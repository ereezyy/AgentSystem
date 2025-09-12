"""
🏦 AgentSystem Stripe Billing Service
Comprehensive subscription and usage-based billing system
"""

import asyncio
import stripe
import hashlib
import hmac
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import logging
from enum import Enum

import asyncpg
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stripe configuration
stripe.api_key = "sk_test_..."  # Will be loaded from environment

class PlanType(str, Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class PaymentStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"

@dataclass
class PricingTier:
    plan_type: str
    monthly_fee: Decimal
    included_tokens: int
    overage_rate: Decimal  # Per token
    features: List[str]
    max_users: int = 10
    api_rate_limit: int = 1000  # requests per minute

# Pricing configuration
PRICING_TIERS = {
    PlanType.STARTER: PricingTier(
        plan_type="starter",
        monthly_fee=Decimal("99.00"),
        included_tokens=100_000,
        overage_rate=Decimal("0.002"),
        features=["basic_ai", "email_support", "5_custom_agents", "basic_analytics"],
        max_users=5,
        api_rate_limit=500
    ),
    PlanType.PROFESSIONAL: PricingTier(
        plan_type="professional",
        monthly_fee=Decimal("299.00"),
        included_tokens=500_000,
        overage_rate=Decimal("0.0015"),
        features=["all_ai", "agent_swarms", "priority_support", "unlimited_agents", "integrations", "advanced_analytics"],
        max_users=25,
        api_rate_limit=2000
    ),
    PlanType.ENTERPRISE: PricingTier(
        plan_type="enterprise",
        monthly_fee=Decimal("999.00"),
        included_tokens=2_000_000,
        overage_rate=Decimal("0.001"),
        features=["unlimited", "custom_agents", "dedicated_support", "white_label", "sso", "compliance", "custom_integrations"],
        max_users=100,
        api_rate_limit=5000
    ),
    PlanType.CUSTOM: PricingTier(
        plan_type="custom",
        monthly_fee=Decimal("2999.00"),  # Starting point for negotiation
        included_tokens=10_000_000,
        overage_rate=Decimal("0.0005"),
        features=["everything", "custom_development", "dedicated_infrastructure", "24_7_support"],
        max_users=1000,
        api_rate_limit=10000
    )
}

class StripeService:
    """Comprehensive Stripe billing integration"""

    def __init__(self, db_pool: asyncpg.Pool, stripe_api_key: str, webhook_secret: str):
        self.db_pool = db_pool
        stripe.api_key = stripe_api_key
        self.webhook_secret = webhook_secret
        self.pricing_tiers = PRICING_TIERS

    async def create_customer(self, tenant_id: str, email: str, name: str,
                            plan_type: PlanType = PlanType.STARTER) -> Dict[str, Any]:
        """Create a new Stripe customer and subscription"""
        try:
            # Create Stripe customer
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={
                    'tenant_id': tenant_id,
                    'plan_type': plan_type.value
                }
            )

            # Create subscription with the selected plan
            subscription = await self._create_subscription(customer.id, plan_type)

            # Update tenant record with Stripe customer ID
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE tenant_management.tenants
                    SET stripe_customer_id = $1, plan_type = $2,
                        current_period_start = $3, current_period_end = $4,
                        monthly_token_limit = $5, api_rate_limit = $6
                    WHERE id = $7
                """, customer.id, plan_type.value,
                    datetime.fromtimestamp(subscription.current_period_start),
                    datetime.fromtimestamp(subscription.current_period_end),
                    self.pricing_tiers[plan_type].included_tokens,
                    self.pricing_tiers[plan_type].api_rate_limit,
                    tenant_id)

            logger.info(f"Created Stripe customer {customer.id} for tenant {tenant_id}")

            return {
                'customer_id': customer.id,
                'subscription_id': subscription.id,
                'client_secret': subscription.latest_invoice.payment_intent.client_secret,
                'plan_details': asdict(self.pricing_tiers[plan_type])
            }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating customer: {e}")
            raise HTTPException(status_code=400, detail=f"Payment error: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating customer: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _create_subscription(self, customer_id: str, plan_type: PlanType) -> stripe.Subscription:
        """Create a Stripe subscription for the given plan"""

        # Create or get the price object for the plan
        price_id = await self._get_or_create_price(plan_type)

        subscription = stripe.Subscription.create(
            customer=customer_id,
            items=[{
                'price': price_id,
            }],
            payment_behavior='default_incomplete',
            payment_settings={'save_default_payment_method': 'on_subscription'},
            expand=['latest_invoice.payment_intent'],
            metadata={
                'plan_type': plan_type.value,
                'included_tokens': str(self.pricing_tiers[plan_type].included_tokens)
            }
        )

        return subscription

    async def _get_or_create_price(self, plan_type: PlanType) -> str:
        """Get or create a Stripe price object for the plan"""

        tier = self.pricing_tiers[plan_type]

        # Try to find existing price
        prices = stripe.Price.list(
            product_data={'name': f'AgentSystem {plan_type.value.title()} Plan'},
            active=True,
            limit=1
        )

        if prices.data:
            return prices.data[0].id

        # Create new price
        price = stripe.Price.create(
            currency='usd',
            unit_amount=int(tier.monthly_fee * 100),  # Convert to cents
            recurring={'interval': 'month'},
            product_data={
                'name': f'AgentSystem {plan_type.value.title()} Plan',
                'description': f'Includes {tier.included_tokens:,} tokens, {", ".join(tier.features[:3])}',
                'metadata': {
                    'plan_type': plan_type.value,
                    'included_tokens': str(tier.included_tokens),
                    'overage_rate': str(tier.overage_rate)
                }
            }
        )

        return price.id

    async def change_subscription_plan(self, tenant_id: str, new_plan_type: PlanType) -> Dict[str, Any]:
        """Change a customer's subscription plan"""
        try:
            # Get tenant's Stripe customer ID
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT stripe_customer_id FROM tenant_management.tenants
                    WHERE id = $1
                """, tenant_id)

                if not row or not row['stripe_customer_id']:
                    raise HTTPException(status_code=404, detail="Customer not found")

                customer_id = row['stripe_customer_id']

            # Get current subscription
            subscriptions = stripe.Subscription.list(customer=customer_id, status='active')
            if not subscriptions.data:
                raise HTTPException(status_code=404, detail="No active subscription found")

            subscription = subscriptions.data[0]

            # Get new price ID
            new_price_id = await self._get_or_create_price(new_plan_type)

            # Update subscription
            updated_subscription = stripe.Subscription.modify(
                subscription.id,
                items=[{
                    'id': subscription['items']['data'][0].id,
                    'price': new_price_id,
                }],
                proration_behavior='create_prorations',
                metadata={
                    'plan_type': new_plan_type.value,
                    'included_tokens': str(self.pricing_tiers[new_plan_type].included_tokens)
                }
            )

            # Update tenant record
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE tenant_management.tenants
                    SET plan_type = $1, monthly_token_limit = $2, api_rate_limit = $3
                    WHERE id = $4
                """, new_plan_type.value,
                    self.pricing_tiers[new_plan_type].included_tokens,
                    self.pricing_tiers[new_plan_type].api_rate_limit,
                    tenant_id)

            logger.info(f"Changed plan for tenant {tenant_id} to {new_plan_type.value}")

            return {
                'subscription_id': updated_subscription.id,
                'new_plan': new_plan_type.value,
                'plan_details': asdict(self.pricing_tiers[new_plan_type])
            }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error changing plan: {e}")
            raise HTTPException(status_code=400, detail=f"Payment error: {str(e)}")

    async def calculate_usage_charges(self, tenant_id: str, billing_period_start: datetime,
                                    billing_period_end: datetime) -> Dict[str, Any]:
        """Calculate usage-based charges for a billing period"""

        async with self.db_pool.acquire() as conn:
            # Get tenant plan info
            tenant_row = await conn.fetchrow("""
                SELECT plan_type, monthly_token_limit FROM tenant_management.tenants
                WHERE id = $1
            """, tenant_id)

            if not tenant_row:
                raise HTTPException(status_code=404, detail="Tenant not found")

            plan_type = PlanType(tenant_row['plan_type'])
            included_tokens = tenant_row['monthly_token_limit']

            # Get usage for the billing period
            usage_row = await conn.fetchrow("""
                SELECT
                    SUM(tokens_used) as total_tokens_used,
                    SUM(calculated_cost_usd) as total_calculated_cost,
                    COUNT(*) as total_requests
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= $2
                    AND created_at < $3
            """, tenant_id, billing_period_start, billing_period_end)

            total_tokens_used = usage_row['total_tokens_used'] or 0
            total_calculated_cost = Decimal(str(usage_row['total_calculated_cost'] or 0))
            total_requests = usage_row['total_requests'] or 0

            # Calculate overage
            overage_tokens = max(0, total_tokens_used - included_tokens)
            overage_rate = self.pricing_tiers[plan_type].overage_rate
            overage_charges = Decimal(str(overage_tokens)) * overage_rate

            # Get base subscription fee
            base_fee = self.pricing_tiers[plan_type].monthly_fee

            # Calculate total
            subtotal = base_fee + overage_charges
            tax_rate = Decimal("0.08")  # 8% tax (would be calculated based on location)
            tax_amount = subtotal * tax_rate
            total_amount = subtotal + tax_amount

            return {
                'base_subscription_fee': float(base_fee),
                'included_tokens': included_tokens,
                'total_tokens_used': total_tokens_used,
                'overage_tokens': overage_tokens,
                'overage_rate': float(overage_rate),
                'overage_charges': float(overage_charges),
                'subtotal': float(subtotal),
                'tax_amount': float(tax_amount),
                'total_amount': float(total_amount),
                'total_requests': total_requests,
                'period_start': billing_period_start.isoformat(),
                'period_end': billing_period_end.isoformat()
            }

    async def create_usage_invoice(self, tenant_id: str, billing_period_start: datetime,
                                 billing_period_end: datetime) -> Dict[str, Any]:
        """Create an invoice for usage charges"""

        try:
            # Calculate usage charges
            usage_data = await self.calculate_usage_charges(tenant_id, billing_period_start, billing_period_end)

            # Get customer ID
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT stripe_customer_id FROM tenant_management.tenants
                    WHERE id = $1
                """, tenant_id)

                if not row or not row['stripe_customer_id']:
                    raise HTTPException(status_code=404, detail="Customer not found")

                customer_id = row['stripe_customer_id']

            # Create invoice items for overage charges (if any)
            if usage_data['overage_charges'] > 0:
                stripe.InvoiceItem.create(
                    customer=customer_id,
                    amount=int(usage_data['overage_charges'] * 100),  # Convert to cents
                    currency='usd',
                    description=f"Token overage: {usage_data['overage_tokens']:,} tokens @ ${usage_data['overage_rate']:.4f} each",
                    metadata={
                        'tenant_id': tenant_id,
                        'billing_period_start': billing_period_start.isoformat(),
                        'billing_period_end': billing_period_end.isoformat(),
                        'overage_tokens': str(usage_data['overage_tokens'])
                    }
                )

            # Create and finalize invoice
            invoice = stripe.Invoice.create(
                customer=customer_id,
                auto_advance=True,  # Automatically finalize
                description=f"AgentSystem usage for {billing_period_start.strftime('%B %Y')}",
                metadata={
                    'tenant_id': tenant_id,
                    'billing_period_start': billing_period_start.isoformat(),
                    'billing_period_end': billing_period_end.isoformat()
                }
            )

            # Save billing record to database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO tenant_management.monthly_bills (
                        tenant_id, billing_month, billing_year, period_start, period_end,
                        base_subscription_fee, included_tokens, total_tokens_used,
                        overage_tokens, overage_charges, subtotal, tax_amount, total_amount,
                        stripe_invoice_id, payment_status, usage_breakdown
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    tenant_id, billing_period_start.month, billing_period_start.year,
                    billing_period_start.date(), billing_period_end.date(),
                    usage_data['base_subscription_fee'], usage_data['included_tokens'],
                    usage_data['total_tokens_used'], usage_data['overage_tokens'],
                    usage_data['overage_charges'], usage_data['subtotal'],
                    usage_data['tax_amount'], usage_data['total_amount'],
                    invoice.id, PaymentStatus.PENDING.value, json.dumps(usage_data)
                )

            logger.info(f"Created usage invoice {invoice.id} for tenant {tenant_id}")

            return {
                'invoice_id': invoice.id,
                'invoice_url': invoice.hosted_invoice_url,
                'amount_due': invoice.amount_due / 100,  # Convert from cents
                'usage_data': usage_data
            }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error creating invoice: {e}")
            raise HTTPException(status_code=400, detail=f"Payment error: {str(e)}")

    async def handle_webhook(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        """Handle Stripe webhook events"""

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
        except ValueError:
            logger.error("Invalid payload in webhook")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid signature in webhook")
            raise HTTPException(status_code=400, detail="Invalid signature")

        logger.info(f"Received Stripe webhook: {event['type']}")

        # Handle different event types
        if event['type'] == 'customer.subscription.created':
            await self._handle_subscription_created(event['data']['object'])
        elif event['type'] == 'customer.subscription.updated':
            await self._handle_subscription_updated(event['data']['object'])
        elif event['type'] == 'customer.subscription.deleted':
            await self._handle_subscription_deleted(event['data']['object'])
        elif event['type'] == 'invoice.payment_succeeded':
            await self._handle_payment_succeeded(event['data']['object'])
        elif event['type'] == 'invoice.payment_failed':
            await self._handle_payment_failed(event['data']['object'])
        elif event['type'] == 'customer.created':
            await self._handle_customer_created(event['data']['object'])

        return {'status': 'success', 'event_type': event['type']}

    async def _handle_subscription_created(self, subscription: Dict[str, Any]):
        """Handle subscription created event"""
        customer_id = subscription['customer']

        # Get tenant ID from customer metadata
        customer = stripe.Customer.retrieve(customer_id)
        tenant_id = customer.metadata.get('tenant_id')

        if tenant_id:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE tenant_management.tenants
                    SET status = 'active',
                        current_period_start = $1,
                        current_period_end = $2
                    WHERE id = $3
                """,
                    datetime.fromtimestamp(subscription['current_period_start']),
                    datetime.fromtimestamp(subscription['current_period_end']),
                    tenant_id
                )

        logger.info(f"Subscription created for customer {customer_id}")

    async def _handle_payment_succeeded(self, invoice: Dict[str, Any]):
        """Handle successful payment"""

        # Update payment status in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE tenant_management.monthly_bills
                SET payment_status = $1, paid_at = $2
                WHERE stripe_invoice_id = $3
            """, PaymentStatus.PAID.value, datetime.now(), invoice['id'])

        logger.info(f"Payment succeeded for invoice {invoice['id']}")

    async def _handle_payment_failed(self, invoice: Dict[str, Any]):
        """Handle failed payment"""

        # Update payment status and potentially suspend account
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE tenant_management.monthly_bills
                SET payment_status = $1
                WHERE stripe_invoice_id = $2
            """, PaymentStatus.FAILED.value, invoice['id'])

            # Check if we should suspend the account (after 3 failed payments)
            failed_count = await conn.fetchval("""
                SELECT COUNT(*) FROM tenant_management.monthly_bills
                WHERE stripe_invoice_id IN (
                    SELECT stripe_invoice_id FROM tenant_management.monthly_bills
                    WHERE stripe_invoice_id = $1
                ) AND payment_status = 'failed'
            """, invoice['id'])

            if failed_count >= 3:
                # Suspend account
                await conn.execute("""
                    UPDATE tenant_management.tenants
                    SET status = 'suspended'
                    WHERE stripe_customer_id = $1
                """, invoice['customer'])

        logger.warning(f"Payment failed for invoice {invoice['id']}")

    async def _handle_subscription_deleted(self, subscription: Dict[str, Any]):
        """Handle subscription cancellation"""
        customer_id = subscription['customer']

        # Update tenant status
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE tenant_management.tenants
                SET status = 'cancelled'
                WHERE stripe_customer_id = $1
            """, customer_id)

        logger.info(f"Subscription cancelled for customer {customer_id}")

    async def _handle_customer_created(self, customer: Dict[str, Any]):
        """Handle customer created event"""
        logger.info(f"Customer created: {customer['id']}")
        # Additional customer setup logic can be added here

    async def get_billing_history(self, tenant_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        """Get billing history for a tenant"""

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM tenant_management.monthly_bills
                WHERE tenant_id = $1
                ORDER BY billing_year DESC, billing_month DESC
                LIMIT $2
            """, tenant_id, limit)

            return [dict(row) for row in rows]

    async def get_current_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get current month usage for a tenant"""

        # Get current billing period
        now = datetime.now()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        async with self.db_pool.acquire() as conn:
            # Get tenant plan info
            tenant_row = await conn.fetchrow("""
                SELECT plan_type, monthly_token_limit FROM tenant_management.tenants
                WHERE id = $1
            """, tenant_id)

            if not tenant_row:
                raise HTTPException(status_code=404, detail="Tenant not found")

            # Get current usage
            usage_row = await conn.fetchrow("""
                SELECT
                    SUM(tokens_used) as total_tokens_used,
                    COUNT(*) as total_requests,
                    SUM(calculated_cost_usd) as estimated_cost
                FROM tenant_management.usage_events
                WHERE tenant_id = $1 AND created_at >= $2
            """, tenant_id, period_start)

            total_tokens_used = usage_row['total_tokens_used'] or 0
            total_requests = usage_row['total_requests'] or 0
            estimated_cost = float(usage_row['estimated_cost'] or 0)

            included_tokens = tenant_row['monthly_token_limit']
            usage_percentage = (total_tokens_used / included_tokens * 100) if included_tokens > 0 else 0

            return {
                'current_period_start': period_start.isoformat(),
                'total_tokens_used': total_tokens_used,
                'included_tokens': included_tokens,
                'usage_percentage': min(usage_percentage, 100),
                'overage_tokens': max(0, total_tokens_used - included_tokens),
                'total_requests': total_requests,
                'estimated_cost': estimated_cost,
                'days_remaining': (period_start.replace(month=period_start.month + 1) - now).days
            }

# Pydantic models for API requests/responses
class CreateCustomerRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant UUID")
    email: str = Field(..., description="Customer email")
    name: str = Field(..., description="Customer name")
    plan_type: PlanType = Field(default=PlanType.STARTER, description="Initial plan type")

class ChangePlanRequest(BaseModel):
    new_plan_type: PlanType = Field(..., description="New plan type")

class UsageCalculationRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant UUID")
    period_start: datetime = Field(..., description="Billing period start")
    period_end: datetime = Field(..., description="Billing period end")

# Usage tracking decorator
def track_usage(service_type: str, tokens_used: int = 0):
    """Decorator to automatically track usage for API endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract tenant_id from request
            tenant_id = kwargs.get('tenant_id') or getattr(args[0], 'tenant_id', None)

            if tenant_id:
                # Record usage event
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO tenant_management.usage_events (
                            tenant_id, event_type, service_type, tokens_used,
                            billing_period_start
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, tenant_id, 'api_request', service_type, tokens_used,
                        datetime.now().replace(day=1).date())

            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Export main class and utilities
__all__ = ['StripeService', 'PlanType', 'PaymentStatus', 'PricingTier', 'PRICING_TIERS', 'track_usage']
