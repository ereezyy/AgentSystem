"""
Overage Billing Module for AgentSystem
Handles billing for usage beyond plan limits, specifically for the Pro tier's credits-per-task model
"""

import logging
import os
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
from decimal import Decimal
import stripe
import asyncpg
import aioredis

logger = logging.getLogger(__name__)

class OverageBilling:
    """Manages overage billing for usage beyond plan limits"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        """Initialize the overage billing system"""
        self.db_pool = db_pool
        self.redis = redis_client
        stripe.api_key = os.getenv("STRIPE_API_KEY")
        self.overage_cost_per_credit = Decimal("0.99")  # Cost per task resolution for Pro tier
        self._running = False
        self._billing_task = None
        self.billing_interval = 3600  # Hourly billing cycle for overages

    async def start(self):
        """Start the overage billing system with background processing"""
        self._running = True
        self._billing_task = asyncio.create_task(self._billing_loop())
        logger.info("Overage billing system started")

    async def stop(self):
        """Stop the overage billing system and process any pending charges"""
        self._running = False
        if self._billing_task:
            self._billing_task.cancel()
            try:
                await self._billing_task
            except asyncio.CancelledError:
                pass

        # Process any remaining overage charges
        await self._process_overage_charges()
        logger.info("Overage billing system stopped")

    async def track_overage_task(self, tenant_id: str, task_id: str, plan_type: str):
        """Track a task for potential overage billing"""
        if plan_type != "Pro":
            return  # Only Pro tier has credits-per-task overage model

        # Increment overage task counter in Redis for quick access
        month_key = f"overage:{tenant_id}:{datetime.now().strftime('%Y-%m')}"
        await self.redis.hincrby(month_key, "tasks", 1)
        await self.redis.expire(month_key, 86400 * 35)  # Expire after 35 days

        # Record task details in database for billing history
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.overage_tasks (tenant_id, task_id, billing_period_start)
                VALUES ($1, $2, $3)
            """, tenant_id, task_id, datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).date())

        logger.info(f"Tracked overage task {task_id} for tenant {tenant_id}")

    async def _billing_loop(self):
        """Background task to process overage billing periodically"""
        while self._running:
            try:
                await asyncio.sleep(self.billing_interval)
                await self._process_overage_charges()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in overage billing loop: {e}")

    async def _process_overage_charges(self):
        """Process overage charges for all tenants"""
        async with self.db_pool.acquire() as conn:
            # Get all Pro tier tenants
            tenants = await conn.fetch("""
                SELECT id, stripe_customer_id, plan_type
                FROM tenant_management.tenants
                WHERE plan_type = 'Pro'
            """)

            for tenant in tenants:
                tenant_id = tenant['id']
                stripe_customer_id = tenant['stripe_customer_id']
                month_key = f"overage:{tenant_id}:{datetime.now().strftime('%Y-%m')}"

                # Get overage task count from Redis
                overage_tasks = await self.redis.hget(month_key, "tasks") or 0
                overage_tasks = int(overage_tasks)

                if overage_tasks > 0:
                    # Calculate overage charge
                    overage_charge = Decimal(str(overage_tasks)) * self.overage_cost_per_credit

                    try:
                        # Create Stripe invoice item for overage
                        invoice_item = stripe.InvoiceItem.create(
                            customer=stripe_customer_id,
                            amount=int(overage_charge * 100),  # Convert to cents
                            currency="usd",
                            description=f"Overage charge for {overage_tasks} tasks beyond Pro plan limit"
                        )

                        # Log billing event in database
                        await conn.execute("""
                            INSERT INTO tenant_management.billing_events
                            (tenant_id, event_type, amount, description, stripe_invoice_item_id)
                            VALUES ($1, $2, $3, $4, $5)
                        """, tenant_id, "overage_charge", overage_charge,
                        f"Overage for {overage_tasks} tasks", invoice_item.id)

                        logger.info(f"Processed overage charge of ${overage_charge} for tenant {tenant_id}")

                        # Reset overage counter after billing
                        await self.redis.hset(month_key, "tasks", 0)

                    except stripe.error.StripeError as e:
                        logger.error(f"Stripe billing error for tenant {tenant_id}: {e}")
                        # Record failed billing attempt for manual resolution
                        await conn.execute("""
                            INSERT INTO tenant_management.billing_errors
                            (tenant_id, amount, error_message)
                            VALUES ($1, $2, $3)
                        """, tenant_id, overage_charge, str(e))

    async def get_overage_summary(self, tenant_id: str, period: str = "current_month") -> Dict[str, Any]:
        """Get overage billing summary for a tenant"""
        now = datetime.now()

        if period == "current_month":
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif period == "last_month":
            last_month = now.replace(month=now.month-1) if now.month > 1 else now.replace(year=now.year-1, month=12)
            start_date = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now

        async with self.db_pool.acquire() as conn:
            overage_tasks = await conn.fetchval("""
                SELECT COUNT(*)
                FROM tenant_management.overage_tasks
                WHERE tenant_id = $1 AND billing_period_start >= $2 AND billing_period_start <= $3
            """, tenant_id, start_date.date(), end_date.date())

            billing_events = await conn.fetch("""
                SELECT amount, description, created_at
                FROM tenant_management.billing_events
                WHERE tenant_id = $1 AND event_type = 'overage_charge'
                AND created_at >= $2 AND created_at <= $3
                ORDER BY created_at DESC
            """, tenant_id, start_date, end_date)

        month_key = f"overage:{tenant_id}:{now.strftime('%Y-%m')}"
        current_overage_tasks = int(await self.redis.hget(month_key, "tasks") or 0)

        total_billed = sum(event['amount'] for event in billing_events)

        return {
            "period": period,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_overage_tasks": overage_tasks,
            "current_unbilled_tasks": current_overage_tasks,
            "total_billed_amount": float(total_billed),
            "cost_per_credit": float(self.overage_cost_per_credit),
            "billing_events": [
                {
                    "amount": float(event['amount']),
                    "description": event['description'],
                    "date": event['created_at'].isoformat()
                }
                for event in billing_events
            ]
        }
