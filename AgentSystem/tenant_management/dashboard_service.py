"""
🏢 AgentSystem Tenant Management Dashboard
Comprehensive tenant onboarding, management, and analytics dashboard
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import bcrypt
from email_validator import validate_email, EmailNotValidError

import asyncpg
import aioredis
from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import jinja2

from ..billing.stripe_service import StripeService, PlanType
from ..usage.usage_tracker import UsageTracker
from ..pricing.pricing_engine import PricingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnboardingStage(str, Enum):
    EMAIL_VERIFICATION = "email_verification"
    ACCOUNT_SETUP = "account_setup"
    PLAN_SELECTION = "plan_selection"
    PAYMENT_SETUP = "payment_setup"
    TEAM_SETUP = "team_setup"
    INTEGRATION_SETUP = "integration_setup"
    COMPLETED = "completed"

class TenantStatus(str, Enum):
    ONBOARDING = "onboarding"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    TRIAL = "trial"

@dataclass
class OnboardingProgress:
    """Track onboarding progress"""
    current_stage: OnboardingStage
    completed_stages: List[OnboardingStage]
    progress_percentage: float
    next_steps: List[Dict[str, Any]]
    estimated_completion_time: int  # minutes

@dataclass
class DashboardMetrics:
    """Dashboard metrics for a tenant"""
    current_usage: Dict[str, Any]
    billing_info: Dict[str, Any]
    performance_stats: Dict[str, Any]
    recent_activity: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    growth_metrics: Dict[str, Any]

class TenantDashboardService:
    """Comprehensive tenant dashboard and management service"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis,
                 stripe_service: StripeService, usage_tracker: UsageTracker,
                 pricing_engine: PricingEngine):
        self.db_pool = db_pool
        self.redis = redis_client
        self.stripe_service = stripe_service
        self.usage_tracker = usage_tracker
        self.pricing_engine = pricing_engine

        # Email templates
        self.email_templates = jinja2.Environment(
            loader=jinja2.DictLoader({
                'welcome_email': '''
                <!DOCTYPE html>
                <html>
                <head><title>Welcome to AgentSystem</title></head>
                <body>
                    <h1>Welcome to AgentSystem, {{ name }}!</h1>
                    <p>Click here to verify your email and complete setup:</p>
                    <a href="{{ verification_url }}">Verify Email</a>
                </body>
                </html>
                ''',
                'onboarding_reminder': '''
                <h1>Complete Your AgentSystem Setup</h1>
                <p>Hi {{ name }}, you're {{ progress_percentage }}% done with setup!</p>
                <p>Next steps: {{ next_steps | join(', ') }}</p>
                <a href="{{ dashboard_url }}">Continue Setup</a>
                '''
            })
        )

    async def start_onboarding(self, email: str, name: str, company: str = "",
                             initial_plan: PlanType = PlanType.STARTER,
                             utm_params: Dict[str, str] = None) -> Dict[str, Any]:
        """Start the onboarding process for a new tenant"""

        try:
            # Validate email
            valid = validate_email(email)
            email = valid.email
        except EmailNotValidError:
            raise HTTPException(status_code=400, detail="Invalid email address")

        # Check if email already exists
        async with self.db_pool.acquire() as conn:
            existing = await conn.fetchrow("""
                SELECT id FROM tenant_management.tenants
                WHERE billing_email = $1
            """, email)

            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")

        # Generate subdomain from company name or email
        subdomain = self._generate_subdomain(company or email.split('@')[0])

        # Create tenant record
        tenant_id = await self._create_tenant_record(
            email, name, company, subdomain, initial_plan, utm_params
        )

        # Generate email verification token
        verification_token = secrets.token_urlsafe(32)
        await self.redis.setex(
            f"email_verification:{tenant_id}",
            3600,  # 1 hour
            verification_token
        )

        # Send welcome email
        await self._send_welcome_email(email, name, verification_token, subdomain)

        # Initialize onboarding progress
        progress = OnboardingProgress(
            current_stage=OnboardingStage.EMAIL_VERIFICATION,
            completed_stages=[],
            progress_percentage=10.0,
            next_steps=[
                {"action": "verify_email", "title": "Verify your email address"},
                {"action": "setup_account", "title": "Complete account setup"}
            ],
            estimated_completion_time=15
        )

        await self._save_onboarding_progress(tenant_id, progress)

        logger.info(f"Started onboarding for tenant {tenant_id} - {email}")

        return {
            'tenant_id': tenant_id,
            'subdomain': subdomain,
            'verification_required': True,
            'next_step': 'email_verification',
            'dashboard_url': f"https://{subdomain}.agentsystem.ai/dashboard"
        }

    async def verify_email(self, tenant_id: str, verification_token: str) -> Dict[str, Any]:
        """Verify email address and advance onboarding"""

        # Check verification token
        stored_token = await self.redis.get(f"email_verification:{tenant_id}")
        if not stored_token or stored_token.decode() != verification_token:
            raise HTTPException(status_code=400, detail="Invalid or expired verification token")

        # Update tenant status
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE tenant_management.tenants
                SET status = $1, updated_at = NOW()
                WHERE id = $2
            """, TenantStatus.ONBOARDING.value, tenant_id)

        # Update onboarding progress
        progress = await self._get_onboarding_progress(tenant_id)
        progress.completed_stages.append(OnboardingStage.EMAIL_VERIFICATION)
        progress.current_stage = OnboardingStage.ACCOUNT_SETUP
        progress.progress_percentage = 25.0
        progress.next_steps = [
            {"action": "setup_account", "title": "Complete account information"},
            {"action": "select_plan", "title": "Choose your plan"}
        ]
        progress.estimated_completion_time = 10

        await self._save_onboarding_progress(tenant_id, progress)

        # Clean up verification token
        await self.redis.delete(f"email_verification:{tenant_id}")

        return {
            'verified': True,
            'next_stage': OnboardingStage.ACCOUNT_SETUP.value,
            'progress': progress.progress_percentage
        }

    async def complete_account_setup(self, tenant_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete account setup information"""

        async with self.db_pool.acquire() as conn:
            # Update tenant information
            await conn.execute("""
                UPDATE tenant_management.tenants
                SET settings = settings || $1, updated_at = NOW()
                WHERE id = $2
            """, json.dumps({
                'account_info': account_data,
                'setup_completed_at': datetime.now().isoformat()
            }), tenant_id)

            # Create owner user record
            await conn.execute("""
                INSERT INTO tenant_management.tenant_users (
                    tenant_id, email, name, role, is_active, joined_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (tenant_id, email) DO UPDATE SET
                name = EXCLUDED.name, joined_at = EXCLUDED.joined_at
            """, tenant_id, account_data.get('email'), account_data.get('name'),
                 'owner', True, datetime.now())

        # Update onboarding progress
        progress = await self._get_onboarding_progress(tenant_id)
        progress.completed_stages.append(OnboardingStage.ACCOUNT_SETUP)
        progress.current_stage = OnboardingStage.PLAN_SELECTION
        progress.progress_percentage = 40.0
        progress.next_steps = [
            {"action": "select_plan", "title": "Choose your subscription plan"},
            {"action": "setup_payment", "title": "Add payment method"}
        ]
        progress.estimated_completion_time = 8

        await self._save_onboarding_progress(tenant_id, progress)

        return {
            'account_setup_complete': True,
            'next_stage': OnboardingStage.PLAN_SELECTION.value,
            'progress': progress.progress_percentage
        }

    async def select_plan(self, tenant_id: str, plan_type: PlanType) -> Dict[str, Any]:
        """Select and configure subscription plan"""

        # Get tenant information for Stripe customer creation
        async with self.db_pool.acquire() as conn:
            tenant_row = await conn.fetchrow("""
                SELECT name, billing_email, settings FROM tenant_management.tenants
                WHERE id = $1
            """, tenant_id)

            if not tenant_row:
                raise HTTPException(status_code=404, detail="Tenant not found")

        settings = json.loads(tenant_row['settings'] or '{}')
        account_info = settings.get('account_info', {})

        # Create Stripe customer and subscription
        stripe_result = await self.stripe_service.create_customer(
            tenant_id=tenant_id,
            email=tenant_row['billing_email'],
            name=account_info.get('name', tenant_row['name']),
            plan_type=plan_type
        )

        # Update onboarding progress
        progress = await self._get_onboarding_progress(tenant_id)
        progress.completed_stages.append(OnboardingStage.PLAN_SELECTION)
        progress.current_stage = OnboardingStage.PAYMENT_SETUP
        progress.progress_percentage = 60.0
        progress.next_steps = [
            {"action": "setup_payment", "title": "Complete payment setup"},
            {"action": "setup_team", "title": "Invite team members (optional)"}
        ]
        progress.estimated_completion_time = 5

        await self._save_onboarding_progress(tenant_id, progress)

        return {
            'plan_selected': True,
            'plan_type': plan_type.value,
            'stripe_client_secret': stripe_result['client_secret'],
            'next_stage': OnboardingStage.PAYMENT_SETUP.value,
            'progress': progress.progress_percentage
        }

    async def complete_payment_setup(self, tenant_id: str, payment_method_id: str) -> Dict[str, Any]:
        """Complete payment setup and activate subscription"""

        # Payment setup is handled by Stripe webhooks, so we just need to verify
        async with self.db_pool.acquire() as conn:
            tenant_row = await conn.fetchrow("""
                SELECT stripe_customer_id FROM tenant_management.tenants
                WHERE id = $1
            """, tenant_id)

            if not tenant_row or not tenant_row['stripe_customer_id']:
                raise HTTPException(status_code=400, detail="Stripe customer not found")

        # Generate API keys
        api_keys = await self._generate_api_keys(tenant_id)

        # Update onboarding progress
        progress = await self._get_onboarding_progress(tenant_id)
        progress.completed_stages.append(OnboardingStage.PAYMENT_SETUP)
        progress.current_stage = OnboardingStage.TEAM_SETUP
        progress.progress_percentage = 80.0
        progress.next_steps = [
            {"action": "setup_team", "title": "Invite team members (optional)"},
            {"action": "setup_integrations", "title": "Configure integrations (optional)"}
        ]
        progress.estimated_completion_time = 3

        await self._save_onboarding_progress(tenant_id, progress)

        return {
            'payment_setup_complete': True,
            'api_keys': api_keys,
            'next_stage': OnboardingStage.TEAM_SETUP.value,
            'progress': progress.progress_percentage
        }

    async def skip_to_completion(self, tenant_id: str) -> Dict[str, Any]:
        """Skip optional steps and complete onboarding"""

        # Update tenant status to active
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE tenant_management.tenants
                SET status = $1, updated_at = NOW()
                WHERE id = $2
            """, TenantStatus.ACTIVE.value, tenant_id)

        # Update onboarding progress
        progress = await self._get_onboarding_progress(tenant_id)
        progress.completed_stages.append(OnboardingStage.COMPLETED)
        progress.current_stage = OnboardingStage.COMPLETED
        progress.progress_percentage = 100.0
        progress.next_steps = []
        progress.estimated_completion_time = 0

        await self._save_onboarding_progress(tenant_id, progress)

        # Start trial if applicable
        await self._start_trial_if_applicable(tenant_id)

        return {
            'onboarding_complete': True,
            'tenant_status': TenantStatus.ACTIVE.value,
            'dashboard_ready': True
        }

    async def get_dashboard_data(self, tenant_id: str) -> DashboardMetrics:
        """Get comprehensive dashboard data for a tenant"""

        # Get current usage
        current_usage = await self.usage_tracker.get_current_usage(tenant_id)

        # Get billing information
        billing_info = await self._get_billing_dashboard_info(tenant_id)

        # Get performance statistics
        performance_stats = await self._get_performance_stats(tenant_id)

        # Get recent activity
        recent_activity = await self._get_recent_activity(tenant_id)

        # Get alerts and notifications
        alerts = await self._get_alerts(tenant_id)

        # Get growth metrics
        growth_metrics = await self._get_growth_metrics(tenant_id)

        return DashboardMetrics(
            current_usage=current_usage,
            billing_info=billing_info,
            performance_stats=performance_stats,
            recent_activity=recent_activity,
            alerts=alerts,
            growth_metrics=growth_metrics
        )

    async def _create_tenant_record(self, email: str, name: str, company: str,
                                  subdomain: str, plan_type: PlanType,
                                  utm_params: Dict[str, str] = None) -> str:
        """Create a new tenant record in the database"""

        async with self.db_pool.acquire() as conn:
            tenant_id = await conn.fetchval("""
                INSERT INTO tenant_management.tenants (
                    name, subdomain, plan_type, status, billing_email,
                    settings, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())
                RETURNING id
            """,
                company or name, subdomain, plan_type.value,
                TenantStatus.ONBOARDING.value, email,
                json.dumps({
                    'utm_params': utm_params or {},
                    'onboarding_started_at': datetime.now().isoformat(),
                    'initial_plan': plan_type.value
                })
            )

            return str(tenant_id)

    def _generate_subdomain(self, base: str) -> str:
        """Generate a unique subdomain from base string"""
        import re

        # Clean and normalize base string
        subdomain = re.sub(r'[^a-zA-Z0-9]', '-', base.lower())
        subdomain = re.sub(r'-+', '-', subdomain)
        subdomain = subdomain.strip('-')[:20]  # Max 20 chars

        if not subdomain or len(subdomain) < 3:
            subdomain = 'tenant-' + secrets.token_hex(4)

        # Add random suffix to ensure uniqueness
        return f"{subdomain}-{secrets.token_hex(2)}"

    async def _send_welcome_email(self, email: str, name: str,
                                verification_token: str, subdomain: str):
        """Send welcome email with verification link"""

        verification_url = f"https://{subdomain}.agentsystem.ai/verify?token={verification_token}"

        template = self.email_templates.get_template('welcome_email')
        html_content = template.render(
            name=name,
            verification_url=verification_url
        )

        # In a real implementation, you'd use a service like SendGrid, SES, etc.
        logger.info(f"Welcome email sent to {email}: {verification_url}")

        # Store email in Redis for potential resending
        await self.redis.setex(
            f"welcome_email:{email}",
            86400,  # 24 hours
            json.dumps({
                'name': name,
                'verification_url': verification_url,
                'sent_at': datetime.now().isoformat()
            })
        )

    async def _save_onboarding_progress(self, tenant_id: str, progress: OnboardingProgress):
        """Save onboarding progress to Redis"""
        await self.redis.setex(
            f"onboarding_progress:{tenant_id}",
            86400 * 7,  # 7 days
            json.dumps(asdict(progress))
        )

    async def _get_onboarding_progress(self, tenant_id: str) -> OnboardingProgress:
        """Get onboarding progress from Redis"""
        progress_data = await self.redis.get(f"onboarding_progress:{tenant_id}")

        if not progress_data:
            # Return default progress
            return OnboardingProgress(
                current_stage=OnboardingStage.EMAIL_VERIFICATION,
                completed_stages=[],
                progress_percentage=0.0,
                next_steps=[],
                estimated_completion_time=15
            )

        data = json.loads(progress_data)
        return OnboardingProgress(**data)

    async def _generate_api_keys(self, tenant_id: str) -> Dict[str, str]:
        """Generate API keys for the tenant"""

        # Generate production and test API keys
        prod_key = f"as_live_{secrets.token_urlsafe(32)}"
        test_key = f"as_test_{secrets.token_urlsafe(32)}"

        # Hash keys for storage
        prod_hash = bcrypt.hashpw(prod_key.encode(), bcrypt.gensalt()).decode()
        test_hash = bcrypt.hashpw(test_key.encode(), bcrypt.gensalt()).decode()

        async with self.db_pool.acquire() as conn:
            # Store hashed keys
            await conn.executemany("""
                INSERT INTO tenant_management.tenant_api_keys (
                    tenant_id, key_name, api_key_hash, permissions, is_active, created_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
            """, [
                (tenant_id, 'Production', prod_hash, json.dumps({'all': True}), True),
                (tenant_id, 'Test', test_hash, json.dumps({'all': True}), True)
            ])

        return {
            'production_key': prod_key,
            'test_key': test_key,
            'warning': 'Store these keys securely - they will not be shown again'
        }

    async def _start_trial_if_applicable(self, tenant_id: str):
        """Start trial period if applicable"""

        async with self.db_pool.acquire() as conn:
            tenant_row = await conn.fetchrow("""
                SELECT plan_type FROM tenant_management.tenants WHERE id = $1
            """, tenant_id)

            if tenant_row and tenant_row['plan_type'] == PlanType.STARTER.value:
                # Start 14-day trial
                await conn.execute("""
                    UPDATE tenant_management.tenants
                    SET status = $1, settings = settings || $2
                    WHERE id = $3
                """, TenantStatus.TRIAL.value, json.dumps({
                    'trial_started_at': datetime.now().isoformat(),
                    'trial_ends_at': (datetime.now() + timedelta(days=14)).isoformat()
                }), tenant_id)

    async def _get_billing_dashboard_info(self, tenant_id: str) -> Dict[str, Any]:
        """Get billing information for dashboard"""

        billing_history = await self.stripe_service.get_billing_history(tenant_id, limit=3)
        current_usage = await self.usage_tracker.get_current_usage(tenant_id)

        return {
            'current_period_usage': current_usage,
            'recent_invoices': billing_history,
            'next_billing_date': current_usage.get('period_end'),
            'estimated_current_bill': current_usage.get('estimated_cost', 0)
        }

    async def _get_performance_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get performance statistics"""

        # Get last 7 days of performance data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        async with self.db_pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT
                    AVG(processing_time_ms) as avg_response_time,
                    COUNT(*) as total_requests,
                    COUNT(*) FILTER (WHERE tokens_used > 0) as successful_requests,
                    SUM(tokens_used) as total_tokens
                FROM tenant_management.usage_events
                WHERE tenant_id = $1 AND created_at >= $2
            """, tenant_id, start_date)

        success_rate = 0.95  # Default success rate
        if stats and stats['total_requests'] > 0:
            success_rate = stats['successful_requests'] / stats['total_requests']

        return {
            'avg_response_time_ms': float(stats['avg_response_time'] or 0),
            'total_requests_7d': stats['total_requests'] or 0,
            'success_rate': success_rate,
            'total_tokens_7d': stats['total_tokens'] or 0
        }

    async def _get_recent_activity(self, tenant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activity for the tenant"""

        async with self.db_pool.acquire() as conn:
            activities = await conn.fetch("""
                SELECT
                    event_type, service_type, tokens_used,
                    calculated_cost_usd, created_at, request_metadata
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, tenant_id, limit)

        return [
            {
                'type': row['event_type'],
                'service': row['service_type'],
                'tokens': row['tokens_used'],
                'cost': float(row['calculated_cost_usd'] or 0),
                'timestamp': row['created_at'].isoformat(),
                'metadata': json.loads(row['request_metadata'] or '{}')
            }
            for row in activities
        ]

    async def _get_alerts(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get alerts and notifications for the tenant"""

        alerts = []

        # Check usage alerts
        current_usage = await self.usage_tracker.get_current_usage(tenant_id)
        if current_usage.get('usage_percentage', 0) > 80:
            alerts.append({
                'type': 'warning',
                'title': 'High Token Usage',
                'message': f"You've used {current_usage.get('usage_percentage', 0):.1f}% of your monthly tokens",
                'action_url': '/billing/upgrade'
            })

        # Check for failed payments (from notifications Redis)
        notifications = await self.redis.lrange(f"notifications:{tenant_id}", 0, 9)
        for notification in notifications:
            data = json.loads(notification)
            if data.get('type') == 'rate_limit_exceeded':
                alerts.append({
                    'type': 'error',
                    'title': 'Rate Limit Exceeded',
                    'message': data.get('message'),
                    'action_url': '/billing/upgrade'
                })

        return alerts

    async def _get_growth_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get growth and trend metrics"""

        # Get usage trends
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        async with self.db_pool.acquire() as conn:
            daily_usage = await conn.fetch("""
                SELECT
                    DATE(created_at) as date,
                    SUM(tokens_used) as daily_tokens,
                    COUNT(*) as daily_requests
                FROM tenant_management.usage_events
                WHERE tenant_id = $1 AND created_at >= $2
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """, tenant_id, start_date)

        # Calculate growth trends
        if len(daily_usage) >= 7:
            recent_week = sum(row['daily_tokens'] for row in daily_usage[-7:])
            previous_week = sum(row['daily_tokens'] for row in daily_usage[-14:-7]) if len(daily_usage) >= 14 else recent_week

            growth_rate = ((recent_week - previous_week) / previous_week * 100) if previous_week > 0 else 0
        else:
            growth_rate = 0

        return {
            'token_growth_rate': growth_rate,
            'daily_usage_trend': [
                {
                    'date': row['date'].isoformat(),
                    'tokens': row['daily_tokens'],
                    'requests': row['daily_requests']
                }
                for row in daily_usage
            ]
        }

# Pydantic models for API requests
class StartOnboardingRequest(BaseModel):
    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User full name")
    company: str = Field("", description="Company name")
    initial_plan: PlanType = Field(default=PlanType.STARTER, description="Initial plan selection")
    utm_source: Optional[str] = Field(None, description="UTM source tracking")
    utm_medium: Optional[str] = Field(None, description="UTM medium tracking")
    utm_campaign: Optional[str] = Field(None, description="UTM campaign tracking")

    @validator('email')
    def validate_email(cls, v):
        try:
            valid = validate_email(v)
            return valid.email
        except EmailNotValidError:
            raise ValueError('Invalid email address')

class AccountSetupRequest(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    company: str = Field("", description="Company name")
    industry: str = Field("", description="Industry")
    use_case: str = Field("", description="Primary use case")
    team_size: str = Field("", description="Team size")

class EmailVerificationRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant ID")
    verification_token: str = Field(..., description="Email verification token")

class PlanSelectionRequest(BaseModel):
    plan_type: PlanType = Field(..., description="Selected plan type")

class PaymentSetupRequest(BaseModel):
    payment_method_id: str = Field(..., description="Stripe payment method ID")

# Export main classes and models
__all__ = [
    'TenantDashboardService', 'OnboardingStage', 'TenantStatus', 'OnboardingProgress',
    'DashboardMetrics', 'StartOnboardingRequest', 'AccountSetupRequest',
    'EmailVerificationRequest', 'PlanSelectionRequest', 'PaymentSetupRequest'
]
