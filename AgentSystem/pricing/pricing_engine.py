
"""
💰 AgentSystem Pricing Engine
Dynamic tiered pricing with intelligent rate limiting and overage billing
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict
import aioredis
import asyncpg
from fastapi import HTTPException, Request
from pydantic import BaseModel

from ..billing.stripe_service import PlanType, PRICING_TIERS
from ..usage.usage_tracker import ServiceType, UsageTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitType(str, Enum):
    REQUESTS_PER_MINUTE = "requests_per_minute"
    TOKENS_PER_HOUR = "tokens_per_hour"
    AGENTS_PER_DAY = "agents_per_day"
    WORKFLOWS_PER_DAY = "workflows_per_day"
    INTEGRATIONS_ACTIVE = "integrations_active"

class OveragePolicy(str, Enum):
    BLOCK = "block"  # Block requests after limit
    CHARGE = "charge"  # Allow with overage charges
    NOTIFY = "notify"  # Allow with notifications only

@dataclass
class PricingFeature:
    """Individual feature configuration"""
    name: str
    enabled: bool = True
    limit: Optional[int] = None  # None = unlimited
    overage_rate: Optional[Decimal] = None  # Cost per unit over limit
    description: str = ""

@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit_type: RateLimitType
    limit: int
    window_seconds: int
    overage_policy: OveragePolicy = OveragePolicy.BLOCK
    overage_rate: Optional[Decimal] = None
    burst_allowance: float = 1.2  # 20% burst capacity

@dataclass
class PlanConfiguration:
    """Complete plan configuration"""
    plan_type: PlanType
    monthly_fee: Decimal

    # Token allocations
    included_tokens: int
    token_overage_rate: Decimal

    # Rate limits
    rate_limits: Dict[RateLimitType, RateLimit]

    # Features
    features: Dict[str, PricingFeature]

    # User and resource limits
    max_users: int
    max_custom_agents: int
    max_workflows: int
    max_integrations: int

    # Support level
    support_level: str
    support_response_time: str

    # Advanced capabilities
    white_label: bool = False
    sso_enabled: bool = False
    compliance_features: List[str] = field(default_factory=list)
    custom_development: bool = False

# Complete pricing configuration
PRICING_CONFIGURATIONS = {
    PlanType.STARTER: PlanConfiguration(
        plan_type=PlanType.STARTER,
        monthly_fee=Decimal("99.00"),
        included_tokens=100_000,
        token_overage_rate=Decimal("0.002"),
        rate_limits={
            RateLimitType.REQUESTS_PER_MINUTE: RateLimit(
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=500,
                window_seconds=60,
                overage_policy=OveragePolicy.BLOCK
            ),
            RateLimitType.TOKENS_PER_HOUR: RateLimit(
                limit_type=RateLimitType.TOKENS_PER_HOUR,
                limit=10_000,
                window_seconds=3600,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.003")
            ),
            RateLimitType.AGENTS_PER_DAY: RateLimit(
                limit_type=RateLimitType.AGENTS_PER_DAY,
                limit=50,
                window_seconds=86400,
                overage_policy=OveragePolicy.BLOCK
            )
        },
        features={
            "basic_ai": PricingFeature("Basic AI Models", True, description="GPT-3.5, Claude Haiku"),
            "email_support": PricingFeature("Email Support", True, description="Business hours email support"),
            "custom_agents": PricingFeature("Custom Agents", True, limit=5, description="Create up to 5 custom agents"),
            "basic_analytics": PricingFeature("Basic Analytics", True, description="Usage reports and basic insights"),
            "api_access": PricingFeature("API Access", True, description="RESTful API access"),
            "webhooks": PricingFeature("Webhooks", True, limit=10, description="Up to 10 webhook endpoints")
        },
        max_users=5,
        max_custom_agents=5,
        max_workflows=25,
        max_integrations=3,
        support_level="email",
        support_response_time="24 hours"
    ),

    PlanType.PROFESSIONAL: PlanConfiguration(
        plan_type=PlanType.PROFESSIONAL,
        monthly_fee=Decimal("299.00"),
        included_tokens=500_000,
        token_overage_rate=Decimal("0.0015"),
        rate_limits={
            RateLimitType.REQUESTS_PER_MINUTE: RateLimit(
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=2000,
                window_seconds=60,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.001")
            ),
            RateLimitType.TOKENS_PER_HOUR: RateLimit(
                limit_type=RateLimitType.TOKENS_PER_HOUR,
                limit=50_000,
                window_seconds=3600,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.002")
            ),
            RateLimitType.AGENTS_PER_DAY: RateLimit(
                limit_type=RateLimitType.AGENTS_PER_DAY,
                limit=250,
                window_seconds=86400,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.10")
            ),
            RateLimitType.WORKFLOWS_PER_DAY: RateLimit(
                limit_type=RateLimitType.WORKFLOWS_PER_DAY,
                limit=500,
                window_seconds=86400,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.05")
            )
        },
        features={
            "all_ai": PricingFeature("All AI Models", True, description="GPT-4, Claude 3, Gemini Pro"),
            "agent_swarms": PricingFeature("Agent Swarms", True, description="Coordinate multiple agents"),
            "priority_support": PricingFeature("Priority Support", True, description="Priority email and chat support"),
            "unlimited_agents": PricingFeature("Unlimited Custom Agents", True, description="Create unlimited custom agents"),
            "integrations": PricingFeature("Integrations", True, description="Salesforce, HubSpot, Slack, Teams"),
            "advanced_analytics": PricingFeature("Advanced Analytics", True, description="Detailed insights and reporting"),
            "workflow_builder": PricingFeature("Visual Workflow Builder", True, description="Drag-and-drop automation"),
            "bulk_operations": PricingFeature("Bulk Operations", True, description="Process large datasets efficiently"),
            "scheduled_agents": PricingFeature("Scheduled Agents", True, description="Time-based automation")
        },
        max_users=25,
        max_custom_agents=-1,  # Unlimited
        max_workflows=500,
        max_integrations=15,
        support_level="priority",
        support_response_time="4 hours"
    ),

    PlanType.ENTERPRISE: PlanConfiguration(
        plan_type=PlanType.ENTERPRISE,
        monthly_fee=Decimal("999.00"),
        included_tokens=2_000_000,
        token_overage_rate=Decimal("0.001"),
        rate_limits={
            RateLimitType.REQUESTS_PER_MINUTE: RateLimit(
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=5000,
                window_seconds=60,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.0005")
            ),
            RateLimitType.TOKENS_PER_HOUR: RateLimit(
                limit_type=RateLimitType.TOKENS_PER_HOUR,
                limit=200_000,
                window_seconds=3600,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.0012")
            ),
            RateLimitType.AGENTS_PER_DAY: RateLimit(
                limit_type=RateLimitType.AGENTS_PER_DAY,
                limit=1000,
                window_seconds=86400,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.08")
            ),
            RateLimitType.WORKFLOWS_PER_DAY: RateLimit(
                limit_type=RateLimitType.WORKFLOWS_PER_DAY,
                limit=2000,
                window_seconds=86400,
                overage_policy=OveragePolicy.CHARGE,
                overage_rate=Decimal("0.03")
            )
        },
        features={
            "unlimited": PricingFeature("Unlimited Everything", True, description="No limits on usage"),
            "custom_agents": PricingFeature("Advanced Custom Agents", True, description="AI agent marketplace access"),
            "dedicated_support": PricingFeature("Dedicated Support", True, description="Dedicated customer success manager"),
            "white_label": PricingFeature("White Label", True, description="Complete branding customization"),
            "sso": PricingFeature("Single Sign-On", True, description="SAML, OIDC, Active Directory"),
            "compliance": PricingFeature("Compliance Suite", True, description="SOC2, GDPR, HIPAA tools"),
            "custom_integrations": PricingFeature("Custom Integrations", True, description="Bespoke integration development"),
            "advanced_security": PricingFeature("Advanced Security", True, description="Enhanced security controls"),
            "audit_logs": PricingFeature("Audit Logs", True, description="Comprehensive audit trail"),
            "data_export": PricingFeature("Data Export", True, description="Full data portability")
        },
        max_users=100,
        max_custom_agents=-1,
        max_workflows=-1,
        max_integrations=-1,
        support_level="dedicated",
        support_response_time="1 hour",
        white_label=True,
        sso_enabled=True,
        compliance_features=["SOC2", "GDPR", "HIPAA"]
    ),

    PlanType.CUSTOM: PlanConfiguration(
        plan_type=PlanType.CUSTOM,
        monthly_fee=Decimal("2999.00"),  # Starting point
        included_tokens=10_000_000,
        token_overage_rate=Decimal("0.0005"),
        rate_limits={
            RateLimitType.REQUESTS_PER_MINUTE: RateLimit(
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                limit=10000,
                window_seconds=60,
                overage_policy=OveragePolicy.NOTIFY
            ),
            RateLimitType.TOKENS_PER_HOUR: RateLimit(
                limit_type=RateLimitType.TOKENS_PER_HOUR,
                limit=1_000_000,
                window_seconds=3600,
                overage_policy=OveragePolicy.NOTIFY
            )
        },
        features={
            "everything": PricingFeature("Everything Included", True, description="All features and capabilities"),
            "custom_development": PricingFeature("Custom Development", True, description="Dedicated development team"),
            "dedicated_infrastructure": PricingFeature("Dedicated Infrastructure", True, description="Isolated cloud resources"),
            "24_7_support": PricingFeature("24/7 Support", True, description="Round-the-clock dedicated support"),
            "onsite_training": PricingFeature("On-site Training", True, description="Professional services and training"),
            "custom_sla": PricingFeature("Custom SLA", True, description="Negotiated service level agreements")
        },
        max_users=1000,
        max_custom_agents=-1,
        max_workflows=-1,
        max_integrations=-1,
        support_level="24x7",
        support_response_time="15 minutes",
        white_label=True,
        sso_enabled=True,
        compliance_features=["SOC2", "GDPR", "HIPAA", "FedRAMP"],
        custom_development=True
    )
}

class PricingEngine:
    """Dynamic pricing and rate limiting engine"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis, usage_tracker: UsageTracker):
        self.db_pool = db_pool
        self.redis = redis_client
        self.usage_tracker = usage_tracker
        self.configurations = PRICING_CONFIGURATIONS

    async def get_plan_configuration(self, tenant_id: str) -> PlanConfiguration:
        """Get pricing configuration for a tenant"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT plan_type, settings FROM tenant_management.tenants
                WHERE id = $1
            """, tenant_id)

            if not row:
                raise HTTPException(status_code=404, detail="Tenant not found")

            plan_type = PlanType(row['plan_type'])
            base_config = self.configurations[plan_type]

            # Apply any custom settings
            settings = json.loads(row['settings'] or '{}')
            custom_config = settings.get('pricing_config', {})

            # Create a copy and apply customizations
            config = base_config
            if custom_config:
                # Apply custom rate limits, features, etc.
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            return config

    async def check_rate_limit(self, tenant_id: str, limit_type: RateLimitType,
                              amount: int = 1) -> Dict[str, Any]:
        """Check and enforce rate limits for a tenant"""

        config = await self.get_plan_configuration(tenant_id)

        if limit_type not in config.rate_limits:
            # No limit configured for this type
            return {
                'allowed': True,
                'limit_type': limit_type.value,
                'remaining': float('inf'),
                'reset_time': None
            }

        rate_limit = config.rate_limits[limit_type]

        # Create Redis key for this limit
        now = datetime.now()
        window_start = int(now.timestamp() // rate_limit.window_seconds) * rate_limit.window_seconds
        window_key = f"rate_limit:{tenant_id}:{limit_type.value}:{window_start}"

        # Get current usage in this window
        current_usage = await self.redis.get(window_key) or 0
        current_usage = int(current_usage)

        # Calculate effective limit (including burst allowance)
        effective_limit = int(rate_limit.limit * rate_limit.burst_allowance)

        # Check if request would exceed limit
        new_usage = current_usage + amount

        if new_usage > effective_limit:
            # Rate limit exceeded
            if rate_limit.overage_policy == OveragePolicy.BLOCK:
                return {
                    'allowed': False,
                    'limit_type': limit_type.value,
                    'limit': rate_limit.limit,
                    'current_usage': current_usage,
                    'remaining': max(0, effective_limit - current_usage),
                    'reset_time': datetime.fromtimestamp(window_start + rate_limit.window_seconds).isoformat(),
                    'retry_after': window_start + rate_limit.window_seconds - int(now.timestamp())
                }

            elif rate_limit.overage_policy == OveragePolicy.CHARGE:
                # Calculate overage charges
                overage_amount = new_usage - rate_limit.limit
                overage_cost = Decimal(str(overage_amount)) * (rate_limit.overage_rate or Decimal("0"))

                # Record overage charge
                await self._record_overage_charge(tenant_id, limit_type, overage_amount, overage_cost)

                # Allow request with charges
                await self.redis.setex(window_key, rate_limit.window_seconds, new_usage)

                return {
                    'allowed': True,
                    'limit_type': limit_type.value,
                    'limit': rate_limit.limit,
                    'current_usage': new_usage,
                    'overage_amount': overage_amount,
                    'overage_cost': float(overage_cost),
                    'remaining': 0,
                    'reset_time': datetime.fromtimestamp(window_start + rate_limit.window_seconds).isoformat()
                }

            elif rate_limit.overage_policy == OveragePolicy.NOTIFY:
                # Allow but send notifications
                await self._send_overage_notification(tenant_id, limit_type, new_usage, rate_limit.limit)

        # Update usage counter
        await self.redis.setex(window_key, rate_limit.window_seconds, new_usage)

        return {
            'allowed': True,
            'limit_type': limit_type.value,
            'limit': rate_limit.limit,
            'current_usage': new_usage,
            'remaining': max(0, effective_limit - new_usage),
            'reset_time': datetime.fromtimestamp(window_start + rate_limit.window_seconds).isoformat()
        }

    async def _record_overage_charge(self, tenant_id: str, limit_type: RateLimitType,
                                   amount: int, cost: Decimal):
        """Record an overage charge in the database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.usage_events (
                    tenant_id, event_type, service_type, tokens_used,
                    calculated_cost_usd, request_metadata, created_at, billing_period_start
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                tenant_id, "overage_charge", f"rate_limit_{limit_type.value}",
                amount, cost, json.dumps({
                    'limit_type': limit_type.value,
                    'overage_amount': amount,
                    'rate': float(cost / Decimal(str(amount))) if amount > 0 else 0
                }),
                datetime.now(),
                datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
            )

        logger.info(f"Recorded overage charge: {tenant_id} - {limit_type.value} - ${cost}")

    async def _send_overage_notification(self, tenant_id: str, limit_type: RateLimitType,
                                       usage: int, limit: int):
        """Send notification about rate limit overage"""

        # Store notification for tenant dashboard
        notification_key = f"notifications:{tenant_id}"
        notification = {
            'type': 'rate_limit_exceeded',
            'limit_type': limit_type.value,
            'usage': usage,
            'limit': limit,
            'timestamp': datetime.now().isoformat(),
            'message': f"Rate limit exceeded for {limit_type.value}: {usage}/{limit}"
        }

        await self.redis.lpush(notification_key, json.dumps(notification))
        await self.redis.ltrim(notification_key, 0, 99)  # Keep last 100 notifications
        await self.redis.expire(notification_key, 86400 * 7)  # Expire after 7 days

        logger.warning(f"Rate limit overage: {tenant_id} - {limit_type.value} - {usage}/{limit}")

    async def calculate_dynamic_pricing(self, tenant_id: str, service_type: ServiceType,
                                      token_count: int) -> Dict[str, Any]:
        """Calculate dynamic pricing based on usage patterns and demand"""

        config = await self.get_plan_configuration(tenant_id)

        # Get historical usage patterns
        usage_summary = await self.usage_tracker.get_usage_summary(tenant_id, "current_month")

        # Base pricing
        base_rate = config.token_overage_rate

        # Dynamic adjustments
        dynamic_multiplier = Decimal("1.0")

        # Volume discount (more usage = lower rate)
        monthly_tokens = usage_summary['summary']['total_tokens']
        if monthly_tokens > 1_000_000:
            dynamic_multiplier *= Decimal("0.9")  # 10% discount
        elif monthly_tokens > 5_000_000:
            dynamic_multiplier *= Decimal("0.8")  # 20% discount

        # Service type adjustments
        if service_type in [ServiceType.OPENAI_GPT4, ServiceType.ANTHROPIC_CLAUDE3_OPUS]:
            dynamic_multiplier *= Decimal("1.1")  # Premium models cost more
        elif service_type in [ServiceType.OPENAI_GPT35_TURBO, ServiceType.GOOGLE_GEMINI_PRO]:
            dynamic_multiplier *= Decimal("0.9")  # Cheaper models cost less

        # Time-based pricing (lower rates during off-peak hours)
        current_hour = datetime.now().hour
        if 2 <= current_hour <= 6:  # Off-peak hours
            dynamic_multiplier *= Decimal("0.85")  # 15% discount
        elif 9 <= current_hour <= 17:  # Peak business hours
            dynamic_multiplier *= Decimal("1.05")  # 5% premium

        # Calculate final price
        final_rate = base_rate * dynamic_multiplier
        total_cost = final_rate * Decimal(str(token_count))

        return {
            'base_rate': float(base_rate),
            'dynamic_multiplier': float(dynamic_multiplier),
            'final_rate': float(final_rate),
            'token_count': token_count,
            'total_cost': float(total_cost),
            'discounts_applied': {
                'volume_discount': monthly_tokens > 1_000_000,
                'off_peak_discount': 2 <= current_hour <= 6,
                'service_adjustment': float(dynamic_multiplier) != 1.0
            }
        }

    async def get_plan_comparison(self) -> Dict[str, Any]:
        """Get a comparison of all available plans"""

        comparison = {}

        for plan_type, config in self.configurations.items():
            comparison[plan_type.value] = {
                'monthly_fee': float(config.monthly_fee),
                'included_tokens': config.included_tokens,
                'token_overage_rate': float(config.token_overage_rate),
                'max_users': config.max_users,
                'max_custom_agents': config.max_custom_agents,
                'max_workflows': config.max_workflows,
                'max_integrations': config.max_integrations,
                'support_level': config.support_level,
                'support_response_time': config.support_response_time,
                'white_label': config.white_label,
                'sso_enabled': config.sso_enabled,
                'features': {
                    name: {
                        'enabled': feature.enabled,
                        'limit': feature.limit,
                        'description': feature.description
                    } for name, feature in config.features.items()
                },
                'rate_limits': {
                    limit_type.value: {
                        'limit': rate_limit.limit,
                        'window_seconds': rate_limit.window_seconds,
                        'overage_policy': rate_limit.overage_policy.value,
                        'overage_rate': float(rate_limit.overage_rate) if rate_limit.overage_rate else None
                    } for limit_type, rate_limit in config.rate_limits.items()
                }
            }

        return comparison

    async def check_feature_access(self, tenant_id: str, feature_name: str) -> Dict[str, Any]:
        """Check if tenant has access to a specific feature"""

        config = await self.get_plan_configuration(tenant_id)

        if feature_name not in config.features:
            return {
                'allowed': False,
                'reason': 'Feature not available in any plan'
            }

        feature = config.features[feature_name]

        if not feature.enabled:
            return {
                'allowed': False,
                'reason': 'Feature not available in current plan',
                'upgrade_required': True
            }

        # Check feature usage limits
        if feature.limit is not None:
            # Get current usage for this feature
            current_usage = await self._get_feature_usage(tenant_id, feature_name)

            if current_usage >= feature.limit:
                return {
                    'allowed': False,
                    'reason': f'Feature limit exceeded: {current_usage}/{feature.limit}',
                    'current_usage': current_usage,
                    'limit': feature.limit,
                    'upgrade_required': True
                }

        return {
            'allowed': True,
            'feature': feature_name,
            'current_usage': await self._get_feature_usage(tenant_id, feature_name),
            'limit': feature.limit
        }

    async def _get_feature_usage(self, tenant_id: str, feature_name: str) -> int:
        """Get current usage count for a feature"""

        feature_usage_map = {
            'custom_agents': 'SELECT COUNT(*) FROM tenant_management.custom_agents WHERE tenant_id = $1 AND is_active = true',
            'workflows': 'SELECT COUNT(*) FROM tenant_management.workflows WHERE tenant_id = $1 AND is_active = true',
            'integrations': 'SELECT COUNT(*) FROM tenant_management.integrations WHERE tenant_id = $1 AND is_active = true',
            'webhooks': 'SELECT COUNT(*) FROM tenant_management.webhook_endpoints WHERE tenant_id = $1 AND is_active = true'
        }

        if feature_name not in feature_usage_map:
            return 0

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval(feature_usage_map[feature_name], tenant_id)
            return result or 0

    async def suggest_plan_upgrade(self, tenant_id: str) -> Dict[str, Any]:
        """Suggest plan upgrades based on usage patterns"""

        config = await self.get_plan_configuration(tenant_id)
        current_plan = config.plan_type

        # Get usage data
        usage_summary = await self.usage_tracker.get_usage_summary(tenant_id, "current_month")

        # Check rate limit violations
        violations = []
        for limit_type in config.rate_limits:
            limit_key = f"rate_limit_violations:{tenant_id}:{limit_type.value}"
            violation_count = await self.redis.get(limit_key) or 0
            if int(violation_count) > 5:  # More than 5 violations this month
                violations.append(limit_type.value)

        # Calculate usage metrics
        token_usage_ratio = usage_summary['summary']['total_tokens'] / config.included_tokens

        suggestions = []

        # Token usage based suggestions
        if token_usage_ratio > 0.8:
            if current_plan == PlanType.STARTER:
                suggestions.append({
                    'type': 'plan_upgrade',
                    'current_plan': current_plan.value,
                    'suggested_plan': PlanType.PROFESSIONAL.value,
                    'reason': 'High token usage approaching limit',
                    'potential_savings': self._calculate_upgrade_savings(
                        current_plan, PlanType.PROFESSIONAL, usage_summary
                    )
                })
            elif current_plan == PlanType.PROFESSIONAL:
                suggestions.append({
                    'type': 'plan_upgrade',
                    'current_plan': current_plan.value,
                    'suggested_plan': PlanType.ENTERPRISE.value,
                    'reason': 'High token usage with significant overages',
                    'potential_savings': self._calculate_upgrade_savings(
                        current_plan, PlanType.ENTERPRISE, usage_summary
                    )
                })

        # Rate limit violation suggestions
        if violations:
            next_plan = self._get_next_plan_tier(current_plan)
            if next_plan:
                suggestions.append({
                    'type': 'plan_upgrade',
                    'current_plan': current_plan.value,
                    'suggested_plan': next_plan.value,
                    'reason': f'Rate limit violations in: {", ".join(violations)}',
                    'violations': violations
                })

        # Feature usage suggestions
        feature_usage = {}
        for feature_name in config.features:
            feature_usage[feature_name] = await self._get_feature_usage(tenant_id, feature_name)

        # Check if approaching feature limits
        for feature_name, usage in feature_usage.items():
            feature = config.features[feature_name]
            if feature.limit and usage >= feature.limit * 0.8:
                next_plan = self._get_next_plan_tier(current_plan)
                if next_plan and feature_name in self.configurations[next_plan].features:
                    next_feature = self.configurations[next_plan].features[feature_name]
                    if not next_feature.limit or next_feature.limit > feature.limit:
                        suggestions.append({
                            'type': 'feature_upgrade',
                            'feature': feature_name,
                            'current_limit': feature.limit,
                            'usage': usage,
                            'suggested_plan': next_plan.value
                        })

        return {
            'current_plan': current_plan.value,
            'usage_metrics': {
                'token_usage_ratio': token_usage_ratio,
                'monthly_tokens': usage_summary['summary']['total_tokens'],
                'monthly_cost': usage_summary['summary']['total_customer_cost'],
                'rate_limit_violations': len(violations)
            },
            'suggestions': suggestions,
            'feature_usage': feature_usage
        }

    def _calculate_upgrade_savings(self, current_plan: PlanType, suggested_plan: PlanType,
                                 usage_summary: Dict[str, Any]) -> float:
        """Calculate potential savings from plan upgrade"""

        current_config = self.configurations[current_plan]
        suggested_config = self.configurations[suggested_plan]

        monthly_tokens = usage_summary['summary']['total_tokens']

        # Current plan cost
        current_base = float(current_config.monthly_fee)
        current_overage = max(0, monthly_tokens - current_config.included_tokens) * float(current_config.token_overage_rate)
        current_total = current_base + current_overage

        # Suggested plan cost
        suggested_base = float(suggested_config.monthly_fee)
        suggested_overage = max(0, monthly_tokens - suggested_config.included_tokens) * float(suggested_config.token_overage_rate)
        suggested_total = suggested_base + suggested_overage

        return current_total - suggested_total

    def _get_next_plan_tier(self, current_plan: PlanType) -> Optional[PlanType]:
        """Get the next tier up from current plan"""

        tier_order = [PlanType.STARTER, PlanType.PROFESSIONAL, PlanType.ENTERPRISE, PlanType.CUSTOM]

        try:
            current_index = tier_order.index(current_plan)
            if current_index < len(tier_order) - 1:
                return tier_order[current_index + 1]
        except ValueError:
            pass

        return None

    async def get_billing_preview(self, tenant_id: str, target_plan: Optional[PlanType] = None) -> Dict[str, Any]:
        """Get billing preview for current or target plan"""

        current_config = await self.get_plan_configuration(tenant_id)
        target_config = self.configurations[target_plan] if target_plan else current_config

        # Get current month usage
        usage_summary = await self.usage_tracker.get_usage_summary(tenant_id, "current_month")
        monthly_tokens = usage_summary['summary']['total_tokens']

        # Calculate costs
        base_fee = float(target_config.monthly_fee)
        included_tokens = target_config.included_tokens
        overage_tokens = max(0, monthly_tokens - included_tokens)
        overage_cost = overage_tokens * float(target_config.token_overage_rate)

        subtotal = base_fee + overage_cost
        tax_rate = 0.08  # 8% tax
        tax_amount = subtotal * tax_rate
        total = subtotal + tax_amount

        # Get feature comparison
        feature_comparison = {}
        if target_plan and target_plan != current_config.plan_type:
            for feature_name in set(list(current_config.features.keys()) + list(target_config.features.keys())):
                current_feature = current_config.features.get(feature_name)
                target_feature = target_config.features.get(feature_name)

                feature_comparison[feature_name] = {
                    'current': {
                        'enabled': current_feature.enabled if current_feature else False,
                        'limit': current_feature.limit if current_feature else None
                    },
                    'target': {
                        'enabled': target_feature.enabled if target_feature else False,
                        'limit': target_feature.limit if target_feature else None
                    },
                    'improved': (
                        (target_feature and target_feature.enabled) and
                        (not current_feature or not current_feature.enabled or
                         (target_feature.limit or float('inf')) > (current_feature.limit or 0))
                    ) if target_feature else False
                }

        return {
            'plan': target_config.plan_type.value,
            'billing_preview': {
                'base_subscription_fee': base_fee,
                'included_tokens': included_tokens,
                'current_token_usage': monthly_tokens,
                'overage_tokens': overage_tokens,
                'overage_rate': float(target_config.token_overage_rate),
                'overage_cost': overage_cost,
                'subtotal': subtotal,
                'tax_rate': tax_rate,
                'tax_amount': tax_amount,
                'total': total
            },
            'feature_comparison': feature_comparison if target_plan else None,
            'savings_vs_current': (
                self._calculate_upgrade_savings(current_config.plan_type, target_plan, usage_summary)
                if target_plan and target_plan != current_config.plan_type else 0
            )
        }

# Rate limiting middleware
class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""

    def __init__(self, pricing_engine: PricingEngine):
        self.pricing_engine = pricing_engine

    async def __call__(self, request: Request, call_next):
        # Extract tenant ID from request (API key, JWT, etc.)
        tenant_id = await self._extract_tenant_id(request)

        if not tenant_id:
            # No tenant ID, skip rate limiting
            return await call_next(request)

        # Check rate limits
        rate_limit_result = await self.pricing_engine.check_rate_limit(
            tenant_id, RateLimitType.REQUESTS_PER_MINUTE
        )

        if not rate_limit_result['allowed']:
            return HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(rate_limit_result['limit']),
                    "X-RateLimit-Remaining": str(rate_limit_result['remaining']),
                    "X-RateLimit-Reset": str(rate_limit_result['reset_time']),
                    "Retry-After": str(rate_limit_result.get('retry_after', 60))
                }
            )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_limit_result['limit'])
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_result['remaining'])
        response.headers["X-RateLimit-Reset"] = str(rate_limit_result['reset_time'])

        return response

    async def _extract_tenant_id(self, request: Request) -> Optional[str]:
        """Extract tenant ID from request"""

        # Try API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Look up tenant by API key
            async with self.pricing_engine.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT tenant_id FROM tenant_management.tenant_api_keys
                    WHERE api_key_hash = $1 AND is_active = true
                """, api_key)  # In practice, you'd hash the API key

                if row:
                    return row['tenant_id']

        # Try JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Decode JWT and extract tenant_id
            # Implementation depends on your JWT setup
            pass

        # Try subdomain
        host = request.headers.get("Host", "")
        if "." in host:
            subdomain = host.split(".")[0]
            async with self.pricing_engine.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id FROM tenant_management.tenants
                    WHERE subdomain = $1 AND status = 'active'
                """, subdomain)

                if row:
                    return row['id']

        return None

# Pydantic models for API
class PlanUpgradeRequest(BaseModel):
    target_plan: PlanType

class RateLimitCheckRequest(BaseModel):
    limit_type: RateLimitType
    amount: int = 1

class FeatureAccessRequest(BaseModel):
    feature_name: str

class BillingPreviewRequest(BaseModel):
    target_plan: Optional[PlanType] = None

# Export main classes
__all__ = [
    'PricingEngine', 'PlanConfiguration', 'RateLimit', 'RateLimitType',
    'OveragePolicy', 'PricingFeature', 'PRICING_CONFIGURATIONS',
    'RateLimitMiddleware', 'PlanUpgradeRequest', 'RateLimitCheckRequest',
    'FeatureAccessRequest', 'BillingPreviewRequest'
]
