"""
📊 AgentSystem Usage Tracking System
Real-time token usage monitoring across all AI providers with cost calculation
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
from contextlib import asynccontextmanager
from ..usage.overage_billing import OverageBilling

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(str, Enum):
    # OpenAI Services
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT4_TURBO = "openai_gpt4_turbo"
    OPENAI_GPT35_TURBO = "openai_gpt35_turbo"
    OPENAI_DALLE3 = "openai_dalle3"
    OPENAI_WHISPER = "openai_whisper"
    OPENAI_TTS = "openai_tts"

    # Anthropic Services
    ANTHROPIC_CLAUDE3_OPUS = "anthropic_claude3_opus"
    ANTHROPIC_CLAUDE3_SONNET = "anthropic_claude3_sonnet"
    ANTHROPIC_CLAUDE3_HAIKU = "anthropic_claude3_haiku"

    # Google Services
    GOOGLE_GEMINI_PRO = "google_gemini_pro"
    GOOGLE_GEMINI_ULTRA = "google_gemini_ultra"

    # Specialized Services
    ELEVENLABS_TTS = "elevenlabs_tts"
    STABILITY_AI = "stability_ai"
    COHERE = "cohere"

    # Agent Services
    AGENT_SWARM = "agent_swarm"
    CUSTOM_AGENT = "custom_agent"
    WORKFLOW_EXECUTION = "workflow_execution"

@dataclass
class ProviderCosts:
    """Cost structure for AI providers"""
    input_cost_per_1k: Decimal  # Cost per 1K input tokens
    output_cost_per_1k: Decimal  # Cost per 1K output tokens
    image_cost_per_unit: Decimal = Decimal("0")  # For image generation
    audio_cost_per_minute: Decimal = Decimal("0")  # For audio processing

    # Our markup rates (how much we charge customers)
    markup_multiplier: Decimal = Decimal("2.5")  # 150% markup

    def calculate_cost(self, input_tokens: int, output_tokens: int,
                      images: int = 0, audio_minutes: float = 0) -> Tuple[Decimal, Decimal]:
        """Calculate provider cost and customer cost"""

        # Provider cost calculation
        provider_cost = (
            (Decimal(input_tokens) / Decimal("1000")) * self.input_cost_per_1k +
            (Decimal(output_tokens) / Decimal("1000")) * self.output_cost_per_1k +
            Decimal(images) * self.image_cost_per_unit +
            Decimal(str(audio_minutes)) * self.audio_cost_per_minute
        )

        # Customer cost (with markup)
        customer_cost = provider_cost * self.markup_multiplier

        return provider_cost, customer_cost

# Cost structure for all providers
PROVIDER_COSTS = {
    ServiceType.OPENAI_GPT4: ProviderCosts(
        input_cost_per_1k=Decimal("0.03"),
        output_cost_per_1k=Decimal("0.06"),
        markup_multiplier=Decimal("2.0")  # Lower markup for premium models
    ),
    ServiceType.OPENAI_GPT4_TURBO: ProviderCosts(
        input_cost_per_1k=Decimal("0.01"),
        output_cost_per_1k=Decimal("0.03"),
        markup_multiplier=Decimal("2.2")
    ),
    ServiceType.OPENAI_GPT35_TURBO: ProviderCosts(
        input_cost_per_1k=Decimal("0.0015"),
        output_cost_per_1k=Decimal("0.002"),
        markup_multiplier=Decimal("3.0")  # Higher markup for cheaper models
    ),
    ServiceType.OPENAI_DALLE3: ProviderCosts(
        input_cost_per_1k=Decimal("0"),
        output_cost_per_1k=Decimal("0"),
        image_cost_per_unit=Decimal("0.04"),  # Per image
        markup_multiplier=Decimal("2.5")
    ),
    ServiceType.OPENAI_WHISPER: ProviderCosts(
        input_cost_per_1k=Decimal("0"),
        output_cost_per_1k=Decimal("0"),
        audio_cost_per_minute=Decimal("0.006"),
        markup_multiplier=Decimal("3.0")
    ),
    ServiceType.OPENAI_TTS: ProviderCosts(
        input_cost_per_1k=Decimal("0.015"),
        output_cost_per_1k=Decimal("0"),
        markup_multiplier=Decimal("2.8")
    ),
    ServiceType.ANTHROPIC_CLAUDE3_OPUS: ProviderCosts(
        input_cost_per_1k=Decimal("0.015"),
        output_cost_per_1k=Decimal("0.075"),
        markup_multiplier=Decimal("2.0")
    ),
    ServiceType.ANTHROPIC_CLAUDE3_SONNET: ProviderCosts(
        input_cost_per_1k=Decimal("0.003"),
        output_cost_per_1k=Decimal("0.015"),
        markup_multiplier=Decimal("2.3")
    ),
    ServiceType.ANTHROPIC_CLAUDE3_HAIKU: ProviderCosts(
        input_cost_per_1k=Decimal("0.00025"),
        output_cost_per_1k=Decimal("0.00125"),
        markup_multiplier=Decimal("4.0")
    ),
    ServiceType.GOOGLE_GEMINI_PRO: ProviderCosts(
        input_cost_per_1k=Decimal("0.001"),
        output_cost_per_1k=Decimal("0.002"),
        markup_multiplier=Decimal("3.5")
    ),
    ServiceType.ELEVENLABS_TTS: ProviderCosts(
        input_cost_per_1k=Decimal("0.30"),  # Per 1K characters
        output_cost_per_1k=Decimal("0"),
        markup_multiplier=Decimal("2.2")
    ),
    ServiceType.STABILITY_AI: ProviderCosts(
        input_cost_per_1k=Decimal("0"),
        output_cost_per_1k=Decimal("0"),
        image_cost_per_unit=Decimal("0.002"),
        markup_multiplier=Decimal("5.0")  # High markup for image generation
    )
}

@dataclass
class UsageEvent:
    """Individual usage event"""
    tenant_id: str
    service_type: ServiceType
    event_type: str  # api_request, agent_execution, workflow_run

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Additional metrics
    images_generated: int = 0
    audio_minutes: float = 0
    processing_time_ms: int = 0

    # Cost information
    provider_cost_usd: Decimal = Decimal("0")
    customer_cost_usd: Decimal = Decimal("0")

    # Metadata
    request_id: str = ""
    user_id: str = ""
    endpoint: str = ""
    model_name: str = ""
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)

    def calculate_costs(self) -> Tuple[Decimal, Decimal]:
        """Calculate provider and customer costs for this event"""
        if self.service_type in PROVIDER_COSTS:
            cost_config = PROVIDER_COSTS[self.service_type]
            return cost_config.calculate_cost(
                self.input_tokens, self.output_tokens,
                self.images_generated, self.audio_minutes
            )
        return Decimal("0"), Decimal("0")

class UsageTracker:
    """Real-time usage tracking system"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis, overage_billing: Optional[OverageBilling] = None):
        self.db_pool = db_pool
        self.redis = redis_client
        self.overage_billing = overage_billing
        self.buffer = defaultdict(list)  # Buffered events for batch processing
        self.buffer_size = 100  # Events before batch write
        self.flush_interval = 30  # Seconds between flushes
        self._running = False
        self._flush_task = None

    async def start(self):
        """Start the usage tracker with background flushing"""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Usage tracker started")

    async def stop(self):
        """Stop the usage tracker and flush remaining events"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_events()
        logger.info("Usage tracker stopped")

    async def track_usage(self, event: UsageEvent) -> Dict[str, Any]:
        """Track a usage event"""

        # Calculate costs
        if event.total_tokens == 0:
            event.total_tokens = event.input_tokens + event.output_tokens

        provider_cost, customer_cost = event.calculate_costs()
        event.provider_cost_usd = provider_cost
        event.customer_cost_usd = customer_cost

        # Add to buffer
        self.buffer[event.tenant_id].append(event)

        # Real-time metrics update in Redis
        # Real-time metrics update in Redis
        await self._update_realtime_metrics(event)

        # Check if we should flush this tenant's buffer
        if len(self.buffer[event.tenant_id]) >= self.buffer_size:
            await self._flush_tenant_events(event.tenant_id)

        # Check rate limits
        rate_limit_info = await self._check_rate_limits(event.tenant_id, event.service_type)

        # Track overage for tasks if on Pro plan
        if event.event_type in ["agent_execution", "workflow_run"] and self.overage_billing:
            plan_type = rate_limit_info.get('plan_type', '')
            if plan_type == 'Pro':
                await self.overage_billing.track_overage_task(event.tenant_id, event.request_id, plan_type)
        return {
            'event_id': event.request_id,
            'provider_cost': float(provider_cost),
            'customer_cost': float(customer_cost),
            'total_tokens': event.total_tokens,
            'rate_limit_info': rate_limit_info,
            'timestamp': event.timestamp.isoformat()
        }

    async def _update_realtime_metrics(self, event: UsageEvent):
        """Update real-time metrics in Redis"""

        # Current month key
        month_key = f"usage:{event.tenant_id}:{event.timestamp.strftime('%Y-%m')}"

        # Increment counters
        pipe = self.redis.pipeline()
        pipe.hincrby(month_key, 'total_tokens', event.total_tokens)
        pipe.hincrby(month_key, 'total_requests', 1)
        pipe.hincrbyfloat(month_key, 'total_cost', float(event.customer_cost_usd))
        pipe.hincrbyfloat(month_key, 'provider_cost', float(event.provider_cost_usd))
        pipe.hincrby(month_key, f'tokens_{event.service_type.value}', event.total_tokens)
        pipe.expire(month_key, 86400 * 35)  # Expire after 35 days

        # Daily usage key
        day_key = f"usage_daily:{event.tenant_id}:{event.timestamp.strftime('%Y-%m-%d')}"
        pipe.hincrby(day_key, 'total_tokens', event.total_tokens)
        pipe.hincrby(day_key, 'total_requests', 1)
        pipe.hincrbyfloat(day_key, 'total_cost', float(event.customer_cost_usd))
        pipe.expire(day_key, 86400 * 7)  # Expire after 7 days

        # Rate limiting counters
        minute_key = f"rate_limit:{event.tenant_id}:{event.timestamp.strftime('%Y-%m-%d-%H-%M')}"
        pipe.incr(minute_key)
        pipe.expire(minute_key, 300)  # 5 minutes

        await pipe.execute()

    async def _check_rate_limits(self, tenant_id: str, service_type: ServiceType) -> Dict[str, Any]:
        """Check if tenant is approaching or exceeding rate limits"""

        # Get tenant limits
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT api_rate_limit, plan_type, monthly_token_limit
                FROM tenant_management.tenants
                WHERE id = $1
            """, tenant_id)

            if not row:
                return {'status': 'error', 'message': 'Tenant not found'}

            api_rate_limit = row['api_rate_limit']
            plan_type = row['plan_type']
            monthly_token_limit = row['monthly_token_limit']

        # Check current minute usage
        current_minute = datetime.now().strftime('%Y-%m-%d-%H-%M')
        minute_key = f"rate_limit:{tenant_id}:{current_minute}"
        current_requests = await self.redis.get(minute_key) or 0
        current_requests = int(current_requests)

        # Check monthly token usage
        month_key = f"usage:{tenant_id}:{datetime.now().strftime('%Y-%m')}"
        monthly_usage = await self.redis.hget(month_key, 'total_tokens') or 0
        monthly_usage = int(monthly_usage)

        # Calculate limits
        rate_limit_exceeded = current_requests >= api_rate_limit
        token_limit_exceeded = monthly_usage >= monthly_token_limit

        # Warning thresholds
        rate_warning = current_requests >= (api_rate_limit * 0.8)
        token_warning = monthly_usage >= (monthly_token_limit * 0.8)

        return {
            'rate_limit': {
                'current': current_requests,
                'limit': api_rate_limit,
                'exceeded': rate_limit_exceeded,
                'warning': rate_warning
            },
            'token_limit': {
                'current': monthly_usage,
                'limit': monthly_token_limit,
                'exceeded': token_limit_exceeded,
                'warning': token_warning,
                'usage_percentage': (monthly_usage / monthly_token_limit * 100) if monthly_token_limit > 0 else 0
            },
            'plan_type': plan_type
        }

    async def _flush_loop(self):
        """Background task to flush events periodically"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")

    async def _flush_events(self):
        """Flush all buffered events to database"""
        for tenant_id in list(self.buffer.keys()):
            await self._flush_tenant_events(tenant_id)

    async def _flush_tenant_events(self, tenant_id: str):
        """Flush events for a specific tenant"""
        if tenant_id not in self.buffer or not self.buffer[tenant_id]:
            return

        events = self.buffer[tenant_id].copy()
        self.buffer[tenant_id].clear()

        if not events:
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Batch insert events
                await conn.executemany("""
                    INSERT INTO tenant_management.usage_events (
                        tenant_id, event_type, service_type, tokens_used, requests_count,
                        processing_time_ms, provider_cost_usd, calculated_cost_usd,
                        request_metadata, response_metadata, created_at, billing_period_start
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, [
                    (
                        event.tenant_id, event.event_type, event.service_type.value,
                        event.total_tokens, 1, event.processing_time_ms,
                        event.provider_cost_usd, event.customer_cost_usd,
                        json.dumps(event.request_metadata),
                        json.dumps(event.response_metadata),
                        event.timestamp,
                        event.timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0).date()
                    ) for event in events
                ])

                logger.info(f"Flushed {len(events)} usage events for tenant {tenant_id}")

        except Exception as e:
            logger.error(f"Error flushing events for tenant {tenant_id}: {e}")
            # Re-add events to buffer for retry
            self.buffer[tenant_id].extend(events)

    async def get_usage_summary(self, tenant_id: str, period: str = "current_month") -> Dict[str, Any]:
        """Get usage summary for a tenant"""

        now = datetime.now()

        if period == "current_month":
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        elif period == "last_month":
            last_month = now.replace(month=now.month-1) if now.month > 1 else now.replace(year=now.year-1, month=12)
            start_date = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == "last_7_days":
            start_date = now - timedelta(days=7)
            end_date = now
        else:
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now

        async with self.db_pool.acquire() as conn:
            # Get detailed usage breakdown
            usage_rows = await conn.fetch("""
                SELECT
                    service_type,
                    SUM(tokens_used) as total_tokens,
                    COUNT(*) as total_requests,
                    SUM(provider_cost_usd) as total_provider_cost,
                    SUM(calculated_cost_usd) as total_customer_cost,
                    AVG(processing_time_ms) as avg_processing_time
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= $2
                    AND created_at <= $3
                GROUP BY service_type
                ORDER BY total_tokens DESC
            """, tenant_id, start_date, end_date)

            # Get total summary
            total_row = await conn.fetchrow("""
                SELECT
                    SUM(tokens_used) as total_tokens,
                    COUNT(*) as total_requests,
                    SUM(provider_cost_usd) as total_provider_cost,
                    SUM(calculated_cost_usd) as total_customer_cost
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= $2
                    AND created_at <= $3
            """, tenant_id, start_date, end_date)

        # Build response
        breakdown = []
        for row in usage_rows:
            breakdown.append({
                'service_type': row['service_type'],
                'total_tokens': row['total_tokens'] or 0,
                'total_requests': row['total_requests'] or 0,
                'provider_cost': float(row['total_provider_cost'] or 0),
                'customer_cost': float(row['total_customer_cost'] or 0),
                'avg_processing_time': float(row['avg_processing_time'] or 0)
            })

        return {
            'period': period,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'summary': {
                'total_tokens': total_row['total_tokens'] or 0,
                'total_requests': total_row['total_requests'] or 0,
                'total_provider_cost': float(total_row['total_provider_cost'] or 0),
                'total_customer_cost': float(total_row['total_customer_cost'] or 0),
                'profit_margin': float(
                    (total_row['total_customer_cost'] or 0) - (total_row['total_provider_cost'] or 0)
                )
            },
            'breakdown_by_service': breakdown
        }

    async def get_cost_optimization_suggestions(self, tenant_id: str) -> Dict[str, Any]:
        """Analyze usage patterns and suggest cost optimizations"""

        # Get last 30 days of usage
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    service_type,
                    SUM(tokens_used) as total_tokens,
                    COUNT(*) as total_requests,
                    SUM(provider_cost_usd) as total_cost,
                    AVG(processing_time_ms) as avg_time
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= $2
                    AND created_at <= $3
                GROUP BY service_type
                ORDER BY total_cost DESC
            """, tenant_id, start_date, end_date)

        suggestions = []
        total_potential_savings = Decimal("0")

        for row in rows:
            service_type = ServiceType(row['service_type'])
            total_cost = Decimal(str(row['total_cost'] or 0))
            total_tokens = row['total_tokens'] or 0
            avg_time = row['avg_time'] or 0

            # Analyze patterns and suggest alternatives
            if service_type == ServiceType.OPENAI_GPT4 and total_tokens > 100000:
                # Suggest GPT-4 Turbo for high volume
                potential_savings = total_cost * Decimal("0.67")  # ~33% savings
                suggestions.append({
                    'type': 'model_downgrade',
                    'current_service': service_type.value,
                    'suggested_service': ServiceType.OPENAI_GPT4_TURBO.value,
                    'potential_monthly_savings': float(potential_savings),
                    'confidence': 0.85,
                    'reason': 'High volume usage could benefit from GPT-4 Turbo pricing'
                })
                total_potential_savings += potential_savings

            elif avg_time > 5000 and total_tokens < 50000:
                # Suggest caching for slow, low-volume requests
                cache_savings = total_cost * Decimal("0.4")  # 40% savings from caching
                suggestions.append({
                    'type': 'caching',
                    'service': service_type.value,
                    'potential_monthly_savings': float(cache_savings),
                    'confidence': 0.7,
                    'reason': 'Slow response times suggest potential for aggressive caching'
                })
                total_potential_savings += cache_savings

            elif service_type in [ServiceType.ANTHROPIC_CLAUDE3_OPUS] and total_tokens > 200000:
                # Suggest Claude Sonnet for high volume
                sonnet_savings = total_cost * Decimal("0.8")  # 80% savings
                suggestions.append({
                    'type': 'model_downgrade',
                    'current_service': service_type.value,
                    'suggested_service': ServiceType.ANTHROPIC_CLAUDE3_SONNET.value,
                    'potential_monthly_savings': float(sonnet_savings),
                    'confidence': 0.75,
                    'reason': 'Claude Sonnet offers similar performance at much lower cost'
                })
                total_potential_savings += sonnet_savings

        return {
            'total_potential_monthly_savings': float(total_potential_savings),
            'suggestions': suggestions,
            'analysis_period_days': 30,
            'confidence_threshold': 0.7
        }

# Context manager for usage tracking
@asynccontextmanager
async def track_ai_request(tracker: UsageTracker, tenant_id: str, service_type: ServiceType,
                          event_type: str = "api_request", **metadata):
    """Context manager to automatically track AI requests"""

    start_time = time.time()
    event = UsageEvent(
        tenant_id=tenant_id,
        service_type=service_type,
        event_type=event_type,
        request_metadata=metadata,
        timestamp=datetime.now()
    )

    try:
        yield event
    finally:
        # Calculate processing time
        event.processing_time_ms = int((time.time() - start_time) * 1000)

        # Track the usage
        await tracker.track_usage(event)

# Decorator for automatic usage tracking
def track_usage_decorator(service_type: ServiceType, event_type: str = "api_request"):
    """Decorator to automatically track usage for functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract tracker and tenant_id from arguments or kwargs
            tracker = kwargs.get('usage_tracker') or getattr(args[0], 'usage_tracker', None)
            tenant_id = kwargs.get('tenant_id') or getattr(args[0], 'tenant_id', None)

            if not tracker or not tenant_id:
                # If no tracker or tenant_id, just run the function normally
                return await func(*args, **kwargs)

            async with track_ai_request(tracker, tenant_id, service_type, event_type) as event:
                result = await func(*args, **kwargs)

                # Try to extract token usage from result
                if isinstance(result, dict):
                    if 'usage' in result:
                        usage = result['usage']
                        event.input_tokens = usage.get('prompt_tokens', 0)
                        event.output_tokens = usage.get('completion_tokens', 0)
                        event.total_tokens = usage.get('total_tokens', 0)

                    # Store response metadata
                    event.response_metadata = {
                        'model': result.get('model', ''),
                        'finish_reason': result.get('choices', [{}])[0].get('finish_reason', ''),
                        'response_id': result.get('id', '')
                    }

                return result
        return wrapper
    return decorator

# Export main classes and utilities
__all__ = [
    'UsageTracker', 'UsageEvent', 'ServiceType', 'ProviderCosts', 'PROVIDER_COSTS',
    'track_ai_request', 'track_usage_decorator'
]

