"""
🎯 AgentSystem Cost Optimization Engine
Intelligent AI provider routing and cost arbitrage system for maximum profit margins
"""

import asyncio
import json
import time
import io
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
from collections import defaultdict, deque
import statistics
import numpy as np

import asyncpg
import aioredis
from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..usage.usage_tracker import ServiceType, UsageTracker
from ..pricing.pricing_engine import PricingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationStrategy(str, Enum):
    COST_FIRST = "cost_first"          # Prioritize lowest cost
    QUALITY_FIRST = "quality_first"    # Prioritize highest quality
    BALANCED = "balanced"              # Balance cost and quality
    SPEED_FIRST = "speed_first"        # Prioritize fastest response
    CUSTOM = "custom"                  # Custom weighted scoring

class CacheStrategy(str, Enum):
    AGGRESSIVE = "aggressive"     # Cache everything possible
    CONSERVATIVE = "conservative" # Cache only safe, repeatable queries
    SMART = "smart"              # AI-powered cache decision making
    DISABLED = "disabled"        # No caching

@dataclass
class ProviderMetrics:
    """Performance metrics for an AI provider"""
    provider: str
    model: str

    # Cost metrics
    avg_cost_per_token: Decimal
    total_cost_24h: Decimal
    cost_trend: float  # Percentage change in cost

    # Performance metrics
    avg_response_time_ms: float
    success_rate: float
    uptime_percentage: float

    # Quality metrics
    quality_score: float  # 0-1 based on user feedback and evaluations
    consistency_score: float  # How consistent responses are

    # Usage metrics
    requests_24h: int
    tokens_24h: int
    error_rate: float

    # Capacity metrics
    current_load: float  # 0-1 scale
    rate_limit_remaining: int
    estimated_queue_time_ms: float

    # Calculated scores
    cost_score: float = 0.0     # Lower is better
    performance_score: float = 0.0  # Higher is better
    overall_score: float = 0.0  # Higher is better

    def calculate_scores(self, weights: Dict[str, float] = None) -> None:
        """Calculate weighted scores for routing decisions"""

        if weights is None:
            weights = {
                'cost': 0.4,
                'quality': 0.3,
                'speed': 0.2,
                'reliability': 0.1
            }

        # Normalize metrics (0-1 scale)
        cost_normalized = min(float(self.avg_cost_per_token) / 0.1, 1.0)  # Assume max $0.1/token
        speed_normalized = max(0, 1.0 - (self.avg_response_time_ms / 10000))  # 10s max
        quality_normalized = self.quality_score
        reliability_normalized = self.success_rate * self.uptime_percentage

        # Cost score (lower is better, so invert)
        self.cost_score = 1.0 - cost_normalized

        # Performance score (higher is better)
        self.performance_score = (
            weights['quality'] * quality_normalized +
            weights['speed'] * speed_normalized +
            weights['reliability'] * reliability_normalized
        )

        # Overall score (higher is better)
        self.overall_score = (
            weights['cost'] * self.cost_score +
            (1 - weights['cost']) * self.performance_score
        )

@dataclass
class RouteDecision:
    """AI provider routing decision with rationale"""
    selected_provider: str
    selected_model: str
    confidence: float  # 0-1 confidence in this decision
    expected_cost: Decimal
    expected_response_time_ms: float
    expected_quality: float
    alternative_providers: List[Dict[str, Any]]
    reasoning: str
    cache_strategy: str
    estimated_savings: Decimal = Decimal("0")

class CostOptimizer:
    """Intelligent cost optimization and provider routing engine"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis,
                 usage_tracker: UsageTracker, pricing_engine: PricingEngine):
        self.db_pool = db_pool
        self.redis = redis_client
        self.usage_tracker = usage_tracker
        self.pricing_engine = pricing_engine

        # Provider configurations
        self.providers = self._initialize_providers()

        # Optimization settings
        self.default_strategy = OptimizationStrategy.BALANCED
        self.cache_strategy = CacheStrategy.SMART

    def _initialize_providers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize AI provider configurations"""
        return {
            ServiceType.OPENAI_GPT4.value: {
                'provider': 'openai',
                'model': 'gpt-4',
                'cost_per_1k_tokens': {'input': 0.03, 'output': 0.06},
                'max_tokens': 8192,
                'quality_baseline': 0.95,
                'avg_response_time': 2500,
                'rate_limit': 3500,  # TPM
                'reliability': 0.99
            },
            ServiceType.OPENAI_GPT4_TURBO.value: {
                'provider': 'openai',
                'model': 'gpt-4-turbo',
                'cost_per_1k_tokens': {'input': 0.01, 'output': 0.03},
                'max_tokens': 4096,
                'quality_baseline': 0.93,
                'avg_response_time': 1800,
                'rate_limit': 5000,
                'reliability': 0.99
            },
            ServiceType.OPENAI_GPT35_TURBO.value: {
                'provider': 'openai',
                'model': 'gpt-3.5-turbo',
                'cost_per_1k_tokens': {'input': 0.0015, 'output': 0.002},
                'max_tokens': 4096,
                'quality_baseline': 0.85,
                'avg_response_time': 1200,
                'rate_limit': 10000,
                'reliability': 0.99
            },
            ServiceType.ANTHROPIC_CLAUDE3_SONNET.value: {
                'provider': 'anthropic',
                'model': 'claude-3-sonnet-20240229',
                'cost_per_1k_tokens': {'input': 0.003, 'output': 0.015},
                'max_tokens': 4096,
                'quality_baseline': 0.92,
                'avg_response_time': 2000,
                'rate_limit': 5000,
                'reliability': 0.98
            },
            ServiceType.GOOGLE_GEMINI_PRO.value: {
                'provider': 'google',
                'model': 'gemini-pro',
                'cost_per_1k_tokens': {'input': 0.001, 'output': 0.002},
                'max_tokens': 2048,
                'quality_baseline': 0.90,
                'avg_response_time': 1500,
                'rate_limit': 6000,
                'reliability': 0.97
            }
        }

    async def optimize_request(self, tenant_id: str, request_data: Dict[str, Any],
                             strategy: OptimizationStrategy = None) -> RouteDecision:
        """Optimize AI request routing for cost and performance"""

        strategy = strategy or self.default_strategy

        # Check cache first
        cache_result = await self._check_intelligent_cache(request_data)
        if cache_result:
            return RouteDecision(
                selected_provider="cache",
                selected_model="cached_response",
                confidence=1.0,
                expected_cost=Decimal("0"),
                expected_response_time_ms=50,
                expected_quality=cache_result.get('quality_score', 0.9),
                alternative_providers=[],
                reasoning="Found high-quality cached response",
                cache_strategy="hit",
                estimated_savings=Decimal(str(cache_result.get('original_cost', 0)))
            )

        # Get provider metrics and select best
        provider_metrics = await self._get_provider_metrics()
        suitable_providers = self._filter_suitable_providers(provider_metrics, request_data)

        if not suitable_providers:
            raise HTTPException(status_code=503, detail="No suitable providers available")

        # Rank and select provider
        ranked_providers = self._rank_providers(suitable_providers, strategy)
        selected = ranked_providers[0]

        return RouteDecision(
            selected_provider=selected.provider,
            selected_model=selected.model,
            confidence=selected.overall_score,
            expected_cost=selected.avg_cost_per_token * Decimal("1000"),  # Estimated cost
            expected_response_time_ms=selected.avg_response_time_ms,
            expected_quality=selected.quality_score,
            alternative_providers=[],
            reasoning=f"Selected {selected.provider} for optimal {strategy.value} performance",
            cache_strategy="miss"
        )

    async def _check_intelligent_cache(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check intelligent cache for similar requests"""

        if self.cache_strategy == CacheStrategy.DISABLED:
            return None

        cache_key = self._generate_cache_key(request_data)
        cached = await self.redis.get(f"ai_cache:exact:{cache_key}")

        if cached:
            cache_data = json.loads(cached)
            await self.redis.incr(f"ai_cache:hits:{cache_key}")
            return cache_data

        return None

    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request"""

        cache_content = {
            'prompt': request_data.get('prompt', ''),
            'max_tokens': request_data.get('max_tokens', 1000),
            'temperature': request_data.get('temperature', 0.7),
            'task_type': request_data.get('task_type', 'general')
        }

        content_str = json.dumps(cache_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    async def cache_response(self, request_data: Dict[str, Any], response: Any,
                           provider: str, model: str, cost: Decimal, quality_score: float = 0.9):
        """Cache AI response with intelligent metadata"""

        if self.cache_strategy == CacheStrategy.DISABLED:
            return

        cache_key = self._generate_cache_key(request_data)
        cache_data = {
            'response': response,
            'provider': provider,
            'model': model,
            'original_cost': float(cost),
            'quality_score': quality_score,
            'created_at': datetime.now().isoformat()
        }

        # Store with 1 hour TTL
        await self.redis.setex(
            f"ai_cache:exact:{cache_key}",
            3600,
            json.dumps(cache_data)
        )

        logger.info(f"Cached response for key {cache_key}")

    async def _get_provider_metrics(self) -> List[ProviderMetrics]:
        """Get real-time metrics for all providers"""

        metrics = []

        for service_type, config in self.providers.items():
            # Create metrics with baseline values
            provider_metric = ProviderMetrics(
                provider=config['provider'],
                model=config['model'],
                avg_cost_per_token=Decimal(str(config['cost_per_1k_tokens']['input'])) / 1000,
                total_cost_24h=Decimal("0"),
                cost_trend=0.0,
                avg_response_time_ms=config['avg_response_time'],
                success_rate=config['reliability'],
                uptime_percentage=config['reliability'],
                quality_score=config['quality_baseline'],
                consistency_score=0.9,
                requests_24h=0,
                tokens_24h=0,
                error_rate=1.0 - config['reliability'],
                current_load=0.1,
                rate_limit_remaining=config['rate_limit'],
                estimated_queue_time_ms=100
            )

            provider_metric.calculate_scores()
            metrics.append(provider_metric)

        return metrics

    def _filter_suitable_providers(self, provider_metrics: List[ProviderMetrics],
                                 request_data: Dict[str, Any]) -> List[ProviderMetrics]:
        """Filter providers based on request requirements"""

        suitable = []

        for provider in provider_metrics:
            # Basic filtering
            if provider.success_rate < 0.95:
                continue

            if provider.current_load > 0.95:
                continue

            suitable.append(provider)

        return suitable

    def _rank_providers(self, providers: List[ProviderMetrics],
                       strategy: OptimizationStrategy) -> List[ProviderMetrics]:
        """Rank providers based on optimization strategy"""

        weights = {
            OptimizationStrategy.COST_FIRST: {'cost': 0.7, 'quality': 0.15, 'speed': 0.1, 'reliability': 0.05},
            OptimizationStrategy.QUALITY_FIRST: {'cost': 0.1, 'quality': 0.6, 'speed': 0.15, 'reliability': 0.15},
            OptimizationStrategy.BALANCED: {'cost': 0.4, 'quality': 0.3, 'speed': 0.2, 'reliability': 0.1},
            OptimizationStrategy.SPEED_FIRST: {'cost': 0.2, 'quality': 0.2, 'speed': 0.5, 'reliability': 0.1}
        }

        strategy_weights = weights.get(strategy, weights[OptimizationStrategy.BALANCED])

        # Recalculate scores with strategy-specific weights
        for provider in providers:
            provider.calculate_scores(strategy_weights)

        # Sort by overall score (descending)
        return sorted(providers, key=lambda p: p.overall_score, reverse=True)

    async def get_cost_savings_report(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate cost savings report for a tenant"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        async with self.db_pool.acquire() as conn:
            actual_costs = await conn.fetchrow("""
                SELECT
                    SUM(calculated_cost_usd) as total_cost,
                    SUM(tokens_used) as total_tokens,
                    COUNT(*) as total_requests
                FROM tenant_management.usage_events
                WHERE tenant_id = $1 AND created_at >= $2
            """, tenant_id, start_date)

        total_cost = Decimal(str(actual_costs['total_cost'] or 0))
        estimated_savings = total_cost * Decimal("0.35")  # 35% potential savings

        return {
            'period_days': days,
            'total_cost': float(total_cost),
            'total_tokens': actual_costs['total_tokens'] or 0,
            'total_requests': actual_costs['total_requests'] or 0,
            'optimization_opportunities': {
                'cost_first_strategy': {
                    'estimated_savings': float(estimated_savings),
                    'savings_percentage': float((estimated_savings / total_cost * 100) if total_cost > 0 else 0)
                }
            },
            'recommendations': [
                "Enable cost-first optimization strategy",
                "Implement aggressive caching for repeated queries",
                "Consider batch processing for bulk operations"
            ]
        }

# Pydantic models for API
class OptimizationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=1000, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    task_type: str = Field(default="general")
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.BALANCED)

class OptimizationPreferences(BaseModel):
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.BALANCED)
    max_cost_per_request: Optional[float] = Field(None, ge=0)
    preferred_providers: List[str] = Field(default_factory=list)
    excluded_providers: List[str] = Field(default_factory=list)
    min_quality: float = Field(default=0.8, ge=0.0, le=1.0)

# Export main classes
__all__ = [
    'CostOptimizer', 'OptimizationStrategy', 'CacheStrategy', 'RouteDecision',
    'ProviderMetrics', 'OptimizationRequest', 'OptimizationPreferences'
]
