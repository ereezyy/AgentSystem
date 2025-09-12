
"""
🎯 AgentSystem Cost Optimization Engine
Intelligent AI provider routing and cost arbitrage system for maximum profit margins
"""

import asyncio
import json
import time
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import asyncpg
import aioredis
from fastapi import HTTPException
import openai
import anthropic
import google.generativeai as genai

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

@dataclass
class CacheEntry:
    """Intelligent cache entry with metadata"""
    key: str
    response: Any
    provider: str
    model: str
    cost: Decimal
    quality_score: float
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = None
    ttl_seconds: int = 3600  # Default 1 hour
    similarity_threshold: float = 0.95  # For semantic caching

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

        # Performance tracking
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_models = {}

        # Cache management
        self.cache_hit_rate = 0.0
        self.cache_savings_24h = Decimal("0")

        # Real-time monitoring
        self.provider_status = {}
        self.load_balancer_weights = {}

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
            ServiceType.ANTHROPIC_CLAUDE3_OPUS.value: {
                'provider': 'anthropic',
                'model': 'claude-3-opus-20240229',
                'cost_per_1k_tokens': {'input': 0.015, 'output': 0.075},
                'max_tokens': 4096,
                'quality_baseline': 0.96,
                'avg_response_time': 3000,
                'rate_limit': 4000,
                'reliability': 0.98
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
            ServiceType.ANTHROPIC_CLAUDE3_HAIKU.value: {
                'provider': 'anthropic',
                'model': 'claude-3-haiku-20240307',
                'cost_per_1k_tokens': {'input': 0.00025, 'output': 0.00125},
                'max_tokens': 4096,
                'quality_baseline': 0.88,
                'avg_response_time': 1000,
                'rate_limit': 8000,
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

        # Analyze request characteristics
        request_analysis = await self._analyze_request(request_data)

        # Get tenant preferences and constraints
        tenant_constraints = await self._get_tenant_constraints(tenant_id)

        # Check cache first
        cache_result = await self._check_intelligent_cache(request_data, tenant_constraints)
        if cache_result:
            return RouteDecision(
                selected_provider="cache",
                selected_model="cached_response",
                confidence=1.0,
                expected_cost=Decimal("0"),
                expected_response_time_ms=50,
                expected_quality=cache_result['quality_score'],
                alternative_providers=[],
                reasoning="Found high-quality cached response",
                cache_strategy="hit",
                estimated_savings=cache_result['original_cost']
            )

        # Get current provider metrics
        provider_metrics = await self._get_provider_metrics()

        # Filter providers based on request requirements
        suitable_providers = self._filter_suitable_providers(
            provider_metrics, request_analysis, tenant_constraints
        )

        if not suitable_providers:
            raise HTTPException(status_code=503, detail="No suitable providers available")

        # Score and rank providers
        ranked_providers = await self._rank_providers(
            suitable_providers, strategy, request_analysis, tenant_id
        )

        # Select best provider with load balancing
        selected = await self._select_provider_with_load_balancing(ranked_providers)

        # Calculate expected savings vs most expensive option
        most_expensive = max(ranked_providers, key=lambda p: p.avg_cost_per_token)
        estimated_savings = (
            most_expensive.avg_cost_per_token - selected.avg_cost_per_token
        ) * Decimal(str(request_analysis['estimated_tokens']))

        return RouteDecision(
            selected_provider=selected.provider,
            selected_model=selected.model,
            confidence=selected.overall_score,
            expected_cost=selected.avg_cost_per_token * Decimal(str(request_analysis['estimated_tokens'])),
            expected_response_time_ms=selected.avg_response_time_ms,
            expected_quality=selected.quality_score,
            alternative_providers=[
                {
                    'provider': p.provider,
                    'model': p.model,
                    'cost': float(p.avg_cost_per_token),
                    'quality': p.quality_score,
                    'response_time': p.avg_response_time_ms
                } for p in ranked_providers[1:4]  # Top 3 alternatives
            ],
            reasoning=self._generate_reasoning(selected, strategy, request_analysis),
            cache_strategy="miss",
            estimated_savings=estimated_savings
        )

    async def _analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze request to determine optimal routing"""

        # Extract request characteristics
        prompt = request_data.get('prompt', '')
        max_tokens = request_data.get('max_tokens', 1000)
        temperature = request_data.get('temperature', 0.7)
        task_type = request_data.get('task_type', 'general')

        # Estimate token usage
        estimated_input_tokens = len(prompt.split()) * 1.3  # Rough approximation
        estimated_total_tokens = estimated_input_tokens + max_tokens

        # Analyze task complexity
        complexity_indicators = {
            'code_generation': ['function', 'class', 'def ', 'return', 'import'],
            'creative_writing': ['story', 'poem', 'creative', 'imagine', 'character'],
            'analysis': ['analyze', 'compare', 'evaluate', 'assess', 'review'],
            'translation': ['translate', 'language', 'français', 'español', 'deutsch'],
            'math': ['calculate', 'equation', 'formula', 'solve', 'mathematics']
        }

        detected_tasks = []
        prompt_lower = prompt.lower()
        for task, indicators in complexity_indicators.items():
            if any(indicator in prompt_lower for indicator in indicators):
                detected_tasks.append(task)

        # Determine quality requirements
        quality_sensitive_tasks = ['code_generation', 'analysis', 'math', 'translation']
        requires_high_quality = any(task in quality_sensitive_tasks for task in detected_tasks)

        # Determine speed requirements
        speed_sensitive_indicators = ['urgent', 'quickly', 'asap', 'real-time', 'immediate']
        requires_speed = any(indicator in prompt_lower for indicator in speed_sensitive_indicators)

        return {
            'estimated_tokens': estimated_total_tokens,
            'estimated_input_tokens': estimated_input_tokens,
            'max_output_tokens': max_tokens,
            'task_types': detected_tasks,
            'requires_high_quality': requires_high_quality,
            'requires_speed': requires_speed,
            'complexity_score': len(detected_tasks) * 0.2 + (len(prompt) / 1000) * 0.3,
            'temperature': temperature,
            'cacheable': temperature < 0.3 and 'creative' not in detected_tasks
        }

    async def _get_tenant_constraints(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant-specific routing constraints and preferences"""

        # Get tenant plan and preferences
        config = await self.pricing_engine.get_plan_configuration(tenant_id)

        # Get optimization preferences from tenant settings
        async with self.db_pool.acquire() as conn:
            settings_row = await conn.fetchrow("""
                SELECT settings FROM tenant_management.tenants WHERE id = $1
            """, tenant_id)

            settings = json.loads(settings_row['settings'] if settings_row else '{}')
            optimization_prefs = settings.get('optimization_preferences', {})

        return {
            'plan_type': config.plan_type,
            'max_cost_per_request': optimization_prefs.get('max_cost_per_request'),
            'preferred_providers': optimization_prefs.get('preferred_providers', []),
            'excluded_providers': optimization_prefs.get('excluded_providers', []),
            'optimization_strategy': optimization_prefs.get('strategy', self.default_strategy),
            'quality_threshold': optimization_prefs.get('min_quality', 0.8),
            'max_response_time': optimization_prefs.get('max_response_time_ms', 10000),
            'cost_weight': optimization_prefs.get('cost_weight', 0.4),
            'quality_weight': optimization_prefs.get('quality_weight', 0.3),
            'speed_weight': optimization_prefs.get('speed_weight', 0.2),
            'reliability_weight': optimization_prefs.get('reliability_weight', 0.1)
        }

    async def _check_intelligent_cache(self, request_data: Dict[str, Any],
                                     constraints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check intelligent cache for similar requests"""

        if self.cache_strategy == CacheStrategy.DISABLED:
            return None

        # Generate cache key based on request content
        cache_key = self._generate_cache_key(request_data)

        # Try exact match first
        cached = await self.redis.get(f"ai_cache:exact:{cache_key}")
        if cached:
            cache_data = json.loads(cached)
            await self.redis.incr(f"ai_cache:hits:{cache_key}")
            cache_data['hit_type'] = 'exact'
            return cache_data

        # Try semantic similarity cache for aggressive/smart strategies
        if self.cache_strategy in [CacheStrategy.AGGRESSIVE, CacheStrategy.SMART]:
            similar_entries = await self._find_similar_cached_requests(request_data)

            for entry in similar_entries:
                if entry['similarity'] >= 0.95:  # Very high similarity
                    await self.redis.incr(f"ai_cache:semantic_hits")
                    entry['hit_type'] = 'semantic'
                    return entry

        return None

    async def _get_provider_metrics(self) -> List[ProviderMetrics]:
        """Get real-time metrics for all providers"""

        metrics = []

        # Get performance data from the last 24 hours
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        async with self.db_pool.acquire() as conn:
            for service_type, config in self.providers.items():
                # Get usage statistics
                stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as request_count,
                        AVG(processing_time_ms) as avg_response_time,
                        AVG(provider_cost_usd) as avg_cost,
                        SUM(tokens_used) as total_tokens,
                        COUNT(*) FILTER (WHERE tokens_used > 0) as successful_requests
                    FROM tenant_management.usage_events
                    WHERE service_type = $1 AND created_at >= $2
                """, service_type, start_time)

                if not stats or stats['request_count'] == 0:
                    # Use baseline values for new/unused providers
                    metrics.append(ProviderMetrics(
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
                        current_load=0.1,  # Low initial load
                        rate_limit_remaining=config['rate_limit'],
                        estimated_queue_time_ms=100
                    ))
                    continue

                # Calculate real metrics
                success_rate = stats['successful_requests'] / stats['request_count'] if stats['request_count'] > 0 else 0
                avg_cost_per_token = Decimal(str(stats['avg_cost'] or 0)) / max(stats['total_tokens'] or 1, 1) * 1000

                # Get quality score from recent feedback (if available)
                quality_score = await self._get_provider_quality_score(service_type)

                # Get current load and rate limits (would integrate with provider APIs)
                current_load, rate_limit_remaining = await self._get_provider_load_info(service_type)

                provider_metric = ProviderMetrics(
                    provider=config['provider'],
                    model=config['model'],
                    avg_cost_per_token=avg_cost_per_token,
                    total_cost_24h=Decimal(str(stats['avg_cost'] or 0)) * stats['request_count'],
                    cost_trend=0.0,  # Would calculate from historical data
                    avg_response_time_ms=float(stats['avg_response_time'] or config['avg_response_time']),
                    success_rate=success_rate,
                    uptime_percentage=success_rate,  # Simplified
                    quality_score=quality_score,
                    consistency_score=0.9,  # Would calculate from response variance
                    requests_24h=stats['request_count'],
                    tokens_24h=stats['total_tokens'] or 0,
                    error_rate=1.0 - success_rate,
                    current_load=current_load,
                    rate_limit_remaining=rate_limit_remaining,
                    estimated_queue_time_ms=max(100, current_load * 1000)
                )

                # Calculate scores
                provider_metric.calculate_scores()
                metrics.append(provider_metric)

        return metrics

    async def _get_provider_quality_score(self, service_type: str) -> float:
        """Get quality score for provider based on feedback and evaluations"""

        # Get from Redis cache or calculate
        cached_score = await self.redis.get(f"quality_score:{service_type}")
        if cached_score:
            return float(cached_score)

        # Default to baseline quality from configuration
        return self.providers[service_type]['quality_baseline']

    async def _get_provider_load_info(self, service_type: str) -> Tuple[float, int]:
        """Get current load and rate limit info for provider"""

        # In a real implementation, this would call provider APIs
        # For now, simulate based on recent usage

        # Get recent usage rate
        recent_requests = await self.redis.get(f"recent_usage:{service_type}") or 0
        max_rate = self.providers[service_type]['rate_limit']

        current_load = min(float(recent_requests) / max_rate, 1.0)
        rate_limit_remaining = max(0, max_rate - int(recent_requests))

        return current_load, rate_limit_remaining

    def _filter_suitable_providers(self, provider_metrics: List[ProviderMetrics],
                                 request_analysis: Dict[str, Any],
                                 constraints: Dict[str, Any]) -> List[ProviderMetrics]:
        """Filter providers based on request requirements and constraints"""

        suitable = []

        for provider in provider_metrics:
            # Check if provider is excluded
            if provider.provider in constraints.get('excluded_providers', []):
                continue

            # Check quality threshold
            if provider.quality_score < constraints.get('quality_threshold', 0.8):
                continue

            # Check response time requirement
            if provider.avg_response_time_ms > constraints.get('max_response_time', 10000):
                continue

            # Check if provider can handle the token count
            config = next(c for c in self.providers.values() if c['model'] == provider.model)
            if request_analysis['max_output_tokens'] > config['max_tokens']:
                continue

            # Check success rate
            if provider.success_rate < 0.95:  # Minimum reliability
                continue

            # Check current capacity
            if provider.current_load > 0.95:  # Too loaded
                continue

            suitable.append(provider)

        return suitable

    async def _rank_providers(self, providers: List[ProviderMetrics],
                            strategy: OptimizationStrategy,
                            request_analysis: Dict[str, Any],
                            tenant_id: str) -> List[ProviderMetrics]:
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

        # Apply request-specific adjustments
        if request_analysis['requires_high_quality']:
            for provider in providers:
                provider.overall_score *= (1.0 + provider.quality_score * 0.2)

        if request_analysis['requires_speed']:
            for provider in providers:
                speed_bonus = max(0, (2000 - provider.avg_response_time_ms) / 2000) * 0.3
                provider.overall_score *= (1.0 + speed_bonus)

        # Sort by overall score (descending)
        return sorted(providers, key=lambda p: p.overall_score, reverse=True)

    async def _select_provider_with_load_balancing(self, ranked_providers: List[ProviderMetrics]) -> ProviderMetrics:
        """Select provider with intelligent load balancing"""

        # Use weighted random selection from top 3 providers
        top_providers = ranked_providers[:3]

        if not top_providers:
            raise HTTPException(status_code=503, detail="No providers available")

        if len(top_providers) == 1:
            return top_providers[0]

        # Calculate selection weights (higher score + lower load = higher weight)
        weights = []
        for provider in top_providers:
            load_penalty = provider.current_load * 0.5
            weight = provider.overall_score * (1.0 - load_penalty)
            weights.append(max(weight, 0.1))  # Minimum weight

        # Weighted random selection
        import random
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return top_providers[i]

        return top_providers[0]  # Fallback

    def _generate_reasoning(self, selected: ProviderMetrics, strategy: OptimizationStrategy,
                          request_analysis: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for provider selection"""

        reasons = []

        if strategy == OptimizationStrategy.COST_FIRST:
            reasons.append(f"Selected for lowest cost (${selected.avg_cost_per_token:.6f}/token)")
        elif strategy == OptimizationStrategy.QUALITY_FIRST:
            reasons.append(f"Selected for highest quality (score: {selected.quality_score:.2f})")
        elif strategy == OptimizationStrategy.SPEED_FIRST:
            reasons.append(f"Selected for fastest response ({selected.avg_response_time_ms:.0f}ms avg)")
        else:
            reasons.append(f"Selected for optimal balance (score: {selected.overall_score:.2f})")

        if selected.current_load < 0.3:
            reasons.append("Low current load ensures fast processing")

        if request_analysis['requires_high_quality'] and selected.quality_score > 0.9:
            reasons.append("High quality score suitable for complex task")

        if request_analysis['cacheable']:
            reasons.append("Response will be cached for future optimization")

        return "; ".join(reasons)

    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request"""

        # Create deterministic hash of request content
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

        # Determine TTL based on content type and quality
        base_ttl = 3600  # 1 hour
        if quality_score > 0.95:
            ttl = base_ttl * 4  # High quality responses cached longer
        elif quality_score < 0.8:
            ttl = base_ttl // 2  # Low quality responses cached shorter
        else:
            ttl = base_ttl

        # Adjust TTL based on request characteristics
        if request_data.get('temperature', 0.7) < 0.3:
            ttl *= 2  # Deterministic responses cached longer

        cache_data = {
            'response': response,
            'provider': provider,
            'model': model,
            'original_cost': float(cost),
            'quality_score': quality_score,
            'created_at': datetime.now().isoformat(),
            'access_count': 1,
            'request_hash': cache_key
        }

        # Store exact match cache
        await self.redis.setex(
            f"ai_cache:exact:{cache_key}",
            ttl,
            json.dumps(cache_data)
        )

        # Store for semantic similarity search if enabled
        if self.cache_strategy in [CacheStrategy.AGGRESSIVE, CacheStrategy.SMART]:
            await self.redis.setex(
                f"ai_cache:semantic:{cache_key}",
                ttl,
                json.dumps({
                    'prompt': request_data.get('prompt', ''),
                    'response': response,
                    'metadata': cache_data
                })
            )

        logger.info(f"Cached response for key {cache_key} with TTL {ttl}s")

    async def _find_similar_cached_requests(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find semantically similar cached requests"""

        # Simple implementation - in production, use vector embeddings
        prompt = request_data.get('prompt', '').lower()
        similar_entries = []

        # Get recent cache entries
        pattern = "ai_cache:semantic:*"
        cache_keys = await self.redis.keys(pattern)

        for key in cache_keys[:50]:  # Limit search to recent entries
            try:
                cache_data = json.loads(await self.redis.get(key))
                cached_prompt = cache_data['prompt'].lower()

                # Simple similarity calculation (Jaccard similarity)
                prompt_words = set(prompt.split())
                cached_words = set(cached_prompt.split())

                if not prompt_words or not cached_words:
                    continue

                intersection = len(prompt_words & cached_words)
                union = len(prompt_words | cached_words)
                similarity = intersection / union if union > 0 else 0

                if similarity >= 0.8:  # Threshold for similarity
                    cache_data['similarity'] = similarity
                    similar_entries.append(cache_data)

            except Exception as e:
                logger.warning(f"Error processing cache entry {key}: {e}")
                continue

        return sorted(similar_entries, key=lambda x: x['similarity'], reverse=True)

    async def get_cost_savings_report(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate cost savings report for a tenant"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        async with self.db_pool.acquire() as conn:
            # Get total actual costs
            actual_costs = await conn.fetchrow("""
                SELECT
                    SUM(calculated_cost_usd) as total_cost,
                    SUM(tokens_used) as total_tokens,
                    COUNT(*) as total_requests
                FROM tenant_management.usage_events
                WHERE tenant_id = $1 AND created_at >= $2
            """, tenant_id, start_date)

            # Get provider breakdown
            provider_breakdown = await conn.fetch("""
                SELECT
                    service_type,
                    SUM(calculated_cost_usd) as cost,
                    SUM(tokens_used) as tokens,
                    COUNT(*) as requests,
                    AVG(processing_time_ms) as avg_time
                FROM tenant_management.usage_events
                WHERE tenant_id = $1 AND created_at >= $2
                GROUP BY service_type
                ORDER BY cost DESC
            """, tenant_id, start_date)

        # Calculate potential savings with different strategies
        total_cost = Decimal(str(actual_costs['total_cost'] or 0))
        total_tokens = actual_costs['total_tokens'] or 0

        # Estimate savings with cost-first strategy
        cost_first_savings = total_cost * Decimal("0.35")  # Estimated 35% savings

        # Estimate cache savings
        cache_hit_rate = await self.redis.get(f"cache_hit_rate:{tenant_id}") or 0
        cache_savings = total_cost * Decimal(str(cache_hit_rate)) * Decimal("0.05")  # 5% per hit

        # Get current optimization settings
        current_strategy = await self._get_tenant_optimization_strategy(tenant_id)

        return {
            'period_days': days,
            'total_cost': float(total_cost),
            'total_tokens': total_tokens,
            'total_requests': actual_costs['total_requests'] or 0,
            'current_strategy': current_strategy,
            'provider_breakdown': [
                {
                    'service': row['service_type'],
                    'cost': float(row['cost']),
                    'tokens': row['tokens'],
                    'requests': row['requests'],
                    'avg_response_time': float(row['avg_time'] or 0)
                }
                for row in provider_breakdown
            ],
            'optimization_opportunities': {
                'cost_first_strategy': {
                    'estimated_savings': float(cost_first_savings),
                    'savings_percentage': float((cost_first_savings / total_cost * 100) if total_cost > 0 else 0)
                },
                'improved_caching': {
                    'current_hit_rate': float(cache_hit_rate),
                    'potential_savings': float(cache_savings),
                    'target_hit_rate': 0.6
                },
                'provider_optimization': await self._get_provider_optimization_suggestions(tenant_id)
            },
            'recommendations': await self._generate_optimization_recommendations(tenant_id, total_cost, total_tokens)
        }

    async def _get_tenant_optimization_strategy(self, tenant_id: str) -> str:
        """Get current optimization strategy for tenant"""

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT settings FROM tenant_management.tenants WHERE id = $1
            """, tenant_id)

            if row:
                settings = json.loads(row['settings'] or '{}')
                return settings.get('optimization_preferences', {}).get('strategy', 'balanced')

            return 'balanced'

    async def _get_provider_optimization_suggestions(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get provider-specific optimization suggestions"""

        suggestions = []

        # Analyze current provider usage
        async with self.db_pool.acquire() as conn:
            provider_usage = await conn.fetch("""
                SELECT
                    service_type,
                    SUM(calculated_cost_usd) as total_cost,
                    SUM(tokens_used) as total_tokens,
                    AVG(processing_time_ms) as avg_time
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '7 days'
                GROUP BY service_type
                HAVING SUM(calculated_cost_usd) > 0
                ORDER BY total_cost DESC
            """, tenant_id)

        for usage in provider_usage:
            service_type = usage['service_type']
            current_cost = Decimal(str(usage['total_cost']))

            # Find cheaper alternatives
            current_config = self.providers.get(service_type)
            if not current_config:
                continue

            # Look for cheaper providers with similar quality
            alternatives = []
            for alt_service, alt_config in self.providers.items():
                if alt_service == service_type:
                    continue

                cost_diff = (Decimal(str(alt_config['cost_per_1k_tokens']['input'])) -
                           Decimal(str(current_config['cost_per_1k_tokens']['input'])))
                quality_diff = alt_config['quality_baseline'] - current_config['quality_baseline']

                if cost_diff < 0 and quality_diff > -0.1:  # Cheaper with similar quality
                    potential_savings = current_cost * abs(cost_diff) / Decimal(str(current_config['cost_per_1k_tokens']['input']))

                    alternatives.append({
                        'provider': alt_config['provider'],
                        'model': alt_config['model'],
                        'cost_savings': float(potential_savings),
                        'quality_impact': quality_diff
                    })

            if alternatives:
                suggestions.append({
                    'current_service': service_type,
                    'current_cost': float(current_cost),
                    'alternatives': sorted(alternatives, key=lambda x: x['cost_savings'], reverse=True)[:3]
                })

        return suggestions

    async def _generate_optimization_recommendations(self, tenant_id: str,
                                                   total_cost: Decimal, total_tokens: int) -> List[str]:
        """Generate actionable optimization recommendations"""

        recommendations = []

        # Cost-based recommendations
        if total_cost > Decimal("100"):  # $100+ monthly spend
            recommendations.append("Consider switching to cost-first optimization strategy to reduce spending by up to 35%")

        # Usage-based recommendations
        if total_tokens > 1_000_000:  # High volume
            recommendations.append("Enable aggressive caching to reduce costs on repeated queries")
            recommendations.append("Consider batch processing for bulk operations")

        # Cache recommendations
        cache_hit_rate = float(await self.redis.get(f"cache_hit_rate:{tenant_id}") or 0)
        if cache_hit_rate < 0.3:
            recommendations.append("Improve cache hit rate by using more deterministic prompts (temperature < 0.3)")

        # Provider diversity recommendations
        provider_count = len(await self._get_active_providers(tenant_id))
        if provider_count < 3:
            recommendations.append("Diversify AI provider usage to take advantage of competitive pricing")

        return recommendations

    async def _get_active_providers(self, tenant_id: str) -> List[str]:
        """Get list of actively used providers"""

        async with self.db_pool.acquire() as conn:
            providers = await conn.fetch("""
                SELECT DISTINCT service_type
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                    AND tokens_used > 0
            """, tenant_id)

            return [row['service_type'] for row in providers]

    async def update_tenant_optimization_preferences(self, tenant_id: str,
                                                   preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update tenant's optimization preferences"""

        # Validate preferences
        valid_strategies = [s.value for s in OptimizationStrategy]
        if preferences.get('strategy') not in valid_strategies:
            raise HTTPException(status_code=400, detail=f"Invalid strategy. Must be one of: {valid_strategies}")

        async with self.db_pool.acquire() as conn:
            # Get current settings
            current_settings = await conn.fetchval("""
                SELECT settings FROM tenant_management.tenants WHERE id = $1
            """, tenant_id)

            settings = json.loads(current_settings or '{}')
            settings['optimization_preferences'] = preferences

            # Update database
            await conn.execute("""
                UPDATE tenant_management.tenants
                SET settings = $1, updated_at = NOW()
                WHERE id = $2
            """, json.dumps(settings), tenant_id)

        # Clear caches
        await self.redis.delete(f"optimization_prefs:{tenant_id}")

        logger.info(f"Updated optimization preferences for tenant {tenant_id}")

        return preferences

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
    max_response_time_ms: int = Field(default=10000, ge=100)
    cost_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    quality_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    speed_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    reliability_weight: float = Field(default=0.1, ge=0.0, le=1.0)

# Export main classes
__all__ = [
    'CostOptimizer', 'OptimizationStrategy', 'CacheStrategy', 'RouteDecision',
    'ProviderMetrics', 'OptimizationRequest', 'OptimizationPreferences'
]
                           provider: str, model: str, cost: Decimal, quality_score: float = 0.9):
        """Cache AI response with intelligent metadata"""

        if self.cache_strategy == CacheStrategy.DISABLED:
            return

        cache_key = self._generate_cache_key(request_data)

        # Determine TTL based on content type and quality
        base_ttl = 3600  # 1 hour
        if quality_score > 0.95:
            ttl = base_ttl * 4  # High quality responses cached longer
        elif quality_score < 0.8:
            ttl = base_ttl // 2  # Low quality responses cached shorter
        else:
            ttl = base_ttl

        # Adjust TTL based on request characteristics
        if request_data.get('temperature', 0.7) < 0.3:
            ttl *= 2  # Deterministic responses cached longer

        cache
