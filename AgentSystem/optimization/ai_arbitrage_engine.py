
"""
Intelligent AI Provider Arbitrage Engine - AgentSystem Profit Machine
Advanced cost optimization through intelligent provider routing and real-time pricing arbitrage
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import asyncpg
import aioredis
import aiohttp
import numpy as np
from collections import defaultdict
import uuid
import hashlib

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    COHERE = "cohere"
    MISTRAL = "mistral"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    GROQ = "groq"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"

class ModelCapability(Enum):
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"
    LONG_CONTEXT = "long_context"
    MULTIMODAL = "multimodal"

class RoutingStrategy(Enum):
    COST_OPTIMAL = "cost_optimal"
    QUALITY_OPTIMAL = "quality_optimal"
    LATENCY_OPTIMAL = "latency_optimal"
    BALANCED = "balanced"
    FALLBACK_CASCADE = "fallback_cascade"

@dataclass
class AIModel:
    model_id: str
    provider: AIProvider
    model_name: str
    capabilities: List[ModelCapability]
    cost_per_input_token: float
    cost_per_output_token: float
    max_context_length: int
    max_output_tokens: int
    quality_score: float  # 0-100
    average_latency_ms: float
    availability_score: float  # 0-100, rolling 24h average
    rate_limit_rpm: int
    rate_limit_tpm: int
    supports_streaming: bool
    supports_function_calling: bool
    supports_vision: bool
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ArbitrageRequest:
    request_id: str
    tenant_id: str
    capability: ModelCapability
    input_tokens: int
    estimated_output_tokens: int
    max_latency_ms: Optional[int]
    quality_threshold: Optional[float]
    cost_budget: Optional[float]
    strategy: RoutingStrategy
    fallback_enabled: bool
    context_length_required: int
    requires_streaming: bool
    requires_function_calling: bool
    requires_vision: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ArbitrageResult:
    request_id: str
    selected_model: AIModel
    estimated_cost: float
    estimated_latency_ms: float
    cost_savings_percent: float
    fallback_models: List[AIModel]
    routing_reason: str
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ProviderMetrics:
    provider: AIProvider
    success_rate: float
    average_latency_ms: float
    average_cost_per_request: float
    total_requests_24h: int
    total_cost_24h: float
    quality_score: float
    last_updated: datetime

@dataclass
class CostOptimizationRule:
    rule_id: str
    tenant_id: str
    name: str
    description: str
    conditions: Dict[str, Any]  # JSON conditions for rule matching
    actions: Dict[str, Any]     # Actions to take when rule matches
    priority: int
    is_active: bool
    cost_savings_target: float  # Target cost savings percentage
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class AIArbitrageEngine:
    """
    Intelligent AI provider arbitrage engine for cost optimization
    """

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis = redis_client

        # In-memory caches for performance
        self.models_cache = {}
        self.pricing_cache = {}
        self.metrics_cache = {}

        # Performance tracking
        self.request_history = defaultdict(list)
        self.cost_savings_history = defaultdict(list)

        # Initialize providers and models
        self.providers = self._initialize_providers()

        # Pricing refresh intervals
        self.pricing_refresh_interval = 300  # 5 minutes
        self.metrics_refresh_interval = 60   # 1 minute

        # Quality thresholds
        self.min_quality_score = 70.0
        self.min_availability_score = 95.0

        # Background tasks
        self._background_tasks = []

    async def initialize(self):
        """Initialize the arbitrage engine"""

        # Load models and pricing data
        await self._load_models_from_database()
        await self._load_pricing_data()
        await self._load_provider_metrics()

        # Start background monitoring tasks
        asyncio.create_task(self._monitor_pricing_loop())
        asyncio.create_task(self._monitor_provider_health_loop())
        asyncio.create_task(self._optimize_routing_algorithms_loop())

        logger.info("AI Arbitrage Engine initialized successfully")

    async def route_request(self, request: ArbitrageRequest) -> ArbitrageResult:
        """
        Main arbitrage routing function - finds optimal AI provider/model
        """

        start_time = time.time()

        # Get eligible models for this request
        eligible_models = await self._get_eligible_models(request)

        if not eligible_models:
            raise ValueError(f"No eligible models found for capability: {request.capability}")

        # Apply routing strategy
        selected_model = await self._apply_routing_strategy(request, eligible_models)

        # Calculate cost and savings
        estimated_cost = await self._calculate_request_cost(request, selected_model)
        baseline_cost = await self._calculate_baseline_cost(request, eligible_models)
        cost_savings_percent = ((baseline_cost - estimated_cost) / baseline_cost * 100) if baseline_cost > 0 else 0

        # Get fallback models
        fallback_models = await self._get_fallback_models(request, eligible_models, selected_model)

        # Calculate confidence score
        confidence_score = await self._calculate_confidence_score(request, selected_model)

        # Create result
        result = ArbitrageResult(
            request_id=request.request_id,
            selected_model=selected_model,
            estimated_cost=estimated_cost,
            estimated_latency_ms=selected_model.average_latency_ms,
            cost_savings_percent=cost_savings_percent,
            fallback_models=fallback_models,
            routing_reason=self._get_routing_reason(request.strategy, selected_model),
            confidence_score=confidence_score
        )

        # Store routing decision for learning
        await self._store_routing_decision(request, result)

        # Update metrics
        routing_time = (time.time() - start_time) * 1000
        await self._update_routing_metrics(request.tenant_id, routing_time, cost_savings_percent)

        logger.info(f"Routed request {request.request_id} to {selected_model.provider.value}:{selected_model.model_name} "
                   f"with {cost_savings_percent:.1f}% cost savings")

        return result

    async def _get_eligible_models(self, request: ArbitrageRequest) -> List[AIModel]:
        """Get models that can handle this request"""

        eligible = []

        for model in self.models_cache.values():
            # Check capability match
            if request.capability not in model.capabilities:
                continue

            # Check context length requirement
            if request.context_length_required > model.max_context_length:
                continue

            # Check streaming requirement
            if request.requires_streaming and not model.supports_streaming:
                continue

            # Check function calling requirement
            if request.requires_function_calling and not model.supports_function_calling:
                continue

            # Check vision requirement
            if request.requires_vision and not model.supports_vision:
                continue

            # Check quality threshold
            if request.quality_threshold and model.quality_score < request.quality_threshold:
                continue

            # Check availability
            if model.availability_score < self.min_availability_score:
                continue

            # Check rate limits (simplified check)
            current_usage = await self._get_current_provider_usage(model.provider)
            if current_usage['rpm'] >= model.rate_limit_rpm * 0.9:  # 90% threshold
                continue

            eligible.append(model)

        return eligible

    async def _apply_routing_strategy(self, request: ArbitrageRequest,
                                    eligible_models: List[AIModel]) -> AIModel:
        """Apply the specified routing strategy to select optimal model"""

        if request.strategy == RoutingStrategy.COST_OPTIMAL:
            return await self._route_cost_optimal(request, eligible_models)

        elif request.strategy == RoutingStrategy.QUALITY_OPTIMAL:
            return await self._route_quality_optimal(request, eligible_models)

        elif request.strategy == RoutingStrategy.LATENCY_OPTIMAL:
            return await self._route_latency_optimal(request, eligible_models)

        elif request.strategy == RoutingStrategy.BALANCED:
            return await self._route_balanced(request, eligible_models)

        elif request.strategy == RoutingStrategy.FALLBACK_CASCADE:
            return await self._route_fallback_cascade(request, eligible_models)

        else:
            # Default to balanced
            return await self._route_balanced(request, eligible_models)

    async def _route_cost_optimal(self, request: ArbitrageRequest,
                                eligible_models: List[AIModel]) -> AIModel:
        """Route to the most cost-effective model"""

        costs = []
        for model in eligible_models:
            cost = await self._calculate_request_cost(request, model)
            costs.append((cost, model))

        # Sort by cost and return cheapest
        costs.sort(key=lambda x: x[0])
        return costs[0][1]

    async def _route_quality_optimal(self, request: ArbitrageRequest,
                                   eligible_models: List[AIModel]) -> AIModel:
        """Route to the highest quality model within budget"""

        # Filter by budget if specified
        if request.cost_budget:
            budget_eligible = []
            for model in eligible_models:
                cost = await self._calculate_request_cost(request, model)
                if cost <= request.cost_budget:
                    budget_eligible.append(model)
            eligible_models = budget_eligible

        if not eligible_models:
            raise ValueError("No models available within specified budget")

        # Sort by quality score
        eligible_models.sort(key=lambda x: x.quality_score, reverse=True)
        return eligible_models[0]

    async def _route_latency_optimal(self, request: ArbitrageRequest,
                                   eligible_models: List[AIModel]) -> AIModel:
        """Route to the fastest model within constraints"""

        # Filter by latency requirement if specified
        if request.max_latency_ms:
            latency_eligible = [
                model for model in eligible_models
                if model.average_latency_ms <= request.max_latency_ms
            ]
            eligible_models = latency_eligible

        if not eligible_models:
            raise ValueError("No models available within latency requirement")

        # Sort by latency
        eligible_models.sort(key=lambda x: x.average_latency_ms)
        return eligible_models[0]

    async def _route_balanced(self, request: ArbitrageRequest,
                            eligible_models: List[AIModel]) -> AIModel:
        """Route using balanced scoring algorithm"""

        scores = []

        for model in eligible_models:
            # Calculate cost score (lower cost = higher score)
            cost = await self._calculate_request_cost(request, model)
            max_cost = max(await self._calculate_request_cost(request, m) for m in eligible_models)
            min_cost = min(await self._calculate_request_cost(request, m) for m in eligible_models)
            cost_score = ((max_cost - cost) / (max_cost - min_cost) * 100) if max_cost > min_cost else 100

            # Calculate latency score (lower latency = higher score)
            max_latency = max(m.average_latency_ms for m in eligible_models)
            min_latency = min(m.average_latency_ms for m in eligible_models)
            latency_score = ((max_latency - model.average_latency_ms) / (max_latency - min_latency) * 100) if max_latency > min_latency else 100

            # Quality score (use directly)
            quality_score = model.quality_score

            # Availability score (use directly)
            availability_score = model.availability_score

            # Weighted composite score
            weights = {
                'cost': 0.35,
                'quality': 0.30,
                'latency': 0.20,
                'availability': 0.15
            }

            composite_score = (
                weights['cost'] * cost_score +
                weights['quality'] * quality_score +
                weights['latency'] * latency_score +
                weights['availability'] * availability_score
            )

            scores.append((composite_score, model))

        # Sort by composite score and return best
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    async def _route_fallback_cascade(self, request: ArbitrageRequest,
                                    eligible_models: List[AIModel]) -> AIModel:
        """Route with intelligent fallback cascade"""

        # Primary: Try cost optimal
        try:
            primary = await self._route_cost_optimal(request, eligible_models)
            if await self._check_model_health(primary):
                return primary
        except Exception:
            pass

        # Secondary: Try balanced
        try:
            secondary = await self._route_balanced(request, eligible_models)
            if await self._check_model_health(secondary):
                return secondary
        except Exception:
            pass

        # Tertiary: Any available model
        for model in eligible_models:
            if await self._check_model_health(model):
                return model

        # Last resort: Return first available
        if eligible_models:
            return eligible_models[0]

        raise ValueError("No healthy models available")

    async def _calculate_request_cost(self, request: ArbitrageRequest, model: AIModel) -> float:
        """Calculate estimated cost for request with specific model"""

        input_cost = request.input_tokens * model.cost_per_input_token / 1000
        output_cost = request.estimated_output_tokens * model.cost_per_output_token / 1000

        return input_cost + output_cost

    async def _calculate_baseline_cost(self, request: ArbitrageRequest,
                                     eligible_models: List[AIModel]) -> float:
        """Calculate baseline cost (most expensive eligible model)"""

        costs = []
        for model in eligible_models:
            cost = await self._calculate_request_cost(request, model)
            costs.append(cost)

        return max(costs) if costs else 0

    async def _get_fallback_models(self, request: ArbitrageRequest,
                                 eligible_models: List[AIModel],
                                 selected_model: AIModel) -> List[AIModel]:
        """Get ordered list of fallback models"""

        fallbacks = [model for model in eligible_models if model != selected_model]

        # Sort fallbacks by balanced score
        scores = []
        for model in fallbacks:
            cost = await self._calculate_request_cost(request, model)
            score = model.quality_score * 0.4 + model.availability_score * 0.3 + (1/cost if cost > 0 else 0) * 0.3
            scores.append((score, model))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [model for _, model in scores[:3]]  # Top 3 fallbacks

    async def _calculate_confidence_score(self, request: ArbitrageRequest,
                                        selected_model: AIModel) -> float:
        """Calculate confidence score for routing decision"""

        # Base confidence from model availability and quality
        base_confidence = (selected_model.availability_score + selected_model.quality_score) / 2

        # Adjust based on historical performance
        historical_performance = await self._get_historical_performance(selected_model)
        performance_adjustment = (historical_performance - 50) * 0.2  # Scale -10 to +10

        # Adjust based on current load
        current_load = await self._get_current_provider_usage(selected_model.provider)
        load_factor = 1 - (current_load['rpm'] / selected_model.rate_limit_rpm)
        load_adjustment = load_factor * 10

        confidence = base_confidence + performance_adjustment + load_adjustment
        return max(0, min(100, confidence))

    async def _get_routing_reason(self, strategy: RoutingStrategy, model: AIModel) -> str:
        """Get human-readable routing reason"""

        reasons = {
            RoutingStrategy.COST_OPTIMAL: f"Selected for lowest cost: ${model.cost_per_input_token:.6f}/1K input tokens",
            RoutingStrategy.QUALITY_OPTIMAL: f"Selected for highest quality: {model.quality_score:.1f}/100 quality score",
            RoutingStrategy.LATENCY_OPTIMAL: f"Selected for lowest latency: {model.average_latency_ms:.0f}ms average",
            RoutingStrategy.BALANCED: f"Selected for optimal balance: {model.provider.value} {model.model_name}",
            RoutingStrategy.FALLBACK_CASCADE: f"Selected via fallback cascade: {model.provider.value} {model.model_name}"
        }

        return reasons.get(strategy, f"Selected: {model.provider.value} {model.model_name}")

    async def track_request_outcome(self, request_id: str, actual_cost: float,
                                  actual_latency_ms: float, quality_rating: Optional[float],
                                  success: bool, error_details: Optional[str] = None):
        """Track actual request outcome for learning"""

        # Store outcome for ML model training
        outcome_data = {
            'request_id': request_id,
            'actual_cost': actual_cost,
            'actual_latency_ms': actual_latency_ms,
            'quality_rating': quality_rating,
            'success': success,
            'error_details': error_details,
            'timestamp': datetime.now().isoformat()
        }

        # Store in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO optimization.arbitrage_outcomes (
                    request_id, actual_cost, actual_latency_ms, quality_rating,
                    success, error_details, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, request_id, actual_cost, actual_latency_ms, quality_rating,
                success, error_details, datetime.now())

        # Update model performance metrics
        await self._update_model_performance_metrics(request_id, outcome_data)

        # Trigger retraining if needed
        if not success:
            await self._trigger_model_retraining(request_id, error_details)

    async def get_cost_savings_analytics(self, tenant_id: str,
                                       days: int = 30) -> Dict[str, Any]:
        """Get cost savings analytics for tenant"""

        async with self.db_pool.acquire() as conn:
            # Get routing decisions and outcomes
            analytics_data = await conn.fetch("""
                SELECT
                    rd.selected_model,
                    rd.estimated_cost,
                    rd.cost_savings_percent,
                    ao.actual_cost,
                    rd.created_at
                FROM optimization.arbitrage_decisions rd
                LEFT JOIN optimization.arbitrage_outcomes ao ON rd.request_id = ao.request_id
                WHERE rd.tenant_id = $1
                AND rd.created_at > NOW() - INTERVAL '%s days'
                ORDER BY rd.created_at DESC
            """ % days, tenant_id)

            if not analytics_data:
                return {
                    'total_requests': 0,
                    'total_cost_savings': 0,
                    'average_savings_percent': 0,
                    'top_providers': [],
                    'savings_trend': []
                }

            # Calculate metrics
            total_requests = len(analytics_data)
            total_estimated_savings = sum(row['cost_savings_percent'] or 0 for row in analytics_data)
            average_savings_percent = total_estimated_savings / total_requests if total_requests > 0 else 0

            # Calculate actual cost savings where available
            actual_savings = []
            for row in analytics_data:
                if row['actual_cost'] and row['estimated_cost']:
                    baseline_cost = row['estimated_cost'] / (1 - (row['cost_savings_percent'] or 0) / 100)
                    actual_saving = baseline_cost - row['actual_cost']
                    actual_savings.append(actual_saving)

            total_actual_savings = sum(actual_savings)

            # Provider usage breakdown
            provider_usage = defaultdict(int)
            for row in analytics_data:
                if row['selected_model']:
                    provider = row['selected_model'].split(':')[0]
                    provider_usage[provider] += 1

            top_providers = sorted(
                [{'provider': k, 'requests': v, 'percentage': v/total_requests*100}
                 for k, v in provider_usage.items()],
                key=lambda x: x['requests'],
                reverse=True
            )[:5]

            # Daily savings trend
            daily_savings = defaultdict(lambda: {'requests': 0, 'savings': 0})
            for row in analytics_data:
                date_key = row['created_at'].date().isoformat()
                daily_savings[date_key]['requests'] += 1
                daily_savings[date_key]['savings'] += row['cost_savings_percent'] or 0

            savings_trend = [
                {
                    'date': date,
                    'requests': data['requests'],
                    'average_savings_percent': data['savings'] / data['requests'] if data['requests'] > 0 else 0
                }
                for date, data in sorted(daily_savings.items())
            ]

        return {
            'total_requests': total_requests,
            'total_estimated_savings_percent': total_estimated_savings,
            'total_actual_savings_dollars': total_actual_savings,
            'average_savings_percent': average_savings_percent,
            'top_providers': top_providers,
            'savings_trend': savings_trend,
            'period_days': days
        }

    async def optimize_pricing_rules(self, tenant_id: str) -> List[CostOptimizationRule]:
        """Generate optimized pricing rules based on usage patterns"""

        # Analyze usage patterns
        usage_patterns = await self._analyze_usage_patterns(tenant_id)

        # Generate optimization rules
        rules = []

        # Rule 1: High-volume cost optimization
        if usage_patterns['daily_requests'] > 1000:
            rules.append(CostOptimizationRule(
                rule_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                name="High Volume Cost Optimization",
                description="Route high-volume requests to most cost-effective providers",
                conditions={
                    "request_volume_threshold": 100,
                    "time_window_minutes": 60
                },
                actions={
                    "routing_strategy": "cost_optimal",
                    "quality_threshold": 75
                },
                priority=1,
                is_active=True,
                cost_savings_target=15.0
            ))

        # Rule 2: Off-peak optimization
        rules.append(CostOptimizationRule(
            rule_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            name="Off-Peak Hours Optimization",
            description="Use cheaper providers during off-peak hours",
            conditions={
                "time_range": ["22:00", "06:00"],
                "timezone": "UTC"
            },
            actions={
                "routing_strategy": "cost_optimal",
                "exclude_premium_providers": True
            },
            priority=2,
            is_active=True,
            cost_savings_target=25.0
        ))

        # Rule 3: Batch processing optimization
        if usage_patterns['batch_requests_percent'] > 20:
            rules.append(CostOptimizationRule(
                rule_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                name="Batch Processing Optimization",
                description="Optimize batch requests for maximum cost savings",
                conditions={
                    "request_type": "batch",
                    "min_batch_size": 10
                },
                actions={
                    "routing_strategy": "cost_optimal",
                    "enable_request_batching": True,
                    "max_wait_time_seconds": 30
                },
                priority=3,
                is_active=True,
                cost_savings_target=30.0
            ))

        # Store rules in database
        for rule in rules:
            await self._store_optimization_rule(rule)

        return rules

    # Background monitoring tasks
    async def _monitor_pricing_loop(self):
        """Background task to monitor and update pricing"""

        while True:
            try:
                await self._update_real_time_pricing()
                await asyncio.sleep(self.pricing_refresh_interval)
            except Exception as e:
                logger.error(f"Error in pricing monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _monitor_provider_health_loop(self):
        """Background task to monitor provider health and availability"""

        while True:
            try:
                await self._update_provider_health_metrics()
                await asyncio.sleep(self.metrics_refresh_interval)
            except Exception as e:
                logger.error(f"Error in provider health monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def _optimize_routing_algorithms_loop(self):
        """Background task to optimize routing algorithms based on performance"""

        while True:
            try:
                await self._optimize_routing_algorithms()
                await asyncio.sleep(3600)  # Optimize every hour
            except Exception as e:
                logger.error(f"Error in routing optimization loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    # Helper methods
    def _initialize_providers(self) -> Dict[AIProvider, Dict[str, Any]]:
        """Initialize AI provider configurations"""

        return {
            AIProvider.OPENAI: {
                'api_base': 'https://api.openai.com/v1',
                'models': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                'pricing_endpoint': 'https://openai.com/pricing'
            },
            AIProvider.ANTHROPIC: {
                'api_base': 'https://api.anthropic.com/v1',
                'models': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
                'pricing_endpoint': 'https://www.anthropic.com/pricing'
            },
            AIProvider.GOOGLE: {
                'api_base': 'https://generativelanguage.googleapis.com/v1',
                'models': ['gemini-pro', 'gemini-pro-vision'],
                'pricing_endpoint': 'https://cloud.google.com/vertex-ai/pricing'
            },
            # Add more providers...
        }

    async def _load_models_from_database(self):
        """Load AI models from database"""

        async with self.db_pool.acquire() as conn:
            models_data = await conn.fetch("""
                SELECT * FROM optimization.ai_models ORDER BY created_at DESC
            """)

            for row in models_data:
                model = AIModel(
                    model_id=row['model_id'],
                    provider=AIProvider(row['provider']),
                    model_name=row['model_name'],
                    capabilities=[ModelCapability(cap) for cap in row['capabilities']],
                    cost_per_input_token=row['cost_per_input_token'],
                    cost_per_output_token=row['cost_per_output_token'],
                    max_context_length=row['max_context_length'],
                    max_output_tokens=row['max_output_tokens'],
                    quality_score=row['quality_score'],
                    average_latency_ms=row['average_latency_ms'],
                    availability_score=row['availability_score'],
                    rate_limit_rpm=row['rate_limit_rpm'],
                    rate_limit_tpm=row['rate_limit_tpm'],
                    supports_streaming=row['supports_streaming'],
                    supports_function_calling=row['supports_function_calling'],
                    supports_vision=row['supports_vision'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                self.models_cache[model.model_id] = model

    async def _load_pricing_data(self):
        """Load current pricing data from cache/database"""

        # Load from Redis cache first
        cached_pricing = await self.redis.get("ai_pricing_data")
        if cached_pricing:
            self.pricing_cache = json.loads(cached_pricing)
        else:
            # Load from database and cache
            async with self.db_pool.acquire() as conn:
                pricing_data = await conn.fetch("""
                    SELECT provider, model_name, cost_per_input_token, cost_per_output_token, updated_at
                    FROM optimization.ai_pricing
                    WHERE updated_at > NOW() - INTERVAL '24 hours'
                """)

                for row in pricing_data:
                    key = f"{row['provider']}:{row['model_name']}"
                    self.pricing_cache[key] = {
                        'input_cost': row['cost_per_input_token'],
                        'output_cost': row['cost_per_output_token'],
                        'updated_at': row['updated_at'].isoformat()
                    }

                # Cache for 1 hour
                await self.redis.setex("ai_pricing_data", 3600, json.dumps(self.pricing_cache, default=str))

    async def _load_provider_metrics(self):
        """Load provider performance metrics"""

        async with self.db_pool.acquire() as conn:
            metrics_data = await conn.fetch("""
                SELECT provider, success_rate, average_latency_ms, average_cost_per_request,
                       total_requests_24h, total_cost_24h, quality_score, last_updated
                FROM optimization.provider_metrics
                WHERE last_updated > NOW() - INTERVAL '1 hour'
            """)

            for row in metrics_data:
                self.metrics_cache[row['provider']] = ProviderMetrics(
                    provider=AIProvider(row['provider']),
                    success_rate=row['success_rate'],
                    average_latency_ms=row['average_latency_ms'],
                    average_cost_per_request=row['average_cost_per_request'],
                    total_requests_24h=row['total_requests_24h'],
                    total_cost_24h=row['total_cost_24h'],
                    quality_score=row['quality_score'],
                    last_updated=row['last_updated']
                )

    async def _get_current_provider_usage(self, provider: AIProvider) -> Dict[str, int]:
        """Get current provider usage (RPM, TPM)"""

        # Get from Redis cache
        usage_key = f"provider_usage:{provider.value}"
        cached_usage = await self.redis.get(usage_key)

        if cached_usage:
            return json.loads(cached_usage)

        # Default usage if not cached
        return {'rpm': 0, 'tpm': 0}

    async def _check_model_health(self, model: AIModel) -> bool:
        """Check if model is currently healthy and available"""

        # Check availability score
        if model.availability_score < self.min_availability_score:
            return False

        # Check recent error rate
        error_rate = await self._get_recent_error_rate(model)
        if error_rate > 0.1:  # 10% error rate threshold
            return False

        # Check rate limits
        current_usage = await self._get_current_provider_usage(model.provider)
        if current_usage['rpm'] >= model.rate_limit_rpm * 0.95:  # 95% threshold
            return False

        return True

    async def _get_recent_error_rate(self, model: AIModel) -> float:
        """Get recent error rate for model"""

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN success = false THEN 1 END) as failed_requests
                FROM optimization.arbitrage_outcomes ao
                JOIN optimization.arbitrage_decisions ad ON ao.request_id = ad.request_id
                WHERE ad.selected_model LIKE $1
                AND ao.created_at > NOW() - INTERVAL '1 hour'
            """, f"%{model.provider.value}:{model.model_name}%")

            if result and result['total_requests'] > 0:
                return result['failed_requests'] / result['total_requests']

            return 0.0

    async def _get_historical_performance(self, model: AIModel) -> float:
        """Get historical performance score for model"""

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT
                    AVG(CASE WHEN success THEN 100 ELSE 0 END) as success_rate,
                    AVG(quality_rating) as avg_quality
                FROM optimization.arbitrage_outcomes ao
                JOIN optimization.arbitrage_decisions ad ON ao.request_id = ad.request_id
                WHERE ad.selected_model LIKE $1
                AND ao.created_at > NOW() - INTERVAL '7 days'
            """, f"%{model.provider.value}:{model.model_name}%")

            if result:
                success_rate = result['success_rate'] or 50
                avg_quality = result['avg_quality'] or 50
                return (success_rate + avg_quality) / 2

            return 50.0  # Default neutral score

    async def _store_routing_decision(self, request: ArbitrageRequest, result: ArbitrageResult):
        """Store routing decision for analytics and learning"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO optimization.arbitrage_decisions (
                    request_id, tenant_id, capability, input_tokens, estimated_output_tokens,
                    strategy, selected_model, estimated_cost, estimated_latency_ms,
                    cost_savings_percent, routing_reason, confidence_score, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, request.request_id, request.tenant_id, request.capability.value,
                request.input_tokens, request.estimated_output_tokens, request.strategy.value,
                f"{result.selected_model.provider.value}:{result.selected_model.model_name}",
                result.estimated_cost, result.estimated_latency_ms, result.cost_savings_percent,
                result.routing_reason, result.confidence_score, result.created_at)

    async def _update_routing_metrics(self, tenant_id: str, routing_time_ms: float,
                                    cost_savings_percent: float):
        """Update routing performance metrics"""

        # Update in Redis for real-time metrics
        metrics_key = f"routing_metrics:{tenant_id}"
        current_metrics = await self.redis.get(metrics_key)

        if current_metrics:
            metrics = json.loads(current_metrics)
        else:
            metrics = {
                'total_requests': 0,
                'total_routing_time_ms': 0,
                'total_cost_savings_percent': 0,
                'last_updated': datetime.now().isoformat()
            }

        metrics['total_requests'] += 1
        metrics['total_routing_time_ms'] += routing_time_ms
        metrics['total_cost_savings_percent'] += cost_savings_percent
        metrics['average_routing_time_ms'] = metrics['total_routing_time_ms'] / metrics['total_requests']
        metrics['average_cost_savings_percent'] = metrics['total_cost_savings_percent'] / metrics['total_requests']
        metrics['last_updated'] = datetime.now().isoformat()

        # Cache for 1 hour
        await self.redis.setex(metrics_key, 3600, json.dumps(metrics))

    async def _update_model_performance_metrics(self, request_id: str, outcome_data: Dict[str, Any]):
        """Update model performance metrics based on actual outcomes"""

        # Get the routing decision
        async with self.db_pool.acquire() as conn:
            decision = await conn.fetchrow("""
                SELECT selected_model, estimated_cost, estimated_latency_ms
                FROM optimization.arbitrage_decisions
                WHERE request_id = $1
            """, request_id)

            if not decision:
                return

            # Update model metrics
            provider_model = decision['selected_model']

            # Calculate accuracy metrics
            cost_accuracy = 1 - abs(decision['estimated_cost'] - outcome_data['actual_cost']) / decision['estimated_cost'] if decision['estimated_cost'] > 0 else 0
            latency_accuracy = 1 - abs(decision['estimated_latency_ms'] - outcome_data['actual_latency_ms']) / decision['estimated_latency_ms'] if decision['estimated_latency_ms'] > 0 else 0

            # Update model performance in database
            await conn.execute("""
                INSERT INTO optimization.model_performance_history (
                    model_identifier, request_id, cost_accuracy, latency_accuracy,
                    quality_rating, success, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, provider_model, request_id, cost_accuracy, latency_accuracy,
                outcome_data.get('quality_rating'), outcome_data['success'], datetime.now())

    async def _trigger_model_retraining(self, request_id: str, error_details: str):
        """Trigger model retraining when failures occur"""

        # Log the failure for analysis
        logger.warning(f"Model failure for request {request_id}: {error_details}")

        # Store failure for retraining dataset
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO optimization.model_failures (
                    request_id, error_details, created_at
                ) VALUES ($1, $2, $3)
            """, request_id, error_details, datetime.now())

        # Check if retraining threshold is reached
        failure_count = await self._get_recent_failure_count()
        if failure_count > 10:  # Threshold for retraining
            await self._schedule_model_retraining()

    async def _get_recent_failure_count(self) -> int:
        """Get count of recent failures"""

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT COUNT(*) as failure_count
                FROM optimization.model_failures
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)

            return result['failure_count'] if result else 0

    async def _schedule_model_retraining(self):
        """Schedule model retraining job"""

        # In a production system, this would trigger a ML pipeline
        logger.info("Scheduling model retraining due to high failure rate")

        # For now, just log and potentially adjust routing weights
        await self._adjust_routing_weights_for_failures()

    async def _adjust_routing_weights_for_failures(self):
        """Adjust routing weights based on recent failures"""

        # Get failure patterns by provider
        async with self.db_pool.acquire() as conn:
            failure_patterns = await conn.fetch("""
                SELECT
                    ad.selected_model,
                    COUNT(*) as failure_count
                FROM optimization.model_failures mf
                JOIN optimization.arbitrage_decisions ad ON mf.request_id = ad.request_id
                WHERE mf.created_at > NOW() - INTERVAL '24 hours'
                GROUP BY ad.selected_model
                ORDER BY failure_count DESC
            """)

            # Temporarily reduce routing to high-failure models
            for pattern in failure_patterns:
                if pattern['failure_count'] > 5:
                    model_key = pattern['selected_model']
                    penalty_key = f"routing_penalty:{model_key}"

                    # Apply penalty for 1 hour
                    await self.redis.setex(penalty_key, 3600, pattern['failure_count'])
                    logger.info(f"Applied routing penalty to {model_key} due to {pattern['failure_count']} failures")

    async def _analyze_usage_patterns(self, tenant_id: str) -> Dict[str, Any]:
        """Analyze tenant usage patterns for optimization"""

        async with self.db_pool.acquire() as conn:
            # Get usage statistics
            usage_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_requests,
                    AVG(input_tokens) as avg_input_tokens,
                    AVG(estimated_output_tokens) as avg_output_tokens,
                    COUNT(*) / 30.0 as daily_requests,
                    COUNT(CASE WHEN capability = 'text_generation' THEN 1 END) * 100.0 / COUNT(*) as text_gen_percent,
                    COUNT(CASE WHEN strategy = 'cost_optimal' THEN 1 END) * 100.0 / COUNT(*) as cost_focused_percent
                FROM optimization.arbitrage_decisions
                WHERE tenant_id = $1
                AND created_at > NOW() - INTERVAL '30 days'
            """, tenant_id)

            # Get time-based patterns
            hourly_patterns = await conn.fetch("""
                SELECT
                    EXTRACT(hour FROM created_at) as hour,
                    COUNT(*) as request_count
                FROM optimization.arbitrage_decisions
                WHERE tenant_id = $1
                AND created_at > NOW() - INTERVAL '7 days'
                GROUP BY EXTRACT(hour FROM created_at)
                ORDER BY hour
            """)

            # Identify peak and off-peak hours
            hourly_counts = {int(row['hour']): row['request_count'] for row in hourly_patterns}
            avg_hourly = sum(hourly_counts.values()) / len(hourly_counts) if hourly_counts else 0
            peak_hours = [hour for hour, count in hourly_counts.items() if count > avg_hourly * 1.5]
            off_peak_hours = [hour for hour, count in hourly_counts.items() if count < avg_hourly * 0.5]

            return {
                'total_requests': usage_stats['total_requests'] or 0,
                'daily_requests': usage_stats['daily_requests'] or 0,
                'avg_input_tokens': usage_stats['avg_input_tokens'] or 0,
                'avg_output_tokens': usage_stats['avg_output_tokens'] or 0,
                'text_gen_percent': usage_stats['text_gen_percent'] or 0,
                'cost_focused_percent': usage_stats['cost_focused_percent'] or 0,
                'batch_requests_percent': 0,  # Would need additional analysis
                'peak_hours': peak_hours,
                'off_peak_hours': off_peak_hours,
                'hourly_patterns': hourly_counts
            }

    async def _store_optimization_rule(self, rule: CostOptimizationRule):
        """Store optimization rule in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO optimization.cost_optimization_rules (
                    rule_id, tenant_id, name, description, conditions, actions,
                    priority, is_active, cost_savings_target, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, rule.rule_id, rule.tenant_id, rule.name, rule.description,
                json.dumps(rule.conditions), json.dumps(rule.actions),
                rule.priority, rule.is_active, rule.cost_savings_target,
                rule.created_at, rule.updated_at)

    async def _update_real_time_pricing(self):
        """Update real-time pricing from providers"""

        # This would integrate with provider APIs to get current pricing
        # For now, simulate with database updates

        try:
            # Update pricing cache from external sources
            for provider, config in self.providers.items():
                pricing_data = await self._fetch_provider_pricing(provider)
                if pricing_data:
                    await self._update_provider_pricing(provider, pricing_data)

            logger.info("Real-time pricing updated successfully")

        except Exception as e:
            logger.error(f"Error updating real-time pricing: {e}")

    async def _fetch_provider_pricing(self, provider: AIProvider) -> Optional[Dict[str, Any]]:
        """Fetch current pricing from provider"""

        # This would make actual API calls to providers
        # For now, return simulated data

        pricing_data = {
            AIProvider.OPENAI: {
                'gpt-4': {'input': 0.00003, 'output': 0.00006},
                'gpt-3.5-turbo': {'input': 0.000001, 'output': 0.000002}
            },
            AIProvider.ANTHROPIC: {
                'claude-3-opus': {'input': 0.000015, 'output': 0.000075},
                'claude-3-sonnet': {'input': 0.000003, 'output': 0.000015}
            }
        }

        return pricing_data.get(provider)

    async def _update_provider_pricing(self, provider: AIProvider, pricing_data: Dict[str, Any]):
        """Update provider pricing in database"""

        async with self.db_pool.acquire() as conn:
            for model_name, costs in pricing_data.items():
                await conn.execute("""
                    INSERT INTO optimization.ai_pricing (
                        provider, model_name, cost_per_input_token, cost_per_output_token, updated_at
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (provider, model_name)
                    DO UPDATE SET
                        cost_per_input_token = EXCLUDED.cost_per_input_token,
                        cost_per_output_token = EXCLUDED.cost_per_output_token,
                        updated_at = EXCLUDED.updated_at
                """, provider.value, model_name, costs['input'], costs['output'], datetime.now())

    async def _update_provider_health_metrics(self):
        """Update provider health and availability metrics"""

        for provider in AIProvider:
            try:
                # Check provider health
                health_data = await self._check_provider_health(provider)

                # Update metrics in database
                await self._store_provider_metrics(provider, health_data)

            except Exception as e:
                logger.error(f"Error updating health metrics for {provider.value}: {e}")

    async def _check_provider_health(self, provider: AIProvider) -> Dict[str, Any]:
        """Check provider health status"""

        # This would make actual health checks to providers
        # For now, simulate health data

        import random

        return {
            'success_rate': random.uniform(0.95, 0.99),
            'average_latency_ms': random.uniform(200, 800),
            'availability_score': random.uniform(95, 99.9),
            'quality_score': random.uniform(75, 95)
        }

    async def _store_provider_metrics(self, provider: AIProvider, health_data: Dict[str, Any]):
        """Store provider metrics in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO optimization.provider_metrics (
                    provider, success_rate, average_latency_ms, average_cost_per_request,
                    total_requests_24h, total_cost_24h, quality_score, last_updated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (provider)
                DO UPDATE SET
                    success_rate = EXCLUDED.success_rate,
                    average_latency_ms = EXCLUDED.average_latency_ms,
                    quality_score = EXCLUDED.quality_score,
                    last_updated = EXCLUDED.last_updated
            """, provider.value, health_data['success_rate'], health_data['average_latency_ms'],
                0.0, 0, 0.0, health_data['quality_score'], datetime.now())

    async def _optimize_routing_algorithms(self):
        """Optimize routing algorithms based on performance data"""

        try:
            # Analyze routing performance
            performance_data = await self._analyze_routing_performance()

            # Adjust routing weights based on performance
            await self._adjust_routing_weights(performance_data)

            logger.info("Routing algorithms optimized successfully")

        except Exception as e:
            logger.error(f"Error optimizing routing algorithms: {e}")

    async def _analyze_routing_performance(self) -> Dict[str, Any]:
        """Analyze routing performance across all strategies"""

        async with self.db_pool.acquire() as conn:
            performance_data = await conn.fetch("""
                SELECT
                    strategy,
                    AVG(cost_savings_percent) as avg_cost_savings,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN ao.success THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM optimization.arbitrage_decisions ad
                LEFT JOIN optimization.arbitrage_outcomes ao ON ad.request_id = ao.request_id
                WHERE ad.created_at > NOW() - INTERVAL '24 hours'
                GROUP BY strategy
            """)

            return {row['strategy']: dict(row) for row in performance_data}

    async def _adjust_routing_weights(self, performance_data: Dict[str, Any]):
        """Adjust routing algorithm weights based on performance"""

        # Calculate optimal weights based on performance
        best_strategy = None
        best_score = 0

        for strategy, data in performance_data.items():
            # Composite score based on cost savings, confidence, and success rate
            score = (
                data.get('avg_cost_savings', 0) * 0.4 +
                data.get('avg_confidence', 0) * 0.3 +
                data.get('success_rate', 0) * 0.3
            )

            if score > best_score:
                best_score = score
                best_strategy = strategy

        if best_strategy:
            # Store optimal strategy weights
            await self.redis.setex(
                "optimal_routing_strategy",
                3600,
                json.dumps({
                    'strategy': best_strategy,
                    'score': best_score,
                    'updated_at': datetime.now().isoformat()
                })
            )

            logger.info(f"Optimal routing strategy updated to: {best_strategy} (score: {best_score:.2f})")


class ArbitrageManager:
    """
    High-level arbitrage management orchestrator
    """

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis = redis_client
        self.engines = {}  # tenant_id -> ArbitrageEngine

    async def get_engine(self, tenant_id: str) -> AIArbitrageEngine:
        """Get or create arbitrage engine for tenant"""

        if tenant_id not in self.engines:
            engine = AIArbitrageEngine(self.db_pool, self.redis)
            await engine.initialize()
            self.engines[tenant_id] = engine

        return self.engines[tenant_id]

    async def route_ai_request(self, tenant_id: str, capability: str,
                             input_tokens: int, estimated_output_tokens: int,
                             strategy: str = "balanced", **kwargs) -> ArbitrageResult:
        """Route AI request through arbitrage engine"""

        engine = await self.get_engine(tenant_id)

        request = ArbitrageRequest(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            capability=ModelCapability(capability),
            input_tokens=input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            strategy=RoutingStrategy(strategy),
            max_latency_ms=kwargs.get('max_latency_ms'),
            quality_threshold=kwargs.get('quality_threshold'),
            cost_budget=kwargs.get('cost_budget'),
            fallback_enabled=kwargs.get('fallback_enabled', True),
            context_length_required=kwargs.get('context_length_required', 4000),
            requires_streaming=kwargs.get('requires_streaming', False),
            requires_function_calling=kwargs.get('requires_function_calling', False),
            requires_vision=kwargs.get('requires_vision', False),
            metadata=kwargs.get('metadata', {})
        )

        return await engine.route_request(request)

    async def track_request_outcome(self, tenant_id: str, request_id: str,
                                  actual_cost: float, actual_latency_ms: float,
                                  quality_rating: Optional[float], success: bool,
                                  error_details: Optional[str] = None):
        """Track actual request outcome"""

        engine = await self.get_engine(tenant_id)
        await engine.track_request_outcome(
            request_id, actual_cost, actual_latency_ms,
            quality_rating, success, error_details
        )

    async def get_cost_savings_report(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """Get cost savings analytics report"""

        engine = await self.get_engine(tenant_id)
        return await engine.get_cost_savings_analytics(tenant_id, days)

    async def optimize_tenant_rules(self, tenant_id: str) -> List[CostOptimizationRule]:
        """Generate optimized rules for tenant"""

        engine = await self.get_engine(tenant_id)
        return await engine.optimize_pricing_rules(tenant_id)
