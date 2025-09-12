"""
AI Arbitrage API Endpoints - AgentSystem Profit Machine
Intelligent AI provider routing and cost optimization endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field
import asyncpg
import uuid

from ..optimization.ai_arbitrage_engine import (
    ArbitrageManager, ModelCapability, RoutingStrategy,
    AIProvider, CostOptimizationRule
)
from ..auth.auth_service import get_current_user, require_permissions
from ..database.connection import get_db_pool
import aioredis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/arbitrage", tags=["arbitrage"])
security = HTTPBearer()

# Pydantic models for request/response
class ArbitrageRequest(BaseModel):
    capability: str = Field(..., description="AI capability needed (text_generation, code_generation, etc.)")
    input_tokens: int = Field(..., description="Number of input tokens")
    estimated_output_tokens: int = Field(..., description="Estimated output tokens")
    strategy: str = Field(default="balanced", description="Routing strategy")
    max_latency_ms: Optional[int] = Field(None, description="Maximum acceptable latency in ms")
    quality_threshold: Optional[float] = Field(None, description="Minimum quality score (0-100)")
    cost_budget: Optional[float] = Field(None, description="Maximum cost budget")
    context_length_required: int = Field(default=4000, description="Required context length")
    requires_streaming: bool = Field(default=False, description="Requires streaming support")
    requires_function_calling: bool = Field(default=False, description="Requires function calling")
    requires_vision: bool = Field(default=False, description="Requires vision capabilities")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

class ArbitrageResponse(BaseModel):
    request_id: str
    selected_model: Dict[str, Any]
    estimated_cost: float
    estimated_latency_ms: float
    cost_savings_percent: float
    fallback_models: List[Dict[str, Any]]
    routing_reason: str
    confidence_score: float
    created_at: datetime

class OutcomeTrackingRequest(BaseModel):
    request_id: str = Field(..., description="Original arbitrage request ID")
    actual_cost: float = Field(..., description="Actual cost incurred")
    actual_latency_ms: float = Field(..., description="Actual latency in milliseconds")
    quality_rating: Optional[float] = Field(None, description="Quality rating (0-100)")
    success: bool = Field(..., description="Whether the request was successful")
    error_details: Optional[str] = Field(None, description="Error details if failed")

class OptimizationRuleRequest(BaseModel):
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    conditions: Dict[str, Any] = Field(..., description="Conditions for rule activation")
    actions: Dict[str, Any] = Field(..., description="Actions to take when rule matches")
    priority: int = Field(default=1, description="Rule priority")
    cost_savings_target: float = Field(default=0, description="Target cost savings percentage")

class CostSavingsAnalytics(BaseModel):
    total_requests: int
    total_estimated_savings_percent: float
    total_actual_savings_dollars: float
    average_savings_percent: float
    top_providers: List[Dict[str, Any]]
    savings_trend: List[Dict[str, Any]]
    period_days: int

# Dependency to get arbitrage manager
async def get_arbitrage_manager() -> ArbitrageManager:
    db_pool = await get_db_pool()
    # Get Redis client (simplified for example)
    redis_client = None  # Would be initialized properly in production
    return ArbitrageManager(db_pool, redis_client)

@router.post("/route", response_model=ArbitrageResponse)
async def route_ai_request(
    request: ArbitrageRequest,
    current_user = Depends(get_current_user),
    arbitrage_manager: ArbitrageManager = Depends(get_arbitrage_manager)
):
    """Route AI request through intelligent arbitrage engine"""

    # Validate capability
    try:
        ModelCapability(request.capability)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid capability: {request.capability}")

    # Validate strategy
    try:
        RoutingStrategy(request.strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")

    try:
        # Route the request
        result = await arbitrage_manager.route_ai_request(
            tenant_id=current_user['tenant_id'],
            capability=request.capability,
            input_tokens=request.input_tokens,
            estimated_output_tokens=request.estimated_output_tokens,
            strategy=request.strategy,
            max_latency_ms=request.max_latency_ms,
            quality_threshold=request.quality_threshold,
            cost_budget=request.cost_budget,
            context_length_required=request.context_length_required,
            requires_streaming=request.requires_streaming,
            requires_function_calling=request.requires_function_calling,
            requires_vision=request.requires_vision,
            metadata=request.metadata
        )

        return ArbitrageResponse(
            request_id=result.request_id,
            selected_model={
                'provider': result.selected_model.provider.value,
                'model_name': result.selected_model.model_name,
                'quality_score': result.selected_model.quality_score,
                'availability_score': result.selected_model.availability_score,
                'cost_per_input_token': result.selected_model.cost_per_input_token,
                'cost_per_output_token': result.selected_model.cost_per_output_token
            },
            estimated_cost=result.estimated_cost,
            estimated_latency_ms=result.estimated_latency_ms,
            cost_savings_percent=result.cost_savings_percent,
            fallback_models=[
                {
                    'provider': model.provider.value,
                    'model_name': model.model_name,
                    'quality_score': model.quality_score
                }
                for model in result.fallback_models
            ],
            routing_reason=result.routing_reason,
            confidence_score=result.confidence_score,
            created_at=result.created_at
        )

    except Exception as e:
        logger.error(f"Error routing AI request: {e}")
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")

@router.post("/track-outcome")
async def track_request_outcome(
    request: OutcomeTrackingRequest,
    current_user = Depends(get_current_user),
    arbitrage_manager: ArbitrageManager = Depends(get_arbitrage_manager)
):
    """Track actual outcome of an AI request for learning and optimization"""

    try:
        await arbitrage_manager.track_request_outcome(
            tenant_id=current_user['tenant_id'],
            request_id=request.request_id,
            actual_cost=request.actual_cost,
            actual_latency_ms=request.actual_latency_ms,
            quality_rating=request.quality_rating,
            success=request.success,
            error_details=request.error_details
        )

        return {"message": "Outcome tracked successfully", "request_id": request.request_id}

    except Exception as e:
        logger.error(f"Error tracking outcome: {e}")
        raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")

@router.get("/analytics/cost-savings", response_model=CostSavingsAnalytics)
async def get_cost_savings_analytics(
    days: int = Query(30, description="Number of days to analyze", le=365),
    current_user = Depends(get_current_user),
    arbitrage_manager: ArbitrageManager = Depends(get_arbitrage_manager)
):
    """Get cost savings analytics and reporting"""

    try:
        analytics = await arbitrage_manager.get_cost_savings_report(
            current_user['tenant_id'], days
        )

        return CostSavingsAnalytics(**analytics)

    except Exception as e:
        logger.error(f"Error getting cost savings analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

@router.get("/models")
async def list_available_models(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    capability: Optional[str] = Query(None, description="Filter by capability"),
    min_quality_score: Optional[float] = Query(None, description="Minimum quality score"),
    active_only: bool = Query(True, description="Show only active models"),
    limit: int = Query(50, le=100, description="Number of models to return"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List available AI models with their capabilities and pricing"""

    query = """
        SELECT model_id, provider, model_name, capabilities, cost_per_input_token,
               cost_per_output_token, max_context_length, max_output_tokens,
               quality_score, average_latency_ms, availability_score,
               rate_limit_rpm, rate_limit_tpm, supports_streaming,
               supports_function_calling, supports_vision, created_at
        FROM optimization.ai_models
        WHERE 1=1
    """
    params = []
    param_count = 0

    if active_only:
        query += " AND is_active = true"

    if provider:
        param_count += 1
        query += f" AND provider = ${param_count}"
        params.append(provider)

    if capability:
        param_count += 1
        query += f" AND ${param_count} = ANY(capabilities)"
        params.append(capability)

    if min_quality_score:
        param_count += 1
        query += f" AND quality_score >= ${param_count}"
        params.append(min_quality_score)

    query += f" ORDER BY quality_score DESC, average_latency_ms ASC LIMIT ${param_count + 1}"
    params.append(limit)

    async with db_pool.acquire() as conn:
        models = await conn.fetch(query, *params)

    return [dict(model) for model in models]

@router.get("/providers")
async def list_providers(
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List AI providers with their current metrics"""

    async with db_pool.acquire() as conn:
        providers = await conn.fetch("""
            SELECT provider, success_rate, average_latency_ms, average_cost_per_request,
                   total_requests_24h, total_cost_24h, quality_score, last_updated
            FROM optimization.provider_metrics
            ORDER BY quality_score DESC, success_rate DESC
        """)

    return [dict(provider) for provider in providers]

@router.post("/optimization-rules")
async def create_optimization_rule(
    request: OptimizationRuleRequest,
    current_user = Depends(get_current_user),
    arbitrage_manager: ArbitrageManager = Depends(get_arbitrage_manager)
):
    """Create a custom cost optimization rule"""

    try:
        rules = await arbitrage_manager.optimize_tenant_rules(current_user['tenant_id'])

        # For now, return the auto-generated rules
        # In production, would create custom rule based on request

        return {
            "message": "Optimization rules created successfully",
            "rules_generated": len(rules),
            "estimated_savings": sum(rule.cost_savings_target for rule in rules)
        }

    except Exception as e:
        logger.error(f"Error creating optimization rule: {e}")
        raise HTTPException(status_code=500, detail=f"Rule creation failed: {str(e)}")

@router.get("/optimization-rules")
async def list_optimization_rules(
    active_only: bool = Query(True, description="Show only active rules"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List cost optimization rules for tenant"""

    query = """
        SELECT rule_id, name, description, conditions, actions, priority,
               is_active, cost_savings_target, times_triggered,
               total_savings_achieved, created_at, updated_at
        FROM optimization.cost_optimization_rules
        WHERE tenant_id = $1
    """
    params = [current_user['tenant_id']]

    if active_only:
        query += " AND is_active = true"

    query += " ORDER BY priority DESC, created_at DESC"

    async with db_pool.acquire() as conn:
        rules = await conn.fetch(query, *params)

    return [dict(rule) for rule in rules]

@router.get("/dashboard")
async def get_arbitrage_dashboard(
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Get arbitrage dashboard overview"""

    async with db_pool.acquire() as conn:
        # Get dashboard data from view
        dashboard_data = await conn.fetchrow("""
            SELECT * FROM optimization.tenant_arbitrage_dashboard
            WHERE tenant_id = $1
        """, current_user['tenant_id'])

        if not dashboard_data:
            return {
                'tenant_id': current_user['tenant_id'],
                'total_requests': 0,
                'avg_cost_savings': 0,
                'total_estimated_cost': 0,
                'total_actual_cost': 0,
                'avg_latency_ms': 0,
                'success_rate': 0,
                'providers_used': 0,
                'models_used': 0,
                'last_request_at': None
            }

        # Get recent routing decisions
        recent_decisions = await conn.fetch("""
            SELECT request_id, capability, strategy, selected_model,
                   estimated_cost, cost_savings_percent, confidence_score, created_at
            FROM optimization.arbitrage_decisions
            WHERE tenant_id = $1
            ORDER BY created_at DESC
            LIMIT 10
        """, current_user['tenant_id'])

        # Get provider usage breakdown
        provider_usage = await conn.fetch("""
            SELECT
                SPLIT_PART(selected_model, ':', 1) as provider,
                COUNT(*) as requests,
                AVG(cost_savings_percent) as avg_savings,
                SUM(estimated_cost) as total_cost
            FROM optimization.arbitrage_decisions
            WHERE tenant_id = $1
            AND created_at > NOW() - INTERVAL '7 days'
            GROUP BY SPLIT_PART(selected_model, ':', 1)
            ORDER BY requests DESC
        """, current_user['tenant_id'])

        # Get cost savings trends
        savings_trends = await conn.fetch("""
            SELECT date, avg_cost_savings, total_requests
            FROM optimization.cost_savings_summary
            WHERE tenant_id = $1
            AND date > CURRENT_DATE - INTERVAL '30 days'
            ORDER BY date DESC
        """, current_user['tenant_id'])

    return {
        'summary': dict(dashboard_data),
        'recent_decisions': [dict(decision) for decision in recent_decisions],
        'provider_usage': [dict(usage) for usage in provider_usage],
        'savings_trends': [dict(trend) for trend in savings_trends]
    }

@router.get("/strategies")
async def get_routing_strategies():
    """Get available routing strategies and their descriptions"""

    strategies = [
        {
            "strategy": "cost_optimal",
            "name": "Cost Optimal",
            "description": "Routes to the most cost-effective model that meets requirements",
            "best_for": "High-volume, cost-sensitive workloads"
        },
        {
            "strategy": "quality_optimal",
            "name": "Quality Optimal",
            "description": "Routes to the highest quality model within budget constraints",
            "best_for": "Critical tasks requiring highest accuracy"
        },
        {
            "strategy": "latency_optimal",
            "name": "Latency Optimal",
            "description": "Routes to the fastest model that meets requirements",
            "best_for": "Real-time applications requiring low latency"
        },
        {
            "strategy": "balanced",
            "name": "Balanced",
            "description": "Optimizes across cost, quality, latency and availability",
            "best_for": "General purpose workloads"
        },
        {
            "strategy": "fallback_cascade",
            "name": "Fallback Cascade",
            "description": "Intelligent fallback routing with multiple backup options",
            "best_for": "High-availability requirements"
        }
    ]

    return {"strategies": strategies}

@router.get("/capabilities")
async def get_model_capabilities():
    """Get available model capabilities"""

    capabilities = [
        {
            "capability": "text_generation",
            "name": "Text Generation",
            "description": "General text generation and completion"
        },
        {
            "capability": "code_generation",
            "name": "Code Generation",
            "description": "Programming code generation and completion"
        },
        {
            "capability": "image_generation",
            "name": "Image Generation",
            "description": "AI image and artwork generation"
        },
        {
            "capability": "image_analysis",
            "name": "Image Analysis",
            "description": "Computer vision and image understanding"
        },
        {
            "capability": "embedding",
            "name": "Embeddings",
            "description": "Text and data embeddings for ML"
        },
        {
            "capability": "function_calling",
            "name": "Function Calling",
            "description": "Structured function and API calling"
        },
        {
            "capability": "long_context",
            "name": "Long Context",
            "description": "Processing of very long documents"
        },
        {
            "capability": "multimodal",
            "name": "Multimodal",
            "description": "Text, image, and mixed media processing"
        }
    ]

    return {"capabilities": capabilities}

@router.post("/batch-optimize")
async def optimize_batch_requests(
    requests: List[ArbitrageRequest],
    current_user = Depends(get_current_user),
    arbitrage_manager: ArbitrageManager = Depends(get_arbitrage_manager)
):
    """Optimize a batch of requests for maximum cost savings"""

    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")

    try:
        results = []
        total_estimated_cost = 0
        total_estimated_savings = 0

        for req in requests:
            result = await arbitrage_manager.route_ai_request(
                tenant_id=current_user['tenant_id'],
                capability=req.capability,
                input_tokens=req.input_tokens,
                estimated_output_tokens=req.estimated_output_tokens,
                strategy="cost_optimal",  # Force cost optimal for batch
                max_latency_ms=req.max_latency_ms,
                quality_threshold=req.quality_threshold,
                cost_budget=req.cost_budget,
                context_length_required=req.context_length_required,
                requires_streaming=req.requires_streaming,
                requires_function_calling=req.requires_function_calling,
                requires_vision=req.requires_vision,
                metadata=req.metadata
            )

            results.append({
                'request_id': result.request_id,
                'selected_provider': result.selected_model.provider.value,
                'selected_model': result.selected_model.model_name,
                'estimated_cost': result.estimated_cost,
                'cost_savings_percent': result.cost_savings_percent
            })

            total_estimated_cost += result.estimated_cost
            total_estimated_savings += result.cost_savings_percent

        avg_savings_percent = total_estimated_savings / len(requests) if requests else 0

        return {
            'batch_id': str(uuid.uuid4()),
            'total_requests': len(requests),
            'total_estimated_cost': total_estimated_cost,
            'average_savings_percent': avg_savings_percent,
            'results': results
        }

    except Exception as e:
        logger.error(f"Error optimizing batch requests: {e}")
        raise HTTPException(status_code=500, detail=f"Batch optimization failed: {str(e)}")

# Include router in main application
def setup_arbitrage_routes(app):
    """Setup arbitrage routes in FastAPI application"""
    app.include_router(router)
