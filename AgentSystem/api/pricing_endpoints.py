
"""
Dynamic Pricing API Endpoints - AgentSystem Profit Machine
Advanced value-based pricing optimization system endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta, date
from uuid import UUID, uuid4
import asyncio
import json
from enum import Enum

from ..auth.auth_service import verify_token, get_current_tenant
from ..database.connection import get_db_connection
from ..pricing.dynamic_pricing_engine import (
    DynamicPricingEngine, PricingStrategy, PricingTier, PriceAdjustmentType,
    PricingRecommendation, CustomerPricingProfile, MarketConditions
)

# Initialize router
router = APIRouter(prefix="/api/v1/pricing", tags=["Dynamic Pricing"])
security = HTTPBearer()

# Enums
class PricingStrategyAPI(str, Enum):
    VALUE_BASED = "value_based"
    USAGE_BASED = "usage_based"
    COMPETITIVE = "competitive"
    PENETRATION = "penetration"
    PREMIUM = "premium"
    DYNAMIC = "dynamic"

class PricingTierAPI(str, Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class PriceAdjustmentTypeAPI(str, Enum):
    DISCOUNT = "discount"
    PREMIUM = "premium"
    LOYALTY = "loyalty"
    VOLUME = "volume"
    COMPETITIVE = "competitive"
    VALUE_REALIZATION = "value_realization"

# Request Models
class PricingRecommendationRequest(BaseModel):
    customer_id: UUID = Field(..., description="Customer ID for pricing recommendation")
    strategy: PricingStrategyAPI = Field(default=PricingStrategyAPI.DYNAMIC, description="Pricing strategy to use")

class BatchPricingRequest(BaseModel):
    customer_ids: List[UUID] = Field(..., description="List of customer IDs")
    strategy: PricingStrategyAPI = Field(default=PricingStrategyAPI.DYNAMIC, description="Pricing strategy to use")

class TierOptimizationRequest(BaseModel):
    target_metrics: Dict[str, float] = Field(..., description="Target metrics for optimization")
    tiers_to_optimize: Optional[List[PricingTierAPI]] = Field(None, description="Specific tiers to optimize")

class PricingExperimentRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    strategy: PricingStrategyAPI = Field(..., description="Pricing strategy to test")
    target_segment: str = Field(..., description="Target customer segment")
    test_price: float = Field(..., ge=0, description="Test price")
    control_price: float = Field(..., ge=0, description="Control price")
    duration_days: int = Field(..., ge=7, le=90, description="Experiment duration in days")
    sample_size: int = Field(..., ge=10, le=10000, description="Sample size")
    success_metrics: List[str] = Field(..., description="Success metrics to track")

class MarketConditionsUpdate(BaseModel):
    competitive_pressure: float = Field(..., ge=0, le=1, description="Competitive pressure (0-1)")
    market_growth_rate: float = Field(..., ge=-1, le=1, description="Market growth rate")
    customer_acquisition_cost: float = Field(..., ge=0, description="Customer acquisition cost")
    average_deal_size: float = Field(..., ge=0, description="Average deal size")
    price_elasticity: float = Field(..., ge=-5, le=0, description="Price elasticity")

class CompetitiveIntelligenceRequest(BaseModel):
    competitor_name: str = Field(..., description="Competitor name")
    product_tier: str = Field(..., description="Product tier")
    price: float = Field(..., ge=0, description="Competitor price")
    features_included: List[str] = Field(default_factory=list, description="Features included")
    pricing_model: str = Field(default="subscription", description="Pricing model")
    contract_terms: Optional[str] = Field(None, description="Contract terms")
    market_position: str = Field(default="mid-market", description="Market position")
    confidence_score: float = Field(default=0.8, ge=0, le=1, description="Data confidence")

# Response Models
class PricingRecommendationResponse(BaseModel):
    recommendation_id: UUID
    customer_id: UUID
    tenant_id: UUID
    current_price: float
    recommended_price: float
    price_change_percent: float
    adjustment_type: PriceAdjustmentTypeAPI
    strategy_used: PricingStrategyAPI
    confidence_score: float
    expected_revenue_impact: float
    expected_churn_impact: float
    implementation_priority: float
    reasoning: List[str]
    supporting_metrics: Dict[str, float]
    effective_date: datetime
    expiry_date: Optional[datetime]

class CustomerPricingProfileResponse(BaseModel):
    customer_id: UUID
    tenant_id: UUID
    current_tier: PricingTierAPI
    monthly_usage: float
    value_score: float
    price_sensitivity: float
    churn_risk: float
    clv_prediction: float
    competitive_position: float
    usage_growth_trend: float
    feature_adoption_score: float
    support_cost_ratio: float
    payment_reliability: float
    contract_length_preference: int
    last_updated: datetime

class TierOptimizationResponse(BaseModel):
    current_performance: Dict[str, Any]
    optimized_tiers: Dict[str, Any]
    expected_impact: Dict[str, float]
    implementation_plan: List[Dict[str, Any]]

class PricingExperimentResponse(BaseModel):
    experiment_id: UUID
    name: str
    description: Optional[str]
    strategy: PricingStrategyAPI
    target_segment: str
    test_price: float
    control_price: float
    start_date: datetime
    end_date: datetime
    sample_size: int
    test_group_size: int
    control_group_size: int
    success_metrics: List[str]
    status: str
    results: Optional[Dict[str, Any]]
    statistical_significance: Optional[float]
    recommendation: Optional[str]

class PricingDashboardResponse(BaseModel):
    customers_with_recommendations: int
    total_recommendations: int
    pending_recommendations: int
    implemented_recommendations: int
    avg_revenue_opportunity: float
    total_revenue_opportunity: float
    total_realized_revenue: float
    avg_priority_score: float
    high_priority_recommendations: int

class PricingAlertResponse(BaseModel):
    alert_id: UUID
    alert_type: str
    severity: str
    title: str
    description: str
    affected_customers: int
    revenue_impact: float
    recommended_actions: List[str]
    status: str
    created_at: datetime

class CompetitiveIntelligenceResponse(BaseModel):
    intelligence_id: UUID
    competitor_name: str
    product_tier: str
    price: float
    features_included: List[str]
    pricing_model: str
    contract_terms: Optional[str]
    market_position: str
    confidence_score: float
    last_updated: datetime

class PriceElasticityResponse(BaseModel):
    customer_segment: Optional[str]
    pricing_tier: Optional[PricingTierAPI]
    elasticity_coefficient: float
    confidence_interval: Dict[str, float]
    r_squared: float
    data_points_count: int
    calculation_date: date

# Initialize pricing engine
pricing_engine = DynamicPricingEngine()

# Endpoints

@router.post("/recommendations", response_model=PricingRecommendationResponse)
async def generate_pricing_recommendation(
    request: PricingRecommendationRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Generate pricing recommendation for a customer"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Generate recommendation
        recommendation = await pricing_engine.generate_pricing_recommendation(
            tenant_id=tenant_id,
            customer_id=request.customer_id,
            strategy=PricingStrategy(request.strategy.value)
        )

        return PricingRecommendationResponse(
            recommendation_id=uuid4(),  # Generated by storage
            customer_id=recommendation.customer_id,
            tenant_id=recommendation.tenant_id,
            current_price=recommendation.current_price,
            recommended_price=recommendation.recommended_price,
            price_change_percent=recommendation.price_change_percent,
            adjustment_type=PriceAdjustmentTypeAPI(recommendation.adjustment_type.value),
            strategy_used=PricingStrategyAPI(recommendation.strategy_used.value),
            confidence_score=recommendation.confidence_score,
            expected_revenue_impact=recommendation.expected_revenue_impact,
            expected_churn_impact=recommendation.expected_churn_impact,
            implementation_priority=recommendation.implementation_priority,
            reasoning=recommendation.reasoning,
            supporting_metrics=recommendation.supporting_metrics,
            effective_date=recommendation.effective_date,
            expiry_date=recommendation.expiry_date
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate pricing recommendation: {str(e)}")

@router.post("/recommendations/batch", response_model=List[PricingRecommendationResponse])
async def generate_batch_pricing_recommendations(
    request: BatchPricingRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Generate pricing recommendations for multiple customers"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate batch size
        if len(request.customer_ids) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 customers per batch")

        # Generate batch recommendations
        recommendations = await pricing_engine.generate_batch_recommendations(
            tenant_id=tenant_id,
            customer_ids=request.customer_ids,
            strategy=PricingStrategy(request.strategy.value)
        )

        return [
            PricingRecommendationResponse(
                recommendation_id=uuid4(),
                customer_id=rec.customer_id,
                tenant_id=rec.tenant_id,
                current_price=rec.current_price,
                recommended_price=rec.recommended_price,
                price_change_percent=rec.price_change_percent,
                adjustment_type=PriceAdjustmentTypeAPI(rec.adjustment_type.value),
                strategy_used=PricingStrategyAPI(rec.strategy_used.value),
                confidence_score=rec.confidence_score,
                expected_revenue_impact=rec.expected_revenue_impact,
                expected_churn_impact=rec.expected_churn_impact,
                implementation_priority=rec.implementation_priority,
                reasoning=rec.reasoning,
                supporting_metrics=rec.supporting_metrics,
                effective_date=rec.effective_date,
                expiry_date=rec.expiry_date
            )
            for rec in recommendations
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate batch recommendations: {str(e)}")

@router.get("/recommendations", response_model=List[PricingRecommendationResponse])
async def get_pricing_recommendations(
    customer_id: Optional[UUID] = None,
    strategy: Optional[PricingStrategyAPI] = None,
    status: Optional[str] = None,
    min_priority: Optional[float] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get pricing recommendations with filtering"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1

        if customer_id:
            param_count += 1
            conditions.append(f"customer_id = ${param_count}")
            params.append(customer_id)

        if strategy:
            param_count += 1
            conditions.append(f"strategy_used = ${param_count}")
            params.append(strategy.value)

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status)

        if min_priority:
            param_count += 1
            conditions.append(f"implementation_priority >= ${param_count}")
            params.append(min_priority)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM pricing.recommendations
                WHERE {' AND '.join(conditions)}
                ORDER BY implementation_priority DESC, created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                PricingRecommendationResponse(
                    recommendation_id=row['recommendation_id'],
                    customer_id=row['customer_id'],
                    tenant_id=row['tenant_id'],
                    current_price=float(row['current_price']),
                    recommended_price=float(row['recommended_price']),
                    price_change_percent=float(row['price_change_percent']),
                    adjustment_type=PriceAdjustmentTypeAPI(row['adjustment_type']),
                    strategy_used=PricingStrategyAPI(row['strategy_used']),
                    confidence_score=float(row['confidence_score']),
                    expected_revenue_impact=float(row['expected_revenue_impact']),
                    expected_churn_impact=float(row['expected_churn_impact']),
                    implementation_priority=float(row['implementation_priority']),
                    reasoning=json.loads(row['reasoning']) if row['reasoning'] else [],
                    supporting_metrics=json.loads(row['supporting_metrics']) if row['supporting_metrics'] else {},
                    effective_date=row['effective_date'],
                    expiry_date=row['expiry_date']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pricing recommendations: {str(e)}")

@router.put("/recommendations/{recommendation_id}/implement")
async def implement_pricing_recommendation(
    recommendation_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Implement a pricing recommendation"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            # Update recommendation status
            query = """
                UPDATE pricing.recommendations
                SET status = 'implemented',
                    implemented_date = NOW(),
                    updated_at = NOW()
                WHERE recommendation_id = $1 AND tenant_id = $2
            """
            result = await conn.execute(query, recommendation_id, tenant_id)

            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Recommendation not found")

            # TODO: Integrate with billing system to actually update prices

            return {"message": "Pricing recommendation implemented successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to implement recommendation: {str(e)}")

@router.get("/customers/{customer_id}/profile", response_model=CustomerPricingProfileResponse)
async def get_customer_pricing_profile(
    customer_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get customer pricing profile"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM pricing.customer_profiles
                WHERE tenant_id = $1 AND customer_id = $2
            """
            result = await conn.fetchrow(query, tenant_id, customer_id)

            if not result:
                # Update profile if not exists
                await conn.execute(
                    "SELECT pricing.update_customer_profile($1, $2)",
                    tenant_id, customer_id
                )
                result = await conn.fetchrow(query, tenant_id, customer_id)

            if not result:
                raise HTTPException(status_code=404, detail="Customer profile not found")

            return CustomerPricingProfileResponse(
                customer_id=result['customer_id'],
                tenant_id=result['tenant_id'],
                current_tier=PricingTierAPI(result['current_tier']),
                monthly_usage=float(result['monthly_usage']),
                value_score=float(result['value_score']),
                price_sensitivity=float(result['price_sensitivity']),
                churn_risk=float(result['churn_risk']),
                clv_prediction=float(result['clv_prediction']),
                competitive_position=float(result['competitive_position']),
                usage_growth_trend=float(result['usage_growth_trend']),
                feature_adoption_score=float(result['feature_adoption_score']),
                support_cost_ratio=float(result['support_cost_ratio']),
                payment_reliability=float(result['payment_reliability']),
                contract_length_preference=result['contract_length_preference'],
                last_updated=result['last_updated']
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get customer profile: {str(e)}")

@router.post("/tiers/optimize", response_model=TierOptimizationResponse)
async def optimize_pricing_tiers(
    request: TierOptimizationRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Optimize pricing tiers based on target metrics"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Run optimization
        optimization_result = await pricing_engine.optimize_pricing_tiers(
            tenant_id=tenant_id,
            target_metrics=request.target_metrics
        )

        return TierOptimizationResponse(**optimization_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize pricing tiers: {str(e)}")

@router.post("/experiments", response_model=PricingExperimentResponse)
async def create_pricing_experiment(
    request: PricingExperimentRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create and start a pricing experiment"""
    try:
        # Verify token
        await verify_token(token.credentials)

        experiment_id = uuid4()
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=request.duration_days)

        async with get_db_connection() as conn:
            query = """
                INSERT INTO pricing.experiments (
                    experiment_id, tenant_id, name, description, strategy,
                    target_segment, test_price, control_price, start_date, end_date,
                    sample_size, success_metrics, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING *
            """
            result = await conn.fetchrow(
                query, experiment_id, tenant_id, request.name, request.description,
                request.strategy.value, request.target_segment, request.test_price,
                request.control_price, start_date, end_date, request.sample_size,
                json.dumps(request.success_metrics), 'running'
            )

            # TODO: Start experiment execution in background

            return PricingExperimentResponse(
                experiment_id=result['experiment_id'],
                name=result['name'],
                description=result['description'],
                strategy=PricingStrategyAPI(result['strategy']),
                target_segment=result['target_segment'],
                test_price=float(result['test_price']),
                control_price=float(result['control_price']),
                start_date=result['start_date'],
                end_date=result['end_date'],
                sample_size=result['sample_size'],
                test_group_size=result['test_group_size'],
                control_group_size=result['control_group_size'],
                success_metrics=json.loads(result['success_metrics']),
                status=result['status'],
                results=json.loads(result['results']) if result['results'] else None,
                statistical_significance=float(result['statistical_significance']) if result['statistical_significance'] else None,
                recommendation=result['recommendation']
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create pricing experiment: {str(e)}")

@router.get("/experiments", response_model=List[PricingExperimentResponse])
async def get_pricing_experiments(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get pricing experiments"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]

        if status:
            conditions.append("status = $2")
            params.append(status)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM pricing.experiments
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                PricingExperimentResponse(
                    experiment_id=row['experiment_id'],
                    name=row['name'],
                    description=row['description'],
                    strategy=PricingStrategyAPI(row['strategy']),
                    target_segment=row['target_segment'],
                    test_price=float(row['test_price']),
                    control_price=float(row['control_price']),
                    start_date=row['start_date'],
                    end_date=row['end_date'],
                    sample_size=row['sample_size'],
                    test_group_size=row['test_group_size'],
                    control_group_size=row['control_group_size'],
                    success_metrics=json.loads(row['success_metrics']),
                    status=row['status'],
                    results=json.loads(row['results']) if row['results'] else None,
                    statistical_significance=float(row['statistical_significance']) if row['statistical_significance'] else None,
                    recommendation=row['recommendation']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pricing experiments: {str(e)}")

@router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get pricing experiment results"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Analyze experiment results
        results = await pricing_engine.analyze_experiment_results(
            tenant_id=tenant_id,
            experiment_id=experiment_id
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment results: {str(e)}")

@router.get("/dashboard", response_model=PricingDashboardResponse)
async def get_pricing_dashboard(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get pricing dashboard data"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM pricing.pricing_dashboard_stats
                WHERE tenant_id = $1
            """
            result = await conn.fetchrow(query, tenant_id)

            if not result:
                # Return empty dashboard
                return PricingDashboardResponse(
                    customers_with_recommendations=0,
                    total_recommendations=0,
                    pending_recommendations=0,
                    implemented_recommendations=0,
                    avg_revenue_opportunity=0,
                    total_revenue_opportunity=0,
                    total_realized_revenue=0,
                    avg_priority_score=0,
                    high_priority_recommendations=0
                )

            return PricingDashboardResponse(
                customers_with_recommendations=result['customers_with_recommendations'],
                total_recommendations=result['total_recommendations'],
                pending_recommendations=result['pending_recommendations'],
                implemented_recommendations=result['implemented_recommendations'],
                avg_revenue_opportunity=float(result['avg_revenue_opportunity'] or 0),
                total_revenue_opportunity=float(result['total_revenue_opportunity'] or 0),
                total_realized_revenue=float(result['total_realized_revenue'] or 0),
                avg_priority_score=float(result['avg_priority_score'] or 0),
                high_priority_recommendations=result['high_priority_recommendations']
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pricing dashboard: {str(e)}")

@router.get("/analytics/performance")
async def get_pricing_performance(
    days_back: int = Query(default=30, ge=1, le=365),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Monitor pricing performance"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get performance data
        performance = await pricing_engine.monitor_pricing_performance(
            tenant_id=tenant_id,
            days_back=days_back
        )

        return performance

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pricing performance: {str(e)}")

@router.get("/analytics/elasticity", response_model=List[PriceElasticityResponse])
async def get_price_elasticity(
    customer_segment: Optional[str] = None,
    pricing_tier: Optional[PricingTierAPI] = None,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get price elasticity analysis"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            # Build query
            conditions = ["tenant_id = $1"]
            params = [tenant_id]

            if customer_segment:
                conditions.append("customer_segment = $2")
                params.append(customer_segment)

            if pricing_tier:
                conditions.append("pricing_tier = $3")
                params.append(pricing_tier.value)

            query = f"""
                SELECT * FROM pricing.price_elasticity
                WHERE {' AND '.join(conditions)}
                ORDER BY calculation_date DESC
                LIMIT 50
            """

            results = await conn.fetch(query, *params)

            return [
                PriceElasticityResponse(
                    customer_segment=row['customer_segment'],
                    pricing_tier=PricingTierAPI(row['pricing_tier']) if row['pricing_tier'] else None,
                    elasticity_coefficient=float(row['elasticity_coefficient']),
                    confidence_interval=json.loads(row['confidence_interval']) if row['confidence_interval'] else {},
                    r_squared=float(row['r_squared'] or 0),
                    data_points_count=row['data_points_count'],
                    calculation_date=row['calculation_date']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get price elasticity: {str(e)}")

@router.put("/market-conditions", response_model=Dict[str, Any])
async def update_market_conditions(
    request: MarketConditionsUpdate,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Update market conditions"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                INSERT INTO pricing.market_conditions (
                    tenant_id, competitive_pressure, market_growth_rate,
                    customer_acquisition_cost, average_deal_size, price_elasticity,
                    data_date
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (tenant_id, data_date)
                DO UPDATE SET
                    competitive_pressure = EXCLUDED.competitive_pressure,
                    market_growth_rate = EXCLUDED.market_growth_rate,
                    customer_acquisition_cost = EXCLUDED.customer_acquisition_cost,
                    average_deal_size = EXCLUDED.average_deal_size,
                    price_elasticity = EXCLUDED.price_elasticity
                RETURNING *
            """
            result = await conn.fetchrow(
                query, tenant_id, request.competitive_pressure, request.market_growth_rate,
                request.customer_acquisition_cost, request.average_deal_size,
                request.price_elasticity, date.today()
            )

            return {"message": "Market conditions updated successfully", "data": dict(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update market conditions: {str(e)}")

@router.post("/competitive-intelligence", response_model=CompetitiveIntelligenceResponse)
async def add_competitive_intelligence(
    request: CompetitiveIntelligenceRequest,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Add competitive intelligence data"""
    try:
        # Verify token
        await verify_token(token.credentials)

        intelligence_id = uuid4()

        async with get_db_connection() as conn:
            query = """
                INSERT INTO pricing.competitive_intelligence (
                    intelligence_id, tenant_id, competitor_name, product_tier,
                    price, features_included, pricing_model, contract_terms,
                    market_position, confidence_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING *
            """
            result = await conn.fetchrow(
                query, intelligence_id, tenant_id, request.competitor_name,
                request.product_tier, request.price, json.dumps(request.features_included),
                request.pricing_model, request.contract_terms, request.market_position,
                request.confidence_score
            )

            return CompetitiveIntelligenceResponse(
                intelligence_id=result['intelligence_id'],
                competitor_name=result['competitor_name'],
                product_tier=result['product_tier'],
                price=float(result['price']),
                features_included=json.loads(result['features_included']),
                pricing_model=result['pricing_model'],
                contract_terms=result['contract_terms'],
                market_position=result['market_position'],
                confidence_score=float(result['confidence_score']),
                last_updated=result['last_updated']
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add competitive intelligence: {str(e)}")

@router.get("/competitive-intelligence", response_model=List[CompetitiveIntelligenceResponse])
async def get_competitive_intelligence(
    competitor_name: Optional[str] = None,
    product_tier: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get competitive intelligence data"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]

        if competitor_name:
            conditions.append("competitor_name ILIKE $2")
            params.append(f"%{competitor_name}%")

        if product_tier:
            conditions.append("product_tier = $3")
            params.append(product_tier)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM pricing.competitive_intelligence
                WHERE {' AND '.join(conditions)}
                ORDER BY last_updated DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                CompetitiveIntelligenceResponse(
                    intelligence_id=row['intelligence_id'],
                    competitor_name=row['competitor_name'],
                    product_tier=row['product_tier'],
                    price=float(row['price']),
                    features_included=json.loads(row['features_included']),
                    pricing_model=row['pricing_model'],
                    contract_terms=row['contract_terms'],
                    market_position=row['market_position'],
                    confidence_score=float(row['confidence_score']),
                    last_updated=row['last_updated']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get competitive intelligence: {str(e)}")

@router.get("/alerts", response_model=List[PricingAlertResponse])
async def get_pricing_alerts(
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get pricing alerts"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]

        if alert_type:
            conditions.append("alert_type = $2")
            params.append(alert_type)

        if severity:
            conditions.append("severity = $3")
            params.append(severity)

        if status:
            conditions.append("status = $4")
            params.append(status)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM pricing.alerts
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                PricingAlertResponse(
                    alert_id=row['alert_id'],
                    alert_type=row['alert_type'],
                    severity=row['severity'],
                    title=row['title'],
                    description=row['description'],
                    affected_customers=row['affected_customers'],
                    revenue_impact=float(row['revenue_impact']),
                    recommended_actions=json.loads(row['recommended_actions']) if row['recommended_actions'] else [],
                    status=row['status'],
                    created_at=row['created_at']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pricing alerts: {str(e)}")

@router.put("/alerts/{alert_id}/acknowledge")
async def acknowledge_pricing_alert(
    alert_id: UUID,
    acknowledged_by: str = Query(..., description="Name of person acknowledging alert"),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Acknowledge a pricing alert"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                UPDATE pricing.alerts
                SET status = 'acknowledged',
                    acknowledged_by = $3,
                    acknowledged_at = NOW()
                WHERE alert_id = $1 AND tenant_id = $2
            """
            result = await conn.execute(query, alert_id, tenant_id, acknowledged_by)

            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Alert not found")

            return {"message": "Alert acknowledged successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.post("/refresh-dashboard")
async def refresh_pricing_dashboard(
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Refresh pricing dashboard materialized view"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            await conn.execute("SELECT pricing.refresh_pricing_dashboard_stats()")

        return {"message": "Pricing dashboard refreshed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh dashboard: {str(e)}")

@router.post("/calculate-elasticity")
async def calculate_price_elasticity(
    customer_segment: Optional[str] = None,
    pricing_tier: Optional[PricingTierAPI] = None,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Calculate price elasticity for segment or tier"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = "SELECT * FROM pricing.calculate_price_elasticity($1, $2, $3)"
            result = await conn.fetchrow(
                query, tenant_id, customer_segment,
                pricing_tier.value if pricing_tier else None
            )

            # Store the calculated elasticity
            insert_query = """
                INSERT INTO pricing.price_elasticity (
                    tenant_id, customer_segment, pricing_tier, price_point,
                    demand_quantity, elasticity_coefficient, confidence_interval,
                    calculation_date, data_points_count, r_squared
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """
            await conn.execute(
                insert_query, tenant_id, customer_segment,
                pricing_tier.value if pricing_tier else None, 100.0, 1000,
                result['elasticity_coefficient'], result['confidence_interval'],
                date.today(), result['data_points'], result['r_squared']
            )

            return {
                "elasticity_coefficient": float(result['elasticity_coefficient']),
                "confidence_interval": result['confidence_interval'],
                "r_squared": float(result['r_squared']),
                "data_points": result['data_points']
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate price elasticity: {str(e)}")

# Health check endpoint
@router.get("/health")
async def pricing_health_check():
    """Health check for pricing system"""
    try:
        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "service": "dynamic_pricing",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
