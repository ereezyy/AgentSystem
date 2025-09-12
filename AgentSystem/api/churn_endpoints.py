
"""
Churn Prediction and Intervention API Endpoints - AgentSystem Profit Machine
Advanced churn prediction and automated intervention system endpoints
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
from ..analytics.churn_predictor import (
    ChurnPredictor, ChurnRiskLevel, InterventionType, InterventionStatus,
    ChurnModel, ChurnPrediction, InterventionPlan
)

# Initialize router
router = APIRouter(prefix="/api/v1/churn", tags=["Churn Prediction & Intervention"])
security = HTTPBearer()

# Enums
class ChurnRiskLevelAPI(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class InterventionTypeAPI(str, Enum):
    EMAIL_OUTREACH = "email_outreach"
    PHONE_CALL = "phone_call"
    DISCOUNT_OFFER = "discount_offer"
    FEATURE_TRAINING = "feature_training"
    ACCOUNT_REVIEW = "account_review"
    PRODUCT_DEMO = "product_demo"
    CUSTOMER_SUCCESS_CALL = "customer_success_call"
    RETENTION_CAMPAIGN = "retention_campaign"

class InterventionStatusAPI(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ChurnModelAPI(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"

# Request Models
class ChurnPredictionRequest(BaseModel):
    customer_id: UUID = Field(..., description="Customer ID to predict churn for")
    model_type: ChurnModelAPI = Field(default=ChurnModelAPI.ENSEMBLE, description="Churn model type to use")

class BatchChurnPredictionRequest(BaseModel):
    customer_ids: List[UUID] = Field(..., description="List of customer IDs")
    model_type: ChurnModelAPI = Field(default=ChurnModelAPI.ENSEMBLE, description="Churn model type to use")

class InterventionPlanRequest(BaseModel):
    customer_id: UUID = Field(..., description="Customer ID for intervention plan")
    custom_interventions: Optional[List[InterventionTypeAPI]] = Field(None, description="Custom intervention types")
    priority_override: Optional[float] = Field(None, ge=0, le=100, description="Priority score override")
    target_completion_days: int = Field(default=30, ge=1, le=90, description="Target completion timeframe")

class InterventionExecutionRequest(BaseModel):
    intervention_id: str = Field(..., description="Intervention ID to execute")
    assigned_agent: Optional[str] = Field(None, description="Agent assigned to execute intervention")
    scheduled_date: Optional[datetime] = Field(None, description="Scheduled execution date")
    notes: Optional[str] = Field(None, description="Execution notes")

class ChurnAlertRequest(BaseModel):
    customer_id: UUID = Field(..., description="Customer ID for alert")
    alert_type: str = Field(..., description="Type of alert")
    message: str = Field(..., description="Alert message")
    priority_score: float = Field(..., ge=0, le=100, description="Alert priority score")

class RetentionCampaignRequest(BaseModel):
    campaign_name: str = Field(..., description="Campaign name")
    campaign_type: str = Field(..., description="Campaign type")
    target_risk_levels: List[ChurnRiskLevelAPI] = Field(..., description="Target risk levels")
    campaign_config: Dict[str, Any] = Field(..., description="Campaign configuration")
    start_date: datetime = Field(..., description="Campaign start date")
    end_date: Optional[datetime] = Field(None, description="Campaign end date")

# Response Models
class ChurnPredictionResponse(BaseModel):
    prediction_id: UUID
    customer_id: UUID
    tenant_id: UUID
    churn_probability: float
    risk_level: ChurnRiskLevelAPI
    confidence_score: float
    time_to_churn_days: Optional[int]
    key_risk_factors: List[str]
    protective_factors: List[str]
    recommended_interventions: List[InterventionTypeAPI]
    early_warning_signals: List[str]
    feature_importance: Dict[str, float]
    model_used: ChurnModelAPI
    prediction_date: datetime

class InterventionPlanResponse(BaseModel):
    plan_id: UUID
    customer_id: UUID
    tenant_id: UUID
    churn_probability: float
    risk_level: ChurnRiskLevelAPI
    interventions: List[Dict[str, Any]]
    priority_score: float
    estimated_success_rate: float
    estimated_cost: float
    estimated_clv_impact: float
    created_date: datetime
    target_completion_date: datetime
    assigned_agent: Optional[str]
    status: str

class InterventionExecutionResponse(BaseModel):
    execution_id: UUID
    plan_id: UUID
    intervention_id: str
    intervention_type: InterventionTypeAPI
    status: InterventionStatusAPI
    scheduled_date: Optional[datetime]
    execution_date: Optional[datetime]
    completion_date: Optional[datetime]
    assigned_agent: Optional[str]
    outcome: Dict[str, Any]
    engagement_score: Optional[float]
    cost: float
    duration_minutes: Optional[int]

class ChurnAlertResponse(BaseModel):
    alert_id: UUID
    customer_id: UUID
    tenant_id: UUID
    alert_type: str
    risk_level: ChurnRiskLevelAPI
    churn_probability: float
    trigger_factors: List[str]
    alert_message: str
    priority_score: float
    status: str
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    created_at: datetime

class ChurnDashboardResponse(BaseModel):
    total_at_risk_customers: int
    high_risk_customers: int
    urgent_customers: int
    avg_churn_probability: float
    active_intervention_plans: int
    pending_interventions: int
    prevented_churns_last_30d: int
    total_clv_protected: float
    avg_intervention_roi: float

class InterventionEffectivenessResponse(BaseModel):
    intervention_type: InterventionTypeAPI
    total_interventions: int
    completed_interventions: int
    avg_engagement_score: float
    avg_cost: float
    successful_preventions: int
    avg_roi: float
    avg_clv_impact: float

class ChurnRiskSummaryResponse(BaseModel):
    risk_level: ChurnRiskLevelAPI
    customer_count: int
    avg_churn_probability: float
    avg_confidence: float
    urgent_cases: int
    with_intervention_plans: int
    avg_estimated_clv_impact: float

class RetentionCampaignResponse(BaseModel):
    campaign_id: UUID
    campaign_name: str
    campaign_type: str
    target_risk_levels: List[ChurnRiskLevelAPI]
    start_date: datetime
    end_date: Optional[datetime]
    is_active: bool
    target_customer_count: int
    enrolled_customer_count: int
    completed_customer_count: int
    success_rate: float
    total_cost: float
    total_clv_impact: float
    roi: float

# Initialize churn predictor
churn_predictor = ChurnPredictor()

# Endpoints

@router.post("/predict", response_model=ChurnPredictionResponse)
async def predict_customer_churn(
    request: ChurnPredictionRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Predict churn probability for a single customer"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Make churn prediction
        prediction = await churn_predictor.predict_churn(
            tenant_id=tenant_id,
            customer_id=request.customer_id,
            model_type=ChurnModel(request.model_type.value)
        )

        # Create alert if high risk
        if prediction.risk_level in [ChurnRiskLevel.HIGH, ChurnRiskLevel.VERY_HIGH, ChurnRiskLevel.CRITICAL]:
            background_tasks.add_task(
                create_churn_alert,
                tenant_id,
                request.customer_id,
                prediction.churn_probability,
                prediction.risk_level,
                prediction.key_risk_factors
            )

        return ChurnPredictionResponse(
            prediction_id=uuid4(),  # Generated by storage
            customer_id=prediction.customer_id,
            tenant_id=prediction.tenant_id,
            churn_probability=prediction.churn_probability,
            risk_level=ChurnRiskLevelAPI(prediction.risk_level.value),
            confidence_score=prediction.confidence_score,
            time_to_churn_days=prediction.time_to_churn_days,
            key_risk_factors=prediction.key_risk_factors,
            protective_factors=prediction.protective_factors,
            recommended_interventions=[InterventionTypeAPI(i.value) for i in prediction.recommended_interventions],
            early_warning_signals=prediction.early_warning_signals,
            feature_importance=prediction.feature_importance,
            model_used=ChurnModelAPI(prediction.model_used.value),
            prediction_date=prediction.prediction_date
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict churn: {str(e)}")

@router.post("/predict/batch", response_model=List[ChurnPredictionResponse])
async def predict_batch_customer_churn(
    request: BatchChurnPredictionRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Predict churn for multiple customers in batch"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate batch size
        if len(request.customer_ids) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 customers per batch")

        # Make batch predictions
        predictions = await churn_predictor.predict_batch_churn(
            tenant_id=tenant_id,
            customer_ids=request.customer_ids,
            model_type=ChurnModel(request.model_type.value)
        )

        # Create alerts for high-risk customers
        for prediction in predictions:
            if prediction.risk_level in [ChurnRiskLevel.HIGH, ChurnRiskLevel.VERY_HIGH, ChurnRiskLevel.CRITICAL]:
                background_tasks.add_task(
                    create_churn_alert,
                    tenant_id,
                    prediction.customer_id,
                    prediction.churn_probability,
                    prediction.risk_level,
                    prediction.key_risk_factors
                )

        return [
            ChurnPredictionResponse(
                prediction_id=uuid4(),
                customer_id=pred.customer_id,
                tenant_id=pred.tenant_id,
                churn_probability=pred.churn_probability,
                risk_level=ChurnRiskLevelAPI(pred.risk_level.value),
                confidence_score=pred.confidence_score,
                time_to_churn_days=pred.time_to_churn_days,
                key_risk_factors=pred.key_risk_factors,
                protective_factors=pred.protective_factors,
                recommended_interventions=[InterventionTypeAPI(i.value) for i in pred.recommended_interventions],
                early_warning_signals=pred.early_warning_signals,
                feature_importance=pred.feature_importance,
                model_used=ChurnModelAPI(pred.model_used.value),
                prediction_date=pred.prediction_date
            )
            for pred in predictions
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict batch churn: {str(e)}")

@router.get("/predictions", response_model=List[ChurnPredictionResponse])
async def get_churn_predictions(
    customer_id: Optional[UUID] = None,
    risk_level: Optional[ChurnRiskLevelAPI] = None,
    model_type: Optional[ChurnModelAPI] = None,
    days_back: int = Query(default=7, ge=1, le=90),
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get churn predictions with filtering"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1", "prediction_date >= $2"]
        params = [tenant_id, datetime.utcnow() - timedelta(days=days_back)]
        param_count = 2

        if customer_id:
            param_count += 1
            conditions.append(f"customer_id = ${param_count}")
            params.append(customer_id)

        if risk_level:
            param_count += 1
            conditions.append(f"risk_level = ${param_count}")
            params.append(risk_level.value)

        if model_type:
            param_count += 1
            conditions.append(f"model_used = ${param_count}")
            params.append(model_type.value)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM analytics.churn_predictions
                WHERE {' AND '.join(conditions)}
                ORDER BY prediction_date DESC, churn_probability DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                ChurnPredictionResponse(
                    prediction_id=row['prediction_id'],
                    customer_id=row['customer_id'],
                    tenant_id=row['tenant_id'],
                    churn_probability=float(row['churn_probability']),
                    risk_level=ChurnRiskLevelAPI(row['risk_level']),
                    confidence_score=float(row['confidence_score']),
                    time_to_churn_days=row['time_to_churn_days'],
                    key_risk_factors=json.loads(row['key_risk_factors']) if row['key_risk_factors'] else [],
                    protective_factors=json.loads(row['protective_factors']) if row['protective_factors'] else [],
                    recommended_interventions=[],  # TODO: Extract from stored data
                    early_warning_signals=json.loads(row['early_warning_signals']) if row['early_warning_signals'] else [],
                    feature_importance=json.loads(row['feature_importance']) if row['feature_importance'] else {},
                    model_used=ChurnModelAPI(row['model_used']),
                    prediction_date=row['prediction_date']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get churn predictions: {str(e)}")

@router.post("/interventions/plan", response_model=InterventionPlanResponse)
async def create_intervention_plan(
    request: InterventionPlanRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create intervention plan for at-risk customer"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get latest churn prediction
        async with get_db_connection() as conn:
            query = """
                SELECT * FROM analytics.churn_predictions
                WHERE tenant_id = $1 AND customer_id = $2
                ORDER BY prediction_date DESC LIMIT 1
            """
            result = await conn.fetchrow(query, tenant_id, request.customer_id)

            if not result:
                raise HTTPException(status_code=404, detail="No churn prediction found for customer")

        # Convert to ChurnPrediction object
        churn_prediction = ChurnPrediction(
            customer_id=result['customer_id'],
            tenant_id=result['tenant_id'],
            churn_probability=float(result['churn_probability']),
            risk_level=ChurnRiskLevel(result['risk_level']),
            confidence_score=float(result['confidence_score']),
            time_to_churn_days=result['time_to_churn_days'],
            key_risk_factors=json.loads(result['key_risk_factors']) if result['key_risk_factors'] else [],
            protective_factors=json.loads(result['protective_factors']) if result['protective_factors'] else [],
            recommended_interventions=[],  # Will be set by intervention plan
            prediction_date=result['prediction_date'],
            model_used=ChurnModel(result['model_used']),
            feature_importance=json.loads(result['feature_importance']) if result['feature_importance'] else {},
            early_warning_signals=json.loads(result['early_warning_signals']) if result['early_warning_signals'] else []
        )

        # Create intervention plan
        plan = await churn_predictor.create_intervention_plan(
            tenant_id=tenant_id,
            customer_id=request.customer_id,
            churn_prediction=churn_prediction
        )

        return InterventionPlanResponse(
            plan_id=plan.plan_id,
            customer_id=plan.customer_id,
            tenant_id=plan.tenant_id,
            churn_probability=plan.churn_probability,
            risk_level=ChurnRiskLevelAPI(plan.risk_level.value),
            interventions=plan.interventions,
            priority_score=plan.priority_score,
            estimated_success_rate=plan.estimated_success_rate,
            estimated_cost=plan.estimated_cost,
            estimated_clv_impact=plan.estimated_clv_impact,
            created_date=plan.created_date,
            target_completion_date=plan.target_completion_date,
            assigned_agent=plan.assigned_agent,
            status="active"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create intervention plan: {str(e)}")

@router.post("/interventions/{plan_id}/execute", response_model=Dict[str, Any])
async def execute_intervention(
    plan_id: UUID,
    request: InterventionExecutionRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Execute a specific intervention"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Execute intervention
        result = await churn_predictor.execute_intervention(
            tenant_id=tenant_id,
            plan_id=plan_id,
            intervention_id=request.intervention_id
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute intervention: {str(e)}")

@router.get("/interventions/plans", response_model=List[InterventionPlanResponse])
async def get_intervention_plans(
    customer_id: Optional[UUID] = None,
    risk_level: Optional[ChurnRiskLevelAPI] = None,
    status: Optional[str] = None,
    assigned_agent: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get intervention plans with filtering"""
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

        if risk_level:
            param_count += 1
            conditions.append(f"risk_level = ${param_count}")
            params.append(risk_level.value)

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status)

        if assigned_agent:
            param_count += 1
            conditions.append(f"assigned_agent = ${param_count}")
            params.append(assigned_agent)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM analytics.intervention_plans
                WHERE {' AND '.join(conditions)}
                ORDER BY priority_score DESC, created_date DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                InterventionPlanResponse(
                    plan_id=row['plan_id'],
                    customer_id=row['customer_id'],
                    tenant_id=row['tenant_id'],
                    churn_probability=float(row['churn_probability']),
                    risk_level=ChurnRiskLevelAPI(row['risk_level']),
                    interventions=json.loads(row['interventions']) if row['interventions'] else [],
                    priority_score=float(row['priority_score']),
                    estimated_success_rate=float(row['estimated_success_rate']),
                    estimated_cost=float(row['estimated_cost']),
                    estimated_clv_impact=float(row['estimated_clv_impact']),
                    created_date=row['created_date'],
                    target_completion_date=row['target_completion_date'],
                    assigned_agent=row['assigned_agent'],
                    status=row['status']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get intervention plans: {str(e)}")

@router.get("/alerts", response_model=List[ChurnAlertResponse])
async def get_churn_alerts(
    customer_id: Optional[UUID] = None,
    status: Optional[str] = None,
    risk_level: Optional[ChurnRiskLevelAPI] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get churn alerts with filtering"""
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

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status)

        if risk_level:
            param_count += 1
            conditions.append(f"risk_level = ${param_count}")
            params.append(risk_level.value)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM analytics.churn_alerts
                WHERE {' AND '.join(conditions)}
                ORDER BY priority_score DESC, created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                ChurnAlertResponse(
                    alert_id=row['alert_id'],
                    customer_id=row['customer_id'],
                    tenant_id=row['tenant_id'],
                    alert_type=row['alert_type'],
                    risk_level=ChurnRiskLevelAPI(row['risk_level']),
                    churn_probability=float(row['churn_probability']),
                    trigger_factors=json.loads(row['trigger_factors']) if row['trigger_factors'] else [],
                    alert_message=row['alert_message'],
                    priority_score=float(row['priority_score']),
                    status=row['status'],
                    acknowledged_by=row['acknowledged_by'],
                    acknowledged_at=row['acknowledged_at'],
                    created_at=row['created_at']
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get churn alerts: {str(e)}")

@router.put("/alerts/{alert_id}/acknowledge")
async def acknowledge_churn_alert(
    alert_id: UUID,
    acknowledged_by: str = Query(..., description="Name of person acknowledging alert"),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Acknowledge a churn alert"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                UPDATE analytics.churn_alerts
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

@router.get("/dashboard", response_model=ChurnDashboardResponse)
async def get_churn_dashboard(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get churn analytics dashboard data"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM analytics.churn_dashboard_stats
                WHERE tenant_id = $1
            """
            result = await conn.fetchrow(query, tenant_id)

            if not result:
                # Return empty dashboard
                return ChurnDashboardResponse(
                    total_at_risk_customers=0,
                    high_risk_customers=0,
                    urgent_customers=0,
                    avg_churn_probability=0,
                    active_intervention_plans=0,
                    pending_interventions=0,
                    prevented_churns_last_30d=0,
                    total_clv_protected=0,
                    avg_intervention_roi=0
                )

            return ChurnDashboardResponse(
                total_at_risk_customers=result['total_at_risk_customers'],
                high_risk_customers=result['high_risk_customers'],
                urgent_customers=result['urgent_customers'],
                avg_churn_probability=float(result['avg_churn_probability'] or 0),
                active_intervention_plans=result['active_intervention_plans'],
                pending_interventions=result['pending_interventions'],
                prevented_churns_last_30d=result['prevented_churns_last_30d'],
                total_clv_protected=float(result['total_clv_protected'] or 0),
                avg_intervention_roi=float(result['avg_intervention_roi'] or 0)
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get churn dashboard: {str(e)}")

@router.get("/analytics/effectiveness", response_model=List[InterventionEffectivenessResponse])
async def get_intervention_effectiveness(
    days_back: int = Query(default=90, ge=1, le=365),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get intervention effectiveness analytics"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM analytics.intervention_effectiveness
                WHERE tenant_id = $1
                ORDER BY avg_roi DESC NULLS LAST
            """
            results = await conn.fetch(query, tenant_id)

            return [
                InterventionEffectivenessResponse(
                    intervention_type=InterventionTypeAPI(row['intervention_type']),
                    total_interventions=row['total_interventions'],
                    completed_interventions=row['completed_interventions'],
                    avg_engagement_score=float(row['avg_engagement_score'] or 0),
                    avg_cost=float(row['avg_cost'] or 0),
                    successful_preventions=row['successful_preventions'],
                    avg_roi=float(row['avg_roi'] or 0),
                    avg_clv_impact=float(row['avg_clv_impact'] or 0)
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get intervention effectiveness: {str(e)}")

@router.get("/analytics/risk-summary", response_model=List[ChurnRiskSummaryResponse])
async def get_churn_risk_summary(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get churn risk level summary"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM analytics.churn_risk_summary
                WHERE tenant_id = $1
                ORDER BY
                    CASE risk_level
                        WHEN 'critical' THEN 1
                        WHEN 'very_high' THEN 2
                        WHEN 'high' THEN 3
                        WHEN 'medium' THEN 4
                        WHEN 'low' THEN 5
                        ELSE 6
                    END
            """
            results = await conn.fetch(query, tenant_id)

            return [
                ChurnRiskSummaryResponse(
                    risk_level=ChurnRiskLevelAPI(row['risk_level']),
                    customer_count=row['customer_count'],
                    avg_churn_probability=float(row['avg_churn_probability']),
                    avg_confidence=float(row['avg_confidence']),
                    urgent_cases=row['urgent_cases'],
                    with_intervention_plans=row['with_intervention_plans'],
                    avg_estimated_clv_impact=float(row['avg_estimated_clv_impact'] or 0)
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get churn risk summary: {str(e)}")

@router.post("/campaigns", response_model=RetentionCampaignResponse)
async def create_retention_campaign(
    request: RetentionCampaignRequest,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create retention campaign"""
    try:
        # Verify token
        await verify_token(token.credentials)

        campaign_id = uuid4()

        async with get_db_connection() as conn:
            query = """
                INSERT INTO analytics.retention_campaigns (
                    campaign_id, tenant_id, campaign_name, campaign_type,
                    target_risk_levels, campaign_config, start_date, end_date
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
            """
            result = await conn.fetchrow(
                query, campaign_id, tenant_id, request.campaign_name,
                request.campaign_type, [r.value for r in request.target_risk_levels],
                json.dumps(request.campaign_config), request.start_date, request.end_date
            )

            return RetentionCampaignResponse(
                campaign_id=result['campaign_id'],
                campaign_name=result['campaign_name'],
                campaign_type=result['campaign_type'],
                target_risk_levels=[ChurnRiskLevelAPI(r) for r in result['target_risk_levels']],
                start_date=result['start_date'],
                end_date=result['end_date'],
                is_active=result['is_active'],
                target_customer_count=result['target_customer_count'],
                enrolled_customer_count=result['enrolled_customer_count'],
                completed_customer_count=result['completed_customer_count'],
                success_rate=float(result['success_rate']),
                total_cost=float(result['total_cost']),
                total_clv_impact=float(result['total_clv_impact']),
                roi=float(result['roi'])
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create retention campaign: {str(e)}")

@router.get("/campaigns", response_model=List[RetentionCampaignResponse])
async def get_retention_campaigns(
    is_active: Optional[bool] = None,
    campaign_type: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get retention campaigns"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1

        if is_active is not None:
            param_count += 1
            conditions.append(f"is_active = ${param_count}")
            params.append(is_active)

        if campaign_type:
            param_count += 1
            conditions.append(f"campaign_type = ${param_count}")
            params.append(campaign_type)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM analytics.retention_campaigns
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [
                RetentionCampaignResponse(
                    campaign_id=row['campaign_id'],
                    campaign_name=row['campaign_name'],
                    campaign_type=row['campaign_type'],
                    target_risk_levels=[ChurnRiskLevelAPI(r) for r in row['target_risk_levels']],
                    start_date=row['start_date'],
                    end_date=row['end_date'],
                    is_active=row['is_active'],
                    target_customer_count=row['target_customer_count'],
                    enrolled_customer_count=row['enrolled_customer_count'],
                    completed_customer_count=row['completed_customer_count'],
                    success_rate=float(row['success_rate']),
                    total_cost=float(row['total_cost']),
                    total_clv_impact=float(row['total_clv_impact']),
                    roi=float(row['roi'])
                )
                for row in results
            ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get retention campaigns: {str(e)}")

@router.get("/analytics/monitor")
async def monitor_intervention_effectiveness(
    days_back: int = Query(default=30, ge=1, le=90),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Monitor intervention effectiveness"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Get effectiveness metrics
        effectiveness = await churn_predictor.monitor_intervention_effectiveness(
            tenant_id=tenant_id,
            days_back=days_back
        )

        return effectiveness

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to monitor effectiveness: {str(e)}")

@router.post("/refresh-dashboard")
async def refresh_churn_dashboard(
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Refresh churn dashboard materialized view"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            await conn.execute("SELECT analytics.refresh_churn_dashboard_stats()")

        return {"message": "Churn dashboard refreshed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh dashboard: {str(e)}")

@router.post("/escalate-alerts")
async def escalate_alerts(
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Manually trigger alert escalation"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            await conn.execute("SELECT analytics.auto_escalate_alerts()")

        return {"message": "Alert escalation completed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to escalate alerts: {str(e)}")

# Background task functions
async def create_churn_alert(
    tenant_id: UUID,
    customer_id: UUID,
    churn_probability: float,
    risk_level: ChurnRiskLevel,
    trigger_factors: List[str]
):
    """Background task to create churn alert"""
    try:
        async with get_db_connection() as conn:
            await conn.execute(
                "SELECT analytics.create_churn_alert($1, $2, $3, $4, $5)",
                tenant_id, customer_id, churn_probability,
                risk_level.value, json.dumps(trigger_factors)
            )
    except Exception as e:
        print(f"Failed to create churn alert: {e}")

# Health check endpoint
@router.get("/health")
async def churn_health_check():
    """Health check for churn prediction system"""
    try:
        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "service": "churn_prediction",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
