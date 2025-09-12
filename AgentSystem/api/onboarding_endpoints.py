"""
AgentSystem Onboarding API Endpoints
Automated customer onboarding and success management endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from ..database.connection import get_db
from ..onboarding.customer_success_engine import CustomerSuccessEngine, OnboardingAutomationEngine
from ..auth.dependencies import get_current_tenant, require_permissions

router = APIRouter(prefix="/api/v2/onboarding", tags=["Customer Success"])

# Initialize customer success engine (would be dependency injected in real app)
cs_engine = CustomerSuccessEngine("postgresql://user:pass@localhost/agentsystem")
automation_engine = OnboardingAutomationEngine(cs_engine)

@router.post("/start/{tenant_id}")
async def start_customer_onboarding(
    tenant_id: str,
    user_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Initialize customer onboarding journey

    Starts the automated onboarding process with personalized flow
    """
    try:
        # Verify tenant access
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Start onboarding journey
        result = await cs_engine.start_customer_onboarding(tenant_id, user_data)

        return {
            "success": True,
            "message": "Onboarding journey started successfully",
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start onboarding: {str(e)}")

@router.get("/journey/{tenant_id}")
async def get_onboarding_journey(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get current onboarding journey status and progress
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Get journey from database
        with cs_engine.Session() as session:
            query = """
                SELECT cj.*,
                       COALESCE(sc.completed_count, 0) as completed_steps_count,
                       ts.total_steps
                FROM customer_journeys cj
                LEFT JOIN (
                    SELECT tenant_id, COUNT(*) as completed_count
                    FROM step_completions
                    GROUP BY tenant_id
                ) sc ON cj.tenant_id = sc.tenant_id
                CROSS JOIN (
                    SELECT COUNT(*) as total_steps
                    FROM onboarding_steps
                    WHERE active = TRUE
                ) ts
                WHERE cj.tenant_id = :tenant_id
            """

            result = session.execute(query, {"tenant_id": tenant_id}).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Onboarding journey not found")

        # Get next steps
        next_steps = await cs_engine._get_next_onboarding_steps(tenant_id)

        # Calculate progress percentage
        progress_percentage = (result.completed_steps_count / result.total_steps) * 100 if result.total_steps > 0 else 0

        return {
            "success": True,
            "journey": {
                "tenant_id": result.tenant_id,
                "current_stage": result.current_stage,
                "health_score": float(result.health_score),
                "health_status": result.health_status,
                "engagement_score": float(result.engagement_score),
                "progress_percentage": progress_percentage,
                "completed_steps": result.completed_steps_count,
                "total_steps": result.total_steps,
                "last_activity": result.last_activity.isoformat(),
                "risk_factors": result.risk_factors,
                "success_milestones": result.success_milestones
            },
            "next_steps": next_steps
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get journey: {str(e)}")

@router.post("/step/{tenant_id}/complete")
async def complete_onboarding_step(
    tenant_id: str,
    step_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Mark an onboarding step as completed

    Updates progress and triggers next automation steps
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        step_id = step_data.get("step_id")
        completion_data = step_data.get("completion_data", {})
        time_spent = step_data.get("time_spent_minutes")
        satisfaction_score = step_data.get("satisfaction_score")

        if not step_id:
            raise HTTPException(status_code=400, detail="step_id is required")

        # Record step completion
        with cs_engine.Session() as session:
            query = """
                INSERT INTO step_completions (
                    tenant_id, step_id, completion_data, time_spent_minutes, satisfaction_score
                ) VALUES (
                    :tenant_id, :step_id, :completion_data, :time_spent, :satisfaction_score
                ) ON CONFLICT (tenant_id, step_id) DO UPDATE SET
                    completion_data = EXCLUDED.completion_data,
                    time_spent_minutes = EXCLUDED.time_spent_minutes,
                    satisfaction_score = EXCLUDED.satisfaction_score,
                    completed_at = CURRENT_TIMESTAMP
            """

            session.execute(query, {
                "tenant_id": tenant_id,
                "step_id": step_id,
                "completion_data": json.dumps(completion_data),
                "time_spent": time_spent,
                "satisfaction_score": satisfaction_score
            })
            session.commit()

        # Update onboarding progress
        result = await cs_engine.update_onboarding_progress(tenant_id, step_id, completion_data)

        return {
            "success": True,
            "message": f"Step '{step_id}' completed successfully",
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete step: {str(e)}")

@router.get("/health/{tenant_id}")
async def get_customer_health(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive customer health assessment
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        health_data = await cs_engine.calculate_customer_health(tenant_id)

        return {
            "success": True,
            "health_assessment": health_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health assessment: {str(e)}")

@router.post("/intervention/{tenant_id}")
async def trigger_customer_intervention(
    tenant_id: str,
    intervention_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Trigger a customer success intervention
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        intervention_type = intervention_data.get("intervention_type")
        context = intervention_data.get("context", {})

        if not intervention_type:
            raise HTTPException(status_code=400, detail="intervention_type is required")

        # Trigger intervention in background
        background_tasks.add_task(
            cs_engine.trigger_intervention,
            tenant_id,
            intervention_type,
            context
        )

        return {
            "success": True,
            "message": f"Intervention '{intervention_type}' triggered successfully",
            "intervention_type": intervention_type,
            "scheduled_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger intervention: {str(e)}")

@router.get("/steps")
async def get_onboarding_steps(
    stage: Optional[str] = Query(None),
    required_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Get available onboarding steps
    """
    try:
        with cs_engine.Session() as session:
            query = """
                SELECT step_id, title, description, stage, required,
                       estimated_time_minutes, help_resources, sort_order
                FROM onboarding_steps
                WHERE active = TRUE
            """

            params = {}
            if stage:
                query += " AND stage = :stage"
                params["stage"] = stage
            if required_only:
                query += " AND required = TRUE"

            query += " ORDER BY sort_order"

            results = session.execute(query, params).fetchall()

        steps = []
        for row in results:
            steps.append({
                "step_id": row.step_id,
                "title": row.title,
                "description": row.description,
                "stage": row.stage,
                "required": row.required,
                "estimated_time_minutes": row.estimated_time_minutes,
                "help_resources": row.help_resources,
                "sort_order": row.sort_order
            })

        return {
            "success": True,
            "steps": steps,
            "total_count": len(steps)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get steps: {str(e)}")

@router.get("/milestones/{tenant_id}")
async def get_customer_milestones(
    tenant_id: str,
    achieved_only: bool = Query(False),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get customer milestone achievements
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with cs_engine.Session() as session:
            if achieved_only:
                query = """
                    SELECT sm.milestone_id, sm.title, sm.description, sm.category,
                           sm.reward_type, ma.achieved_at, ma.reward_claimed
                    FROM success_milestones sm
                    JOIN milestone_achievements ma ON sm.milestone_id = ma.milestone_id
                    WHERE ma.tenant_id = :tenant_id AND sm.active = TRUE
                    ORDER BY ma.achieved_at DESC
                """
            else:
                query = """
                    SELECT sm.milestone_id, sm.title, sm.description, sm.category,
                           sm.reward_type, ma.achieved_at, ma.reward_claimed
                    FROM success_milestones sm
                    LEFT JOIN milestone_achievements ma ON sm.milestone_id = ma.milestone_id
                           AND ma.tenant_id = :tenant_id
                    WHERE sm.active = TRUE
                    ORDER BY sm.sort_order
                """

            results = session.execute(query, {"tenant_id": tenant_id}).fetchall()

        milestones = []
        for row in results:
            milestones.append({
                "milestone_id": row.milestone_id,
                "title": row.title,
                "description": row.description,
                "category": row.category,
                "reward_type": row.reward_type,
                "achieved": row.achieved_at is not None,
                "achieved_at": row.achieved_at.isoformat() if row.achieved_at else None,
                "reward_claimed": row.reward_claimed if row.reward_claimed else False
            })

        return {
            "success": True,
            "milestones": milestones,
            "total_count": len(milestones),
            "achieved_count": len([m for m in milestones if m["achieved"]])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get milestones: {str(e)}")

@router.get("/interventions/{tenant_id}")
async def get_customer_interventions(
    tenant_id: str,
    status: Optional[str] = Query(None),
    intervention_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get customer success interventions history
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with cs_engine.Session() as session:
            query = """
                SELECT intervention_type, trigger_reason, status, actions_taken,
                       scheduled_at, executed_at, completed_at, outcome,
                       effectiveness_score, created_at
                FROM success_interventions
                WHERE tenant_id = :tenant_id
            """

            params = {"tenant_id": tenant_id}

            if status:
                query += " AND status = :status"
                params["status"] = status
            if intervention_type:
                query += " AND intervention_type = :intervention_type"
                params["intervention_type"] = intervention_type

            query += " ORDER BY created_at DESC LIMIT :limit"
            params["limit"] = limit

            results = session.execute(query, params).fetchall()

        interventions = []
        for row in results:
            interventions.append({
                "intervention_type": row.intervention_type,
                "trigger_reason": row.trigger_reason,
                "status": row.status,
                "actions_taken": row.actions_taken,
                "scheduled_at": row.scheduled_at.isoformat() if row.scheduled_at else None,
                "executed_at": row.executed_at.isoformat() if row.executed_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                "outcome": row.outcome,
                "effectiveness_score": float(row.effectiveness_score) if row.effectiveness_score else None,
                "created_at": row.created_at.isoformat()
            })

        return {
            "success": True,
            "interventions": interventions,
            "total_count": len(interventions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interventions: {str(e)}")

@router.post("/feedback/{tenant_id}")
async def submit_customer_feedback(
    tenant_id: str,
    feedback_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Submit customer feedback
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        feedback_type = feedback_data.get("feedback_type", "general")
        question = feedback_data.get("question")
        response = feedback_data.get("response")
        rating = feedback_data.get("rating")
        category = feedback_data.get("category")

        # Store feedback
        with cs_engine.Session() as session:
            query = """
                INSERT INTO customer_feedback (
                    tenant_id, feedback_type, question, response, rating, category
                ) VALUES (
                    :tenant_id, :feedback_type, :question, :response, :rating, :category
                )
            """

            session.execute(query, {
                "tenant_id": tenant_id,
                "feedback_type": feedback_type,
                "question": question,
                "response": response,
                "rating": rating,
                "category": category
            })
            session.commit()

        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "feedback_type": feedback_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/templates")
async def get_onboarding_templates(
    target_segment: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    company_size: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get available onboarding templates
    """
    try:
        with cs_engine.Session() as session:
            query = """
                SELECT template_name, description, target_segment, industry,
                       company_size, use_case, steps, success_criteria
                FROM onboarding_templates
                WHERE active = TRUE
            """

            params = {}
            if target_segment:
                query += " AND target_segment = :target_segment"
                params["target_segment"] = target_segment
            if industry:
                query += " AND industry = :industry"
                params["industry"] = industry
            if company_size:
                query += " AND company_size = :company_size"
                params["company_size"] = company_size

            results = session.execute(query, params).fetchall()

        templates = []
        for row in results:
            templates.append({
                "template_name": row.template_name,
                "description": row.description,
                "target_segment": row.target_segment,
                "industry": row.industry,
                "company_size": row.company_size,
                "use_case": row.use_case,
                "steps": row.steps,
                "success_criteria": row.success_criteria
            })

        return {
            "success": True,
            "templates": templates,
            "total_count": len(templates)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.post("/health-check/run")
async def run_health_checks(
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Manually trigger customer health checks (admin only)
    """
    try:
        if not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin access required")

        # Run health checks in background
        background_tasks.add_task(automation_engine.run_daily_health_checks)

        return {
            "success": True,
            "message": "Customer health checks initiated",
            "scheduled_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run health checks: {str(e)}")

@router.get("/analytics/overview")
async def get_onboarding_analytics(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get onboarding analytics overview (admin only)
    """
    try:
        if not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin access required")

        # Calculate date range
        end_date = datetime.now()
        if period == "7d":
            start_date = end_date - timedelta(days=7)
        elif period == "30d":
            start_date = end_date - timedelta(days=30)
        elif period == "90d":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)

        with cs_engine.Session() as session:
            # Get onboarding metrics
            metrics_query = """
                SELECT
                    COUNT(*) as total_journeys,
                    AVG(health_score) as avg_health_score,
                    COUNT(CASE WHEN health_status = 'excellent' THEN 1 END) as excellent_health,
                    COUNT(CASE WHEN health_status = 'good' THEN 1 END) as good_health,
                    COUNT(CASE WHEN health_status = 'at_risk' THEN 1 END) as at_risk_health,
                    COUNT(CASE WHEN health_status = 'critical' THEN 1 END) as critical_health,
                    COUNT(CASE WHEN current_stage = 'completed' THEN 1 END) as completed_onboarding
                FROM customer_journeys
                WHERE created_at >= :start_date
            """

            metrics_result = session.execute(metrics_query, {"start_date": start_date}).fetchone()

            # Get step completion rates
            completion_query = """
                SELECT os.step_id, os.title,
                       COUNT(sc.tenant_id) as completions,
                       COUNT(cj.tenant_id) as total_customers,
                       (COUNT(sc.tenant_id)::FLOAT / COUNT(cj.tenant_id)::FLOAT * 100) as completion_rate
                FROM onboarding_steps os
                CROSS JOIN customer_journeys cj
                LEFT JOIN step_completions sc ON os.step_id = sc.step_id AND cj.tenant_id = sc.tenant_id
                WHERE os.active = TRUE AND cj.created_at >= :start_date
                GROUP BY os.step_id, os.title, os.sort_order
                ORDER BY os.sort_order
            """

            completion_results = session.execute(completion_query, {"start_date": start_date}).fetchall()

        # Build analytics response
        analytics = {
            "period": period,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "overview": {
                "total_journeys": metrics_result.total_journeys,
                "avg_health_score": float(metrics_result.avg_health_score) if metrics_result.avg_health_score else 0,
                "completed_onboarding": metrics_result.completed_onboarding,
                "completion_rate": (metrics_result.completed_onboarding / metrics_result.total_journeys * 100) if metrics_result.total_journeys > 0 else 0
            },
            "health_distribution": {
                "excellent": metrics_result.excellent_health,
                "good": metrics_result.good_health,
                "at_risk": metrics_result.at_risk_health,
                "critical": metrics_result.critical_health
            },
            "step_completion_rates": [
                {
                    "step_id": row.step_id,
                    "title": row.title,
                    "completions": row.completions,
                    "completion_rate": float(row.completion_rate) if row.completion_rate else 0
                }
                for row in completion_results
            ]
        }

        return {
            "success": True,
            "analytics": analytics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")
