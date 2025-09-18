
"""
AgentSystem Autonomous Operations API Endpoints
Revolutionary self-operating AI system with autonomous decision-making capabilities
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from ..database.connection import get_db
from ..autonomous.autonomous_operations_engine import (
    AutonomousOperationsEngine, AutonomousSelfHealing,
    AutonomyLevel, DecisionType, ActionPriority
)
from ..auth.dependencies import get_current_tenant, require_permissions

router = APIRouter(prefix="/api/v2/autonomous", tags=["Autonomous Operations"])

# Initialize autonomous operations engine (would be dependency injected in real app)
autonomous_engine = AutonomousOperationsEngine("postgresql://user:pass@localhost/agentsystem")
self_healing = AutonomousSelfHealing(autonomous_engine)

@router.post("/start/{tenant_id}")
async def start_autonomous_operations(
    tenant_id: str,
    config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Start autonomous operations for a tenant

    Enables AI-powered autonomous decision-making and execution
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Update autonomous configuration
        autonomy_level = config.get("autonomy_level", "semi_autonomous")
        confidence_threshold = config.get("confidence_threshold", 0.85)

        with autonomous_engine.Session() as session:
            query = """
                UPDATE autonomous_config
                SET autonomy_level = :autonomy_level,
                    decision_confidence_threshold = :confidence_threshold,
                    updated_at = CURRENT_TIMESTAMP
                WHERE tenant_id = :tenant_id
            """

            result = session.execute(query, {
                "tenant_id": tenant_id,
                "autonomy_level": autonomy_level,
                "confidence_threshold": confidence_threshold
            })
            session.commit()

            if result.rowcount == 0:
                # Insert new config if not exists
                session.execute("""
                    INSERT INTO autonomous_config (tenant_id, autonomy_level, decision_confidence_threshold)
                    VALUES (:tenant_id, :autonomy_level, :confidence_threshold)
                """, {
                    "tenant_id": tenant_id,
                    "autonomy_level": autonomy_level,
                    "confidence_threshold": confidence_threshold
                })
                session.commit()

        # Start autonomous operations
        background_tasks.add_task(autonomous_engine.start_autonomous_operations, tenant_id)

        return {
            "success": True,
            "message": "Autonomous operations started successfully",
            "tenant_id": tenant_id,
            "autonomy_level": autonomy_level,
            "confidence_threshold": confidence_threshold,
            "started_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start autonomous operations: {str(e)}")

@router.get("/status/{tenant_id}")
async def get_autonomous_status(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get autonomous operations status and metrics
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Get autonomous status from engine
        status = await autonomous_engine.get_autonomous_status()

        # Get database metrics
        with autonomous_engine.Session() as session:
            dashboard_data = session.execute("""
                SELECT * FROM autonomous_operations_dashboard
                WHERE tenant_id = :tenant_id
            """, {"tenant_id": tenant_id}).fetchone()

            # Get recent decisions
            recent_decisions = session.execute("""
                SELECT decision_type, priority, confidence_score, status,
                       reasoning, created_at, executed_at, completed_at
                FROM autonomous_decisions
                WHERE tenant_id = :tenant_id
                ORDER BY created_at DESC
                LIMIT 10
            """, {"tenant_id": tenant_id}).fetchall()

        dashboard_metrics = {}
        if dashboard_data:
            dashboard_metrics = {
                "autonomy_level": dashboard_data.autonomy_level,
                "total_decisions_24h": dashboard_data.total_decisions_24h,
                "successful_decisions_24h": dashboard_data.successful_decisions_24h,
                "currently_executing": dashboard_data.currently_executing,
                "pending_approvals": dashboard_data.pending_approvals,
                "avg_confidence_24h": float(dashboard_data.avg_confidence_24h) if dashboard_data.avg_confidence_24h else 0,
                "healing_incidents_24h": dashboard_data.healing_incidents_24h,
                "successful_healings_24h": dashboard_data.successful_healings_24h,
                "cost_savings_24h": float(dashboard_data.cost_savings_24h) if dashboard_data.cost_savings_24h else 0
            }

        recent_decisions_data = []
        for row in recent_decisions:
            recent_decisions_data.append({
                "decision_type": row.decision_type,
                "priority": row.priority,
                "confidence_score": float(row.confidence_score),
                "status": row.status,
                "reasoning": row.reasoning,
                "created_at": row.created_at.isoformat(),
                "executed_at": row.executed_at.isoformat() if row.executed_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None
            })

        return {
            "success": True,
            "autonomous_status": {
                **status,
                "dashboard_metrics": dashboard_metrics,
                "recent_decisions": recent_decisions_data
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get autonomous status: {str(e)}")

@router.post("/decisions/trigger/{tenant_id}")
async def trigger_autonomous_decision(
    tenant_id: str,
    trigger_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Manually trigger autonomous decision evaluation
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        trigger_type = trigger_data.get("trigger_type")
        trigger_context = trigger_data.get("context", {})

        if not trigger_type:
            raise HTTPException(status_code=400, detail="trigger_type is required")

        # Analyze current system state
        system_state = await autonomous_engine.analyze_system_state(tenant_id)

        # Make autonomous decision
        decision = await autonomous_engine.make_autonomous_decision(
            {"type": trigger_type, "context": trigger_context},
            system_state
        )

        if not decision:
            return {
                "success": True,
                "message": "No autonomous decision required at this time",
                "system_state_summary": {
                    "health_score": system_state.get("overall_health_score", 0),
                    "optimization_opportunities": len(system_state.get("optimization_opportunities", [])),
                    "decision_triggers": len(system_state.get("decision_triggers", []))
                }
            }

        # Execute if approved for autonomous execution
        execution_result = None
        if not decision.approval_required:
            execution_result = await autonomous_engine.execute_autonomous_action(decision)

        return {
            "success": True,
            "decision": {
                "decision_id": decision.decision_id,
                "decision_type": decision.decision_type.value,
                "priority": decision.priority.value,
                "confidence_score": decision.confidence_score,
                "reasoning": decision.reasoning,
                "proposed_actions": decision.proposed_actions,
                "approval_required": decision.approval_required,
                "execution_deadline": decision.execution_deadline.isoformat() if decision.execution_deadline else None
            },
            "execution_result": execution_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger autonomous decision: {str(e)}")

@router.post("/decisions/{decision_id}/approve")
async def approve_autonomous_decision(
    decision_id: str,
    approval_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Approve a pending autonomous decision for execution
    """
    try:
        # Update decision approval status
        with autonomous_engine.Session() as session:
            query = """
                UPDATE autonomous_decisions
                SET status = 'approved',
                    approved_at = CURRENT_TIMESTAMP,
                    approved_by = :user_id
                WHERE decision_id = :decision_id
                AND status = 'pending'
                RETURNING *
            """

            result = session.execute(query, {
                "decision_id": decision_id,
                "user_id": current_tenant["user_id"]
            }).fetchone()
            session.commit()

        if not result:
            raise HTTPException(status_code=404, detail="Decision not found or already processed")

        # Get decision from pending list
        decision = autonomous_engine.pending_decisions.get(decision_id)
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not in pending queue")

        # Execute the approved decision
        execution_result = await autonomous_engine.execute_autonomous_action(decision, force_execution=True)

        # Remove from pending
        del autonomous_engine.pending_decisions[decision_id]

        return {
            "success": True,
            "message": "Decision approved and executed successfully",
            "decision_id": decision_id,
            "execution_result": execution_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve decision: {str(e)}")

@router.get("/decisions/{tenant_id}")
async def get_autonomous_decisions(
    tenant_id: str,
    status: Optional[str] = Query(None),
    decision_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get autonomous decisions history and status
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with autonomous_engine.Session() as session:
            query = """
                SELECT decision_id, decision_type, priority, confidence_score,
                       reasoning, proposed_actions, expected_outcomes,
                       approval_required, status, created_at, executed_at, completed_at
                FROM autonomous_decisions
                WHERE tenant_id = :tenant_id
            """

            params = {"tenant_id": tenant_id}

            if status:
                query += " AND status = :status"
                params["status"] = status
            if decision_type:
                query += " AND decision_type = :decision_type"
                params["decision_type"] = decision_type

            query += " ORDER BY created_at DESC LIMIT :limit"
            params["limit"] = limit

            results = session.execute(query, params).fetchall()

        decisions = []
        for row in results:
            decisions.append({
                "decision_id": row.decision_id,
                "decision_type": row.decision_type,
                "priority": row.priority,
                "confidence_score": float(row.confidence_score),
                "reasoning": row.reasoning,
                "proposed_actions": row.proposed_actions,
                "expected_outcomes": row.expected_outcomes,
                "approval_required": row.approval_required,
                "status": row.status,
                "created_at": row.created_at.isoformat(),
                "executed_at": row.executed_at.isoformat() if row.executed_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None
            })

        return {
            "success": True,
            "decisions": decisions,
            "total_count": len(decisions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get decisions: {str(e)}")

@router.get("/efficiency/{tenant_id}")
async def get_autonomous_efficiency(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get autonomous system efficiency metrics
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with autonomous_engine.Session() as session:
            efficiency_data = session.execute("""
                SELECT * FROM calculate_autonomous_efficiency()
                WHERE tenant_id = :tenant_id
            """, {"tenant_id": tenant_id}).fetchone()

            if not efficiency_data:
                return {
                    "success": True,
                    "message": "No autonomous operations data available yet",
                    "efficiency_metrics": {}
                }

        efficiency_metrics = {
            "total_decisions": efficiency_data.total_decisions,
            "successful_decisions": efficiency_data.successful_decisions,
            "success_rate": (efficiency_data.successful_decisions / efficiency_data.total_decisions * 100) if efficiency_data.total_decisions > 0 else 0,
            "average_confidence": float(efficiency_data.avg_confidence) if efficiency_data.avg_confidence else 0,
            "average_execution_time_minutes": efficiency_data.avg_execution_time,
            "total_cost_savings": float(efficiency_data.cost_savings_total) if efficiency_data.cost_savings_total else 0,
            "efficiency_score": float(efficiency_data.efficiency_score) if efficiency_data.efficiency_score else 0
        }

        return {
            "success": True,
            "efficiency_metrics": efficiency_metrics,
            "analysis_period": "Last 30 days",
            "calculated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get efficiency metrics: {str(e)}")

@router.post("/self-healing/start/{tenant_id}")
async def start_self_healing(
    tenant_id: str,
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Start autonomous self-healing monitoring
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Start self-healing in background
        background_tasks.add_task(self_healing.detect_and_heal_issues)

        return {
            "success": True,
            "message": "Self-healing monitoring started",
            "tenant_id": tenant_id,
            "started_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start self-healing: {str(e)}")

@router.get("/readiness/{tenant_id}")
async def evaluate_autonomy_readiness(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Evaluate tenant's readiness for higher autonomy levels
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with autonomous_engine.Session() as session:
            readiness_data = session.execute("""
                SELECT evaluate_autonomy_readiness(:tenant_id) as readiness
            """, {"tenant_id": tenant_id}).scalar()

        return {
            "success": True,
            "readiness_assessment": readiness_data,
            "current_autonomy_level": await autonomous_engine._get_current_autonomy_level(tenant_id),
            "recommended_actions": await autonomous_engine._get_readiness_recommendations(tenant_id, readiness_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate readiness: {str(e)}")

@router.post("/optimize/{tenant_id}")
async def trigger_autonomous_optimization(
    tenant_id: str,
    optimization_params: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Trigger autonomous system optimization
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        optimization_type = optimization_params.get("optimization_type", "full")
        target_metrics = optimization_params.get("target_metrics", {})

        # Analyze current system state
        system_state = await autonomous_engine.analyze_system_state(tenant_id)

        # Trigger optimization decisions
        if optimization_type == "performance":
            decision = await autonomous_engine.autonomous_scaling_decision(
                system_state["performance_metrics"]
            )
        elif optimization_type == "cost":
            decisions = await autonomous_engine.autonomous_cost_optimization(
                system_state["cost_efficiency"]
            )
            decision = decisions[0] if decisions else None
        else:
            # Full optimization - analyze all areas
            decision = await autonomous_engine.make_autonomous_decision(
                {"type": "optimization", "context": optimization_params},
                system_state
            )

        if not decision:
            return {
                "success": True,
                "message": "No optimization opportunities identified at this time",
                "system_health_score": system_state.get("overall_health_score", 0)
            }

        # Execute optimization if autonomous
        execution_result = None
        if not decision.approval_required:
            execution_result = await autonomous_engine.execute_autonomous_action(decision)

        return {
            "success": True,
            "optimization_decision": {
                "decision_id": decision.decision_id,
                "optimization_type": optimization_type,
                "confidence_score": decision.confidence_score,
                "expected_impact": decision.estimated_impact,
                "approval_required": decision.approval_required
            },
            "execution_result": execution_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger optimization: {str(e)}")

@router.get("/learning/{tenant_id}")
async def get_autonomous_learning(
    tenant_id: str,
    learning_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get autonomous learning insights and patterns
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with autonomous_engine.Session() as session:
            query = """
                SELECT learning_type, trigger_scenario, action_taken,
                       outcome_success, confidence_adjustment, pattern_recognition,
                       improvement_suggestions, applied_to_future_decisions, learned_at
                FROM autonomous_learning
                WHERE tenant_id = :tenant_id
            """

            params = {"tenant_id": tenant_id}

            if learning_type:
                query += " AND learning_type = :learning_type"
                params["learning_type"] = learning_type

            query += " ORDER BY learned_at DESC LIMIT :limit"
            params["limit"] = limit

            results = session.execute(query, params).fetchall()

        learning_data = []
        for row in results:
            learning_data.append({
                "learning_type": row.learning_type,
                "trigger_scenario": row.trigger_scenario,
                "action_taken": row.action_taken,
                "outcome_success": row.outcome_success,
                "confidence_adjustment": float(row.confidence_adjustment) if row.confidence_adjustment else 0,
                "pattern_recognition": row.pattern_recognition,
                "improvement_suggestions": row.improvement_suggestions,
                "applied_to_future": row.applied_to_future_decisions,
                "learned_at": row.learned_at.isoformat()
            })

        # Calculate learning summary
        total_learnings = len(learning_data)
        successful_outcomes = len([l for l in learning_data if l["outcome_success"]])
        success_rate = (successful_outcomes / total_learnings * 100) if total_learnings > 0 else 0

        return {
            "success": True,
            "learning_data": learning_data,
            "learning_summary": {
                "total_learnings": total_learnings,
                "successful_outcomes": successful_outcomes,
                "success_rate": success_rate,
                "applied_improvements": len([l for l in learning_data if l["applied_to_future"]])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get learning data: {str(e)}")

@router.post("/config/{tenant_id}")
async def update_autonomous_config(
    tenant_id: str,
    config_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Update autonomous operations configuration
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Validate configuration
        autonomy_level = config_data.get("autonomy_level")
        if autonomy_level and autonomy_level not in ["monitoring", "advisory", "semi_autonomous", "autonomous", "fully_autonomous"]:
            raise HTTPException(status_code=400, detail="Invalid autonomy_level")

        confidence_threshold = config_data.get("confidence_threshold", 0.85)
        if not 0.5 <= confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="confidence_threshold must be between 0.5 and 1.0")

        # Update configuration
        with autonomous_engine.Session() as session:
            update_fields = []
            params = {"tenant_id": tenant_id}

            if autonomy_level:
                update_fields.append("autonomy_level = :autonomy_level")
                params["autonomy_level"] = autonomy_level

            if "confidence_threshold" in config_data:
                update_fields.append("decision_confidence_threshold = :confidence_threshold")
                params["confidence_threshold"] = confidence_threshold

            if "auto_scaling_enabled" in config_data:
                update_fields.append("auto_scaling_enabled = :auto_scaling")
                params["auto_scaling"] = config_data["auto_scaling_enabled"]

            if "auto_optimization_enabled" in config_data:
                update_fields.append("auto_optimization_enabled = :auto_optimization")
                params["auto_optimization"] = config_data["auto_optimization_enabled"]

            if update_fields:
                query = f"""
                    UPDATE autonomous_config
                    SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                    WHERE tenant_id = :tenant_id
                """
                session.execute(query, params)
                session.commit()

        return {
            "success": True,
            "message": "Autonomous configuration updated successfully",
            "updated_config": config_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@router.get("/dashboard/{tenant_id}")
async def get_autonomous_dashboard(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Get comprehensive autonomous operations dashboard
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Get dashboard data from autonomous engine
        status = await autonomous_engine.get_autonomous_status()

        # Get additional metrics from database
        with autonomous_engine.Session() as session:
            # Recent optimizations
            recent_optimizations = session.execute("""
                SELECT optimization_type, improvement_percentage, cost_savings,
                       optimization_started_at, optimization_completed_at
                FROM autonomous_optimizations
                WHERE tenant_id = :tenant_id
                AND optimization_started_at >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY optimization_started_at DESC
                LIMIT 10
            """, {"tenant_id": tenant_id}).fetchall()

            # Self-healing incidents
            healing_incidents = session.execute("""
                SELECT incident_type, severity, healing_status,
                       time_to_detection_seconds, time_to_healing_seconds,
                       detected_at, healed_at
                FROM self_healing_incidents
                WHERE tenant_id = :tenant_id
                AND detected_at >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY detected_at DESC
                LIMIT 10
            """, {"tenant_id": tenant_id}).fetchall()

        dashboard_data = {
            "autonomous_status": status,
            "recent_optimizations": [
                {
                    "optimization_type": row.optimization_type,
                    "improvement_percentage": float(row.improvement_percentage) if row.improvement_percentage else 0,
                    "cost_savings": float(row.cost_savings) if row.cost_savings else 0,
                    "started_at": row.optimization_started_at.isoformat(),
                    "completed_at": row.optimization_completed_at.isoformat() if row.optimization_completed_at else None
                }
                for row in recent_optimizations
            ],
            "self_healing_incidents": [
                {
                    "incident_type": row.incident_type,
                    "severity": row.severity,
                    "healing_status": row.healing_status,
                    "detection_time_seconds": row.time_to_detection_seconds,
                    "healing_time_seconds": row.time_to_healing_seconds,
                    "detected_at": row.detected_at.isoformat(),
                    "healed_at": row.healed_at.isoformat() if row.healed_at else None
                }
                for row in healing_incidents
            ],
            "summary_metrics": {
                "total_optimizations_7d": len(recent_optimizations),
                "total_cost_savings_7d": sum(float(row.cost_savings) if row.cost_savings else 0 for row in recent_optimizations),
                "healing_incidents_7d": len(healing_incidents),
                "avg_healing_time": sum(row.time_to_healing_seconds or 0 for row in healing_incidents) / len(healing_incidents) if healing_incidents else 0
            },
            "dashboard_generated_at": datetime.now().isoformat()
        }

        return {
            "success": True,
            "dashboard": dashboard_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")

@router.post("/emergency-stop/{tenant_id}")
async def emergency_stop_autonomous_operations(
    tenant_id: str,
    reason: str,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Emergency stop for autonomous operations (admin only)
    """
    try:
        if not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin access required for emergency stop")

        # Stop autonomous operations
        await autonomous_engine.stop_autonomous_operations()

        # Cancel all pending executions
        with autonomous_engine.Session() as session:
            session.execute("""
                UPDATE autonomous_decisions
                SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                WHERE tenant_id = :tenant_id
                AND status IN ('pending', 'approved', 'executing')
            """, {"tenant_id": tenant_id})
            session.commit()

        # Log emergency stop
        with autonomous_engine.Session() as session:
            session.execute("""
                INSERT INTO autonomous_bi_alerts (
                    tenant_id, alert_type, severity, trigger_conditions,
                    recommended_autonomous_actions, human_override_required
                ) VALUES (
                    :tenant_id, 'emergency_stop', 'critical',
                    :reason, '[]'::jsonb, TRUE
                )
            """, {"tenant_id": tenant_id, "reason": json.dumps({"reason": reason})})
            session.commit()

        return {
            "success": True,
            "message": "Emergency stop executed successfully",
            "reason": reason,
            "stopped_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute emergency stop: {str(e)}")

@router.get("/analytics/overview")
async def get_autonomous_analytics_overview(
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Get autonomous operations analytics overview (admin only)
    """
    try:
        if not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin access required")

        # Calculate date range
        if period == "7d":
            start_date = datetime.now() - timedelta(days=7)
        elif period == "30d":
            start_date = datetime.now() - timedelta(days=30)
        elif period == "90d":
            start_date = datetime.now() - timedelta(days=90)
        else:
            start_date = datetime.now() - timedelta(days=365)

        # Get autonomous operations analytics
        with autonomous_engine.Session() as session:
            analytics_data = session.execute("""
                SELECT
                    COUNT(*) as total_decisions,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_decisions,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(DISTINCT tenant_id) as active_tenants,
                    SUM(CASE WHEN decision_type = 'scaling' THEN 1 ELSE 0 END) as scaling_decisions,
                    SUM(CASE WHEN decision_type = 'cost_management' THEN 1 ELSE 0 END) as cost_decisions,
                    SUM(CASE WHEN decision_type = 'customer_intervention' THEN 1 ELSE 0 END) as customer_decisions
                FROM autonomous_decisions
                WHERE created_at >= :start_date
            """, {"start_date": start_date}).fetchone()

            # Get cost savings
            cost_savings = session.execute("""
                SELECT COALESCE(SUM(cost_savings), 0) as total_savings
                FROM autonomous_optimizations
                WHERE optimization_started_at >= :start_date
            """, {"start_date": start_date}).scalar()

        analytics = {
            "period": period,
            "total_decisions": analytics_data.total_decisions,
            "successful_decisions": analytics_data.successful_decisions,
            "success_rate": (analytics_data.successful_decisions / analytics_data.total_decisions * 100) if analytics_data.total_decisions > 0 else 0,
            "average_confidence": float(analytics_data.avg_confidence) if analytics_data.avg_confidence else 0,
            "active_tenants": analytics_data.active_tenants,
            "decision_breakdown": {
                "scaling": analytics_data.scaling_decisions,
                "cost_management": analytics_data.cost_decisions,
                "customer_intervention": analytics_data.customer_decisions
            },
            "total_cost_savings": float(cost_savings) if cost_savings else 0,
            "analysis_period": {
                "start": start_date.isoformat(),
                "end": datetime.now().isoformat()
            }
        }

        return {
            "success": True,
            "analytics": analytics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics overview: {str(e)}")
