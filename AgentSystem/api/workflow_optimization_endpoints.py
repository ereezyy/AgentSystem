"""
AgentSystem Self-Optimizing Workflow Engine API Endpoints
Intelligent workflow optimization through machine learning and performance analysis
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from ..database.connection import get_db
from ..optimization.self_optimizing_workflow_engine import (
    SelfOptimizingWorkflowEngine, WorkflowLearningEngine,
    OptimizationType, OptimizationStrategy
)
from ..auth.dependencies import get_current_tenant, require_permissions

router = APIRouter(prefix="/api/v2/workflow-optimization", tags=["Workflow Optimization"])

# Initialize optimization engine (would be dependency injected in real app)
optimization_engine = SelfOptimizingWorkflowEngine("postgresql://user:pass@localhost/agentsystem")
learning_engine = WorkflowLearningEngine(optimization_engine)

@router.get("/analyze/{workflow_id}")
async def analyze_workflow_performance(
    workflow_id: str,
    analysis_period_days: int = Query(7, ge=1, le=90),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Analyze comprehensive workflow performance metrics

    Provides detailed performance analysis for optimization opportunities
    """
    try:
        # Verify workflow access
        workflow_owner = await optimization_engine._get_workflow_owner(workflow_id)
        if workflow_owner != current_tenant["id"] and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to workflow data")

        # Analyze performance
        performance_metrics = await optimization_engine.analyze_workflow_performance(
            workflow_id, analysis_period_days
        )

        # Get historical comparison
        baseline_comparison = await optimization_engine._get_baseline_comparison(workflow_id)

        # Get bottleneck analysis
        bottlenecks = await optimization_engine._identify_bottlenecks(workflow_id)

        return {
            "success": True,
            "workflow_id": workflow_id,
            "analysis_period_days": analysis_period_days,
            "performance_metrics": {
                "execution_time_seconds": performance_metrics.execution_time_seconds,
                "success_rate": performance_metrics.success_rate,
                "error_rate": performance_metrics.error_rate,
                "cost_per_execution": performance_metrics.cost_per_execution,
                "throughput_per_hour": performance_metrics.throughput_per_hour,
                "user_satisfaction_score": performance_metrics.user_satisfaction_score,
                "resource_utilization": performance_metrics.resource_utilization,
                "latency_percentiles": performance_metrics.latency_percentiles
            },
            "baseline_comparison": baseline_comparison,
            "bottlenecks": bottlenecks,
            "analysis_timestamp": performance_metrics.timestamp.isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze workflow performance: {str(e)}")

@router.get("/opportunities/{workflow_id}")
async def get_optimization_opportunities(
    workflow_id: str,
    optimization_type: Optional[str] = Query(None),
    min_confidence: float = Query(0.7, ge=0.5, le=1.0),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get identified optimization opportunities for a workflow
    """
    try:
        # Verify workflow access
        workflow_owner = await optimization_engine._get_workflow_owner(workflow_id)
        if workflow_owner != current_tenant["id"] and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to workflow data")

        # Get optimization opportunities
        opportunities = await optimization_engine.identify_optimization_opportunities(workflow_id)

        # Filter by criteria
        filtered_opportunities = [
            opp for opp in opportunities
            if opp.confidence_score >= min_confidence
            and (not optimization_type or opp.optimization_type.value == optimization_type)
        ]

        # Convert to response format
        opportunities_data = []
        for opp in filtered_opportunities:
            opportunities_data.append({
                "opportunity_id": opp.opportunity_id,
                "optimization_type": opp.optimization_type.value,
                "strategy": opp.strategy.value,
                "potential_improvement": opp.potential_improvement,
                "implementation_effort": opp.implementation_effort,
                "confidence_score": opp.confidence_score,
                "business_impact": opp.business_impact,
                "recommended_actions": opp.recommended_actions
            })

        return {
            "success": True,
            "workflow_id": workflow_id,
            "optimization_opportunities": opportunities_data,
            "total_opportunities": len(opportunities_data),
            "avg_confidence": sum(opp["confidence_score"] for opp in opportunities_data) / len(opportunities_data) if opportunities_data else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization opportunities: {str(e)}")

@router.post("/optimize/{workflow_id}")
async def implement_workflow_optimization(
    workflow_id: str,
    optimization_params: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Implement workflow optimization based on identified opportunities
    """
    try:
        # Verify workflow access
        workflow_owner = await optimization_engine._get_workflow_owner(workflow_id)
        if workflow_owner != current_tenant["id"] and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to workflow data")

        opportunity_id = optimization_params.get("opportunity_id")
        auto_implement = optimization_params.get("auto_implement", False)

        if not opportunity_id:
            raise HTTPException(status_code=400, detail="opportunity_id is required")

        # Get the specific opportunity
        opportunities = await optimization_engine.identify_optimization_opportunities(workflow_id)
        target_opportunity = next((opp for opp in opportunities if opp.opportunity_id == opportunity_id), None)

        if not target_opportunity:
            raise HTTPException(status_code=404, detail="Optimization opportunity not found")

        if auto_implement:
            # Implement optimization in background
            background_tasks.add_task(
                optimization_engine.implement_optimization,
                target_opportunity
            )

            return {
                "success": True,
                "message": "Optimization implementation started",
                "opportunity_id": opportunity_id,
                "estimated_completion": "5-15 minutes",
                "monitoring_enabled": True
            }
        else:
            # Return implementation plan for approval
            implementation_plan = await optimization_engine._create_implementation_plan(target_opportunity)

            return {
                "success": True,
                "opportunity": {
                    "opportunity_id": target_opportunity.opportunity_id,
                    "optimization_type": target_opportunity.optimization_type.value,
                    "strategy": target_opportunity.strategy.value,
                    "confidence_score": target_opportunity.confidence_score,
                    "business_impact": target_opportunity.business_impact
                },
                "implementation_plan": implementation_plan,
                "requires_approval": True
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to implement optimization: {str(e)}")

@router.post("/auto-optimize/{workflow_id}")
async def auto_optimize_workflow(
    workflow_id: str,
    auto_params: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Automatically optimize workflow with multiple strategies
    """
    try:
        # Verify workflow access
        workflow_owner = await optimization_engine._get_workflow_owner(workflow_id)
        if workflow_owner != current_tenant["id"] and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to workflow data")

        max_optimizations = auto_params.get("max_optimizations", 3)
        confidence_threshold = auto_params.get("confidence_threshold", 0.8)

        # Set optimization engine parameters
        optimization_engine.decision_confidence_threshold = confidence_threshold

        # Start auto-optimization in background
        background_tasks.add_task(
            optimization_engine.auto_optimize_workflow,
            workflow_id,
            max_optimizations
        )

        return {
            "success": True,
            "message": "Auto-optimization started",
            "workflow_id": workflow_id,
            "max_optimizations": max_optimizations,
            "confidence_threshold": confidence_threshold,
            "estimated_completion": "15-30 minutes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start auto-optimization: {str(e)}")

@router.get("/recommendations/{tenant_id}")
async def get_optimization_recommendations(
    tenant_id: str,
    priority: Optional[str] = Query(None, regex="^(high|medium|low)$"),
    optimization_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get optimization recommendations for all tenant workflows
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        recommendations = await optimization_engine.generate_optimization_recommendations(tenant_id)

        # Filter recommendations
        filtered_recommendations = recommendations["workflow_recommendations"]

        if priority:
            # Filter by business impact (proxy for priority)
            impact_threshold = {"high": 1000, "medium": 500, "low": 0}[priority]
            filtered_recommendations = [
                rec for rec in filtered_recommendations
                if any(
                    opp.get("business_impact", {}).get("monthly_cost_savings", 0) >= impact_threshold
                    for opp in rec["optimization_opportunities"]
                )
            ]

        if optimization_type:
            # Filter by optimization type
            filtered_recommendations = [
                {
                    **rec,
                    "optimization_opportunities": [
                        opp for opp in rec["optimization_opportunities"]
                        if opp.get("optimization_type") == optimization_type
                    ]
                }
                for rec in filtered_recommendations
            ]
            # Remove workflows with no matching opportunities
            filtered_recommendations = [rec for rec in filtered_recommendations if rec["optimization_opportunities"]]

        # Limit results
        filtered_recommendations = filtered_recommendations[:limit]

        return {
            "success": True,
            "recommendations": {
                "workflow_recommendations": filtered_recommendations,
                "portfolio_recommendations": recommendations["portfolio_recommendations"],
                "summary": {
                    "total_workflows": recommendations["total_workflows_analyzed"],
                    "total_opportunities": recommendations["optimization_opportunities"],
                    "filtered_results": len(filtered_recommendations)
                }
            },
            "generated_at": recommendations["analysis_timestamp"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@router.get("/monitoring/{optimization_id}")
async def get_optimization_monitoring(
    optimization_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get real-time monitoring data for an active optimization
    """
    try:
        with optimization_engine.Session() as session:
            # Get optimization details
            optimization_data = session.execute("""
                SELECT wo.*, oo.workflow_id, oo.tenant_id
                FROM workflow_optimizations wo
                JOIN optimization_opportunities oo ON wo.opportunity_id = oo.opportunity_id
                WHERE wo.optimization_id = :optimization_id
            """, {"optimization_id": optimization_id}).fetchone()

            if not optimization_data:
                raise HTTPException(status_code=404, detail="Optimization not found")

            # Verify access
            if optimization_data.tenant_id != current_tenant["id"] and not current_tenant.get("is_admin"):
                raise HTTPException(status_code=403, detail="Access denied to optimization data")

            # Get monitoring data
            monitoring_data = session.execute("""
                SELECT monitoring_timestamp, performance_metrics, improvement_metrics,
                       anomalies_detected, optimization_health_score, rollback_recommended
                FROM optimization_monitoring
                WHERE optimization_id = :optimization_id
                ORDER BY monitoring_timestamp DESC
                LIMIT 50
            """, {"optimization_id": optimization_id}).fetchall()

        monitoring_history = []
        for row in monitoring_data:
            monitoring_history.append({
                "timestamp": row.monitoring_timestamp.isoformat(),
                "performance_metrics": row.performance_metrics,
                "improvement_metrics": row.improvement_metrics,
                "anomalies_detected": row.anomalies_detected,
                "health_score": float(row.optimization_health_score) if row.optimization_health_score else 0,
                "rollback_recommended": row.rollback_recommended
            })

        return {
            "success": True,
            "optimization_id": optimization_id,
            "workflow_id": optimization_data.workflow_id,
            "optimization_status": optimization_data.status if hasattr(optimization_data, 'status') else 'unknown',
            "monitoring_history": monitoring_history,
            "current_health_score": monitoring_history[0]["health_score"] if monitoring_history else 0,
            "rollback_available": optimization_data.rollback_available,
            "rollback_recommended": monitoring_history[0]["rollback_recommended"] if monitoring_history else False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization monitoring: {str(e)}")

@router.get("/dashboard/{tenant_id}")
async def get_optimization_dashboard(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive workflow optimization dashboard
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with optimization_engine.Session() as session:
            # Get dashboard summary from view
            dashboard_data = session.execute("""
                SELECT * FROM workflow_optimization_dashboard
                WHERE tenant_id = :tenant_id
            """, {"tenant_id": tenant_id}).fetchall()

            # Get optimization analytics
            analytics_data = session.execute("""
                SELECT * FROM optimization_analytics
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY date DESC
            """).fetchall()

            # Get recent optimizations
            recent_optimizations = session.execute("""
                SELECT wo.optimization_id, wo.workflow_id, wo.optimization_strategy,
                       wo.actual_improvement, wo.implementation_started_at, wo.implementation_completed_at,
                       oo.optimization_type, oo.confidence_score
                FROM workflow_optimizations wo
                JOIN optimization_opportunities oo ON wo.opportunity_id = oo.opportunity_id
                WHERE wo.tenant_id = :tenant_id
                AND wo.implementation_started_at >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY wo.implementation_started_at DESC
                LIMIT 10
            """, {"tenant_id": tenant_id}).fetchall()

        # Build dashboard response
        workflows_summary = []
        for row in dashboard_data:
            workflows_summary.append({
                "workflow_id": row.workflow_id,
                "execution_time": float(row.execution_time_seconds),
                "success_rate": float(row.success_rate),
                "cost_per_execution": float(row.cost_per_execution),
                "satisfaction_score": float(row.user_satisfaction_score) if row.user_satisfaction_score else 0,
                "optimization_opportunities": row.optimization_opportunities,
                "active_optimizations": row.active_optimizations,
                "total_cost_savings": float(row.total_cost_savings),
                "avg_performance_gain": float(row.avg_performance_gain),
                "optimization_score": float(row.optimization_score)
            })

        optimization_analytics = []
        for row in analytics_data:
            optimization_analytics.append({
                "date": row.date.isoformat(),
                "workflows_analyzed": row.workflows_analyzed,
                "avg_execution_time": float(row.avg_execution_time),
                "avg_success_rate": float(row.avg_success_rate),
                "avg_cost_per_execution": float(row.avg_cost_per_execution),
                "opportunities_identified": row.opportunities_identified,
                "optimizations_implemented": row.optimizations_implemented,
                "total_cost_savings": float(row.total_cost_savings),
                "avg_performance_improvement": float(row.avg_performance_improvement)
            })

        recent_optimizations_data = []
        for row in recent_optimizations:
            recent_optimizations_data.append({
                "optimization_id": row.optimization_id,
                "workflow_id": row.workflow_id,
                "optimization_type": row.optimization_type,
                "strategy": row.optimization_strategy,
                "confidence_score": float(row.confidence_score),
                "actual_improvement": row.actual_improvement,
                "started_at": row.implementation_started_at.isoformat(),
                "completed_at": row.implementation_completed_at.isoformat() if row.implementation_completed_at else None
            })

        return {
            "success": True,
            "dashboard": {
                "tenant_id": tenant_id,
                "workflows_summary": workflows_summary,
                "optimization_analytics": optimization_analytics,
                "recent_optimizations": recent_optimizations_data,
                "summary_metrics": {
                    "total_workflows": len(workflows_summary),
                    "total_opportunities": sum(w["optimization_opportunities"] for w in workflows_summary),
                    "total_cost_savings": sum(w["total_cost_savings"] for w in workflows_summary),
                    "avg_optimization_score": sum(w["optimization_score"] for w in workflows_summary) / len(workflows_summary) if workflows_summary else 0
                }
            },
            "dashboard_generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization dashboard: {str(e)}")

@router.post("/continuous/{tenant_id}/start")
async def start_continuous_optimization(
    tenant_id: str,
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Start continuous workflow optimization monitoring
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Start continuous optimization monitoring
        background_tasks.add_task(optimization_engine.continuous_optimization_monitoring)

        return {
            "success": True,
            "message": "Continuous optimization monitoring started",
            "tenant_id": tenant_id,
            "monitoring_interval": "5 minutes",
            "started_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start continuous optimization: {str(e)}")

@router.get("/learning/{tenant_id}")
async def get_optimization_learning(
    tenant_id: str,
    learning_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get optimization learning insights and patterns
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with optimization_engine.Session() as session:
            query = """
                SELECT optimization_pattern, workflow_characteristics, strategy_applied,
                       outcome_success, improvement_achieved, confidence_level,
                       pattern_frequency, success_rate, learned_at, last_applied
                FROM optimization_learning
                WHERE tenant_id = :tenant_id
            """

            params = {"tenant_id": tenant_id}

            if learning_type:
                query += " AND optimization_pattern LIKE :learning_type"
                params["learning_type"] = f"%{learning_type}%"

            query += " ORDER BY learned_at DESC LIMIT :limit"
            params["limit"] = limit

            results = session.execute(query, params).fetchall()

        learning_data = []
        for row in results:
            learning_data.append({
                "optimization_pattern": row.optimization_pattern,
                "workflow_characteristics": row.workflow_characteristics,
                "strategy_applied": row.strategy_applied,
                "outcome_success": row.outcome_success,
                "improvement_achieved": row.improvement_achieved,
                "confidence_level": float(row.confidence_level) if row.confidence_level else 0,
                "pattern_frequency": row.pattern_frequency,
                "success_rate": float(row.success_rate) if row.success_rate else 0,
                "learned_at": row.learned_at.isoformat(),
                "last_applied": row.last_applied.isoformat() if row.last_applied else None
            })

        # Calculate learning summary
        total_patterns = len(learning_data)
        successful_patterns = len([l for l in learning_data if l["outcome_success"]])
        avg_success_rate = sum(l["success_rate"] for l in learning_data) / total_patterns if total_patterns > 0 else 0

        return {
            "success": True,
            "learning_data": learning_data,
            "learning_summary": {
                "total_patterns": total_patterns,
                "successful_patterns": successful_patterns,
                "overall_success_rate": avg_success_rate,
                "most_successful_strategy": await optimization_engine._get_most_successful_strategy(tenant_id)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get learning data: {str(e)}")

@router.post("/predict-success/{workflow_id}")
async def predict_optimization_success(
    workflow_id: str,
    prediction_params: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Predict success probability of optimization strategy
    """
    try:
        # Verify workflow access
        workflow_owner = await optimization_engine._get_workflow_owner(workflow_id)
        if workflow_owner != current_tenant["id"] and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to workflow data")

        strategy_name = prediction_params.get("strategy")
        if not strategy_name:
            raise HTTPException(status_code=400, detail="strategy is required")

        try:
            strategy = OptimizationStrategy(strategy_name)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid optimization strategy")

        # Predict success probability
        prediction = await learning_engine.predict_optimization_success(workflow_id, strategy)

        return {
            "success": True,
            "workflow_id": workflow_id,
            "strategy": strategy_name,
            "prediction": prediction,
            "predicted_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict optimization success: {str(e)}")

@router.post("/rollback/{optimization_id}")
async def rollback_optimization(
    optimization_id: str,
    rollback_reason: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Rollback a workflow optimization
    """
    try:
        # Get optimization details for access verification
        with optimization_engine.Session() as session:
            optimization_data = session.execute("""
                SELECT wo.*, oo.tenant_id, oo.workflow_id
                FROM workflow_optimizations wo
                JOIN optimization_opportunities oo ON wo.opportunity_id = oo.opportunity_id
                WHERE wo.optimization_id = :optimization_id
            """, {"optimization_id": optimization_id}).fetchone()

        if not optimization_data:
            raise HTTPException(status_code=404, detail="Optimization not found")

        # Verify access
        if optimization_data.tenant_id != current_tenant["id"] and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to optimization data")

        # Check if rollback is available
        if not optimization_data.rollback_available:
            raise HTTPException(status_code=400, detail="Rollback not available for this optimization")

        if optimization_data.rollback_executed:
            raise HTTPException(status_code=400, detail="Optimization already rolled back")

        # Execute rollback
        rollback_result = await optimization_engine._rollback_optimization(
            optimization_data.workflow_id, optimization_id, rollback_reason
        )

        # Update optimization record
        with optimization_engine.Session() as session:
            session.execute("""
                UPDATE workflow_optimizations
                SET rollback_executed = TRUE,
                    rollback_reason = :reason,
                    rollback_completed_at = CURRENT_TIMESTAMP
                WHERE optimization_id = :optimization_id
            """, {"optimization_id": optimization_id, "reason": rollback_reason})
            session.commit()

        return {
            "success": True,
            "message": "Optimization rolled back successfully",
            "optimization_id": optimization_id,
            "rollback_reason": rollback_reason,
            "rollback_result": rollback_result,
            "rolled_back_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rollback optimization: {str(e)}")

@router.get("/analytics/overview/{tenant_id}")
async def get_optimization_analytics(
    tenant_id: str,
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Get workflow optimization analytics overview
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Calculate date range
        if period == "7d":
            days = 7
        elif period == "30d":
            days = 30
        elif period == "90d":
            days = 90
        else:
            days = 365

        with optimization_engine.Session() as session:
            analytics = session.execute("""
                SELECT
                    COUNT(DISTINCT wpm.workflow_id) as total_workflows,
                    COUNT(oo.id) as total_opportunities,
                    COUNT(wo.id) as total_optimizations,
                    COUNT(CASE WHEN wo.implementation_completed_at IS NOT NULL THEN 1 END) as completed_optimizations,
                    AVG(wpm.execution_time_seconds) as avg_execution_time,
                    AVG(wpm.success_rate) as avg_success_rate,
                    AVG(wpm.cost_per_execution) as avg_cost_per_execution,
                    SUM(COALESCE((wo.actual_improvement->>'cost_reduction')::DECIMAL, 0)) as total_cost_savings,
                    AVG(COALESCE((wo.actual_improvement->>'performance_improvement')::DECIMAL, 0)) as avg_performance_improvement
                FROM workflow_performance_metrics wpm
                LEFT JOIN optimization_opportunities oo ON wpm.workflow_id = oo.workflow_id
                LEFT JOIN workflow_optimizations wo ON oo.opportunity_id = wo.opportunity_id
                WHERE wpm.tenant_id = :tenant_id
                AND wpm.created_at >= CURRENT_DATE - INTERVAL :days DAY
            """, {"tenant_id": tenant_id, "days": days}).fetchone()

        return {
            "success": True,
            "analytics": {
                "period": period,
                "total_workflows": analytics.total_workflows,
                "total_opportunities": analytics.total_opportunities,
                "total_optimizations": analytics.total_optimizations,
                "completed_optimizations": analytics.completed_optimizations,
                "completion_rate": (analytics.completed_optimizations / analytics.total_optimizations * 100) if analytics.total_optimizations > 0 else 0,
                "performance_metrics": {
                    "avg_execution_time": float(analytics.avg_execution_time) if analytics.avg_execution_time else 0,
                    "avg_success_rate": float(analytics.avg_success_rate) if analytics.avg_success_rate else 0,
                    "avg_cost_per_execution": float(analytics.avg_cost_per_execution) if analytics.avg_cost_per_execution else 0
                },
                "optimization_impact": {
                    "total_cost_savings": float(analytics.total_cost_savings) if analytics.total_cost_savings else 0,
                    "avg_performance_improvement": float(analytics.avg_performance_improvement) if analytics.avg_performance_improvement else 0
                }
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization analytics: {str(e)}")
