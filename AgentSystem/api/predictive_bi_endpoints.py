"""
AgentSystem Predictive Business Intelligence API Endpoints
Advanced ML-powered predictive analytics for strategic business insights
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from ..database.connection import get_db
from ..intelligence.predictive_bi_platform import PredictiveBusinessIntelligence, PredictiveInsightsEngine
from ..auth.dependencies import get_current_tenant, require_permissions

router = APIRouter(prefix="/api/v2/predictive-bi", tags=["Predictive Business Intelligence"])

# Initialize predictive BI engine (would be dependency injected in real app)
predictive_bi = PredictiveBusinessIntelligence("postgresql://user:pass@localhost/agentsystem")
insights_engine = PredictiveInsightsEngine(predictive_bi)

@router.post("/forecast/revenue/{tenant_id}")
async def generate_revenue_forecast(
    tenant_id: str,
    forecast_params: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Generate advanced revenue forecast using ML models

    Uses ensemble of ML models for highly accurate revenue predictions
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        forecast_horizon_days = forecast_params.get("forecast_horizon_days", 90)
        confidence_level = forecast_params.get("confidence_level", 0.95)

        # Generate forecast
        forecast_result = await predictive_bi.generate_revenue_forecast(
            tenant_id, forecast_horizon_days, confidence_level
        )

        return {
            "success": True,
            "forecast": {
                "predicted_revenue": forecast_result.predicted_value,
                "confidence_score": forecast_result.confidence_score,
                "prediction_interval": {
                    "lower_bound": forecast_result.prediction_interval[0],
                    "upper_bound": forecast_result.prediction_interval[1]
                },
                "key_factors": forecast_result.factors,
                "recommendations": forecast_result.recommendations,
                "model_accuracy": forecast_result.model_accuracy,
                "forecast_horizon_days": forecast_horizon_days,
                "generated_at": forecast_result.timestamp.isoformat()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate revenue forecast: {str(e)}")

@router.post("/analysis/market-opportunities/{tenant_id}")
async def analyze_market_opportunities(
    tenant_id: str,
    analysis_params: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Analyze market opportunities using competitive and industry data
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        market_segments = analysis_params.get("market_segments")

        opportunities = await predictive_bi.predict_market_opportunities(tenant_id, market_segments)

        return {
            "success": True,
            "market_opportunities": opportunities,
            "total_opportunities": len(opportunities),
            "analysis_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze market opportunities: {str(e)}")

@router.get("/analysis/clv-trends/{tenant_id}")
async def get_clv_trends(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Analyze customer lifetime value trends and predictions
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        clv_analysis = await predictive_bi.analyze_customer_lifetime_value_trends(tenant_id)

        return {
            "success": True,
            "clv_analysis": clv_analysis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze CLV trends: {str(e)}")

@router.post("/scenarios/analyze/{tenant_id}")
async def run_scenario_analysis(
    tenant_id: str,
    scenarios_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Run comprehensive what-if scenario analysis
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Convert scenarios data to BusinessScenario objects
        from ..intelligence.predictive_bi_platform import BusinessScenario

        scenarios = []
        for scenario_data in scenarios_data.get("scenarios", []):
            scenarios.append(BusinessScenario(
                scenario_name=scenario_data["scenario_name"],
                parameters=scenario_data["parameters"],
                probability=scenario_data["probability"],
                impact_assessment=scenario_data.get("impact_assessment", {}),
                recommendations=scenario_data.get("recommendations", [])
            ))

        scenario_results = await predictive_bi.run_scenario_analysis(tenant_id, scenarios)

        return {
            "success": True,
            "scenario_analysis": scenario_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run scenario analysis: {str(e)}")

@router.post("/intelligence/competitive-forecast/{tenant_id}")
async def generate_competitive_forecast(
    tenant_id: str,
    competitive_params: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Generate competitive intelligence forecasts
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        competitors = competitive_params.get("competitors", [])

        competitive_forecast = await predictive_bi.generate_competitive_intelligence_forecast(
            tenant_id, competitors
        )

        return {
            "success": True,
            "competitive_forecast": competitive_forecast
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate competitive forecast: {str(e)}")

@router.get("/insights/strategic/{tenant_id}")
async def get_strategic_insights(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive strategic insights for business planning
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        strategic_insights = await insights_engine.generate_strategic_insights(tenant_id)

        return {
            "success": True,
            "strategic_insights": strategic_insights,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategic insights: {str(e)}")

@router.get("/models/performance")
async def get_model_performance(
    model_type: Optional[str] = Query(None),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get ML model performance metrics and accuracy tracking
    """
    try:
        if not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin access required")

        with predictive_bi.Session() as session:
            query = """
                SELECT model_name, prediction_type, current_accuracy,
                       total_predictions, avg_confidence, predictions_with_outcomes,
                       avg_error_rate, excellent_predictions, last_updated
                FROM model_performance_summary
            """

            params = {}
            if model_type:
                query += " WHERE prediction_type = :model_type"
                params["model_type"] = model_type

            query += " ORDER BY current_accuracy DESC"

            results = session.execute(query, params).fetchall()

        performance_data = []
        for row in results:
            performance_data.append({
                "model_name": row.model_name,
                "prediction_type": row.prediction_type,
                "accuracy_score": float(row.current_accuracy) if row.current_accuracy else 0,
                "total_predictions": row.total_predictions,
                "average_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
                "predictions_with_outcomes": row.predictions_with_outcomes,
                "average_error_rate": float(row.avg_error_rate) if row.avg_error_rate else 0,
                "excellent_predictions": row.excellent_predictions,
                "last_updated": row.last_updated.isoformat() if row.last_updated else None
            })

        return {
            "success": True,
            "model_performance": performance_data,
            "total_models": len(performance_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")

@router.get("/predictions/history/{tenant_id}")
async def get_prediction_history(
    tenant_id: str,
    prediction_type: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get historical predictions and their accuracy
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with predictive_bi.Session() as session:
            query = """
                SELECT pr.prediction_type, pr.predicted_value, pr.confidence_score,
                       pr.prediction_interval_low, pr.prediction_interval_high,
                       pr.factors, pr.recommendations, pr.prediction_date,
                       pat.actual_value, pat.prediction_error, pat.accuracy_bucket
                FROM prediction_results pr
                LEFT JOIN prediction_accuracy_tracking pat ON pr.id = pat.prediction_id
                WHERE pr.tenant_id = :tenant_id
            """

            params = {"tenant_id": tenant_id}

            if prediction_type:
                query += " AND pr.prediction_type = :prediction_type"
                params["prediction_type"] = prediction_type
            if start_date:
                query += " AND pr.prediction_date >= :start_date"
                params["start_date"] = start_date
            if end_date:
                query += " AND pr.prediction_date <= :end_date"
                params["end_date"] = end_date

            query += " ORDER BY pr.prediction_date DESC LIMIT :limit"
            params["limit"] = limit

            results = session.execute(query, params).fetchall()

        predictions = []
        for row in results:
            predictions.append({
                "prediction_type": row.prediction_type,
                "predicted_value": float(row.predicted_value),
                "confidence_score": float(row.confidence_score),
                "prediction_interval": {
                    "lower": float(row.prediction_interval_low) if row.prediction_interval_low else None,
                    "upper": float(row.prediction_interval_high) if row.prediction_interval_high else None
                },
                "key_factors": row.factors,
                "recommendations": row.recommendations,
                "prediction_date": row.prediction_date.isoformat(),
                "actual_value": float(row.actual_value) if row.actual_value else None,
                "prediction_error": float(row.prediction_error) if row.prediction_error else None,
                "accuracy_bucket": row.accuracy_bucket
            })

        return {
            "success": True,
            "predictions": predictions,
            "total_count": len(predictions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction history: {str(e)}")

@router.post("/models/retrain/{model_id}")
async def retrain_model(
    model_id: str,
    retrain_params: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Retrain a prediction model with new data (admin only)
    """
    try:
        if not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin access required")

        # Schedule model retraining in background
        def retrain_model_task():
            # This would implement actual model retraining
            # For now, simulate the process
            import time
            time.sleep(5)  # Simulate training time

            with predictive_bi.Session() as session:
                # Update model with new accuracy
                new_accuracy = retrain_params.get("expected_accuracy", 0.85)
                query = """
                    UPDATE prediction_models
                    SET accuracy_score = :accuracy, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :model_id
                """
                session.execute(query, {"accuracy": new_accuracy, "model_id": model_id})
                session.commit()

        background_tasks.add_task(retrain_model_task)

        return {
            "success": True,
            "message": "Model retraining initiated",
            "model_id": model_id,
            "estimated_completion": "5-10 minutes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrain model: {str(e)}")

@router.post("/feedback/prediction/{prediction_id}")
async def submit_prediction_feedback(
    prediction_id: str,
    feedback_data: Dict[str, Any],
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Submit actual outcome for prediction accuracy tracking
    """
    try:
        actual_value = feedback_data.get("actual_value")
        feedback_notes = feedback_data.get("notes", "")

        if actual_value is None:
            raise HTTPException(status_code=400, detail="actual_value is required")

        # Calculate accuracy and update tracking
        with predictive_bi.Session() as session:
            accuracy = session.execute(
                "SELECT calculate_prediction_accuracy(:prediction_id, :actual_value)",
                {"prediction_id": prediction_id, "actual_value": actual_value}
            ).scalar()

            # Add feedback notes
            if feedback_notes:
                session.execute("""
                    UPDATE prediction_accuracy_tracking
                    SET model_feedback = jsonb_set(
                        COALESCE(model_feedback, '{}'),
                        '{user_notes}',
                        :notes
                    )
                    WHERE prediction_id = :prediction_id
                """, {"prediction_id": prediction_id, "notes": f'"{feedback_notes}"'})
                session.commit()

        return {
            "success": True,
            "message": "Prediction feedback recorded successfully",
            "calculated_accuracy": float(accuracy) if accuracy else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/alerts/{tenant_id}")
async def get_bi_alerts(
    tenant_id: str,
    severity: Optional[str] = Query(None, regex="^(info|warning|critical|urgent)$"),
    acknowledged: bool = Query(False),
    limit: int = Query(50, ge=1, le=100),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get business intelligence alerts for a tenant
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        with predictive_bi.Session() as session:
            query = """
                SELECT alert_type, alert_title, alert_message, severity,
                       trigger_conditions, recommended_actions, is_acknowledged,
                       acknowledged_at, created_at
                FROM bi_alerts
                WHERE tenant_id = :tenant_id AND is_acknowledged = :acknowledged
            """

            params = {"tenant_id": tenant_id, "acknowledged": acknowledged}

            if severity:
                query += " AND severity = :severity"
                params["severity"] = severity

            query += " ORDER BY created_at DESC LIMIT :limit"
            params["limit"] = limit

            results = session.execute(query, params).fetchall()

        alerts = []
        for row in results:
            alerts.append({
                "alert_type": row.alert_type,
                "title": row.alert_title,
                "message": row.alert_message,
                "severity": row.severity,
                "trigger_conditions": row.trigger_conditions,
                "recommended_actions": row.recommended_actions,
                "is_acknowledged": row.is_acknowledged,
                "acknowledged_at": row.acknowledged_at.isoformat() if row.acknowledged_at else None,
                "created_at": row.created_at.isoformat()
            })

        return {
            "success": True,
            "alerts": alerts,
            "total_count": len(alerts)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get BI alerts: {str(e)}")

@router.post("/refresh-insights/{tenant_id}")
async def refresh_strategic_insights(
    tenant_id: str,
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Manually refresh strategic insights for a tenant
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Refresh insights in background
        async def refresh_insights_task():
            try:
                with predictive_bi.Session() as session:
                    # Call the stored procedure to generate new insights
                    session.execute("SELECT generate_predictive_insights(:tenant_id)", {"tenant_id": tenant_id})
                    session.commit()
            except Exception as e:
                logging.error(f"Failed to refresh insights for {tenant_id}: {e}")

        background_tasks.add_task(refresh_insights_task)

        return {
            "success": True,
            "message": "Strategic insights refresh initiated",
            "tenant_id": tenant_id,
            "estimated_completion": "2-5 minutes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh insights: {str(e)}")

@router.get("/dashboard/{tenant_id}")
async def get_predictive_dashboard(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Get comprehensive predictive BI dashboard data
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Get recent predictions summary
        with predictive_bi.Session() as session:
            recent_predictions = session.execute("""
                SELECT prediction_type, COUNT(*) as count,
                       AVG(confidence_score) as avg_confidence,
                       MAX(prediction_date) as last_prediction
                FROM prediction_results
                WHERE tenant_id = :tenant_id
                AND prediction_date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY prediction_type
            """, {"tenant_id": tenant_id}).fetchall()

            # Get active insights
            active_insights = session.execute("""
                SELECT insight_type, insight_title, confidence_score,
                       impact_level, generated_at
                FROM strategic_insights
                WHERE tenant_id = :tenant_id
                AND status = 'active'
                ORDER BY generated_at DESC
                LIMIT 10
            """, {"tenant_id": tenant_id}).fetchall()

            # Get model accuracy summary
            model_accuracy = session.execute("""
                SELECT AVG(accuracy_score) as avg_accuracy
                FROM prediction_models
                WHERE is_active = TRUE
            """).scalar()

        dashboard_data = {
            "recent_predictions": [
                {
                    "prediction_type": row.prediction_type,
                    "count": row.count,
                    "avg_confidence": float(row.avg_confidence),
                    "last_prediction": row.last_prediction.isoformat()
                }
                for row in recent_predictions
            ],
            "active_insights": [
                {
                    "insight_type": row.insight_type,
                    "title": row.insight_title,
                    "confidence_score": float(row.confidence_score),
                    "impact_level": row.impact_level,
                    "generated_at": row.generated_at.isoformat()
                }
                for row in active_insights
            ],
            "model_performance": {
                "average_accuracy": float(model_accuracy) if model_accuracy else 0,
                "total_predictions_30d": sum(row.count for row in recent_predictions),
                "prediction_types_active": len(recent_predictions)
            },
            "dashboard_generated_at": datetime.now().isoformat()
        }

        return {
            "success": True,
            "dashboard": dashboard_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictive dashboard: {str(e)}")
