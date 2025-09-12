"""
AgentSystem Analytics API Endpoints
Advanced analytics and business intelligence endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from ..database.connection import get_db
from ..analytics.business_intelligence import AdvancedAnalyticsEngine, RealtimeAnalyticsDashboard
from ..auth.dependencies import get_current_tenant, require_permissions

router = APIRouter(prefix="/api/v2/analytics", tags=["Analytics"])

# Initialize analytics engine (would be dependency injected in real app)
analytics_engine = AdvancedAnalyticsEngine("postgresql://user:pass@localhost/agentsystem")
realtime_dashboard = RealtimeAnalyticsDashboard(analytics_engine)

@router.get("/dashboard/{tenant_id}")
async def get_executive_dashboard(
    tenant_id: str,
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive executive dashboard with key business metrics

    Returns revenue, usage, customer, and operational metrics with insights
    """
    try:
        # Verify tenant access
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        dashboard_data = await analytics_engine.generate_executive_dashboard(tenant_id, period)

        return {
            "success": True,
            "data": dashboard_data,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")

@router.get("/metrics/{tenant_id}")
async def get_analytics_metrics(
    tenant_id: str,
    metric_name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get specific analytics metrics with filtering options
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Build query filters
        filters = {"tenant_id": tenant_id}
        if metric_name:
            filters["metric_name"] = metric_name
        if category:
            filters["category"] = category
        if start_date:
            filters["start_date"] = datetime.fromisoformat(start_date)
        if end_date:
            filters["end_date"] = datetime.fromisoformat(end_date)

        # Query metrics from database
        query = """
            SELECT metric_name, metric_value, change_percent, trend,
                   category, period_start, period_end, created_at
            FROM analytics_metrics
            WHERE tenant_id = %(tenant_id)s
        """

        if metric_name:
            query += " AND metric_name = %(metric_name)s"
        if category:
            query += " AND category = %(category)s"
        if start_date:
            query += " AND created_at >= %(start_date)s"
        if end_date:
            query += " AND created_at <= %(end_date)s"

        query += " ORDER BY created_at DESC LIMIT 100"

        with analytics_engine.Session() as session:
            result = session.execute(query, filters).fetchall()

        metrics = []
        for row in result:
            metrics.append({
                "name": row.metric_name,
                "value": float(row.metric_value),
                "change_percent": float(row.change_percent) if row.change_percent else 0.0,
                "trend": row.trend,
                "category": row.category,
                "period_start": row.period_start.isoformat(),
                "period_end": row.period_end.isoformat(),
                "timestamp": row.created_at.isoformat()
            })

        return {
            "success": True,
            "metrics": metrics,
            "total_count": len(metrics)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/insights/{tenant_id}")
async def get_business_insights(
    tenant_id: str,
    impact_level: Optional[str] = Query(None, regex="^(high|medium|low)$"),
    category: Optional[str] = Query(None),
    status: str = Query("active", regex="^(active|resolved|dismissed)$"),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get AI-generated business insights and recommendations
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Query insights from database
        query = """
            SELECT title, description, impact_level, category, confidence_score,
                   data_points, recommendations, status, created_at, updated_at
            FROM business_insights
            WHERE tenant_id = %(tenant_id)s AND status = %(status)s
        """

        filters = {"tenant_id": tenant_id, "status": status}

        if impact_level:
            query += " AND impact_level = %(impact_level)s"
            filters["impact_level"] = impact_level
        if category:
            query += " AND category = %(category)s"
            filters["category"] = category

        query += " ORDER BY confidence_score DESC, created_at DESC"

        with analytics_engine.Session() as session:
            result = session.execute(query, filters).fetchall()

        insights = []
        for row in result:
            insights.append({
                "title": row.title,
                "description": row.description,
                "impact_level": row.impact_level,
                "category": row.category,
                "confidence_score": float(row.confidence_score),
                "data_points": row.data_points,
                "recommendations": row.recommendations,
                "status": row.status,
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat()
            })

        return {
            "success": True,
            "insights": insights,
            "total_count": len(insights)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@router.get("/forecast/{tenant_id}")
async def get_predictive_forecast(
    tenant_id: str,
    forecast_days: int = Query(30, ge=7, le=365),
    forecast_type: str = Query("revenue", regex="^(revenue|usage|customers)$"),
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Generate predictive forecast for business metrics
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        forecast_data = await analytics_engine.generate_predictive_forecast(tenant_id, forecast_days)

        return {
            "success": True,
            "forecast": forecast_data,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")

@router.post("/realtime/{tenant_id}/start")
async def start_realtime_monitoring(
    tenant_id: str,
    update_interval: int = Query(60, ge=30, le=3600),
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Start real-time analytics monitoring for a tenant
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Start real-time monitoring
        await realtime_dashboard.start_realtime_monitoring(tenant_id, update_interval)

        return {
            "success": True,
            "message": f"Real-time monitoring started for tenant {tenant_id}",
            "update_interval": update_interval
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/realtime/{tenant_id}/stop")
async def stop_realtime_monitoring(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Stop real-time analytics monitoring for a tenant
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        await realtime_dashboard.stop_realtime_monitoring(tenant_id)

        return {
            "success": True,
            "message": f"Real-time monitoring stopped for tenant {tenant_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.get("/realtime/{tenant_id}")
async def get_realtime_data(
    tenant_id: str,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Get latest real-time analytics data
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        realtime_data = await realtime_dashboard.get_realtime_data(tenant_id)

        return {
            "success": True,
            "data": realtime_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get realtime data: {str(e)}")

@router.get("/alerts/{tenant_id}")
async def get_analytics_alerts(
    tenant_id: str,
    alert_type: Optional[str] = Query(None, regex="^(critical|warning|info)$"),
    resolved: bool = Query(False),
    limit: int = Query(50, ge=1, le=100),
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Get analytics alerts for a tenant
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Query alerts from database
        query = """
            SELECT alert_type, title, message, action_required, resolved,
                   resolved_at, metadata, created_at
            FROM analytics_alerts
            WHERE tenant_id = %(tenant_id)s AND resolved = %(resolved)s
        """

        filters = {"tenant_id": tenant_id, "resolved": resolved}

        if alert_type:
            query += " AND alert_type = %(alert_type)s"
            filters["alert_type"] = alert_type

        query += " ORDER BY created_at DESC LIMIT %(limit)s"
        filters["limit"] = limit

        with analytics_engine.Session() as session:
            result = session.execute(query, filters).fetchall()

        alerts = []
        for row in result:
            alerts.append({
                "type": row.alert_type,
                "title": row.title,
                "message": row.message,
                "action_required": row.action_required,
                "resolved": row.resolved,
                "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
                "metadata": row.metadata,
                "created_at": row.created_at.isoformat()
            })

        return {
            "success": True,
            "alerts": alerts,
            "total_count": len(alerts)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/{tenant_id}/{alert_id}/resolve")
async def resolve_alert(
    tenant_id: str,
    alert_id: str,
    current_tenant: dict = Depends(get_current_tenant),
    db: Session = Depends(get_db)
):
    """
    Mark an analytics alert as resolved
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Update alert status
        query = """
            UPDATE analytics_alerts
            SET resolved = true, resolved_at = CURRENT_TIMESTAMP, resolved_by = %(user_id)s
            WHERE id = %(alert_id)s AND tenant_id = %(tenant_id)s
        """

        with analytics_engine.Session() as session:
            result = session.execute(query, {
                "alert_id": alert_id,
                "tenant_id": tenant_id,
                "user_id": current_tenant["user_id"]
            })
            session.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {
            "success": True,
            "message": "Alert resolved successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")

@router.get("/performance/{tenant_id}")
async def get_performance_metrics(
    tenant_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Get system performance metrics for a tenant
    """
    try:
        if current_tenant["id"] != tenant_id and not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied to tenant data")

        # Default to last 24 hours if no dates provided
        if not end_date:
            end_date = datetime.now()
        else:
            end_date = datetime.fromisoformat(end_date)

        if not start_date:
            start_date = end_date - timedelta(hours=24)
        else:
            start_date = datetime.fromisoformat(start_date)

        # Query performance metrics
        query = """
            SELECT uptime_percentage, response_time_avg, error_rate,
                   throughput_requests_per_second, cpu_usage_percent,
                   memory_usage_percent, timestamp
            FROM system_health_metrics
            WHERE tenant_id = %(tenant_id)s
            AND timestamp BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY timestamp DESC
        """

        with analytics_engine.Session() as session:
            result = session.execute(query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchall()

        metrics = []
        for row in result:
            metrics.append({
                "uptime_percentage": float(row.uptime_percentage) if row.uptime_percentage else None,
                "response_time_avg": float(row.response_time_avg) if row.response_time_avg else None,
                "error_rate": float(row.error_rate) if row.error_rate else None,
                "throughput_rps": float(row.throughput_requests_per_second) if row.throughput_requests_per_second else None,
                "cpu_usage_percent": float(row.cpu_usage_percent) if row.cpu_usage_percent else None,
                "memory_usage_percent": float(row.memory_usage_percent) if row.memory_usage_percent else None,
                "timestamp": row.timestamp.isoformat()
            })

        # Calculate averages
        if metrics:
            avg_uptime = sum(m["uptime_percentage"] for m in metrics if m["uptime_percentage"]) / len([m for m in metrics if m["uptime_percentage"]])
            avg_response_time = sum(m["response_time_avg"] for m in metrics if m["response_time_avg"]) / len([m for m in metrics if m["response_time_avg"]])
        else:
            avg_uptime = 0
            avg_response_time = 0

        return {
            "success": True,
            "performance_metrics": metrics,
            "summary": {
                "average_uptime": avg_uptime,
                "average_response_time": avg_response_time,
                "total_data_points": len(metrics)
            },
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/refresh-views")
async def refresh_analytics_views(
    background_tasks: BackgroundTasks,
    current_tenant: dict = Depends(get_current_tenant)
):
    """
    Manually refresh analytics materialized views (admin only)
    """
    try:
        if not current_tenant.get("is_admin"):
            raise HTTPException(status_code=403, detail="Admin access required")

        # Refresh materialized views in background
        def refresh_views():
            with analytics_engine.Session() as session:
                session.execute("SELECT refresh_analytics_views();")
                session.commit()

        background_tasks.add_task(refresh_views)

        return {
            "success": True,
            "message": "Analytics views refresh initiated"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh views: {str(e)}")
