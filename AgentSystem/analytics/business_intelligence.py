"""
AgentSystem Advanced Analytics and Business Intelligence
Real-time business intelligence platform for data-driven decision making
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

@dataclass
class AnalyticsMetric:
    """Analytics metric definition"""
    name: str
    value: float
    change_percent: float
    trend: str  # 'up', 'down', 'stable'
    period: str
    timestamp: datetime

@dataclass
class BusinessInsight:
    """Business insight with actionable recommendations"""
    title: str
    description: str
    impact_level: str  # 'high', 'medium', 'low'
    category: str  # 'revenue', 'cost', 'customer', 'operational'
    recommendations: List[str]
    confidence_score: float
    data_points: Dict[str, Any]

class AdvancedAnalyticsEngine:
    """Advanced analytics and business intelligence engine"""

    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)

    async def generate_executive_dashboard(self, tenant_id: str, period: str = "30d") -> Dict[str, Any]:
        """Generate comprehensive executive dashboard"""

        # Calculate date range
        end_date = datetime.now()
        if period == "7d":
            start_date = end_date - timedelta(days=7)
        elif period == "30d":
            start_date = end_date - timedelta(days=30)
        elif period == "90d":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=30)

        # Gather key metrics
        revenue_metrics = await self._calculate_revenue_metrics(tenant_id, start_date, end_date)
        usage_metrics = await self._calculate_usage_metrics(tenant_id, start_date, end_date)
        customer_metrics = await self._calculate_customer_metrics(tenant_id, start_date, end_date)
        operational_metrics = await self._calculate_operational_metrics(tenant_id, start_date, end_date)

        # Generate insights
        insights = await self._generate_business_insights(tenant_id, {
            "revenue": revenue_metrics,
            "usage": usage_metrics,
            "customer": customer_metrics,
            "operational": operational_metrics
        })

        dashboard = {
            "tenant_id": tenant_id,
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "key_metrics": {
                "revenue": revenue_metrics,
                "usage": usage_metrics,
                "customer": customer_metrics,
                "operational": operational_metrics
            },
            "insights": insights,
            "recommendations": await self._generate_recommendations(insights),
            "alerts": await self._check_alerts(tenant_id, revenue_metrics, usage_metrics, customer_metrics)
        }

        return dashboard

    async def _calculate_revenue_metrics(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, AnalyticsMetric]:
        """Calculate revenue-related metrics"""

        with self.Session() as session:
            # Current period revenue
            current_revenue_query = text("""
                SELECT COALESCE(SUM(amount), 0) as total_revenue
                FROM billing_transactions
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_date AND :end_date
                AND status = 'completed'
            """)

            current_result = session.execute(current_revenue_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            current_revenue = float(current_result.total_revenue) if current_result.total_revenue else 0.0

            # Previous period for comparison
            period_length = end_date - start_date
            prev_start = start_date - period_length
            prev_end = start_date

            prev_result = session.execute(current_revenue_query, {
                "tenant_id": tenant_id,
                "start_date": prev_start,
                "end_date": prev_end
            }).fetchone()

            prev_revenue = float(prev_result.total_revenue) if prev_result.total_revenue else 0.0

            # Calculate growth
            revenue_growth = ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0.0

            # Monthly Recurring Revenue (MRR)
            mrr_query = text("""
                SELECT COALESCE(SUM(amount), 0) as mrr
                FROM subscriptions s
                JOIN billing_transactions bt ON s.id = bt.subscription_id
                WHERE s.tenant_id = :tenant_id
                AND s.status = 'active'
                AND bt.created_at >= :start_date
                AND bt.billing_cycle = 'monthly'
            """)

            mrr_result = session.execute(mrr_query, {
                "tenant_id": tenant_id,
                "start_date": start_date
            }).fetchone()

            mrr = float(mrr_result.mrr) if mrr_result.mrr else 0.0

            # Average Revenue Per User (ARPU)
            arpu_query = text("""
                SELECT COUNT(DISTINCT tenant_id) as active_users
                FROM usage_tracking
                WHERE tenant_id = :tenant_id
                AND timestamp BETWEEN :start_date AND :end_date
            """)

            arpu_result = session.execute(arpu_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            active_users = arpu_result.active_users if arpu_result.active_users else 1
            arpu = current_revenue / active_users if active_users > 0 else 0.0

        return {
            "total_revenue": AnalyticsMetric(
                name="Total Revenue",
                value=current_revenue,
                change_percent=revenue_growth,
                trend="up" if revenue_growth > 0 else "down" if revenue_growth < 0 else "stable",
                period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                timestamp=datetime.now()
            ),
            "mrr": AnalyticsMetric(
                name="Monthly Recurring Revenue",
                value=mrr,
                change_percent=0.0,  # Would need historical data for comparison
                trend="stable",
                period="Current",
                timestamp=datetime.now()
            ),
            "arpu": AnalyticsMetric(
                name="Average Revenue Per User",
                value=arpu,
                change_percent=0.0,
                trend="stable",
                period="Current",
                timestamp=datetime.now()
            )
        }

    async def _calculate_usage_metrics(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, AnalyticsMetric]:
        """Calculate usage-related metrics"""

        with self.Session() as session:
            # API calls
            api_calls_query = text("""
                SELECT COALESCE(SUM(api_calls), 0) as total_calls
                FROM usage_tracking
                WHERE tenant_id = :tenant_id
                AND timestamp BETWEEN :start_date AND :end_date
            """)

            api_result = session.execute(api_calls_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            total_api_calls = int(api_result.total_calls) if api_result.total_calls else 0

            # AI tokens used
            tokens_query = text("""
                SELECT COALESCE(SUM(ai_tokens), 0) as total_tokens
                FROM usage_tracking
                WHERE tenant_id = :tenant_id
                AND timestamp BETWEEN :start_date AND :end_date
            """)

            tokens_result = session.execute(tokens_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            total_tokens = int(tokens_result.total_tokens) if tokens_result.total_tokens else 0

            # Cost efficiency (revenue per token)
            cost_efficiency = (await self._calculate_revenue_metrics(tenant_id, start_date, end_date))["total_revenue"].value / total_tokens if total_tokens > 0 else 0.0

        return {
            "api_calls": AnalyticsMetric(
                name="Total API Calls",
                value=float(total_api_calls),
                change_percent=0.0,
                trend="stable",
                period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                timestamp=datetime.now()
            ),
            "ai_tokens": AnalyticsMetric(
                name="AI Tokens Used",
                value=float(total_tokens),
                change_percent=0.0,
                trend="stable",
                period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                timestamp=datetime.now()
            ),
            "cost_efficiency": AnalyticsMetric(
                name="Revenue per AI Token",
                value=cost_efficiency,
                change_percent=0.0,
                trend="stable",
                period="Current",
                timestamp=datetime.now()
            )
        }

    async def _calculate_customer_metrics(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, AnalyticsMetric]:
        """Calculate customer-related metrics"""

        with self.Session() as session:
            # Active users
            active_users_query = text("""
                SELECT COUNT(DISTINCT user_id) as active_users
                FROM usage_tracking
                WHERE tenant_id = :tenant_id
                AND timestamp BETWEEN :start_date AND :end_date
            """)

            users_result = session.execute(active_users_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            active_users = int(users_result.active_users) if users_result.active_users else 0

            # Customer satisfaction (from support tickets)
            satisfaction_query = text("""
                SELECT AVG(satisfaction_score) as avg_satisfaction
                FROM support_tickets
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_date AND :end_date
                AND satisfaction_score IS NOT NULL
            """)

            satisfaction_result = session.execute(satisfaction_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            avg_satisfaction = float(satisfaction_result.avg_satisfaction) if satisfaction_result.avg_satisfaction else 0.0

        return {
            "active_users": AnalyticsMetric(
                name="Active Users",
                value=float(active_users),
                change_percent=0.0,
                trend="stable",
                period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                timestamp=datetime.now()
            ),
            "satisfaction": AnalyticsMetric(
                name="Customer Satisfaction",
                value=avg_satisfaction,
                change_percent=0.0,
                trend="stable",
                period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                timestamp=datetime.now()
            )
        }

    async def _calculate_operational_metrics(self, tenant_id: str, start_date: datetime, end_date: datetime) -> Dict[str, AnalyticsMetric]:
        """Calculate operational metrics"""

        with self.Session() as session:
            # System uptime
            uptime_query = text("""
                SELECT AVG(uptime_percentage) as avg_uptime
                FROM system_health_metrics
                WHERE tenant_id = :tenant_id
                AND timestamp BETWEEN :start_date AND :end_date
            """)

            uptime_result = session.execute(uptime_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            avg_uptime = float(uptime_result.avg_uptime) if uptime_result.avg_uptime else 99.9

            # Response time
            response_time_query = text("""
                SELECT AVG(response_time_ms) as avg_response_time
                FROM api_performance_metrics
                WHERE tenant_id = :tenant_id
                AND timestamp BETWEEN :start_date AND :end_date
            """)

            response_result = session.execute(response_time_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()

            avg_response_time = float(response_result.avg_response_time) if response_result.avg_response_time else 200.0

        return {
            "uptime": AnalyticsMetric(
                name="System Uptime",
                value=avg_uptime,
                change_percent=0.0,
                trend="stable",
                period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                timestamp=datetime.now()
            ),
            "response_time": AnalyticsMetric(
                name="Average Response Time (ms)",
                value=avg_response_time,
                change_percent=0.0,
                trend="stable",
                period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                timestamp=datetime.now()
            )
        }

    async def _generate_business_insights(self, tenant_id: str, metrics: Dict[str, Any]) -> List[BusinessInsight]:
        """Generate actionable business insights from metrics"""
        insights = []

        revenue_metrics = metrics["revenue"]
        usage_metrics = metrics["usage"]
        customer_metrics = metrics["customer"]

        # Revenue growth insight
        total_revenue = revenue_metrics["total_revenue"]
        if total_revenue.change_percent > 20:
            insights.append(BusinessInsight(
                title="Strong Revenue Growth",
                description=f"Revenue has grown by {total_revenue.change_percent:.1f}% in the current period, indicating strong business momentum.",
                impact_level="high",
                category="revenue",
                recommendations=[
                    "Consider increasing marketing spend to capitalize on growth",
                    "Analyze successful customer segments for expansion",
                    "Evaluate pricing strategy for optimization"
                ],
                confidence_score=0.9,
                data_points={"growth_rate": total_revenue.change_percent, "revenue": total_revenue.value}
            ))
        elif total_revenue.change_percent < -10:
            insights.append(BusinessInsight(
                title="Revenue Decline Alert",
                description=f"Revenue has declined by {abs(total_revenue.change_percent):.1f}% in the current period, requiring immediate attention.",
                impact_level="high",
                category="revenue",
                recommendations=[
                    "Conduct customer churn analysis",
                    "Review pricing and competitive positioning",
                    "Implement customer retention campaigns"
                ],
                confidence_score=0.95,
                data_points={"decline_rate": total_revenue.change_percent, "revenue": total_revenue.value}
            ))

        # Usage efficiency insight
        cost_efficiency = usage_metrics["cost_efficiency"]
        if cost_efficiency.value > 0.01:  # $0.01 per token
            insights.append(BusinessInsight(
                title="High Cost Efficiency",
                description=f"Revenue per AI token is ${cost_efficiency.value:.4f}, indicating efficient resource utilization.",
                impact_level="medium",
                category="operational",
                recommendations=[
                    "Document best practices for efficient AI usage",
                    "Consider premium pricing for high-efficiency features",
                    "Expand successful use cases"
                ],
                confidence_score=0.8,
                data_points={"efficiency": cost_efficiency.value}
            ))

        # Customer satisfaction insight
        satisfaction = customer_metrics["satisfaction"]
        if satisfaction.value < 3.5:  # Assuming 1-5 scale
            insights.append(BusinessInsight(
                title="Customer Satisfaction Concern",
                description=f"Customer satisfaction score is {satisfaction.value:.1f}, below optimal levels.",
                impact_level="high",
                category="customer",
                recommendations=[
                    "Implement customer feedback collection system",
                    "Increase customer support team capacity",
                    "Conduct customer interviews to identify pain points"
                ],
                confidence_score=0.85,
                data_points={"satisfaction_score": satisfaction.value}
            ))

        return insights

    async def _generate_recommendations(self, insights: List[BusinessInsight]) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations based on insights"""
        recommendations = []

        # Group recommendations by category and impact
        high_impact_insights = [i for i in insights if i.impact_level == "high"]

        for insight in high_impact_insights:
            for rec in insight.recommendations:
                recommendations.append({
                    "recommendation": rec,
                    "category": insight.category,
                    "priority": "high",
                    "related_insight": insight.title,
                    "confidence": insight.confidence_score
                })

        # Sort by confidence score
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    async def _check_alerts(self, tenant_id: str, revenue_metrics: Dict, usage_metrics: Dict, customer_metrics: Dict) -> List[Dict[str, Any]]:
        """Check for critical alerts that need immediate attention"""
        alerts = []

        # Revenue decline alert
        if revenue_metrics["total_revenue"].change_percent < -15:
            alerts.append({
                "type": "critical",
                "title": "Significant Revenue Decline",
                "message": f"Revenue has declined by {abs(revenue_metrics['total_revenue'].change_percent):.1f}%",
                "action_required": True,
                "timestamp": datetime.now().isoformat()
            })

        # High usage without revenue growth
        if usage_metrics["ai_tokens"].value > 1000000 and revenue_metrics["total_revenue"].change_percent < 5:
            alerts.append({
                "type": "warning",
                "title": "High Usage, Low Revenue Growth",
                "message": "High AI token usage not translating to proportional revenue growth",
                "action_required": True,
                "timestamp": datetime.now().isoformat()
            })

        # Low customer satisfaction
        if customer_metrics["satisfaction"].value < 3.0:
            alerts.append({
                "type": "warning",
                "title": "Low Customer Satisfaction",
                "message": f"Customer satisfaction at {customer_metrics['satisfaction'].value:.1f}/5.0",
                "action_required": True,
                "timestamp": datetime.now().isoformat()
            })

        return alerts

    async def generate_predictive_forecast(self, tenant_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """Generate predictive forecast for key metrics"""

        # Get historical data for trend analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 90 days of history

        with self.Session() as session:
            # Revenue trend
            revenue_query = text("""
                SELECT DATE(created_at) as date, SUM(amount) as daily_revenue
                FROM billing_transactions
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_date AND :end_date
                AND status = 'completed'
                GROUP BY DATE(created_at)
                ORDER BY date
            """)

            revenue_data = session.execute(revenue_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchall()

            # Usage trend
            usage_query = text("""
                SELECT DATE(timestamp) as date, SUM(ai_tokens) as daily_tokens
                FROM usage_tracking
                WHERE tenant_id = :tenant_id
                AND timestamp BETWEEN :start_date AND :end_date
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)

            usage_data = session.execute(usage_query, {
                "tenant_id": tenant_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchall()

        # Simple linear regression for forecasting
        revenue_forecast = self._calculate_linear_forecast(revenue_data, forecast_days)
        usage_forecast = self._calculate_linear_forecast(usage_data, forecast_days)

        return {
            "tenant_id": tenant_id,
            "forecast_period_days": forecast_days,
            "generated_at": datetime.now().isoformat(),
            "revenue_forecast": revenue_forecast,
            "usage_forecast": usage_forecast,
            "confidence_level": 0.75  # Based on historical data quality
        }

    def _calculate_linear_forecast(self, historical_data: List, forecast_days: int) -> Dict[str, Any]:
        """Calculate linear forecast from historical data"""
        if len(historical_data) < 7:  # Need minimum data points
            return {"error": "Insufficient historical data"}

        # Convert to numpy arrays for calculation
        dates = [i for i in range(len(historical_data))]
        values = [float(row[1]) if row[1] else 0.0 for row in historical_data]

        if len(values) == 0:
            return {"error": "No valid data points"}

        # Simple linear regression
        z = np.polyfit(dates, values, 1)
        slope, intercept = z[0], z[1]

        # Generate forecast
        forecast_dates = []
        forecast_values = []

        start_index = len(historical_data)
        for i in range(forecast_days):
            forecast_date = datetime.now() + timedelta(days=i+1)
            forecast_value = slope * (start_index + i) + intercept

            forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
            forecast_values.append(max(0, forecast_value))  # Ensure non-negative

        return {
            "trend_slope": slope,
            "forecast_dates": forecast_dates,
            "forecast_values": forecast_values,
            "total_forecast": sum(forecast_values)
        }

class RealtimeAnalyticsDashboard:
    """Real-time analytics dashboard for live monitoring"""

    def __init__(self, analytics_engine: AdvancedAnalyticsEngine):
        self.analytics_engine = analytics_engine
        self.active_dashboards = {}

    async def start_realtime_monitoring(self, tenant_id: str, update_interval: int = 60):
        """Start real-time monitoring for a tenant"""

        async def monitor_loop():
            while tenant_id in self.active_dashboards:
                try:
                    # Generate current metrics
                    dashboard_data = await self.analytics_engine.generate_executive_dashboard(tenant_id, "24h")

                    # Store latest data
                    self.active_dashboards[tenant_id] = {
                        "data": dashboard_data,
                        "last_updated": datetime.now(),
                        "status": "active"
                    }

                    # Check for critical alerts
                    alerts = dashboard_data.get("alerts", [])
                    critical_alerts = [a for a in alerts if a["type"] == "critical"]

                    if critical_alerts:
                        await self._send_alert_notifications(tenant_id, critical_alerts)

                    await asyncio.sleep(update_interval)

                except Exception as e:
                    logging.error(f"Error in realtime monitoring for {tenant_id}: {e}")
                    await asyncio.sleep(update_interval)

        # Start monitoring
        self.active_dashboards[tenant_id] = {"status": "starting"}
        asyncio.create_task(monitor_loop())

    async def stop_realtime_monitoring(self, tenant_id: str):
        """Stop real-time monitoring for a tenant"""
        if tenant_id in self.active_dashboards:
            del self.active_dashboards[tenant_id]

    async def get_realtime_data(self, tenant_id: str) -> Dict[str, Any]:
        """Get latest real-time data for a tenant"""
        return self.active_dashboards.get(tenant_id, {"error": "No active monitoring"})

    async def _send_alert_notifications(self, tenant_id: str, alerts: List[Dict[str, Any]]):
        """Send alert notifications (placeholder for notification system)"""
        for alert in alerts:
            self.analytics_engine.logger.warning(f"ALERT for {tenant_id}: {alert['title']} - {alert['message']}")
            # In real implementation, would send email, Slack, etc.
