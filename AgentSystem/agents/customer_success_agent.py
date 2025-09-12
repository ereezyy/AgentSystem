
"""
🤝 AgentSystem Customer Success Agent
AI-powered customer success automation for retention, health monitoring, and expansion
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
from collections import defaultdict
import statistics

import asyncpg
import aioredis
from fastapi import HTTPException
from pydantic import BaseModel, Field, EmailStr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

from ..core.agent_swarm import SpecializedAgent, AgentCapability
from ..usage.usage_tracker import ServiceType, UsageTracker
from ..pricing.pricing_engine import PricingEngine
from ..billing.stripe_service import StripeService
from ..optimization.cost_optimizer_clean import CostOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    EXCELLENT = "excellent"     # 90-100 score
    HEALTHY = "healthy"         # 70-89 score
    AT_RISK = "at_risk"        # 50-69 score
    CRITICAL = "critical"       # 30-49 score
    CHURNED = "churned"        # 0-29 score

class ChurnRisk(str, Enum):
    LOW = "low"           # <20% probability
    MEDIUM = "medium"     # 20-50% probability
    HIGH = "high"         # 50-80% probability
    CRITICAL = "critical" # >80% probability

class InterventionType(str, Enum):
    ONBOARDING_HELP = "onboarding_help"
    USAGE_OPTIMIZATION = "usage_optimization"
    TRAINING_NEEDED = "training_needed"
    BILLING_ISSUE = "billing_issue"
    FEATURE_REQUEST = "feature_request"
    TECHNICAL_SUPPORT = "technical_support"
    EXPANSION_OPPORTUNITY = "expansion_opportunity"
    WIN_BACK_CAMPAIGN = "win_back_campaign"

class OnboardingStage(str, Enum):
    NOT_STARTED = "not_started"
    EMAIL_VERIFIED = "email_verified"
    FIRST_LOGIN = "first_login"
    API_CONFIGURED = "api_configured"
    FIRST_REQUEST = "first_request"
    VALUE_REALIZED = "value_realized"
    FULLY_ADOPTED = "fully_adopted"

@dataclass
class CustomerHealthScore:
    """Customer health score with detailed breakdown"""
    tenant_id: str
    overall_score: float
    health_status: HealthStatus
    churn_risk: ChurnRisk
    churn_probability: float

    # Component scores
    usage_score: float
    engagement_score: float
    support_score: float
    payment_score: float
    adoption_score: float

    # Trend indicators
    score_trend: str  # improving, stable, declining
    velocity: float   # Rate of change

    # Contextual data
    days_since_signup: int
    last_activity: datetime
    key_metrics: Dict[str, float]
    risk_factors: List[str]
    positive_indicators: List[str]

    calculated_at: datetime

@dataclass
class ChurnPrediction:
    """Churn prediction with AI analysis"""
    tenant_id: str
    churn_probability: float
    churn_risk: ChurnRisk
    predicted_churn_date: Optional[datetime]

    # Contributing factors
    risk_factors: List[Dict[str, Any]]
    protective_factors: List[Dict[str, Any]]

    # Intervention recommendations
    recommended_interventions: List[InterventionType]
    intervention_priority: str
    estimated_intervention_success: float

    # Financial impact
    clv_at_risk: float  # Customer lifetime value at risk
    retention_value: float  # Value of successful retention

    confidence_score: float
    model_version: str

@dataclass
class ExpansionOpportunity:
    """Revenue expansion opportunity"""
    tenant_id: str
    opportunity_type: str  # upgrade, add_users, new_features
    current_plan: str
    recommended_plan: str
    estimated_revenue_increase: float

    # Qualifying factors
    usage_patterns: Dict[str, Any]
    growth_indicators: List[str]
    feature_requests: List[str]

    # Timing and approach
    readiness_score: float
    best_approach: str
    recommended_timing: str
    success_probability: float

    # ROI calculation
    implementation_effort: str
    expected_roi: float

class CustomerSuccessAgent(SpecializedAgent):
    """AI-powered customer success automation agent"""

    def __init__(self, tenant_id: str, db_pool: asyncpg.Pool, redis_client: aioredis.Redis,
                 usage_tracker: UsageTracker, pricing_engine: PricingEngine,
                 stripe_service: StripeService, cost_optimizer: CostOptimizer):
        super().__init__(
            agent_id=f"customer_success_agent_{tenant_id}",
            agent_type="customer_success",
            capabilities=[
                AgentCapability.HEALTH_MONITORING,
                AgentCapability.CHURN_PREVENTION,
                AgentCapability.ONBOARDING_AUTOMATION,
                AgentCapability.EXPANSION_IDENTIFICATION,
                AgentCapability.PREDICTIVE_ANALYTICS
            ]
        )

        self.tenant_id = tenant_id
        self.db_pool = db_pool
        self.redis = redis_client
        self.usage_tracker = usage_tracker
        self.pricing_engine = pricing_engine
        self.stripe_service = stripe_service
        self.cost_optimizer = cost_optimizer

        # Health scoring weights
        self.health_weights = {
            'usage_score': 0.30,      # How actively they use the platform
            'engagement_score': 0.25,  # Email opens, logins, feature usage
            'support_score': 0.15,     # Support ticket sentiment and resolution
            'payment_score': 0.15,     # Payment history and billing health
            'adoption_score': 0.15     # Feature adoption and onboarding progress
        }

        # Churn prediction model (would be trained ML model in production)
        self.churn_model = None  # Placeholder for ML model
        self.feature_scaler = StandardScaler()

        # Intervention templates
        self.intervention_templates = self._initialize_intervention_templates()

    def _initialize_intervention_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intervention templates for different scenarios"""
        return {
            InterventionType.ONBOARDING_HELP: {
                'title': 'Let\'s get you up and running!',
                'email_template': '''
                Hi {customer_name},

                I noticed you haven't completed your onboarding yet. I'm here to help you get the most out of {platform_name}.

                Here's what I can help you with:
                - Setting up your first AI workflow
                - Integrating with your existing tools
                - Best practices for your use case

                Would you like to schedule a 15-minute setup call?

                Best regards,
                {cs_rep_name}
                ''',
                'priority': 'high',
                'timing': 'within_24_hours'
            },

            InterventionType.USAGE_OPTIMIZATION: {
                'title': 'Optimize your AI usage and save costs',
                'email_template': '''
                Hi {customer_name},

                I've been analyzing your usage patterns and found some opportunities to help you:

                🎯 Potential monthly savings: ${estimated_savings}
                📊 Usage optimization suggestions:
                {optimization_suggestions}

                I'd love to show you how to get better results while reducing costs.

                Best,
                {cs_rep_name}
                ''',
                'priority': 'medium',
                'timing': 'next_business_day'
            },

            InterventionType.EXPANSION_OPPORTUNITY: {
                'title': 'Ready to scale your AI automation?',
                'email_template': '''
                Hi {customer_name},

                Great news! Based on your usage growth ({usage_growth}% increase),
                you're ready to unlock more powerful features.

                Upgrading to {recommended_plan} would give you:
                {upgrade_benefits}

                Estimated ROI: {estimated_roi}% in the first 90 days.

                Shall we schedule a brief call to discuss?

                {cs_rep_name}
                ''',
                'priority': 'medium',
                'timing': 'within_week'
            }
        }

    async def calculate_health_score(self, target_tenant_id: Optional[str] = None) -> CustomerHealthScore:
        """Calculate comprehensive customer health score"""

        tenant_id = target_tenant_id or self.tenant_id

        try:
            # Gather health metrics
            usage_metrics = await self._get_usage_metrics(tenant_id)
            engagement_metrics = await self._get_engagement_metrics(tenant_id)
            support_metrics = await self._get_support_metrics(tenant_id)
            payment_metrics = await self._get_payment_metrics(tenant_id)
            adoption_metrics = await self._get_adoption_metrics(tenant_id)

            # Calculate component scores
            usage_score = self._calculate_usage_score(usage_metrics)
            engagement_score = self._calculate_engagement_score(engagement_metrics)
            support_score = self._calculate_support_score(support_metrics)
            payment_score = self._calculate_payment_score(payment_metrics)
            adoption_score = self._calculate_adoption_score(adoption_metrics)

            # Calculate weighted overall score
            overall_score = (
                usage_score * self.health_weights['usage_score'] +
                engagement_score * self.health_weights['engagement_score'] +
                support_score * self.health_weights['support_score'] +
                payment_score * self.health_weights['payment_score'] +
                adoption_score * self.health_weights['adoption_score']
            )

            # Determine health status and churn risk
            health_status = self._get_health_status(overall_score)
            churn_risk, churn_probability = await self._calculate_churn_risk(
                tenant_id, overall_score, usage_metrics, engagement_metrics
            )

            # Calculate trends
            score_trend, velocity = await self._calculate_score_trend(tenant_id)

            # Get contextual data
            days_since_signup = await self._get_days_since_signup(tenant_id)
            last_activity = await self._get_last_activity(tenant_id)

            # Identify risk factors and positive indicators
            risk_factors = self._identify_risk_factors(
                usage_score, engagement_score, support_score, payment_score, adoption_score
            )
            positive_indicators = self._identify_positive_indicators(
                usage_score, engagement_score, support_score, payment_score, adoption_score
            )

            health_score = CustomerHealthScore(
                tenant_id=tenant_id,
                overall_score=overall_score,
                health_status=health_status,
                churn_risk=churn_risk,
                churn_probability=churn_probability,
                usage_score=usage_score,
                engagement_score=engagement_score,
                support_score=support_score,
                payment_score=payment_score,
                adoption_score=adoption_score,
                score_trend=score_trend,
                velocity=velocity,
                days_since_signup=days_since_signup,
                last_activity=last_activity,
                key_metrics={
                    'monthly_usage': usage_metrics.get('monthly_tokens', 0),
                    'feature_adoption_rate': adoption_metrics.get('adoption_rate', 0),
                    'support_satisfaction': support_metrics.get('satisfaction_score', 0),
                    'payment_health': payment_metrics.get('payment_health', 100)
                },
                risk_factors=risk_factors,
                positive_indicators=positive_indicators,
                calculated_at=datetime.now()
            )

            # Store health score
            await self._store_health_score(health_score)

            return health_score

        except Exception as e:
            logger.error(f"Failed to calculate health score for {tenant_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Health score calculation failed: {str(e)}")

    async def _get_usage_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage-based metrics"""

        # Get current usage from usage tracker
        current_usage = await self.usage_tracker.get_current_usage(tenant_id)
        usage_summary = await self.usage_tracker.get_usage_summary(tenant_id, "current_month")

        # Calculate usage trends
        last_month_usage = await self.usage_tracker.get_usage_summary(tenant_id, "last_month")

        usage_growth = 0.0
        if last_month_usage['summary']['total_tokens'] > 0:
            current_tokens = usage_summary['summary']['total_tokens']
            last_tokens = last_month_usage['summary']['total_tokens']
            usage_growth = ((current_tokens - last_tokens) / last_tokens) * 100

        return {
            'monthly_tokens': usage_summary['summary']['total_tokens'],
            'monthly_requests': usage_summary['summary']['total_requests'],
            'usage_percentage': current_usage.get('usage_percentage', 0),
            'usage_growth': usage_growth,
            'days_since_last_use': (datetime.now() - datetime.fromisoformat(current_usage['period_start'])).days,
            'service_diversity': len(usage_summary['breakdown_by_service'])
        }

    async def _get_engagement_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get engagement-based metrics"""

        # Get login frequency, feature usage, email engagement
        async with self.db_pool.acquire() as conn:
            login_data = await conn.fetchrow("""
                SELECT
                    COUNT(*) as login_count,
                    MAX(last_login_at) as last_login,
                    MIN(joined_at) as first_login
                FROM tenant_management.tenant_users
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

            feature_usage = await conn.fetchrow("""
                SELECT
                    COUNT(DISTINCT service_type) as unique_services,
                    COUNT(*) as total_interactions
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
            """, tenant_id)

        # Calculate engagement metrics
        days_since_signup = (datetime.now() - (login_data['first_login'] or datetime.now())).days
        days_since_last_login = (datetime.now() - (login_data['last_login'] or datetime.now())).days

        return {
            'login_frequency': (login_data['login_count'] or 0) / max(days_since_signup, 1),
            'days_since_last_login': days_since_last_login,
            'feature_diversity': feature_usage['unique_services'] or 0,
            'interaction_frequency': (feature_usage['total_interactions'] or 0) / 30,
            'email_engagement': 0.7  # Placeholder - would track email opens/clicks
        }

    async def _get_support_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get support-based metrics"""

        # Placeholder for support ticket analysis
        # In production, would integrate with support system
        return {
            'open_tickets': 0,
            'avg_resolution_time': 24,  # hours
            'satisfaction_score': 4.2,  # out of 5
            'escalation_rate': 0.1,
            'self_service_usage': 0.8
        }

    async def _get_payment_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get payment and billing health metrics"""

        try:
            billing_history = await self.stripe_service.get_billing_history(tenant_id, limit=6)

            # Analyze payment patterns
            failed_payments = sum(1 for bill in billing_history if bill.get('payment_status') == 'failed')
            on_time_payments = sum(1 for bill in billing_history if bill.get('payment_status') == 'paid')

            payment_health = 100
            if billing_history:
                payment_health = (on_time_payments / len(billing_history)) * 100

            return {
                'payment_health': payment_health,
                'failed_payment_count': failed_payments,
                'days_overdue': 0,  # Would calculate from Stripe data
                'payment_method_issues': failed_payments > 0,
                'subscription_status': 'active'  # Would get from Stripe
            }

        except Exception as e:
            logger.warning(f"Could not get payment metrics for {tenant_id}: {e}")
            return {
                'payment_health': 100,
                'failed_payment_count': 0,
                'days_overdue': 0,
                'payment_method_issues': False,
                'subscription_status': 'active'
            }

    async def _get_adoption_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get feature adoption and onboarding progress metrics"""

        # Get onboarding progress
        onboarding_progress = await self._get_onboarding_progress(tenant_id)

        # Calculate feature adoption
        async with self.db_pool.acquire() as conn:
            feature_usage = await conn.fetch("""
                SELECT DISTINCT service_type
                FROM tenant_management.usage_events
                WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
            """, tenant_id)

            total_features = len(ServiceType)  # All available services
            adopted_features = len(feature_usage)
            adoption_rate = (adopted_features / total_features) * 100 if total_features > 0 else 0

        return {
            'onboarding_completion': onboarding_progress,
            'adoption_rate': adoption_rate,
            'features_adopted': adopted_features,
            'time_to_value': 7,  # days (would calculate actual)
            'advanced_features_used': adopted_features > 3
        }

    async def _get_onboarding_progress(self, tenant_id: str) -> float:
        """Calculate onboarding completion percentage"""

        progress = 0.0

        async with self.db_pool.acquire() as conn:
            tenant_data = await conn.fetchrow("""
                SELECT status, settings FROM tenant_management.tenants WHERE id = $1
            """, tenant_id)

            if not tenant_data:
                return 0.0

            # Basic onboarding steps
            if tenant_data['status'] in ['active', 'trial']:
                progress += 20  # Email verified and account active

            # Check for API key generation
            api_keys = await conn.fetchval("""
                SELECT COUNT(*) FROM tenant_management.tenant_api_keys
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

            if api_keys > 0:
                progress += 20  # API configured

            # Check for first API usage
            usage_count = await conn.fetchval("""
                SELECT COUNT(*) FROM tenant_management.usage_events WHERE tenant_id = $1
            """, tenant_id)

            if usage_count > 0:
                progress += 30  # First request made

            if usage_count > 10:
                progress += 20  # Regular usage established

            if usage_count > 100:
                progress += 10  # Value realized

        return min(progress, 100.0)

    def _calculate_usage_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate usage component score (0-100)"""

        score = 50  # Base score

        # Usage percentage bonus/penalty
        usage_pct = metrics.get('usage_percentage', 0)
        if usage_pct > 80:
            score += 30
        elif usage_pct > 50:
            score += 20
        elif usage_pct > 20:
            score += 10
        elif usage_pct < 5:
            score -= 30

        # Usage growth bonus
        growth = metrics.get('usage_growth', 0)
        if growth > 50:
            score += 15
        elif growth > 20:
            score += 10
        elif growth < -20:
            score -= 15

        # Recency penalty
        days_since_use = metrics.get('days_since_last_use', 0)
        if days_since_use > 14:
            score -= 25
        elif days_since_use > 7:
            score -= 15
        elif days_since_use > 3:
            score -= 5

        return max(0, min(100, score))

    def _calculate_engagement_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate engagement component score (0-100)"""

        score = 50  # Base score

        # Login frequency
        login_freq = metrics.get('login_frequency', 0)
        if login_freq > 1:  # More than daily
            score += 25
        elif login_freq > 0.5:  # Every other day
            score += 15
        elif login_freq > 0.2:  # Few times per week
            score += 5
        else:
            score -= 15

        # Days since last login penalty
        days_since_login = metrics.get('days_since_last_login', 0)
        if days_since_login > 30:
            score -= 30
        elif days_since_login > 14:
            score -= 20
        elif days_since_login > 7:
            score -= 10

        # Feature diversity bonus
        feature_diversity = metrics.get('feature_diversity', 0)
        score += min(feature_diversity * 5, 25)

        return max(0, min(100, score))

    def _calculate_support_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate support component score (0-100)"""

        score = 80  # Default good score

        # Open tickets penalty
        open_tickets = metrics.get('open_tickets', 0)
        score -= min(open_tickets * 10, 30)

        # Satisfaction bonus/penalty
        satisfaction = metrics.get('satisfaction_score', 4.0)
        if satisfaction > 4.5:
            score += 15
        elif satisfaction > 4.0:
            score += 5
        elif satisfaction < 3.0:
            score -= 20

        return max(0, min(100, score))

    def _calculate_payment_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate payment component score (0-100)"""

        payment_health = metrics.get('payment_health', 100)

        # Deductions for payment issues
        if metrics.get('days_overdue', 0) > 0:
            payment_health -= 30

        if metrics.get('payment_method_issues', False):
            payment_health -= 15

        if metrics.get('subscription_status') != 'active':
            payment_health -= 50

        return max(0, min(100, payment_health))

    def _calculate_adoption_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate adoption component score (0-100)"""

        # Base score from onboarding completion
        score = metrics.get('onboarding_completion', 0)

        # Feature adoption bonus
        adoption_rate = metrics.get('adoption_rate', 0)
        score = (score + adoption_rate) / 2  # Average of onboarding and adoption

        # Advanced features bonus
        if metrics.get('advanced_features_used', False):
            score += 10

        return max(0, min(100, score))

    def _get_health_status(self, overall_score: float) -> HealthStatus:
        """Determine health status from overall score"""

        if overall_score >= 90:
            return HealthStatus.EXCELLENT
        elif overall_score >= 70:
            return HealthStatus.HEALTHY
        elif overall_score >= 50:
            return HealthStatus.AT_RISK
        elif overall_score >= 30:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.CHURNED

    async def _calculate_churn_risk(self, tenant_id: str, overall_score: float,
                                  usage_metrics: Dict, engagement_metrics: Dict) -> Tuple[ChurnRisk, float]:
        """Calculate churn risk and probability"""

        # Simple rule-based approach (would use ML model in production)
        risk_factors = 0
        probability = 0.1  # Base probability

        # Score-based risk
        if overall_score < 30:
            risk_factors += 3
            probability += 0.6
        elif overall_score < 50:
            risk_factors += 2
            probability += 0.3
        elif overall_score < 70:
            risk_factors += 1
            probability += 0.1

        # Usage-based risk
        if usage_metrics.get('days_since_last_use', 0) > 14:
            risk_factors += 2
            probability += 0.2

        if usage_metrics.get('usage_growth', 0) < -50:
            risk_factors += 2
            probability += 0.2

        # Engagement-based risk
        if engagement_metrics.get('days_since_last_login', 0) > 30:
            risk_factors += 2
            probability += 0.2

        # Determine risk level
        probability = min(probability, 0.95)

        if probability > 0.8:
            churn_risk = ChurnRisk.CRITICAL
        elif probability > 0.5:
            churn_risk = ChurnRisk.HIGH
        elif probability > 0.2:
            churn_risk = ChurnRisk.MEDIUM
        else:
            churn_risk = ChurnRisk.LOW

        return churn_risk, probability

    async def _calculate_score_trend(self, tenant_id: str) -> Tuple[str, float]:
        """Calculate health score trend and velocity"""

        # Get historical health scores
        async with self.db_pool.acquire() as conn:
            historical_scores = await conn.fetch("""
                SELECT overall_score, calculated_at
                FROM tenant_management.customer_health_scores
                WHERE tenant_id = $1
                ORDER BY calculated_at DESC
                LIMIT 5
            """, tenant_id)

        if len(historical_scores) < 2:
            return "stable", 0.0

        # Calculate trend
        scores = [float(score['overall_score']) for score in historical_scores]

        # Simple trend calculation
        if len(scores) >= 3:
            recent_avg = sum(scores[:2]) / 2
            older_avg = sum(scores[-2:]) / 2

            velocity = recent_avg - older_avg

            if velocity > 5:
                trend = "improving"
            elif velocity < -5:
                trend = "declining"
            else:
                trend = "stable"

            return trend, velocity

        return "stable", 0.0

    async def _get_days_since_signup(self, tenant_id: str) -> int:
        """Get days since tenant signup"""

        async with self.db_pool.acquire() as conn:
            signup_date = await conn.fetchval("""
                SELECT created_at FROM tenant_management.tenants WHERE id = $1
            """, tenant_id)

            if signup_date:
                return (datetime.now() - signup_date).days

            return 0

    async def _get_last_activity(self, tenant_id: str) -> datetime:
        """Get last activity timestamp"""

        async with self.db_pool.acquire() as conn:
            last_activity = await conn.fetchval("""
                SELECT MAX(created_at) FROM tenant_management.usage_events
                WHERE tenant_id = $1
            """, tenant_id)

            return last_activity or datetime.now()

    def _identify_risk_factors(self, usage_score: float, engagement_score: float,
                             support_score: float, payment_score: float, adoption_score: float) -> List[str]:
        """Identify specific risk factors"""

        risks = []

        if usage_score < 40:
            risks.append("Low platform usage")
        if engagement_score < 40:
            risks.append("Poor engagement metrics")
        if support_score < 60:
            risks.append("Support issues")
        if payment_score < 80:
            risks.append("Payment problems")
        if adoption_score < 50:
            risks.append("Slow feature adoption")

        return risks

    def _identify_positive_indicators(self, usage_score: float, engagement_score: float,
                                    support_score: float, payment_score: float, adoption_score: float) -> List[str]:
        """Identify positive indicators"""

        positives = []

        if usage_score > 80:
            positives.append("High platform usage")
        if engagement_score > 80:
            positives.append("Strong engagement")
        if support_score > 90:
            positives.append("Excellent support experience")
        if payment_score > 95:
            positives.append("Perfect payment history")
        if adoption_score > 80:
            positives.append("Strong feature adoption")

        return positives

    async def _store_health_score(self, health_score: CustomerHealthScore):
        """Store health score in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.customer_health_scores (
                    tenant_id, calculated_date, usage_score, engagement_score,
                    support_score, payment_score, overall_health_score,
                    health_trend, churn_risk_score, predicted_clv, factors
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (tenant_id, calculated_date) DO UPDATE SET
                    usage_score = EXCLUDED.usage_score,
                    engagement_score = EXCLUDED.engagement_score,
                    support_score = EXCLUDED.support_score,
                    payment_score = EXCLUDED.payment_score,
                    overall_health_score = EXCLUDED.overall_health_score,
                    health_trend = EXCLUDED.health_trend,
                    churn_risk_score = EXCLUDED.churn_risk_score,
                    predicted_clv = EXCLUDED.predicted_clv,
                    factors = EXCLUDED.factors
            """,
                health_score.tenant_id, health_score.calculated_at.date(),
                health_score.usage_score, health_score.engagement_score,
                health_score.support_score, health_score.payment_score,
                health_score.overall_score, health_score.score_trend,
                health_score.churn_probability, 5000.0,  # Placeholder CLV
                json.dumps({
                    'risk_factors': health_score.risk_factors,
                    'positive_indicators': health_score.positive_indicators,
                    'key_metrics': health_score.key_metrics
                })
            )

    async def predict_churn(self, target_tenant_id: Optional[str] = None) -> ChurnPrediction:
        """Predict churn probability and recommend interventions"""

        tenant_id = target_tenant_id or self.tenant_id

        # Get current health score
        health_score = await self.calculate_health_score(tenant_id)

        # Calculate predicted churn date
        predicted_churn_date = None
        if health_score.churn_probability > 0.5:
            # Estimate churn date based on score velocity
            days_to_churn = max(7, int(30 * (1 - health_score.churn_probability)))
            predicted_churn_date = datetime.now() + timedelta(days=days_to_churn)

        # Identify risk and protective factors
        risk_factors = [
            {'factor': factor, 'impact': 'high', 'weight': 0.8}
            for factor in health_score.risk_factors
        ]

        protective_factors = [
            {'factor': factor, 'impact': 'positive', 'weight': 0.7}
            for factor in health_score.positive_indicators
        ]

        # Recommend interventions
        recommended_interventions = self._recommend_interventions(health_score)

        # Calculate financial impact
        clv_at_risk = await self._calculate_clv_at_risk(tenant_id)
        retention_value = clv_at_risk * 0.8  # 80% of CLV if retained

        prediction = ChurnPrediction(
            tenant_id=tenant_id,
            churn_probability=health_score.churn_probability,
            churn_risk=health_score.churn_risk,
            predicted_churn_date=predicted_churn_date,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            recommended_interventions=recommended_interventions,
            intervention_priority='high' if health_score.churn_probability > 0.7 else 'medium',
            estimated_intervention_success=0.6,  # 60% success rate
            clv_at_risk=clv_at_risk,
            retention_value=retention_value,
            confidence_score=0.85,
            model_version="v1.0"
        )

        # Store prediction
        await self._store_churn_prediction(prediction)

        return prediction

    def _recommend_interventions(self, health_score: CustomerHealthScore) -> List[InterventionType]:
        """Recommend interventions based on health score"""

        interventions = []

        # Usage-based interventions
        if health_score.usage_score < 40:
            interventions.append(InterventionType.USAGE_OPTIMIZATION)
            if health_score.adoption_score < 50:
                interventions.append(InterventionType.ONBOARDING_HELP)

        # Engagement-based interventions
        if health_score.engagement_score < 40:
            interventions.append(InterventionType.TRAINING_NEEDED)

        # Support-based interventions
        if health_score.support_score < 60:
            interventions.append(InterventionType.TECHNICAL_SUPPORT)

        # Payment-based interventions
        if health_score.payment_score < 80:
            interventions.append(InterventionType.BILLING_ISSUE)

        # Expansion opportunities
        if (health_score.overall_score > 80 and
            health_score.usage_score > 75 and
            health_score.key_metrics.get('monthly_usage', 0) > 50000):
            interventions.append(InterventionType.EXPANSION_OPPORTUNITY)

        return interventions

    async def _calculate_clv_at_risk(self, tenant_id: str) -> float:
        """Calculate customer lifetime value at risk"""

        # Get tenant plan information
        config = await self.pricing_engine.get_plan_configuration(tenant_id)
        monthly_revenue = float(config.monthly_fee)

        # Simple CLV calculation (would use more sophisticated model in production)
        # Assume average customer lifetime of 24 months
        estimated_clv = monthly_revenue * 24

        return estimated_clv

    async def _store_churn_prediction(self, prediction: ChurnPrediction):
        """Store churn prediction in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO tenant_management.churn_predictions (
                    tenant_id, churn_probability, churn_risk, predicted_churn_date,
                    risk_factors, protective_factors, recommended_interventions,
                    intervention_priority, clv_at_risk, retention_value,
                    confidence_score, model_version, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
                prediction.tenant_id, prediction.churn_probability,
                prediction.churn_risk.value, prediction.predicted_churn_date,
                json.dumps(prediction.risk_factors),
                json.dumps(prediction.protective_factors),
                json.dumps([i.value for i in prediction.recommended_interventions]),
                prediction.intervention_priority, prediction.clv_at_risk,
                prediction.retention_value, prediction.confidence_score,
                prediction.model_version, datetime.now()
            )

    async def identify_expansion_opportunities(self, target_tenant_id: Optional[str] = None) -> List[ExpansionOpportunity]:
        """Identify revenue expansion opportunities"""

        tenant_id = target_tenant_id or self.tenant_id

        opportunities = []

        # Get current usage and plan
        usage_summary = await self.usage_tracker.get_usage_summary(tenant_id, "current_month")
        config = await self.pricing_engine.get_plan_configuration(tenant_id)

        # Plan upgrade opportunity
        if usage_summary['summary']['total_tokens'] > config.included_tokens * 0.8:
            upgrade_opportunity = await self._analyze_plan_upgrade_opportunity(tenant_id, usage_summary, config)
            if upgrade_opportunity:
                opportunities.append(upgrade_opportunity)

        # User seat expansion
        user_expansion = await self._analyze_user_expansion_opportunity(tenant_id, config)
        if user_expansion:
            opportunities.append(user_expansion)

        # Feature expansion
        feature_expansion = await self._analyze_feature_expansion_opportunity(tenant_id, usage_summary)
        if feature_expansion:
            opportunities.append(feature_expansion)

        return opportunities

    async def _analyze_plan_upgrade_opportunity(self, tenant_id: str, usage_summary: Dict,
                                             config: Any) -> Optional[ExpansionOpportunity]:
        """Analyze plan upgrade opportunity"""

        current_plan = config.plan_type.value
        monthly_tokens = usage_summary['summary']['total_tokens']

        # Determine recommended plan
        if current_plan == 'starter' and monthly_tokens > 80000:
            recommended_plan = 'professional'
            revenue_increase = 299 - 99  # $200 increase
        elif current_plan == 'professional' and monthly_tokens > 400000:
            recommended_plan = 'enterprise'
            revenue_increase = 999 - 299  # $700 increase
        else:
            return None

        # Calculate readiness score
        usage_growth = self._calculate_usage_growth(usage_summary)
        readiness_score = min(100, (monthly_tokens / config.included_tokens) * 100 + usage_growth)

        return ExpansionOpportunity(
            tenant_id=tenant_id,
            opportunity_type='plan_upgrade',
            current_plan=current_plan,
            recommended_plan=recommended_plan,
            estimated_revenue_increase=revenue_increase,
            usage_patterns={'monthly_tokens': monthly_tokens, 'growth': usage_growth},
            growth_indicators=['High token usage', 'Consistent growth'],
            feature_requests=[],
            readiness_score=readiness_score,
            best_approach='usage_based_upgrade',
            recommended_timing='end_of_month',
            success_probability=0.7,
            implementation_effort='low',
            expected_roi=300.0
        )

    async def _analyze_user_expansion_opportunity(self, tenant_id: str, config: Any) -> Optional[ExpansionOpportunity]:
        """Analyze user seat expansion opportunity"""

        async with self.db_pool.acquire() as conn:
            current_users = await conn.fetchval("""
                SELECT COUNT(*) FROM tenant_management.tenant_users
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

        if current_users >= config.max_users * 0.8:  # 80% of user limit
            additional_seats = max(5, current_users // 2)
            revenue_per_seat = 25  # $25 per additional user
            revenue_increase = additional_seats * revenue_per_seat

            return ExpansionOpportunity(
                tenant_id=tenant_id,
                opportunity_type='add_users',
                current_plan=config.plan_type.value,
                recommended_plan=config.plan_type.value,
                estimated_revenue_increase=revenue_increase,
                usage_patterns={'current_users': current_users, 'limit': config.max_users},
                growth_indicators=['High user utilization'],
                feature_requests=[],
                readiness_score=85.0,
                best_approach='team_growth_discussion',
                recommended_timing='immediate',
                success_probability=0.8,
                implementation_effort='low',
                expected_roi=400.0
            )

        return None

    async def _analyze_feature_expansion_opportunity(self, tenant_id: str, usage_summary: Dict) -> Optional[ExpansionOpportunity]:
        """Analyze feature expansion opportunity"""

        # Check if customer is using advanced features heavily
        advanced_usage = sum(
            service['total_tokens'] for service in usage_summary['breakdown_by_service']
            if 'gpt4' in service['service_type'] or 'claude3' in service['service_type']
        )

        if advanced_usage > 10000:  # Heavy advanced feature usage
            return ExpansionOpportunity(
                tenant_id=tenant_id,
                opportunity_type='new_features',
                current_plan='current',
                recommended_plan='enterprise_features',
                estimated_revenue_increase=200.0,
                usage_patterns={'advanced_usage': advanced_usage},
                growth_indicators=['Heavy AI model usage'],
                feature_requests=['Custom agents', 'Advanced analytics'],
                readiness_score=75.0,
                best_approach='feature_value_demonstration',
                recommended_timing='next_quarter',
                success_probability=0.6,
                implementation_effort='medium',
                expected_roi=250.0
            )

        return None

    def _calculate_usage_growth(self, usage_summary: Dict) -> float:
        """Calculate usage growth rate"""
        # Placeholder - would calculate actual growth from historical data
        return 25.0  # 25% growth

# Database schema for customer success
CUSTOMER_SUCCESS_SCHEMA_SQL = """
-- Churn predictions
CREATE TABLE IF NOT EXISTS tenant_management.churn_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    churn_probability FLOAT NOT NULL,
    churn_risk VARCHAR(50) NOT NULL,
    predicted_churn_date TIMESTAMP WITH TIME ZONE,
    risk_factors JSONB DEFAULT '[]',
    protective_factors JSONB DEFAULT '[]',
    recommended_interventions JSONB DEFAULT '[]',
    intervention_priority VARCHAR(50),
    clv_at_risk FLOAT DEFAULT 0,
    retention_value FLOAT DEFAULT 0,
    confidence_score FLOAT DEFAULT 0,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Expansion opportunities
CREATE TABLE IF NOT EXISTS tenant_management.expansion_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    opportunity_type VARCHAR(100) NOT NULL,
    current_plan VARCHAR(100),
    recommended_plan VARCHAR(100),
    estimated_revenue_increase FLOAT DEFAULT 0,
    usage_patterns JSONB DEFAULT '{}',
    growth_indicators JSONB DEFAULT '[]',
    feature_requests JSONB DEFAULT '[]',
    readiness_score FLOAT DEFAULT 0,
    best_approach VARCHAR(255),
    recommended_timing VARCHAR(100),
    success_probability FLOAT DEFAULT 0,
    implementation_effort VARCHAR(50),
    expected_roi FLOAT DEFAULT 0,
    status VARCHAR(100) DEFAULT 'identified',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customer interventions tracking
CREATE TABLE IF NOT EXISTS tenant_management.customer_interventions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_management.tenants(id) ON DELETE CASCADE,
    intervention_type VARCHAR(100) NOT NULL,
    trigger_reason VARCHAR(255),
    intervention_data JSONB DEFAULT '{}',
    status VARCHAR(100) DEFAULT 'planned',
    scheduled_at TIMESTAMP WITH TIME ZONE,
    executed_at TIMESTAMP WITH TIME ZONE,
    outcome VARCHAR(255),
    success_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_churn_predictions_tenant_risk
ON tenant_management.churn_predictions(tenant_id, churn_risk);

CREATE INDEX IF NOT EXISTS idx_expansion_opportunities_tenant_status
ON tenant_management.expansion_opportunities(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_customer_interventions_tenant_type
ON tenant_management.customer_interventions(tenant_id, intervention_type);
"""

# Pydantic models for API
class HealthScoreRequest(BaseModel):
    tenant_id: Optional[str] = Field(None, description="Target tenant ID (defaults to current)")

class ChurnPredictionRequest(BaseModel):
    tenant_id: Optional[str] = Field(None, description="Target tenant ID (defaults to current)")

class ExpansionAnalysisRequest(BaseModel):
    tenant_id: Optional[str] = Field(None, description="Target tenant ID (defaults to current)")

class InterventionRequest(BaseModel):
    tenant_id: str = Field(..., description="Target tenant ID")
    intervention_type: InterventionType = Field(..., description="Type of intervention")
    custom_message: Optional[str] = Field(None, description="Custom intervention message")

# Export main classes
__all__ = [
    'CustomerSuccessAgent', 'CustomerHealthScore', 'ChurnPrediction', 'ExpansionOpportunity',
    'HealthStatus', 'ChurnRisk', 'InterventionType', 'OnboardingStage',
    'HealthScoreRequest', 'ChurnPredictionRequest', 'ExpansionAnalysisRequest',
    'InterventionRequest', 'CUSTOMER_SUCCESS_SCHEMA_SQL'
]
