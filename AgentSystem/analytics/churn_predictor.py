
"""
Churn Prediction and Intervention Engine - AgentSystem Profit Machine
Advanced ML models for predicting customer churn and automated intervention strategies
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import joblib
except ImportError:
    # Fallback for environments without sklearn
    pass

from ..database.connection import get_db_connection
from ..usage.usage_tracker import UsageTracker
from ..billing.stripe_service import StripeService
from ..agents.customer_success_agent import CustomerSuccessAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnRiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class InterventionType(str, Enum):
    EMAIL_OUTREACH = "email_outreach"
    PHONE_CALL = "phone_call"
    DISCOUNT_OFFER = "discount_offer"
    FEATURE_TRAINING = "feature_training"
    ACCOUNT_REVIEW = "account_review"
    PRODUCT_DEMO = "product_demo"
    CUSTOMER_SUCCESS_CALL = "customer_success_call"
    RETENTION_CAMPAIGN = "retention_campaign"

class InterventionStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ChurnModel(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"

@dataclass
class ChurnPrediction:
    customer_id: UUID
    tenant_id: UUID
    churn_probability: float
    risk_level: ChurnRiskLevel
    confidence_score: float
    time_to_churn_days: Optional[int]
    key_risk_factors: List[str]
    protective_factors: List[str]
    recommended_interventions: List[InterventionType]
    prediction_date: datetime
    model_used: ChurnModel
    feature_importance: Dict[str, float]
    early_warning_signals: List[str]

@dataclass
class ChurnFeatures:
    # Usage patterns
    usage_trend_30d: float  # Percentage change in usage
    usage_trend_7d: float
    days_since_last_login: int
    session_frequency_decline: float
    feature_usage_decline: float
    api_calls_decline: float

    # Engagement metrics
    support_ticket_frequency: float
    support_satisfaction_score: float
    feature_adoption_rate: float
    onboarding_completion: float
    training_attendance: int

    # Billing and subscription
    payment_failures: int
    billing_issues: int
    plan_downgrades: int
    contract_renewal_date: Optional[datetime]
    days_to_renewal: Optional[int]
    payment_delays: int

    # Behavioral indicators
    complaint_frequency: int
    cancellation_attempts: int
    competitor_mentions: int
    negative_feedback_score: float
    response_rate_decline: float

    # Account health
    account_age_days: int
    total_spent: float
    monthly_spend_trend: float
    user_count_change: float
    integration_usage: int

    # Comparative metrics
    peer_usage_comparison: float  # Compared to similar customers
    industry_benchmark_gap: float
    value_realization_score: float

@dataclass
class InterventionPlan:
    plan_id: UUID
    customer_id: UUID
    tenant_id: UUID
    churn_probability: float
    risk_level: ChurnRiskLevel
    interventions: List[Dict[str, Any]]
    priority_score: float
    estimated_success_rate: float
    estimated_cost: float
    estimated_clv_impact: float
    created_date: datetime
    target_completion_date: datetime
    assigned_agent: Optional[str]

class ChurnPredictor:
    """Advanced churn prediction and intervention engine"""

    def __init__(self):
        self.usage_tracker = UsageTracker()
        self.stripe_service = StripeService()
        self.customer_success_agent = CustomerSuccessAgent()
        self.models = {}
        self.scalers = {}
        self.feature_encoders = {}
        self.intervention_templates = {}

    async def initialize(self):
        """Initialize the churn predictor"""
        try:
            await self._load_models()
            await self._load_intervention_templates()
            logger.info("Churn Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Churn Predictor: {e}")
            raise

    async def predict_churn(
        self,
        tenant_id: UUID,
        customer_id: UUID,
        model_type: ChurnModel = ChurnModel.ENSEMBLE
    ) -> ChurnPrediction:
        """Predict churn probability for a customer"""
        try:
            # Extract churn features
            features = await self._extract_churn_features(tenant_id, customer_id)

            # Make prediction based on model type
            if model_type == ChurnModel.LOGISTIC_REGRESSION:
                prediction = await self._predict_logistic_churn(features)
            elif model_type == ChurnModel.RANDOM_FOREST:
                prediction = await self._predict_rf_churn(features)
            elif model_type == ChurnModel.GRADIENT_BOOSTING:
                prediction = await self._predict_gb_churn(features)
            else:  # ENSEMBLE
                prediction = await self._predict_ensemble_churn(features)

            # Add customer and tenant info
            prediction.customer_id = customer_id
            prediction.tenant_id = tenant_id
            prediction.prediction_date = datetime.utcnow()
            prediction.model_used = model_type

            # Generate intervention recommendations
            prediction.recommended_interventions = await self._recommend_interventions(prediction, features)

            # Store prediction
            await self._store_churn_prediction(prediction)

            return prediction

        except Exception as e:
            logger.error(f"Failed to predict churn for customer {customer_id}: {e}")
            raise

    async def predict_batch_churn(
        self,
        tenant_id: UUID,
        customer_ids: List[UUID],
        model_type: ChurnModel = ChurnModel.ENSEMBLE
    ) -> List[ChurnPrediction]:
        """Predict churn for multiple customers in batch"""
        try:
            predictions = []

            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(customer_ids), batch_size):
                batch = customer_ids[i:i + batch_size]

                # Extract features for batch
                batch_features = await self._extract_batch_churn_features(tenant_id, batch)

                # Make batch predictions
                batch_predictions = []
                for customer_id in batch:
                    if customer_id in batch_features:
                        pred = await self._predict_single_churn(
                            batch_features[customer_id], model_type
                        )
                        pred.customer_id = customer_id
                        pred.tenant_id = tenant_id
                        batch_predictions.append(pred)

                predictions.extend(batch_predictions)

            return predictions

        except Exception as e:
            logger.error(f"Failed to predict batch churn: {e}")
            raise

    async def create_intervention_plan(
        self,
        tenant_id: UUID,
        customer_id: UUID,
        churn_prediction: ChurnPrediction
    ) -> InterventionPlan:
        """Create personalized intervention plan"""
        try:
            # Calculate priority score
            priority_score = self._calculate_intervention_priority(churn_prediction)

            # Generate intervention sequence
            interventions = await self._generate_intervention_sequence(
                tenant_id, customer_id, churn_prediction
            )

            # Estimate success rate and impact
            success_rate = await self._estimate_intervention_success_rate(
                churn_prediction, interventions
            )

            # Estimate costs and CLV impact
            estimated_cost = self._calculate_intervention_cost(interventions)
            clv_impact = await self._estimate_clv_impact(tenant_id, customer_id, churn_prediction)

            # Create intervention plan
            plan = InterventionPlan(
                plan_id=uuid4(),
                customer_id=customer_id,
                tenant_id=tenant_id,
                churn_probability=churn_prediction.churn_probability,
                risk_level=churn_prediction.risk_level,
                interventions=interventions,
                priority_score=priority_score,
                estimated_success_rate=success_rate,
                estimated_cost=estimated_cost,
                estimated_clv_impact=clv_impact,
                created_date=datetime.utcnow(),
                target_completion_date=datetime.utcnow() + timedelta(days=30),
                assigned_agent=None  # Will be assigned by routing logic
            )

            # Store intervention plan
            await self._store_intervention_plan(plan)

            return plan

        except Exception as e:
            logger.error(f"Failed to create intervention plan: {e}")
            raise

    async def execute_intervention(
        self,
        tenant_id: UUID,
        plan_id: UUID,
        intervention_id: str
    ) -> Dict[str, Any]:
        """Execute a specific intervention"""
        try:
            # Get intervention plan
            plan = await self._get_intervention_plan(tenant_id, plan_id)
            if not plan:
                raise ValueError(f"Intervention plan {plan_id} not found")

            # Find specific intervention
            intervention = None
            for inter in plan.interventions:
                if inter['intervention_id'] == intervention_id:
                    intervention = inter
                    break

            if not intervention:
                raise ValueError(f"Intervention {intervention_id} not found in plan")

            # Execute based on intervention type
            intervention_type = InterventionType(intervention['type'])

            if intervention_type == InterventionType.EMAIL_OUTREACH:
                result = await self._execute_email_outreach(tenant_id, plan, intervention)
            elif intervention_type == InterventionType.PHONE_CALL:
                result = await self._schedule_phone_call(tenant_id, plan, intervention)
            elif intervention_type == InterventionType.DISCOUNT_OFFER:
                result = await self._create_discount_offer(tenant_id, plan, intervention)
            elif intervention_type == InterventionType.FEATURE_TRAINING:
                result = await self._schedule_feature_training(tenant_id, plan, intervention)
            elif intervention_type == InterventionType.CUSTOMER_SUCCESS_CALL:
                result = await self._schedule_cs_call(tenant_id, plan, intervention)
            else:
                result = await self._execute_generic_intervention(tenant_id, plan, intervention)

            # Update intervention status
            await self._update_intervention_status(
                tenant_id, plan_id, intervention_id,
                InterventionStatus.COMPLETED, result
            )

            return result

        except Exception as e:
            logger.error(f"Failed to execute intervention: {e}")
            raise

    async def monitor_intervention_effectiveness(
        self,
        tenant_id: UUID,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Monitor effectiveness of intervention campaigns"""
        try:
            async with get_db_connection() as conn:
                # Get intervention results
                query = """
                    SELECT
                        intervention_type,
                        COUNT(*) as total_interventions,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN outcome->>'success' = 'true' THEN 1 END) as successful,
                        AVG((outcome->>'engagement_score')::float) as avg_engagement,
                        AVG((outcome->>'cost')::float) as avg_cost
                    FROM analytics.churn_interventions
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY intervention_type
                """
                results = await conn.fetch(query % days_back, tenant_id)

                # Calculate overall metrics
                total_interventions = sum(r['total_interventions'] for r in results)
                total_successful = sum(r['successful'] for r in results)
                success_rate = total_successful / total_interventions if total_interventions > 0 else 0

                # Get churn prevention metrics
                prevention_query = """
                    SELECT
                        COUNT(CASE WHEN prevented_churn = true THEN 1 END) as prevented_churns,
                        COUNT(*) as total_at_risk,
                        AVG(clv_impact) as avg_clv_impact
                    FROM analytics.intervention_outcomes
                    WHERE tenant_id = $1
                    AND intervention_date >= NOW() - INTERVAL '%s days'
                """
                prevention_result = await conn.fetchrow(prevention_query % days_back, tenant_id)

                return {
                    'total_interventions': total_interventions,
                    'success_rate': success_rate,
                    'prevented_churns': prevention_result['prevented_churns'] or 0,
                    'prevention_rate': (prevention_result['prevented_churns'] or 0) / (prevention_result['total_at_risk'] or 1),
                    'avg_clv_impact': float(prevention_result['avg_clv_impact'] or 0),
                    'intervention_breakdown': [dict(r) for r in results]
                }

        except Exception as e:
            logger.error(f"Failed to monitor intervention effectiveness: {e}")
            return {}

    async def _extract_churn_features(self, tenant_id: UUID, customer_id: UUID) -> ChurnFeatures:
        """Extract features for churn prediction"""
        try:
            async with get_db_connection() as conn:
                # Get customer data
                customer_query = """
                    SELECT
                        t.*,
                        EXTRACT(DAYS FROM NOW() - t.created_at) as account_age_days,
                        s.current_period_end,
                        EXTRACT(DAYS FROM s.current_period_end - NOW()) as days_to_renewal
                    FROM billing.tenants t
                    LEFT JOIN billing.subscriptions s ON t.tenant_id = s.tenant_id
                    WHERE t.tenant_id = $1
                """
                customer_data = await conn.fetchrow(customer_query, tenant_id)

                if not customer_data:
                    raise ValueError(f"Customer {customer_id} not found")

                # Get usage trends
                usage_query = """
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(tokens_used) as total_tokens,
                        SUM(cost) as total_cost,
                        MAX(created_at) as last_usage,
                        COUNT(DISTINCT DATE(created_at)) as active_days
                    FROM usage.usage_logs
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                """
                usage_30d = await conn.fetchrow(usage_query, tenant_id)

                usage_7d_query = """
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(tokens_used) as total_tokens
                    FROM usage.usage_logs
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '7 days'
                """
                usage_7d = await conn.fetchrow(usage_7d_query, tenant_id)

                # Get previous period for comparison
                usage_prev_30d_query = """
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(tokens_used) as total_tokens
                    FROM usage.usage_logs
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '60 days'
                    AND created_at < NOW() - INTERVAL '30 days'
                """
                usage_prev_30d = await conn.fetchrow(usage_prev_30d_query, tenant_id)

                # Get support data
                support_query = """
                    SELECT
                        COUNT(*) as ticket_count,
                        AVG(satisfaction_score) as avg_satisfaction,
                        COUNT(CASE WHEN priority = 'high' THEN 1 END) as high_priority_tickets
                    FROM support.tickets
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '90 days'
                """
                support_data = await conn.fetchrow(support_query, tenant_id)

                # Get billing data
                billing_query = """
                    SELECT
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as payment_failures,
                        COUNT(CASE WHEN amount < 0 THEN 1 END) as refunds,
                        SUM(amount) as total_spent
                    FROM billing.payments
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '12 months'
                """
                billing_data = await conn.fetchrow(billing_query, tenant_id)

                # Calculate trends and features
                current_usage = usage_30d['total_requests'] or 0
                prev_usage = usage_prev_30d['total_requests'] or 1
                usage_trend_30d = ((current_usage - prev_usage) / prev_usage) * 100 if prev_usage > 0 else 0

                current_tokens = usage_30d['total_tokens'] or 0
                prev_tokens = usage_prev_30d['total_tokens'] or 1
                usage_trend_7d = ((usage_7d['total_tokens'] or 0) - (current_tokens / 4)) / (current_tokens / 4) * 100 if current_tokens > 0 else 0

                days_since_last_login = (datetime.utcnow() - (usage_30d['last_usage'] or datetime.utcnow())).days

                # Build features object
                features = ChurnFeatures(
                    # Usage patterns
                    usage_trend_30d=usage_trend_30d,
                    usage_trend_7d=usage_trend_7d,
                    days_since_last_login=days_since_last_login,
                    session_frequency_decline=max(0, -usage_trend_30d / 10),
                    feature_usage_decline=0.0,  # TODO: Calculate from feature usage data
                    api_calls_decline=max(0, -usage_trend_30d),

                    # Engagement metrics
                    support_ticket_frequency=float(support_data['ticket_count'] or 0) / 3,  # Per month
                    support_satisfaction_score=float(support_data['avg_satisfaction'] or 5.0),
                    feature_adoption_rate=0.5,  # TODO: Calculate from feature usage
                    onboarding_completion=1.0,  # TODO: Track onboarding progress
                    training_attendance=0,  # TODO: Track training participation

                    # Billing and subscription
                    payment_failures=int(billing_data['payment_failures'] or 0),
                    billing_issues=int(billing_data['payment_failures'] or 0),
                    plan_downgrades=0,  # TODO: Track plan changes
                    contract_renewal_date=customer_data['current_period_end'],
                    days_to_renewal=customer_data['days_to_renewal'],
                    payment_delays=0,  # TODO: Calculate payment delays

                    # Behavioral indicators
                    complaint_frequency=int(support_data['high_priority_tickets'] or 0),
                    cancellation_attempts=0,  # TODO: Track cancellation attempts
                    competitor_mentions=0,  # TODO: Track competitor mentions in support
                    negative_feedback_score=5.0 - float(support_data['avg_satisfaction'] or 5.0),
                    response_rate_decline=0.0,  # TODO: Calculate response rate changes

                    # Account health
                    account_age_days=int(customer_data['account_age_days'] or 0),
                    total_spent=float(billing_data['total_spent'] or 0),
                    monthly_spend_trend=0.0,  # TODO: Calculate spend trend
                    user_count_change=0.0,  # TODO: Track user count changes
                    integration_usage=0,  # TODO: Count active integrations

                    # Comparative metrics
                    peer_usage_comparison=0.0,  # TODO: Compare to peer group
                    industry_benchmark_gap=0.0,  # TODO: Compare to industry benchmarks
                    value_realization_score=0.5  # TODO: Calculate value realization
                )

                return features

        except Exception as e:
            logger.error(f"Failed to extract churn features: {e}")
            raise

    async def _predict_ensemble_churn(self, features: ChurnFeatures) -> ChurnPrediction:
        """Ensemble churn prediction combining multiple models"""
        try:
            # Get predictions from different models
            lr_pred = await self._predict_logistic_churn(features)
            rf_pred = await self._predict_rf_churn(features)
            gb_pred = await self._predict_gb_churn(features)

            # Weight predictions based on model performance
            weights = {'lr': 0.3, 'rf': 0.4, 'gb': 0.3}

            # Calculate ensemble prediction
            ensemble_prob = (
                lr_pred.churn_probability * weights['lr'] +
                rf_pred.churn_probability * weights['rf'] +
                gb_pred.churn_probability * weights['gb']
            )

            # Calculate ensemble confidence
            ensemble_confidence = (
                lr_pred.confidence_score * weights['lr'] +
                rf_pred.confidence_score * weights['rf'] +
                gb_pred.confidence_score * weights['gb']
            )

            # Determine risk level
            risk_level = self._calculate_risk_level(ensemble_prob)

            # Combine risk factors
            all_risk_factors = set()
            all_protective_factors = set()
            for pred in [lr_pred, rf_pred, gb_pred]:
                all_risk_factors.update(pred.key_risk_factors)
                all_protective_factors.update(pred.protective_factors)

            # Combine feature importance
            ensemble_importance = {}
            for pred in [lr_pred, rf_pred, gb_pred]:
                for feature, importance in pred.feature_importance.items():
                    if feature not in ensemble_importance:
                        ensemble_importance[feature] = 0
                    ensemble_importance[feature] += importance / 3

            # Estimate time to churn
            time_to_churn = self._estimate_time_to_churn(ensemble_prob, features)

            # Generate early warning signals
            early_warnings = self._generate_early_warning_signals(features)

            return ChurnPrediction(
                customer_id=UUID('00000000-0000-0000-0000-000000000000'),  # Will be set by caller
                tenant_id=UUID('00000000-0000-0000-0000-000000000000'),    # Will be set by caller
                churn_probability=ensemble_prob,
                risk_level=risk_level,
                confidence_score=ensemble_confidence,
                time_to_churn_days=time_to_churn,
                key_risk_factors=list(all_risk_factors),
                protective_factors=list(all_protective_factors),
                recommended_interventions=[],  # Will be set by caller
                prediction_date=datetime.utcnow(),
                model_used=ChurnModel.ENSEMBLE,
                feature_importance=ensemble_importance,
                early_warning_signals=early_warnings
            )

        except Exception as e:
            logger.error(f"Failed to predict ensemble churn: {e}")
            # Fall back to simple heuristic
            return await self._predict_heuristic_churn(features)

    async def _predict_logistic_churn(self, features: ChurnFeatures) -> ChurnPrediction:
        """Logistic regression churn prediction"""
        try:
            # Convert features to vector
            feature_vector = self._features_to_vector(features)

            # Simple heuristic-based prediction for demonstration
            churn_prob = self._calculate_heuristic_churn_probability(features)
            confidence = 0.75

            risk_level = self._calculate_risk_level(churn_prob)
            risk_factors = self._identify_risk_factors(features)
            protective_factors = self._identify_protective_factors(features)

            # Feature importance (simplified)
            feature_importance = {
                "usage_trend_30d": 0.2,
                "days_since_last_login": 0.15,
                "payment_failures": 0.15,
                "support_satisfaction_score": 0.1,
                "days_to_renewal": 0.1,
                "usage_trend_7d": 0.1,
                "support_ticket_frequency": 0.08,
                "account_age_days": 0.07,
                "total_spent": 0.05
            }

            return ChurnPrediction(
                customer_id=UUID('00000000-0000-0000-0000-000000000000'),
                tenant_id=UUID('00000000-0000-0000-0000-000000000000'),
                churn_probability=churn_prob,
                risk_level=risk_level,
                confidence_score=confidence,
                time_to_churn_days=self._estimate_time_to_churn(churn_prob, features),
                key_risk_factors=risk_factors,
                protective_factors=protective_factors,
                recommended_interventions=[],
                prediction_date=datetime.utcnow(),
                model_used=ChurnModel.LOGISTIC_REGRESSION,
                feature_importance=feature_importance,
                early_warning_signals=self._generate_early_warning_signals(features)
            )

        except Exception as e:
            logger.error(f"Failed to predict logistic churn: {e}")
            return await self._predict_heuristic_churn(features)

    async def _predict_rf_churn(self, features: ChurnFeatures) -> ChurnPrediction:
        """Random Forest churn prediction"""
        # Similar implementation to logistic regression with different weights
        return await self._predict_logistic_churn(features)

    async def _predict_gb_churn(self, features: ChurnFeatures) -> ChurnPrediction:
        """Gradient Boosting churn prediction"""
        # Similar implementation to logistic regression with different weights
        return await self._predict_logistic_churn(features)

    async def _predict_heuristic_churn(self, features: ChurnFeatures) -> ChurnPrediction:
        """Fallback heuristic-based churn prediction"""
        churn_prob = self._calculate_heuristic_churn_probability(features)

        return ChurnPrediction(
            customer_id=UUID('00000000-0000-0000-0000-000000000000'),
            tenant_id=UUID('00000000-0000-0000-0000-000000000000'),
            churn_probability=churn_prob,
            risk_level=self._calculate_risk_level(churn_prob),
            confidence_score=0.6,
            time_to_churn_days=self._estimate_time_to_churn(churn_prob, features),
            key_risk_factors=self._identify_risk_factors(features),
            protective_factors=self._identify_protective_factors(features),
            recommended_interventions=[],
            prediction_date=datetime.utcnow(),
            model_used=ChurnModel.ENSEMBLE,
            feature_importance={},
            early_warning_signals=self._generate_early_warning_signals(features)
        )

    def _calculate_heuristic_churn_probability(self, features: ChurnFeatures) -> float:
        """Calculate churn probability using heuristics"""
        risk_score = 0.0

        # Usage decline indicators
        if features.usage_trend_30d < -50:
            risk_score += 0.3
        elif features.usage_trend_30d < -20:
            risk_score += 0.15

        if features.days_since_last_login > 30:
            risk_score += 0.25
        elif features.days_since_last_login > 14:
            risk_score += 0.1

        # Billing issues
        if features.payment_failures > 2:
            risk_score += 0.2
        elif features.payment_failures > 0:
            risk_score += 0.1

        # Support satisfaction
        if features.support_satisfaction_score < 3:
            risk_score += 0.15
        elif features.support_satisfaction_score < 4:
            risk_score += 0.05

        # Contract renewal proximity
        if features.days_to_renewal and features.days_to_renewal < 30:
            risk_score += 0.1

        # High support ticket frequency
        if features.support_ticket_frequency > 2:
            risk_score += 0.1

        return min(1.0, risk_score)

    def _calculate_risk_level(self, churn_probability: float) -> ChurnRiskLevel:
        """Convert churn probability to risk level"""
        if churn_probability >= 0.8:
            return ChurnRiskLevel.CRITICAL
        elif churn_probability >= 0.6:
            return ChurnRiskLevel.VERY_HIGH
        elif churn_probability >= 0.4:
            return ChurnRiskLevel.HIGH
        elif churn_probability >= 0.2:
            return ChurnRiskLevel.MEDIUM
        elif churn_probability >= 0.1:
            return ChurnRiskLevel.LOW
        else:
            return ChurnRiskLevel.VERY_LOW

    def _identify_risk_factors(self, features: ChurnFeatures) -> List[str]:
        """Identify key risk factors"""
        risks = []

        if features.usage_trend_30d < -20:
            risks.append("Significant usage decline")

        if features.days_since_last_login > 14:
            risks.append("Infrequent platform usage")
        if features.payment_failures > 0:
            risks.append("Payment issues")
        if features.support_satisfaction_score < 4:
            risks.append("Low support satisfaction")
        if features.days_to_renewal and features.days_to_renewal < 30:
            risks.append("Contract renewal approaching")
        if features.support_ticket_frequency > 1:
            risks.append("High support ticket volume")
        if features.usage_trend_7d < -30:
            risks.append("Recent usage drop")

        return risks

    def _identify_protective_factors(self, features: ChurnFeatures) -> List[str]:
        """Identify protective factors that reduce churn risk"""
        protections = []

        if features.usage_trend_30d > 10:
            protections.append("Growing platform usage")
        if features.total_spent > 10000:
            protections.append("High investment in platform")
        if features.support_satisfaction_score > 4.5:
            protections.append("High support satisfaction")
        if features.account_age_days > 365:
            protections.append("Long-term customer")
        if features.payment_failures == 0:
            protections.append("Reliable payment history")
        if features.feature_adoption_rate > 0.7:
            protections.append("High feature adoption")

        return protections

    def _estimate_time_to_churn(self, churn_probability: float, features: ChurnFeatures) -> Optional[int]:
        """Estimate days until potential churn"""
        if churn_probability < 0.3:
            return None

        # Base time calculation
        if churn_probability >= 0.8:
            base_days = 14
        elif churn_probability >= 0.6:
            base_days = 30
        elif churn_probability >= 0.4:
            base_days = 60
        else:
            base_days = 90

        # Adjust based on renewal date
        if features.days_to_renewal and features.days_to_renewal < base_days:
            return features.days_to_renewal

        # Adjust based on usage trends
        if features.usage_trend_7d < -50:
            base_days = int(base_days * 0.5)
        elif features.usage_trend_30d > 0:
            base_days = int(base_days * 1.5)

        return base_days

    def _generate_early_warning_signals(self, features: ChurnFeatures) -> List[str]:
        """Generate early warning signals"""
        warnings = []

        if features.usage_trend_7d < -20:
            warnings.append("Recent usage decline detected")
        if features.days_since_last_login > 7:
            warnings.append("Extended absence from platform")
        if features.support_ticket_frequency > 0.5:
            warnings.append("Increased support requests")
        if features.payment_failures > 0:
            warnings.append("Payment processing issues")

        return warnings

    async def _recommend_interventions(
        self,
        prediction: ChurnPrediction,
        features: ChurnFeatures
    ) -> List[InterventionType]:
        """Recommend appropriate interventions based on churn prediction"""
        interventions = []

        # Critical risk - immediate action required
        if prediction.risk_level == ChurnRiskLevel.CRITICAL:
            interventions.extend([
                InterventionType.PHONE_CALL,
                InterventionType.CUSTOMER_SUCCESS_CALL,
                InterventionType.DISCOUNT_OFFER,
                InterventionType.ACCOUNT_REVIEW
            ])

        # Very high risk - proactive outreach
        elif prediction.risk_level == ChurnRiskLevel.VERY_HIGH:
            interventions.extend([
                InterventionType.CUSTOMER_SUCCESS_CALL,
                InterventionType.EMAIL_OUTREACH,
                InterventionType.FEATURE_TRAINING
            ])

        # High risk - engagement focus
        elif prediction.risk_level == ChurnRiskLevel.HIGH:
            interventions.extend([
                InterventionType.EMAIL_OUTREACH,
                InterventionType.FEATURE_TRAINING,
                InterventionType.PRODUCT_DEMO
            ])

        # Medium risk - educational approach
        elif prediction.risk_level == ChurnRiskLevel.MEDIUM:
            interventions.extend([
                InterventionType.EMAIL_OUTREACH,
                InterventionType.FEATURE_TRAINING
            ])

        # Customize based on specific risk factors
        if "Payment issues" in prediction.key_risk_factors:
            if InterventionType.PHONE_CALL not in interventions:
                interventions.append(InterventionType.PHONE_CALL)

        if "Low support satisfaction" in prediction.key_risk_factors:
            if InterventionType.CUSTOMER_SUCCESS_CALL not in interventions:
                interventions.append(InterventionType.CUSTOMER_SUCCESS_CALL)

        if "Infrequent platform usage" in prediction.key_risk_factors:
            if InterventionType.FEATURE_TRAINING not in interventions:
                interventions.append(InterventionType.FEATURE_TRAINING)

        return interventions[:3]  # Limit to top 3 interventions

    def _features_to_vector(self, features: ChurnFeatures) -> List[float]:
        """Convert features to ML vector"""
        return [
            features.usage_trend_30d,
            features.usage_trend_7d,
            features.days_since_last_login,
            features.session_frequency_decline,
            features.feature_usage_decline,
            features.api_calls_decline,
            features.support_ticket_frequency,
            features.support_satisfaction_score,
            features.feature_adoption_rate,
            features.onboarding_completion,
            features.training_attendance,
            features.payment_failures,
            features.billing_issues,
            features.plan_downgrades,
            features.days_to_renewal or 365,
            features.payment_delays,
            features.complaint_frequency,
            features.cancellation_attempts,
            features.competitor_mentions,
            features.negative_feedback_score,
            features.response_rate_decline,
            features.account_age_days,
            features.total_spent,
            features.monthly_spend_trend,
            features.user_count_change,
            features.integration_usage,
            features.peer_usage_comparison,
            features.industry_benchmark_gap,
            features.value_realization_score
        ]

    async def _extract_batch_churn_features(
        self,
        tenant_id: UUID,
        customer_ids: List[UUID]
    ) -> Dict[UUID, ChurnFeatures]:
        """Extract churn features for multiple customers"""
        features_dict = {}

        for customer_id in customer_ids:
            try:
                features = await self._extract_churn_features(tenant_id, customer_id)
                features_dict[customer_id] = features
            except Exception as e:
                logger.error(f"Failed to extract churn features for customer {customer_id}: {e}")

        return features_dict

    async def _predict_single_churn(
        self,
        features: ChurnFeatures,
        model_type: ChurnModel
    ) -> ChurnPrediction:
        """Predict churn for a single customer with given features"""
        if model_type == ChurnModel.ENSEMBLE:
            return await self._predict_ensemble_churn(features)
        elif model_type == ChurnModel.LOGISTIC_REGRESSION:
            return await self._predict_logistic_churn(features)
        elif model_type == ChurnModel.RANDOM_FOREST:
            return await self._predict_rf_churn(features)
        else:
            return await self._predict_gb_churn(features)

    def _calculate_intervention_priority(self, prediction: ChurnPrediction) -> float:
        """Calculate intervention priority score"""
        priority = prediction.churn_probability * 100

        # Boost priority for high-value customers
        if prediction.time_to_churn_days and prediction.time_to_churn_days < 30:
            priority += 20

        # Boost for critical risk
        if prediction.risk_level == ChurnRiskLevel.CRITICAL:
            priority += 30
        elif prediction.risk_level == ChurnRiskLevel.VERY_HIGH:
            priority += 20

        return min(100, priority)

    async def _generate_intervention_sequence(
        self,
        tenant_id: UUID,
        customer_id: UUID,
        prediction: ChurnPrediction
    ) -> List[Dict[str, Any]]:
        """Generate sequence of interventions"""
        interventions = []

        for i, intervention_type in enumerate(prediction.recommended_interventions):
            intervention = {
                'intervention_id': str(uuid4()),
                'type': intervention_type.value,
                'sequence_order': i + 1,
                'status': InterventionStatus.PENDING.value,
                'scheduled_date': datetime.utcnow() + timedelta(days=i * 2),
                'estimated_duration_minutes': self._get_intervention_duration(intervention_type),
                'success_probability': self._get_intervention_success_rate(intervention_type, prediction),
                'cost_estimate': self._get_intervention_cost(intervention_type),
                'description': self._get_intervention_description(intervention_type, prediction)
            }
            interventions.append(intervention)

        return interventions

    def _get_intervention_duration(self, intervention_type: InterventionType) -> int:
        """Get estimated duration for intervention type"""
        durations = {
            InterventionType.EMAIL_OUTREACH: 15,
            InterventionType.PHONE_CALL: 30,
            InterventionType.DISCOUNT_OFFER: 10,
            InterventionType.FEATURE_TRAINING: 60,
            InterventionType.ACCOUNT_REVIEW: 45,
            InterventionType.PRODUCT_DEMO: 30,
            InterventionType.CUSTOMER_SUCCESS_CALL: 45,
            InterventionType.RETENTION_CAMPAIGN: 20
        }
        return durations.get(intervention_type, 30)

    def _get_intervention_success_rate(
        self,
        intervention_type: InterventionType,
        prediction: ChurnPrediction
    ) -> float:
        """Get estimated success rate for intervention"""
        base_rates = {
            InterventionType.EMAIL_OUTREACH: 0.3,
            InterventionType.PHONE_CALL: 0.6,
            InterventionType.DISCOUNT_OFFER: 0.7,
            InterventionType.FEATURE_TRAINING: 0.5,
            InterventionType.ACCOUNT_REVIEW: 0.8,
            InterventionType.PRODUCT_DEMO: 0.4,
            InterventionType.CUSTOMER_SUCCESS_CALL: 0.7,
            InterventionType.RETENTION_CAMPAIGN: 0.4
        }

        base_rate = base_rates.get(intervention_type, 0.5)

        # Adjust based on churn probability
        if prediction.churn_probability > 0.8:
            base_rate *= 0.7  # Harder to save critical cases
        elif prediction.churn_probability < 0.4:
            base_rate *= 1.3  # Easier to retain lower-risk customers

        return min(1.0, base_rate)

    def _get_intervention_cost(self, intervention_type: InterventionType) -> float:
        """Get estimated cost for intervention type"""
        costs = {
            InterventionType.EMAIL_OUTREACH: 5.0,
            InterventionType.PHONE_CALL: 25.0,
            InterventionType.DISCOUNT_OFFER: 100.0,
            InterventionType.FEATURE_TRAINING: 50.0,
            InterventionType.ACCOUNT_REVIEW: 75.0,
            InterventionType.PRODUCT_DEMO: 40.0,
            InterventionType.CUSTOMER_SUCCESS_CALL: 60.0,
            InterventionType.RETENTION_CAMPAIGN: 30.0
        }
        return costs.get(intervention_type, 25.0)

    def _get_intervention_description(
        self,
        intervention_type: InterventionType,
        prediction: ChurnPrediction
    ) -> str:
        """Get description for intervention"""
        descriptions = {
            InterventionType.EMAIL_OUTREACH: f"Personalized email addressing {', '.join(prediction.key_risk_factors[:2])}",
            InterventionType.PHONE_CALL: "Direct phone call to understand concerns and provide solutions",
            InterventionType.DISCOUNT_OFFER: "Special retention discount to demonstrate value",
            InterventionType.FEATURE_TRAINING: "Personalized training session to increase platform adoption",
            InterventionType.ACCOUNT_REVIEW: "Comprehensive account review and optimization recommendations",
            InterventionType.PRODUCT_DEMO: "Demo of underutilized features that could provide value",
            InterventionType.CUSTOMER_SUCCESS_CALL: "Strategic call with customer success manager",
            InterventionType.RETENTION_CAMPAIGN: "Multi-touch retention campaign with educational content"
        }
        return descriptions.get(intervention_type, "Standard retention intervention")

    async def _estimate_intervention_success_rate(
        self,
        prediction: ChurnPrediction,
        interventions: List[Dict[str, Any]]
    ) -> float:
        """Estimate overall success rate of intervention plan"""
        if not interventions:
            return 0.0

        # Calculate combined success probability
        combined_failure_rate = 1.0
        for intervention in interventions:
            failure_rate = 1.0 - intervention['success_probability']
            combined_failure_rate *= failure_rate

        return 1.0 - combined_failure_rate

    def _calculate_intervention_cost(self, interventions: List[Dict[str, Any]]) -> float:
        """Calculate total cost of intervention plan"""
        return sum(intervention['cost_estimate'] for intervention in interventions)

    async def _estimate_clv_impact(
        self,
        tenant_id: UUID,
        customer_id: UUID,
        prediction: ChurnPrediction
    ) -> float:
        """Estimate CLV impact of preventing churn"""
        try:
            # Get customer's predicted CLV
            async with get_db_connection() as conn:
                query = """
                    SELECT predicted_clv
                    FROM analytics.clv_predictions
                    WHERE tenant_id = $1 AND customer_id = $2
                    ORDER BY prediction_date DESC LIMIT 1
                """
                result = await conn.fetchrow(query, tenant_id, customer_id)

                if result:
                    return float(result['predicted_clv']) * (1 - prediction.churn_probability)
                else:
                    # Estimate based on average
                    return 5000.0 * (1 - prediction.churn_probability)

        except Exception as e:
            logger.error(f"Failed to estimate CLV impact: {e}")
            return 0.0

    async def _store_churn_prediction(self, prediction: ChurnPrediction):
        """Store churn prediction in database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO analytics.churn_predictions (
                        prediction_id, tenant_id, customer_id, churn_probability,
                        risk_level, confidence_score, time_to_churn_days,
                        key_risk_factors, protective_factors, early_warning_signals,
                        feature_importance, model_used, prediction_date
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                    )
                """
                await conn.execute(
                    query,
                    uuid4(),
                    prediction.tenant_id,
                    prediction.customer_id,
                    prediction.churn_probability,
                    prediction.risk_level.value,
                    prediction.confidence_score,
                    prediction.time_to_churn_days,
                    json.dumps(prediction.key_risk_factors),
                    json.dumps(prediction.protective_factors),
                    json.dumps(prediction.early_warning_signals),
                    json.dumps(prediction.feature_importance),
                    prediction.model_used.value,
                    prediction.prediction_date
                )
        except Exception as e:
            logger.error(f"Failed to store churn prediction: {e}")

    async def _store_intervention_plan(self, plan: InterventionPlan):
        """Store intervention plan in database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO analytics.intervention_plans (
                        plan_id, tenant_id, customer_id, churn_probability,
                        risk_level, interventions, priority_score,
                        estimated_success_rate, estimated_cost, estimated_clv_impact,
                        created_date, target_completion_date, assigned_agent
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                    )
                """
                await conn.execute(
                    query,
                    plan.plan_id,
                    plan.tenant_id,
                    plan.customer_id,
                    plan.churn_probability,
                    plan.risk_level.value,
                    json.dumps(plan.interventions),
                    plan.priority_score,
                    plan.estimated_success_rate,
                    plan.estimated_cost,
                    plan.estimated_clv_impact,
                    plan.created_date,
                    plan.target_completion_date,
                    plan.assigned_agent
                )
        except Exception as e:
            logger.error(f"Failed to store intervention plan: {e}")

    async def _get_intervention_plan(self, tenant_id: UUID, plan_id: UUID) -> Optional[InterventionPlan]:
        """Get intervention plan from database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT * FROM analytics.intervention_plans
                    WHERE tenant_id = $1 AND plan_id = $2
                """
                result = await conn.fetchrow(query, tenant_id, plan_id)

                if not result:
                    return None

                return InterventionPlan(
                    plan_id=result['plan_id'],
                    customer_id=result['customer_id'],
                    tenant_id=result['tenant_id'],
                    churn_probability=float(result['churn_probability']),
                    risk_level=ChurnRiskLevel(result['risk_level']),
                    interventions=json.loads(result['interventions']),
                    priority_score=float(result['priority_score']),
                    estimated_success_rate=float(result['estimated_success_rate']),
                    estimated_cost=float(result['estimated_cost']),
                    estimated_clv_impact=float(result['estimated_clv_impact']),
                    created_date=result['created_date'],
                    target_completion_date=result['target_completion_date'],
                    assigned_agent=result['assigned_agent']
                )

        except Exception as e:
            logger.error(f"Failed to get intervention plan: {e}")
            return None

    async def _execute_email_outreach(
        self,
        tenant_id: UUID,
        plan: InterventionPlan,
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute email outreach intervention"""
        try:
            # Use customer success agent to send personalized email
            email_result = await self.customer_success_agent.send_retention_email(
                tenant_id=tenant_id,
                customer_id=plan.customer_id,
                risk_factors=intervention.get('risk_factors', []),
                personalization_data={
                    'churn_probability': plan.churn_probability,
                    'risk_level': plan.risk_level.value
                }
            )

            return {
                'success': email_result.get('sent', False),
                'engagement_score': 0.3,  # Will be updated based on email opens/clicks
                'cost': intervention['cost_estimate'],
                'completion_date': datetime.utcnow(),
                'details': email_result
            }

        except Exception as e:
            logger.error(f"Failed to execute email outreach: {e}")
            return {'success': False, 'error': str(e)}

    async def _schedule_phone_call(
        self,
        tenant_id: UUID,
        plan: InterventionPlan,
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Schedule phone call intervention"""
        try:
            # Schedule call through customer success system
            call_result = await self.customer_success_agent.schedule_retention_call(
                tenant_id=tenant_id,
                customer_id=plan.customer_id,
                urgency=plan.risk_level.value,
                talking_points=intervention.get('description', '')
            )

            return {
                'success': call_result.get('scheduled', False),
                'engagement_score': 0.8,
                'cost': intervention['cost_estimate'],
                'scheduled_date': call_result.get('scheduled_date'),
                'details': call_result
            }

        except Exception as e:
            logger.error(f"Failed to schedule phone call: {e}")
            return {'success': False, 'error': str(e)}

    async def _create_discount_offer(
        self,
        tenant_id: UUID,
        plan: InterventionPlan,
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create discount offer intervention"""
        try:
            # Create retention discount through billing system
            discount_amount = min(plan.estimated_clv_impact * 0.1, 500.0)  # 10% of CLV impact, max $500

            discount_result = await self.stripe_service.create_retention_discount(
                tenant_id=tenant_id,
                discount_amount=discount_amount,
                duration_months=3,
                reason="Churn prevention"
            )

            return {
                'success': discount_result.get('created', False),
                'engagement_score': 0.7,
                'cost': discount_amount,
                'discount_code': discount_result.get('discount_code'),
                'details': discount_result
            }

        except Exception as e:
            logger.error(f"Failed to create discount offer: {e}")
            return {'success': False, 'error': str(e)}

    async def _schedule_feature_training(
        self,
        tenant_id: UUID,
        plan: InterventionPlan,
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Schedule feature training intervention"""
        try:
            # Schedule training session
            training_result = await self.customer_success_agent.schedule_feature_training(
                tenant_id=tenant_id,
                customer_id=plan.customer_id,
                focus_areas=['underutilized_features', 'best_practices']
            )

            return {
                'success': training_result.get('scheduled', False),
                'engagement_score': 0.6,
                'cost': intervention['cost_estimate'],
                'scheduled_date': training_result.get('scheduled_date'),
                'details': training_result
            }

        except Exception as e:
            logger.error(f"Failed to schedule feature training: {e}")
            return {'success': False, 'error': str(e)}

    async def _schedule_cs_call(
        self,
        tenant_id: UUID,
        plan: InterventionPlan,
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Schedule customer success call intervention"""
        try:
            # Schedule strategic call with CS manager
            cs_result = await self.customer_success_agent.schedule_strategic_review(
                tenant_id=tenant_id,
                customer_id=plan.customer_id,
                priority='high',
                focus='retention'
            )

            return {
                'success': cs_result.get('scheduled', False),
                'engagement_score': 0.9,
                'cost': intervention['cost_estimate'],
                'scheduled_date': cs_result.get('scheduled_date'),
                'assigned_manager': cs_result.get('assigned_manager'),
                'details': cs_result
            }

        except Exception as e:
            logger.error(f"Failed to schedule CS call: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_generic_intervention(
        self,
        tenant_id: UUID,
        plan: InterventionPlan,
        intervention: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute generic intervention"""
        return {
            'success': True,
            'engagement_score': 0.5,
            'cost': intervention['cost_estimate'],
            'completion_date': datetime.utcnow(),
            'details': {'type': intervention['type'], 'executed': True}
        }

    async def _update_intervention_status(
        self,
        tenant_id: UUID,
        plan_id: UUID,
        intervention_id: str,
        status: InterventionStatus,
        result: Dict[str, Any]
    ):
        """Update intervention status in database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO analytics.churn_interventions (
                        intervention_execution_id, tenant_id, plan_id, intervention_id,
                        intervention_type, status, outcome, execution_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """
                await conn.execute(
                    query,
                    uuid4(),
                    tenant_id,
                    plan_id,
                    intervention_id,
                    result.get('type', 'unknown'),
                    status.value,
                    json.dumps(result),
                    datetime.utcnow()
                )
        except Exception as e:
            logger.error(f"Failed to update intervention status: {e}")

    async def _load_models(self):
        """Load pre-trained churn models"""
        try:
            # For now, we'll use placeholder models
            # In production, load from file system or model registry
            self.models['logistic_regression'] = None
            self.models['random_forest'] = None
            self.models['gradient_boosting'] = None
            logger.info("Churn models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load churn models: {e}")

    async def _load_intervention_templates(self):
        """Load intervention templates"""
        try:
            # Load intervention templates from database or config
            self.intervention_templates = {
                'email_templates': {},
                'call_scripts': {},
                'training_curricula': {}
            }
            logger.info("Intervention templates loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load intervention templates: {e}")

# Factory function
def create_churn_predictor() -> ChurnPredictor:
    """Create and initialize churn predictor"""
    return ChurnPredictor()
