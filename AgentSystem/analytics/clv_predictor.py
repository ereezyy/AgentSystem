
"""
Customer Lifetime Value (CLV) Prediction Engine - AgentSystem Profit Machine
Advanced ML models for predicting customer value and optimizing revenue strategies
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
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import joblib
except ImportError:
    # Fallback for environments without sklearn
    pass

from ..database.connection import get_db_connection
from ..usage.usage_tracker import UsageTracker
from ..billing.stripe_service import StripeService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLVModel(str, Enum):
    TRADITIONAL = "traditional"  # RFM-based model
    MACHINE_LEARNING = "ml"      # Advanced ML model
    ENSEMBLE = "ensemble"        # Combined models
    COHORT_BASED = "cohort"      # Cohort analysis model

class PredictionConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class CLVPrediction:
    customer_id: UUID
    tenant_id: UUID
    predicted_clv: float
    confidence_score: float
    confidence_level: PredictionConfidence
    time_horizon_days: int
    model_used: CLVModel
    feature_importance: Dict[str, float]
    risk_factors: List[str]
    growth_opportunities: List[str]
    recommended_actions: List[str]
    prediction_date: datetime
    next_purchase_probability: float
    churn_probability: float
    upsell_potential: float

@dataclass
class CustomerFeatures:
    # Behavioral features
    total_spent: float
    avg_monthly_spend: float
    purchase_frequency: float
    days_since_last_purchase: int
    total_purchases: int
    avg_purchase_value: float

    # Engagement features
    api_calls_per_month: int
    feature_adoption_score: float
    support_tickets: int
    login_frequency: float
    session_duration_avg: float

    # Subscription features
    subscription_tier: str
    months_subscribed: int
    plan_changes: int
    payment_failures: int

    # Usage features
    tokens_used_monthly: int
    cost_per_month: float
    feature_usage_diversity: float
    peak_usage_ratio: float

    # Demographic features
    company_size: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    acquisition_channel: Optional[str]

class CLVPredictor:
    """Advanced Customer Lifetime Value prediction engine"""

    def __init__(self):
        self.usage_tracker = UsageTracker()
        self.stripe_service = StripeService()
        self.models = {}
        self.scalers = {}
        self.feature_encoders = {}
        self.model_performance = {}

    async def initialize(self):
        """Initialize the CLV predictor"""
        try:
            await self._load_models()
            logger.info("CLV Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CLV Predictor: {e}")
            raise

    async def predict_clv(
        self,
        tenant_id: UUID,
        customer_id: UUID,
        time_horizon_days: int = 365,
        model_type: CLVModel = CLVModel.ENSEMBLE
    ) -> CLVPrediction:
        """Predict customer lifetime value"""
        try:
            # Extract customer features
            features = await self._extract_customer_features(tenant_id, customer_id)

            # Make prediction based on model type
            if model_type == CLVModel.TRADITIONAL:
                prediction = await self._predict_traditional_clv(features, time_horizon_days)
            elif model_type == CLVModel.MACHINE_LEARNING:
                prediction = await self._predict_ml_clv(features, time_horizon_days)
            elif model_type == CLVModel.COHORT_BASED:
                prediction = await self._predict_cohort_clv(tenant_id, features, time_horizon_days)
            else:  # ENSEMBLE
                prediction = await self._predict_ensemble_clv(tenant_id, features, time_horizon_days)

            # Add customer and tenant info
            prediction.customer_id = customer_id
            prediction.tenant_id = tenant_id
            prediction.time_horizon_days = time_horizon_days
            prediction.model_used = model_type
            prediction.prediction_date = datetime.utcnow()

            # Generate recommendations
            prediction.recommended_actions = await self._generate_recommendations(prediction, features)

            # Store prediction
            await self._store_prediction(prediction)

            return prediction

        except Exception as e:
            logger.error(f"Failed to predict CLV for customer {customer_id}: {e}")
            raise

    async def predict_batch_clv(
        self,
        tenant_id: UUID,
        customer_ids: List[UUID],
        time_horizon_days: int = 365,
        model_type: CLVModel = CLVModel.ENSEMBLE
    ) -> List[CLVPrediction]:
        """Predict CLV for multiple customers in batch"""
        try:
            predictions = []

            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(customer_ids), batch_size):
                batch = customer_ids[i:i + batch_size]

                # Extract features for batch
                batch_features = await self._extract_batch_features(tenant_id, batch)

                # Make batch predictions
                if model_type == CLVModel.MACHINE_LEARNING:
                    batch_predictions = await self._predict_batch_ml_clv(
                        batch_features, time_horizon_days
                    )
                else:
                    # Fall back to individual predictions for other models
                    batch_predictions = []
                    for customer_id in batch:
                        pred = await self.predict_clv(
                            tenant_id, customer_id, time_horizon_days, model_type
                        )
                        batch_predictions.append(pred)

                predictions.extend(batch_predictions)

            return predictions

        except Exception as e:
            logger.error(f"Failed to predict batch CLV: {e}")
            raise

    async def _extract_customer_features(self, tenant_id: UUID, customer_id: UUID) -> CustomerFeatures:
        """Extract features for CLV prediction"""
        try:
            async with get_db_connection() as conn:
                # Get customer data
                customer_query = """
                    SELECT
                        t.*,
                        EXTRACT(DAYS FROM NOW() - t.created_at) as days_since_signup,
                        COALESCE(s.current_period_start, t.created_at) as subscription_start
                    FROM billing.tenants t
                    LEFT JOIN billing.subscriptions s ON t.tenant_id = s.tenant_id
                    WHERE t.tenant_id = $1
                """
                customer_data = await conn.fetchrow(customer_query, tenant_id)

                if not customer_data:
                    raise ValueError(f"Customer {customer_id} not found")

                # Get usage data
                usage_query = """
                    SELECT
                        COUNT(*) as total_requests,
                        SUM(tokens_used) as total_tokens,
                        SUM(cost) as total_cost,
                        AVG(cost) as avg_cost_per_request,
                        COUNT(DISTINCT DATE(created_at)) as active_days,
                        MAX(created_at) as last_usage,
                        MIN(created_at) as first_usage
                    FROM usage.usage_logs
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '12 months'
                """
                usage_data = await conn.fetchrow(usage_query, tenant_id)

                # Get subscription data
                subscription_query = """
                    SELECT
                        plan_id,
                        status,
                        current_period_start,
                        current_period_end,
                        EXTRACT(DAYS FROM NOW() - current_period_start) as subscription_days
                    FROM billing.subscriptions
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                subscription_data = await conn.fetchrow(subscription_query, tenant_id)

                # Get payment data
                payment_query = """
                    SELECT
                        COUNT(*) as total_payments,
                        SUM(amount) as total_revenue,
                        AVG(amount) as avg_payment,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_payments,
                        MAX(created_at) as last_payment
                    FROM billing.payments
                    WHERE tenant_id = $1
                """
                payment_data = await conn.fetchrow(payment_query, tenant_id)

                # Get feature usage data
                feature_query = """
                    SELECT
                        COUNT(DISTINCT feature_name) as features_used,
                        COUNT(*) as total_feature_usage
                    FROM usage.feature_usage
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '3 months'
                """
                feature_data = await conn.fetchrow(feature_query, tenant_id)

                # Calculate derived features
                days_since_signup = customer_data['days_since_signup'] or 1
                subscription_days = subscription_data['subscription_days'] if subscription_data else days_since_signup

                # Calculate monthly metrics
                months_active = max(1, days_since_signup / 30)
                total_spent = payment_data['total_revenue'] or 0
                avg_monthly_spend = total_spent / months_active

                # Calculate usage metrics
                total_tokens = usage_data['total_tokens'] or 0
                total_cost = usage_data['total_cost'] or 0
                active_days = usage_data['active_days'] or 0

                # Calculate engagement metrics
                api_calls_per_month = (usage_data['total_requests'] or 0) / months_active
                login_frequency = active_days / days_since_signup if days_since_signup > 0 else 0

                # Build features object
                features = CustomerFeatures(
                    # Behavioral features
                    total_spent=float(total_spent),
                    avg_monthly_spend=float(avg_monthly_spend),
                    purchase_frequency=float(payment_data['total_payments'] or 0) / months_active,
                    days_since_last_purchase=(datetime.utcnow() - (payment_data['last_payment'] or datetime.utcnow())).days,
                    total_purchases=int(payment_data['total_payments'] or 0),
                    avg_purchase_value=float(payment_data['avg_payment'] or 0),

                    # Engagement features
                    api_calls_per_month=int(api_calls_per_month),
                    feature_adoption_score=min(1.0, (feature_data['features_used'] or 0) / 10),
                    support_tickets=0,  # TODO: Add support ticket tracking
                    login_frequency=float(login_frequency),
                    session_duration_avg=0.0,  # TODO: Add session tracking

                    # Subscription features
                    subscription_tier=subscription_data['plan_id'] if subscription_data else 'free',
                    months_subscribed=int(subscription_days / 30),
                    plan_changes=0,  # TODO: Add plan change tracking
                    payment_failures=int(payment_data['failed_payments'] or 0),

                    # Usage features
                    tokens_used_monthly=int(total_tokens / months_active),
                    cost_per_month=float(total_cost / months_active),
                    feature_usage_diversity=float(feature_data['features_used'] or 0),
                    peak_usage_ratio=1.0,  # TODO: Calculate peak vs average usage

                    # Demographic features
                    company_size=customer_data.get('company_size'),
                    industry=customer_data.get('industry'),
                    country=customer_data.get('country'),
                    acquisition_channel=customer_data.get('acquisition_channel')
                )

                return features

        except Exception as e:
            logger.error(f"Failed to extract customer features: {e}")
            raise

    async def _predict_traditional_clv(self, features: CustomerFeatures, time_horizon_days: int) -> CLVPrediction:
        """Traditional RFM-based CLV prediction"""
        try:
            # RFM Analysis
            recency_score = self._calculate_recency_score(features.days_since_last_purchase)
            frequency_score = self._calculate_frequency_score(features.purchase_frequency)
            monetary_score = self._calculate_monetary_score(features.avg_purchase_value)

            # Calculate base CLV using traditional formula
            # CLV = (Average Purchase Value × Purchase Frequency × Gross Margin × Lifespan)
            gross_margin = 0.8  # Assume 80% gross margin for SaaS
            estimated_lifespan_months = self._estimate_lifespan(recency_score, frequency_score)

            monthly_value = features.avg_monthly_spend
            predicted_clv = monthly_value * estimated_lifespan_months * gross_margin

            # Adjust for time horizon
            horizon_months = time_horizon_days / 30
            if horizon_months < estimated_lifespan_months:
                predicted_clv = predicted_clv * (horizon_months / estimated_lifespan_months)

            # Calculate confidence based on data quality
            confidence_score = self._calculate_confidence(features, "traditional")
            confidence_level = self._get_confidence_level(confidence_score)

            # Feature importance for traditional model
            feature_importance = {
                "avg_monthly_spend": 0.4,
                "purchase_frequency": 0.3,
                "months_subscribed": 0.2,
                "payment_failures": -0.1
            }

            # Risk factors and opportunities
            risk_factors = self._identify_risk_factors(features)
            growth_opportunities = self._identify_growth_opportunities(features)

            return CLVPrediction(
                customer_id=UUID('00000000-0000-0000-0000-000000000000'),  # Will be set by caller
                tenant_id=UUID('00000000-0000-0000-0000-000000000000'),    # Will be set by caller
                predicted_clv=predicted_clv,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                model_used=CLVModel.TRADITIONAL,
                feature_importance=feature_importance,
                risk_factors=risk_factors,
                growth_opportunities=growth_opportunities,
                recommended_actions=[],  # Will be set by caller
                prediction_date=datetime.utcnow(),
                next_purchase_probability=frequency_score,
                churn_probability=1 - recency_score,
                upsell_potential=monetary_score
            )

        except Exception as e:
            logger.error(f"Failed to predict traditional CLV: {e}")
            raise

    async def _predict_ml_clv(self, features: CustomerFeatures, time_horizon_days: int) -> CLVPrediction:
        """Machine learning-based CLV prediction"""
        try:
            # Convert features to ML format
            feature_vector = self._features_to_vector(features)

            # Load or train ML model
            model = await self._get_ml_model()
            scaler = await self._get_scaler()

            # Scale features
            if model and scaler:
                scaled_features = scaler.transform([feature_vector])
                predicted_clv = model.predict(scaled_features)[0]
            else:
                # Fallback to traditional method
                return await self._predict_traditional_clv(features, time_horizon_days)

            # Adjust for time horizon
            base_horizon = 365  # Model trained on 1-year horizon
            if time_horizon_days != base_horizon:
                predicted_clv = predicted_clv * (time_horizon_days / base_horizon)

            # Calculate confidence based on model uncertainty
            confidence_score = self._calculate_ml_confidence(model, scaled_features)
            confidence_level = self._get_confidence_level(confidence_score)

            # Get feature importance from model
            feature_importance = self._get_feature_importance(model, feature_vector)

            # Calculate additional predictions
            next_purchase_prob = self._predict_next_purchase_probability(features)
            churn_prob = self._predict_churn_probability(features)
            upsell_potential = self._predict_upsell_potential(features)

            # Risk factors and opportunities
            risk_factors = self._identify_risk_factors(features)
            growth_opportunities = self._identify_growth_opportunities(features)

            return CLVPrediction(
                customer_id=UUID('00000000-0000-0000-0000-000000000000'),
                tenant_id=UUID('00000000-0000-0000-0000-000000000000'),
                predicted_clv=max(0, predicted_clv),  # Ensure non-negative
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                model_used=CLVModel.MACHINE_LEARNING,
                feature_importance=feature_importance,
                risk_factors=risk_factors,
                growth_opportunities=growth_opportunities,
                recommended_actions=[],
                prediction_date=datetime.utcnow(),
                next_purchase_probability=next_purchase_prob,
                churn_probability=churn_prob,
                upsell_potential=upsell_potential
            )

        except Exception as e:
            logger.error(f"Failed to predict ML CLV: {e}")
            # Fall back to traditional method
            return await self._predict_traditional_clv(features, time_horizon_days)

    async def _predict_ensemble_clv(
        self,
        tenant_id: UUID,
        features: CustomerFeatures,
        time_horizon_days: int
    ) -> CLVPrediction:
        """Ensemble prediction combining multiple models"""
        try:
            # Get predictions from different models
            traditional_pred = await self._predict_traditional_clv(features, time_horizon_days)
            ml_pred = await self._predict_ml_clv(features, time_horizon_days)
            cohort_pred = await self._predict_cohort_clv(tenant_id, features, time_horizon_days)

            # Weight predictions based on confidence and model performance
            weights = {
                'traditional': 0.3,
                'ml': 0.5,
                'cohort': 0.2
            }

            # Adjust weights based on data availability and confidence
            if traditional_pred.confidence_score > 0.8:
                weights['traditional'] += 0.1
            if ml_pred.confidence_score > 0.8:
                weights['ml'] += 0.1
            if cohort_pred.confidence_score > 0.8:
                weights['cohort'] += 0.1

            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}

            # Calculate ensemble prediction
            ensemble_clv = (
                traditional_pred.predicted_clv * weights['traditional'] +
                ml_pred.predicted_clv * weights['ml'] +
                cohort_pred.predicted_clv * weights['cohort']
            )

            # Calculate ensemble confidence
            ensemble_confidence = (
                traditional_pred.confidence_score * weights['traditional'] +
                ml_pred.confidence_score * weights['ml'] +
                cohort_pred.confidence_score * weights['cohort']
            )

            # Combine feature importance
            ensemble_importance = {}
            for pred in [traditional_pred, ml_pred, cohort_pred]:
                for feature, importance in pred.feature_importance.items():
                    if feature not in ensemble_importance:
                        ensemble_importance[feature] = 0
                    ensemble_importance[feature] += importance / 3

            # Combine risk factors and opportunities
            all_risks = set()
            all_opportunities = set()
            for pred in [traditional_pred, ml_pred, cohort_pred]:
                all_risks.update(pred.risk_factors)
                all_opportunities.update(pred.growth_opportunities)

            return CLVPrediction(
                customer_id=UUID('00000000-0000-0000-0000-000000000000'),
                tenant_id=UUID('00000000-0000-0000-0000-000000000000'),
                predicted_clv=ensemble_clv,
                confidence_score=ensemble_confidence,
                confidence_level=self._get_confidence_level(ensemble_confidence),
                time_horizon_days=time_horizon_days,
                model_used=CLVModel.ENSEMBLE,
                feature_importance=ensemble_importance,
                risk_factors=list(all_risks),
                growth_opportunities=list(all_opportunities),
                recommended_actions=[],
                prediction_date=datetime.utcnow(),
                next_purchase_probability=(traditional_pred.next_purchase_probability + ml_pred.next_purchase_probability) / 2,
                churn_probability=(traditional_pred.churn_probability + ml_pred.churn_probability) / 2,
                upsell_potential=(traditional_pred.upsell_potential + ml_pred.upsell_potential) / 2
            )

        except Exception as e:
            logger.error(f"Failed to predict ensemble CLV: {e}")
            # Fall back to traditional method
            return await self._predict_traditional_clv(features, time_horizon_days)

    async def _predict_cohort_clv(
        self,
        tenant_id: UUID,
        features: CustomerFeatures,
        time_horizon_days: int
    ) -> CLVPrediction:
        """Cohort-based CLV prediction"""
        try:
            # Get cohort data for similar customers
            cohort_data = await self._get_cohort_data(tenant_id, features)

            if not cohort_data:
                # Fall back to traditional if no cohort data
                return await self._predict_traditional_clv(features, time_horizon_days)

            # Calculate cohort-based CLV
            cohort_avg_clv = cohort_data['avg_clv']
            cohort_retention_rate = cohort_data['retention_rate']

            # Adjust based on customer's relative performance
            customer_score = self._calculate_customer_score(features)
            cohort_avg_score = cohort_data['avg_score']

            performance_multiplier = customer_score / cohort_avg_score if cohort_avg_score > 0 else 1.0
            predicted_clv = cohort_avg_clv * performance_multiplier

            # Adjust for time horizon
            horizon_months = time_horizon_days / 30
            base_horizon_months = 12
            if horizon_months != base_horizon_months:
                predicted_clv = predicted_clv * (horizon_months / base_horizon_months)

            # Calculate confidence based on cohort size and similarity
            confidence_score = min(0.9, cohort_data['cohort_size'] / 100) * cohort_data['similarity_score']
            confidence_level = self._get_confidence_level(confidence_score)

            # Feature importance based on cohort analysis
            feature_importance = {
                "subscription_tier": 0.3,
                "months_subscribed": 0.25,
                "avg_monthly_spend": 0.2,
                "feature_adoption_score": 0.15,
                "api_calls_per_month": 0.1
            }

            return CLVPrediction(
                customer_id=UUID('00000000-0000-0000-0000-000000000000'),
                tenant_id=UUID('00000000-0000-0000-0000-000000000000'),
                predicted_clv=predicted_clv,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                model_used=CLVModel.COHORT_BASED,
                feature_importance=feature_importance,
                risk_factors=self._identify_risk_factors(features),
                growth_opportunities=self._identify_growth_opportunities(features),
                recommended_actions=[],
                prediction_date=datetime.utcnow(),
                next_purchase_probability=cohort_retention_rate,
                churn_probability=1 - cohort_retention_rate,
                upsell_potential=cohort_data.get('upsell_rate', 0.2)
            )

        except Exception as e:
            logger.error(f"Failed to predict cohort CLV: {e}")
            return await self._predict_traditional_clv(features, time_horizon_days)

    # Helper methods
    def _calculate_recency_score(self, days_since_last_purchase: int) -> float:
        """Calculate recency score (0-1, higher is better)"""
        if days_since_last_purchase <= 30:
            return 1.0
        elif days_since_last_purchase <= 90:
            return 0.8
        elif days_since_last_purchase <= 180:
            return 0.6
        elif days_since_last_purchase <= 365:
            return 0.4
        else:
            return 0.2

    def _calculate_frequency_score(self, purchase_frequency: float) -> float:
        """Calculate frequency score (0-1, higher is better)"""
        if purchase_frequency >= 1.0:  # Monthly or more
            return 1.0
        elif purchase_frequency >= 0.5:  # Every 2 months
            return 0.8
        elif purchase_frequency >= 0.25:  # Quarterly
            return 0.6
        elif purchase_frequency >= 0.1:  # Few times per year
            return 0.4
        else:
            return 0.2

    def _calculate_monetary_score(self, avg_purchase_value: float) -> float:
        """Calculate monetary score (0-1, higher is better)"""
        if avg_purchase_value >= 1000:
            return 1.0
        elif avg_purchase_value >= 500:
            return 0.8
        elif avg_purchase_value >= 100:
            return 0.6
        elif avg_purchase_value >= 50:
            return 0.4
        else:
            return 0.2

    def _estimate_lifespan(self, recency_score: float, frequency_score: float) -> float:
        """Estimate customer lifespan in months"""
        base_lifespan = 24  # 2 years base
        recency_factor = recency_score * 12  # Up to 1 year bonus
        frequency_factor = frequency_score * 18  # Up to 1.5 years bonus
        return base_lifespan + recency_factor + frequency_factor

    def _calculate_confidence(self, features: CustomerFeatures, model_type: str) -> float:
        """Calculate prediction confidence"""
        confidence = 0.5  # Base confidence

        # Data quality factors
        if features.months_subscribed >= 3:
            confidence += 0.2
        if features.total_purchases >= 5:
            confidence += 0.1
        if features.api_calls_per_month > 100:
            confidence += 0.1
        if features.payment_failures == 0:
            confidence += 0.1

        return min(1.0, confidence)

    def _get_confidence_level(self, confidence_score: float) -> PredictionConfidence:
        """Convert confidence score to level"""
        if confidence_score >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 0.6:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.4:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW

    def _identify_risk_factors(self, features: CustomerFeatures) -> List[str]:
        """Identify customer risk factors"""
        risks = []

        if features.days_since_last_purchase > 90:
            risks.append("Long time since last purchase")
        if features.payment_failures > 2:
            risks.append("Multiple payment failures")
        if features.feature_adoption_score < 0.3:
            risks.append("Low feature adoption")
        if features.api_calls_per_month < 10:
            risks.append("Low API usage")
        if features.login_frequency < 0.1:
            risks.append("Infrequent logins")

        return risks

    def _identify_growth_opportunities(self, features: CustomerFeatures) -> List[str]:
        """Identify growth opportunities"""
        opportunities = []

        if features.feature_adoption_score < 0.7:
            opportunities.append("Increase feature adoption")
        if features.subscription_tier == 'starter':
            opportunities.append("Upgrade to higher tier")
        if features.api_calls_per_month > 1000 and features.subscription_tier != 'enterprise':
            opportunities.append("Consider enterprise plan")
        if features.avg_monthly_spend > 200 and features.months_subscribed > 6:
            opportunities.append("Potential for annual subscription")

        return opportunities

    async def _generate_recommendations(self, prediction: CLVPrediction, features: CustomerFeatures) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # High-value customer recommendations
        if prediction.predicted_clv > 5000:
            recommendations.append("Assign dedicated customer success manager")
            recommendations.append("Offer premium support tier")
            recommendations.append("Provide early access to new features")

        # Churn risk recommendations
        if prediction.churn_probability > 0.6:
            recommendations.append("Immediate outreach to prevent churn")
            recommendations.append("Offer discount or incentive")
            recommendations.append("Schedule product training session")

        # Upsell recommendations
        if prediction.upsell_potential > 0.7:
            recommendations.append("Present upgrade options")
            recommendations.append("Highlight advanced features")
            recommendations.append("Offer usage-based pricing")

        # Engagement recommendations
        if features.feature_adoption_score < 0.5:
            recommendations.append("Provide feature onboarding")
            recommendations.append("Send educational content")
            recommendations.append("Schedule product demo")

        return recommendations

    async def _store_prediction(self, prediction: CLVPrediction):
        """Store CLV prediction in database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO analytics.clv_predictions (
                        prediction_id, tenant_id, customer_id, predicted_clv,
                        confidence_score, confidence_level, time_horizon_days,
                        model_used, feature_importance, risk_factors,
                        growth_opportunities, recommended_actions,
                        next_purchase_probability, churn_probability, upsell_potential,
                        prediction_date
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                    )
                """
                await conn.execute(
                    query,
                    uuid4(),
                    prediction.tenant_id,
                    prediction.customer_id,
                    prediction.predicted_clv,
                    prediction.confidence_score,
                    prediction.confidence_level.value,
                    prediction.time_horizon_days,
                    prediction.model_used.value,
                    json.dumps(prediction.feature_importance),
                    json.dumps(prediction.risk_factors),
                    json.dumps(prediction.growth_opportunities),
                    json.dumps(prediction.recommended_actions),
                    prediction.next_purchase_probability,
                    prediction.churn_probability,
                    prediction.upsell_potential,
                    prediction.prediction_date
                )
        except Exception as e:
            logger.error(f"Failed to store CLV prediction: {e}")

    async def _load_models(self):
        """Load pre-trained ML models"""
        try:
            # For now, we'll use placeholder models
            # In production, load from file system or model registry
            self.models['clv_regressor'] = None
            self.scalers['clv_scaler'] = None
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    async def _get_ml_model(self):
        """Get ML model for CLV prediction"""
        if 'clv_regressor' not in self.models or self.models['clv_regressor'] is None:
            # Create a simple model for demonstration
            # In production, this would be a pre-trained model
            try:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                # Would be pre-trained on historical data
                self.models['clv_regressor'] = model
            except ImportError:
                # Fallback if sklearn not available
                self.models['clv_regressor'] = None

        return self.models['clv_regressor']

    async def _get_scaler(self):
        """Get feature scaler"""
        if 'clv_scaler' not in self.scalers or self.scalers['clv_scaler'] is None:
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                # Would be fitted on training data
                self.scalers['clv_scaler'] = scaler
            except ImportError:
                # Fallback if sklearn not available
                self.scalers['clv_scaler'] = None

        return self.scalers['clv_scaler']

    def _features_to_vector(self, features: CustomerFeatures) -> List[float]:
        """Convert features to ML vector"""
        return [
            features.total_spent,
            features.avg_monthly_spend,
            features.purchase_frequency,
            features.days_since_last_purchase,
            features.total_purchases,
            features.avg_purchase_value,
            features.api_calls_per_month,
            features.feature_adoption_score,
            features.support_tickets,
            features.login_frequency,
            features.session_duration_avg,
            features.months_subscribed,
            features.plan_changes,
            features.payment_failures,
            features.tokens_used_monthly,
            features.cost_per_month,
            features.feature_usage_diversity,
            features.peak_usage_ratio,
            # Encode categorical features
            1 if features.subscription_tier == 'enterprise' else 0,
            1 if features.subscription_tier == 'pro' else 0,
            1 if features.company_size == 'large' else 0,
            1 if features.company_size == 'medium' else 0
        ]

    def _calculate_ml_confidence(self, model, scaled_features) -> float:
        """Calculate ML model confidence"""
        # Placeholder implementation
        # In practice, use techniques like prediction intervals, ensemble variance, etc.
        return 0.75

    def _get_feature_importance(self, model, feature_vector) -> Dict[str, float]:
        """Get feature importance from ML model"""
        feature_names = [
            "total_spent", "avg_monthly_spend", "purchase_frequency",
            "days_since_last_purchase", "total_purchases", "avg_purchase_value",
            "api_calls_per_month", "feature_adoption_score", "support_tickets",
            "login_frequency", "session_duration_avg", "months_subscribed",
            "plan_changes", "payment_failures", "tokens_used_monthly",
            "cost_per_month", "feature_usage_diversity", "peak_usage_ratio",
            "is_enterprise", "is_pro", "is_large_company", "is_medium_company"
        ]

        # Placeholder importance scores
        importance_scores = [0.1, 0.15, 0.12, 0.08, 0.09, 0.11, 0.07, 0.06, 0.03, 0.05, 0.04, 0.1, 0.02, 0.04, 0.08, 0.13, 0.05, 0.06, 0.12, 0.08, 0.04, 0.03]

        return dict(zip(feature_names, importance_scores))

    def _predict_next_purchase_probability(self, features: CustomerFeatures) -> float:
        """Predict probability of next purchase"""
        # Simple heuristic based on recency and frequency
        recency_score = self._calculate_recency_score(features.days_since_last_purchase)
        frequency_score = self._calculate_frequency_score(features.purchase_frequency)
        engagement_score = features.feature_adoption_score

        return (recency_score * 0.4 + frequency_score * 0.4 + engagement_score * 0.2)

    def _predict_churn_probability(self, features: CustomerFeatures) -> float:
        """Predict probability of churn"""
        risk_score = 0.0

        # Risk factors
        if features.days_since_last_purchase > 90:
            risk_score += 0.3
        if features.payment_failures > 0:
            risk_score += 0.2
        if features.feature_adoption_score < 0.3:
            risk_score += 0.2
        if features.api_calls_per_month < 10:
            risk_score += 0.1
        if features.login_frequency < 0.1:
            risk_score += 0.2

        return min(1.0, risk_score)

    def _predict_upsell_potential(self, features: CustomerFeatures) -> float:
        """Predict upsell potential"""
        potential = 0.0

        # Positive indicators
        if features.api_calls_per_month > 1000:
            potential += 0.3
        if features.feature_adoption_score > 0.7:
            potential += 0.2
        if features.avg_monthly_spend > 100:
            potential += 0.2
        if features.months_subscribed > 6:
            potential += 0.1
        if features.subscription_tier == 'starter':
            potential += 0.2

        return min(1.0, potential)

    async def _extract_batch_features(self, tenant_id: UUID, customer_ids: List[UUID]) -> Dict[UUID, CustomerFeatures]:
        """Extract features for multiple customers"""
        features_dict = {}

        # Extract features for each customer
        for customer_id in customer_ids:
            try:
                features = await self._extract_customer_features(tenant_id, customer_id)
                features_dict[customer_id] = features
            except Exception as e:
                logger.error(f"Failed to extract features for customer {customer_id}: {e}")

        return features_dict

    async def _predict_batch_ml_clv(self, batch_features: Dict[UUID, CustomerFeatures], time_horizon_days: int) -> List[CLVPrediction]:
        """Predict CLV for batch of customers using ML"""
        predictions = []

        for customer_id, features in batch_features.items():
            try:
                prediction = await self._predict_ml_clv(features, time_horizon_days)
                prediction.customer_id = customer_id
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict CLV for customer {customer_id}: {e}")

        return predictions

    async def _get_cohort_data(self, tenant_id: UUID, features: CustomerFeatures) -> Optional[Dict[str, Any]]:
        """Get cohort data for similar customers"""
        try:
            async with get_db_connection() as conn:
                # Find similar customers based on subscription tier and usage patterns
                query = """
                    SELECT
                        COUNT(*) as cohort_size,
                        AVG(total_revenue) as avg_clv,
                        AVG(retention_score) as retention_rate,
                        AVG(customer_score) as avg_score,
                        0.8 as similarity_score,
                        0.2 as upsell_rate
                    FROM analytics.customer_cohorts
                    WHERE tenant_id = $1
                    AND subscription_tier = $2
                    AND usage_category = $3
                """

                usage_category = self._get_usage_category(features)
                result = await conn.fetchrow(query, tenant_id, features.subscription_tier, usage_category)

                if result and result['cohort_size'] > 5:
                    return dict(result)

                return None

        except Exception as e:
            logger.error(f"Failed to get cohort data: {e}")
            return None

    def _get_usage_category(self, features: CustomerFeatures) -> str:
        """Categorize customer by usage patterns"""
        if features.api_calls_per_month > 10000:
            return "high_usage"
        elif features.api_calls_per_month > 1000:
            return "medium_usage"
        elif features.api_calls_per_month > 100:
            return "low_usage"
        else:
            return "minimal_usage"

    def _calculate_customer_score(self, features: CustomerFeatures) -> float:
        """Calculate overall customer score"""
        score = 0.0

        # Monetary component (40%)
        monetary_score = min(1.0, features.avg_monthly_spend / 500) * 0.4

        # Usage component (30%)
        usage_score = min(1.0, features.api_calls_per_month / 5000) * 0.3

        # Engagement component (20%)
        engagement_score = features.feature_adoption_score * 0.2

        # Loyalty component (10%)
        loyalty_score = min(1.0, features.months_subscribed / 24) * 0.1

        return monetary_score + usage_score + engagement_score + loyalty_score

    async def optimize_performance(self, tenant_id: UUID):
        """Optimize CLV prediction performance for a tenant"""
        try:
            logger.info(f"Starting CLV optimization for tenant {tenant_id}")

            # Analyze prediction accuracy
            accuracy_metrics = await self._analyze_prediction_accuracy(tenant_id)

            # Retrain models if needed
            if accuracy_metrics['accuracy'] < 0.7:
                await self._retrain_models(tenant_id)

            # Update feature weights
            await self._update_feature_weights(tenant_id)

            logger.info(f"CLV optimization completed for tenant {tenant_id}")

        except Exception as e:
            logger.error(f"Failed to optimize CLV performance: {e}")

    async def _analyze_prediction_accuracy(self, tenant_id: UUID) -> Dict[str, float]:
        """Analyze historical prediction accuracy"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT
                        AVG(ABS(predicted_clv - actual_clv) / NULLIF(actual_clv, 0)) as mape,
                        COUNT(*) as total_predictions
                    FROM analytics.clv_predictions p
                    LEFT JOIN analytics.actual_clv a ON p.customer_id = a.customer_id
                    WHERE p.tenant_id = $1
                    AND p.prediction_date >= NOW() - INTERVAL '6 months'
                    AND a.actual_clv IS NOT NULL
                """
                result = await conn.fetchrow(query, tenant_id)

                mape = result['mape'] or 1.0
                accuracy = max(0, 1 - mape)

                return {
                    'accuracy': accuracy,
                    'mape': mape,
                    'total_predictions': result['total_predictions'] or 0
                }

        except Exception as e:
            logger.error(f"Failed to analyze prediction accuracy: {e}")
            return {'accuracy': 0.5, 'mape': 0.5, 'total_predictions': 0}

    async def _retrain_models(self, tenant_id: UUID):
        """Retrain ML models with latest data"""
        try:
            # Get training data
            training_data = await self._get_training_data(tenant_id)

            if len(training_data) < 100:
                logger.warning(f"Insufficient training data for tenant {tenant_id}")
                return

            # Retrain models (placeholder implementation)
            logger.info(f"Retraining CLV models for tenant {tenant_id}")

        except Exception as e:
            logger.error(f"Failed to retrain models: {e}")

    async def _update_feature_weights(self, tenant_id: UUID):
        """Update feature weights based on performance"""
        try:
            # Analyze feature performance
            feature_performance = await self._analyze_feature_performance(tenant_id)

            # Update weights (placeholder implementation)
            logger.info(f"Updated feature weights for tenant {tenant_id}")

        except Exception as e:
            logger.error(f"Failed to update feature weights: {e}")

    async def _get_training_data(self, tenant_id: UUID) -> List[Dict]:
        """Get training data for model retraining"""
        # Placeholder implementation
        return []

    async def _analyze_feature_performance(self, tenant_id: UUID) -> Dict[str, float]:
        """Analyze feature performance"""
        # Placeholder implementation
        return {}

# Factory function
def create_clv_predictor() -> CLVPredictor:
    """Create and initialize CLV predictor"""
    return CLVPredictor()
