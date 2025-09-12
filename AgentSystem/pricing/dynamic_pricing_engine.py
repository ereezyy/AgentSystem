
"""
Dynamic Pricing Engine - AgentSystem Profit Machine
Advanced value-based pricing optimization system
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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import joblib
except ImportError:
    # Fallback for environments without sklearn
    pass

from ..database.connection import get_db_connection
from ..usage.usage_tracker import UsageTracker
from ..billing.stripe_service import StripeService
from ..analytics.clv_predictor import CLVPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricingStrategy(str, Enum):
    VALUE_BASED = "value_based"
    USAGE_BASED = "usage_based"
    COMPETITIVE = "competitive"
    PENETRATION = "penetration"
    PREMIUM = "premium"
    DYNAMIC = "dynamic"

class PricingTier(str, Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class PriceAdjustmentType(str, Enum):
    DISCOUNT = "discount"
    PREMIUM = "premium"
    LOYALTY = "loyalty"
    VOLUME = "volume"
    COMPETITIVE = "competitive"
    VALUE_REALIZATION = "value_realization"

@dataclass
class CustomerPricingProfile:
    customer_id: UUID
    tenant_id: UUID
    current_tier: PricingTier
    monthly_usage: float
    value_score: float
    price_sensitivity: float
    churn_risk: float
    clv_prediction: float
    competitive_position: float
    usage_growth_trend: float
    feature_adoption_score: float
    support_cost_ratio: float
    payment_reliability: float
    contract_length_preference: int  # months

@dataclass
class PricingRecommendation:
    customer_id: UUID
    tenant_id: UUID
    current_price: float
    recommended_price: float
    price_change_percent: float
    adjustment_type: PriceAdjustmentType
    strategy_used: PricingStrategy
    confidence_score: float
    expected_revenue_impact: float
    expected_churn_impact: float
    implementation_priority: float
    reasoning: List[str]
    supporting_metrics: Dict[str, float]
    effective_date: datetime
    expiry_date: Optional[datetime]

@dataclass
class MarketConditions:
    competitive_pressure: float
    market_growth_rate: float
    customer_acquisition_cost: float
    average_deal_size: float
    price_elasticity: float
    seasonal_factor: float
    economic_indicator: float

@dataclass
class PricingExperiment:
    experiment_id: UUID
    name: str
    description: str
    strategy: PricingStrategy
    target_segment: str
    test_price: float
    control_price: float
    start_date: datetime
    end_date: datetime
    sample_size: int
    success_metrics: List[str]
    results: Optional[Dict[str, float]]

class DynamicPricingEngine:
    """Advanced dynamic pricing engine based on customer value and market conditions"""

    def __init__(self):
        self.usage_tracker = UsageTracker()
        self.stripe_service = StripeService()
        self.clv_predictor = CLVPredictor()
        self.pricing_models = {}
        self.market_data = {}
        self.pricing_rules = {}

    async def initialize(self):
        """Initialize the dynamic pricing engine"""
        try:
            await self._load_pricing_models()
            await self._load_market_data()
            await self._load_pricing_rules()
            logger.info("Dynamic Pricing Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Dynamic Pricing Engine: {e}")
            raise

    async def generate_pricing_recommendation(
        self,
        tenant_id: UUID,
        customer_id: UUID,
        strategy: PricingStrategy = PricingStrategy.DYNAMIC
    ) -> PricingRecommendation:
        """Generate pricing recommendation for a customer"""
        try:
            # Extract customer pricing profile
            profile = await self._extract_customer_profile(tenant_id, customer_id)

            # Get market conditions
            market_conditions = await self._get_market_conditions(tenant_id)

            # Generate recommendation based on strategy
            if strategy == PricingStrategy.VALUE_BASED:
                recommendation = await self._value_based_pricing(profile, market_conditions)
            elif strategy == PricingStrategy.USAGE_BASED:
                recommendation = await self._usage_based_pricing(profile, market_conditions)
            elif strategy == PricingStrategy.COMPETITIVE:
                recommendation = await self._competitive_pricing(profile, market_conditions)
            elif strategy == PricingStrategy.PENETRATION:
                recommendation = await self._penetration_pricing(profile, market_conditions)
            elif strategy == PricingStrategy.PREMIUM:
                recommendation = await self._premium_pricing(profile, market_conditions)
            else:  # DYNAMIC
                recommendation = await self._dynamic_pricing(profile, market_conditions)

            # Add customer and tenant info
            recommendation.customer_id = customer_id
            recommendation.tenant_id = tenant_id
            recommendation.strategy_used = strategy
            recommendation.effective_date = datetime.utcnow()

            # Store recommendation
            await self._store_pricing_recommendation(recommendation)

            return recommendation

        except Exception as e:
            logger.error(f"Failed to generate pricing recommendation for customer {customer_id}: {e}")
            raise

    async def generate_batch_recommendations(
        self,
        tenant_id: UUID,
        customer_ids: List[UUID],
        strategy: PricingStrategy = PricingStrategy.DYNAMIC
    ) -> List[PricingRecommendation]:
        """Generate pricing recommendations for multiple customers"""
        try:
            recommendations = []

            # Process in batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(customer_ids), batch_size):
                batch = customer_ids[i:i + batch_size]

                # Extract profiles for batch
                batch_profiles = await self._extract_batch_profiles(tenant_id, batch)

                # Get market conditions once for the batch
                market_conditions = await self._get_market_conditions(tenant_id)

                # Generate recommendations for batch
                for customer_id in batch:
                    if customer_id in batch_profiles:
                        try:
                            profile = batch_profiles[customer_id]

                            if strategy == PricingStrategy.DYNAMIC:
                                rec = await self._dynamic_pricing(profile, market_conditions)
                            else:
                                rec = await self._apply_strategy_pricing(profile, market_conditions, strategy)

                            rec.customer_id = customer_id
                            rec.tenant_id = tenant_id
                            rec.strategy_used = strategy

                            recommendations.append(rec)
                        except Exception as e:
                            logger.error(f"Failed to generate recommendation for customer {customer_id}: {e}")

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate batch pricing recommendations: {e}")
            raise

    async def optimize_pricing_tiers(
        self,
        tenant_id: UUID,
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize pricing tiers based on target metrics"""
        try:
            # Get current pricing performance
            current_performance = await self._analyze_pricing_performance(tenant_id)

            # Get customer segments
            segments = await self._segment_customers_by_value(tenant_id)

            # Optimize each tier
            optimized_tiers = {}

            for tier in PricingTier:
                tier_optimization = await self._optimize_tier_pricing(
                    tenant_id, tier, segments, target_metrics
                )
                optimized_tiers[tier.value] = tier_optimization

            # Calculate expected impact
            expected_impact = await self._calculate_optimization_impact(
                tenant_id, optimized_tiers, current_performance
            )

            return {
                'current_performance': current_performance,
                'optimized_tiers': optimized_tiers,
                'expected_impact': expected_impact,
                'implementation_plan': await self._create_implementation_plan(optimized_tiers)
            }

        except Exception as e:
            logger.error(f"Failed to optimize pricing tiers: {e}")
            raise

    async def run_pricing_experiment(
        self,
        tenant_id: UUID,
        experiment: PricingExperiment
    ) -> UUID:
        """Run a pricing experiment"""
        try:
            # Validate experiment parameters
            await self._validate_experiment(experiment)

            # Select test and control groups
            test_group, control_group = await self._select_experiment_groups(
                tenant_id, experiment
            )

            # Store experiment
            experiment_id = await self._store_experiment(tenant_id, experiment, test_group, control_group)

            # Apply test pricing to test group
            await self._apply_experiment_pricing(tenant_id, experiment, test_group)

            return experiment_id

        except Exception as e:
            logger.error(f"Failed to run pricing experiment: {e}")
            raise

    async def analyze_experiment_results(
        self,
        tenant_id: UUID,
        experiment_id: UUID
    ) -> Dict[str, Any]:
        """Analyze pricing experiment results"""
        try:
            # Get experiment details
            experiment = await self._get_experiment(tenant_id, experiment_id)

            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")

            # Get test and control group data
            test_results = await self._get_experiment_group_results(
                tenant_id, experiment_id, 'test'
            )
            control_results = await self._get_experiment_group_results(
                tenant_id, experiment_id, 'control'
            )

            # Calculate statistical significance
            significance_results = await self._calculate_statistical_significance(
                test_results, control_results, experiment.success_metrics
            )

            # Generate insights and recommendations
            insights = await self._generate_experiment_insights(
                experiment, test_results, control_results, significance_results
            )

            return {
                'experiment': experiment,
                'test_results': test_results,
                'control_results': control_results,
                'significance': significance_results,
                'insights': insights,
                'recommendation': await self._get_experiment_recommendation(insights)
            }

        except Exception as e:
            logger.error(f"Failed to analyze experiment results: {e}")
            raise

    async def monitor_pricing_performance(
        self,
        tenant_id: UUID,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Monitor pricing performance and identify optimization opportunities"""
        try:
            # Get pricing metrics
            metrics = await self._get_pricing_metrics(tenant_id, days_back)

            # Identify underperforming segments
            underperforming = await self._identify_underperforming_segments(tenant_id, metrics)

            # Get pricing alerts
            alerts = await self._get_pricing_alerts(tenant_id)

            # Calculate price elasticity
            elasticity = await self._calculate_price_elasticity(tenant_id, days_back)

            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                tenant_id, metrics, underperforming, elasticity
            )

            return {
                'metrics': metrics,
                'underperforming_segments': underperforming,
                'alerts': alerts,
                'price_elasticity': elasticity,
                'recommendations': recommendations,
                'next_actions': await self._prioritize_pricing_actions(recommendations)
            }

        except Exception as e:
            logger.error(f"Failed to monitor pricing performance: {e}")
            raise

    async def _extract_customer_profile(self, tenant_id: UUID, customer_id: UUID) -> CustomerPricingProfile:
        """Extract customer pricing profile"""
        try:
            async with get_db_connection() as conn:
                # Get customer data
                customer_query = """
                    SELECT
                        t.*,
                        s.plan_id as current_tier,
                        s.current_period_start,
                        s.current_period_end
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
                        AVG(cost) as avg_cost_per_request
                    FROM usage.usage_logs
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                """
                usage_data = await conn.fetchrow(usage_query, tenant_id)

                # Get payment data
                payment_query = """
                    SELECT
                        COUNT(*) as total_payments,
                        SUM(amount) as total_revenue,
                        AVG(amount) as avg_payment,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_payments
                    FROM billing.payments
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '12 months'
                """
                payment_data = await conn.fetchrow(payment_query, tenant_id)

                # Get CLV prediction
                clv_query = """
                    SELECT predicted_clv, churn_probability
                    FROM analytics.clv_predictions
                    WHERE tenant_id = $1 AND customer_id = $2
                    ORDER BY prediction_date DESC LIMIT 1
                """
                clv_data = await conn.fetchrow(clv_query, tenant_id, customer_id)

                # Get churn prediction
                churn_query = """
                    SELECT churn_probability
                    FROM analytics.churn_predictions
                    WHERE tenant_id = $1 AND customer_id = $2
                    ORDER BY prediction_date DESC LIMIT 1
                """
                churn_data = await conn.fetchrow(churn_query, tenant_id, customer_id)

                # Calculate derived metrics
                monthly_usage = float(usage_data['total_cost'] or 0)
                total_revenue = float(payment_data['total_revenue'] or 0)
                failed_payments = int(payment_data['failed_payments'] or 0)
                total_payments = int(payment_data['total_payments'] or 1)

                # Calculate value score (0-1)
                value_score = min(1.0, total_revenue / 10000)  # Normalize to $10k max

                # Calculate price sensitivity (0-1, higher = more sensitive)
                price_sensitivity = 0.5  # Default
                if total_revenue > 0:
                    price_sensitivity = max(0.1, min(0.9, 1 - (total_revenue / 5000)))

                # Calculate payment reliability (0-1)
                payment_reliability = max(0.1, 1 - (failed_payments / total_payments))

                # Build profile
                profile = CustomerPricingProfile(
                    customer_id=customer_id,
                    tenant_id=tenant_id,
                    current_tier=PricingTier(customer_data['current_tier'] or 'starter'),
                    monthly_usage=monthly_usage,
                    value_score=value_score,
                    price_sensitivity=price_sensitivity,
                    churn_risk=float(churn_data['churn_probability'] or 0.3) if churn_data else 0.3,
                    clv_prediction=float(clv_data['predicted_clv'] or 5000) if clv_data else 5000,
                    competitive_position=0.5,  # TODO: Calculate from market data
                    usage_growth_trend=0.1,   # TODO: Calculate from usage trends
                    feature_adoption_score=0.5,  # TODO: Calculate from feature usage
                    support_cost_ratio=0.1,   # TODO: Calculate from support costs
                    payment_reliability=payment_reliability,
                    contract_length_preference=12  # Default to 12 months
                )

                return profile

        except Exception as e:
            logger.error(f"Failed to extract customer profile: {e}")
            raise

    async def _value_based_pricing(
        self,
        profile: CustomerPricingProfile,
        market: MarketConditions
    ) -> PricingRecommendation:
        """Generate value-based pricing recommendation"""
        try:
            # Calculate value-based price
            base_value = profile.clv_prediction * 0.1  # 10% of CLV as monthly price

            # Adjust for value realization
            value_adjustment = profile.value_score * 0.2  # Up to 20% adjustment

            # Adjust for feature adoption
            adoption_adjustment = profile.feature_adoption_score * 0.15  # Up to 15% adjustment

            # Calculate recommended price
            recommended_price = base_value * (1 + value_adjustment + adoption_adjustment)

            # Apply market conditions
            recommended_price *= (1 + market.competitive_pressure * 0.1)

            # Ensure minimum viable price
            recommended_price = max(recommended_price, 50.0)  # Minimum $50/month

            # Calculate price change
            current_price = await self._get_current_price(profile.tenant_id, profile.customer_id)
            price_change_percent = ((recommended_price - current_price) / current_price) * 100

            # Determine adjustment type
            if price_change_percent > 5:
                adjustment_type = PriceAdjustmentType.VALUE_REALIZATION
            elif price_change_percent < -5:
                adjustment_type = PriceAdjustmentType.DISCOUNT
            else:
                adjustment_type = PriceAdjustmentType.PREMIUM

            # Calculate confidence and impacts
            confidence_score = 0.8 if profile.value_score > 0.6 else 0.6
            revenue_impact = (recommended_price - current_price) * 12  # Annual impact
            churn_impact = self._estimate_churn_impact(price_change_percent, profile.price_sensitivity)

            return PricingRecommendation(
                customer_id=profile.customer_id,
                tenant_id=profile.tenant_id,
                current_price=current_price,
                recommended_price=recommended_price,
                price_change_percent=price_change_percent,
                adjustment_type=adjustment_type,
                strategy_used=PricingStrategy.VALUE_BASED,
                confidence_score=confidence_score,
                expected_revenue_impact=revenue_impact,
                expected_churn_impact=churn_impact,
                implementation_priority=self._calculate_priority(revenue_impact, churn_impact, confidence_score),
                reasoning=[
                    f"Customer CLV of ${profile.clv_prediction:,.0f} supports higher pricing",
                    f"Value score of {profile.value_score:.1%} indicates strong value realization",
                    f"Feature adoption of {profile.feature_adoption_score:.1%} shows engagement"
                ],
                supporting_metrics={
                    'clv_prediction': profile.clv_prediction,
                    'value_score': profile.value_score,
                    'feature_adoption': profile.feature_adoption_score,
                    'price_sensitivity': profile.price_sensitivity
                },
                effective_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=90)
            )

        except Exception as e:
            logger.error(f"Failed to generate value-based pricing: {e}")
            raise

    async def _usage_based_pricing(
        self,
        profile: CustomerPricingProfile,
        market: MarketConditions
    ) -> PricingRecommendation:
        """Generate usage-based pricing recommendation"""
        try:
            # Calculate usage-based price
            base_rate = 0.01  # $0.01 per usage unit
            usage_price = profile.monthly_usage * base_rate

            # Apply volume discounts
            if profile.monthly_usage > 10000:
                usage_price *= 0.8  # 20% volume discount
            elif profile.monthly_usage > 5000:
                usage_price *= 0.9  # 10% volume discount

            # Add base subscription fee
            base_fee = 50.0  # Minimum base fee
            recommended_price = usage_price + base_fee

            # Apply growth trend adjustment
            if profile.usage_growth_trend > 0.2:  # 20% growth
                recommended_price *= 1.1  # 10% premium for growing usage

            current_price = await self._get_current_price(profile.tenant_id, profile.customer_id)
            price_change_percent = ((recommended_price - current_price) / current_price) * 100

            return PricingRecommendation(
                customer_id=profile.customer_id,
                tenant_id=profile.tenant_id,
                current_price=current_price,
                recommended_price=recommended_price,
                price_change_percent=price_change_percent,
                adjustment_type=PriceAdjustmentType.VOLUME if profile.monthly_usage > 5000 else PriceAdjustmentType.PREMIUM,
                strategy_used=PricingStrategy.USAGE_BASED,
                confidence_score=0.9,  # High confidence for usage-based
                expected_revenue_impact=(recommended_price - current_price) * 12,
                expected_churn_impact=self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                implementation_priority=self._calculate_priority(
                    (recommended_price - current_price) * 12,
                    self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                    0.9
                ),
                reasoning=[
                    f"Monthly usage of ${profile.monthly_usage:,.0f} supports usage-based pricing",
                    f"Usage growth trend of {profile.usage_growth_trend:.1%} indicates value",
                    "Fair pricing aligned with actual consumption"
                ],
                supporting_metrics={
                    'monthly_usage': profile.monthly_usage,
                    'usage_growth_trend': profile.usage_growth_trend,
                    'volume_discount_applied': profile.monthly_usage > 5000
                },
                effective_date=datetime.utcnow(),
                expiry_date=None  # Usage-based pricing doesn't expire
            )

        except Exception as e:
            logger.error(f"Failed to generate usage-based pricing: {e}")
            raise

    async def _dynamic_pricing(
        self,
        profile: CustomerPricingProfile,
        market: MarketConditions
    ) -> PricingRecommendation:
        """Generate dynamic pricing recommendation using ML model"""
        try:
            # Get recommendations from different strategies
            value_rec = await self._value_based_pricing(profile, market)
            usage_rec = await self._usage_based_pricing(profile, market)

            # Weight recommendations based on customer characteristics
            if profile.value_score > 0.7:
                # High-value customers: favor value-based pricing
                weights = {'value': 0.7, 'usage': 0.3}
            elif profile.monthly_usage > 5000:
                # High-usage customers: favor usage-based pricing
                weights = {'value': 0.3, 'usage': 0.7}
            else:
                # Balanced approach
                weights = {'value': 0.5, 'usage': 0.5}

            # Calculate weighted price
            recommended_price = (
                value_rec.recommended_price * weights['value'] +
                usage_rec.recommended_price * weights['usage']
            )

            # Apply market adjustments
            if market.competitive_pressure > 0.7:
                recommended_price *= 0.95  # 5% discount for high competition

            if market.market_growth_rate > 0.1:
                recommended_price *= 1.05  # 5% premium for growing market

            # Apply customer-specific adjustments
            if profile.churn_risk > 0.6:
                recommended_price *= 0.9  # 10% discount for high churn risk

            if profile.payment_reliability > 0.9:
                recommended_price *= 1.02  # 2% premium for reliable payers

            current_price = await self._get_current_price(profile.tenant_id, profile.customer_id)
            price_change_percent = ((recommended_price - current_price) / current_price) * 100

            # Determine adjustment type
            adjustment_type = PriceAdjustmentType.DISCOUNT
            if price_change_percent > 10:
                adjustment_type = PriceAdjustmentType.VALUE_REALIZATION
            elif price_change_percent > 0:
                adjustment_type = PriceAdjustmentType.PREMIUM
            elif profile.churn_risk > 0.6:
                adjustment_type = PriceAdjustmentType.LOYALTY

            # Calculate confidence (average of component confidences)
            confidence_score = (value_rec.confidence_score + usage_rec.confidence_score) / 2

            return PricingRecommendation(
                customer_id=profile.customer_id,
                tenant_id=profile.tenant_id,
                current_price=current_price,
                recommended_price=recommended_price,
                price_change_percent=price_change_percent,
                adjustment_type=adjustment_type,
                strategy_used=PricingStrategy.DYNAMIC,
                confidence_score=confidence_score,
                expected_revenue_impact=(recommended_price - current_price) * 12,
                expected_churn_impact=self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                implementation_priority=self._calculate_priority(
                    (recommended_price - current_price) * 12,
                    self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                    confidence_score
                ),
                reasoning=[
                    f"Dynamic pricing combining value-based ({weights['value']:.0%}) and usage-based ({weights['usage']:.0%})",
                    f"Market conditions: competition {market.competitive_pressure:.1%}, growth {market.market_growth_rate:.1%}",
                    f"Customer risk factors: churn {profile.churn_risk:.1%}, reliability {profile.payment_reliability:.1%}"
                ],
                supporting_metrics={
                    'value_weight': weights['value'],
                    'usage_weight': weights['usage'],
                    'market_adjustment': market.competitive_pressure,
                    'churn_adjustment': profile.churn_risk,
                    'reliability_bonus': profile.payment_reliability
                },
                effective_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=30)  # Dynamic pricing expires monthly
            )

        except Exception as e:
            logger.error(f"Failed to generate dynamic pricing: {e}")
            raise

    def _estimate_churn_impact(self, price_change_percent: float, price_sensitivity: float) -> float:
        """Estimate churn impact of price change"""
        # Simple elasticity model: churn_impact = price_change * sensitivity
        base_impact = (price_change_percent / 100) * price_sensitivity

        # Apply diminishing returns for small changes
        if abs(price_change_percent) < 5:
            base_impact *= 0.5

        # Cap the impact
        return max(-0.5, min(0.5, base_impact))

    def _calculate_priority(self, revenue_impact: float, churn_impact: float, confidence: float) -> float:
        """Calculate implementation priority score"""
        # Positive revenue impact increases priority
        revenue_score = max(0, revenue_impact / 1000)  # Normalize to $1k

        # Negative churn impact (reduced churn) increases priority
        churn_score = max(0, -churn_impact * 100)

        # Confidence multiplier
        priority = (revenue_score + churn_score) * confidence

        return min(100, priority)

    async def _get_current_price(self, tenant_id: UUID, customer_id: UUID) -> float:
        """Get customer's current price"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT amount
                    FROM billing.payments
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                result = await conn.fetchrow(query, tenant_id)
                return float(result['amount']) if result else 100.0  # Default $100

        except Exception as e:
            logger.error(f"Failed to get current price: {e}")
            return 100.0  # Default fallback price

    async def _get_market_conditions(self, tenant_id: UUID) -> MarketConditions:
        """Get current market conditions"""
        try:
            # In a real implementation, this would fetch from market data APIs
            # For now, return simulated market conditions
            return MarketConditions(
                competitive_pressure=0.6,  # Moderate competition
                market_growth_rate=0.15,   # 15% market growth
                customer_acquisition_cost=500.0,
                average_deal_size=2000.0,
                price_elasticity=0.8,      # Somewhat elastic
                seasonal_factor=1.0,       # No seasonal adjustment
                economic_indicator=1.0     # Neutral economic conditions
            )
        except Exception as e:
            logger.error(f"Failed to get market conditions: {e}")
            # Return default conditions
            return MarketConditions(0.5, 0.1, 500.0, 2000.0, 0.8, 1.0, 1.0)

    async def _extract_batch_profiles(
        self,
        tenant_id: UUID,
        customer_ids: List[UUID]
    ) -> Dict[UUID, CustomerPricingProfile]:
        """Extract pricing profiles for multiple customers"""
        profiles = {}

        for customer_id in customer_ids:
            try:
                profile = await self._extract_customer_profile(tenant_id, customer_id)
                profiles[customer_id] = profile
            except Exception as e:
                logger.error(f"Failed to extract profile for customer {customer_id}: {e}")

        return profiles

    async def _apply_strategy_pricing(
        self,
        profile: CustomerPricingProfile,
        market: MarketConditions,
        strategy: PricingStrategy
    ) -> PricingRecommendation:
        """Apply specific pricing strategy"""
        if strategy == PricingStrategy.VALUE_BASED:
            return await self._value_based_pricing(profile, market)
        elif strategy == PricingStrategy.USAGE_BASED:
            return await self._usage_based_pricing(profile, market)
        elif strategy == PricingStrategy.COMPETITIVE:
            return await self._competitive_pricing(profile, market)
        elif strategy == PricingStrategy.PENETRATION:
            return await self._penetration_pricing(profile, market)
        elif strategy == PricingStrategy.PREMIUM:
            return await self._premium_pricing(profile, market)
        else:
            return await self._dynamic_pricing(profile, market)

    async def _competitive_pricing(
        self,
        profile: CustomerPricingProfile,
        market: MarketConditions
    ) -> PricingRecommendation:
        """Generate competitive pricing recommendation"""
        try:
            # Get competitor pricing (simulated)
            competitor_price = await self._get_competitor_pricing(profile.current_tier)

            # Position relative to competition
            if profile.value_score > 0.7:
                # High value customers can support premium pricing
                recommended_price = competitor_price * 1.1  # 10% premium
                adjustment_type = PriceAdjustmentType.PREMIUM
            elif profile.churn_risk > 0.6:
                # High churn risk customers need competitive pricing
                recommended_price = competitor_price * 0.95  # 5% discount
                adjustment_type = PriceAdjustmentType.COMPETITIVE
            else:
                # Match competition
                recommended_price = competitor_price
                adjustment_type = PriceAdjustmentType.COMPETITIVE

            current_price = await self._get_current_price(profile.tenant_id, profile.customer_id)
            price_change_percent = ((recommended_price - current_price) / current_price) * 100

            return PricingRecommendation(
                customer_id=profile.customer_id,
                tenant_id=profile.tenant_id,
                current_price=current_price,
                recommended_price=recommended_price,
                price_change_percent=price_change_percent,
                adjustment_type=adjustment_type,
                strategy_used=PricingStrategy.COMPETITIVE,
                confidence_score=0.7,
                expected_revenue_impact=(recommended_price - current_price) * 12,
                expected_churn_impact=self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                implementation_priority=self._calculate_priority(
                    (recommended_price - current_price) * 12,
                    self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                    0.7
                ),
                reasoning=[
                    f"Competitive benchmark: ${competitor_price:,.0f}",
                    f"Market positioning based on value score: {profile.value_score:.1%}",
                    f"Churn risk consideration: {profile.churn_risk:.1%}"
                ],
                supporting_metrics={
                    'competitor_price': competitor_price,
                    'value_score': profile.value_score,
                    'churn_risk': profile.churn_risk
                },
                effective_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=60)
            )

        except Exception as e:
            logger.error(f"Failed to generate competitive pricing: {e}")
            raise

    async def _penetration_pricing(
        self,
        profile: CustomerPricingProfile,
        market: MarketConditions
    ) -> PricingRecommendation:
        """Generate penetration pricing recommendation"""
        try:
            # Aggressive pricing to gain market share
            base_price = await self._get_current_price(profile.tenant_id, profile.customer_id)

            # Apply penetration discount (20-40% off)
            discount_rate = 0.3  # 30% default discount

            # Adjust discount based on customer characteristics
            if profile.value_score > 0.8:
                discount_rate = 0.2  # Smaller discount for high-value customers
            elif profile.churn_risk > 0.7:
                discount_rate = 0.4  # Larger discount for high-risk customers

            recommended_price = base_price * (1 - discount_rate)
            price_change_percent = -discount_rate * 100

            return PricingRecommendation(
                customer_id=profile.customer_id,
                tenant_id=profile.tenant_id,
                current_price=base_price,
                recommended_price=recommended_price,
                price_change_percent=price_change_percent,
                adjustment_type=PriceAdjustmentType.DISCOUNT,
                strategy_used=PricingStrategy.PENETRATION,
                confidence_score=0.8,
                expected_revenue_impact=(recommended_price - base_price) * 12,
                expected_churn_impact=-0.2,  # Significant churn reduction expected
                implementation_priority=80.0,  # High priority for penetration
                reasoning=[
                    f"Penetration pricing with {discount_rate:.0%} discount",
                    "Designed to gain market share and reduce churn",
                    f"Adjusted for customer value score: {profile.value_score:.1%}"
                ],
                supporting_metrics={
                    'discount_rate': discount_rate,
                    'market_share_strategy': True,
                    'churn_reduction_focus': True
                },
                effective_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=180)  # Longer term strategy
            )

        except Exception as e:
            logger.error(f"Failed to generate penetration pricing: {e}")
            raise

    async def _premium_pricing(
        self,
        profile: CustomerPricingProfile,
        market: MarketConditions
    ) -> PricingRecommendation:
        """Generate premium pricing recommendation"""
        try:
            # Premium pricing for high-value customers
            base_price = await self._get_current_price(profile.tenant_id, profile.customer_id)

            # Apply premium based on customer value
            if profile.value_score > 0.8:
                premium_rate = 0.25  # 25% premium
            elif profile.value_score > 0.6:
                premium_rate = 0.15  # 15% premium
            else:
                premium_rate = 0.05  # 5% premium

            # Adjust for payment reliability
            if profile.payment_reliability > 0.9:
                premium_rate += 0.05  # Additional 5% for reliable payers

            recommended_price = base_price * (1 + premium_rate)
            price_change_percent = premium_rate * 100

            return PricingRecommendation(
                customer_id=profile.customer_id,
                tenant_id=profile.tenant_id,
                current_price=base_price,
                recommended_price=recommended_price,
                price_change_percent=price_change_percent,
                adjustment_type=PriceAdjustmentType.PREMIUM,
                strategy_used=PricingStrategy.PREMIUM,
                confidence_score=0.6 if profile.price_sensitivity > 0.7 else 0.8,
                expected_revenue_impact=(recommended_price - base_price) * 12,
                expected_churn_impact=self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                implementation_priority=self._calculate_priority(
                    (recommended_price - base_price) * 12,
                    self._estimate_churn_impact(price_change_percent, profile.price_sensitivity),
                    0.7
                ),
                reasoning=[
                    f"Premium pricing with {premium_rate:.0%} increase",
                    f"Justified by high value score: {profile.value_score:.1%}",
                    f"Payment reliability: {profile.payment_reliability:.1%}"
                ],
                supporting_metrics={
                    'premium_rate': premium_rate,
                    'value_score': profile.value_score,
                    'payment_reliability': profile.payment_reliability,
                    'price_sensitivity': profile.price_sensitivity
                },
                effective_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=120)
            )

        except Exception as e:
            logger.error(f"Failed to generate premium pricing: {e}")
            raise

    async def _get_competitor_pricing(self, tier: PricingTier) -> float:
        """Get competitor pricing for a given tier"""
        # Simulated competitor pricing
        competitor_prices = {
            PricingTier.STARTER: 79.0,
            PricingTier.PROFESSIONAL: 199.0,
            PricingTier.ENTERPRISE: 499.0,
            PricingTier.CUSTOM: 999.0
        }
        return competitor_prices.get(tier, 199.0)

    async def _store_pricing_recommendation(self, recommendation: PricingRecommendation):
        """Store pricing recommendation in database"""
        try:
            async with get_db_connection() as conn:
                query = """
                    INSERT INTO pricing.recommendations (
                        recommendation_id, tenant_id, customer_id, current_price,
                        recommended_price, price_change_percent, adjustment_type,
                        strategy_used, confidence_score, expected_revenue_impact,
                        expected_churn_impact, implementation_priority, reasoning,
                        supporting_metrics, effective_date, expiry_date
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                    )
                """
                await conn.execute(
                    query,
                    uuid4(),
                    recommendation.tenant_id,
                    recommendation.customer_id,
                    recommendation.current_price,
                    recommendation.recommended_price,
                    recommendation.price_change_percent,
                    recommendation.adjustment_type.value,
                    recommendation.strategy_used.value,
                    recommendation.confidence_score,
                    recommendation.expected_revenue_impact,
                    recommendation.expected_churn_impact,
                    recommendation.implementation_priority,
                    json.dumps(recommendation.reasoning),
                    json.dumps(recommendation.supporting_metrics),
                    recommendation.effective_date,
                    recommendation.expiry_date
                )
        except Exception as e:
            logger.error(f"Failed to store pricing recommendation: {e}")

    async def _analyze_pricing_performance(self, tenant_id: UUID) -> Dict[str, Any]:
        """Analyze current pricing performance"""
        try:
            async with get_db_connection() as conn:
                # Get pricing metrics
                metrics_query = """
                    SELECT
                        COUNT(DISTINCT customer_id) as total_customers,
                        AVG(amount) as avg_price,
                        SUM(amount) as total_revenue,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_payments
                    FROM billing.payments
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '30 days'
                """
                metrics = await conn.fetchrow(metrics_query, tenant_id)

                return {
                    'total_customers': metrics['total_customers'] or 0,
                    'avg_price': float(metrics['avg_price'] or 0),
                    'total_revenue': float(metrics['total_revenue'] or 0),
                    'payment_success_rate': 1 - (metrics['failed_payments'] or 0) / max(1, metrics['total_customers'] or 1),
                    'revenue_per_customer': float(metrics['total_revenue'] or 0) / max(1, metrics['total_customers'] or 1)
                }

        except Exception as e:
            logger.error(f"Failed to analyze pricing performance: {e}")
            return {}

    async def _segment_customers_by_value(self, tenant_id: UUID) -> Dict[str, List[UUID]]:
        """Segment customers by value for pricing optimization"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT
                        customer_id,
                        SUM(amount) as total_revenue,
                        COUNT(*) as payment_count,
                        AVG(amount) as avg_payment
                    FROM billing.payments
                    WHERE tenant_id = $1
                    AND created_at >= NOW() - INTERVAL '12 months'
                    GROUP BY customer_id
                    ORDER BY total_revenue DESC
                """
                results = await conn.fetch(query, tenant_id)

                # Segment customers
                segments = {
                    'high_value': [],
                    'medium_value': [],
                    'low_value': []
                }

                for row in results:
                    revenue = float(row['total_revenue'])
                    if revenue > 5000:
                        segments['high_value'].append(row['customer_id'])
                    elif revenue > 1000:
                        segments['medium_value'].append(row['customer_id'])
                    else:
                        segments['low_value'].append(row['customer_id'])

                return segments

        except Exception as e:
            logger.error(f"Failed to segment customers: {e}")
            return {'high_value': [], 'medium_value': [], 'low_value': []}

    async def _optimize_tier_pricing(
        self,
        tenant_id: UUID,
        tier: PricingTier,
        segments: Dict[str, List[UUID]],
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Optimize pricing for a specific tier"""
        try:
            # Get current tier performance
            current_performance = await self._get_tier_performance(tenant_id, tier)

            # Calculate optimal price based on elasticity
            current_price = current_performance.get('avg_price', 100.0)
            elasticity = target_metrics.get('price_elasticity', -0.5)

            # Target revenue growth
            target_growth = target_metrics.get('revenue_growth', 0.2)

            # Calculate price change needed
            # Using elasticity: % change in quantity = elasticity * % change in price
            # Revenue = Price * Quantity, so we need to balance price vs. quantity

            if target_growth > 0:
                # For growth, we need to find optimal price point
                optimal_price_change = target_growth / (1 + elasticity)
                optimal_price = current_price * (1 + optimal_price_change)
            else:
                optimal_price = current_price

            return {
                'tier': tier.value,
                'current_price': current_price,
                'optimal_price': optimal_price,
                'price_change_percent': ((optimal_price - current_price) / current_price) * 100,
                'expected_impact': {
                    'revenue_change': target_growth,
                    'customer_impact': elasticity * optimal_price_change,
                    'confidence': 0.7
                }
            }

        except Exception as e:
            logger.error(f"Failed to optimize tier pricing: {e}")
            return {}

    async def _get_tier_performance(self, tenant_id: UUID, tier: PricingTier) -> Dict[str, Any]:
        """Get performance metrics for a pricing tier"""
        try:
            async with get_db_connection() as conn:
                query = """
                    SELECT
                        COUNT(DISTINCT p.customer_id) as customer_count,
                        AVG(p.amount) as avg_price,
                        SUM(p.amount) as total_revenue
                    FROM billing.payments p
                    JOIN billing.subscriptions s ON p.tenant_id = s.tenant_id
                    WHERE p.tenant_id = $1 AND s.plan_id = $2
                    AND p.created_at >= NOW() - INTERVAL '30 days'
                """
                result = await conn.fetchrow(query, tenant_id, tier.value)

                return {
                    'customer_count': result['customer_count'] or 0,
                    'avg_price': float(result['avg_price'] or 0),
                    'total_revenue': float(result['total_revenue'] or 0)
                }

        except Exception as e:
            logger.error(f"Failed to get tier performance: {e}")
            return {}

    async def _calculate_optimization_impact(
        self,
        tenant_id: UUID,
        optimized_tiers: Dict[str, Any],
        current_performance: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate expected impact of pricing optimization"""
        try:
            total_revenue_impact = 0
            total_customer_impact = 0

            for tier_name, optimization in optimized_tiers.items():
                price_change = optimization.get('price_change_percent', 0)
                current_revenue = optimization.get('current_price', 0) * 12  # Annual

                # Calculate revenue impact
                revenue_impact = current_revenue * (price_change / 100)
                total_revenue_impact += revenue_impact

                # Estimate customer impact (simplified)
                customer_impact = price_change * -0.3  # Assume -0.3 elasticity
                total_customer_impact += customer_impact

            return {
                'annual_revenue_impact': total_revenue_impact,
                'customer_retention_impact': total_customer_impact,
                'roi_estimate': total_revenue_impact / max(1000, abs(total_revenue_impact) * 0.1),  # Assuming 10% implementation cost
                'implementation_complexity': len(optimized_tiers) * 0.2  # Complexity score
            }

        except Exception as e:
            logger.error(f"Failed to calculate optimization impact: {e}")
            return {}

    async def _create_implementation_plan(self, optimized_tiers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create implementation plan for pricing optimization"""
        try:
            plan = []

            for tier_name, optimization in optimized_tiers.items():
                price_change = optimization.get('price_change_percent', 0)

                if abs(price_change) > 5:  # Only implement significant changes
                    plan.append({
                        'tier': tier_name,
                        'action': 'price_change',
                        'current_price': optimization.get('current_price'),
                        'new_price': optimization.get('optimal_price'),
                        'change_percent': price_change,
                        'priority': 'high' if abs(price_change) > 15 else 'medium',
                        'timeline': '30 days' if abs(price_change) < 10 else '60 days',
                        'risk_level': 'low' if price_change < 0 else 'medium'
                    })

            # Sort by priority and expected impact
            plan.sort(key=lambda x: (x['priority'] == 'high', abs(x['change_percent'])), reverse=True)

            return plan

        except Exception as e:
            logger.error(f"Failed to create implementation plan: {e}")
            return []

    async def _load_pricing_models(self):
        """Load pricing ML models"""
        try:
            # For now, initialize placeholder models
            self.pricing_models = {
                'elasticity_model': None,
                'value_model': None,
                'churn_impact_model': None
            }
            logger.info("Pricing models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pricing models: {e}")

    async def _load_market_data(self):
        """Load market data and competitive intelligence"""
        try:
            # For now, initialize with default market data
            self.market_data = {
                'competitor_prices': {},
                'market_trends': {},
                'economic_indicators': {}
            }
            logger.info("Market data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")

    async def _load_pricing_rules(self):
        """Load pricing business rules and constraints"""
        try:
            # Initialize default pricing rules
            self.pricing_rules = {
                'max_price_increase': 0.25,  # 25% max increase
                'max_price_decrease': 0.40,  # 40% max decrease
                'min_price': 10.0,           # $10 minimum
                'max_price': 10000.0,        # $10k maximum
                'price_change_frequency': 30 # Days between changes
            }
            logger.info("Pricing rules loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pricing rules: {e}")

    # Placeholder methods for experiment functionality
    async def _validate_experiment(self, experiment: PricingExperiment):
        """Validate experiment parameters"""
        pass

    async def _select_experiment_groups(self, tenant_id: UUID, experiment: PricingExperiment):
        """Select test and control groups for experiment"""
        return [], []

    async def _store_experiment(self, tenant_id: UUID, experiment: PricingExperiment, test_group: List[UUID], control_group: List[UUID]) -> UUID:
        """Store experiment in database"""
        return uuid4()

    async def _apply_experiment_pricing(self, tenant_id: UUID, experiment: PricingExperiment, test_group: List[UUID]):
        """Apply experimental pricing to test group"""
        pass

    async def _get_experiment(self, tenant_id: UUID, experiment_id: UUID):
        """Get experiment details"""
        return None

    async def _get_experiment_group_results(self, tenant_id: UUID, experiment_id: UUID, group_type: str):
        """Get experiment group results"""
        return {}

    async def _calculate_statistical_significance(self, test_results: Dict, control_results: Dict, metrics: List[str]):
        """Calculate statistical significance of experiment results"""
        return {}

    async def _generate_experiment_insights(self, experiment: PricingExperiment, test_results: Dict, control_results: Dict, significance: Dict):
        """Generate insights from experiment results"""
        return {}

    async def _get_experiment_recommendation(self, insights: Dict):
        """Get recommendation based on experiment insights"""
        return {}

    async def _get_pricing_metrics(self, tenant_id: UUID, days_back: int):
        """Get pricing performance metrics"""
        return {}

    async def _identify_underperforming_segments(self, tenant_id: UUID, metrics: Dict):
        """Identify underperforming customer segments"""
        return {}

    async def _get_pricing_alerts(self, tenant_id: UUID):
        """Get pricing-related alerts"""
        return []

    async def _calculate_price_elasticity(self, tenant_id: UUID, days_back: int):
        """Calculate price elasticity"""
        return {}

    async def _generate_optimization_recommendations(self, tenant_id: UUID, metrics: Dict, underperforming: Dict, elasticity: Dict):
        """Generate pricing optimization recommendations"""
        return []

    async def _prioritize_pricing_actions(self, recommendations: List):
        """Prioritize pricing actions by impact and effort"""
        return []

# Factory function
def create_dynamic_pricing_engine() -> DynamicPricingEngine:
    """Create and initialize dynamic pricing engine"""
    return DynamicPricingEngine()
