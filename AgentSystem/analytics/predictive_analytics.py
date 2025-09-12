"""
Predictive Analytics System
AI-powered insights for performance prediction, cost optimization, and failure prevention
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import get_logger
from ..monitoring.realtime_dashboard import dashboard_service

logger = get_logger(__name__)

@dataclass
class Prediction:
    """Prediction result"""
    metric_name: str
    predicted_value: float
    confidence: float
    time_horizon: int  # seconds into future
    timestamp: float
    model_used: str
    features_used: List[str]

@dataclass
class Anomaly:
    """Detected anomaly"""
    metric_name: str
    timestamp: float
    actual_value: float
    expected_value: float
    anomaly_score: float
    severity: str  # "low", "medium", "high"
    description: str

@dataclass
class Insight:
    """Generated insight"""
    id: str
    category: str  # "performance", "cost", "failure", "optimization"
    title: str
    description: str
    impact: str  # "low", "medium", "high"
    confidence: float
    recommendations: List[str]
    timestamp: float
    data_points: Dict[str, Any]

class PredictiveModel:
    """Base class for predictive models"""

    def __init__(self, name: str, model_type: str = "regression"):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training = None
        self.feature_names = []
        self.performance_metrics = {}

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for training/prediction"""
        # Extract time-based features
        data['hour'] = pd.to_datetime(data['timestamp'], unit='s').dt.hour
        data['day_of_week'] = pd.to_datetime(data['timestamp'], unit='s').dt.dayofweek
        data['minute'] = pd.to_datetime(data['timestamp'], unit='s').dt.minute

        # Calculate rolling statistics
        for window in [5, 15, 30]:
            data[f'rolling_mean_{window}'] = data['value'].rolling(window=window, min_periods=1).mean()
            data[f'rolling_std_{window}'] = data['value'].rolling(window=window, min_periods=1).std().fillna(0)

        # Calculate differences and trends
        data['diff_1'] = data['value'].diff().fillna(0)
        data['diff_5'] = data['value'].diff(5).fillna(0)

        # Select feature columns
        feature_columns = [
            'hour', 'day_of_week', 'minute',
            'rolling_mean_5', 'rolling_mean_15', 'rolling_mean_30',
            'rolling_std_5', 'rolling_std_15', 'rolling_std_30',
            'diff_1', 'diff_5'
        ]

        self.feature_names = feature_columns
        return data[feature_columns].fillna(0).values

    def train(self, data: pd.DataFrame, target_column: str = 'value') -> Dict[str, float]:
        """Train the model"""
        if len(data) < 50:  # Need minimum data points
            return {"error": "Insufficient data for training"}

        try:
            # Prepare features and target
            X = self.prepare_features(data)
            y = data[target_column].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model based on type
            if self.model_type == "regression":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == "anomaly":
                self.model = IsolationForest(contamination=0.1, random_state=42)

            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            if self.model_type == "regression":
                y_pred = self.model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                self.performance_metrics = {
                    "mae": mae,
                    "r2_score": r2,
                    "training_samples": len(X_train)
                }

            self.is_trained = True
            self.last_training = datetime.now()

            return self.performance_metrics

        except Exception as e:
            logger.error(f"Model training error for {self.name}: {e}")
            return {"error": str(e)}

    def predict(self, data: pd.DataFrame, steps_ahead: int = 1) -> Optional[Prediction]:
        """Make prediction"""
        if not self.is_trained:
            return None

        try:
            # Prepare features for latest data point
            X = self.prepare_features(data)
            X_scaled = self.scaler.transform(X[-1:])

            # Make prediction
            if self.model_type == "regression":
                pred_value = self.model.predict(X_scaled)[0]

                # Calculate confidence based on model performance
                confidence = max(0.1, min(0.95, self.performance_metrics.get("r2_score", 0.5)))

                return Prediction(
                    metric_name=self.name,
                    predicted_value=pred_value,
                    confidence=confidence,
                    time_horizon=steps_ahead * 300,  # 5 minutes per step
                    timestamp=datetime.now().timestamp(),
                    model_used="RandomForest",
                    features_used=self.feature_names
                )

        except Exception as e:
            logger.error(f"Prediction error for {self.name}: {e}")
            return None

    def detect_anomaly(self, data: pd.DataFrame) -> Optional[Anomaly]:
        """Detect anomalies in data"""
        if not self.is_trained or self.model_type != "anomaly":
            return None

        try:
            X = self.prepare_features(data)
            X_scaled = self.scaler.transform(X[-1:])

            # Get anomaly score
            anomaly_score = self.model.decision_function(X_scaled)[0]
            is_anomaly = self.model.predict(X_scaled)[0] == -1

            if is_anomaly:
                latest_value = data['value'].iloc[-1]
                expected_value = data['value'].rolling(window=10).mean().iloc[-1]

                # Determine severity
                severity = "low"
                if abs(anomaly_score) > 0.5:
                    severity = "high"
                elif abs(anomaly_score) > 0.3:
                    severity = "medium"

                return Anomaly(
                    metric_name=self.name,
                    timestamp=data['timestamp'].iloc[-1],
                    actual_value=latest_value,
                    expected_value=expected_value,
                    anomaly_score=anomaly_score,
                    severity=severity,
                    description=f"Anomalous {self.name} detected: {latest_value:.2f} (expected: {expected_value:.2f})"
                )

        except Exception as e:
            logger.error(f"Anomaly detection error for {self.name}: {e}")

        return None

class AnalyticsEngine:
    """Main analytics engine"""

    def __init__(self):
        self.models: Dict[str, PredictiveModel] = {}
        self.insights_history: List[Insight] = []
        self.anomalies_history: List[Anomaly] = []
        self.predictions_history: List[Prediction] = []

        # Initialize models for key metrics
        self._initialize_models()

        # Start background analytics
        asyncio.create_task(self._analytics_loop())

    def _initialize_models(self):
        """Initialize predictive models for key metrics"""
        metrics_to_model = [
            "cpu_usage", "memory_usage", "response_time",
            "active_agents", "streaming_sessions", "error_rate"
        ]

        for metric in metrics_to_model:
            # Regression model for prediction
            self.models[f"{metric}_predictor"] = PredictiveModel(
                name=metric, model_type="regression"
            )

            # Anomaly detection model
            self.models[f"{metric}_anomaly"] = PredictiveModel(
                name=metric, model_type="anomaly"
            )

    async def _analytics_loop(self):
        """Background analytics processing"""
        while True:
            try:
                await self._update_models()
                await self._generate_predictions()
                await self._detect_anomalies()
                await self._generate_insights()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Analytics loop error: {e}")
                await asyncio.sleep(60)

    async def _update_models(self):
        """Update models with latest data"""
        for model_name, model in self.models.items():
            if "_predictor" in model_name or "_anomaly" in model_name:
                metric_name = model_name.replace("_predictor", "").replace("_anomaly", "")

                # Get historical data
                data_points = dashboard_service.metrics_collector.get_metrics_data(
                    metric_name, time_range=86400  # 24 hours
                )

                if len(data_points) > 50:
                    # Convert to DataFrame
                    df = pd.DataFrame([
                        {"timestamp": p.timestamp, "value": p.value}
                        for p in data_points
                    ])

                    # Train model if not trained recently
                    if (not model.is_trained or
                        (model.last_training and
                         datetime.now() - model.last_training > timedelta(hours=6))):

                        performance = model.train(df)
                        logger.info(f"Updated model {model_name}: {performance}")

    async def _generate_predictions(self):
        """Generate predictions for key metrics"""
        for model_name, model in self.models.items():
            if "_predictor" in model_name and model.is_trained:
                metric_name = model_name.replace("_predictor", "")

                # Get recent data
                data_points = dashboard_service.metrics_collector.get_metrics_data(
                    metric_name, time_range=3600  # 1 hour
                )

                if len(data_points) > 10:
                    df = pd.DataFrame([
                        {"timestamp": p.timestamp, "value": p.value}
                        for p in data_points
                    ])

                    # Generate predictions for next 1, 3, and 6 hours
                    for steps in [12, 36, 72]:  # 5-minute intervals
                        prediction = model.predict(df, steps_ahead=steps)
                        if prediction:
                            self.predictions_history.append(prediction)

    async def _detect_anomalies(self):
        """Detect anomalies in metrics"""
        for model_name, model in self.models.items():
            if "_anomaly" in model_name and model.is_trained:
                metric_name = model_name.replace("_anomaly", "")

                # Get recent data
                data_points = dashboard_service.metrics_collector.get_metrics_data(
                    metric_name, time_range=1800  # 30 minutes
                )

                if len(data_points) > 5:
                    df = pd.DataFrame([
                        {"timestamp": p.timestamp, "value": p.value}
                        for p in data_points
                    ])

                    anomaly = model.detect_anomaly(df)
                    if anomaly:
                        self.anomalies_history.append(anomaly)
                        logger.warning(f"Anomaly detected: {anomaly.description}")

    async def _generate_insights(self):
        """Generate AI-powered insights"""
        current_time = datetime.now().timestamp()

        # Performance insights
        await self._generate_performance_insights()

        # Cost optimization insights
        await self._generate_cost_insights()

        # Failure prediction insights
        await self._generate_failure_insights()

        # Usage pattern insights
        await self._generate_usage_insights()

    async def _generate_performance_insights(self):
        """Generate performance-related insights"""
        # Analyze response time trends
        response_time_data = dashboard_service.metrics_collector.get_metrics_data(
            "response_time", time_range=3600
        )

        if len(response_time_data) > 20:
            values = [p.value for p in response_time_data]
            avg_response_time = np.mean(values)
            trend = np.polyfit(range(len(values)), values, 1)[0]

            if avg_response_time > 2.0:
                insight = Insight(
                    id=f"perf_response_time_{int(datetime.now().timestamp())}",
                    category="performance",
                    title="High Response Time Detected",
                    description=f"Average response time is {avg_response_time:.2f}s, above optimal threshold",
                    impact="high" if avg_response_time > 5.0 else "medium",
                    confidence=0.85,
                    recommendations=[
                        "Consider scaling up server resources",
                        "Optimize database queries",
                        "Implement caching strategies",
                        "Review AI provider response times"
                    ],
                    timestamp=datetime.now().timestamp(),
                    data_points={"avg_response_time": avg_response_time, "trend": trend}
                )
                self.insights_history.append(insight)

    async def _generate_cost_insights(self):
        """Generate cost optimization insights"""
        # Analyze AI provider usage patterns
        ai_requests = dashboard_service.metrics_collector.get_metrics_data(
            "ai_requests", time_range=86400
        )

        if len(ai_requests) > 10:
            total_requests = sum(p.value for p in ai_requests)

            # Estimate costs (placeholder logic)
            estimated_daily_cost = total_requests * 0.002  # $0.002 per request estimate

            if estimated_daily_cost > 50:  # $50 daily threshold
                insight = Insight(
                    id=f"cost_optimization_{int(datetime.now().timestamp())}",
                    category="cost",
                    title="High AI Usage Cost Detected",
                    description=f"Estimated daily AI cost: ${estimated_daily_cost:.2f}",
                    impact="medium",
                    confidence=0.75,
                    recommendations=[
                        "Implement request caching to reduce redundant calls",
                        "Use cheaper models for simple tasks",
                        "Implement request batching",
                        "Set up usage alerts and limits"
                    ],
                    timestamp=datetime.now().timestamp(),
                    data_points={"daily_cost": estimated_daily_cost, "total_requests": total_requests}
                )
                self.insights_history.append(insight)

    async def _generate_failure_insights(self):
        """Generate failure prediction insights"""
        # Analyze system resource trends
        cpu_data = dashboard_service.metrics_collector.get_metrics_data("cpu_usage", time_range=3600)
        memory_data = dashboard_service.metrics_collector.get_metrics_data("memory_usage", time_range=3600)

        if len(cpu_data) > 10 and len(memory_data) > 10:
            cpu_trend = np.polyfit(range(len(cpu_data)), [p.value for p in cpu_data], 1)[0]
            memory_trend = np.polyfit(range(len(memory_data)), [p.value for p in memory_data], 1)[0]

            current_cpu = cpu_data[-1].value
            current_memory = memory_data[-1].value

            # Predict resource exhaustion
            if cpu_trend > 0.1 and current_cpu > 70:
                hours_to_exhaustion = (95 - current_cpu) / (cpu_trend * 12)  # 12 data points per hour

                insight = Insight(
                    id=f"failure_cpu_{int(datetime.now().timestamp())}",
                    category="failure",
                    title="CPU Exhaustion Risk",
                    description=f"CPU usage trending upward. Estimated {hours_to_exhaustion:.1f} hours to critical level",
                    impact="high",
                    confidence=0.80,
                    recommendations=[
                        "Scale up CPU resources immediately",
                        "Optimize high-CPU processes",
                        "Implement auto-scaling",
                        "Review agent workload distribution"
                    ],
                    timestamp=datetime.now().timestamp(),
                    data_points={"current_cpu": current_cpu, "trend": cpu_trend, "hours_to_exhaustion": hours_to_exhaustion}
                )
                self.insights_history.append(insight)

    async def _generate_usage_insights(self):
        """Generate usage pattern insights"""
        # Analyze agent utilization patterns
        agent_data = dashboard_service.metrics_collector.get_metrics_data("active_agents", time_range=86400)

        if len(agent_data) > 24:  # At least 24 hours of data
            hourly_usage = defaultdict(list)

            for point in agent_data:
                hour = datetime.fromtimestamp(point.timestamp).hour
                hourly_usage[hour].append(point.value)

            # Calculate average usage by hour
            avg_hourly = {hour: np.mean(values) for hour, values in hourly_usage.items()}
            peak_hour = max(avg_hourly, key=avg_hourly.get)
            low_hour = min(avg_hourly, key=avg_hourly.get)

            insight = Insight(
                id=f"usage_pattern_{int(datetime.now().timestamp())}",
                category="optimization",
                title="Agent Usage Pattern Analysis",
                description=f"Peak usage at hour {peak_hour}, lowest at hour {low_hour}",
                impact="medium",
                confidence=0.90,
                recommendations=[
                    f"Consider scaling up resources before hour {peak_hour}",
                    f"Schedule maintenance during hour {low_hour}",
                    "Implement predictive auto-scaling",
                    "Optimize agent distribution based on usage patterns"
                ],
                timestamp=datetime.now().timestamp(),
                data_points={"peak_hour": peak_hour, "low_hour": low_hour, "hourly_avg": avg_hourly}
            )
            self.insights_history.append(insight)

    async def get_predictions(self, metric_name: str, time_horizon: int = 3600) -> List[Prediction]:
        """Get predictions for a specific metric"""
        cutoff_time = datetime.now().timestamp() - time_horizon
        return [
            pred for pred in self.predictions_history
            if pred.metric_name == metric_name and pred.timestamp >= cutoff_time
        ]

    async def get_anomalies(self, time_range: int = 3600) -> List[Anomaly]:
        """Get recent anomalies"""
        cutoff_time = datetime.now().timestamp() - time_range
        return [
            anomaly for anomaly in self.anomalies_history
            if anomaly.timestamp >= cutoff_time
        ]

    async def get_insights(self, category: str = None, time_range: int = 86400) -> List[Insight]:
        """Get recent insights"""
        cutoff_time = datetime.now().timestamp() - time_range
        insights = [
            insight for insight in self.insights_history
            if insight.timestamp >= cutoff_time
        ]

        if category:
            insights = [insight for insight in insights if insight.category == category]

        return insights

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        current_time = datetime.now().timestamp()

        # Recent predictions
        recent_predictions = [
            asdict(pred) for pred in self.predictions_history
            if current_time - pred.timestamp < 3600
        ]

        # Recent anomalies
        recent_anomalies = [
            asdict(anomaly) for anomaly in self.anomalies_history
            if current_time - anomaly.timestamp < 3600
        ]

        # Recent insights
        recent_insights = [
            asdict(insight) for insight in self.insights_history
            if current_time - insight.timestamp < 86400
        ]

        # Model status
        model_status = {}
        for name, model in self.models.items():
            model_status[name] = {
                "is_trained": model.is_trained,
                "last_training": model.last_training.isoformat() if model.last_training else None,
                "performance": model.performance_metrics
            }

        return {
            "timestamp": current_time,
            "predictions": recent_predictions,
            "anomalies": recent_anomalies,
            "insights": recent_insights,
            "model_status": model_status,
            "summary": {
                "total_predictions": len(recent_predictions),
                "total_anomalies": len(recent_anomalies),
                "total_insights": len(recent_insights),
                "high_impact_insights": len([i for i in recent_insights if i.get("impact") == "high"])
            }
        }

# Global analytics engine instance
analytics_engine = AnalyticsEngine()
