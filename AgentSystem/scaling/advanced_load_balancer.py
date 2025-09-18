"""
Advanced Load Balancer
Implements sophisticated load balancing strategies with predictive scaling
"""

import asyncio
import time
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import os

from ..utils.logger import get_logger
from ..monitoring.realtime_dashboard import dashboard_service
from .auto_scaler import auto_scaler, ServiceManager, LoadBalancer as BaseLoadBalancer

logger = get_logger(__name__)

@dataclass
class LoadDistribution:
    """Load distribution configuration"""
    service_name: str
    weight_distribution: List[float]  # Weights for each instance
    health_scores: List[float]        # Health scores for each instance
    last_updated: float

@dataclass
class PredictiveScalingModel:
    """Predictive scaling model parameters"""
    service_name: str
    historical_data_points: int
    prediction_horizon: int  # in minutes
    model_weights: Dict[str, float]  # weights for different metrics
    last_trained: float

class AdvancedLoadBalancer(BaseLoadBalancer):
    """Enhanced load balancer with predictive analytics and dynamic weighting"""

    def __init__(self):
        super().__init__()
        self.distribution_configs: Dict[str, LoadDistribution] = {}
        self.predictive_models: Dict[str, PredictiveScalingModel] = {}
        self.prediction_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._prediction_task = None

    async def initialize_advanced_balancing(self):
        """Initialize advanced load balancing strategies"""
        for service_name in auto_scaler.scaling_rules.keys():
            self.distribution_configs[service_name] = LoadDistribution(
                service_name=service_name,
                weight_distribution=[],
                health_scores=[],
                last_updated=0
            )

            self.predictive_models[service_name] = PredictiveScalingModel(
                service_name=service_name,
                historical_data_points=288,  # 24 hours at 5 min intervals
                prediction_horizon=30,
                model_weights={
                    "cpu_usage": 0.3,
                    "response_time": 0.3,
                    "request_rate": 0.2,
                    "error_rate": 0.2
                },
                last_trained=0
            )

        logger.info("Initialized advanced load balancing for services")
        await self.start_prediction_loop()

    async def start_prediction_loop(self):
        """Start the predictive scaling analysis loop"""
        if self._prediction_task is None:
            self._prediction_task = asyncio.create_task(self._prediction_loop())
            logger.info("Started predictive scaling analysis loop")
        else:
            logger.warning("Prediction loop already started")

    async def _prediction_loop(self):
        """Main loop for predictive analysis"""
        while True:
            try:
                for service_name, model in self.predictive_models.items():
                    await self._update_predictions(service_name, model)
                await asyncio.sleep(300)  # Update predictions every 5 minutes
            except Exception as e:
                logger.error(f"Predictive analysis loop error: {e}")
                await asyncio.sleep(600)

    async def _update_predictions(self, service_name: str, model: PredictiveScalingModel):
        """Update predictive scaling model and generate predictions"""
        current_time = time.time()

        # Check if model needs retraining
        if current_time - model.last_trained > 3600:  # Retrain hourly
            await self._retrain_model(service_name, model)
            model.last_trained = current_time

        # Gather recent metrics for prediction
        metrics_data = {
            metric: dashboard_service.metrics_collector.get_metrics_data(
                metric, time_range=model.historical_data_points * 300  # 5 min intervals
            )
            for metric in model.model_weights.keys()
        }

        # Generate predictions for the horizon
        predictions = {}
        for metric, weight in model.model_weights.items():
            metric_data = metrics_data.get(metric, [])
            if len(metric_data) > model.historical_data_points // 2:
                # Simple moving average prediction as baseline
                values = [point.value for point in metric_data[-model.historical_data_points:]]
                predicted_value = np.mean(values[-10:]) if len(values) >= 10 else np.mean(values)
                predictions[metric] = predicted_value * weight

        self.prediction_cache[service_name] = predictions

        # Check if predictive scaling is needed
        await self._evaluate_predictive_scaling(service_name, predictions)

        logger.info(f"Updated predictions for {service_name}")

    async def _retrain_model(self, service_name: str, model: PredictiveScalingModel):
        """Retrain the predictive model based on historical patterns"""
        try:
            # Placeholder for more sophisticated ML model training
            # In a real implementation, this would use historical data to train time series models
            logger.info(f"Retraining predictive model for {service_name}")

            # Adjust weights based on recent performance
            for metric in model.model_weights.keys():
                model.model_weights[metric] = max(0.1, min(0.5, model.model_weights[metric]))

        except Exception as e:
            logger.error(f"Error retraining model for {service_name}: {e}")

    async def _evaluate_predictive_scaling(self, service_name: str, predictions: Dict[str, float]):
        """Evaluate if predictive scaling action is needed"""
        rule = auto_scaler.scaling_rules.get(service_name)
        if not rule:
            return

        current_instances = auto_scaler.service_manager.get_service_instances(service_name)
        predicted_load = sum(predictions.values())
        action = None
        target_instances = current_instances

        # Simple threshold-based prediction
        if predicted_load > rule.scale_up_threshold * 0.8 and current_instances < rule.max_instances:
            action = "predictive_scale_up"
            target_instances = min(current_instances + rule.scale_up_step, rule.max_instances)
        elif predicted_load < rule.scale_down_threshold * 1.2 and current_instances > rule.min_instances:
            action = "predictive_scale_down"
            target_instances = max(current_instances - rule.scale_down_step, rule.min_instances)

        if action and target_instances != current_instances:
            logger.info(f"Predictive scaling: {action} for {service_name} based on predicted load {predicted_load}")
            success = await auto_scaler.service_manager.scale_service(service_name, target_instances)

            if success:
                await auto_scaler._update_load_balancer_endpoints(service_name, target_instances)
                logger.info(f"Successful predictive scaling {action} for {service_name} to {target_instances} instances")

    async def update_dynamic_weights(self, service_name: str):
        """Update dynamic load distribution weights based on health and performance"""
        if service_name not in self.service_endpoints:
            return

        endpoints = self.service_endpoints[service_name]
        if not endpoints:
            return

        health_results = self.health_check_results.get(service_name, {})
        weights = []
        health_scores = []

        for endpoint in endpoints:
            health_info = health_results.get(endpoint, {})
            if health_info.get("status") == "healthy":
                response_time = health_info.get("response_time", 1.0)
                health_score = max(0.1, 1.0 - response_time)  # Simple scoring
            else:
                health_score = 0.1  # Minimal weight for unhealthy instances

            health_scores.append(health_score)
            weights.append(health_score)

        # Normalize weights to sum to 1.0
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        else:
            weights = [1.0/len(weights)] * len(weights)

        self.distribution_configs[service_name] = LoadDistribution(
            service_name=service_name,
            weight_distribution=weights,
            health_scores=health_scores,
            last_updated=time.time()
        )

        # Apply weights to HAProxy configuration
        await self._apply_dynamic_weights(service_name, weights)

        logger.info(f"Updated dynamic weights for {service_name}: {weights}")

    async def _apply_dynamic_weights(self, service_name: str, weights: List[float]):
        """Apply dynamic weights to load balancer configuration"""
        try:
            # This would update HAProxy configuration with new weights
            # In a real implementation, use HAProxy runtime API to update weights
            pass
        except Exception as e:
            logger.error(f"Error applying dynamic weights for {service_name}: {e}")

    async def get_advanced_balancing_status(self) -> Dict[str, Any]:
        """Get status of advanced load balancing"""
        status = {
            "timestamp": time.time(),
            "services": {},
            "predictions": dict(self.prediction_cache)
        }

        for service_name, config in self.distribution_configs.items():
            status["services"][service_name] = {
                "weight_distribution": config.weight_distribution,
                "health_scores": config.health_scores,
                "last_updated": config.last_updated,
                "endpoints": self.service_endpoints.get(service_name, [])
            }

        return status

# Global advanced load balancer instance
advanced_load_balancer = AdvancedLoadBalancer()
