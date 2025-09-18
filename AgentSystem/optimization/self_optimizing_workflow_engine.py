"""
AgentSystem Self-Optimizing Workflow Engine
Intelligent workflow optimization through machine learning and performance analysis
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
import statistics

class OptimizationType(Enum):
    """Types of workflow optimizations"""
    PERFORMANCE = "performance"
    COST = "cost"
    RELIABILITY = "reliability"
    USER_EXPERIENCE = "user_experience"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    EXECUTION_TIME = "execution_time"

class OptimizationStrategy(Enum):
    """Optimization strategies"""
    PARALLEL_EXECUTION = "parallel_execution"
    CACHING = "caching"
    BATCH_PROCESSING = "batch_processing"
    INTELLIGENT_ROUTING = "intelligent_routing"
    RESOURCE_POOLING = "resource_pooling"
    PREDICTIVE_SCALING = "predictive_scaling"
    DYNAMIC_CONFIGURATION = "dynamic_configuration"

@dataclass
class WorkflowPerformanceMetrics:
    """Comprehensive workflow performance metrics"""
    workflow_id: str
    execution_time_seconds: float
    success_rate: float
    error_rate: float
    cost_per_execution: float
    resource_utilization: Dict[str, float]
    user_satisfaction_score: float
    throughput_per_hour: float
    latency_percentiles: Dict[str, float]
    timestamp: datetime

@dataclass
class OptimizationOpportunity:
    """Identified optimization opportunity"""
    opportunity_id: str
    workflow_id: str
    optimization_type: OptimizationType
    strategy: OptimizationStrategy
    potential_improvement: Dict[str, float]
    implementation_effort: str  # low, medium, high
    confidence_score: float
    business_impact: Dict[str, Any]
    recommended_actions: List[str]

class SelfOptimizingWorkflowEngine:
    """Revolutionary self-optimizing workflow engine with ML-powered optimization"""

    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self.logger = logging.getLogger(__name__)

        # Optimization models and algorithms
        self.optimization_models = {}
        self.performance_baselines = {}
        self.optimization_history = {}

        # Real-time monitoring
        self.active_workflows = {}
        self.optimization_queue = asyncio.Queue()

        # Performance thresholds
        self.performance_thresholds = {
            "execution_time_threshold_seconds": 30,
            "success_rate_minimum": 0.95,
            "cost_efficiency_target": 0.01,  # $0.01 per execution
            "user_satisfaction_minimum": 4.0
        }

        # Initialize optimization engine
        asyncio.create_task(self._initialize_optimization_engine())

    async def _initialize_optimization_engine(self):
        """Initialize the self-optimization engine"""

        # Load historical performance data
        await self._load_performance_baselines()

        # Start optimization monitoring
        await self._start_optimization_monitoring()

        # Initialize ML models for optimization
        await self._initialize_optimization_models()

        self.logger.info("Self-optimizing workflow engine initialized successfully")

    async def analyze_workflow_performance(self, workflow_id: str, analysis_period_days: int = 7) -> WorkflowPerformanceMetrics:
        """Comprehensive workflow performance analysis"""

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=analysis_period_days)

            with self.Session() as session:
                # Get execution metrics
                execution_data = session.execute(text("""
                    SELECT
                        AVG(execution_time_seconds) as avg_execution_time,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END)::FLOAT / COUNT(*) as success_rate,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END)::FLOAT / COUNT(*) as error_rate,
                        AVG(cost) as avg_cost,
                        COUNT(*) as total_executions,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY execution_time_seconds) as p50_latency,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_seconds) as p95_latency,
                        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY execution_time_seconds) as p99_latency
                    FROM workflow_executions
                    WHERE workflow_id = :workflow_id
                    AND executed_at BETWEEN :start_date AND :end_date
                """), {
                    "workflow_id": workflow_id,
                    "start_date": start_date,
                    "end_date": end_date
                }).fetchone()

                # Get resource utilization
                resource_data = session.execute(text("""
                    SELECT
                        AVG(cpu_usage) as avg_cpu,
                        AVG(memory_usage) as avg_memory,
                        AVG(io_operations) as avg_io
                    FROM workflow_resource_usage
                    WHERE workflow_id = :workflow_id
                    AND timestamp BETWEEN :start_date AND :end_date
                """), {
                    "workflow_id": workflow_id,
                    "start_date": start_date,
                    "end_date": end_date
                }).fetchone()

                # Get user satisfaction
                satisfaction_data = session.execute(text("""
                    SELECT AVG(satisfaction_score) as avg_satisfaction
                    FROM workflow_feedback
                    WHERE workflow_id = :workflow_id
                    AND created_at BETWEEN :start_date AND :end_date
                """), {
                    "workflow_id": workflow_id,
                    "start_date": start_date,
                    "end_date": end_date
                }).fetchone()

            if not execution_data or execution_data.total_executions == 0:
                raise ValueError(f"No execution data found for workflow {workflow_id}")

            # Calculate throughput
            throughput = execution_data.total_executions / (analysis_period_days * 24) if analysis_period_days > 0 else 0

            metrics = WorkflowPerformanceMetrics(
                workflow_id=workflow_id,
                execution_time_seconds=float(execution_data.avg_execution_time or 0),
                success_rate=float(execution_data.success_rate or 0),
                error_rate=float(execution_data.error_rate or 0),
                cost_per_execution=float(execution_data.avg_cost or 0),
                resource_utilization={
                    "cpu": float(resource_data.avg_cpu or 0) if resource_data else 0,
                    "memory": float(resource_data.avg_memory or 0) if resource_data else 0,
                    "io": float(resource_data.avg_io or 0) if resource_data else 0
                },
                user_satisfaction_score=float(satisfaction_data.avg_satisfaction or 0) if satisfaction_data else 0,
                throughput_per_hour=throughput,
                latency_percentiles={
                    "p50": float(execution_data.p50_latency or 0),
                    "p95": float(execution_data.p95_latency or 0),
                    "p99": float(execution_data.p99_latency or 0)
                },
                timestamp=datetime.now()
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to analyze workflow performance for {workflow_id}: {e}")
            raise

    async def identify_optimization_opportunities(self, workflow_id: str) -> List[OptimizationOpportunity]:
        """Identify optimization opportunities using ML analysis"""

        try:
            # Get current performance metrics
            current_metrics = await self.analyze_workflow_performance(workflow_id)

            # Get historical performance for comparison
            historical_metrics = await self._get_historical_performance(workflow_id, days=30)

            # Get baseline performance
            baseline = self.performance_baselines.get(workflow_id, {})

            opportunities = []

            # Performance optimization opportunities
            if current_metrics.execution_time_seconds > self.performance_thresholds["execution_time_threshold_seconds"]:
                opportunities.append(await self._analyze_execution_time_optimization(workflow_id, current_metrics))

            # Cost optimization opportunities
            if current_metrics.cost_per_execution > self.performance_thresholds["cost_efficiency_target"]:
                opportunities.append(await self._analyze_cost_optimization(workflow_id, current_metrics))

            # Reliability optimization opportunities
            if current_metrics.success_rate < self.performance_thresholds["success_rate_minimum"]:
                opportunities.append(await self._analyze_reliability_optimization(workflow_id, current_metrics))

            # Resource efficiency opportunities
            resource_optimization = await self._analyze_resource_optimization(workflow_id, current_metrics)
            if resource_optimization:
                opportunities.append(resource_optimization)

            # User experience optimization
            if current_metrics.user_satisfaction_score < self.performance_thresholds["user_satisfaction_minimum"]:
                opportunities.append(await self._analyze_ux_optimization(workflow_id, current_metrics))

            # Filter out None values and sort by potential impact
            opportunities = [opp for opp in opportunities if opp is not None]
            opportunities.sort(key=lambda x: x.confidence_score * sum(x.potential_improvement.values()), reverse=True)

            return opportunities

        except Exception as e:
            self.logger.error(f"Failed to identify optimization opportunities for {workflow_id}: {e}")
            raise

    async def _analyze_execution_time_optimization(self, workflow_id: str, metrics: WorkflowPerformanceMetrics) -> OptimizationOpportunity:
        """Analyze execution time optimization opportunities"""

        # Analyze workflow steps for bottlenecks
        bottlenecks = await self._identify_bottlenecks(workflow_id)

        # Determine optimization strategy
        if len(bottlenecks) > 2:
            strategy = OptimizationStrategy.PARALLEL_EXECUTION
            potential_improvement = {"execution_time_reduction": 0.4, "throughput_increase": 0.6}
        elif metrics.resource_utilization["cpu"] < 50:
            strategy = OptimizationStrategy.INTELLIGENT_ROUTING
            potential_improvement = {"execution_time_reduction": 0.25, "resource_efficiency": 0.3}
        else:
            strategy = OptimizationStrategy.CACHING
            potential_improvement = {"execution_time_reduction": 0.3, "cost_reduction": 0.2}

        return OptimizationOpportunity(
            opportunity_id=f"exec_time_{workflow_id}_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            optimization_type=OptimizationType.EXECUTION_TIME,
            strategy=strategy,
            potential_improvement=potential_improvement,
            implementation_effort="medium",
            confidence_score=0.85,
            business_impact={
                "time_savings_per_execution": metrics.execution_time_seconds * potential_improvement.get("execution_time_reduction", 0),
                "annual_cost_savings": await self._calculate_annual_savings(workflow_id, potential_improvement)
            },
            recommended_actions=[
                f"Implement {strategy.value} optimization",
                "Monitor performance improvement",
                "A/B test optimization impact"
            ]
        )

    async def _analyze_cost_optimization(self, workflow_id: str, metrics: WorkflowPerformanceMetrics) -> OptimizationOpportunity:
        """Analyze cost optimization opportunities"""

        # Analyze cost breakdown
        cost_breakdown = await self._analyze_cost_breakdown(workflow_id)

        # Identify highest cost components
        highest_cost_component = max(cost_breakdown.items(), key=lambda x: x[1])[0]

        if highest_cost_component == "ai_tokens":
            strategy = OptimizationStrategy.INTELLIGENT_ROUTING
            potential_improvement = {"cost_reduction": 0.35, "performance_maintained": 0.95}
        elif highest_cost_component == "compute":
            strategy = OptimizationStrategy.RESOURCE_POOLING
            potential_improvement = {"cost_reduction": 0.25, "resource_efficiency": 0.4}
        else:
            strategy = OptimizationStrategy.BATCH_PROCESSING
            potential_improvement = {"cost_reduction": 0.3, "throughput_increase": 0.2}

        return OptimizationOpportunity(
            opportunity_id=f"cost_{workflow_id}_{datetime.now().timestamp()}",
            workflow_id=workflow_id,
            optimization_type=OptimizationType.COST,
            strategy=strategy,
            potential_improvement=potential_improvement,
            implementation_effort="low",
            confidence_score=0.82,
            business_impact={
                "monthly_cost_savings": metrics.cost_per_execution * potential_improvement.get("cost_reduction", 0) * await self._get_monthly_executions(workflow_id),
                "roi_timeline": "30-60 days"
            },
            recommended_actions=[
                f"Implement {strategy.value} for cost optimization",
                "Monitor cost reduction metrics",
                "Validate performance maintained"
            ]
        )

    async def implement_optimization(self, opportunity: OptimizationOpportunity) -> Dict[str, Any]:
        """Implement workflow optimization based on identified opportunity"""

        try:
            optimization_id = f"opt_{opportunity.opportunity_id}_{datetime.now().timestamp()}"

            # Get current baseline
            baseline_metrics = await self.analyze_workflow_performance(opportunity.workflow_id)

            # Implement optimization strategy
            implementation_result = await self._execute_optimization_strategy(
                opportunity.workflow_id,
                opportunity.strategy,
                opportunity.potential_improvement
            )

            # Monitor implementation
            monitoring_result = await self._monitor_optimization_impact(
                optimization_id,
                opportunity.workflow_id,
                baseline_metrics,
                implementation_result
            )

            # Calculate actual improvement
            post_optimization_metrics = await self.analyze_workflow_performance(opportunity.workflow_id)
            actual_improvement = await self._calculate_actual_improvement(
                baseline_metrics, post_optimization_metrics
            )

            # Store optimization results
            await self._store_optimization_results(
                optimization_id, opportunity, baseline_metrics,
                post_optimization_metrics, actual_improvement
            )

            # Update optimization models with results
            await self._update_optimization_models(opportunity, actual_improvement)

            return {
                "optimization_id": optimization_id,
                "status": "completed",
                "baseline_metrics": asdict(baseline_metrics),
                "post_optimization_metrics": asdict(post_optimization_metrics),
                "actual_improvement": actual_improvement,
                "success_rate": implementation_result.get("success_rate", 0),
                "rollback_available": implementation_result.get("rollback_available", True),
                "implementation_time": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to implement optimization: {e}")
            # Attempt rollback on failure
            await self._rollback_optimization(opportunity.workflow_id)
            raise

    async def continuous_optimization_monitoring(self):
        """Continuous monitoring and optimization of all workflows"""

        while True:
            try:
                # Get all active workflows
                active_workflows = await self._get_active_workflows()

                for workflow_id in active_workflows:
                    # Analyze performance
                    current_metrics = await self.analyze_workflow_performance(workflow_id)

                    # Check if optimization is needed
                    needs_optimization = await self._evaluate_optimization_need(workflow_id, current_metrics)

                    if needs_optimization:
                        # Identify optimization opportunities
                        opportunities = await self.identify_optimization_opportunities(workflow_id)

                        # Select best opportunity
                        if opportunities:
                            best_opportunity = opportunities[0]  # Already sorted by impact

                            # Implement optimization if confidence is high enough
                            if best_opportunity.confidence_score > 0.8:
                                await self.optimization_queue.put(best_opportunity)

                # Process optimization queue
                await self._process_optimization_queue()

                # Sleep between monitoring cycles
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"Error in continuous optimization monitoring: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error

    async def _execute_optimization_strategy(
        self,
        workflow_id: str,
        strategy: OptimizationStrategy,
        potential_improvement: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute specific optimization strategy"""

        implementation_results = {}

        if strategy == OptimizationStrategy.PARALLEL_EXECUTION:
            # Implement parallel execution optimization
            result = await self._implement_parallel_execution(workflow_id)
            implementation_results = {
                "strategy": "parallel_execution",
                "changes_made": result["changes"],
                "expected_improvement": potential_improvement,
                "rollback_available": True
            }

        elif strategy == OptimizationStrategy.CACHING:
            # Implement intelligent caching
            result = await self._implement_intelligent_caching(workflow_id)
            implementation_results = {
                "strategy": "caching",
                "cache_configuration": result["cache_config"],
                "expected_improvement": potential_improvement,
                "rollback_available": True
            }

        elif strategy == OptimizationStrategy.INTELLIGENT_ROUTING:
            # Implement intelligent routing optimization
            result = await self._implement_intelligent_routing(workflow_id)
            implementation_results = {
                "strategy": "intelligent_routing",
                "routing_rules": result["routing_rules"],
                "expected_improvement": potential_improvement,
                "rollback_available": True
            }

        elif strategy == OptimizationStrategy.BATCH_PROCESSING:
            # Implement batch processing optimization
            result = await self._implement_batch_processing(workflow_id)
            implementation_results = {
                "strategy": "batch_processing",
                "batch_configuration": result["batch_config"],
                "expected_improvement": potential_improvement,
                "rollback_available": True
            }

        else:
            # Dynamic configuration optimization
            result = await self._implement_dynamic_configuration(workflow_id, potential_improvement)
            implementation_results = {
                "strategy": "dynamic_configuration",
                "configuration_changes": result["config_changes"],
                "expected_improvement": potential_improvement,
                "rollback_available": True
            }

        return implementation_results

    async def _implement_parallel_execution(self, workflow_id: str) -> Dict[str, Any]:
        """Implement parallel execution optimization"""

        # Analyze workflow DAG for parallelization opportunities
        workflow_steps = await self._get_workflow_steps(workflow_id)

        # Identify independent steps that can run in parallel
        parallel_groups = await self._identify_parallel_groups(workflow_steps)

        # Update workflow configuration for parallel execution
        changes = []
        for group in parallel_groups:
            if len(group) > 1:
                # Configure parallel execution for this group
                parallel_config = {
                    "type": "parallel_group",
                    "steps": group,
                    "max_concurrency": min(len(group), 5),  # Limit concurrency
                    "timeout_seconds": 300,
                    "failure_strategy": "partial_success"
                }

                await self._update_workflow_step_config(workflow_id, group, parallel_config)
                changes.append(f"Parallelized {len(group)} steps: {', '.join(group)}")

        return {
            "changes": changes,
            "parallel_groups": len(parallel_groups),
            "optimization_type": "parallel_execution"
        }

    async def _implement_intelligent_caching(self, workflow_id: str) -> Dict[str, Any]:
        """Implement intelligent caching optimization"""

        # Analyze cacheable operations
        cacheable_operations = await self._identify_cacheable_operations(workflow_id)

        cache_config = {}
        for operation in cacheable_operations:
            cache_strategy = await self._determine_cache_strategy(operation)

            cache_config[operation["step_id"]] = {
                "cache_type": cache_strategy["type"],
                "ttl_seconds": cache_strategy["ttl"],
                "cache_key_pattern": cache_strategy["key_pattern"],
                "invalidation_triggers": cache_strategy["invalidation_triggers"]
            }

            # Apply caching configuration
            await self._apply_cache_configuration(workflow_id, operation["step_id"], cache_strategy)

        return {
            "cache_config": cache_config,
            "cached_operations": len(cacheable_operations),
            "optimization_type": "intelligent_caching"
        }

    async def _implement_intelligent_routing(self, workflow_id: str) -> Dict[str, Any]:
        """Implement intelligent routing optimization"""

        # Analyze AI provider usage and costs
        ai_usage = await self._analyze_ai_provider_usage(workflow_id)

        # Create intelligent routing rules
        routing_rules = []
        for step_id, usage_data in ai_usage.items():
            optimal_provider = await self._determine_optimal_provider(usage_data)

            routing_rule = {
                "step_id": step_id,
                "provider_priority": optimal_provider["priority_list"],
                "cost_threshold": optimal_provider["cost_threshold"],
                "performance_threshold": optimal_provider["performance_threshold"],
                "fallback_strategy": optimal_provider["fallback"]
            }

            routing_rules.append(routing_rule)

            # Apply routing configuration
            await self._apply_routing_configuration(workflow_id, step_id, routing_rule)

        return {
            "routing_rules": routing_rules,
            "optimized_steps": len(routing_rules),
            "optimization_type": "intelligent_routing"
        }

    async def auto_optimize_workflow(self, workflow_id: str, max_optimizations: int = 3) -> Dict[str, Any]:
        """Automatically optimize workflow with multiple strategies"""

        try:
            optimization_results = []

            # Get initial baseline
            initial_metrics = await self.analyze_workflow_performance(workflow_id)

            # Identify and implement top optimization opportunities
            opportunities = await self.identify_optimization_opportunities(workflow_id)

            implemented_count = 0
            for opportunity in opportunities[:max_optimizations]:
                if implemented_count >= max_optimizations:
                    break

                try:
                    # Implement optimization
                    result = await self.implement_optimization(opportunity)
                    optimization_results.append(result)
                    implemented_count += 1

                    # Wait between implementations to measure impact
                    await asyncio.sleep(60)

                except Exception as e:
                    self.logger.error(f"Failed to implement optimization {opportunity.opportunity_id}: {e}")
                    continue

            # Get final metrics after all optimizations
            final_metrics = await self.analyze_workflow_performance(workflow_id)

            # Calculate overall improvement
            overall_improvement = await self._calculate_overall_improvement(
                initial_metrics, final_metrics
            )

            return {
                "workflow_id": workflow_id,
                "optimizations_implemented": implemented_count,
                "optimization_results": optimization_results,
                "initial_metrics": asdict(initial_metrics),
                "final_metrics": asdict(final_metrics),
                "overall_improvement": overall_improvement,
                "optimization_completed_at": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to auto-optimize workflow {workflow_id}: {e}")
            raise

    async def generate_optimization_recommendations(self, tenant_id: str) -> Dict[str, Any]:
        """Generate optimization recommendations for all tenant workflows"""

        try:
            # Get all workflows for tenant
            tenant_workflows = await self._get_tenant_workflows(tenant_id)

            all_recommendations = []

            for workflow_id in tenant_workflows:
                # Analyze workflow performance
                metrics = await self.analyze_workflow_performance(workflow_id)

                # Get optimization opportunities
                opportunities = await self.identify_optimization_opportunities(workflow_id)

                if opportunities:
                    workflow_recommendations = {
                        "workflow_id": workflow_id,
                        "current_performance": asdict(metrics),
                        "optimization_opportunities": [asdict(opp) for opp in opportunities],
                        "recommended_priority": opportunities[0].optimization_type.value if opportunities else None,
                        "potential_impact": await self._calculate_portfolio_impact(opportunities)
                    }

                    all_recommendations.append(workflow_recommendations)

            # Sort by potential business impact
            all_recommendations.sort(
                key=lambda x: x["potential_impact"]["business_value"], reverse=True
            )

            # Generate portfolio-level recommendations
            portfolio_recommendations = await self._generate_portfolio_recommendations(
                all_recommendations
            )

            return {
                "tenant_id": tenant_id,
                "workflow_recommendations": all_recommendations,
                "portfolio_recommendations": portfolio_recommendations,
                "total_workflows_analyzed": len(tenant_workflows),
                "optimization_opportunities": sum(len(rec["optimization_opportunities"]) for rec in all_recommendations),
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations for {tenant_id}: {e}")
            raise

class WorkflowLearningEngine:
    """Machine learning engine for workflow optimization"""

    def __init__(self, optimization_engine: SelfOptimizingWorkflowEngine):
        self.optimization_engine = optimization_engine
        self.logger = logging.getLogger(__name__)

    async def learn_from_optimization_outcomes(self, optimization_results: Dict[str, Any]):
        """Learn from optimization outcomes to improve future decisions"""

        try:
            # Extract learning features
            features = await self._extract_learning_features(optimization_results)

            # Update optimization models
            await self._update_models_with_outcome(features, optimization_results)

            # Identify patterns in successful optimizations
            patterns = await self._identify_optimization_patterns(optimization_results)

            # Update recommendation algorithms
            await self._update_recommendation_algorithms(patterns)

            return {
                "learning_completed": True,
                "features_extracted": len(features),
                "patterns_identified": len(patterns),
                "models_updated": True
            }

        except Exception as e:
            self.logger.error(f"Failed to learn from optimization outcomes: {e}")
            raise

    async def predict_optimization_success(
        self,
        workflow_id: str,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Predict success probability of optimization strategy"""

        # Get workflow characteristics
        workflow_chars = await self._get_workflow_characteristics(workflow_id)

        # Get historical success rates for similar workflows
        similar_optimizations = await self._get_similar_optimization_history(
            workflow_chars, strategy
        )

        # Calculate success probability
        if similar_optimizations:
            success_rate = sum(opt["success"] for opt in similar_optimizations) / len(similar_optimizations)
            confidence = min(0.95, len(similar_optimizations) / 10)  # More data = higher confidence
        else:
            success_rate = 0.7  # Default conservative estimate
            confidence = 0.5

        return {
            "predicted_success_rate": success_rate,
            "confidence_level": confidence,
            "similar_cases": len(similar_optimizations),
            "recommendation": "proceed" if success_rate > 0.75 and confidence > 0.6 else "caution"
        }
