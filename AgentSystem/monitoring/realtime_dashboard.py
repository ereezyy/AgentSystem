"""
Real-Time Dashboard System
Enhanced monitoring with Grafana integration, custom metrics, and performance alerts
"""

import asyncio
import time
import json
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from ..utils.logger import get_logger
from ..core.agent_swarm import swarm_coordinator
from ..services.streaming_service import streaming_service

logger = get_logger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str]

@dataclass
class Alert:
    """Performance alert"""
    id: str
    severity: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    timestamp: float
    metric_name: str
    threshold: float
    current_value: float
    resolved: bool = False

class MetricsCollector:
    """Collects and manages custom metrics"""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.alert_thresholds = {
            "cpu_usage": {"high": 80, "critical": 95},
            "memory_usage": {"high": 80, "critical": 95},
            "response_time": {"high": 2.0, "critical": 5.0},
            "error_rate": {"high": 0.05, "critical": 0.1},
            "active_agents": {"low": 1, "critical": 0}
        }

        # Prometheus metrics
        self._setup_prometheus_metrics()

        # Background collection will be started later when event loop is available
        self._metrics_task = None

    def _setup_prometheus_metrics(self):
        """Set up Prometheus metrics"""
        self.request_counter = Counter(
            'agentsystem_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.response_time_histogram = Histogram(
            'agentsystem_response_time_seconds',
            'Response time in seconds',
            ['endpoint'],
            registry=self.registry
        )

        self.active_agents_gauge = Gauge(
            'agentsystem_active_agents',
            'Number of active agents',
            ['agent_type'],
            registry=self.registry
        )

        self.system_cpu_gauge = Gauge(
            'agentsystem_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.system_memory_gauge = Gauge(
            'agentsystem_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )

        self.ai_provider_requests = Counter(
            'agentsystem_ai_provider_requests_total',
            'Total AI provider requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )

        self.streaming_sessions_gauge = Gauge(
            'agentsystem_streaming_sessions',
            'Number of active streaming sessions',
            registry=self.registry
        )

    async def _collect_metrics_loop(self):
        """Background loop to collect metrics"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_agent_metrics()
                await self._collect_streaming_metrics()
                await self._check_alerts()
                await asyncio.sleep(5)  # Collect every 5 seconds
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)

    def start_metrics_collection(self):
        """Start the metrics collection loop"""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(self._collect_metrics_loop())
            logger.info("Started metrics collection")
        else:
            logger.warning("Metrics collection already started")

    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        timestamp = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu_gauge.set(cpu_percent)
        self._add_metric_point("cpu_usage", timestamp, cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.system_memory_gauge.set(memory_percent)
        self._add_metric_point("memory_usage", timestamp, memory_percent)

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric_point("disk_usage", timestamp, disk_percent)

        # Network I/O
        network = psutil.net_io_counters()
        self._add_metric_point("network_bytes_sent", timestamp, network.bytes_sent)
        self._add_metric_point("network_bytes_recv", timestamp, network.bytes_recv)

    async def _collect_agent_metrics(self):
        """Collect agent swarm metrics"""
        timestamp = time.time()

        try:
            swarm_status = await swarm_coordinator.get_swarm_status()

            # Active agents by type
            agent_counts = defaultdict(int)
            for agent_id, agent_info in swarm_status["agent_status"].items():
                if agent_info["status"] in ["idle", "working"]:
                    agent_counts[agent_info["role"]] += 1

            for agent_type, count in agent_counts.items():
                self.active_agents_gauge.labels(agent_type=agent_type).set(count)

            # Total active agents
            total_active = sum(agent_counts.values())
            self._add_metric_point("active_agents", timestamp, total_active)

            # Task metrics
            self._add_metric_point("active_tasks", timestamp, swarm_status["active_tasks"])
            self._add_metric_point("completed_tasks", timestamp, swarm_status["completed_tasks"])

        except Exception as e:
            logger.error(f"Agent metrics collection error: {e}")

    async def _collect_streaming_metrics(self):
        """Collect streaming service metrics"""
        timestamp = time.time()

        try:
            active_streams = await streaming_service.get_active_streams()

            # Active streaming sessions
            self.streaming_sessions_gauge.set(len(active_streams))
            self._add_metric_point("streaming_sessions", timestamp, len(active_streams))

            # Streaming by provider
            provider_counts = defaultdict(int)
            for stream_info in active_streams.values():
                provider_counts[stream_info["provider"]] += 1

            for provider, count in provider_counts.items():
                self._add_metric_point(f"streaming_sessions_{provider}", timestamp, count)

        except Exception as e:
            logger.error(f"Streaming metrics collection error: {e}")

    def _add_metric_point(self, metric_name: str, timestamp: float, value: float, labels: Dict[str, str] = None):
        """Add a metric point to history"""
        point = MetricPoint(
            timestamp=timestamp,
            value=value,
            labels=labels or {}
        )
        self.metrics_history[metric_name].append(point)

    async def _check_alerts(self):
        """Check for alert conditions"""
        current_time = time.time()

        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in self.metrics_history:
                latest_points = list(self.metrics_history[metric_name])
                if latest_points:
                    current_value = latest_points[-1].value

                    # Check thresholds
                    for severity, threshold in thresholds.items():
                        condition_met = False

                        if severity in ["high", "critical"]:
                            condition_met = current_value > threshold
                        elif severity in ["low"]:
                            condition_met = current_value < threshold

                        if condition_met:
                            await self._create_alert(
                                severity=severity,
                                metric_name=metric_name,
                                threshold=threshold,
                                current_value=current_value
                            )

    async def _create_alert(self, severity: str, metric_name: str, threshold: float, current_value: float):
        """Create a new alert"""
        # Check if similar alert already exists and is not resolved
        existing_alert = None
        for alert in self.alerts:
            if (alert.metric_name == metric_name and
                alert.severity == severity and
                not alert.resolved and
                time.time() - alert.timestamp < 300):  # 5 minutes
                existing_alert = alert
                break

        if existing_alert:
            return  # Don't create duplicate alerts

        alert = Alert(
            id=f"{metric_name}_{severity}_{int(time.time())}",
            severity=severity,
            title=f"{metric_name.replace('_', ' ').title()} {severity.title()} Alert",
            description=f"{metric_name} is {current_value:.2f}, threshold: {threshold}",
            timestamp=time.time(),
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value
        )

        self.alerts.append(alert)
        logger.warning(f"Alert created: {alert.title} - {alert.description}")

    def record_request(self, method: str, endpoint: str, status: str, response_time: float):
        """Record API request metrics"""
        self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
        self.response_time_histogram.labels(endpoint=endpoint).observe(response_time)

        # Add to custom metrics
        self._add_metric_point("response_time", time.time(), response_time, {
            "endpoint": endpoint,
            "method": method,
            "status": status
        })

    def record_ai_request(self, provider: str, model: str, status: str):
        """Record AI provider request"""
        self.ai_provider_requests.labels(provider=provider, model=model, status=status).inc()

    def get_metrics_data(self, metric_name: str, time_range: int = 3600) -> List[MetricPoint]:
        """Get metrics data for a specific time range"""
        if metric_name not in self.metrics_history:
            return []

        cutoff_time = time.time() - time_range
        return [
            point for point in self.metrics_history[metric_name]
            if point.timestamp >= cutoff_time
        ]

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False

class DashboardService:
    """Main dashboard service"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self._metrics_task = None
        self.dashboard_config = {
            "refresh_interval": 5,
            "chart_types": ["line", "gauge", "bar"],
            "time_ranges": [300, 900, 3600, 86400],  # 5m, 15m, 1h, 1d
            "alert_channels": ["console", "webhook"]
        }

    async def get_dashboard_data(self, time_range: int = 3600) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        current_time = time.time()

        # System metrics
        system_metrics = {
            "cpu_usage": self.metrics_collector.get_metrics_data("cpu_usage", time_range),
            "memory_usage": self.metrics_collector.get_metrics_data("memory_usage", time_range),
            "disk_usage": self.metrics_collector.get_metrics_data("disk_usage", time_range)
        }

        # Agent metrics
        agent_metrics = {
            "active_agents": self.metrics_collector.get_metrics_data("active_agents", time_range),
            "active_tasks": self.metrics_collector.get_metrics_data("active_tasks", time_range),
            "completed_tasks": self.metrics_collector.get_metrics_data("completed_tasks", time_range)
        }

        # Streaming metrics
        streaming_metrics = {
            "streaming_sessions": self.metrics_collector.get_metrics_data("streaming_sessions", time_range)
        }

        # Performance metrics
        performance_metrics = {
            "response_time": self.metrics_collector.get_metrics_data("response_time", time_range)
        }

        # Current status
        swarm_status = await swarm_coordinator.get_swarm_status()
        streaming_status = await streaming_service.get_active_streams()

        # Alerts
        active_alerts = self.metrics_collector.get_active_alerts()

        return {
            "timestamp": current_time,
            "system_metrics": system_metrics,
            "agent_metrics": agent_metrics,
            "streaming_metrics": streaming_metrics,
            "performance_metrics": performance_metrics,
            "current_status": {
                "swarm": swarm_status,
                "streaming": streaming_status
            },
            "alerts": [asdict(alert) for alert in active_alerts],
            "config": self.dashboard_config
        }

    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        return prometheus_client.generate_latest(self.metrics_collector.registry).decode('utf-8')

    async def create_grafana_dashboard(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration"""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "AgentSystem Real-Time Dashboard",
                "tags": ["agentsystem", "ai", "monitoring"],
                "timezone": "browser",
                "refresh": "5s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "System CPU Usage",
                        "type": "stat",
                        "targets": [{
                            "expr": "agentsystem_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        }],
                        "fieldConfig": {
                            "defaults": {
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 70},
                                        {"color": "red", "value": 90}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Active Agents",
                        "type": "stat",
                        "targets": [{
                            "expr": "sum(agentsystem_active_agents)",
                            "legendFormat": "Active Agents"
                        }]
                    },
                    {
                        "id": 3,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(agentsystem_response_time_seconds_sum[5m]) / rate(agentsystem_response_time_seconds_count[5m])",
                            "legendFormat": "Avg Response Time"
                        }]
                    },
                    {
                        "id": 4,
                        "title": "AI Provider Requests",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(agentsystem_ai_provider_requests_total[5m])",
                            "legendFormat": "{{provider}} - {{model}}"
                        }]
                    }
                ]
            }
        }
        return dashboard

    def record_request_metric(self, method: str, endpoint: str, status: str, response_time: float):
        """Record request metrics"""
        self.metrics_collector.record_request(method, endpoint, status, response_time)

    def record_ai_request_metric(self, provider: str, model: str, status: str):
        """Record AI request metrics"""
        self.metrics_collector.record_ai_request(provider, model, status)

# Global dashboard service instance
dashboard_service = DashboardService()

# Middleware for automatic request tracking
class MetricsMiddleware:
    """Middleware to automatically track request metrics"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, environ, start_response):
        start_time = time.time()
        method = environ.get('REQUEST_METHOD', 'GET')
        path = environ.get('PATH_INFO', '/')

        # Call the actual application
        response = await self.app(environ, start_response)

        # Record metrics
        response_time = time.time() - start_time
        status = "200"  # Default, should be extracted from actual response

        dashboard_service.record_request_metric(method, path, status, response_time)

        return response
