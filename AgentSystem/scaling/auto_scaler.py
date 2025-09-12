"""
Auto-Scaling System
Implements dynamic scaling based on metrics and load balancing
"""

import asyncio
import time
import json
import docker
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import yaml
import os

from ..utils.logger import get_logger
from ..monitoring.realtime_dashboard import dashboard_service

logger = get_logger(__name__)

@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    service_name: str
    metric_name: str
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int
    max_instances: int
    cooldown_period: int  # seconds
    scale_up_step: int
    scale_down_step: int
    evaluation_period: int  # seconds

@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: float
    service_name: str
    action: str  # "scale_up", "scale_down"
    from_instances: int
    to_instances: int
    trigger_metric: str
    trigger_value: float
    threshold: float
    reason: str

class ServiceManager:
    """Manages service instances and scaling operations"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.current_instances = defaultdict(int)
        self.last_scaling_events = {}
        self.service_configs = {}

        # Initialize current instance counts
        self._discover_current_instances()

    def _discover_current_instances(self):
        """Discover currently running service instances"""
        try:
            containers = self.docker_client.containers.list()
            service_counts = defaultdict(int)

            for container in containers:
                # Extract service name from container labels or name
                service_name = self._extract_service_name(container)
                if service_name:
                    service_counts[service_name] += 1

            self.current_instances = service_counts
            logger.info(f"Discovered current instances: {dict(service_counts)}")

        except Exception as e:
            logger.error(f"Error discovering instances: {e}")

    def _extract_service_name(self, container) -> Optional[str]:
        """Extract service name from container"""
        try:
            # Try to get from compose service label
            if 'com.docker.compose.service' in container.labels:
                return container.labels['com.docker.compose.service']

            # Try to extract from container name
            name = container.name
            if 'agentsystem-' in name:
                return name.replace('agentsystem-', '').split('-')[0]

            return None
        except Exception:
            return None

    async def scale_service(self, service_name: str, target_instances: int) -> bool:
        """Scale a service to target number of instances"""
        try:
            current = self.current_instances[service_name]

            if target_instances == current:
                return True

            if target_instances > current:
                # Scale up
                for i in range(target_instances - current):
                    await self._start_service_instance(service_name)
            else:
                # Scale down
                for i in range(current - target_instances):
                    await self._stop_service_instance(service_name)

            self.current_instances[service_name] = target_instances
            logger.info(f"Scaled {service_name} from {current} to {target_instances} instances")
            return True

        except Exception as e:
            logger.error(f"Error scaling {service_name}: {e}")
            return False

    async def _start_service_instance(self, service_name: str):
        """Start a new instance of a service"""
        try:
            # Use docker-compose to scale up
            compose_file = "docker-compose.microservices.yml"
            current_count = self.current_instances[service_name]
            new_count = current_count + 1

            # Execute docker-compose scale command
            import subprocess
            result = subprocess.run([
                "docker-compose", "-f", compose_file,
                "scale", f"{service_name}={new_count}"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Started new instance of {service_name}")
            else:
                logger.error(f"Failed to start {service_name}: {result.stderr}")

        except Exception as e:
            logger.error(f"Error starting {service_name} instance: {e}")

    async def _stop_service_instance(self, service_name: str):
        """Stop an instance of a service"""
        try:
            # Find containers for this service
            containers = self.docker_client.containers.list(
                filters={"label": f"com.docker.compose.service={service_name}"}
            )

            if containers:
                # Stop the most recently created container
                container_to_stop = max(containers, key=lambda c: c.attrs['Created'])
                container_to_stop.stop()
                logger.info(f"Stopped instance of {service_name}: {container_to_stop.short_id}")

        except Exception as e:
            logger.error(f"Error stopping {service_name} instance: {e}")

    def get_service_instances(self, service_name: str) -> int:
        """Get current number of instances for a service"""
        return self.current_instances.get(service_name, 0)

class LoadBalancer:
    """Manages load balancing configuration"""

    def __init__(self):
        self.haproxy_config_path = "/usr/local/etc/haproxy/haproxy.cfg"
        self.service_endpoints = defaultdict(list)
        self.health_check_results = defaultdict(dict)

    async def update_service_endpoints(self, service_name: str, endpoints: List[str]):
        """Update endpoints for a service"""
        self.service_endpoints[service_name] = endpoints
        await self._regenerate_haproxy_config()

    async def _regenerate_haproxy_config(self):
        """Regenerate HAProxy configuration"""
        try:
            config = self._generate_haproxy_config()

            # Write new config
            with open(self.haproxy_config_path, 'w') as f:
                f.write(config)

            # Reload HAProxy
            await self._reload_haproxy()

        except Exception as e:
            logger.error(f"Error regenerating HAProxy config: {e}")

    def _generate_haproxy_config(self) -> str:
        """Generate HAProxy configuration"""
        config = """
global
    daemon
    maxconn 4096
    log stdout local0

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option redispatch
    retries 3

# Stats interface
stats enable
stats uri /haproxy-stats
stats refresh 30s
stats admin if TRUE

# Frontend for main application
frontend agentsystem_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/agentsystem.pem
    redirect scheme https if !{ ssl_fc }

    # Route based on path
    acl is_api path_beg /api
    acl is_streaming path_beg /stream
    acl is_analytics path_beg /analytics
    acl is_notifications path_beg /notifications

    use_backend api_backend if is_api
    use_backend streaming_backend if is_streaming
    use_backend analytics_backend if is_analytics
    use_backend notifications_backend if is_notifications
    default_backend orchestrator_backend

# Backend for orchestrator service
backend orchestrator_backend
    balance roundrobin
    option httpchk GET /health
"""

        # Add orchestrator endpoints
        for i, endpoint in enumerate(self.service_endpoints.get('agent-orchestrator', [])):
            config += f"    server orchestrator{i+1} {endpoint} check\n"

        # Add AI provider backend
        config += """
backend api_backend
    balance roundrobin
    option httpchk GET /health
"""
        for i, endpoint in enumerate(self.service_endpoints.get('ai-provider-service', [])):
            config += f"    server ai_provider{i+1} {endpoint} check\n"

        # Add streaming backend
        config += """
backend streaming_backend
    balance roundrobin
    option httpchk GET /health
"""
        for i, endpoint in enumerate(self.service_endpoints.get('streaming-service', [])):
            config += f"    server streaming{i+1} {endpoint} check\n"

        # Add analytics backend
        config += """
backend analytics_backend
    balance roundrobin
    option httpchk GET /health
"""
        for i, endpoint in enumerate(self.service_endpoints.get('analytics-service', [])):
            config += f"    server analytics{i+1} {endpoint} check\n"

        # Add notifications backend
        config += """
backend notifications_backend
    balance roundrobin
    option httpchk GET /health
"""
        for i, endpoint in enumerate(self.service_endpoints.get('notification-service', [])):
            config += f"    server notifications{i+1} {endpoint} check\n"

        return config

    async def _reload_haproxy(self):
        """Reload HAProxy configuration"""
        try:
            import subprocess
            result = subprocess.run([
                "docker", "exec", "agentsystem-loadbalancer",
                "haproxy", "-f", "/usr/local/etc/haproxy/haproxy.cfg", "-sf", "$(cat /var/run/haproxy.pid)"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("HAProxy configuration reloaded successfully")
            else:
                logger.error(f"Failed to reload HAProxy: {result.stderr}")

        except Exception as e:
            logger.error(f"Error reloading HAProxy: {e}")

    async def health_check_services(self):
        """Perform health checks on all services"""
        for service_name, endpoints in self.service_endpoints.items():
            for endpoint in endpoints:
                try:
                    response = requests.get(f"http://{endpoint}/health", timeout=5)
                    self.health_check_results[service_name][endpoint] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "response_time": response.elapsed.total_seconds(),
                        "last_check": time.time()
                    }
                except Exception as e:
                    self.health_check_results[service_name][endpoint] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "last_check": time.time()
                    }

class AutoScaler:
    """Main auto-scaling orchestrator"""

    def __init__(self, config_path: str = "config/scaling-rules.yml"):
        self.service_manager = ServiceManager()
        self.load_balancer = LoadBalancer()
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.config_path = config_path

        # Load scaling rules
        self._load_scaling_rules()

        # Start auto-scaling loop
        asyncio.create_task(self._auto_scaling_loop())
        asyncio.create_task(self._health_check_loop())

    def _load_scaling_rules(self):
        """Load scaling rules from configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)

                for rule_config in config.get('scaling_rules', []):
                    rule = ScalingRule(**rule_config)
                    self.scaling_rules[rule.service_name] = rule

                logger.info(f"Loaded {len(self.scaling_rules)} scaling rules")
            else:
                # Create default rules
                self._create_default_rules()

        except Exception as e:
            logger.error(f"Error loading scaling rules: {e}")
            self._create_default_rules()

    def _create_default_rules(self):
        """Create default scaling rules"""
        default_rules = [
            ScalingRule(
                service_name="ai-provider-service",
                metric_name="cpu_usage",
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                min_instances=1,
                max_instances=5,
                cooldown_period=300,
                scale_up_step=1,
                scale_down_step=1,
                evaluation_period=60
            ),
            ScalingRule(
                service_name="task-processor",
                metric_name="queue_length",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                min_instances=1,
                max_instances=10,
                cooldown_period=180,
                scale_up_step=2,
                scale_down_step=1,
                evaluation_period=30
            ),
            ScalingRule(
                service_name="streaming-service",
                metric_name="active_connections",
                scale_up_threshold=50.0,
                scale_down_threshold=10.0,
                min_instances=1,
                max_instances=3,
                cooldown_period=120,
                scale_up_step=1,
                scale_down_step=1,
                evaluation_period=45
            )
        ]

        for rule in default_rules:
            self.scaling_rules[rule.service_name] = rule

    async def _auto_scaling_loop(self):
        """Main auto-scaling loop"""
        while True:
            try:
                await self._evaluate_scaling_rules()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(60)

    async def _health_check_loop(self):
        """Health check loop"""
        while True:
            try:
                await self.load_balancer.health_check_services()
                await asyncio.sleep(30)  # Health check every 30 seconds
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)

    async def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules"""
        for service_name, rule in self.scaling_rules.items():
            try:
                await self._evaluate_service_scaling(service_name, rule)
            except Exception as e:
                logger.error(f"Error evaluating scaling for {service_name}: {e}")

    async def _evaluate_service_scaling(self, service_name: str, rule: ScalingRule):
        """Evaluate scaling for a specific service"""
        # Check cooldown period
        last_event = self.service_manager.last_scaling_events.get(service_name)
        if last_event and time.time() - last_event < rule.cooldown_period:
            return

        # Get current metric value
        metric_data = dashboard_service.metrics_collector.get_metrics_data(
            rule.metric_name, time_range=rule.evaluation_period
        )

        if not metric_data:
            return

        # Calculate average metric value over evaluation period
        avg_value = sum(point.value for point in metric_data) / len(metric_data)
        current_instances = self.service_manager.get_service_instances(service_name)

        # Determine scaling action
        action = None
        target_instances = current_instances

        if avg_value > rule.scale_up_threshold and current_instances < rule.max_instances:
            action = "scale_up"
            target_instances = min(current_instances + rule.scale_up_step, rule.max_instances)
        elif avg_value < rule.scale_down_threshold and current_instances > rule.min_instances:
            action = "scale_down"
            target_instances = max(current_instances - rule.scale_down_step, rule.min_instances)

        # Execute scaling action
        if action and target_instances != current_instances:
            success = await self.service_manager.scale_service(service_name, target_instances)

            if success:
                # Record scaling event
                event = ScalingEvent(
                    timestamp=time.time(),
                    service_name=service_name,
                    action=action,
                    from_instances=current_instances,
                    to_instances=target_instances,
                    trigger_metric=rule.metric_name,
                    trigger_value=avg_value,
                    threshold=rule.scale_up_threshold if action == "scale_up" else rule.scale_down_threshold,
                    reason=f"{rule.metric_name} {action.replace('_', ' ')} threshold triggered"
                )

                self.scaling_history.append(event)
                self.service_manager.last_scaling_events[service_name] = time.time()

                logger.info(f"Scaling event: {action} {service_name} from {current_instances} to {target_instances}")

                # Update load balancer
                await self._update_load_balancer_endpoints(service_name, target_instances)

    async def _update_load_balancer_endpoints(self, service_name: str, instance_count: int):
        """Update load balancer with new service endpoints"""
        try:
            # Generate endpoints based on service naming convention
            endpoints = []
            base_port = self._get_service_base_port(service_name)

            for i in range(instance_count):
                endpoint = f"{service_name}-{i+1}:{base_port}"
                endpoints.append(endpoint)

            await self.load_balancer.update_service_endpoints(service_name, endpoints)

        except Exception as e:
            logger.error(f"Error updating load balancer endpoints: {e}")

    def _get_service_base_port(self, service_name: str) -> int:
        """Get base port for a service"""
        port_mapping = {
            "agent-orchestrator": 8000,
            "ai-provider-service": 8001,
            "notification-service": 8002,
            "analytics-service": 8003,
            "streaming-service": 8004,
            "task-processor": 8006
        }
        return port_mapping.get(service_name, 8000)

    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        status = {
            "timestamp": time.time(),
            "services": {},
            "recent_events": [],
            "load_balancer_status": {}
        }

        # Service status
        for service_name, rule in self.scaling_rules.items():
            current_instances = self.service_manager.get_service_instances(service_name)
            status["services"][service_name] = {
                "current_instances": current_instances,
                "min_instances": rule.min_instances,
                "max_instances": rule.max_instances,
                "scaling_metric": rule.metric_name,
                "scale_up_threshold": rule.scale_up_threshold,
                "scale_down_threshold": rule.scale_down_threshold
            }

        # Recent scaling events
        recent_events = [
            asdict(event) for event in self.scaling_history
            if time.time() - event.timestamp < 3600  # Last hour
        ]
        status["recent_events"] = recent_events[-10:]  # Last 10 events

        # Load balancer status
        status["load_balancer_status"] = self.load_balancer.health_check_results

        return status

    async def manual_scale(self, service_name: str, target_instances: int) -> bool:
        """Manually scale a service"""
        if service_name not in self.scaling_rules:
            return False

        rule = self.scaling_rules[service_name]
        if target_instances < rule.min_instances or target_instances > rule.max_instances:
            return False

        current_instances = self.service_manager.get_service_instances(service_name)
        success = await self.service_manager.scale_service(service_name, target_instances)

        if success:
            # Record manual scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                service_name=service_name,
                action="manual_scale",
                from_instances=current_instances,
                to_instances=target_instances,
                trigger_metric="manual",
                trigger_value=0,
                threshold=0,
                reason="Manual scaling operation"
            )

            self.scaling_history.append(event)
            await self._update_load_balancer_endpoints(service_name, target_instances)

        return success

# Global auto-scaler instance
auto_scaler = AutoScaler()
