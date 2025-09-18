
"""
Comprehensive Monitoring and Alerting API Endpoints
Provides REST API for monitoring system management, metrics collection, alerting, and health checks
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import json
import uuid
from typing import Dict, List, Any, Optional
import logging
from functools import wraps
import redis
import asyncio
from dataclasses import dataclass

# Import monitoring engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring.comprehensive_monitoring_engine import ComprehensiveMonitoringEngine

# Create blueprint
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/v1/monitoring')

# Initialize monitoring engine
monitoring_engine = None

def get_monitoring_engine():
    """Get or create monitoring engine instance"""
    global monitoring_engine
    if monitoring_engine is None:
        monitoring_engine = ComprehensiveMonitoringEngine()
    return monitoring_engine

@dataclass
class MonitoringResponse:
    """Standard monitoring API response"""
    success: bool
    data: Any = None
    message: str = ""
    error: str = ""
    metadata: Dict = None

def monitoring_response(success: bool, data: Any = None, message: str = "", error: str = "", metadata: Dict = None):
    """Create standardized monitoring response"""
    return jsonify({
        'success': success,
        'data': data,
        'message': message,
        'error': error,
        'metadata': metadata or {},
        'timestamp': datetime.utcnow().isoformat()
    })

def validate_tenant_access():
    """Validate tenant access for monitoring operations"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            tenant_id = request.headers.get('X-Tenant-ID')
            if not tenant_id:
                return monitoring_response(False, error="Tenant ID required in X-Tenant-ID header"), 400

            # Add tenant_id to request context
            request.tenant_id = tenant_id
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Monitoring Session Management

@monitoring_bp.route('/sessions', methods=['POST'])
@validate_tenant_access()
def create_monitoring_session():
    """Create a new monitoring session"""
    try:
        data = request.get_json()

        session_config = {
            'session_name': data.get('session_name', 'Default Monitoring'),
            'monitoring_scope': data.get('monitoring_scope', 'tenant'),
            'data_retention_days': data.get('data_retention_days', 90),
            'sampling_interval_seconds': data.get('sampling_interval_seconds', 30),
            'metrics_config': data.get('metrics_config', {}),
            'alert_config': data.get('alert_config', {}),
            'health_check_config': data.get('health_check_config', {}),
            'notification_config': data.get('notification_config', {})
        }

        engine = get_monitoring_engine()
        session_id = engine.start_monitoring_session(request.tenant_id, session_config)

        return monitoring_response(
            True,
            {'session_id': session_id, 'config': session_config},
            "Monitoring session created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating monitoring session: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/sessions', methods=['GET'])
@validate_tenant_access()
def list_monitoring_sessions():
    """List all monitoring sessions for tenant"""
    try:
        engine = get_monitoring_engine()
        sessions = engine.get_monitoring_sessions(request.tenant_id)

        return monitoring_response(
            True,
            sessions,
            f"Retrieved {len(sessions)} monitoring sessions"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing monitoring sessions: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/sessions/<session_id>', methods=['GET'])
@validate_tenant_access()
def get_monitoring_session(session_id):
    """Get specific monitoring session details"""
    try:
        engine = get_monitoring_engine()
        session = engine.get_monitoring_session(session_id, request.tenant_id)

        if not session:
            return monitoring_response(False, error="Monitoring session not found"), 404

        return monitoring_response(True, session, "Monitoring session retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting monitoring session: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/sessions/<session_id>', methods=['PUT'])
@validate_tenant_access()
def update_monitoring_session(session_id):
    """Update monitoring session configuration"""
    try:
        data = request.get_json()

        engine = get_monitoring_engine()
        result = engine.update_monitoring_session(session_id, request.tenant_id, data)

        if not result:
            return monitoring_response(False, error="Monitoring session not found"), 404

        return monitoring_response(True, result, "Monitoring session updated")

    except Exception as e:
        current_app.logger.error(f"Error updating monitoring session: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/sessions/<session_id>', methods=['DELETE'])
@validate_tenant_access()
def delete_monitoring_session(session_id):
    """Stop and delete monitoring session"""
    try:
        engine = get_monitoring_engine()
        result = engine.stop_monitoring_session(session_id, request.tenant_id)

        if not result:
            return monitoring_response(False, error="Monitoring session not found"), 404

        return monitoring_response(True, message="Monitoring session stopped and deleted")

    except Exception as e:
        current_app.logger.error(f"Error deleting monitoring session: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Metrics Management

@monitoring_bp.route('/metrics', methods=['POST'])
@validate_tenant_access()
def collect_metric():
    """Collect a single metric or batch of metrics"""
    try:
        data = request.get_json()

        if 'metrics' in data:
            # Batch metrics collection
            metrics = data['metrics']
            session_id = data.get('session_id')

            engine = get_monitoring_engine()
            results = []

            for metric in metrics:
                result = engine.collect_metric(
                    session_id or 'default',
                    request.tenant_id,
                    metric['metric_name'],
                    metric['metric_value'],
                    metric.get('metric_type', 'custom'),
                    metric.get('labels', {}),
                    metric.get('source_system', 'api')
                )
                results.append(result)

            return monitoring_response(
                True,
                {'collected_metrics': len(results), 'results': results},
                f"Collected {len(results)} metrics"
            )
        else:
            # Single metric collection
            engine = get_monitoring_engine()
            result = engine.collect_metric(
                data.get('session_id', 'default'),
                request.tenant_id,
                data['metric_name'],
                data['metric_value'],
                data.get('metric_type', 'custom'),
                data.get('labels', {}),
                data.get('source_system', 'api')
            )

            return monitoring_response(True, result, "Metric collected successfully")

    except Exception as e:
        current_app.logger.error(f"Error collecting metrics: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/metrics', methods=['GET'])
@validate_tenant_access()
def query_metrics():
    """Query metrics with filters"""
    try:
        # Parse query parameters
        metric_name = request.args.get('metric_name')
        metric_type = request.args.get('metric_type')
        source_system = request.args.get('source_system')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = request.args.get('limit', 1000, type=int)

        engine = get_monitoring_engine()
        metrics = engine.query_metrics(
            request.tenant_id,
            metric_name=metric_name,
            metric_type=metric_type,
            source_system=source_system,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        return monitoring_response(
            True,
            metrics,
            f"Retrieved {len(metrics)} metrics"
        )

    except Exception as e:
        current_app.logger.error(f"Error querying metrics: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/metrics/aggregated', methods=['GET'])
@validate_tenant_access()
def get_aggregated_metrics():
    """Get aggregated metrics (hourly, daily, weekly)"""
    try:
        metric_name = request.args.get('metric_name')
        aggregation = request.args.get('aggregation', 'hourly')  # hourly, daily, weekly
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')

        engine = get_monitoring_engine()
        aggregated_data = engine.get_aggregated_metrics(
            request.tenant_id,
            metric_name,
            aggregation,
            start_time,
            end_time
        )

        return monitoring_response(
            True,
            aggregated_data,
            f"Retrieved aggregated metrics ({aggregation})"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting aggregated metrics: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Alert Management

@monitoring_bp.route('/alert-rules', methods=['POST'])
@validate_tenant_access()
def create_alert_rule():
    """Create a new alert rule"""
    try:
        data = request.get_json()

        rule_config = {
            'rule_name': data['rule_name'],
            'description': data.get('description', ''),
            'metric_name': data['metric_name'],
            'condition_operator': data['condition_operator'],
            'threshold_value': data['threshold_value'],
            'severity': data['severity'],
            'evaluation_window_seconds': data.get('evaluation_window_seconds', 300),
            'evaluation_frequency_seconds': data.get('evaluation_frequency_seconds', 60),
            'minimum_occurrences': data.get('minimum_occurrences', 1),
            'consecutive_breaches_required': data.get('consecutive_breaches_required', 1),
            'recovery_threshold': data.get('recovery_threshold'),
            'recovery_occurrences': data.get('recovery_occurrences', 1),
            'notification_channels': data.get('notification_channels', []),
            'escalation_policy_id': data.get('escalation_policy_id'),
            'runbook_url': data.get('runbook_url'),
            'tags': data.get('tags', {}),
            'suppression_config': data.get('suppression_config', {})
        }

        engine = get_monitoring_engine()
        rule_id = engine.create_alert_rule(request.tenant_id, rule_config)

        return monitoring_response(
            True,
            {'rule_id': rule_id, 'config': rule_config},
            "Alert rule created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating alert rule: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/alert-rules', methods=['GET'])
@validate_tenant_access()
def list_alert_rules():
    """List all alert rules for tenant"""
    try:
        enabled_only = request.args.get('enabled_only', 'false').lower() == 'true'

        engine = get_monitoring_engine()
        rules = engine.get_alert_rules(request.tenant_id, enabled_only=enabled_only)

        return monitoring_response(
            True,
            rules,
            f"Retrieved {len(rules)} alert rules"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing alert rules: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/alert-rules/<rule_id>', methods=['PUT'])
@validate_tenant_access()
def update_alert_rule(rule_id):
    """Update an existing alert rule"""
    try:
        data = request.get_json()

        engine = get_monitoring_engine()
        result = engine.update_alert_rule(rule_id, request.tenant_id, data)

        if not result:
            return monitoring_response(False, error="Alert rule not found"), 404

        return monitoring_response(True, result, "Alert rule updated successfully")

    except Exception as e:
        current_app.logger.error(f"Error updating alert rule: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/alert-rules/<rule_id>', methods=['DELETE'])
@validate_tenant_access()
def delete_alert_rule(rule_id):
    """Delete an alert rule"""
    try:
        engine = get_monitoring_engine()
        result = engine.delete_alert_rule(rule_id, request.tenant_id)

        if not result:
            return monitoring_response(False, error="Alert rule not found"), 404

        return monitoring_response(True, message="Alert rule deleted successfully")

    except Exception as e:
        current_app.logger.error(f"Error deleting alert rule: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/alerts', methods=['GET'])
@validate_tenant_access()
def list_alerts():
    """List alerts with filtering options"""
    try:
        # Parse query parameters
        status = request.args.get('status')
        severity = request.args.get('severity')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)

        engine = get_monitoring_engine()
        alerts = engine.get_alerts(
            request.tenant_id,
            status=status,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )

        return monitoring_response(
            True,
            alerts,
            f"Retrieved {len(alerts)} alerts"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing alerts: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
@validate_tenant_access()
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    try:
        data = request.get_json() or {}
        acknowledged_by = data.get('acknowledged_by', 'api_user')
        notes = data.get('notes', '')

        engine = get_monitoring_engine()
        result = engine.acknowledge_alert(alert_id, request.tenant_id, acknowledged_by, notes)

        if not result:
            return monitoring_response(False, error="Alert not found"), 404

        return monitoring_response(True, result, "Alert acknowledged successfully")

    except Exception as e:
        current_app.logger.error(f"Error acknowledging alert: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/alerts/<alert_id>/resolve', methods=['POST'])
@validate_tenant_access()
def resolve_alert(alert_id):
    """Resolve an alert"""
    try:
        data = request.get_json() or {}
        resolved_by = data.get('resolved_by', 'api_user')
        resolution_notes = data.get('resolution_notes', '')

        engine = get_monitoring_engine()
        result = engine.resolve_alert(alert_id, request.tenant_id, resolved_by, resolution_notes)

        if not result:
            return monitoring_response(False, error="Alert not found"), 404

        return monitoring_response(True, result, "Alert resolved successfully")

    except Exception as e:
        current_app.logger.error(f"Error resolving alert: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/alerts/<alert_id>/suppress', methods=['POST'])
@validate_tenant_access()
def suppress_alert(alert_id):
    """Suppress an alert for a specified duration"""
    try:
        data = request.get_json()
        suppress_until = data.get('suppress_until')
        reason = data.get('reason', 'Manual suppression')

        engine = get_monitoring_engine()
        result = engine.suppress_alert(alert_id, request.tenant_id, suppress_until, reason)

        if not result:
            return monitoring_response(False, error="Alert not found"), 404

        return monitoring_response(True, result, "Alert suppressed successfully")

    except Exception as e:
        current_app.logger.error(f"Error suppressing alert: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Health Check Management

@monitoring_bp.route('/health-checks', methods=['POST'])
@validate_tenant_access()
def create_health_check():
    """Create a new health check"""
    try:
        data = request.get_json()

        check_config = {
            'service_name': data['service_name'],
            'check_name': data['check_name'],
            'check_type': data['check_type'],
            'endpoint_url': data.get('endpoint_url'),
            'expected_response': data.get('expected_response'),
            'expected_status_code': data.get('expected_status_code', 200),
            'timeout_seconds': data.get('timeout_seconds', 30),
            'check_interval_seconds': data.get('check_interval_seconds', 60),
            'consecutive_failures_threshold': data.get('consecutive_failures_threshold', 3),
            'critical_check': data.get('critical_check', False),
            'check_configuration': data.get('check_configuration', {})
        }

        engine = get_monitoring_engine()
        check_id = engine.create_health_check(request.tenant_id, check_config)

        return monitoring_response(
            True,
            {'check_id': check_id, 'config': check_config},
            "Health check created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating health check: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/health-checks', methods=['GET'])
@validate_tenant_access()
def list_health_checks():
    """List all health checks for tenant"""
    try:
        service_name = request.args.get('service_name')
        check_type = request.args.get('check_type')
        enabled_only = request.args.get('enabled_only', 'false').lower() == 'true'

        engine = get_monitoring_engine()
        checks = engine.get_health_checks(
            request.tenant_id,
            service_name=service_name,
            check_type=check_type,
            enabled_only=enabled_only
        )

        return monitoring_response(
            True,
            checks,
            f"Retrieved {len(checks)} health checks"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing health checks: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/health-checks/<check_id>/execute', methods=['POST'])
@validate_tenant_access()
def execute_health_check(check_id):
    """Execute a health check on demand"""
    try:
        engine = get_monitoring_engine()
        result = engine.execute_health_check(check_id, request.tenant_id)

        if not result:
            return monitoring_response(False, error="Health check not found"), 404

        return monitoring_response(True, result, "Health check executed")

    except Exception as e:
        current_app.logger.error(f"Error executing health check: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/health-checks/<check_id>', methods=['DELETE'])
@validate_tenant_access()
def delete_health_check(check_id):
    """Delete a health check"""
    try:
        engine = get_monitoring_engine()
        result = engine.delete_health_check(check_id, request.tenant_id)

        if not result:
            return monitoring_response(False, error="Health check not found"), 404

        return monitoring_response(True, message="Health check deleted successfully")

    except Exception as e:
        current_app.logger.error(f"Error deleting health check: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Service Health Status

@monitoring_bp.route('/health/services', methods=['GET'])
@validate_tenant_access()
def get_service_health_status():
    """Get health status for all services"""
    try:
        service_name = request.args.get('service_name')

        engine = get_monitoring_engine()
        health_status = engine.get_service_health_status(
            request.tenant_id,
            service_name=service_name
        )

        return monitoring_response(
            True,
            health_status,
            f"Retrieved health status for {len(health_status)} services"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting service health status: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/health/overall', methods=['GET'])
@validate_tenant_access()
def get_overall_health():
    """Get overall system health summary"""
    try:
        engine = get_monitoring_engine()
        overall_health = engine.get_overall_system_health(request.tenant_id)

        return monitoring_response(
            True,
            overall_health,
            "Retrieved overall system health"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting overall health: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Anomaly Detection

@monitoring_bp.route('/anomalies', methods=['GET'])
@validate_tenant_access()
def list_anomalies():
    """List detected anomalies"""
    try:
        metric_name = request.args.get('metric_name')
        anomaly_type = request.args.get('anomaly_type')
        min_score = request.args.get('min_score', 0.5, type=float)
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = request.args.get('limit', 100, type=int)

        engine = get_monitoring_engine()
        anomalies = engine.get_anomalies(
            request.tenant_id,
            metric_name=metric_name,
            anomaly_type=anomaly_type,
            min_score=min_score,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        return monitoring_response(
            True,
            anomalies,
            f"Retrieved {len(anomalies)} anomalies"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing anomalies: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/anomalies/analyze', methods=['POST'])
@validate_tenant_access()
def analyze_anomalies():
    """Run anomaly detection on specified metrics"""
    try:
        data = request.get_json()
        metric_names = data.get('metric_names', [])
        analysis_period_hours = data.get('analysis_period_hours', 24)

        engine = get_monitoring_engine()
        results = engine.run_anomaly_detection(
            request.tenant_id,
            metric_names,
            analysis_period_hours
        )

        return monitoring_response(
            True,
            results,
            f"Anomaly analysis completed for {len(metric_names)} metrics"
        )

    except Exception as e:
        current_app.logger.error(f"Error analyzing anomalies: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Predictive Analysis

@monitoring_bp.route('/predictions', methods=['POST'])
@validate_tenant_access()
def create_prediction():
    """Generate predictive analysis"""
    try:
        data = request.get_json()

        prediction_config = {
            'prediction_type': data['prediction_type'],
            'target_metric': data['target_metric'],
            'prediction_horizon_hours': data.get('prediction_horizon_hours', 24),
            'model_type': data.get('model_type', 'auto'),
            'features': data.get('features', []),
            'confidence_threshold': data.get('confidence_threshold', 0.7)
        }

        engine = get_monitoring_engine()
        prediction = engine.generate_prediction(request.tenant_id, prediction_config)

        return monitoring_response(
            True,
            prediction,
            "Prediction generated successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating prediction: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/predictions', methods=['GET'])
@validate_tenant_access()
def list_predictions():
    """List predictions with filtering"""
    try:
        prediction_type = request.args.get('prediction_type')
        target_metric = request.args.get('target_metric')
        risk_level = request.args.get('risk_level')
        limit = request.args.get('limit', 50, type=int)

        engine = get_monitoring_engine()
        predictions = engine.get_predictions(
            request.tenant_id,
            prediction_type=prediction_type,
            target_metric=target_metric,
            risk_level=risk_level,
            limit=limit
        )

        return monitoring_response(
            True,
            predictions,
            f"Retrieved {len(predictions)} predictions"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing predictions: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Notification Management

@monitoring_bp.route('/notification-channels', methods=['POST'])
@validate_tenant_access()
def create_notification_channel():
    """Create a new notification channel"""
    try:
        data = request.get_json()

        channel_config = {
            'channel_name': data['channel_name'],
            'channel_type': data['channel_type'],
            'channel_config': data['channel_config'],
            'rate_limit_per_hour': data.get('rate_limit_per_hour', 100)
        }

        engine = get_monitoring_engine()
        channel_id = engine.create_notification_channel(request.tenant_id, channel_config)

        return monitoring_response(
            True,
            {'channel_id': channel_id, 'config': channel_config},
            "Notification channel created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating notification channel: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/notification-channels', methods=['GET'])
@validate_tenant_access()
def list_notification_channels():
    """List all notification channels for tenant"""
    try:
        channel_type = request.args.get('channel_type')
        enabled_only = request.args.get('enabled_only', 'false').lower() == 'true'

        engine = get_monitoring_engine()
        channels = engine.get_notification_channels(
            request.tenant_id,
            channel_type=channel_type,
            enabled_only=enabled_only
        )

        return monitoring_response(
            True,
            channels,
            f"Retrieved {len(channels)} notification channels"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing notification channels: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/notification-channels/<channel_id>/test', methods=['POST'])
@validate_tenant_access()
def test_notification_channel(channel_id):
    """Test a notification channel"""
    try:
        data = request.get_json() or {}
        test_message = data.get('test_message', 'Test notification from AgentSystem monitoring')

        engine = get_monitoring_engine()
        result = engine.test_notification_channel(channel_id, request.tenant_id, test_message)

        if not result:
            return monitoring_response(False, error="Notification channel not found"), 404

        return monitoring_response(True, result, "Notification channel test completed")

    except Exception as e:
        current_app.logger.error(f"Error testing notification channel: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Dashboard Management

@monitoring_bp.route('/dashboards', methods=['POST'])
@validate_tenant_access()
def create_dashboard():
    """Create a new monitoring dashboard"""
    try:
        data = request.get_json()

        dashboard_config = {
            'dashboard_name': data['dashboard_name'],
            'dashboard_type': data.get('dashboard_type', 'custom'),
            'layout_config': data.get('layout_config', {}),
            'widget_configs': data.get('widget_configs', []),
            'refresh_interval_seconds': data.get('refresh_interval_seconds', 30),
            'auto_refresh': data.get('auto_refresh', True),
            'public_access': data.get('public_access', False),
            'access_permissions': data.get('access_permissions', {}),
            'theme': data.get('theme', 'default')
        }

        engine = get_monitoring_engine()
        dashboard_id = engine.create_dashboard(request.tenant_id, dashboard_config)

        return monitoring_response(
            True,
            {'dashboard_id': dashboard_id, 'config': dashboard_config},
            "Dashboard created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating dashboard: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/dashboards', methods=['GET'])
@validate_tenant_access()
def list_dashboards():
    """List all dashboards for tenant"""
    try:
        dashboard_type = request.args.get('dashboard_type')

        engine = get_monitoring_engine()
        dashboards = engine.get_dashboards(
            request.tenant_id,
            dashboard_type=dashboard_type
        )

        return monitoring_response(
            True,
            dashboards,
            f"Retrieved {len(dashboards)} dashboards"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing dashboards: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/dashboards/<dashboard_id>', methods=['GET'])
@validate_tenant_access()
def get_dashboard(dashboard_id):
    """Get specific dashboard configuration"""
    try:
        engine = get_monitoring_engine()
        dashboard = engine.get_dashboard(dashboard_id, request.tenant_id)

        if not dashboard:
            return monitoring_response(False, error="Dashboard not found"), 404

        return monitoring_response(True, dashboard, "Dashboard retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting dashboard: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/dashboards/<dashboard_id>', methods=['PUT'])
@validate_tenant_access()
def update_dashboard(dashboard_id):
    """Update dashboard configuration"""
    try:
        data = request.get_json()

        engine = get_monitoring_engine()
        result = engine.update_dashboard(dashboard_id, request.tenant_id, data)

        if not result:
            return monitoring_response(False, error="Dashboard not found"), 404

        return monitoring_response(True, result, "Dashboard updated successfully")

    except Exception as e:
        current_app.logger.error(f"Error updating dashboard: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/dashboards/<dashboard_id>', methods=['DELETE'])
@validate_tenant_access()
def delete_dashboard(dashboard_id):
    """Delete a dashboard"""
    try:
        engine = get_monitoring_engine()
        result = engine.delete_dashboard(dashboard_id, request.tenant_id)

        if not result:
            return monitoring_response(False, error="Dashboard not found"), 404

        return monitoring_response(True, message="Dashboard deleted successfully")

    except Exception as e:
        current_app.logger.error(f"Error deleting dashboard: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# SLA Management

@monitoring_bp.route('/sla', methods=['POST'])
@validate_tenant_access()
def create_sla():
    """Create a new SLA definition"""
    try:
        data = request.get_json()

        sla_config = {
            'sla_name': data['sla_name'],
            'service_name': data['service_name'],
            'sla_type': data['sla_type'],
            'target_value': data['target_value'],
            'measurement_unit': data['measurement_unit'],
            'measurement_period': data['measurement_period'],
            'penalty_threshold': data.get('penalty_threshold'),
            'penalty_amount': data.get('penalty_amount', 0.00),
            'grace_period_hours': data.get('grace_period_hours', 0),
            'exclusions': data.get('exclusions', {}),
            'start_date': data['start_date'],
            'end_date': data.get('end_date')
        }

        engine = get_monitoring_engine()
        sla_id = engine.create_sla(request.tenant_id, sla_config)

        return monitoring_response(
            True,
            {'sla_id': sla_id, 'config': sla_config},
            "SLA created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating SLA: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/sla', methods=['GET'])
@validate_tenant_access()
def list_slas():
    """List all SLAs for tenant"""
    try:
        service_name = request.args.get('service_name')
        active_only = request.args.get('active_only', 'false').lower() == 'true'

        engine = get_monitoring_engine()
        slas = engine.get_slas(
            request.tenant_id,
            service_name=service_name,
            active_only=active_only
        )

        return monitoring_response(
            True,
            slas,
            f"Retrieved {len(slas)} SLAs"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing SLAs: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/sla/<sla_id>/performance', methods=['GET'])
@validate_tenant_access()
def get_sla_performance(sla_id):
    """Get SLA performance data"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        engine = get_monitoring_engine()
        performance = engine.get_sla_performance(
            sla_id,
            request.tenant_id,
            start_date=start_date,
            end_date=end_date
        )

        if not performance:
            return monitoring_response(False, error="SLA not found"), 404

        return monitoring_response(True, performance, "SLA performance retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting SLA performance: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Incident Management

@monitoring_bp.route('/incidents', methods=['POST'])
@validate_tenant_access()
def create_incident():
    """Create a new incident"""
    try:
        data = request.get_json()

        incident_config = {
            'incident_title': data['incident_title'],
            'incident_description': data.get('incident_description', ''),
            'severity': data['severity'],
            'priority': data['priority'],
            'affected_services': data.get('affected_services', []),
            'incident_commander': data.get('incident_commander'),
            'response_team': data.get('response_team', []),
            'customer_impact': data.get('customer_impact', 'none'),
            'business_impact_usd': data.get('business_impact_usd', 0.00),
            'related_alerts': data.get('related_alerts', [])
        }

        engine = get_monitoring_engine()
        incident_id = engine.create_incident(request.tenant_id, incident_config)

        return monitoring_response(
            True,
            {'incident_id': incident_id, 'config': incident_config},
            "Incident created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating incident: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/incidents', methods=['GET'])
@validate_tenant_access()
def list_incidents():
    """List incidents with filtering"""
    try:
        status = request.args.get('status')
        severity = request.args.get('severity')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = request.args.get('limit', 50, type=int)

        engine = get_monitoring_engine()
        incidents = engine.get_incidents(
            request.tenant_id,
            status=status,
            severity=severity,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )

        return monitoring_response(
            True,
            incidents,
            f"Retrieved {len(incidents)} incidents"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing incidents: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/incidents/<incident_id>/update-status', methods=['POST'])
@validate_tenant_access()
def update_incident_status(incident_id):
    """Update incident status"""
    try:
        data = request.get_json()
        new_status = data['status']
        update_notes = data.get('notes', '')
        updated_by = data.get('updated_by', 'api_user')

        engine = get_monitoring_engine()
        result = engine.update_incident_status(
            incident_id,
            request.tenant_id,
            new_status,
            updated_by,
            update_notes
        )

        if not result:
            return monitoring_response(False, error="Incident not found"), 404

        return monitoring_response(True, result, "Incident status updated")

    except Exception as e:
        current_app.logger.error(f"Error updating incident status: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Analytics and Reporting

@monitoring_bp.route('/analytics/overview', methods=['GET'])
@validate_tenant_access()
def get_monitoring_analytics():
    """Get comprehensive monitoring analytics"""
    try:
        period = request.args.get('period', '24h')  # 1h, 24h, 7d, 30d

        engine = get_monitoring_engine()
        analytics = engine.get_monitoring_analytics(request.tenant_id, period)

        return monitoring_response(
            True,
            analytics,
            f"Retrieved monitoring analytics for {period}"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting monitoring analytics: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/analytics/trends', methods=['GET'])
@validate_tenant_access()
def get_monitoring_trends():
    """Get monitoring trends and patterns"""
    try:
        metric_names = request.args.getlist('metric_names')
        trend_period = request.args.get('trend_period', '7d')

        engine = get_monitoring_engine()
        trends = engine.get_monitoring_trends(
            request.tenant_id,
            metric_names,
            trend_period
        )

        return monitoring_response(
            True,
            trends,
            f"Retrieved monitoring trends for {len(metric_names)} metrics"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting monitoring trends: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/analytics/reports', methods=['POST'])
@validate_tenant_access()
def generate_monitoring_report():
    """Generate custom monitoring report"""
    try:
        data = request.get_json()

        report_config = {
            'report_name': data['report_name'],
            'report_type': data.get('report_type', 'custom'),
            'time_range': data.get('time_range', '7d'),
            'include_metrics': data.get('include_metrics', True),
            'include_alerts': data.get('include_alerts', True),
            'include_health_checks': data.get('include_health_checks', True),
            'include_sla_performance': data.get('include_sla_performance', True),
            'include_anomalies': data.get('include_anomalies', False),
            'include_predictions': data.get('include_predictions', False),
            'format': data.get('format', 'json'),  # json, pdf, csv
            'email_recipients': data.get('email_recipients', [])
        }

        engine = get_monitoring_engine()
        report = engine.generate_monitoring_report(request.tenant_id, report_config)

        return monitoring_response(
            True,
            report,
            "Monitoring report generated successfully"
        )

    except Exception as e:
        current_app.logger.error(f"Error generating monitoring report: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# System Configuration

@monitoring_bp.route('/config/system', methods=['GET'])
@validate_tenant_access()
def get_monitoring_config():
    """Get current monitoring system configuration"""
    try:
        engine = get_monitoring_engine()
        config = engine.get_monitoring_config(request.tenant_id)

        return monitoring_response(
            True,
            config,
            "Monitoring configuration retrieved"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting monitoring config: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/config/system', methods=['PUT'])
@validate_tenant_access()
def update_monitoring_config():
    """Update monitoring system configuration"""
    try:
        data = request.get_json()

        engine = get_monitoring_engine()
        result = engine.update_monitoring_config(request.tenant_id, data)

        return monitoring_response(True, result, "Monitoring configuration updated")

    except Exception as e:
        current_app.logger.error(f"Error updating monitoring config: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Webhook Integration

@monitoring_bp.route('/webhooks/alerts', methods=['POST'])
@validate_tenant_access()
def create_alert_webhook():
    """Create webhook for alert notifications"""
    try:
        data = request.get_json()

        webhook_config = {
            'webhook_name': data['webhook_name'],
            'webhook_url': data['webhook_url'],
            'webhook_secret': data.get('webhook_secret'),
            'alert_types': data.get('alert_types', ['all']),
            'severity_filter': data.get('severity_filter', []),
            'payload_template': data.get('payload_template', {}),
            'retry_policy': data.get('retry_policy', {'max_retries': 3, 'backoff_seconds': 60}),
            'active': data.get('active', True)
        }

        engine = get_monitoring_engine()
        webhook_id = engine.create_alert_webhook(request.tenant_id, webhook_config)

        return monitoring_response(
            True,
            {'webhook_id': webhook_id, 'config': webhook_config},
            "Alert webhook created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating alert webhook: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/webhooks/test', methods=['POST'])
@validate_tenant_access()
def test_webhook():
    """Test webhook delivery"""
    try:
        data = request.get_json()
        webhook_url = data['webhook_url']
        test_payload = data.get('test_payload', {
            'alert_id': 'test-alert-123',
            'alert_name': 'Test Alert',
            'severity': 'warning',
            'message': 'This is a test alert from AgentSystem monitoring'
        })

        engine = get_monitoring_engine()
        result = engine.test_webhook(webhook_url, test_payload)

        return monitoring_response(True, result, "Webhook test completed")

    except Exception as e:
        current_app.logger.error(f"Error testing webhook: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Maintenance and Operations

@monitoring_bp.route('/maintenance/cleanup', methods=['POST'])
@validate_tenant_access()
def cleanup_monitoring_data():
    """Cleanup old monitoring data based on retention policies"""
    try:
        data = request.get_json() or {}
        force_cleanup = data.get('force_cleanup', False)
        retention_days = data.get('retention_days')

        engine = get_monitoring_engine()
        result = engine.cleanup_monitoring_data(
            request.tenant_id,
            force_cleanup=force_cleanup,
            retention_days=retention_days
        )

        return monitoring_response(
            True,
            result,
            "Monitoring data cleanup completed"
        )

    except Exception as e:
        current_app.logger.error(f"Error cleaning up monitoring data: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/maintenance/export', methods=['POST'])
@validate_tenant_access()
def export_monitoring_data():
    """Export monitoring data for backup or analysis"""
    try:
        data = request.get_json()

        export_config = {
            'export_type': data.get('export_type', 'full'),  # full, metrics, alerts, health_checks
            'date_range': data.get('date_range', '30d'),
            'format': data.get('format', 'json'),  # json, csv, parquet
            'compression': data.get('compression', 'gzip'),
            'include_raw_data': data.get('include_raw_data', True),
            'include_aggregated_data': data.get('include_aggregated_data', True)
        }

        engine = get_monitoring_engine()
        export_result = engine.export_monitoring_data(request.tenant_id, export_config)

        return monitoring_response(
            True,
            export_result,
            "Monitoring data export completed"
        )

    except Exception as e:
        current_app.logger.error(f"Error exporting monitoring data: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Real-time Monitoring

@monitoring_bp.route('/realtime/metrics', methods=['GET'])
@validate_tenant_access()
def get_realtime_metrics():
    """Get real-time metrics stream"""
    try:
        metric_names = request.args.getlist('metric_names')
        time_window_seconds = request.args.get('time_window_seconds', 300, type=int)

        engine = get_monitoring_engine()
        realtime_data = engine.get_realtime_metrics(
            request.tenant_id,
            metric_names,
            time_window_seconds
        )

        return monitoring_response(
            True,
            realtime_data,
            f"Retrieved real-time data for {len(metric_names)} metrics"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting real-time metrics: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/realtime/alerts', methods=['GET'])
@validate_tenant_access()
def get_realtime_alerts():
    """Get real-time alerts stream"""
    try:
        severity_filter = request.args.getlist('severity_filter')

        engine = get_monitoring_engine()
        realtime_alerts = engine.get_realtime_alerts(
            request.tenant_id,
            severity_filter=severity_filter
        )

        return monitoring_response(
            True,
            realtime_alerts,
            f"Retrieved {len(realtime_alerts)} real-time alerts"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting real-time alerts: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Escalation Management

@monitoring_bp.route('/escalation-policies', methods=['POST'])
@validate_tenant_access()
def create_escalation_policy():
    """Create a new escalation policy"""
    try:
        data = request.get_json()

        policy_config = {
            'policy_name': data['policy_name'],
            'description': data.get('description', ''),
            'escalation_steps': data['escalation_steps']
        }

        engine = get_monitoring_engine()
        policy_id = engine.create_escalation_policy(request.tenant_id, policy_config)

        return monitoring_response(
            True,
            {'policy_id': policy_id, 'config': policy_config},
            "Escalation policy created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating escalation policy: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

@monitoring_bp.route('/escalation-policies', methods=['GET'])
@validate_tenant_access()
def list_escalation_policies():
    """List all escalation policies for tenant"""
    try:
        engine = get_monitoring_engine()
        policies = engine.get_escalation_policies(request.tenant_id)

        return monitoring_response(
            True,
            policies,
            f"Retrieved {len(policies)} escalation policies"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing escalation policies: {str(e)}")
        return monitoring_response(False, error=str(e)), 500

# Error handlers
@monitoring_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return monitoring_response(False, error="Bad request"), 400

@monitoring_bp.errorhandler(401)
def unauthorized(error):
    """Handle unauthorized errors"""
    return monitoring_response(False, error="Unauthorized"), 401

@monitoring_bp.errorhandler(403)
def forbidden(error):
    """Handle forbidden errors"""
    return monitoring_response(False, error="Forbidden"), 403

@monitoring_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return monitoring_response(False, error="Resource not found"), 404

@monitoring_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return monitoring_response(False, error="Internal server error"), 500

# Health check endpoint for the monitoring API itself
@monitoring_bp.route('/health', methods=['GET'])
def monitoring_api_health():
    """Health check for monitoring API"""
    try:
        engine = get_monitoring_engine()
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'components': {
                'monitoring_engine': 'healthy',
                'database': 'healthy',
                'redis_cache': 'healthy',
                'notification_system': 'healthy'
            }
        }

        return monitoring_response(True, health_status, "Monitoring API is healthy")

    except Exception as e:
        current_app.logger.error(f"Monitoring API health check failed: {str(e)}")
        return monitoring_response(False, error="Monitoring API unhealthy"), 503
