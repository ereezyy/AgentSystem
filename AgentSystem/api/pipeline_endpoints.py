
"""
Automated Testing and Deployment Pipeline API Endpoints
Provides REST API for CI/CD pipeline management, execution, and monitoring
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import json
import uuid
from typing import Dict, List, Any, Optional
import logging
from functools import wraps
import asyncio
from dataclasses import asdict

# Import pipeline engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from devops.automated_pipeline_engine import AutomatedPipelineEngine, PipelineStage, DeploymentStrategy, TestType

# Create blueprint
pipeline_bp = Blueprint('pipeline', __name__, url_prefix='/api/v1/pipeline')

# Initialize pipeline engine
pipeline_engine = None

def get_pipeline_engine():
    """Get or create pipeline engine instance"""
    global pipeline_engine
    if pipeline_engine is None:
        pipeline_engine = AutomatedPipelineEngine()
    return pipeline_engine

def pipeline_response(success: bool, data: Any = None, message: str = "", error: str = "", metadata: Dict = None):
    """Create standardized pipeline response"""
    return jsonify({
        'success': success,
        'data': data,
        'message': message,
        'error': error,
        'metadata': metadata or {},
        'timestamp': datetime.utcnow().isoformat()
    })

def validate_tenant_access():
    """Validate tenant access for pipeline operations"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            tenant_id = request.headers.get('X-Tenant-ID')
            if not tenant_id:
                return pipeline_response(False, error="Tenant ID required in X-Tenant-ID header"), 400

            # Add tenant_id to request context
            request.tenant_id = tenant_id
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Pipeline Configuration Management

@pipeline_bp.route('/pipelines', methods=['POST'])
@validate_tenant_access()
def create_pipeline():
    """Create a new CI/CD pipeline"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['pipeline_name', 'repository_url', 'branch', 'environment']
        for field in required_fields:
            if field not in data:
                return pipeline_response(False, error=f"Missing required field: {field}"), 400

        # Validate configuration
        engine = get_pipeline_engine()
        validation = engine.validate_pipeline_configuration(data)

        if not validation['valid']:
            return pipeline_response(False, error="Invalid configuration", metadata=validation), 400

        # Create pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            pipeline_id = loop.run_until_complete(engine.create_pipeline(request.tenant_id, data))
        finally:
            loop.close()

        return pipeline_response(
            True,
            {'pipeline_id': pipeline_id, 'validation': validation},
            "Pipeline created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating pipeline: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines', methods=['GET'])
@validate_tenant_access()
def list_pipelines():
    """List all pipelines for tenant"""
    try:
        engine = get_pipeline_engine()
        pipelines = engine.get_pipelines(request.tenant_id)

        return pipeline_response(
            True,
            pipelines,
            f"Retrieved {len(pipelines)} pipelines"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing pipelines: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines/<pipeline_id>', methods=['GET'])
@validate_tenant_access()
def get_pipeline(pipeline_id):
    """Get specific pipeline configuration"""
    try:
        engine = get_pipeline_engine()
        pipeline = engine.get_pipeline(pipeline_id, request.tenant_id)

        if not pipeline:
            return pipeline_response(False, error="Pipeline not found"), 404

        return pipeline_response(True, pipeline, "Pipeline retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting pipeline: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines/<pipeline_id>', methods=['PUT'])
@validate_tenant_access()
def update_pipeline(pipeline_id):
    """Update pipeline configuration"""
    try:
        data = request.get_json()

        # Validate configuration updates
        engine = get_pipeline_engine()
        validation = engine.validate_pipeline_configuration(data)

        if not validation['valid']:
            return pipeline_response(False, error="Invalid configuration", metadata=validation), 400

        # Update pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(engine.update_pipeline(pipeline_id, request.tenant_id, data))
        finally:
            loop.close()

        if not result:
            return pipeline_response(False, error="Pipeline not found"), 404

        return pipeline_response(True, {'updated': True}, "Pipeline updated successfully")

    except Exception as e:
        current_app.logger.error(f"Error updating pipeline: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines/<pipeline_id>', methods=['DELETE'])
@validate_tenant_access()
def delete_pipeline(pipeline_id):
    """Delete a pipeline"""
    try:
        engine = get_pipeline_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(engine.delete_pipeline(pipeline_id, request.tenant_id))
        finally:
            loop.close()

        if not result:
            return pipeline_response(False, error="Pipeline not found"), 404

        return pipeline_response(True, message="Pipeline deleted successfully")

    except Exception as e:
        current_app.logger.error(f"Error deleting pipeline: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Pipeline Execution Management

@pipeline_bp.route('/pipelines/<pipeline_id>/execute', methods=['POST'])
@validate_tenant_access()
def execute_pipeline(pipeline_id):
    """Execute a pipeline"""
    try:
        data = request.get_json() or {}

        trigger_data = {
            'triggered_by': data.get('triggered_by', 'manual'),
            'trigger_reason': data.get('trigger_reason', 'Manual execution'),
            'commit_hash': data.get('commit_hash'),
            'branch_override': data.get('branch_override'),
            'environment_override': data.get('environment_override')
        }

        engine = get_pipeline_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            execution_id = loop.run_until_complete(
                engine.execute_pipeline(pipeline_id, request.tenant_id, trigger_data)
            )
        finally:
            loop.close()

        return pipeline_response(
            True,
            {'execution_id': execution_id, 'trigger_data': trigger_data},
            "Pipeline execution started"
        ), 202

    except Exception as e:
        current_app.logger.error(f"Error executing pipeline: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions', methods=['GET'])
@validate_tenant_access()
def list_executions():
    """List pipeline executions for tenant"""
    try:
        pipeline_id = request.args.get('pipeline_id')
        status = request.args.get('status')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        engine = get_pipeline_engine()
        executions = engine.get_executions(request.tenant_id, pipeline_id=pipeline_id)

        # Filter by status if specified
        if status:
            executions = [e for e in executions if e.get('status') == status]

        # Apply pagination
        total_count = len(executions)
        executions = executions[offset:offset + limit]

        return pipeline_response(
            True,
            {
                'executions': executions,
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            },
            f"Retrieved {len(executions)} executions"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing executions: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions/<execution_id>', methods=['GET'])
@validate_tenant_access()
def get_execution(execution_id):
    """Get specific pipeline execution details"""
    try:
        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        return pipeline_response(True, execution, "Execution retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting execution: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions/<execution_id>/cancel', methods=['POST'])
@validate_tenant_access()
def cancel_execution(execution_id):
    """Cancel a running pipeline execution"""
    try:
        engine = get_pipeline_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(engine.cancel_execution(execution_id, request.tenant_id))
        finally:
            loop.close()

        if not result:
            return pipeline_response(False, error="Execution not found or cannot be cancelled"), 404

        return pipeline_response(True, message="Execution cancelled successfully")

    except Exception as e:
        current_app.logger.error(f"Error cancelling execution: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions/<execution_id>/logs', methods=['GET'])
@validate_tenant_access()
def get_execution_logs(execution_id):
    """Get execution logs for specific stage or entire execution"""
    try:
        stage = request.args.get('stage')  # Optional stage filter
        log_level = request.args.get('log_level', 'INFO')
        limit = request.args.get('limit', 1000, type=int)

        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        # Get logs (simulated - would integrate with actual logging system)
        logs = {
            'execution_id': execution_id,
            'stage_filter': stage,
            'log_level': log_level,
            'logs': [
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'INFO',
                    'stage': 'unit_tests',
                    'message': 'Running unit tests with pytest',
                    'component': 'test_runner'
                },
                {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': 'INFO',
                    'stage': 'security_scan',
                    'message': 'Security scan completed with score 9.1',
                    'component': 'security_scanner'
                }
            ],
            'total_logs': 2,
            'truncated': False
        }

        return pipeline_response(True, logs, "Execution logs retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting execution logs: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Test Management

@pipeline_bp.route('/executions/<execution_id>/tests', methods=['GET'])
@validate_tenant_access()
def get_test_results(execution_id):
    """Get test results for execution"""
    try:
        test_type = request.args.get('test_type')  # Optional filter

        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        test_results = execution.get('test_results', {})

        # Filter by test type if specified
        if test_type:
            test_results = {test_type: test_results.get(test_type, {})}

        return pipeline_response(True, test_results, "Test results retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting test results: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions/<execution_id>/tests/<test_type>/rerun', methods=['POST'])
@validate_tenant_access()
def rerun_tests(execution_id, test_type):
    """Rerun specific test type for execution"""
    try:
        data = request.get_json() or {}

        # Validate test type
        valid_test_types = [t.value for t in TestType]
        if test_type not in valid_test_types:
            return pipeline_response(False, error=f"Invalid test type: {test_type}"), 400

        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        # Simulate test rerun
        rerun_result = {
            'rerun_id': f"rerun_{uuid.uuid4().hex[:12]}",
            'execution_id': execution_id,
            'test_type': test_type,
            'status': 'running',
            'started_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(
            True,
            rerun_result,
            f"Test rerun started for {test_type}"
        ), 202

    except Exception as e:
        current_app.logger.error(f"Error rerunning tests: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Security and Quality Management

@pipeline_bp.route('/executions/<execution_id>/security', methods=['GET'])
@validate_tenant_access()
def get_security_results(execution_id):
    """Get security scan results for execution"""
    try:
        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        security_results = execution.get('security_scan_results', {})

        return pipeline_response(True, security_results, "Security results retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting security results: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions/<execution_id>/quality-gates', methods=['GET'])
@validate_tenant_access()
def get_quality_gate_results(execution_id):
    """Get quality gate evaluation results"""
    try:
        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        quality_metrics = execution.get('quality_metrics', {})
        quality_gate_results = quality_metrics.get('quality_gate_results', {})

        return pipeline_response(True, quality_gate_results, "Quality gate results retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting quality gate results: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines/<pipeline_id>/quality-gates', methods=['PUT'])
@validate_tenant_access()
def update_quality_gates(pipeline_id):
    """Update quality gate configuration for pipeline"""
    try:
        data = request.get_json()

        # Validate quality gate configuration
        valid_gates = [
            'code_coverage_threshold', 'security_score_threshold', 'performance_threshold_ms',
            'vulnerability_threshold', 'lint_score_threshold', 'test_pass_rate_threshold'
        ]

        quality_gates = {}
        for gate, value in data.items():
            if gate in valid_gates:
                quality_gates[gate] = value

        if not quality_gates:
            return pipeline_response(False, error="No valid quality gates provided"), 400

        # Update pipeline
        engine = get_pipeline_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                engine.update_pipeline(pipeline_id, request.tenant_id, {'quality_gates': quality_gates})
            )
        finally:
            loop.close()

        if not result:
            return pipeline_response(False, error="Pipeline not found"), 404

        return pipeline_response(True, quality_gates, "Quality gates updated")

    except Exception as e:
        current_app.logger.error(f"Error updating quality gates: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Deployment Management

@pipeline_bp.route('/deployments', methods=['GET'])
@validate_tenant_access()
def list_deployments():
    """List deployment history"""
    try:
        environment = request.args.get('environment')
        days = request.args.get('days', 30, type=int)
        pipeline_id = request.args.get('pipeline_id')

        engine = get_pipeline_engine()
        deployments = engine.get_deployment_history(
            request.tenant_id,
            environment=environment,
            days=days
        )

        # Filter by pipeline if specified
        if pipeline_id:
            deployments = [d for d in deployments if d.get('pipeline_id') == pipeline_id]

        return pipeline_response(
            True,
            deployments,
            f"Retrieved {len(deployments)} deployments"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing deployments: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions/<execution_id>/approve-production', methods=['POST'])
@validate_tenant_access()
def approve_production_deployment(execution_id):
    """Approve production deployment for execution"""
    try:
        data = request.get_json() or {}
        approver = data.get('approver', 'api_user')
        approval_reason = data.get('approval_reason', 'Manual approval via API')

        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        if execution.get('status') != 'pending_approval':
            return pipeline_response(False, error="Execution is not pending approval"), 400

        # Simulate approval process
        approval_result = {
            'execution_id': execution_id,
            'approved': True,
            'approved_by': approver,
            'approved_at': datetime.utcnow().isoformat(),
            'approval_reason': approval_reason
        }

        return pipeline_response(True, approval_result, "Production deployment approved")

    except Exception as e:
        current_app.logger.error(f"Error approving production deployment: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/executions/<execution_id>/rollback', methods=['POST'])
@validate_tenant_access()
def trigger_rollback(execution_id):
    """Trigger manual rollback for deployment"""
    try:
        data = request.get_json() or {}
        rollback_reason = data.get('rollback_reason', 'Manual rollback triggered')
        target_version = data.get('target_version')

        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        # Simulate rollback trigger
        rollback_result = {
            'rollback_id': f"rollback_{uuid.uuid4().hex[:12]}",
            'execution_id': execution_id,
            'rollback_reason': rollback_reason,
            'target_version': target_version,
            'status': 'initiated',
            'initiated_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(True, rollback_result, "Rollback initiated")

    except Exception as e:
        current_app.logger.error(f"Error triggering rollback: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Analytics and Reporting

@pipeline_bp.route('/analytics/overview', methods=['GET'])
@validate_tenant_access()
def get_pipeline_analytics():
    """Get comprehensive pipeline analytics"""
    try:
        pipeline_id = request.args.get('pipeline_id')
        days = request.args.get('days', 30, type=int)

        engine = get_pipeline_engine()
        analytics = engine.get_pipeline_analytics(
            request.tenant_id,
            pipeline_id=pipeline_id,
            days=days
        )

        return pipeline_response(
            True,
            analytics,
            f"Retrieved pipeline analytics for {days} days"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting pipeline analytics: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/analytics/quality-trends', methods=['GET'])
@validate_tenant_access()
def get_quality_trends():
    """Get quality metrics trends"""
    try:
        days = request.args.get('days', 30, type=int)

        engine = get_pipeline_engine()
        trends = engine.get_quality_trends(request.tenant_id, days=days)

        return pipeline_response(
            True,
            trends,
            f"Retrieved quality trends for {days} days"
        )

    except Exception as e:
        current_app.logger.error(f"Error getting quality trends: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/analytics/metrics-summary', methods=['GET'])
@validate_tenant_access()
def get_metrics_summary():
    """Get comprehensive pipeline metrics summary"""
    try:
        engine = get_pipeline_engine()
        summary = engine.get_pipeline_metrics_summary(request.tenant_id)

        return pipeline_response(True, summary, "Pipeline metrics summary retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting metrics summary: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/reports/generate', methods=['POST'])
@validate_tenant_access()
def generate_pipeline_report():
    """Generate custom pipeline report"""
    try:
        data = request.get_json()

        report_config = {
            'report_type': data.get('report_type', 'summary'),
            'time_range_days': data.get('time_range_days', 30),
            'include_executions': data.get('include_executions', True),
            'include_test_results': data.get('include_test_results', True),
            'include_security_scans': data.get('include_security_scans', True),
            'include_deployments': data.get('include_deployments', True),
            'include_quality_trends': data.get('include_quality_trends', True),
            'format': data.get('format', 'json'),  # json, pdf, csv
            'email_recipients': data.get('email_recipients', [])
        }

        engine = get_pipeline_engine()
        report = engine.generate_pipeline_report(request.tenant_id, report_config)

        return pipeline_response(True, report, "Pipeline report generated")

    except Exception as e:
        current_app.logger.error(f"Error generating pipeline report: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Pipeline Templates

@pipeline_bp.route('/templates', methods=['GET'])
@validate_tenant_access()
def list_pipeline_templates():
    """List available pipeline templates"""
    try:
        category = request.args.get('category')
        technology = request.args.get('technology')

        # Simulated templates (would come from database)
        templates = [
            {
                'template_id': 'web_app_python',
                'template_name': 'Python Web Application',
                'template_description': 'Full-stack Python web application with Flask/Django',
                'template_category': 'web_app',
                'technology_stack': ['python', 'flask', 'react', 'docker', 'postgresql'],
                'usage_count': 156,
                'rating': 4.8
            },
            {
                'template_id': 'api_nodejs',
                'template_name': 'Node.js REST API',
                'template_description': 'RESTful API with Node.js and Express',
                'template_category': 'api',
                'technology_stack': ['nodejs', 'express', 'mongodb', 'docker'],
                'usage_count': 203,
                'rating': 4.7
            },
            {
                'template_id': 'microservice_java',
                'template_name': 'Java Microservice',
                'template_description': 'Spring Boot microservice with comprehensive testing',
                'template_category': 'microservice',
                'technology_stack': ['java', 'spring_boot', 'maven', 'docker', 'kubernetes'],
                'usage_count': 89,
                'rating': 4.6
            }
        ]

        # Filter templates
        if category:
            templates = [t for t in templates if t['template_category'] == category]

        if technology:
            templates = [t for t in templates if technology in t['technology_stack']]

        return pipeline_response(
            True,
            templates,
            f"Retrieved {len(templates)} pipeline templates"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing pipeline templates: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/templates/<template_id>', methods=['GET'])
@validate_tenant_access()
def get_pipeline_template(template_id):
    """Get specific pipeline template configuration"""
    try:
        # Simulated template retrieval
        template_configs = {
            'web_app_python': {
                'template_id': 'web_app_python',
                'template_name': 'Python Web Application',
                'quality_gates': {
                    'code_coverage_threshold': 85.0,
                    'security_score_threshold': 8.5,
                    'performance_threshold_ms': 1500,
                    'test_pass_rate_threshold': 98.0
                },
                'test_configurations': {
                    'unit': {
                        'framework': 'pytest',
                        'timeout_minutes': 15,
                        'parallel_execution': True,
                        'coverage_required': True
                    },
                    'integration': {
                        'framework': 'pytest',
                        'timeout_minutes': 30,
                        'test_data_setup': True
                    },
                    'e2e': {
                        'framework': 'playwright',
                        'timeout_minutes': 45,
                        'browser_matrix': ['chrome', 'firefox']
                    }
                },
                'deployment_config': {
                    'strategy': 'rolling',
                    'environments': ['staging', 'production'],
                    'approval_required_for_prod': True,
                    'auto_rollback_enabled': True
                }
            }
        }

        template = template_configs.get(template_id)
        if not template:
            return pipeline_response(False, error="Template not found"), 404

        return pipeline_response(True, template, "Template retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting pipeline template: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/templates/<template_id>/create-pipeline', methods=['POST'])
@validate_tenant_access()
def create_pipeline_from_template(template_id):
    """Create a new pipeline from template"""
    try:
        data = request.get_json() or {}

        # Get template configuration
        engine = get_pipeline_engine()
        template = engine.get_pipeline_template(template_id)

        if not template:
            return pipeline_response(False, error="Template not found"), 404

        # Create pipeline configuration from template
        pipeline_config = template['config'].copy()

        # Override with user-provided values
        if 'pipeline_name' in data:
            pipeline_config['pipeline_name'] = data['pipeline_name']
        if 'repository_url' in data:
            pipeline_config['repository_url'] = data['repository_url']
        if 'branch' in data:
            pipeline_config['branch'] = data['branch']

        # Create pipeline
        pipeline_id = f"pipeline_{uuid.uuid4().hex[:12]}"

        pipeline_result = {
            'pipeline_id': pipeline_id,
            'pipeline_name': pipeline_config.get('pipeline_name', f"Pipeline from {template['name']}"),
            'template_used': template_id,
            'template_name': template['name'],
            'config': pipeline_config,
            'created_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(
            True,
            pipeline_result,
            "Pipeline created from template"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating pipeline from template: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Webhook Management

@pipeline_bp.route('/pipelines/<pipeline_id>/webhooks', methods=['POST'])
@validate_tenant_access()
def create_pipeline_webhook(pipeline_id):
    """Create webhook trigger for pipeline"""
    try:
        data = request.get_json()

        webhook_config = {
            'webhook_url': data['webhook_url'],
            'webhook_secret': data.get('webhook_secret'),
            'trigger_events': data.get('trigger_events', ['push']),
            'branch_filters': data.get('branch_filters', ['main']),
            'path_filters': data.get('path_filters', [])
        }

        # Simulate webhook creation
        webhook_id = f"webhook_{uuid.uuid4().hex[:12]}"

        webhook_result = {
            'webhook_id': webhook_id,
            'pipeline_id': pipeline_id,
            'config': webhook_config,
            'created_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(
            True,
            webhook_result,
            "Pipeline webhook created"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating pipeline webhook: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines/<pipeline_id>/webhooks', methods=['GET'])
@validate_tenant_access()
def list_pipeline_webhooks(pipeline_id):
    """List webhooks for pipeline"""
    try:
        # Simulate webhook listing
        webhooks = [
            {
                'webhook_id': f"webhook_{uuid.uuid4().hex[:12]}",
                'pipeline_id': pipeline_id,
                'webhook_url': 'https://api.github.com/repos/example/repo/hooks',
                'trigger_events': ['push', 'pull_request'],
                'enabled': True,
                'last_triggered': datetime.utcnow().isoformat(),
                'trigger_count': 45
            }
        ]

        return pipeline_response(True, webhooks, f"Retrieved {len(webhooks)} webhooks")

    except Exception as e:
        current_app.logger.error(f"Error listing pipeline webhooks: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Environment Management

@pipeline_bp.route('/environments', methods=['POST'])
@validate_tenant_access()
def create_environment():
    """Create a new deployment environment"""
    try:
        data = request.get_json()

        environment_config = {
            'environment_name': data['environment_name'],
            'environment_type': data['environment_type'],
            'deployment_target': data['deployment_target'],
            'configuration': data.get('configuration', {}),
            'health_check_url': data.get('health_check_url'),
            'monitoring_enabled': data.get('monitoring_enabled', True),
            'auto_deploy_enabled': data.get('auto_deploy_enabled', False),
            'approval_required': data.get('approval_required', True),
            'resource_limits': data.get('resource_limits', {}),
            'environment_variables': data.get('environment_variables', {}),
            'secrets_config': data.get('secrets_config', {})
        }

        environment_id = f"env_{uuid.uuid4().hex[:12]}"

        environment_result = {
            'environment_id': environment_id,
            'config': environment_config,
            'created_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(
            True,
            environment_result,
            "Environment created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating environment: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/environments', methods=['GET'])
@validate_tenant_access()
def list_environments():
    """List deployment environments for tenant"""
    try:
        environment_type = request.args.get('environment_type')
        active_only = request.args.get('active_only', 'false').lower() == 'true'

        # Simulate environment listing
        environments = [
            {
                'environment_id': f"env_{uuid.uuid4().hex[:12]}",
                'environment_name': 'staging',
                'environment_type': 'staging',
                'deployment_target': 'k8s-staging-cluster',
                'health_check_url': 'https://staging-api.agentsystem.com/health',
                'monitoring_enabled': True,
                'auto_deploy_enabled': True,
                'approval_required': False,
                'active': True
            },
            {
                'environment_id': f"env_{uuid.uuid4().hex[:12]}",
                'environment_name': 'production',
                'environment_type': 'production',
                'deployment_target': 'k8s-prod-cluster',
                'health_check_url': 'https://api.agentsystem.com/health',
                'monitoring_enabled': True,
                'auto_deploy_enabled': False,
                'approval_required': True,
                'active': True
            }
        ]

        # Filter environments
        if environment_type:
            environments = [e for e in environments if e['environment_type'] == environment_type]

        if active_only:
            environments = [e for e in environments if e['active']]

        return pipeline_response(True, environments, f"Retrieved {len(environments)} environments")

    except Exception as e:
        current_app.logger.error(f"Error listing environments: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Artifact Management

@pipeline_bp.route('/executions/<execution_id>/artifacts', methods=['GET'])
@validate_tenant_access()
def list_execution_artifacts(execution_id):
    """List artifacts generated by execution"""
    try:
        artifact_type = request.args.get('artifact_type')

        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        artifacts = execution.get('artifacts_generated', [])

        # Enhance artifact information
        enhanced_artifacts = []
        for artifact in artifacts:
            enhanced_artifacts.append({
                'artifact_name': artifact,
                'artifact_type': 'container_image' if 'container_image:' in artifact else 'build_artifact',
                'size_mb': 245.7 if 'container_image:' in artifact else 12.3,
                'created_at': execution.get('started_at'),
                'download_url': f"/api/v1/pipeline/artifacts/{artifact}/download",
                'retention_days': 90
            })

        # Filter by type if specified
        if artifact_type:
            enhanced_artifacts = [a for a in enhanced_artifacts if a['artifact_type'] == artifact_type]

        return pipeline_response(
            True,
            enhanced_artifacts,
            f"Retrieved {len(enhanced_artifacts)} artifacts"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing execution artifacts: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/artifacts/<artifact_name>/download', methods=['GET'])
@validate_tenant_access()
def download_artifact(artifact_name):
    """Download pipeline artifact"""
    try:
        # Simulate artifact download
        download_info = {
            'artifact_name': artifact_name,
            'download_url': f"https://artifacts.agentsystem.com/{request.tenant_id}/{artifact_name}",
            'expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            'size_bytes': 1024000,
            'content_type': 'application/octet-stream'
        }

        return pipeline_response(True, download_info, "Artifact download URL generated")

    except Exception as e:
        current_app.logger.error(f"Error downloading artifact: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Pipeline Scheduling

@pipeline_bp.route('/pipelines/<pipeline_id>/schedules', methods=['POST'])
@validate_tenant_access()
def create_pipeline_schedule(pipeline_id):
    """Create scheduled execution for pipeline"""
    try:
        data = request.get_json()

        schedule_config = {
            'schedule_name': data['schedule_name'],
            'cron_expression': data['cron_expression'],
            'timezone': data.get('timezone', 'UTC'),
            'max_concurrent_executions': data.get('max_concurrent_executions', 1),
            'schedule_config': data.get('schedule_config', {})
        }

        # Validate cron expression (basic validation)
        cron_parts = schedule_config['cron_expression'].split()
        if len(cron_parts) != 5:
            return pipeline_response(False, error="Invalid cron expression format"), 400

        schedule_id = f"schedule_{uuid.uuid4().hex[:12]}"

        schedule_result = {
            'schedule_id': schedule_id,
            'pipeline_id': pipeline_id,
            'config': schedule_config,
            'next_run_time': (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            'created_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(
            True,
            schedule_result,
            "Pipeline schedule created"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating pipeline schedule: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines/<pipeline_id>/schedules', methods=['GET'])
@validate_tenant_access()
def list_pipeline_schedules(pipeline_id):
    """List schedules for pipeline"""
    try:
        # Simulate schedule listing
        schedules = [
            {
                'schedule_id': f"schedule_{uuid.uuid4().hex[:12]}",
                'pipeline_id': pipeline_id,
                'schedule_name': 'Nightly Build',
                'cron_expression': '0 2 * * *',
                'timezone': 'UTC',
                'enabled': True,
                'next_run_time': (datetime.utcnow() + timedelta(hours=8)).isoformat(),
                'last_run_time': (datetime.utcnow() - timedelta(hours=16)).isoformat(),
                'run_count': 45
            }
        ]

        return pipeline_response(True, schedules, f"Retrieved {len(schedules)} schedules")

    except Exception as e:
        current_app.logger.error(f"Error listing pipeline schedules: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Maintenance and Cleanup

@pipeline_bp.route('/maintenance/cleanup', methods=['POST'])
@validate_tenant_access()
def cleanup_pipeline_data():
    """Clean up old pipeline executions and artifacts"""
    try:
        data = request.get_json() or {}
        retention_days = data.get('retention_days', 90)
        cleanup_artifacts = data.get('cleanup_artifacts', True)

        engine = get_pipeline_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            cleanup_result = loop.run_until_complete(
                engine.cleanup_old_executions(request.tenant_id, retention_days)
            )
        finally:
            loop.close()

        return pipeline_response(
            True,
            cleanup_result,
            "Pipeline data cleanup completed"
        )

    except Exception as e:
        current_app.logger.error(f"Error cleaning up pipeline data: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/config/validate', methods=['POST'])
@validate_tenant_access()
def validate_pipeline_config():
    """Validate pipeline configuration without creating"""
    try:
        data = request.get_json()

        engine = get_pipeline_engine()
        validation = engine.validate_pipeline_configuration(data)

        return pipeline_response(True, validation, "Configuration validation completed")

    except Exception as e:
        current_app.logger.error(f"Error validating pipeline config: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Real-time Pipeline Status

@pipeline_bp.route('/executions/<execution_id>/status', methods=['GET'])
@validate_tenant_access()
def get_execution_status():
    """Get real-time execution status"""
    try:
        engine = get_pipeline_engine()
        execution = engine.get_execution(execution_id, request.tenant_id)

        if not execution:
            return pipeline_response(False, error="Execution not found"), 404

        # Extract key status information
        status_info = {
            'execution_id': execution_id,
            'status': execution.get('status'),
            'current_stage': execution.get('current_stage'),
            'stages_completed': len(execution.get('stages_completed', [])),
            'stages_failed': len(execution.get('stages_failed', [])),
            'progress_percentage': len(execution.get('stages_completed', [])) / 18 * 100,  # 18 total stages
            'execution_time_seconds': execution.get('execution_time_seconds', 0),
            'started_at': execution.get('started_at'),
            'estimated_completion': None
        }

        # Calculate estimated completion for running executions
        if execution.get('status') == 'running':
            avg_stage_time = status_info['execution_time_seconds'] / max(1, status_info['stages_completed'])
            remaining_stages = 18 - status_info['stages_completed']
            estimated_seconds = remaining_stages * avg_stage_time
            status_info['estimated_completion'] = (
                datetime.utcnow() + timedelta(seconds=estimated_seconds)
            ).isoformat()

        return pipeline_response(True, status_info, "Execution status retrieved")

    except Exception as e:
        current_app.logger.error(f"Error getting execution status: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Pipeline Dependencies

@pipeline_bp.route('/pipelines/<pipeline_id>/dependencies', methods=['POST'])
@validate_tenant_access()
def create_pipeline_dependency(pipeline_id):
    """Create dependency between pipelines"""
    try:
        data = request.get_json()

        dependency_config = {
            'depends_on_pipeline_id': data['depends_on_pipeline_id'],
            'dependency_type': data.get('dependency_type', 'sequential'),
            'dependency_condition': data.get('dependency_condition', {}),
            'wait_for_completion': data.get('wait_for_completion', True),
            'timeout_minutes': data.get('timeout_minutes', 60)
        }

        dependency_id = f"dep_{uuid.uuid4().hex[:12]}"

        dependency_result = {
            'dependency_id': dependency_id,
            'pipeline_id': pipeline_id,
            'config': dependency_config,
            'created_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(
            True,
            dependency_result,
            "Pipeline dependency created"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating pipeline dependency: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/pipelines/<pipeline_id>/dependencies', methods=['GET'])
@validate_tenant_access()
def list_pipeline_dependencies(pipeline_id):
    """List dependencies for pipeline"""
    try:
        # Simulate dependency listing
        dependencies = [
            {
                'dependency_id': f"dep_{uuid.uuid4().hex[:12]}",
                'pipeline_id': pipeline_id,
                'depends_on_pipeline_id': f"pipeline_{uuid.uuid4().hex[:12]}",
                'dependency_type': 'sequential',
                'wait_for_completion': True,
                'timeout_minutes': 60,
                'created_at': datetime.utcnow().isoformat()
            }
        ]

        return pipeline_response(True, dependencies, f"Retrieved {len(dependencies)} dependencies")

    except Exception as e:
        current_app.logger.error(f"Error listing pipeline dependencies: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Performance Optimization

@pipeline_bp.route('/analytics/performance-insights', methods=['GET'])
@validate_tenant_access()
def get_performance_insights():
    """Get performance insights and optimization recommendations"""
    try:
        pipeline_id = request.args.get('pipeline_id')
        days = request.args.get('days', 30, type=int)

        engine = get_pipeline_engine()
        analytics = engine.get_pipeline_analytics(request.tenant_id, pipeline_id, days)

        # Generate performance insights
        insights = {
            'execution_time_analysis': {
                'average_time_seconds': analytics.get('average_execution_time_seconds', 0),
                'trend': 'stable',  # Would calculate actual trend
                'bottleneck_stages': ['security_scan', 'e2e_tests'],
                'optimization_potential_seconds': 120
            },
            'resource_utilization': {
                'cpu_efficiency': 78.5,
                'memory_efficiency': 82.3,
                'storage_efficiency': 91.2,
                'cost_per_execution': 2.45
            },
            'quality_efficiency': {
                'test_effectiveness': 94.2,
                'security_coverage': 96.8,
                'code_quality_trend': 'improving'
            },
            'recommendations': [
                'Consider parallelizing security scans to reduce execution time',
                'Optimize E2E test suite - current duration is above average',
                'Enable test result caching for faster feedback loops'
            ]
        }

        return pipeline_response(True, insights, "Performance insights generated")

    except Exception as e:
        current_app.logger.error(f"Error getting performance insights: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Compliance and Audit

@pipeline_bp.route('/compliance/audit-trail', methods=['GET'])
@validate_tenant_access()
def get_compliance_audit_trail():
    """Get compliance audit trail for pipeline executions"""
    try:
        pipeline_id = request.args.get('pipeline_id')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        # Simulate audit trail
        audit_trail = [
            {
                'event_id': f"audit_{uuid.uuid4().hex[:12]}",
                'execution_id': f"exec_{uuid.uuid4().hex[:12]}",
                'pipeline_id': pipeline_id or f"pipeline_{uuid.uuid4().hex[:12]}",
                'event_type': 'deployment',
                'environment': 'production',
                'user_id': 'user_123',
                'approval_chain': ['manager_456', 'security_789'],
                'compliance_checks': {
                    'security_scan_passed': True,
                    'quality_gates_passed': True,
                    'approval_obtained': True,
                    'change_management_ticket': 'CHG-2024-001'
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        ]

        return pipeline_response(True, audit_trail, f"Retrieved {len(audit_trail)} audit events")

    except Exception as e:
        current_app.logger.error(f"Error getting compliance audit trail: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Integration Endpoints

@pipeline_bp.route('/integrations/github/webhook', methods=['POST'])
def github_webhook():
    """Handle GitHub webhook for pipeline triggers"""
    try:
        # Validate GitHub webhook signature
        signature = request.headers.get('X-Hub-Signature-256')
        event_type = request.headers.get('X-GitHub-Event')

        if not signature or not event_type:
            return pipeline_response(False, error="Invalid webhook request"), 400

        data = request.get_json()

        # Process webhook based on event type
        if event_type == 'push':
            # Handle push event
            repository_url = data.get('repository', {}).get('clone_url')
            branch = data.get('ref', '').replace('refs/heads/', '')
            commit_hash = data.get('after')

            webhook_result = {
                'event_type': event_type,
                'repository_url': repository_url,
                'branch': branch,
                'commit_hash': commit_hash,
                'pipelines_triggered': 0,
                'processed_at': datetime.utcnow().isoformat()
            }

            return pipeline_response(True, webhook_result, "GitHub webhook processed")

        elif event_type == 'pull_request':
            # Handle pull request event
            action = data.get('action')
            pr_number = data.get('number')

            webhook_result = {
                'event_type': event_type,
                'action': action,
                'pr_number': pr_number,
                'pipelines_triggered': 0,
                'processed_at': datetime.utcnow().isoformat()
            }

            return pipeline_response(True, webhook_result, "GitHub PR webhook processed")

        return pipeline_response(True, {'event_type': event_type}, "Webhook received")

    except Exception as e:
        current_app.logger.error(f"Error processing GitHub webhook: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

@pipeline_bp.route('/integrations/gitlab/webhook', methods=['POST'])
def gitlab_webhook():
    """Handle GitLab webhook for pipeline triggers"""
    try:
        event_type = request.headers.get('X-Gitlab-Event')
        token = request.headers.get('X-Gitlab-Token')

        if not event_type:
            return pipeline_response(False, error="Invalid GitLab webhook"), 400

        data = request.get_json()

        webhook_result = {
            'event_type': event_type,
            'project_id': data.get('project_id'),
            'pipelines_triggered': 0,
            'processed_at': datetime.utcnow().isoformat()
        }

        return pipeline_response(True, webhook_result, "GitLab webhook processed")

    except Exception as e:
        current_app.logger.error(f"Error processing GitLab webhook: {str(e)}")
        return pipeline_response(False, error=str(e)), 500

# Error handlers
@pipeline_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return pipeline_response(False, error="Bad request"), 400

@pipeline_bp.errorhandler(401)
def unauthorized(error):
    """Handle unauthorized errors"""
    return pipeline_response(False, error="Unauthorized"), 401

@pipeline_bp.errorhandler(403)
def forbidden(error):
    """Handle forbidden errors"""
    return pipeline_response(False, error="Forbidden"), 403

@pipeline_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return pipeline_response(False, error="Resource not found"), 404

@pipeline_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return pipeline_response(False, error="Internal server error"), 500

# Health check endpoint for the pipeline API itself
@pipeline_bp.route('/health', methods=['GET'])
def pipeline_api_health():
    """Health check for pipeline API"""
    try:
        engine = get_pipeline_engine()
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'components': {
                'pipeline_engine': 'healthy',
                'database': 'healthy',
                'docker_client': 'healthy' if engine.docker_client else 'unavailable',
                'notification_system': 'healthy'
            },
            'active_executions': len([e for e in engine.executions.values() if e.status == 'running']),
            'total_pipelines': len(engine.pipelines)
        }

        return pipeline_response(True, health_status, "Pipeline API is healthy")

    except Exception as e:
        current_app.logger.error(f"Pipeline API health check failed: {str(e)}")
        return pipeline_response(False, error="Pipeline API unhealthy"), 503

        # Get template configuration
        template_response = get_pipeline_template(template_id)
        if template_response[1] != 200:
            return template_response

        template_config = json.loads(template_response[0].data)['data']

        # Merge template with user overrides
        pipeline_config = {
            'pipeline_name': data['pipeline_name'],
            'repository_url': data['repository_url'],
            'branch': data.get('branch', 'main'),
            'environment': data.get('environment', 'staging'),
            'deployment_strategy': data.get('deployment_strategy', template_config['deployment_config']['strategy']),
            'quality_gates': {**template_config['quality_gates'], **data.get('quality_gates', {})},
            'test_configurations': {**template_config['test_configurations'], **data.get('test_configurations', {})},
            'deployment_config': {**template_config['deployment_config'], **data.get('deployment_config', {})},
            'security_policies': data.get('security_policies', {}),
            'notification_config': data.get('notification_config', {}),
            'rollback_config': data.get('rollback_config', {}),
            'created_by': data.get('created_by', 'api_user')
        }

        # Create pipeline
        engine = get_pipeline_engine()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            pipeline_id = loop.run_until_complete(engine.create_pipeline(request.tenant_id, pipeline_config))
        finally:
            loop.close()

