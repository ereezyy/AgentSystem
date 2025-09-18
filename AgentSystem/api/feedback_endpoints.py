"""
Customer Feedback and Feature Request API Endpoints

This module provides REST API endpoints for submitting, managing, and analyzing
customer feedback and feature requests in a multi-tenant SaaS environment.
"""

from flask import Blueprint, request, current_app
from datetime import datetime
import asyncio
import uuid
from ..utils.security import validate_tenant_access

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/v1/feedback')

def get_feedback_engine():
    """Helper to get the feedback engine instance from the app context"""
    return current_app.feedback_engine

def feedback_response(success: bool, data=None, message: str = "", error: str = ""):
    """
    Standard response format for feedback API endpoints

    Args:
        success: Boolean indicating operation success
        data: Response data payload
        message: Success message
        error: Error message if operation failed

    Returns:
        dict: Standardized response dictionary
    """
    response = {"success": success, "message": message}
    if data is not None:
        response["data"] = data
    if error:
        response["error"] = error
    return response

# Feedback Submission Endpoints

@feedback_bp.route('/submit', methods=['POST'])
@validate_tenant_access()
def submit_feedback():
    """Submit new customer feedback"""
    try:
        data = request.get_json()

        required_fields = ['title', 'description']
        if not all(field in data for field in required_fields):
            return feedback_response(False, error="Missing required fields: title and description"), 400

        feedback_id = asyncio.run(
            get_feedback_engine().submit_feedback(
                request.tenant_id,
                request.user_id,
                data
            )
        )

        return feedback_response(
            True,
            {'feedback_id': feedback_id},
            "Feedback submitted successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error submitting feedback: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/<feedback_id>', methods=['GET'])
@validate_tenant_access()
def get_feedback(feedback_id):
    """Retrieve specific feedback details"""
    try:
        feedback = get_feedback_engine().get_feedback(feedback_id, request.tenant_id)

        if not feedback:
            return feedback_response(False, error="Feedback not found or access denied"), 404

        return feedback_response(True, feedback, "Feedback retrieved successfully")

    except Exception as e:
        current_app.logger.error(f"Error retrieving feedback: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/list', methods=['GET'])
@validate_tenant_access()
def list_feedback():
    """List feedback submissions with optional filters"""
    try:
        filters = {}
        if 'category' in request.args:
            filters['category'] = request.args.get('category')
        if 'status' in request.args:
            filters['status'] = request.args.get('status')
        if 'priority' in request.args:
            filters['priority'] = request.args.get('priority')
        if 'sentiment' in request.args:
            filters['sentiment'] = request.args.get('sentiment')

        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        feedback_list = get_feedback_engine().list_feedback(request.tenant_id, filters)

        # Apply pagination
        total_items = len(feedback_list)
        feedback_list = feedback_list[offset:offset + limit]

        pagination = {
            'total_items': total_items,
            'returned_items': len(feedback_list),
            'limit': limit,
            'offset': offset,
            'has_more': offset + limit < total_items
        }

        return feedback_response(
            True,
            {
                'feedback': feedback_list,
                'pagination': pagination
            },
            f"Retrieved {len(feedback_list)} feedback items"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing feedback: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/<feedback_id>/status', methods=['PUT'])
@validate_tenant_access()
def update_feedback_status(feedback_id):
    """Update the status of a feedback item"""
    try:
        data = request.get_json()
        status = data.get('status')
        notes = data.get('notes', '')

        if not status:
            return feedback_response(False, error="Status is required"), 400

        success = asyncio.run(
            get_feedback_engine().update_feedback_status(
                feedback_id,
                request.tenant_id,
                status,
                notes
            )
        )

        if not success:
            return feedback_response(False, error="Feedback not found or access denied"), 404

        return feedback_response(True, message="Feedback status updated successfully")

    except Exception as e:
        current_app.logger.error(f"Error updating feedback status: {str(e)}")
        return feedback_response(False, error=str(e)), 500

# Feature Request Endpoints

@feedback_bp.route('/features/submit', methods=['POST'])
@validate_tenant_access()
def submit_feature_request():
    """Submit a new feature request"""
    try:
        data = request.get_json()

        required_fields = ['title', 'description']
        if not all(field in data for field in required_fields):
            return feedback_response(False, error="Missing required fields: title and description"), 400

        request_id = asyncio.run(
            get_feedback_engine().submit_feature_request(
                request.tenant_id,
                request.user_id,
                data
            )
        )

        return feedback_response(
            True,
            {'request_id': request_id},
            "Feature request submitted successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error submitting feature request: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/features/<request_id>', methods=['GET'])
@validate_tenant_access()
def get_feature_request(request_id):
    """Retrieve specific feature request details"""
    try:
        feature_request = get_feedback_engine().get_feature_request(request_id, request.tenant_id)

        if not feature_request:
            return feedback_response(False, error="Feature request not found or access denied"), 404

        return feedback_response(True, feature_request, "Feature request retrieved successfully")

    except Exception as e:
        current_app.logger.error(f"Error retrieving feature request: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/features/list', methods=['GET'])
@validate_tenant_access()
def list_feature_requests():
    """List feature requests with optional filters"""
    try:
        filters = {}
        if 'status' in request.args:
            filters['status'] = request.args.get('status')
        if 'priority' in request.args:
            filters['priority'] = request.args.get('priority')

        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        sort_by = request.args.get('sort_by', 'votes')  # votes or impact_score

        feature_list = get_feedback_engine().list_feature_requests(request.tenant_id, filters)

        # Apply custom sorting if needed
        if sort_by == 'impact_score':
            feature_list = sorted(feature_list, key=lambda x: x['impact_score'], reverse=True)

        # Apply pagination
        total_items = len(feature_list)
        feature_list = feature_list[offset:offset + limit]

        pagination = {
            'total_items': total_items,
            'returned_items': len(feature_list),
            'limit': limit,
            'offset': offset,
            'has_more': offset + limit < total_items
        }

        return feedback_response(
            True,
            {
                'feature_requests': feature_list,
                'pagination': pagination
            },
            f"Retrieved {len(feature_list)} feature requests"
        )

    except Exception as e:
        current_app.logger.error(f"Error listing feature requests: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/features/<request_id>/vote', methods=['POST'])
@validate_tenant_access()
def vote_feature_request(request_id):
    """Vote for a feature request to increase its priority"""
    try:
        success = asyncio.run(
            get_feedback_engine().vote_feature_request(
                request.tenant_id,
                request.user_id,
                request_id
            )
        )

        if not success:
            return feedback_response(False, error="Feature request not found or already voted"), 404

        return feedback_response(True, message="Vote recorded successfully")

    except Exception as e:
        current_app.logger.error(f"Error voting for feature request: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/features/<request_id>/status', methods=['PUT'])
@validate_tenant_access()
def update_feature_status(request_id):
    """Update the status of a feature request"""
    try:
        data = request.get_json()
        status = data.get('status')
        notes = data.get('notes', '')

        if not status:
            return feedback_response(False, error="Status is required"), 400

        success = asyncio.run(
            get_feedback_engine().update_feature_status(
                request_id,
                request.tenant_id,
                status,
                notes
            )
        )

        if not success:
            return feedback_response(False, error="Feature request not found or access denied"), 404

        return feedback_response(True, message="Feature request status updated successfully")

    except Exception as e:
        current_app.logger.error(f"Error updating feature request status: {str(e)}")
        return feedback_response(False, error=str(e)), 500

# Analysis Endpoints

@feedback_bp.route('/analysis/feedback-trends', methods=['GET'])
@validate_tenant_access()
def analyze_feedback_trends():
    """Analyze feedback trends over a specified time period"""
    try:
        time_range_days = request.args.get('time_range_days', 30, type=int)

        analysis = get_feedback_engine().analyze_feedback_trends(request.tenant_id, time_range_days)

        return feedback_response(True, analysis, "Feedback trends analysis completed")

    except Exception as e:
        current_app.logger.error(f"Error analyzing feedback trends: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/analysis/feature-demand', methods=['GET'])
@validate_tenant_access()
def analyze_feature_demand():
    """Analyze demand and prioritization for feature requests"""
    try:
        analysis = get_feedback_engine().analyze_feature_demand(request.tenant_id)

        return feedback_response(True, analysis, "Feature demand analysis completed")

    except Exception as e:
        current_app.logger.error(f"Error analyzing feature demand: {str(e)}")
        return feedback_response(False, error=str(e)), 500

# Category Management Endpoints

@feedback_bp.route('/categories', methods=['GET'])
@validate_tenant_access()
def list_feedback_categories():
    """List available feedback categories for the tenant"""
    try:
        # This would normally query the database, but for now we'll return a static list
        categories = [
            {
                'category_id': 'cat_general',
                'category_name': 'General',
                'description': 'General feedback about the product or service',
                'is_default': True
            },
            {
                'category_id': 'cat_bug',
                'category_name': 'Bug',
                'description': 'Issues or bugs encountered while using the product',
                'is_default': True
            },
            {
                'category_id': 'cat_feature',
                'category_name': 'Feature Request',
                'description': 'Suggestions for new features or enhancements',
                'is_default': True
            },
            {
                'category_id': 'cat_ui',
                'category_name': 'User Interface',
                'description': 'Feedback related to UI/UX design',
                'is_default': True
            },
            {
                'category_id': 'cat_performance',
                'category_name': 'Performance',
                'description': 'Feedback related to speed and performance',
                'is_default': True
            },
            {
                'category_id': 'cat_support',
                'category_name': 'Customer Support',
                'description': 'Feedback about customer support experiences',
                'is_default': True
            }
        ]

        return feedback_response(True, categories, f"Retrieved {len(categories)} feedback categories")

    except Exception as e:
        current_app.logger.error(f"Error listing feedback categories: {str(e)}")
        return feedback_response(False, error=str(e)), 500

@feedback_bp.route('/categories', methods=['POST'])
@validate_tenant_access()
def create_feedback_category():
    """Create a custom feedback category for the tenant"""
    try:
        data = request.get_json()
        category_name = data.get('category_name')
        description = data.get('description', '')

        if not category_name:
            return feedback_response(False, error="Category name is required"), 400

        category_id = f"cat_{uuid.uuid4().hex[:12]}"
        category = {
            'category_id': category_id,
            'category_name': category_name,
            'description': description,
            'is_default': False,
            'created_at': datetime.utcnow().isoformat()
        }

        return feedback_response(
            True,
            category,
            "Feedback category created successfully"
        ), 201

    except Exception as e:
        current_app.logger.error(f"Error creating feedback category: {str(e)}")
        return feedback_response(False, error=str(e)), 500

# Error handlers
@feedback_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return feedback_response(False, error="Bad request"), 400

@feedback_bp.errorhandler(401)
def unauthorized(error):
    """Handle unauthorized errors"""
    return feedback_response(False, error="Unauthorized"), 401

@feedback_bp.errorhandler(403)
def forbidden(error):
    """Handle forbidden errors"""
    return feedback_response(False, error="Forbidden"), 403

@feedback_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return feedback_response(False, error="Resource not found"), 404

@feedback_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return feedback_response(False, error="Internal server error"), 500

# Health check endpoint for the feedback API
@feedback_bp.route('/health', methods=['GET'])
def feedback_api_health():
    """Health check for feedback API"""
    try:
        engine = get_feedback_engine()
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'components': {
                'feedback_engine': 'healthy',
                'database': 'healthy' if hasattr(engine, 'db_connection') and engine.db_connection else 'unavailable',
                'notification_system': 'healthy' if hasattr(engine, 'notification_service') and engine.notification_service else 'unavailable'
            },
            'total_feedback': len(engine.feedback_store),
            'total_feature_requests': len(engine.feature_requests)
        }

        return feedback_response(True, health_status, "Feedback API is healthy")

    except Exception as e:
        current_app.logger.error(f"Feedback API health check failed: {str(e)}")
        return feedback_response(False, error="Feedback API unhealthy"), 503
