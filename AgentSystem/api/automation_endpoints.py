"""
Workflow Automation Platform API Endpoints - AgentSystem Profit Machine
RESTful API for no-code workflow automation with visual builder and execution engine
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import json
import logging

from ..auth.auth_middleware import verify_token, get_current_tenant
from ..automation.workflow_platform import WorkflowPlatform, create_workflow_platform
from ..database.connection import get_db_connection
from ..models.api_models import StandardResponse, PaginatedResponse
from ..usage.usage_tracker import UsageTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/automation", tags=["Workflow Automation"])
security = HTTPBearer()

# Initialize workflow platform
workflow_platform = create_workflow_platform()

@router.on_event("startup")
async def startup_event():
    """Initialize workflow platform on startup"""
    await workflow_platform.initialize()

# Workflow Management Endpoints

@router.post("/workflows", response_model=StandardResponse)
async def create_workflow(
    workflow_spec: Dict[str, Any],
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create a new workflow"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate workflow specification
        required_fields = ['name', 'trigger', 'nodes']
        for field in required_fields:
            if field not in workflow_spec:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )

        # Create workflow
        workflow_id = await workflow_platform.create_workflow(tenant_id, workflow_spec)

        return StandardResponse(
            success=True,
            message="Workflow created successfully",
            data={"workflow_id": str(workflow_id)}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@router.get("/workflows", response_model=PaginatedResponse)
async def get_workflows(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get workflows for tenant"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit

        async with get_db_connection() as conn:
            # Build query
            where_conditions = ["tenant_id = $1"]
            params = [tenant_id]
            param_count = 1

            if category:
                param_count += 1
                where_conditions.append(f"category = ${param_count}")
                params.append(category)

            if status:
                param_count += 1
                where_conditions.append(f"status = ${param_count}")
                params.append(status)

            if search:
                param_count += 1
                where_conditions.append(f"(name ILIKE ${param_count} OR description ILIKE ${param_count})")
                params.append(f"%{search}%")

            where_clause = " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM automation.workflows
                WHERE {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get workflows
            query = f"""
                SELECT
                    workflow_id, name, description, version, category,
                    tags, status, created_date, updated_date,
                    execution_count, success_rate, avg_duration_ms, total_cost
                FROM automation.workflows
                WHERE {where_clause}
                ORDER BY created_date DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            workflows = []
            for result in results:
                workflow = {
                    'workflow_id': str(result['workflow_id']),
                    'name': result['name'],
                    'description': result['description'],
                    'version': result['version'],
                    'category': result['category'],
                    'tags': json.loads(result['tags']),
                    'status': result['status'],
                    'created_date': result['created_date'].isoformat(),
                    'updated_date': result['updated_date'].isoformat(),
                    'execution_count': result['execution_count'],
                    'success_rate': float(result['success_rate'] or 0),
                    'avg_duration_ms': result['avg_duration_ms'],
                    'total_cost': float(result['total_cost'])
                }
                workflows.append(workflow)

            return PaginatedResponse(
                success=True,
                message="Workflows retrieved successfully",
                data=workflows,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get workflows: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflows")

@router.get("/workflows/{workflow_id}", response_model=StandardResponse)
async def get_workflow(
    workflow_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get specific workflow"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT
                    workflow_id, name, description, version, category,
                    tags, trigger_config, nodes_config, global_variables,
                    error_handling, status, created_date, updated_date,
                    execution_count, success_rate, avg_duration_ms, total_cost
                FROM automation.workflows
                WHERE workflow_id = $1 AND tenant_id = $2
            """
            result = await conn.fetchrow(query, workflow_id, tenant_id)

            if not result:
                raise HTTPException(status_code=404, detail="Workflow not found")

            workflow = {
                'workflow_id': str(result['workflow_id']),
                'name': result['name'],
                'description': result['description'],
                'version': result['version'],
                'category': result['category'],
                'tags': json.loads(result['tags']),
                'trigger': json.loads(result['trigger_config']),
                'nodes': json.loads(result['nodes_config']),
                'global_variables': json.loads(result['global_variables']),
                'error_handling': json.loads(result['error_handling']),
                'status': result['status'],
                'created_date': result['created_date'].isoformat(),
                'updated_date': result['updated_date'].isoformat(),
                'execution_count': result['execution_count'],
                'success_rate': float(result['success_rate'] or 0),
                'avg_duration_ms': result['avg_duration_ms'],
                'total_cost': float(result['total_cost'])
            }

            return StandardResponse(
                success=True,
                message="Workflow retrieved successfully",
                data=workflow
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow")

@router.put("/workflows/{workflow_id}", response_model=StandardResponse)
async def update_workflow(
    workflow_id: UUID,
    workflow_update: Dict[str, Any],
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Update workflow"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            # Check if workflow exists
            check_query = """
                SELECT workflow_id FROM automation.workflows
                WHERE workflow_id = $1 AND tenant_id = $2
            """
            exists = await conn.fetchval(check_query, workflow_id, tenant_id)

            if not exists:
                raise HTTPException(status_code=404, detail="Workflow not found")

            # Build update query
            update_fields = []
            params = []
            param_count = 0

            allowed_fields = [
                'name', 'description', 'category', 'tags', 'trigger_config',
                'nodes_config', 'global_variables', 'error_handling', 'status'
            ]

            for field in allowed_fields:
                if field in workflow_update:
                    param_count += 1
                    update_fields.append(f"{field} = ${param_count}")

                    # Handle JSON fields
                    if field in ['tags', 'trigger_config', 'nodes_config', 'global_variables', 'error_handling']:
                        params.append(json.dumps(workflow_update[field]))
                    else:
                        params.append(workflow_update[field])

            if not update_fields:
                raise HTTPException(status_code=400, detail="No valid fields to update")

            # Add updated_date
            param_count += 1
            update_fields.append(f"updated_date = ${param_count}")
            params.append(datetime.utcnow())

            # Add WHERE conditions
            param_count += 1
            params.append(workflow_id)
            param_count += 1
            params.append(tenant_id)

            update_query = f"""
                UPDATE automation.workflows
                SET {', '.join(update_fields)}
                WHERE workflow_id = ${param_count - 1} AND tenant_id = ${param_count}
            """

            await conn.execute(update_query, *params)

            return StandardResponse(
                success=True,
                message="Workflow updated successfully",
                data={"workflow_id": str(workflow_id)}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to update workflow")

@router.delete("/workflows/{workflow_id}", response_model=StandardResponse)
async def delete_workflow(
    workflow_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Delete workflow"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            # Check if workflow exists
            check_query = """
                SELECT workflow_id FROM automation.workflows
                WHERE workflow_id = $1 AND tenant_id = $2
            """
            exists = await conn.fetchval(check_query, workflow_id, tenant_id)

            if not exists:
                raise HTTPException(status_code=404, detail="Workflow not found")

            # Delete workflow (cascades to related tables)
            delete_query = """
                DELETE FROM automation.workflows
                WHERE workflow_id = $1 AND tenant_id = $2
            """
            await conn.execute(delete_query, workflow_id, tenant_id)

            return StandardResponse(
                success=True,
                message="Workflow deleted successfully",
                data={"workflow_id": str(workflow_id)}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete workflow")

# Workflow Execution Endpoints

@router.post("/workflows/{workflow_id}/execute", response_model=StandardResponse)
async def execute_workflow(
    workflow_id: UUID,
    execution_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Execute workflow"""
    try:
        # Verify token
        await verify_token(token.credentials)

        trigger_data = execution_data.get('trigger_data', {})
        execution_context = execution_data.get('execution_context', {})

        # Execute workflow
        execution_id = await workflow_platform.execute_workflow(
            tenant_id, workflow_id, trigger_data, execution_context
        )

        return StandardResponse(
            success=True,
            message="Workflow execution started",
            data={"execution_id": str(execution_id)}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute workflow")

@router.get("/workflows/{workflow_id}/executions", response_model=PaginatedResponse)
async def get_workflow_executions(
    workflow_id: UUID,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get workflow executions"""
    try:
        # Verify token
        await verify_token(token.credentials)

        offset = (page - 1) * limit

        async with get_db_connection() as conn:
            # Build query
            where_conditions = ["workflow_id = $1", "tenant_id = $2"]
            params = [workflow_id, tenant_id]
            param_count = 2

            if status:
                param_count += 1
                where_conditions.append(f"status = ${param_count}")
                params.append(status)

            where_clause = " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM automation.workflow_executions
                WHERE {where_clause}
            """
            total_count = await conn.fetchval(count_query, *params)

            # Get executions
            query = f"""
                SELECT
                    execution_id, status, start_time, end_time,
                    total_duration_ms, nodes_executed, nodes_successful,
                    nodes_failed, total_cost, error_details
                FROM automation.workflow_executions
                WHERE {where_clause}
                ORDER BY start_time DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            executions = []
            for result in results:
                execution = {
                    'execution_id': str(result['execution_id']),
                    'status': result['status'],
                    'start_time': result['start_time'].isoformat(),
                    'end_time': result['end_time'].isoformat() if result['end_time'] else None,
                    'total_duration_ms': result['total_duration_ms'],
                    'nodes_executed': result['nodes_executed'],
                    'nodes_successful': result['nodes_successful'],
                    'nodes_failed': result['nodes_failed'],
                    'total_cost': float(result['total_cost']),
                    'error_details': result['error_details']
                }
                executions.append(execution)

            return PaginatedResponse(
                success=True,
                message="Workflow executions retrieved successfully",
                data=executions,
                pagination={
                    'page': page,
                    'limit': limit,
                    'total': total_count,
                    'pages': (total_count + limit - 1) // limit
                }
            )

    except Exception as e:
        logger.error(f"Failed to get workflow executions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow executions")

@router.get("/executions/{execution_id}", response_model=StandardResponse)
async def get_execution_details(
    execution_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get execution details"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            # Get execution details
            execution_query = """
                SELECT
                    execution_id, workflow_id, trigger_data, execution_context,
                    current_node, status, start_time, end_time, total_duration_ms,
                    nodes_executed, nodes_successful, nodes_failed, total_cost,
                    error_details, output_data
                FROM automation.workflow_executions
                WHERE execution_id = $1 AND tenant_id = $2
            """
            execution_result = await conn.fetchrow(execution_query, execution_id, tenant_id)

            if not execution_result:
                raise HTTPException(status_code=404, detail="Execution not found")

            # Get node executions
            nodes_query = """
                SELECT
                    node_execution_id, node_id, node_type, status,
                    start_time, end_time, duration_ms, cost, error_message,
                    retry_count
                FROM automation.node_executions
                WHERE execution_id = $1
                ORDER BY start_time
            """
            node_results = await conn.fetch(nodes_query, execution_id)

            node_executions = []
            for node_result in node_results:
                node_execution = {
                    'node_execution_id': str(node_result['node_execution_id']),
                    'node_id': node_result['node_id'],
                    'node_type': node_result['node_type'],
                    'status': node_result['status'],
                    'start_time': node_result['start_time'].isoformat(),
                    'end_time': node_result['end_time'].isoformat() if node_result['end_time'] else None,
                    'duration_ms': node_result['duration_ms'],
                    'cost': float(node_result['cost']),
                    'error_message': node_result['error_message'],
                    'retry_count': node_result['retry_count']
                }
                node_executions.append(node_execution)

            execution_details = {
                'execution_id': str(execution_result['execution_id']),
                'workflow_id': str(execution_result['workflow_id']),
                'trigger_data': json.loads(execution_result['trigger_data']),
                'execution_context': json.loads(execution_result['execution_context']),
                'current_node': execution_result['current_node'],
                'status': execution_result['status'],
                'start_time': execution_result['start_time'].isoformat(),
                'end_time': execution_result['end_time'].isoformat() if execution_result['end_time'] else None,
                'total_duration_ms': execution_result['total_duration_ms'],
                'nodes_executed': execution_result['nodes_executed'],
                'nodes_successful': execution_result['nodes_successful'],
                'nodes_failed': execution_result['nodes_failed'],
                'total_cost': float(execution_result['total_cost']),
                'error_details': execution_result['error_details'],
                'output_data': json.loads(execution_result['output_data']) if execution_result['output_data'] else None,
                'node_executions': node_executions
            }

            return StandardResponse(
                success=True,
                message="Execution details retrieved successfully",
                data=execution_details
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get execution details")

# Workflow Templates Endpoints

@router.get("/templates", response_model=PaginatedResponse)
async def get_workflow_templates(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    featured: Optional[bool] = None,
    search: Optional[str] = None,
    token: str = Depends(security)
):
    """Get workflow templates"""
    try:
        # Verify token
        await verify_token(token.credentials)

        templates = await workflow_platform.get_workflow_templates(category)

        # Apply filters
        if featured is not None:
            templates = [t for t in templates if t.get('is_featured') == featured]

        if search:
            search_lower = search.lower()
            templates = [
                t for t in templates
                if search_lower in t['name'].lower() or search_lower in t['description'].lower()
            ]

        # Apply pagination
        total_count = len(templates)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_templates = templates[start_idx:end_idx]

        return PaginatedResponse(
            success=True,
            message="Workflow templates retrieved successfully",
            data=paginated_templates,
            pagination={
                'page': page,
                'limit': limit,
                'total': total_count,
                'pages': (total_count + limit - 1) // limit
            }
        )

    except Exception as e:
        logger.error(f"Failed to get workflow templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow templates")

@router.post("/templates/{template_id}/create-workflow", response_model=StandardResponse)
async def create_workflow_from_template(
    template_id: UUID,
    customization: Dict[str, Any],
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create workflow from template"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Create workflow from template
        workflow_id = await workflow_platform.create_workflow_from_template(
            tenant_id, template_id, customization
        )

        return StandardResponse(
            success=True,
            message="Workflow created from template successfully",
            data={"workflow_id": str(workflow_id)}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create workflow from template: {e}")
        raise HTTPException(status_code=500, detail="Failed to create workflow from template")

# Workflow Analytics Endpoints

@router.get("/analytics/overview", response_model=StandardResponse)
async def get_automation_analytics(
    days: int = Query(30, ge=1, le=365),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get automation analytics overview"""
    try:
        # Verify token
        await verify_token(token.credentials)

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)

        async with get_db_connection() as conn:
            # Get workflow stats
            workflow_stats_query = """
                SELECT
                    COUNT(*) as total_workflows,
                    COUNT(*) FILTER (WHERE status = 'active') as active_workflows,
                    COUNT(*) FILTER (WHERE status = 'draft') as draft_workflows,
                    AVG(success_rate) as avg_success_rate,
                    SUM(execution_count) as total_executions,
                    SUM(total_cost) as total_cost
                FROM automation.workflows
                WHERE tenant_id = $1
            """
            workflow_stats = await conn.fetchrow(workflow_stats_query, tenant_id)

            # Get execution stats for period
            execution_stats_query = """
                SELECT
                    COUNT(*) as total_executions,
                    COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
                    AVG(total_duration_ms) as avg_duration_ms,
                    SUM(total_cost) as period_cost,
                    SUM(nodes_executed) as total_nodes_executed
                FROM automation.workflow_executions
                WHERE tenant_id = $1 AND start_time::date BETWEEN $2 AND $3
            """
            execution_stats = await conn.fetchrow(execution_stats_query, tenant_id, start_date, end_date)

            # Get daily execution trends
            daily_trends_query = """
                SELECT
                    start_time::date as execution_date,
                    COUNT(*) as executions,
                    COUNT(*) FILTER (WHERE status = 'completed') as successful,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    SUM(total_cost) as daily_cost
                FROM automation.workflow_executions
                WHERE tenant_id = $1 AND start_time::date BETWEEN $2 AND $3
                GROUP BY start_time::date
                ORDER BY execution_date
            """
            daily_trends = await conn.fetch(daily_trends_query, tenant_id, start_date, end_date)

            # Get top workflows by executions
            top_workflows_query = """
                SELECT
                    w.workflow_id,
                    w.name,
                    w.category,
                    COUNT(e.execution_id) as execution_count,
                    AVG(e.total_duration_ms) as avg_duration,
                    SUM(e.total_cost) as total_cost
                FROM automation.workflows w
                LEFT JOIN automation.workflow_executions e ON w.workflow_id = e.workflow_id
                    AND e.start_time::date BETWEEN $2 AND $3
                WHERE w.tenant_id = $1
                GROUP BY w.workflow_id, w.name, w.category
                ORDER BY execution_count DESC
                LIMIT 10
            """
            top_workflows = await conn.fetch(top_workflows_query, tenant_id, start_date, end_date)

            analytics = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'workflow_stats': {
                    'total_workflows': workflow_stats['total_workflows'],
                    'active_workflows': workflow_stats['active_workflows'],
                    'draft_workflows': workflow_stats['draft_workflows'],
                    'avg_success_rate': float(workflow_stats['avg_success_rate'] or 0),
                    'total_executions': workflow_stats['total_executions'],
                    'total_cost': float(workflow_stats['total_cost'] or 0)
                },
                'execution_stats': {
                    'total_executions': execution_stats['total_executions'],
                    'successful_executions': execution_stats['successful_executions'],
                    'failed_executions': execution_stats['failed_executions'],
                    'success_rate': (execution_stats['successful_executions'] / max(execution_stats['total_executions'], 1)) * 100,
                    'avg_duration_ms': int(execution_stats['avg_duration_ms'] or 0),
                    'period_cost': float(execution_stats['period_cost'] or 0),
                    'total_nodes_executed': execution_stats['total_nodes_executed']
                },
                'daily_trends': [
                    {
                        'date': trend['execution_date'].isoformat(),
                        'executions': trend['executions'],
                        'successful': trend['successful'],
                        'failed': trend['failed'],
                        'daily_cost': float(trend['daily_cost'] or 0)
                    }
                    for trend in daily_trends
                ],
                'top_workflows': [
                    {
                        'workflow_id': str(workflow['workflow_id']),
                        'name': workflow['name'],
                        'category': workflow['category'],
                        'execution_count': workflow['execution_count'],
                        'avg_duration': int(workflow['avg_duration'] or 0),
                        'total_cost': float(workflow['total_cost'] or 0)
                    }
                    for workflow in top_workflows
                ]
            }

            return StandardResponse(
                success=True,
                message="Automation analytics retrieved successfully",
                data=analytics
            )

    except Exception as e:
        logger.error(f"Failed to get automation analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get automation analytics")

# Webhook Endpoints

@router.post("/webhooks/{workflow_id}", response_model=StandardResponse)
async def trigger_workflow_webhook(
    workflow_id: UUID,
    webhook_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Trigger workflow via webhook"""
    try:
        # Get workflow and tenant info from webhook URL
        async with get_db_connection() as conn:
            query = """
                SELECT w.tenant_id, w.status, wh.is_active
                FROM automation.workflows w
                JOIN automation.workflow_webhooks wh ON w.workflow_id = wh.workflow_id
                WHERE w.workflow_id = $1
            """
            result = await conn.fetchrow(query, workflow_id)

            if not result:
                raise HTTPException(status_code=404, detail="Webhook not found")

            if result['status'] != 'active' or not result['is_active']:
                raise HTTPException(status_code=400, detail="Workflow or webhook is not active")

            tenant_id = result['tenant_id']

        # Execute workflow
        execution_id = await workflow_platform.execute_workflow(
            tenant_id, workflow_id, webhook_data, {}
        )

        return StandardResponse(
            success=True,
            message="Workflow triggered successfully",
            data={"execution_id": str(execution_id)}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger workflow webhook: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger workflow")

# Export router
__all__ = ["router"]
