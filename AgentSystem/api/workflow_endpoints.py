"""
Workflow Builder API Endpoints - AgentSystem Profit Machine
FastAPI endpoints for visual workflow creation, execution, and management
"""

from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
from datetime import datetime, timedelta
import asyncpg
import uuid

from ..workflows.workflow_builder import (
    WorkflowEngine, Workflow, WorkflowNode, WorkflowConnection, WorkflowExecution,
    WorkflowStatus, TriggerType, ActionType, ConditionOperator
)
from ..auth.tenant_auth import get_current_tenant
from ..database.connection import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/workflows", tags=["Workflow Builder"])
security = HTTPBearer()

# Pydantic models for API requests/responses
class CreateWorkflowRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    folder_id: Optional[str] = Field(None, description="Folder ID for organization")

class WorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    description: Optional[str]
    status: str
    execution_count: int
    success_count: int
    error_count: int
    last_executed_at: Optional[datetime]
    created_by: str
    created_at: datetime
    updated_at: datetime

class AddNodeRequest(BaseModel):
    node_type: str = Field(..., description="Node type (trigger, action, condition)")
    name: str = Field(..., description="Node display name")
    config: Dict[str, Any] = Field(..., description="Node configuration")
    position: Dict[str, float] = Field(..., description="Node position (x, y)")
    description: Optional[str] = Field(None, description="Node description")

class AddConnectionRequest(BaseModel):
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    source_port: str = Field(default="output", description="Source port")
    target_port: str = Field(default="input", description="Target port")
    condition: Optional[Dict[str, Any]] = Field(None, description="Connection condition")

class ExecuteWorkflowRequest(BaseModel):
    trigger_data: Dict[str, Any] = Field(..., description="Trigger data")
    execution_context: Optional[Dict[str, Any]] = Field(None, description="Execution context")

class WorkflowExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str
    trigger_data: Dict[str, Any]
    execution_time_ms: Optional[int]
    nodes_executed: List[str]
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]

class WorkflowTemplateResponse(BaseModel):
    template_id: str
    name: str
    description: str
    category: str
    subcategory: Optional[str]
    tags: List[str]
    difficulty_level: str
    downloads_count: int
    rating_average: float
    rating_count: int
    is_featured: bool
    required_integrations: List[str]

class CreateFolderRequest(BaseModel):
    name: str = Field(..., description="Folder name")
    description: Optional[str] = Field(None, description="Folder description")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")
    color: str = Field(default="#6366f1", description="Folder color")

class WorkflowAnalyticsResponse(BaseModel):
    workflow_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_execution_time_ms: float
    total_tokens_used: int
    total_cost_usd: float
    executions_last_24h: int
    daily_stats: List[Dict[str, Any]]

# Global WorkflowEngine instance
workflow_engine: Optional[WorkflowEngine] = None

async def get_workflow_engine() -> WorkflowEngine:
    """Get initialized WorkflowEngine instance"""
    global workflow_engine
    if not workflow_engine:
        db_pool = await get_db_pool()

        # Initialize with integrations
        integrations = {
            'email_service': None,  # Will be injected
            'slack_service': None,  # Will be injected
            'teams_service': None,  # Will be injected
            'ai_service': None,     # Will be injected
        }

        workflow_engine = WorkflowEngine(db_pool=db_pool, integrations=integrations)
    return workflow_engine

@router.post("/create", response_model=WorkflowResponse)
async def create_workflow(
    request: CreateWorkflowRequest,
    tenant = Depends(get_current_tenant),
    engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Create a new workflow
    """
    try:
        workflow = await engine.create_workflow(
            tenant_id=tenant.id,
            name=request.name,
            description=request.description,
            created_by=tenant.name,  # Or get from auth context
            folder_id=request.folder_id
        )

        return WorkflowResponse(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            status=workflow.status.value,
            execution_count=workflow.execution_count,
            success_count=0,
            error_count=0,
            last_executed_at=workflow.last_executed_at,
            created_by=workflow.created_by,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at
        )

    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@router.get("/list", response_model=List[WorkflowResponse])
async def list_workflows(
    status: Optional[str] = Query(None, description="Filter by status"),
    folder_id: Optional[str] = Query(None, description="Filter by folder"),
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    List workflows for a tenant
    """
    try:
        async with db_pool.acquire() as conn:
            query = """
                SELECT w.*,
                       COALESCE(stats.success_count, 0) as success_count,
                       COALESCE(stats.error_count, 0) as error_count
                FROM workflows.workflows w
                LEFT JOIN (
                    SELECT workflow_id,
                           COUNT(*) FILTER (WHERE status = 'completed') as success_count,
                           COUNT(*) FILTER (WHERE status = 'failed') as error_count
                    FROM workflows.workflow_executions
                    GROUP BY workflow_id
                ) stats ON w.workflow_id = stats.workflow_id
                WHERE w.tenant_id = $1
            """
            params = [tenant.id]

            if status:
                query += " AND w.status = $2"
                params.append(status)

            if folder_id:
                if status:
                    query += " AND w.folder_id = $3"
                else:
                    query += " AND w.folder_id = $2"
                params.append(folder_id)

            query += " ORDER BY w.updated_at DESC"

            rows = await conn.fetch(query, *params)

            workflows = []
            for row in rows:
                workflows.append(WorkflowResponse(
                    workflow_id=row['workflow_id'],
                    name=row['name'],
                    description=row['description'],
                    status=row['status'],
                    execution_count=row['execution_count'],
                    success_count=row['success_count'] or 0,
                    error_count=row['error_count'] or 0,
                    last_executed_at=row['last_executed_at'],
                    created_by=row['created_by'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))

            return workflows

    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail="Failed to list workflows")

@router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    tenant = Depends(get_current_tenant),
    engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Get workflow details
    """
    try:
        workflow = await engine._get_workflow(workflow_id)

        if not workflow or workflow.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status.value,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "name": node.name,
                    "description": node.description,
                    "config": node.config,
                    "position": node.position,
                    "is_enabled": node.is_enabled
                }
                for node in workflow.nodes
            ],
            "connections": [
                {
                    "connection_id": conn.connection_id,
                    "source_node_id": conn.source_node_id,
                    "target_node_id": conn.target_node_id,
                    "source_port": conn.source_port,
                    "target_port": conn.target_port,
                    "condition": conn.condition
                }
                for conn in workflow.connections
            ],
            "variables": workflow.variables,
            "tags": workflow.tags,
            "execution_count": workflow.execution_count,
            "last_executed_at": workflow.last_executed_at,
            "created_by": workflow.created_by,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow")

@router.post("/{workflow_id}/nodes")
async def add_node(
    workflow_id: str,
    request: AddNodeRequest,
    tenant = Depends(get_current_tenant),
    engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Add a node to workflow
    """
    try:
        # Verify workflow ownership
        workflow = await engine._get_workflow(workflow_id)
        if not workflow or workflow.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Workflow not found")

        node = await engine.add_node(
            workflow_id=workflow_id,
            node_type=request.node_type,
            name=request.name,
            config=request.config,
            position=request.position,
            description=request.description
        )

        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "name": node.name,
            "description": node.description,
            "config": node.config,
            "position": node.position,
            "is_enabled": node.is_enabled,
            "created_at": node.created_at
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add node: {e}")
        raise HTTPException(status_code=500, detail="Failed to add node")

@router.post("/{workflow_id}/connections")
async def add_connection(
    workflow_id: str,
    request: AddConnectionRequest,
    tenant = Depends(get_current_tenant),
    engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Add a connection between nodes
    """
    try:
        # Verify workflow ownership
        workflow = await engine._get_workflow(workflow_id)
        if not workflow or workflow.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Workflow not found")

        connection = await engine.add_connection(
            workflow_id=workflow_id,
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            source_port=request.source_port,
            target_port=request.target_port,
            condition=request.condition
        )

        return {
            "connection_id": connection.connection_id,
            "source_node_id": connection.source_node_id,
            "target_node_id": connection.target_node_id,
            "source_port": connection.source_port,
            "target_port": connection.target_port,
            "condition": connection.condition,
            "created_at": connection.created_at
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add connection: {e}")
        raise HTTPException(status_code=500, detail="Failed to add connection")

@router.post("/{workflow_id}/activate")
async def activate_workflow(
    workflow_id: str,
    tenant = Depends(get_current_tenant),
    engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Activate a workflow for execution
    """
    try:
        # Verify workflow ownership
        workflow = await engine._get_workflow(workflow_id)
        if not workflow or workflow.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Workflow not found")

        success = await engine.activate_workflow(workflow_id)

        if success:
            return {"success": True, "message": "Workflow activated successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to activate workflow")

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to activate workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate workflow")

@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    request: ExecuteWorkflowRequest,
    background_tasks: BackgroundTasks,
    tenant = Depends(get_current_tenant),
    engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Execute a workflow manually
    """
    try:
        # Verify workflow ownership
        workflow = await engine._get_workflow(workflow_id)
        if not workflow or workflow.tenant_id != tenant.id:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Execute workflow in background
        background_tasks.add_task(
            execute_workflow_background,
            engine,
            workflow_id,
            request.trigger_data,
            request.execution_context
        )

        # Return execution started response
        execution_id = str(uuid.uuid4())
        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status="running",
            trigger_data=request.trigger_data,
            execution_time_ms=None,
            nodes_executed=[],
            error_message=None,
            started_at=datetime.now(),
            completed_at=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute workflow")

@router.get("/{workflow_id}/executions")
async def get_workflow_executions(
    workflow_id: str,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0),
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get workflow execution history
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify workflow ownership
            workflow = await conn.fetchrow("""
                SELECT workflow_id FROM workflows.workflows
                WHERE workflow_id = $1 AND tenant_id = $2
            """, workflow_id, tenant.id)

            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

            # Get executions
            executions = await conn.fetch("""
                SELECT * FROM workflows.workflow_executions
                WHERE workflow_id = $1
                ORDER BY started_at DESC
                LIMIT $2 OFFSET $3
            """, workflow_id, limit, offset)

            return [
                {
                    "execution_id": row['execution_id'],
                    "status": row['status'],
                    "trigger_type": row['trigger_type'],
                    "execution_time_ms": row['execution_time_ms'],
                    "nodes_executed": row['nodes_executed'],
                    "error_message": row['error_message'],
                    "started_at": row['started_at'],
                    "completed_at": row['completed_at']
                }
                for row in executions
            ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow executions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get executions")

@router.get("/{workflow_id}/analytics", response_model=WorkflowAnalyticsResponse)
async def get_workflow_analytics(
    workflow_id: str,
    days: int = Query(default=30, le=90),
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Get workflow analytics
    """
    try:
        async with db_pool.acquire() as conn:
            # Verify workflow ownership
            workflow = await conn.fetchrow("""
                SELECT workflow_id FROM workflows.workflows
                WHERE workflow_id = $1 AND tenant_id = $2
            """, workflow_id, tenant.id)

            if not workflow:
                raise HTTPException(status_code=404, detail="Workflow not found")

            # Get overall stats
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_executions,
                    COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
                    AVG(execution_time_ms) as avg_execution_time_ms,
                    COALESCE(SUM(tokens_used), 0) as total_tokens_used,
                    COUNT(*) FILTER (WHERE started_at > NOW() - INTERVAL '24 hours') as executions_last_24h
                FROM workflows.workflow_executions
                WHERE workflow_id = $1 AND started_at > NOW() - INTERVAL '%s days'
            """ % days, workflow_id)

            # Get daily stats
            daily_stats = await conn.fetch("""
                SELECT
                    date,
                    executions_total,
                    executions_successful,
                    executions_failed,
                    avg_execution_time_ms,
                    total_tokens_used,
                    cost_usd
                FROM workflows.workflow_analytics_daily
                WHERE workflow_id = $1 AND date > CURRENT_DATE - INTERVAL '%s days'
                ORDER BY date DESC
            """ % days, workflow_id)

            success_rate = 0.0
            if stats['total_executions'] > 0:
                success_rate = (stats['successful_executions'] / stats['total_executions']) * 100

            return WorkflowAnalyticsResponse(
                workflow_id=workflow_id,
                total_executions=stats['total_executions'] or 0,
                successful_executions=stats['successful_executions'] or 0,
                failed_executions=stats['failed_executions'] or 0,
                success_rate=success_rate,
                avg_execution_time_ms=float(stats['avg_execution_time_ms'] or 0),
                total_tokens_used=stats['total_tokens_used'] or 0,
                total_cost_usd=float((stats['total_tokens_used'] or 0) * 0.002),  # Estimated
                executions_last_24h=stats['executions_last_24h'] or 0,
                daily_stats=[
                    {
                        "date": row['date'].isoformat(),
                        "executions": row['executions_total'],
                        "success_rate": (row['executions_successful'] / max(row['executions_total'], 1)) * 100,
                        "avg_time_ms": row['avg_execution_time_ms'],
                        "tokens_used": row['total_tokens_used'],
                        "cost_usd": float(row['cost_usd'] or 0)
                    }
                    for row in daily_stats
                ]
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@router.post("/webhook/{workflow_id}/{trigger_node_id}")
async def workflow_webhook(
    workflow_id: str,
    trigger_node_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Webhook endpoint for triggering workflows
    """
    try:
        # Get request data
        body = await request.body()
        headers = dict(request.headers)

        trigger_data = {
            'body': body.decode() if body else '',
            'headers': headers,
            'method': request.method,
            'url': str(request.url),
            'query_params': dict(request.query_params)
        }

        # Try to parse JSON body
        if body:
            try:
                trigger_data['json'] = json.loads(body.decode())
            except:
                pass

        # Execute workflow in background
        background_tasks.add_task(
            execute_workflow_background,
            engine,
            workflow_id,
            trigger_data,
            {'trigger_node_id': trigger_node_id}
        )

        return {"status": "webhook_received", "workflow_id": workflow_id}

    except Exception as e:
        logger.error(f"Webhook execution failed: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@router.get("/templates", response_model=List[WorkflowTemplateResponse])
async def get_workflow_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    featured: Optional[bool] = Query(None, description="Filter featured templates"),
    limit: int = Query(default=20, le=50),
    db_pool = Depends(get_db_pool)
):
    """
    Get workflow templates from marketplace
    """
    try:
        async with db_pool.acquire() as conn:
            query = """
                SELECT t.*, COALESCE(AVG(r.rating), 0) as rating_average, COUNT(r.rating_id) as rating_count
                FROM workflows.workflow_templates t
                LEFT JOIN workflows.workflow_template_ratings r ON t.template_id = r.template_id
                WHERE t.is_approved = true
            """
            params = []

            if category:
                query += " AND t.category = $1"
                params.append(category)

            if featured is not None:
                if params:
                    query += f" AND t.is_featured = ${len(params) + 1}"
                else:
                    query += " AND t.is_featured = $1"
                params.append(featured)

            query += """
                GROUP BY t.template_id
                ORDER BY t.downloads_count DESC, rating_average DESC
                LIMIT $%d
            """ % (len(params) + 1)
            params.append(limit)

            rows = await conn.fetch(query, *params)

            templates = []
            for row in rows:
                templates.append(WorkflowTemplateResponse(
                    template_id=row['template_id'],
                    name=row['name'],
                    description=row['description'],
                    category=row['category'],
                    subcategory=row['subcategory'],
                    tags=row['tags'] or [],
                    difficulty_level=row['difficulty_level'],
                    downloads_count=row['downloads_count'],
                    rating_average=float(row['rating_average']),
                    rating_count=row['rating_count'],
                    is_featured=row['is_featured'],
                    required_integrations=row['required_integrations'] or []
                ))

            return templates

    except Exception as e:
        logger.error(f"Failed to get workflow templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to get templates")

@router.post("/folders", response_model=Dict[str, str])
async def create_folder(
    request: CreateFolderRequest,
    tenant = Depends(get_current_tenant),
    db_pool = Depends(get_db_pool)
):
    """
    Create a workflow folder
    """
    try:
        folder_id = str(uuid.uuid4())

        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflows.workflow_folders (
                    folder_id, tenant_id, parent_folder_id, name, description, color
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, folder_id, tenant.id, request.parent_folder_id,
                request.name, request.description, request.color)

            return {"folder_id": folder_id, "message": "Folder created successfully"}

    except Exception as e:
        logger.error(f"Failed to create folder: {e}")
        raise HTTPException(status_code=500, detail="Failed to create folder")

# Background task functions
async def execute_workflow_background(engine: WorkflowEngine, workflow_id: str,
                                    trigger_data: Dict[str, Any],
                                    execution_context: Dict[str, Any] = None):
    """Execute workflow in background"""
    try:
        await engine.execute_workflow(workflow_id, trigger_data, execution_context)
    except Exception as e:
        logger.error(f"Background workflow execution failed: {e}")

# Health check endpoint
@router.get("/health")
async def workflow_health_check():
    """Health check for workflow builder"""
    return {
        "status": "healthy",
        "service": "workflow_builder",
        "timestamp": datetime.now().isoformat()
    }

# Export router
__all__ = ["router"]
