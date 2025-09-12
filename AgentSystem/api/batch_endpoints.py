"""
Batch Processing API Endpoints - AgentSystem Profit Machine
Advanced batch optimization for bulk AI operations and cost reduction
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4
import asyncio
import json
from enum import Enum

from ..auth.auth_service import verify_token, get_current_tenant
from ..database.connection import get_db_connection
from ..batch.batch_processor import BatchProcessor
from ..usage.usage_tracker import UsageTracker
from ..billing.stripe_service import StripeService

# Initialize router
router = APIRouter(prefix="/api/v1/batch", tags=["Batch Processing"])
security = HTTPBearer()

# Enums
class BatchStrategy(str, Enum):
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"
    COST_BASED = "cost_based"
    LATENCY_BASED = "latency_based"
    ADAPTIVE = "adaptive"

class BatchPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class BatchStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class RequestStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

# Request Models
class BatchRequestCreate(BaseModel):
    model: str = Field(..., description="AI model to use")
    capability: str = Field(..., description="AI capability (chat, completion, embedding, etc.)")
    content: str = Field(..., description="Request content/prompt")
    priority: BatchPriority = Field(default=BatchPriority.NORMAL, description="Request priority")
    max_latency_ms: Optional[int] = Field(None, description="Maximum acceptable latency in milliseconds")
    cost_budget: Optional[float] = Field(None, description="Maximum cost budget for this request")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    deadline: Optional[datetime] = Field(None, description="Request deadline")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

class BatchRequestBulk(BaseModel):
    requests: List[BatchRequestCreate] = Field(..., description="List of batch requests")
    batch_config: Optional[Dict[str, Any]] = Field(None, description="Override batch configuration")

class BatchConfigUpdate(BaseModel):
    strategy: Optional[BatchStrategy] = Field(None, description="Batch processing strategy")
    max_batch_size: Optional[int] = Field(None, ge=1, le=100, description="Maximum batch size")
    min_batch_size: Optional[int] = Field(None, ge=1, le=50, description="Minimum batch size")
    max_wait_time_seconds: Optional[int] = Field(None, ge=1, le=300, description="Maximum wait time")
    cost_threshold: Optional[float] = Field(None, ge=0, description="Cost threshold for batching")
    enable_smart_grouping: Optional[bool] = Field(None, description="Enable smart request grouping")
    enable_priority_queue: Optional[bool] = Field(None, description="Enable priority-based queuing")
    retry_failed_requests: Optional[bool] = Field(None, description="Retry failed requests")
    parallel_batches: Optional[int] = Field(None, ge=1, le=10, description="Number of parallel batches")
    cost_savings_target: Optional[float] = Field(None, ge=0, le=100, description="Target cost savings percentage")
    auto_optimization: Optional[bool] = Field(None, description="Enable automatic optimization")

class BatchScheduleCreate(BaseModel):
    schedule_name: str = Field(..., description="Schedule name")
    cron_expression: str = Field(..., description="Cron expression for scheduling")
    batch_config: Dict[str, Any] = Field(..., description="Batch configuration")
    is_active: bool = Field(default=True, description="Whether schedule is active")

# Response Models
class BatchRequestResponse(BaseModel):
    request_id: UUID
    tenant_id: UUID
    batch_id: Optional[UUID]
    model: str
    capability: str
    priority: BatchPriority
    status: RequestStatus
    estimated_tokens: int
    estimated_cost: float
    actual_tokens: Optional[int]
    actual_cost: Optional[float]
    retry_count: int
    max_retries: int
    response_content: Optional[str]
    error_details: Optional[str]
    processing_time_ms: Optional[int]
    deadline: Optional[datetime]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

class BatchResponse(BaseModel):
    batch_id: UUID
    tenant_id: UUID
    model: str
    capability: str
    strategy: BatchStrategy
    status: BatchStatus
    priority: BatchPriority
    batch_size: int
    estimated_cost: float
    estimated_tokens: int
    estimated_processing_time: float
    actual_cost: Optional[float]
    actual_tokens: Optional[int]
    actual_processing_time: Optional[float]
    cost_savings: float
    cost_savings_percent: float
    throughput_improvement: float
    scheduled_at: Optional[datetime]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_details: Optional[str]

class BatchConfigResponse(BaseModel):
    config_id: UUID
    tenant_id: UUID
    strategy: BatchStrategy
    max_batch_size: int
    min_batch_size: int
    max_wait_time_seconds: int
    cost_threshold: float
    enable_smart_grouping: bool
    enable_priority_queue: bool
    retry_failed_requests: bool
    parallel_batches: int
    cost_savings_target: float
    auto_optimization: bool
    created_at: datetime
    updated_at: datetime

class BatchMetricsResponse(BaseModel):
    tenant_id: UUID
    metric_date: datetime
    total_requests: int
    total_batches: int
    avg_batch_size: float
    avg_processing_time: float
    total_cost_savings: float
    cost_savings_percent: float
    success_rate: float
    throughput_per_hour: float
    avg_latency_reduction: float
    efficiency_score: float

class BatchEfficiencyResponse(BaseModel):
    efficiency_score: float
    cost_savings_score: float
    throughput_score: float
    latency_score: float
    recommendation: str

class QueueStatusResponse(BaseModel):
    priority: BatchPriority
    queue_size: int
    avg_wait_time_seconds: float
    max_wait_time_seconds: float
    throughput_per_minute: float

class BatchDashboardResponse(BaseModel):
    total_requests: int
    total_batches: int
    avg_batch_size: float
    completed_requests: int
    failed_requests: int
    pending_requests: int
    processing_requests: int
    avg_processing_time_ms: float
    total_cost_savings: float
    avg_cost_savings_percent: float
    unique_models: int
    last_request_at: Optional[datetime]
    requests_last_24h: int

# Initialize services
batch_processor = BatchProcessor()
usage_tracker = UsageTracker()
stripe_service = StripeService()

# Endpoints

@router.post("/requests", response_model=BatchRequestResponse)
async def create_batch_request(
    request: BatchRequestCreate,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create a new batch request"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Create batch request
        request_id = await batch_processor.submit_request(
            tenant_id=tenant_id,
            model=request.model,
            capability=request.capability,
            content=request.content,
            priority=request.priority.value,
            max_latency_ms=request.max_latency_ms,
            cost_budget=request.cost_budget,
            callback_url=request.callback_url,
            metadata=request.metadata,
            deadline=request.deadline,
            max_retries=request.max_retries
        )

        # Start background processing
        background_tasks.add_task(batch_processor.process_queue, tenant_id)

        # Get request details
        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.requests
                WHERE request_id = $1 AND tenant_id = $2
            """
            result = await conn.fetchrow(query, request_id, tenant_id)

            if not result:
                raise HTTPException(status_code=404, detail="Request not found")

            return BatchRequestResponse(**dict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create batch request: {str(e)}")

@router.post("/requests/bulk", response_model=List[BatchRequestResponse])
async def create_bulk_batch_requests(
    bulk_request: BatchRequestBulk,
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create multiple batch requests in bulk"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Validate bulk request
        if len(bulk_request.requests) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 requests per bulk operation")

        # Create all requests
        request_ids = []
        for req in bulk_request.requests:
            request_id = await batch_processor.submit_request(
                tenant_id=tenant_id,
                model=req.model,
                capability=req.capability,
                content=req.content,
                priority=req.priority.value,
                max_latency_ms=req.max_latency_ms,
                cost_budget=req.cost_budget,
                callback_url=req.callback_url,
                metadata=req.metadata,
                deadline=req.deadline,
                max_retries=req.max_retries
            )
            request_ids.append(request_id)

        # Start background processing
        background_tasks.add_task(batch_processor.process_queue, tenant_id)

        # Get all request details
        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.requests
                WHERE request_id = ANY($1) AND tenant_id = $2
                ORDER BY created_at DESC
            """
            results = await conn.fetch(query, request_ids, tenant_id)

            return [BatchRequestResponse(**dict(result)) for result in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create bulk batch requests: {str(e)}")

@router.get("/requests/{request_id}", response_model=BatchRequestResponse)
async def get_batch_request(
    request_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get batch request details"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.requests
                WHERE request_id = $1 AND tenant_id = $2
            """
            result = await conn.fetchrow(query, request_id, tenant_id)

            if not result:
                raise HTTPException(status_code=404, detail="Request not found")

            return BatchRequestResponse(**dict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch request: {str(e)}")

@router.get("/requests", response_model=List[BatchRequestResponse])
async def list_batch_requests(
    status: Optional[RequestStatus] = None,
    model: Optional[str] = None,
    priority: Optional[BatchPriority] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """List batch requests with filtering"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status.value)

        if model:
            param_count += 1
            conditions.append(f"model = ${param_count}")
            params.append(model)

        if priority:
            param_count += 1
            conditions.append(f"priority = ${param_count}")
            params.append(priority.value)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM batch.requests
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [BatchRequestResponse(**dict(result)) for result in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list batch requests: {str(e)}")

@router.delete("/requests/{request_id}")
async def cancel_batch_request(
    request_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Cancel a pending batch request"""
    try:
        # Verify token
        await verify_token(token.credentials)

        success = await batch_processor.cancel_request(tenant_id, request_id)

        if not success:
            raise HTTPException(status_code=404, detail="Request not found or cannot be cancelled")

        return {"message": "Request cancelled successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel batch request: {str(e)}")

@router.get("/batches", response_model=List[BatchResponse])
async def list_batches(
    status: Optional[BatchStatus] = None,
    model: Optional[str] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """List batches with filtering"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build query
        conditions = ["tenant_id = $1"]
        params = [tenant_id]
        param_count = 1

        if status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(status.value)

        if model:
            param_count += 1
            conditions.append(f"model = ${param_count}")
            params.append(model)

        async with get_db_connection() as conn:
            query = f"""
                SELECT * FROM batch.batches
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ${param_count + 1} OFFSET ${param_count + 2}
            """
            params.extend([limit, offset])

            results = await conn.fetch(query, *params)

            return [BatchResponse(**dict(result)) for result in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list batches: {str(e)}")

@router.get("/batches/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get batch details"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.batches
                WHERE batch_id = $1 AND tenant_id = $2
            """
            result = await conn.fetchrow(query, batch_id, tenant_id)

            if not result:
                raise HTTPException(status_code=404, detail="Batch not found")

            return BatchResponse(**dict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch: {str(e)}")

@router.get("/batches/{batch_id}/requests", response_model=List[BatchRequestResponse])
async def get_batch_requests(
    batch_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get all requests in a batch"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.requests
                WHERE batch_id = $1 AND tenant_id = $2
                ORDER BY created_at ASC
            """
            results = await conn.fetch(query, batch_id, tenant_id)

            return [BatchRequestResponse(**dict(result)) for result in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch requests: {str(e)}")

@router.get("/config", response_model=BatchConfigResponse)
async def get_batch_config(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get batch configuration"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.configs
                WHERE tenant_id = $1
            """
            result = await conn.fetchrow(query, tenant_id)

            if not result:
                # Create default config
                config_id = uuid4()
                insert_query = """
                    INSERT INTO batch.configs (config_id, tenant_id)
                    VALUES ($1, $2)
                    RETURNING *
                """
                result = await conn.fetchrow(insert_query, config_id, tenant_id)

            return BatchConfigResponse(**dict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch config: {str(e)}")

@router.put("/config", response_model=BatchConfigResponse)
async def update_batch_config(
    config: BatchConfigUpdate,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Update batch configuration"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Build update query
        updates = []
        params = [tenant_id]
        param_count = 1

        for field, value in config.dict(exclude_unset=True).items():
            if value is not None:
                param_count += 1
                updates.append(f"{field} = ${param_count}")
                params.append(value)

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        async with get_db_connection() as conn:
            query = f"""
                UPDATE batch.configs
                SET {', '.join(updates)}, updated_at = NOW()
                WHERE tenant_id = $1
                RETURNING *
            """
            result = await conn.fetchrow(query, *params)

            if not result:
                raise HTTPException(status_code=404, detail="Config not found")

            return BatchConfigResponse(**dict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update batch config: {str(e)}")

@router.get("/metrics", response_model=List[BatchMetricsResponse])
async def get_batch_metrics(
    days: int = Query(default=7, ge=1, le=90),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get batch processing metrics"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.performance_metrics
                WHERE tenant_id = $1
                AND metric_date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY metric_date DESC
            """
            results = await conn.fetch(query % days, tenant_id)

            return [BatchMetricsResponse(**dict(result)) for result in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch metrics: {str(e)}")

@router.get("/efficiency", response_model=BatchEfficiencyResponse)
async def get_batch_efficiency(
    days: int = Query(default=7, ge=1, le=90),
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get batch efficiency analysis"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.calculate_batch_efficiency($1, $2)
            """
            result = await conn.fetchrow(query, tenant_id, days)

            return BatchEfficiencyResponse(**dict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch efficiency: {str(e)}")

@router.get("/queue/status", response_model=List[QueueStatusResponse])
async def get_queue_status(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get current queue status"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.queue_performance
                WHERE tenant_id = $1
                ORDER BY priority DESC
            """
            results = await conn.fetch(query, tenant_id)

            return [QueueStatusResponse(**dict(result)) for result in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

@router.get("/dashboard", response_model=BatchDashboardResponse)
async def get_batch_dashboard(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Get batch processing dashboard data"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.tenant_dashboard_stats
                WHERE tenant_id = $1
            """
            result = await conn.fetchrow(query, tenant_id)

            if not result:
                # Return empty dashboard
                return BatchDashboardResponse(
                    total_requests=0,
                    total_batches=0,
                    avg_batch_size=0,
                    completed_requests=0,
                    failed_requests=0,
                    pending_requests=0,
                    processing_requests=0,
                    avg_processing_time_ms=0,
                    total_cost_savings=0,
                    avg_cost_savings_percent=0,
                    unique_models=0,
                    last_request_at=None,
                    requests_last_24h=0
                )

            return BatchDashboardResponse(**dict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get batch dashboard: {str(e)}")

@router.post("/optimize")
async def optimize_batch_performance(
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Trigger batch performance optimization"""
    try:
        # Verify token
        await verify_token(token.credentials)

        # Start optimization in background
        background_tasks.add_task(batch_processor.optimize_performance, tenant_id)

        return {"message": "Batch optimization started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

@router.post("/schedules", response_model=dict)
async def create_batch_schedule(
    schedule: BatchScheduleCreate,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Create a batch processing schedule"""
    try:
        # Verify token
        await verify_token(token.credentials)

        schedule_id = uuid4()

        async with get_db_connection() as conn:
            query = """
                INSERT INTO batch.schedules (
                    schedule_id, tenant_id, schedule_name, cron_expression,
                    batch_config, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING schedule_id
            """
            result = await conn.fetchrow(
                query, schedule_id, tenant_id, schedule.schedule_name,
                schedule.cron_expression, json.dumps(schedule.batch_config),
                schedule.is_active
            )

            return {"schedule_id": result["schedule_id"], "message": "Schedule created successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create schedule: {str(e)}")

@router.get("/schedules")
async def list_batch_schedules(
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """List batch processing schedules"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                SELECT * FROM batch.schedules
                WHERE tenant_id = $1
                ORDER BY created_at DESC
            """
            results = await conn.fetch(query, tenant_id)

            return [dict(result) for result in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list schedules: {str(e)}")

@router.delete("/schedules/{schedule_id}")
async def delete_batch_schedule(
    schedule_id: UUID,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Delete a batch processing schedule"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            query = """
                DELETE FROM batch.schedules
                WHERE schedule_id = $1 AND tenant_id = $2
            """
            result = await conn.execute(query, schedule_id, tenant_id)

            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Schedule not found")

            return {"message": "Schedule deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete schedule: {str(e)}")

@router.post("/refresh-dashboard")
async def refresh_dashboard(
    background_tasks: BackgroundTasks,
    tenant_id: UUID = Depends(get_current_tenant),
    token: str = Depends(security)
):
    """Refresh dashboard materialized view"""
    try:
        # Verify token
        await verify_token(token.credentials)

        async with get_db_connection() as conn:
            await conn.execute("SELECT batch.refresh_dashboard_stats()")

        return {"message": "Dashboard refreshed successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh dashboard: {str(e)}")

# Health check endpoint
@router.get("/health")
async def batch_health_check():
    """Health check for batch processing system"""
    try:
        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")

        return {
            "status": "healthy",
            "service": "batch_processing",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
