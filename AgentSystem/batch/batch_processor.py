"""
Batch Processing Engine - AgentSystem Profit Machine
Advanced batch optimization for bulk AI operations and cost reduction
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict, field
import asyncpg
import aioredis
import heapq
import uuid
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class BatchStrategy(Enum):
    TIME_BASED = "time_based"        # Batch by time window
    SIZE_BASED = "size_based"        # Batch by request count
    COST_BASED = "cost_based"        # Batch for optimal cost
    LATENCY_BASED = "latency_based"  # Batch for optimal latency
    ADAPTIVE = "adaptive"            # AI-driven adaptive batching

class BatchPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class BatchStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class BatchRequest:
    request_id: str
    tenant_id: str
    model: str
    capability: str
    content: str
    priority: BatchPriority
    max_latency_ms: Optional[int]
    cost_budget: Optional[float]
    callback_url: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0
    estimated_cost: float = 0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None

@dataclass
class Batch:
    batch_id: str
    tenant_id: str
    model: str
    capability: str
    strategy: BatchStrategy
    requests: List[BatchRequest] = field(default_factory=list)
    status: BatchStatus = BatchStatus.QUEUED
    priority: BatchPriority = BatchPriority.NORMAL
    estimated_cost: float = 0
    estimated_tokens: int = 0
    estimated_processing_time: float = 0
    actual_cost: Optional[float] = None
    actual_processing_time: Optional[float] = None
    cost_savings: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_details: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BatchConfig:
    tenant_id: str
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    max_batch_size: int = 50
    min_batch_size: int = 2
    max_wait_time_seconds: int = 30
    cost_threshold: float = 10.0
    enable_smart_grouping: bool = True
    enable_priority_queue: bool = True
    retry_failed_requests: bool = True
    parallel_batches: int = 3
    cost_savings_target: float = 20.0  # Target cost savings percentage

@dataclass
class BatchMetrics:
    tenant_id: str
    total_requests: int
    total_batches: int
    avg_batch_size: float
    avg_processing_time: float
    total_cost_savings: float
    success_rate: float
    throughput_per_hour: float
    avg_latency_reduction: float
    period_start: datetime
    period_end: datetime

class BatchProcessor:
    """
    Advanced batch processing engine for AI requests
    """

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis = redis_client

        # Request queues by priority
        self.request_queues = {
            BatchPriority.CRITICAL: deque(),
            BatchPriority.HIGH: deque(),
            BatchPriority.NORMAL: deque(),
            BatchPriority.LOW: deque()
        }

        # Active batches by tenant
        self.active_batches = defaultdict(list)

        # Batch configurations by tenant
        self.batch_configs = {}

        # Processing statistics
        self.stats = defaultdict(lambda: {
            'requests_processed': 0,
            'batches_created': 0,
            'cost_savings': 0.0,
            'avg_processing_time': 0.0,
            'success_rate': 0.0
        })

        # Background workers
        self.workers = []
        self.is_running = False

    async def initialize(self):
        """Initialize the batch processing engine"""

        # Load batch configurations
        await self._load_batch_configs()

        # Start background workers
        self.is_running = True

        # Start different types of workers
        for i in range(3):  # 3 general workers
            worker = asyncio.create_task(self._batch_worker(f"worker-{i}"))
            self.workers.append(worker)

        # Priority worker for critical requests
        priority_worker = asyncio.create_task(self._priority_worker())
        self.workers.append(priority_worker)

        # Batch optimizer
        optimizer = asyncio.create_task(self._batch_optimizer())
        self.workers.append(optimizer)

        # Metrics aggregator
        metrics_worker = asyncio.create_task(self._metrics_aggregator())
        self.workers.append(metrics_worker)

        logger.info("Batch Processing Engine initialized successfully")

    async def submit_request(self, request: BatchRequest) -> str:
        """Submit a request for batch processing"""

        # Estimate tokens and cost
        request.estimated_tokens = self._estimate_tokens(request.content)
        request.estimated_cost = await self._estimate_cost(request)

        # Set deadline if not provided
        if not request.deadline and request.max_latency_ms:
            request.deadline = datetime.now() + timedelta(milliseconds=request.max_latency_ms)

        # Add to appropriate queue
        await self._enqueue_request(request)

        # Store request in database
        await self._store_request(request)

        # Trigger immediate processing if high priority
        if request.priority in [BatchPriority.CRITICAL, BatchPriority.HIGH]:
            await self._trigger_immediate_processing(request.tenant_id)

        logger.debug(f"Submitted request {request.request_id} for batch processing")

        return request.request_id

    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of a batch request"""

        async with self.db_pool.acquire() as conn:
            # Get request details
            request_data = await conn.fetchrow("""
                SELECT br.*, b.batch_id, b.status as batch_status, b.started_at, b.completed_at
                FROM batch.requests br
                LEFT JOIN batch.batches b ON br.batch_id = b.batch_id
                WHERE br.request_id = $1
            """, request_id)

            if not request_data:
                raise ValueError(f"Request {request_id} not found")

            return {
                'request_id': request_data['request_id'],
                'status': request_data['status'],
                'batch_id': request_data['batch_id'],
                'batch_status': request_data['batch_status'],
                'priority': request_data['priority'],
                'estimated_cost': request_data['estimated_cost'],
                'actual_cost': request_data['actual_cost'],
                'created_at': request_data['created_at'],
                'started_at': request_data['started_at'],
                'completed_at': request_data['completed_at'],
                'retry_count': request_data['retry_count'],
                'error_details': request_data['error_details']
            }

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending batch request"""

        async with self.db_pool.acquire() as conn:
            # Check if request can be cancelled (not yet processing)
            request_data = await conn.fetchrow("""
                SELECT br.status, b.status as batch_status
                FROM batch.requests br
                LEFT JOIN batch.batches b ON br.batch_id = b.batch_id
                WHERE br.request_id = $1
            """, request_id)

            if not request_data:
                return False

            if request_data['status'] in ['completed', 'failed']:
                return False

            if request_data['batch_status'] == 'processing':
                return False

            # Mark as cancelled
            await conn.execute("""
                UPDATE batch.requests
                SET status = 'cancelled', updated_at = NOW()
                WHERE request_id = $1
            """, request_id)

            return True

    async def get_batch_analytics(self, tenant_id: str, days: int = 7) -> BatchMetrics:
        """Get batch processing analytics"""

        async with self.db_pool.acquire() as conn:
            # Get batch metrics
            metrics_data = await conn.fetchrow("""
                SELECT
                    COUNT(DISTINCT br.request_id) as total_requests,
                    COUNT(DISTINCT b.batch_id) as total_batches,
                    AVG(b.batch_size) as avg_batch_size,
                    AVG(EXTRACT(EPOCH FROM (b.completed_at - b.started_at))) as avg_processing_time,
                    SUM(b.cost_savings) as total_cost_savings,
                    COUNT(CASE WHEN br.status = 'completed' THEN 1 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(DISTINCT br.request_id) / EXTRACT(EPOCH FROM INTERVAL '%s days') * 3600 as throughput_per_hour
                FROM batch.requests br
                LEFT JOIN batch.batches b ON br.batch_id = b.batch_id
                WHERE br.tenant_id = $1
                AND br.created_at > NOW() - INTERVAL '%s days'
            """ % (days, days), tenant_id)

            if not metrics_data:
                return BatchMetrics(
                    tenant_id=tenant_id,
                    total_requests=0,
                    total_batches=0,
                    avg_batch_size=0,
                    avg_processing_time=0,
                    total_cost_savings=0,
                    success_rate=0,
                    throughput_per_hour=0,
                    avg_latency_reduction=0,
                    period_start=datetime.now() - timedelta(days=days),
                    period_end=datetime.now()
                )

            return BatchMetrics(
                tenant_id=tenant_id,
                total_requests=metrics_data['total_requests'] or 0,
                total_batches=metrics_data['total_batches'] or 0,
                avg_batch_size=metrics_data['avg_batch_size'] or 0,
                avg_processing_time=metrics_data['avg_processing_time'] or 0,
                total_cost_savings=metrics_data['total_cost_savings'] or 0,
                success_rate=metrics_data['success_rate'] or 0,
                throughput_per_hour=metrics_data['throughput_per_hour'] or 0,
                avg_latency_reduction=0,  # Would calculate from detailed metrics
                period_start=datetime.now() - timedelta(days=days),
                period_end=datetime.now()
            )

    async def optimize_batch_config(self, tenant_id: str) -> BatchConfig:
        """Optimize batch configuration based on usage patterns"""

        # Analyze recent performance
        performance_data = await self._analyze_batch_performance(tenant_id)

        # Get current config
        current_config = await self._get_batch_config(tenant_id)

        # Optimize parameters
        optimized_config = await self._optimize_config_parameters(current_config, performance_data)

        # Store optimized config
        await self._store_batch_config(optimized_config)

        self.batch_configs[tenant_id] = optimized_config

        return optimized_config

    # Private worker methods
    async def _batch_worker(self, worker_id: str):
        """Background worker for processing batches"""

        logger.info(f"Started batch worker: {worker_id}")

        while self.is_running:
            try:
                # Find pending batches to process
                batch = await self._get_next_batch()

                if batch:
                    await self._process_batch(batch)
                else:
                    # No batches ready, sleep briefly
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in batch worker {worker_id}: {e}")
                await asyncio.sleep(5)

    async def _priority_worker(self):
        """Worker dedicated to high-priority requests"""

        logger.info("Started priority batch worker")

        while self.is_running:
            try:
                # Process critical and high priority requests immediately
                for priority in [BatchPriority.CRITICAL, BatchPriority.HIGH]:
                    queue = self.request_queues[priority]

                    if queue:
                        # Create immediate mini-batches for high priority
                        requests = []

                        # Take up to 5 requests for quick processing
                        for _ in range(min(5, len(queue))):
                            if queue:
                                requests.append(queue.popleft())

                        if requests:
                            # Group by tenant and model
                            grouped = defaultdict(list)
                            for req in requests:
                                key = (req.tenant_id, req.model)
                                grouped[key].append(req)

                            # Create mini-batches
                            for (tenant_id, model), req_group in grouped.items():
                                batch = Batch(
                                    batch_id=str(uuid.uuid4()),
                                    tenant_id=tenant_id,
                                    model=model,
                                    capability=req_group[0].capability,
                                    strategy=BatchStrategy.LATENCY_BASED,
                                    requests=req_group,
                                    priority=priority
                                )

                                await self._process_batch(batch)

                await asyncio.sleep(0.5)  # Check frequently for priority requests

            except Exception as e:
                logger.error(f"Error in priority worker: {e}")
                await asyncio.sleep(2)

    async def _batch_optimizer(self):
        """Background optimizer for creating optimal batches"""

        logger.info("Started batch optimizer")

        while self.is_running:
            try:
                # Analyze queued requests and create optimal batches
                await self._create_optimal_batches()

                # Sleep based on queue activity
                total_queued = sum(len(q) for q in self.request_queues.values())
                sleep_time = 1 if total_queued > 100 else 5
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in batch optimizer: {e}")
                await asyncio.sleep(10)

    async def _metrics_aggregator(self):
        """Background worker for aggregating metrics"""

        logger.info("Started metrics aggregator")

        while self.is_running:
            try:
                await self._aggregate_batch_metrics()
                await asyncio.sleep(60)  # Aggregate every minute

            except Exception as e:
                logger.error(f"Error in metrics aggregator: {e}")
                await asyncio.sleep(30)

    # Core processing methods
    async def _enqueue_request(self, request: BatchRequest):
        """Add request to appropriate priority queue"""

        self.request_queues[request.priority].append(request)

        # Also store in Redis for persistence
        queue_key = f"batch_queue:{request.tenant_id}:{request.priority.value}"
        await self.redis.lpush(queue_key, json.dumps(asdict(request), default=str))

    async def _get_next_batch(self) -> Optional[Batch]:
        """Get next batch ready for processing"""

        # Look for existing ready batches
        async with self.db_pool.acquire() as conn:
            batch_data = await conn.fetchrow("""
                SELECT * FROM batch.batches
                WHERE status = 'queued'
                AND (scheduled_at IS NULL OR scheduled_at <= NOW())
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """)

            if batch_data:
                # Load batch with requests
                return await self._load_batch(batch_data['batch_id'])

        return None

    async def _create_optimal_batches(self):
        """Create optimal batches from queued requests"""

        # Group requests by tenant and model
        tenant_groups = defaultdict(lambda: defaultdict(list))

        # Collect requests from all priority queues
        for priority, queue in self.request_queues.items():
            while queue:
                request = queue.popleft()

                # Check if request has expired
                if request.deadline and datetime.now() > request.deadline:
                    await self._handle_expired_request(request)
                    continue

                tenant_groups[request.tenant_id][request.model].append(request)

        # Create batches for each tenant/model combination
        for tenant_id, model_groups in tenant_groups.items():
            config = await self._get_batch_config(tenant_id)

            for model, requests in model_groups.items():
                if len(requests) >= config.min_batch_size:
                    # Create optimized batches
                    batches = await self._optimize_batch_creation(requests, config)

                    for batch in batches:
                        await self._store_batch(batch)
                        self.active_batches[tenant_id].append(batch)

    async def _optimize_batch_creation(self, requests: List[BatchRequest],
                                     config: BatchConfig) -> List[Batch]:
        """Create optimally sized batches from requests"""

        batches = []

        # Sort requests by priority and deadline
        requests.sort(key=lambda r: (r.priority.value, r.deadline or datetime.max), reverse=True)

        current_batch = []
        current_cost = 0
        current_tokens = 0

        for request in requests:
            # Check if adding this request would exceed limits
            if (len(current_batch) >= config.max_batch_size or
                current_cost + request.estimated_cost > config.cost_threshold):

                # Create batch from current requests
                if current_batch:
                    batch = await self._create_batch_from_requests(current_batch, config)
                    batches.append(batch)

                    # Start new batch
                    current_batch = [request]
                    current_cost = request.estimated_cost
                    current_tokens = request.estimated_tokens
                else:
                    current_batch.append(request)
                    current_cost += request.estimated_cost
                    current_tokens += request.estimated_tokens
            else:
                current_batch.append(request)
                current_cost += request.estimated_cost
                current_tokens += request.estimated_tokens

        # Create final batch if there are remaining requests
        if current_batch:
            batch = await self._create_batch_from_requests(current_batch, config)
            batches.append(batch)

        return batches

    async def _create_batch_from_requests(self, requests: List[BatchRequest],
                                        config: BatchConfig) -> Batch:
        """Create a batch object from a list of requests"""

        if not requests:
            raise ValueError("Cannot create batch from empty request list")

        # Calculate batch metrics
        total_cost = sum(r.estimated_cost for r in requests)
        total_tokens = sum(r.estimated_tokens for r in requests)
        max_priority = max(r.priority for r in requests)

        # Estimate cost savings from batching
        individual_cost = sum(r.estimated_cost * 1.2 for r in requests)  # 20% overhead for individual requests
        cost_savings = individual_cost - total_cost

        batch = Batch(
            batch_id=str(uuid.uuid4()),
            tenant_id=requests[0].tenant_id,
            model=requests[0].model,
            capability=requests[0].capability,
            strategy=config.strategy,
            requests=requests,
            priority=max_priority,
            estimated_cost=total_cost,
            estimated_tokens=total_tokens,
            cost_savings=cost_savings,
            estimated_processing_time=self._estimate_processing_time(len(requests), total_tokens)
        )

        return batch

    async def _process_batch(self, batch: Batch):
        """Process a complete batch"""

        start_time = time.time()
        batch.status = BatchStatus.PROCESSING
        batch.started_at = datetime.now()

        try:
            # Update batch status in database
            await self._update_batch_status(batch, BatchStatus.PROCESSING)

            # Process requests in the batch
            results = await self._execute_batch_requests(batch)

            # Update results
            batch.results = results
            batch.status = BatchStatus.COMPLETED
            batch.completed_at = datetime.now()
            batch.actual_processing_time = time.time() - start_time

            # Calculate actual cost savings
            await self._calculate_actual_savings(batch)

            # Update database
            await self._update_batch_completion(batch)

            # Send callbacks if configured
            await self._send_batch_callbacks(batch)

            logger.info(f"Completed batch {batch.batch_id} with {len(batch.requests)} requests")

        except Exception as e:
            batch.status = BatchStatus.FAILED
            batch.error_details = str(e)
            batch.completed_at = datetime.now()

            await self._update_batch_status(batch, BatchStatus.FAILED)
            await self._handle_batch_failure(batch)

            logger.error(f"Failed to process batch {batch.batch_id}: {e}")

    async def _execute_batch_requests(self, batch: Batch) -> List[Dict[str, Any]]:
        """Execute the actual AI requests in a batch"""

        # This would integrate with the arbitrage engine and AI providers
        # For now, simulate batch processing

        results = []

        for request in batch.requests:
            try:
                # Simulate AI processing
                await asyncio.sleep(0.1)  # Simulate processing time

                result = {
                    'request_id': request.request_id,
                    'status': 'completed',
                    'response': f"Batch processed response for: {request.content[:100]}...",
                    'tokens_used': request.estimated_tokens,
                    'cost': request.estimated_cost * 0.8,  # 20% batch discount
                    'processing_time_ms': 100
                }

                results.append(result)

                # Update individual request status
                await self._update_request_completion(request, result)

            except Exception as e:
                result = {
                    'request_id': request.request_id,
                    'status': 'failed',
                    'error': str(e)
                }
                results.append(result)

                await self._update_request_failure(request, str(e))

        return results

    # Helper methods
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content"""
        # Simplified token estimation
        return len(content.split()) * 1.3  # Rough approximation

    async def _estimate_cost(self, request: BatchRequest) -> float:
        """Estimate cost for a request"""
        # Would integrate with pricing engine
        base_cost_per_token = 0.00002  # Example rate
        return request.estimated_tokens * base_cost_per_token

    def _estimate_processing_time(self, request_count: int, total_tokens: int) -> float:
        """Estimate batch processing time"""
        # Base time + token processing time + batch overhead
        base_time = 1.0  # 1 second base
        token_time = total_tokens * 0.00001  # 0.01ms per token
        batch_overhead = request_count * 0.1  # 100ms per request

        return base_time + token_time + batch_overhead

    async def _load_batch_configs(self):
        """Load batch configurations from database"""

        async with self.db_pool.acquire() as conn:
            configs = await conn.fetch("""
                SELECT * FROM batch.configs
            """)

            for config_row in configs:
                config = BatchConfig(
                    tenant_id=config_row['tenant_id'],
                    strategy=BatchStrategy(config_row['strategy']),
                    max_batch_size=config_row['max_batch_size'],
                    min_batch_size=config_row['min_batch_size'],
                    max_wait_time_seconds=config_row['max_wait_time_seconds'],
                    cost_threshold=config_row['cost_threshold'],
                    enable_smart_grouping=config_row['enable_smart_grouping'],
                    enable_priority_queue=config_row['enable_priority_queue'],
                    retry_failed_requests=config_row['retry_failed_requests'],
                    parallel_batches=config_row['parallel_batches'],
                    cost_savings_target=config_row['cost_savings_target']
                )

                self.batch_configs[config.tenant_id] = config

    async def _get_batch_config(self, tenant_id: str) -> BatchConfig:
        """Get batch configuration for tenant"""

        if tenant_id not in self.batch_configs:
            # Return default configuration
            return BatchConfig(tenant_id=tenant_id)

        return self.batch_configs[tenant_id]

    async def _trigger_immediate_processing(self, tenant_id: str):
        """Trigger immediate processing for high-priority requests"""

        # Signal priority worker to check queues
        await self.redis.publish(f"batch_priority_trigger:{tenant_id}", "process")

    # Database operations
    async def _store_request(self, request: BatchRequest):
        """Store batch request in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO batch.requests (
                    request_id, tenant_id, model, capability, content, priority,
                    max_latency_ms, cost_budget, callback_url, metadata,
                    estimated_tokens, estimated_cost, max_retries, deadline, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """, request.request_id, request.tenant_id, request.model, request.capability,
                request.content, request.priority.value, request.max_latency_ms,
                request.cost_budget, request.callback_url, json.dumps(request.metadata),
                request.estimated_tokens, request.estimated_cost, request.max_retries,
                request.deadline, request.created_at)

    async def _store_batch(self, batch: Batch):
        """Store batch in database"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO batch.batches (
                    batch_id, tenant_id, model, capability, strategy, status, priority,
                    batch_size, estimated_cost, estimated_tokens, estimated_processing_time,
                    cost_savings, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """, batch.batch_id, batch.tenant_id, batch.model, batch.capability,
                batch.strategy.value, batch.status.value, batch.priority.value,
                len(batch.requests), batch.estimated_cost, batch.estimated_tokens,
                batch.estimated_processing_time, batch.cost_savings, batch.created_at)

            # Link requests to batch
            for request in batch.requests:
                await conn.execute("""
                    UPDATE batch.requests
                    SET batch_id = $1, status = 'queued'
                    WHERE request_id = $2
                """, batch.batch_id, request.request_id)


class BatchManager:
    """
    High-level batch processing management orchestrator
    """

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis = redis_client
        self.processors = {}  # tenant_id -> BatchProcessor

    async def get_processor(self, tenant_id: str) -> BatchProcessor:
        """Get or create batch processor for tenant"""

        if tenant_id not in self.processors:
            processor = BatchProcessor(self.db_pool, self.redis)
            await processor.initialize()
            self.processors[tenant_id] = processor

        return self.processors[tenant_id]

    async def submit_batch_request(self, tenant_id: str, **kwargs) -> str:
        """Submit request for batch processing"""

        processor = await self.get_processor(tenant_id)

        request = BatchRequest(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            **kwargs
        )

        return await processor.submit_request(request)

    async def get_analytics(self, tenant_id: str, days: int = 7) -> BatchMetrics:
        """Get batch processing analytics"""

        processor = await self.get_processor(tenant_id)
        return await processor.get_batch_analytics(tenant_id, days)
