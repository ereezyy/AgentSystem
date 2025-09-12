"""
Intelligent Caching API Endpoints - AgentSystem Profit Machine
Advanced multi-level caching system endpoints for 60% AI cost reduction
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field
import asyncpg
import uuid

from ..caching.intelligent_cache import (
    CacheManager, CacheHit, CacheStats, CacheLevel, CacheStrategy
)
from ..auth.auth_service import get_current_user, require_permissions
from ..database.connection import get_db_pool
import aioredis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/cache", tags=["caching"])
security = HTTPBearer()

# Pydantic models for request/response
class CacheQueryRequest(BaseModel):
    request_content: str = Field(..., description="Content to check in cache")
    model: str = Field(..., description="AI model being used")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

class CacheStoreRequest(BaseModel):
    request_content: str = Field(..., description="Original request content")
    response_content: str = Field(..., description="AI response content")
    model: str = Field(..., description="AI model used")
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    cost: float = Field(..., description="Cost of the request")
    quality_score: float = Field(default=85.0, description="Quality score (0-100)")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=1000, description="Maximum tokens")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")

class CacheHitResponse(BaseModel):
    cache_id: str
    similarity_score: float
    cache_level: str
    adaptation_needed: bool
    estimated_cost_savings: float
    request_content: str
    response_content: str
    quality_score: float
    access_count: int
    created_at: datetime

class CacheStatsResponse(BaseModel):
    tenant_id: str
    total_requests: int
    cache_hits: int
    exact_hits: int
    semantic_hits: int
    partial_hits: int
    template_hits: int
    cache_miss: int
    hit_rate: float
    cost_savings: float
    storage_used_mb: float
    avg_response_time_ms: float
    period_days: int

class CacheConfigRequest(BaseModel):
    strategy: str = Field(default="balanced", description="Caching strategy")
    exact_match_ttl: int = Field(default=604800, description="TTL for exact matches (seconds)")
    semantic_match_ttl: int = Field(default=259200, description="TTL for semantic matches (seconds)")
    similarity_threshold: float = Field(default=0.85, description="Similarity threshold (0-1)")
    max_cache_size_mb: int = Field(default=1000, description="Max cache size in MB")
    warming_enabled: bool = Field(default=True, description="Enable cache warming")
    compression_enabled: bool = Field(default=True, description="Enable compression")

class CacheWarmingRequest(BaseModel):
    predictions: List[Dict[str, Any]] = Field(..., description="List of predicted requests to warm")

class CacheInvalidationRequest(BaseModel):
    pattern: Optional[str] = Field(None, description="Pattern to match for invalidation")
    model: Optional[str] = Field(None, description="Model to invalidate cache for")
    tags: Optional[List[str]] = Field(None, description="Tags to match for invalidation")

# Dependency to get cache manager
async def get_cache_manager() -> CacheManager:
    db_pool = await get_db_pool()
    # Get Redis client (simplified for example)
    redis_client = None  # Would be initialized properly in production
    return CacheManager(db_pool, redis_client)

@router.post("/query", response_model=Optional[CacheHitResponse])
async def query_cache(
    request: CacheQueryRequest,
    current_user = Depends(get_current_user),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Query cache for existing AI response"""

    try:
        cache_hit = await cache_manager.get_cached_response(
            tenant_id=current_user['tenant_id'],
            request_content=request.request_content,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata=request.metadata
        )

        if cache_hit:
            return CacheHitResponse(
                cache_id=cache_hit.cache_entry.cache_id,
                similarity_score=cache_hit.similarity_score,
                cache_level=cache_hit.cache_level.value,
                adaptation_needed=cache_hit.adaptation_needed,
                estimated_cost_savings=cache_hit.estimated_cost_savings,
                request_content=cache_hit.cache_entry.request_content,
                response_content=cache_hit.cache_entry.response_content,
                quality_score=cache_hit.cache_entry.quality_score,
                access_count=cache_hit.cache_entry.access_count,
                created_at=cache_hit.cache_entry.created_at
            )

        return None

    except Exception as e:
        logger.error(f"Error querying cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache query failed: {str(e)}")

@router.post("/store")
async def store_in_cache(
    request: CacheStoreRequest,
    current_user = Depends(get_current_user),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Store AI response in intelligent cache"""

    try:
        cache_id = await cache_manager.store_response(
            tenant_id=current_user['tenant_id'],
            request_content=request.request_content,
            response_content=request.response_content,
            model=request.model,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            cost=request.cost,
            quality_score=request.quality_score,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata=request.metadata
        )

        return {
            "cache_id": cache_id,
            "message": "Response stored in cache successfully"
        }

    except Exception as e:
        logger.error(f"Error storing in cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache storage failed: {str(e)}")

@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_statistics(
    days: int = Query(7, description="Number of days to analyze", le=90),
    current_user = Depends(get_current_user),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Get detailed cache statistics and analytics"""

    try:
        stats = await cache_manager.get_analytics(current_user['tenant_id'], days)

        return CacheStatsResponse(
            tenant_id=stats.tenant_id,
            total_requests=stats.total_requests,
            cache_hits=stats.cache_hits,
            exact_hits=stats.exact_hits,
            semantic_hits=stats.semantic_hits,
            partial_hits=stats.partial_hits,
            template_hits=stats.template_hits,
            cache_miss=stats.cache_miss,
            hit_rate=stats.hit_rate,
            cost_savings=stats.cost_savings,
            storage_used_mb=stats.storage_used_mb,
            avg_response_time_ms=stats.avg_response_time_ms,
            period_days=days
        )

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@router.get("/dashboard")
async def get_cache_dashboard(
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Get cache dashboard overview"""

    async with db_pool.acquire() as conn:
        # Get dashboard data from materialized view
        dashboard_data = await conn.fetchrow("""
            SELECT * FROM caching.cache_dashboard_stats
            WHERE tenant_id = $1
        """, current_user['tenant_id'])

        if not dashboard_data:
            return {
                'tenant_id': current_user['tenant_id'],
                'total_entries': 0,
                'reused_entries': 0,
                'avg_access_count': 0,
                'total_cost_saved': 0,
                'avg_quality': 0,
                'storage_mb': 0,
                'unique_models': 0,
                'last_activity': None,
                'entries_last_24h': 0
            }

        # Get recent cache activity
        recent_activity = await conn.fetch("""
            SELECT cache_level, cache_hit, cost_savings, created_at
            FROM caching.cache_requests
            WHERE tenant_id = $1
            ORDER BY created_at DESC
            LIMIT 10
        """, current_user['tenant_id'])

        # Get cache hit rate trends
        hit_rate_trends = await conn.fetch("""
            SELECT date, hit_rate, total_requests, total_cost_savings
            FROM caching.cache_hit_rate_daily
            WHERE tenant_id = $1
            AND date > CURRENT_DATE - INTERVAL '30 days'
            ORDER BY date DESC
        """, current_user['tenant_id'])

        # Get top cached models
        top_models = await conn.fetch("""
            SELECT model_used, cache_entries, total_accesses, total_cost_saved
            FROM caching.top_cached_models
            WHERE tenant_id = $1
            ORDER BY total_accesses DESC
            LIMIT 5
        """, current_user['tenant_id'])

        # Get cache efficiency score
        efficiency_data = await conn.fetchrow("""
            SELECT * FROM caching.calculate_cache_efficiency($1, 7)
        """, current_user['tenant_id'])

    return {
        'summary': dict(dashboard_data),
        'recent_activity': [dict(activity) for activity in recent_activity],
        'hit_rate_trends': [dict(trend) for trend in hit_rate_trends],
        'top_models': [dict(model) for model in top_models],
        'efficiency': dict(efficiency_data) if efficiency_data else None
    }

@router.get("/config")
async def get_cache_config(
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Get current cache configuration"""

    async with db_pool.acquire() as conn:
        config = await conn.fetchrow("""
            SELECT strategy, exact_match_ttl, semantic_match_ttl, partial_match_ttl,
                   template_match_ttl, similarity_threshold, max_cache_size_mb,
                   max_entries_per_tenant, warming_enabled, compression_enabled,
                   auto_cleanup_enabled, created_at, updated_at
            FROM caching.cache_config
            WHERE tenant_id = $1
        """, current_user['tenant_id'])

        if not config:
            # Return default configuration
            return {
                'strategy': 'balanced',
                'exact_match_ttl': 604800,
                'semantic_match_ttl': 259200,
                'partial_match_ttl': 86400,
                'template_match_ttl': 43200,
                'similarity_threshold': 0.85,
                'max_cache_size_mb': 1000,
                'max_entries_per_tenant': 10000,
                'warming_enabled': True,
                'compression_enabled': True,
                'auto_cleanup_enabled': True
            }

        return dict(config)

@router.put("/config")
async def update_cache_config(
    request: CacheConfigRequest,
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Update cache configuration"""

    # Validate strategy
    valid_strategies = ['aggressive', 'balanced', 'conservative', 'custom']
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"
        )

    async with db_pool.acquire() as conn:
        # Update or insert configuration
        await conn.execute("""
            INSERT INTO caching.cache_config (
                tenant_id, strategy, exact_match_ttl, semantic_match_ttl,
                similarity_threshold, max_cache_size_mb, warming_enabled,
                compression_enabled, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (tenant_id)
            DO UPDATE SET
                strategy = EXCLUDED.strategy,
                exact_match_ttl = EXCLUDED.exact_match_ttl,
                semantic_match_ttl = EXCLUDED.semantic_match_ttl,
                similarity_threshold = EXCLUDED.similarity_threshold,
                max_cache_size_mb = EXCLUDED.max_cache_size_mb,
                warming_enabled = EXCLUDED.warming_enabled,
                compression_enabled = EXCLUDED.compression_enabled,
                updated_at = EXCLUDED.updated_at
        """, current_user['tenant_id'], request.strategy, request.exact_match_ttl,
            request.semantic_match_ttl, request.similarity_threshold,
            request.max_cache_size_mb, request.warming_enabled, request.compression_enabled)

    return {"message": "Cache configuration updated successfully"}

@router.post("/warm")
async def warm_cache(
    request: CacheWarmingRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Warm cache with predicted requests"""

    if len(request.predictions) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 predictions per request")

    # Validate predictions format
    for prediction in request.predictions:
        required_fields = ['content', 'model']
        if not all(field in prediction for field in required_fields):
            raise HTTPException(
                status_code=400,
                detail=f"Each prediction must contain: {', '.join(required_fields)}"
            )

    try:
        # Queue cache warming in background
        async def perform_warming():
            cache = await cache_manager.get_cache(current_user['tenant_id'])
            warmed_count = await cache.warm_cache(current_user['tenant_id'], request.predictions)
            logger.info(f"Warmed {warmed_count} cache entries for tenant {current_user['tenant_id']}")

        background_tasks.add_task(perform_warming)

        return {
            "message": "Cache warming queued successfully",
            "predictions_queued": len(request.predictions)
        }

    except Exception as e:
        logger.error(f"Error queuing cache warming: {e}")
        raise HTTPException(status_code=500, detail=f"Cache warming failed: {str(e)}")

@router.post("/invalidate")
async def invalidate_cache(
    request: CacheInvalidationRequest,
    current_user = Depends(get_current_user),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Invalidate cache entries based on criteria"""

    if not any([request.pattern, request.model, request.tags]):
        raise HTTPException(
            status_code=400,
            detail="Must specify at least one invalidation criteria: pattern, model, or tags"
        )

    try:
        invalidated_count = await cache_manager.invalidate_cache(
            tenant_id=current_user['tenant_id'],
            pattern=request.pattern,
            model=request.model,
            tags=request.tags
        )

        return {
            "message": "Cache invalidation completed",
            "entries_invalidated": invalidated_count
        }

    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache invalidation failed: {str(e)}")

@router.get("/entries")
async def list_cache_entries(
    model: Optional[str] = Query(None, description="Filter by model"),
    min_quality: Optional[float] = Query(None, description="Minimum quality score"),
    min_access_count: Optional[int] = Query(None, description="Minimum access count"),
    limit: int = Query(50, le=100, description="Number of entries to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """List cache entries with filtering options"""

    query = """
        SELECT cache_id, request_hash, model_used, quality_score, cost_saved,
               access_count, last_accessed, ttl_seconds, tags, created_at,
               LENGTH(request_content) as request_size,
               LENGTH(response_content) as response_size
        FROM caching.cache_entries
        WHERE tenant_id = $1 AND is_active = true
    """
    params = [current_user['tenant_id']]
    param_count = 1

    if model:
        param_count += 1
        query += f" AND model_used = ${param_count}"
        params.append(model)

    if min_quality:
        param_count += 1
        query += f" AND quality_score >= ${param_count}"
        params.append(min_quality)

    if min_access_count:
        param_count += 1
        query += f" AND access_count >= ${param_count}"
        params.append(min_access_count)

    query += f" ORDER BY access_count DESC, last_accessed DESC LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
    params.extend([limit, offset])

    async with db_pool.acquire() as conn:
        entries = await conn.fetch(query, *params)

    return [dict(entry) for entry in entries]

@router.delete("/entries/{cache_id}")
async def delete_cache_entry(
    cache_id: str,
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Delete a specific cache entry"""

    async with db_pool.acquire() as conn:
        # Verify entry belongs to tenant
        entry = await conn.fetchrow("""
            SELECT cache_id FROM caching.cache_entries
            WHERE cache_id = $1 AND tenant_id = $2
        """, cache_id, current_user['tenant_id'])

        if not entry:
            raise HTTPException(status_code=404, detail="Cache entry not found")

        # Mark as inactive
        await conn.execute("""
            UPDATE caching.cache_entries
            SET is_active = false, updated_at = NOW()
            WHERE cache_id = $1
        """, cache_id)

    return {"message": "Cache entry deleted successfully"}

@router.get("/analytics/efficiency")
async def get_cache_efficiency_analytics(
    days: int = Query(7, description="Number of days to analyze", le=90),
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Get detailed cache efficiency analytics"""

    async with db_pool.acquire() as conn:
        # Get efficiency data
        efficiency = await conn.fetchrow("""
            SELECT * FROM caching.calculate_cache_efficiency($1, $2)
        """, current_user['tenant_id'], days)

        # Get efficiency trends
        efficiency_trends = await conn.fetch("""
            SELECT date, reuse_rate, avg_quality, total_cost_saved, storage_mb
            FROM caching.cache_efficiency_summary
            WHERE tenant_id = $1
            AND date > CURRENT_DATE - INTERVAL '%s days'
            ORDER BY date DESC
        """ % days, current_user['tenant_id'])

        # Get storage usage breakdown
        storage_usage = await conn.fetchrow("""
            SELECT * FROM caching.cache_storage_usage
            WHERE tenant_id = $1
        """, current_user['tenant_id'])

    return {
        'efficiency_score': dict(efficiency) if efficiency else None,
        'efficiency_trends': [dict(trend) for trend in efficiency_trends],
        'storage_usage': dict(storage_usage) if storage_usage else None
    }

@router.post("/cleanup")
async def manual_cache_cleanup(
    current_user = Depends(get_current_user),
    db_pool: asyncpg.Pool = Depends(get_db_pool)
):
    """Manually trigger cache cleanup"""

    async with db_pool.acquire() as conn:
        # Clean up expired entries
        expired_count = await conn.fetchval("""
            UPDATE caching.cache_entries
            SET is_active = false
            WHERE tenant_id = $1 AND expires_at < NOW() AND is_active = true
            RETURNING COUNT(*)
        """, current_user['tenant_id'])

        # Clean up least accessed entries if over limit
        config = await conn.fetchrow("""
            SELECT max_entries_per_tenant FROM caching.cache_config
            WHERE tenant_id = $1
        """, current_user['tenant_id'])

        max_entries = config['max_entries_per_tenant'] if config else 10000

        # Count current active entries
        current_count = await conn.fetchval("""
            SELECT COUNT(*) FROM caching.cache_entries
            WHERE tenant_id = $1 AND is_active = true
        """, current_user['tenant_id'])

        lru_cleaned = 0
        if current_count > max_entries:
            lru_cleaned = await conn.fetchval("""
                UPDATE caching.cache_entries
                SET is_active = false
                WHERE cache_id IN (
                    SELECT cache_id FROM caching.cache_entries
                    WHERE tenant_id = $1 AND is_active = true
                    ORDER BY last_accessed ASC, access_count ASC
                    LIMIT $2
                ) RETURNING COUNT(*)
            """, current_user['tenant_id'], current_count - max_entries)

    return {
        "message": "Cache cleanup completed",
        "expired_entries_cleaned": expired_count or 0,
        "lru_entries_cleaned": lru_cleaned or 0
    }

@router.get("/strategies")
async def get_cache_strategies():
    """Get available caching strategies"""

    strategies = [
        {
            "strategy": "aggressive",
            "name": "Aggressive Caching",
            "description": "Cache everything with high hit rates, higher storage usage",
            "recommended_for": "High-volume, repeated workloads"
        },
        {
            "strategy": "balanced",
            "name": "Balanced Caching",
            "description": "Optimal balance between hit rate and storage efficiency",
            "recommended_for": "General purpose usage"
        },
        {
            "strategy": "conservative",
            "name": "Conservative Caching",
            "description": "Cache only high-value items, lower storage usage",
            "recommended_for": "Storage-constrained environments"
        },
        {
            "strategy": "custom",
            "name": "Custom Configuration",
            "description": "Manually configured caching rules and thresholds",
            "recommended_for": "Specific optimization requirements"
        }
    ]

    return {"strategies": strategies}

# Include router in main application
def setup_caching_routes(app):
    """Setup caching routes in FastAPI application"""
    app.include_router(router)
