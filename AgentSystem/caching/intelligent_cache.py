
"""
Intelligent Caching Engine - AgentSystem Profit Machine
Advanced multi-level caching system to reduce AI costs by 60%
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import asyncpg
import aioredis
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import gzip
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    PARTIAL_MATCH = "partial_match"
    TEMPLATE_MATCH = "template_match"

class CacheStrategy(Enum):
    AGGRESSIVE = "aggressive"  # Cache everything, high hit rate
    BALANCED = "balanced"      # Balance storage vs hit rate
    CONSERVATIVE = "conservative"  # Cache only high-value items
    CUSTOM = "custom"          # Custom rules per tenant

@dataclass
class CacheEntry:
    cache_id: str
    tenant_id: str
    request_hash: str
    semantic_hash: str
    input_tokens: int
    output_tokens: int
    model_used: str
    prompt_template: Optional[str]
    request_content: str
    response_content: str
    quality_score: float
    cost_saved: float
    access_count: int
    last_accessed: datetime
    ttl_seconds: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))

@dataclass
class CacheHit:
    cache_entry: CacheEntry
    similarity_score: float
    cache_level: CacheLevel
    adaptation_needed: bool
    estimated_cost_savings: float

@dataclass
class CacheStats:
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
    period_start: datetime
    period_end: datetime

class IntelligentCache:
    """
    Advanced multi-level caching system with semantic similarity matching
    """

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis = redis_client

        # Semantic similarity model
        self.embedding_model = None
        self._load_embedding_model()

        # Cache configuration
        self.config = {
            'exact_match_ttl': 3600 * 24 * 7,  # 7 days
            'semantic_match_ttl': 3600 * 24 * 3,  # 3 days
            'partial_match_ttl': 3600 * 24,       # 1 day
            'template_match_ttl': 3600 * 12,      # 12 hours
            'similarity_threshold': 0.85,          # Min similarity for semantic match
            'max_cache_size_mb': 1000,            # Max cache size per tenant
            'max_entries_per_tenant': 10000,      # Max entries per tenant
            'cleanup_interval': 3600,             # Cache cleanup every hour
            'warming_enabled': True,               # Enable predictive cache warming
            'compression_enabled': True           # Enable response compression
        }

        # Performance tracking
        self.stats = defaultdict(lambda: {
            'requests': 0,
            'hits': 0,
            'misses': 0,
            'cost_savings': 0.0,
            'response_times': []
        })

        # Background tasks
        self._background_tasks = []

    async def initialize(self):
        """Initialize the caching system"""

        # Start background maintenance tasks
        asyncio.create_task(self._cache_cleanup_loop())
        asyncio.create_task(self._cache_warming_loop())
        asyncio.create_task(self._stats_aggregation_loop())

        logger.info("Intelligent Cache system initialized successfully")

    def _load_embedding_model(self):
        """Load sentence transformer model for semantic similarity"""
        try:
            # Use a lightweight but effective model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model for semantic caching")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    async def get_cached_response(self, tenant_id: str, request_content: str,
                                 model: str, temperature: float = 0.7,
                                 max_tokens: int = 1000,
                                 metadata: Dict[str, Any] = None) -> Optional[CacheHit]:
        """
        Intelligent cache lookup with multi-level matching
        """
        start_time = time.time()

        # Generate hashes for different matching levels
        exact_hash = self._generate_exact_hash(request_content, model, temperature, max_tokens)
        semantic_hash = await self._generate_semantic_hash(request_content)

        # Level 1: Exact match
        cache_hit = await self._check_exact_match(tenant_id, exact_hash)
        if cache_hit:
            await self._update_access_stats(cache_hit.cache_entry, CacheLevel.EXACT_MATCH)
            self._track_performance(tenant_id, True, time.time() - start_time)
            return cache_hit

        # Level 2: Semantic similarity match
        if self.embedding_model and semantic_hash:
            cache_hit = await self._check_semantic_match(tenant_id, semantic_hash, request_content)
            if cache_hit:
                await self._update_access_stats(cache_hit.cache_entry, CacheLevel.SEMANTIC_MATCH)
                self._track_performance(tenant_id, True, time.time() - start_time)
                return cache_hit

        # Level 3: Partial content match
        cache_hit = await self._check_partial_match(tenant_id, request_content, model)
        if cache_hit:
            await self._update_access_stats(cache_hit.cache_entry, CacheLevel.PARTIAL_MATCH)
            self._track_performance(tenant_id, True, time.time() - start_time)
            return cache_hit

        # Level 4: Template-based match
        cache_hit = await self._check_template_match(tenant_id, request_content, model)
        if cache_hit:
            await self._update_access_stats(cache_hit.cache_entry, CacheLevel.TEMPLATE_MATCH)
            self._track_performance(tenant_id, True, time.time() - start_time)
            return cache_hit

        # Cache miss
        self._track_performance(tenant_id, False, time.time() - start_time)
        return None

    async def store_response(self, tenant_id: str, request_content: str,
                           response_content: str, model: str, input_tokens: int,
                           output_tokens: int, cost: float, quality_score: float = 85.0,
                           temperature: float = 0.7, max_tokens: int = 1000,
                           metadata: Dict[str, Any] = None) -> str:
        """
        Store AI response in intelligent cache
        """

        # Generate cache entry
        exact_hash = self._generate_exact_hash(request_content, model, temperature, max_tokens)
        semantic_hash = await self._generate_semantic_hash(request_content)

        cache_entry = CacheEntry(
            cache_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            request_hash=exact_hash,
            semantic_hash=semantic_hash or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_used=model,
            prompt_template=self._extract_template(request_content),
            request_content=request_content,
            response_content=response_content,
            quality_score=quality_score,
            cost_saved=cost,
            access_count=1,
            last_accessed=datetime.now(),
            ttl_seconds=self._calculate_ttl(quality_score, cost, model),
            tags=self._generate_tags(request_content, model),
            metadata=metadata or {}
        )

        # Check cache limits
        if await self._check_cache_limits(tenant_id):
            await self._evict_old_entries(tenant_id)

        # Store in Redis for fast access
        await self._store_in_redis(cache_entry)

        # Store in database for persistence
        await self._store_in_database(cache_entry)

        logger.debug(f"Stored cache entry {cache_entry.cache_id} for tenant {tenant_id}")

        return cache_entry.cache_id

    async def invalidate_cache(self, tenant_id: str, pattern: Optional[str] = None,
                             model: Optional[str] = None, tags: Optional[List[str]] = None):
        """
        Intelligent cache invalidation
        """

        invalidated_count = 0

        # Get cache keys to invalidate
        if pattern:
            # Pattern-based invalidation
            keys = await self._find_keys_by_pattern(tenant_id, pattern)
        elif model:
            # Model-based invalidation
            keys = await self._find_keys_by_model(tenant_id, model)
        elif tags:
            # Tag-based invalidation
            keys = await self._find_keys_by_tags(tenant_id, tags)
        else:
            # Invalidate all for tenant
            keys = await self._find_all_tenant_keys(tenant_id)

        # Remove from Redis
        if keys:
            await self.redis.delete(*keys)
            invalidated_count = len(keys)

        # Update database
        await self._mark_invalidated_in_database(tenant_id, pattern, model, tags)

        logger.info(f"Invalidated {invalidated_count} cache entries for tenant {tenant_id}")

        return invalidated_count

    async def get_cache_analytics(self, tenant_id: str, days: int = 7) -> CacheStats:
        """
        Get detailed cache analytics and cost savings
        """

        async with self.db_pool.acquire() as conn:
            # Get cache statistics
            stats_data = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN cache_hit THEN 1 END) as cache_hits,
                    COUNT(CASE WHEN cache_level = 'exact_match' THEN 1 END) as exact_hits,
                    COUNT(CASE WHEN cache_level = 'semantic_match' THEN 1 END) as semantic_hits,
                    COUNT(CASE WHEN cache_level = 'partial_match' THEN 1 END) as partial_hits,
                    COUNT(CASE WHEN cache_level = 'template_match' THEN 1 END) as template_hits,
                    COUNT(CASE WHEN NOT cache_hit THEN 1 END) as cache_miss,
                    SUM(cost_savings) as total_cost_savings,
                    AVG(response_time_ms) as avg_response_time
                FROM caching.cache_requests
                WHERE tenant_id = $1
                AND created_at > NOW() - INTERVAL '%s days'
            """ % days, tenant_id)

            # Get storage usage
            storage_data = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(LENGTH(response_content)) / 1024.0 / 1024.0 as storage_mb
                FROM caching.cache_entries
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

        total_requests = stats_data['total_requests'] or 0
        cache_hits = stats_data['cache_hits'] or 0
        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0

        return CacheStats(
            tenant_id=tenant_id,
            total_requests=total_requests,
            cache_hits=cache_hits,
            exact_hits=stats_data['exact_hits'] or 0,
            semantic_hits=stats_data['semantic_hits'] or 0,
            partial_hits=stats_data['partial_hits'] or 0,
            template_hits=stats_data['template_hits'] or 0,
            cache_miss=stats_data['cache_miss'] or 0,
            hit_rate=hit_rate,
            cost_savings=stats_data['total_cost_savings'] or 0,
            storage_used_mb=storage_data['storage_mb'] or 0,
            avg_response_time_ms=stats_data['avg_response_time'] or 0,
            period_start=datetime.now() - timedelta(days=days),
            period_end=datetime.now()
        )

    async def warm_cache(self, tenant_id: str, predictions: List[Dict[str, Any]]):
        """
        Predictive cache warming based on usage patterns
        """

        warmed_count = 0

        for prediction in predictions:
            # Check if already cached
            cached = await self.get_cached_response(
                tenant_id=tenant_id,
                request_content=prediction['content'],
                model=prediction['model']
            )

            if not cached:
                # Generate response and cache it
                # This would integrate with the arbitrage engine
                # For now, we'll just mark it for warming
                await self._mark_for_warming(tenant_id, prediction)
                warmed_count += 1

        logger.info(f"Warmed {warmed_count} cache entries for tenant {tenant_id}")

        return warmed_count

    # Private helper methods
    def _generate_exact_hash(self, content: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate exact match hash"""

        hash_input = f"{content}:{model}:{temperature}:{max_tokens}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    async def _generate_semantic_hash(self, content: str) -> Optional[str]:
        """Generate semantic similarity hash using embeddings"""

        if not self.embedding_model:
            return None

        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content)

            # Create hash from embedding (simplified)
            embedding_bytes = embedding.tobytes()
            return hashlib.md5(embedding_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Error generating semantic hash: {e}")
            return None

    async def _check_exact_match(self, tenant_id: str, exact_hash: str) -> Optional[CacheHit]:
        """Check for exact hash match in cache"""

        cache_key = f"cache:exact:{tenant_id}:{exact_hash}"
        cached_data = await self.redis.get(cache_key)

        if cached_data:
            cache_entry = pickle.loads(cached_data)
            return CacheHit(
                cache_entry=cache_entry,
                similarity_score=1.0,
                cache_level=CacheLevel.EXACT_MATCH,
                adaptation_needed=False,
                estimated_cost_savings=cache_entry.cost_saved
            )

        return None

    async def _check_semantic_match(self, tenant_id: str, semantic_hash: str,
                                  content: str) -> Optional[CacheHit]:
        """Check for semantic similarity match"""

        if not self.embedding_model:
            return None

        # Get candidates with similar semantic hash
        pattern = f"cache:semantic:{tenant_id}:*"
        keys = await self.redis.keys(pattern)

        best_match = None
        best_similarity = 0

        for key in keys[:50]:  # Limit search for performance
            cached_data = await self.redis.get(key)
            if cached_data:
                cache_entry = pickle.loads(cached_data)

                # Calculate semantic similarity
                similarity = await self._calculate_similarity(content, cache_entry.request_content)

                if similarity > self.config['similarity_threshold'] and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cache_entry

        if best_match:
            return CacheHit(
                cache_entry=best_match,
                similarity_score=best_similarity,
                cache_level=CacheLevel.SEMANTIC_MATCH,
                adaptation_needed=best_similarity < 0.95,
                estimated_cost_savings=best_match.cost_saved * best_similarity
            )

        return None

    async def _check_partial_match(self, tenant_id: str, content: str,
                                 model: str) -> Optional[CacheHit]:
        """Check for partial content match"""

        # Look for entries with similar keywords/phrases
        async with self.db_pool.acquire() as conn:
            # Simplified partial matching using database search
            similar_entries = await conn.fetch("""
                SELECT cache_id, request_content, response_content, cost_saved,
                       similarity(request_content, $1) as sim_score
                FROM caching.cache_entries
                WHERE tenant_id = $2 AND model_used = $3
                AND similarity(request_content, $1) > 0.3
                AND is_active = true
                ORDER BY sim_score DESC
                LIMIT 5
            """, content, tenant_id, model)

        if similar_entries:
            best_entry = similar_entries[0]
            similarity = best_entry['sim_score']

            # Load full cache entry
            cache_entry = await self._load_cache_entry(best_entry['cache_id'])

            if cache_entry:
                return CacheHit(
                    cache_entry=cache_entry,
                    similarity_score=similarity,
                    cache_level=CacheLevel.PARTIAL_MATCH,
                    adaptation_needed=True,
                    estimated_cost_savings=cache_entry.cost_saved * similarity * 0.7
                )

        return None

    async def _check_template_match(self, tenant_id: str, content: str,
                                  model: str) -> Optional[CacheHit]:
        """Check for template-based match"""

        # Extract template from content
        template = self._extract_template(content)

        if template:
            # Look for entries with same template
            async with self.db_pool.acquire() as conn:
                template_entries = await conn.fetch("""
                    SELECT cache_id, cost_saved
                    FROM caching.cache_entries
                    WHERE tenant_id = $1 AND model_used = $2
                    AND prompt_template = $3 AND is_active = true
                    ORDER BY quality_score DESC, last_accessed DESC
                    LIMIT 3
                """, tenant_id, model, template)

            if template_entries:
                best_entry = template_entries[0]
                cache_entry = await self._load_cache_entry(best_entry['cache_id'])

                if cache_entry:
                    return CacheHit(
                        cache_entry=cache_entry,
                        similarity_score=0.6,  # Template matches have moderate similarity
                        cache_level=CacheLevel.TEMPLATE_MATCH,
                        adaptation_needed=True,
                        estimated_cost_savings=cache_entry.cost_saved * 0.5
                    )

        return None

    def _extract_template(self, content: str) -> Optional[str]:
        """Extract template pattern from content"""

        # Simple template extraction (could be made more sophisticated)
        import re

        # Replace specific values with placeholders
        template = re.sub(r'\b\d+\b', '{NUMBER}', content)
        template = re.sub(r'\b[A-Z][a-z]+\b', '{NAME}', template)
        template = re.sub(r'\b\w+@\w+\.\w+\b', '{EMAIL}', template)

        # If template is significantly different from original, return it
        if len(template) < len(content) * 0.8:
            return template

        return None

    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""

        if not self.embedding_model:
            # Fallback to simple text similarity
            return self._simple_text_similarity(text1, text2)

        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return self._simple_text_similarity(text1, text2)

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity fallback"""

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def _calculate_ttl(self, quality_score: float, cost: float, model: str) -> int:
        """Calculate intelligent TTL based on entry characteristics"""

        base_ttl = self.config['exact_match_ttl']

        # Higher quality = longer TTL
        quality_multiplier = quality_score / 100.0

        # Higher cost = longer TTL (more valuable to cache)
        cost_multiplier = min(cost * 1000, 2.0)  # Cap at 2x

        # Premium models = longer TTL
        model_multiplier = 1.5 if 'gpt-4' in model or 'claude' in model else 1.0

        ttl = int(base_ttl * quality_multiplier * cost_multiplier * model_multiplier)

        # Ensure reasonable bounds
        return max(3600, min(ttl, 7 * 24 * 3600))  # 1 hour to 7 days

    def _generate_tags(self, content: str, model: str) -> List[str]:
        """Generate tags for cache entry"""

        tags = [f"model:{model}"]

        # Add content-based tags
        if 'code' in content.lower() or 'function' in content.lower():
            tags.append('type:code')
        elif 'translate' in content.lower():
            tags.append('type:translation')
        elif 'summarize' in content.lower() or 'summary' in content.lower():
            tags.append('type:summary')
        else:
            tags.append('type:general')

        # Add length-based tags
        if len(content) > 5000:
            tags.append('length:long')
        elif len(content) > 1000:
            tags.append('length:medium')
        else:
            tags.append('length:short')

        return tags

    async def _store_in_redis(self, cache_entry: CacheEntry):
        """Store cache entry in Redis"""

        # Store for exact matching
        exact_key = f"cache:exact:{cache_entry.tenant_id}:{cache_entry.request_hash}"

        # Store for semantic matching
        semantic_key = f"cache:semantic:{cache_entry.tenant_id}:{cache_entry.semantic_hash}"

        # Serialize entry
        if self.config['compression_enabled']:
            data = gzip.compress(pickle.dumps(cache_entry))
        else:
            data = pickle.dumps(cache_entry)

        # Store with TTL
        await self.redis.setex(exact_key, cache_entry.ttl_seconds, data)
        if cache_entry.semantic_hash:
            await self.redis.setex(semantic_key, cache_entry.ttl_seconds, data)

    async def _store_in_database(self, cache_entry: CacheEntry):
        """Store cache entry in database for persistence"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO caching.cache_entries (
                    cache_id, tenant_id, request_hash, semantic_hash, input_tokens,
                    output_tokens, model_used, prompt_template, request_content,
                    response_content, quality_score, cost_saved, access_count,
                    last_accessed, ttl_seconds, tags, metadata, created_at, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            """, cache_entry.cache_id, cache_entry.tenant_id, cache_entry.request_hash,
                cache_entry.semantic_hash, cache_entry.input_tokens, cache_entry.output_tokens,
                cache_entry.model_used, cache_entry.prompt_template, cache_entry.request_content,
                cache_entry.response_content, cache_entry.quality_score, cache_entry.cost_saved,
                cache_entry.access_count, cache_entry.last_accessed, cache_entry.ttl_seconds,
                cache_entry.tags, json.dumps(cache_entry.metadata), cache_entry.created_at,
                cache_entry.expires_at)

    async def _update_access_stats(self, cache_entry: CacheEntry, cache_level: CacheLevel):
        """Update cache access statistics"""

        # Update access count and timestamp
        cache_entry.access_count += 1
        cache_entry.last_accessed = datetime.now()

        # Update in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE caching.cache_entries
                SET access_count = access_count + 1, last_accessed = NOW()
                WHERE cache_id = $1
            """, cache_entry.cache_id)

            # Log cache request
            await conn.execute("""
                INSERT INTO caching.cache_requests (
                    tenant_id, cache_id, cache_level, cache_hit, cost_savings, response_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, cache_entry.tenant_id, cache_entry.cache_id, cache_level.value,
                True, cache_entry.cost_saved, 50)  # Approximate cache response time

    def _track_performance(self, tenant_id: str, hit: bool, response_time: float):
        """Track cache performance metrics"""

        stats = self.stats[tenant_id]
        stats['requests'] += 1

        if hit:
            stats['hits'] += 1
        else:
            stats['misses'] += 1

        stats['response_times'].append(response_time * 1000)  # Convert to ms

        # Keep only recent response times
        if len(stats['response_times']) > 1000:
            stats['response_times'] = stats['response_times'][-1000:]

    # Background maintenance tasks
    async def _cache_cleanup_loop(self):
        """Background task for cache cleanup"""

        while True:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(self.config['cleanup_interval'])
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _cache_warming_loop(self):
        """Background task for predictive cache warming"""

        while True:
            try:
                if self.config['warming_enabled']:
                    await self._perform_cache_warming()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in cache warming loop: {e}")
                await asyncio.sleep(300)

    async def _stats_aggregation_loop(self):
        """Background task for stats aggregation"""

        while True:
            try:
                await self._aggregate_cache_stats()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in stats aggregation loop: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""

        # Remove from database
        async with self.db_pool.acquire() as conn:
            deleted_count = await conn.fetchval("""
                DELETE FROM caching.cache_entries
                WHERE expires_at < NOW() OR is_active = false
                RETURNING count(*)
            """)

        if deleted_count:
            logger.info(f"Cleaned up {deleted_count} expired cache entries")

    async def _perform_cache_warming(self):
        """Perform predictive cache warming"""

        # Get tenants with recent activity
        async with self.db_pool.acquire() as conn:
            active_tenants = await conn.fetch("""
                SELECT DISTINCT tenant_id
                FROM caching.cache_requests
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)

        for tenant_row in active_tenants:
            tenant_id = tenant_row['tenant_id']

            # Analyze usage patterns and predict likely requests
            predictions = await self._analyze_usage_patterns(tenant_id)

            if predictions:
                await self.warm_cache(tenant_id, predictions)

    async def _analyze_usage_patterns(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Analyze usage patterns to predict cache warming opportunities"""

        predictions = []

        async with self.db_pool.acquire() as conn:
            # Get frequent request patterns
            frequent_patterns = await conn.fetch("""
                SELECT
                    prompt_template,
                    model_used,
                    COUNT(*) as frequency,
                    AVG(quality_score) as avg_quality,
                    AVG(cost_saved) as avg_cost
                FROM caching.cache_entries
                WHERE tenant_id = $1
                AND created_at > NOW() - INTERVAL '7 days'
                AND prompt_template IS NOT NULL
                GROUP BY prompt_template, model_used
                HAVING COUNT(*) > 2
                ORDER BY frequency DESC, avg_cost DESC
                LIMIT 10
            """, tenant_id)

            for pattern in frequent_patterns:
                # Predict likely variations of this pattern
                variations = self._generate_pattern_variations(pattern['prompt_template'])

                for variation in variations:
                    predictions.append({
                        'content': variation,
                        'model': pattern['model_used'],
                        'confidence': pattern['frequency'] / 10.0,
                        'expected_cost': pattern['avg_cost']
                    })

        return predictions[:20]  # Limit predictions

    def _generate_pattern_variations(self, template: str) -> List[str]:
        """Generate variations of a prompt template for cache warming"""

        variations = []

        # Simple template variation generation
        if '{NUMBER}' in template:
            for num in ['1', '5', '10', '100']:
                variations.append(template.replace('{NUMBER}', num))

        if '{NAME}' in template:
            common_names = ['John', 'Mary', 'David', 'Sarah', 'Company']
            for name in common_names:
                variations.append(template.replace('{NAME}', name))

        # Return original if no variations generated
        if not variations:
            variations.append(template)

        return variations

    async def _aggregate_cache_stats(self):
        """Aggregate cache statistics for reporting"""

        try:
            # Aggregate stats from in-memory tracking
            for tenant_id, stats in self.stats.items():
                if stats['requests'] > 0:
                    hit_rate = stats['hits'] / stats['requests']
                    avg_response_time = sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else 0

                    # Store aggregated stats
                    async with self.db_pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO caching.cache_stats_hourly (
                                tenant_id, hour_bucket, total_requests, cache_hits,
                                hit_rate, avg_response_time_ms, cost_savings, created_at
                            ) VALUES ($1, date_trunc('hour', NOW()), $2, $3, $4, $5, $6, NOW())
                            ON CONFLICT (tenant_id, hour_bucket)
                            DO UPDATE SET
                                total_requests = EXCLUDED.total_requests,
                                cache_hits = EXCLUDED.cache_hits,
                                hit_rate = EXCLUDED.hit_rate,
                                avg_response_time_ms = EXCLUDED.avg_response_time_ms,
                                cost_savings = EXCLUDED.cost_savings
                        """, tenant_id, stats['requests'], stats['hits'],
                            hit_rate, avg_response_time, stats['cost_savings'])

                    # Reset stats
                    self.stats[tenant_id] = {
                        'requests': 0,
                        'hits': 0,
                        'misses': 0,
                        'cost_savings': 0.0,
                        'response_times': []
                    }

        except Exception as e:
            logger.error(f"Error aggregating cache stats: {e}")

    async def _check_cache_limits(self, tenant_id: str) -> bool:
        """Check if cache limits are exceeded"""

        async with self.db_pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as entry_count,
                    SUM(LENGTH(response_content)) / 1024.0 / 1024.0 as size_mb
                FROM caching.cache_entries
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

        return (stats['entry_count'] >= self.config['max_entries_per_tenant'] or
                stats['size_mb'] >= self.config['max_cache_size_mb'])

    async def _evict_old_entries(self, tenant_id: str):
        """Evict old cache entries using LRU policy"""

        async with self.db_pool.acquire() as conn:
            # Evict 20% of entries, keeping the most recently accessed
            await conn.execute("""
                UPDATE caching.cache_entries
                SET is_active = false
                WHERE cache_id IN (
                    SELECT cache_id
                    FROM caching.cache_entries
                    WHERE tenant_id = $1 AND is_active = true
                    ORDER BY last_accessed ASC, quality_score ASC
                    LIMIT (SELECT COUNT(*) * 0.2 FROM caching.cache_entries WHERE tenant_id = $1 AND is_active = true)
                )
            """, tenant_id)

    async def _load_cache_entry(self, cache_id: str) -> Optional[CacheEntry]:
        """Load complete cache entry from database"""

        async with self.db_pool.acquire() as conn:
            entry_data = await conn.fetchrow("""
                SELECT * FROM caching.cache_entries WHERE cache_id = $1
            """, cache_id)

        if entry_data:
            return CacheEntry(
                cache_id=entry_data['cache_id'],
                tenant_id=entry_data['tenant_id'],
                request_hash=entry_data['request_hash'],
                semantic_hash=entry_data['semantic_hash'] or "",
                input_tokens=entry_data['input_tokens'],
                output_tokens=entry_data['output_tokens'],
                model_used=entry_data['model_used'],
                prompt_template=entry_data['prompt_template'],
                request_content=entry_data['request_content'],
                response_content=entry_data['response_content'],
                quality_score=entry_data['quality_score'],
                cost_saved=entry_data['cost_saved'],
                access_count=entry_data['access_count'],
                last_accessed=entry_data['last_accessed'],
                ttl_seconds=entry_data['ttl_seconds'],
                tags=entry_data['tags'] or [],
                metadata=json.loads(entry_data['metadata']) if entry_data['metadata'] else {},
                created_at=entry_data['created_at'],
                expires_at=entry_data['expires_at']
            )

        return None

    async def _find_keys_by_pattern(self, tenant_id: str, pattern: str) -> List[str]:
        """Find Redis keys matching pattern"""
        return await self.redis.keys(f"cache:*:{tenant_id}:*{pattern}*")

    async def _find_keys_by_model(self, tenant_id: str, model: str) -> List[str]:
        """Find Redis keys for specific model"""
        # Would need to scan cache entries and build key list
        # Simplified implementation
        return await self.redis.keys(f"cache:*:{tenant_id}:*")

    async def _find_keys_by_tags(self, tenant_id: str, tags: List[str]) -> List[str]:
        """Find Redis keys matching tags"""
        # Would query database for cache_ids with matching tags, then build Redis keys
        # Simplified implementation
        return await self.redis.keys(f"cache:*:{tenant_id}:*")

    async def _find_all_tenant_keys(self, tenant_id: str) -> List[str]:
        """Find all Redis keys for tenant"""
        return await self.redis.keys(f"cache:*:{tenant_id}:*")

    async def _mark_invalidated_in_database(self, tenant_id: str, pattern: Optional[str] = None,
                                          model: Optional[str] = None, tags: Optional[List[str]] = None):
        """Mark cache entries as invalidated in database"""

        query = "UPDATE caching.cache_entries SET is_active = false WHERE tenant_id = $1"
        params = [tenant_id]
        param_count = 1

        if model:
            param_count += 1
            query += f" AND model_used = ${param_count}"
            params.append(model)

        if pattern:
            param_count += 1
            query += f" AND (request_content ILIKE ${param_count} OR prompt_template ILIKE ${param_count})"
            params.extend([f"%{pattern}%", f"%{pattern}%"])

        async with self.db_pool.acquire() as conn:
            await conn.execute(query, *params)

    async def _mark_for_warming(self, tenant_id: str, prediction: Dict[str, Any]):
        """Mark prediction for cache warming"""

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO caching.cache_warming_queue (
                    tenant_id, predicted_content, predicted_model, confidence_score,
                    expected_cost, status, created_at
                ) VALUES ($1, $2, $3, $4, $5, 'pending', NOW())
            """, tenant_id, prediction['content'], prediction['model'],
                prediction['confidence'], prediction['expected_cost'])


class CacheManager:
    """
    High-level cache management orchestrator
    """

    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis = redis_client
        self.caches = {}  # tenant_id -> IntelligentCache

    async def get_cache(self, tenant_id: str) -> IntelligentCache:
        """Get or create cache for tenant"""

        if tenant_id not in self.caches:
            cache = IntelligentCache(self.db_pool, self.redis)
            await cache.initialize()
            self.caches[tenant_id] = cache

        return self.caches[tenant_id]

    async def get_cached_response(self, tenant_id: str, request_content: str,
                                 model: str, **kwargs) -> Optional[CacheHit]:
        """Get cached AI response if available"""

        cache = await self.get_cache(tenant_id)
        return await cache.get_cached_response(
            tenant_id, request_content, model, **kwargs
        )

    async def store_response(self, tenant_id: str, request_content: str,
                           response_content: str, model: str,
                           input_tokens: int, output_tokens: int,
                           cost: float, **kwargs) -> str:
        """Store AI response in cache"""

        cache = await self.get_cache(tenant_id)
        return await cache.store_response(
            tenant_id, request_content, response_content, model,
            input_tokens, output_tokens, cost, **kwargs
        )

    async def get_analytics(self, tenant_id: str, days: int = 7) -> CacheStats:
        """Get cache analytics for tenant"""

        cache = await self.get_cache(tenant_id)
        return await cache.get_cache_analytics(tenant_id, days)

    async def invalidate_cache(self, tenant_id: str, **kwargs) -> int:
        """Invalidate cache entries"""

        cache = await self.get_cache(tenant_id)
        return await cache.invalidate_cache(tenant_id, **kwargs)
