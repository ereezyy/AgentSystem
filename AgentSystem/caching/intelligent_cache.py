"""
Intelligent Caching Engine
Multi-level AI response caching with exact + semantic matching
"""

import asyncio
import json
import logging
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field

import asyncpg
import aioredis
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import gzip

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    EXACT    = "exact"
    SEMANTIC = "semantic"
    # PARTIAL and TEMPLATE levels removed for simplicity / realism in v1


@dataclass
class CacheEntry:
    cache_id: str
    tenant_id: str
    request_hash: str               # exact match
    embedding: Optional[np.ndarray] = None   # for pgvector
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    request_content: str = ""
    response_content: str = ""
    quality_score: float = 80.0
    cost_saved: float = 0.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=7))
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheHit:
    entry: CacheEntry
    similarity: float
    level: CacheLevel
    adaptation_needed: bool = False
    estimated_savings: float = 0.0


class IntelligentCache:
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        redis: aioredis.Redis,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        default_ttl_days: int = 7,
        max_size_mb_per_tenant: float = 800.0,
    ):
        self.db_pool = db_pool
        self.redis = redis
        self.embedding_model_name = embedding_model_name
        self.embedding_model: Optional[SentenceTransformer] = None
        self.model_load_event = asyncio.Event()
        self.model_load_error: Optional[Exception] = None

        self.default_ttl_days = default_ttl_days
        self.max_size_mb_per_tenant = max_size_mb_per_tenant

        # Start background model loading
        asyncio.create_task(self._load_embedding_model_background())

    async def _load_embedding_model_background(self):
        try:
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.embedding_model_name)
            )
            self.embedding_model = model
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            self.model_load_error = e
            logger.error("Failed to load embedding model", exc_info=True)
        finally:
            self.model_load_event.set()

    async def _ensure_model_loaded(self):
        if self.embedding_model is not None:
            return
        if self.model_load_error:
            raise RuntimeError("Embedding model failed to load") from self.model_load_error
        await self.model_load_event.wait()
        if self.embedding_model is None:
            raise RuntimeError("Embedding model load failed")

    # ────────────────────────────────────────────────
    #   Hashing
    # ────────────────────────────────────────────────

    def _exact_hash(self, content: str, model: str, temperature: float, max_tokens: int) -> str:
        key = f"{content}|{model}|{temperature:.2f}|{max_tokens}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _embedding_or_none(self, text: str) -> Optional[np.ndarray]:
        if not self.embedding_model:
            return None
        try:
            return self.embedding_model.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    # ────────────────────────────────────────────────
    #   Main public API
    # ────────────────────────────────────────────────

    async def get(
        self,
        tenant_id: str,
        request_content: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Optional[CacheHit]:
        start = time.perf_counter()

        exact_hash = self._exact_hash(request_content, model, temperature, max_tokens)

        # 1. Fast path: exact match in Redis
        hit = await self._lookup_exact(tenant_id, exact_hash)
        if hit:
            hit.estimated_savings = hit.entry.cost_saved
            self._track(tenant_id, hit=True, latency_ms=(time.perf_counter() - start) * 1000)
            return hit

        # 2. Semantic match (if model loaded)
        if self.embedding_model:
            emb = self._embedding_or_none(request_content)
            if emb is not None:
                hit = await self._lookup_semantic(tenant_id, emb, request_content)
                if hit and hit.similarity >= 0.84:
                    self._track(tenant_id, hit=True, latency_ms=(time.perf_counter() - start) * 1000)
                    return hit

        self._track(tenant_id, hit=False, latency_ms=(time.perf_counter() - start) * 1000)
        return None

    async def put(
        self,
        tenant_id: str,
        request_content: str,
        response_content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        quality_score: float = 85.0,
        metadata: Optional[Dict] = None,
    ) -> str:
        exact_hash = self._exact_hash(request_content, model, temperature, max_tokens)
        embedding = self._embedding_or_none(request_content)

        entry = CacheEntry(
            cache_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            request_hash=exact_hash,
            embedding=embedding,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            request_content=request_content,
            response_content=response_content,
            quality_score=quality_score,
            cost_saved=cost_usd,
            access_count=1,
            metadata=metadata or {},
        )

        # Enforce limits → evict before insert if needed
        if await self._over_limit(tenant_id):
            await self._evict_lru(tenant_id, fraction=0.25)

        # Store
        await self._store_redis(entry)
        await self._store_postgres(entry)

        return entry.cache_id

    # ────────────────────────────────────────────────
    #   Storage layers
    # ────────────────────────────────────────────────

    async def _store_redis(self, entry: CacheEntry):
        key = f"cache:exact:{entry.tenant_id}:{entry.request_hash}"

        payload = pickle.dumps(entry)
        if len(payload) > 32_000:  # rough threshold
            payload = gzip.compress(payload)

        ttl = int(timedelta(days=self.default_ttl_days).total_seconds())
        await self.redis.setex(key, ttl, payload)

    async def _store_postgres(self, entry: CacheEntry):
        embedding_list = entry.embedding.tolist() if entry.embedding is not None else None

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO cache_entries (
                    cache_id, tenant_id, request_hash, embedding,
                    input_tokens, output_tokens, model, request_content,
                    response_content, quality_score, cost_saved,
                    access_count, last_accessed, created_at, expires_at,
                    is_active, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                          $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (cache_id) DO NOTHING
            """,
                entry.cache_id,
                entry.tenant_id,
                entry.request_hash,
                embedding_list,
                entry.input_tokens,
                entry.output_tokens,
                entry.model,
                entry.request_content,
                entry.response_content,
                entry.quality_score,
                entry.cost_saved,
                entry.access_count,
                entry.last_accessed,
                entry.created_at,
                entry.expires_at,
                entry.is_active,
                json.dumps(entry.metadata)
            )

    async def _lookup_exact(self, tenant_id: str, exact_hash: str) -> Optional[CacheHit]:
        key = f"cache:exact:{tenant_id}:{exact_hash}"
        data = await self.redis.get(key)
        if not data:
            return None

        try:
            if data.startswith(b'\x1f\x8b'):  # gzip magic
                data = gzip.decompress(data)
            entry: CacheEntry = pickle.loads(data)

            if entry.expires_at < datetime.now():
                await self.redis.delete(key)
                return None

            return CacheHit(
                entry=entry,
                similarity=1.0,
                level=CacheLevel.EXACT,
                adaptation_needed=False
            )
        except Exception as e:
            logger.warning(f"Corrupted redis cache entry {key}: {e}")
            return None

    async def _lookup_semantic(
        self,
        tenant_id: str,
        query_emb: np.ndarray,
        original_text: str
    ) -> Optional[CacheHit]:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    cache_id,
                    request_content,
                    response_content,
                    cost_saved,
                    quality_score,
                    1 - (embedding <=> $1::vector) AS cosine_sim
                FROM cache_entries
                WHERE tenant_id = $2
                  AND is_active = true
                  AND expires_at > NOW()
                ORDER BY embedding <=> $1::vector
                LIMIT 1
            """, query_emb.tolist(), tenant_id)

            if not row or row["cosine_sim"] < 0.84:
                return None

            full_entry = await self._load_full_entry(row["cache_id"])
            if not full_entry:
                return None

            return CacheHit(
                entry=full_entry,
                similarity=float(row["cosine_sim"]),
                level=CacheLevel.SEMANTIC,
                adaptation_needed=row["cosine_sim"] < 0.94,
                estimated_savings=full_entry.cost_saved * row["cosine_sim"]
            )

    async def _load_full_entry(self, cache_id: str) -> Optional[CacheEntry]:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM cache_entries WHERE cache_id = $1",
                cache_id
            )
        if not row:
            return None

        emb_array = np.array(row["embedding"]) if row["embedding"] else None

        return CacheEntry(
            cache_id=row["cache_id"],
            tenant_id=row["tenant_id"],
            request_hash=row["request_hash"],
            embedding=emb_array,
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            model=row["model"],
            request_content=row["request_content"],
            response_content=row["response_content"],
            quality_score=row["quality_score"],
            cost_saved=row["cost_saved"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            is_active=row["is_active"],
            metadata=json.loads(row["metadata"] or "{}")
        )

    # ────────────────────────────────────────────────
    #   Maintenance
    # ────────────────────────────────────────────────

    async def _over_limit(self, tenant_id: str) -> bool:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    COUNT(*) AS cnt,
                    SUM(LENGTH(response_content) + LENGTH(request_content)) / 1048576.0 AS mb
                FROM cache_entries
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

        if not row:
            return False
        return row["cnt"] > 15000 or row["mb"] > self.max_size_mb_per_tenant

    async def _evict_lru(self, tenant_id: str, fraction: float = 0.2):
        async with self.db_pool.acquire() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM cache_entries
                WHERE tenant_id = $1 AND is_active = true
            """, tenant_id)

            if count == 0:
                return

            to_remove = int(count * fraction) or 1

            await conn.execute("""
                UPDATE cache_entries
                SET is_active = false
                WHERE cache_id IN (
                    SELECT cache_id
                    FROM cache_entries
                    WHERE tenant_id = $1 AND is_active = true
                    ORDER BY last_accessed ASC
                    LIMIT $2
                )
            """, tenant_id, to_remove)

            logger.info(f"Evicted {to_remove} old entries for tenant {tenant_id}")

    def _track(self, tenant_id: str, hit: bool, latency_ms: float):
        # In real system → prometheus / statsd / timescaledb
        pass  # placeholder

    async def invalidate_all(self, tenant_id: str):
        await self.redis.delete(*(await self.redis.keys(f"cache:exact:{tenant_id}:*")))
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE cache_entries SET is_active = false WHERE tenant_id = $1",
                tenant_id
            )


# Factory / per-tenant manager
class CacheManager:
    def __init__(self, db_pool: asyncpg.Pool, redis: aioredis.Redis):
        self.db_pool = db_pool
        self.redis = redis
        self._caches: Dict[str, IntelligentCache] = {}

    async def get_cache(self, tenant_id: str) -> IntelligentCache:
        if tenant_id not in self._caches:
            self._caches[tenant_id] = IntelligentCache(self.db_pool, self.redis)
        return self._caches[tenant_id]

    async def get(self, tenant_id: str, *args, **kwargs) -> Optional[CacheHit]:
        cache = await self.get_cache(tenant_id)
        return await cache.get(tenant_id, *args, **kwargs)

    async def put(self, tenant_id: str, *args, **kwargs) -> str:
        cache = await self.get_cache(tenant_id)
        return await cache.put(tenant_id, *args, **kwargs)