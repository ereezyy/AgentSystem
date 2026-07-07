import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

# Mock missing dependencies
sys.modules['aioredis'] = MagicMock()
sys.modules['asyncpg'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['dotenv'] = MagicMock()

# Now we can import the module
from AgentSystem.caching.intelligent_cache import IntelligentCache

class AsyncIter:
    def __init__(self, items):
        self.items = items
        self.idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.idx >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.idx]
        self.idx += 1
        return item

class TestIntelligentCachePerformance(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_db_pool = AsyncMock()
        self.mock_redis = MagicMock() # Sync mock to hold async methods
        self.mock_redis.get = AsyncMock()
        self.mock_redis.keys = AsyncMock()

        # Setup default scan_iter behavior
        # It needs to return an async iterator
        self.mock_keys = [f"key:{i}".encode() for i in range(10)]
        self.mock_redis.scan_iter = MagicMock(return_value=AsyncIter(self.mock_keys))
        # Ensure that if it's called multiple times, it returns fresh iterators
        self.mock_redis.scan_iter.side_effect = lambda *args, **kwargs: AsyncIter(self.mock_keys)

        self.cache = IntelligentCache(self.mock_db_pool, self.mock_redis)

        # Mock embedding_model on the instance to avoid calls
        self.cache.embedding_model = MagicMock()
        self.cache.embedding_model.encode.return_value = MagicMock()
        self.cache.embedding_model.encode.return_value.tobytes.return_value = b'hash'
        self.cache._calculate_similarity = AsyncMock(return_value=0.9)

    async def test_check_semantic_match_uses_scan_iter(self):
        # Setup
        with unittest.mock.patch('AgentSystem.caching.intelligent_cache.pickle') as mock_pickle:
            mock_entry = MagicMock()
            mock_entry.request_content = "content"
            mock_entry.cost_saved = 0.1
            mock_pickle.loads.return_value = mock_entry

            # Redis get should return bytes
            self.mock_redis.get.return_value = b'data'

            await self.cache._check_semantic_match("tenant1", "hash", "content")

        self.mock_redis.scan_iter.assert_called()
        self.mock_redis.keys.assert_not_called()

        # Verify match argument
        call_args = self.mock_redis.scan_iter.call_args
        self.assertIn("match", call_args.kwargs)
        self.assertIn("count", call_args.kwargs)
        self.assertEqual(call_args.kwargs["count"], 100)

    async def test_find_keys_by_pattern_uses_scan_iter(self):
        keys = await self.cache._find_keys_by_pattern("tenant1", "pattern")
        self.assertEqual(len(keys), 10)
        self.mock_redis.scan_iter.assert_called()
        self.mock_redis.keys.assert_not_called()

        call_args = self.mock_redis.scan_iter.call_args
        self.assertIn("match", call_args.kwargs)
        self.assertIn("pattern", call_args.kwargs["match"])

    async def test_find_keys_by_model_uses_scan_iter(self):
        keys = await self.cache._find_keys_by_model("tenant1", "gpt-4")
        self.assertEqual(len(keys), 10)
        self.mock_redis.scan_iter.assert_called()
        self.mock_redis.keys.assert_not_called()

    async def test_find_keys_by_tags_uses_scan_iter(self):
        keys = await self.cache._find_keys_by_tags("tenant1", ["tag"])
        self.assertEqual(len(keys), 10)
        self.mock_redis.scan_iter.assert_called()
        self.mock_redis.keys.assert_not_called()

    async def test_find_all_tenant_keys_uses_scan_iter(self):
        keys = await self.cache._find_all_tenant_keys("tenant1")
        self.assertEqual(len(keys), 10)
        self.mock_redis.scan_iter.assert_called()
        self.mock_redis.keys.assert_not_called()

if __name__ == '__main__':
    unittest.main()
