import unittest
import shutil
import os
import sqlite3
import threading
from unittest.mock import MagicMock, patch
import sys
import json

# Mock necessary modules
sys.modules["dotenv"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["aioredis"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["prometheus_client"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["AgentSystem.database.connection"] = MagicMock()

# Now import the class under test
from AgentSystem.core.memory import Memory

class TestMemoryPerformance(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./data/test_memory_perf"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        self.memory = Memory(storage_path=self.test_dir)

    def tearDown(self):
        self.memory.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization_creates_connection(self):
        """Test that initialization creates a persistent connection."""
        self.assertIsNotNone(self.memory._conn)
        self.assertIsInstance(self.memory._conn, sqlite3.Connection)

        # Verify table creation
        with self.memory._db_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'")
            self.assertIsNotNone(cursor.fetchone())

    def test_add_and_get(self):
        """Test adding and retrieving items."""
        mid = self.memory.add("test_content", importance=0.8, memory_type="persistent")

        # Should be in working memory
        self.assertEqual(len(self.memory.working_memory), 1)

        # Should be in DB (importance > 0.7)
        with self.memory._db_cursor() as cursor:
            cursor.execute("SELECT content FROM memories WHERE id=?", (mid,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "test_content")

        # Get should work
        item = self.memory.get(mid)
        self.assertIsNotNone(item)
        self.assertEqual(item.content, "test_content")

    def test_update(self):
        """Test updating items."""
        mid = self.memory.add("content", importance=0.8)
        success = self.memory.update(mid, content="updated", importance=0.9)
        self.assertTrue(success)

        item = self.memory.get(mid)
        self.assertEqual(item.content, "updated")

        # Verify DB update
        with self.memory._db_cursor() as cursor:
            cursor.execute("SELECT content FROM memories WHERE id=?", (mid,))
            row = cursor.fetchone()
            self.assertEqual(row[0], "updated")

    def test_forget(self):
        """Test removing items."""
        mid = self.memory.add("content", importance=0.8)
        removed = self.memory.forget(mid)
        self.assertTrue(removed)

        item = self.memory.get(mid)
        self.assertIsNone(item)

        with self.memory._db_cursor() as cursor:
            cursor.execute("SELECT * FROM memories WHERE id=?", (mid,))
            self.assertIsNone(cursor.fetchone())

    def test_thread_safety(self):
        """Test concurrent access."""

        def worker():
            for i in range(10):
                self.memory.add(f"content_{threading.get_ident()}_{i}", importance=0.8)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check count
        with self.memory._db_cursor() as cursor:
            cursor.execute("SELECT count(*) FROM memories")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 50)

    def test_close(self):
        """Test resource cleanup."""
        conn = self.memory._conn
        self.memory.close()
        self.assertIsNone(self.memory._conn)

        # Check that connection is actually closed
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

if __name__ == "__main__":
    unittest.main()
