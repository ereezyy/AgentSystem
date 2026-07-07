"""
Memory System Tests
-------------------
Unit tests for the memory system components.
"""

import unittest
import tempfile
import shutil
import os
import json
import uuid
import time
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to sys.path
sys.path.append(os.getcwd())

# Mock dotenv if not available
try:
    import dotenv
except ImportError:
    sys.modules['dotenv'] = MagicMock()

# Now import AgentSystem modules
from AgentSystem.core.memory import Memory, MemoryItem

class TestMemoryItem(unittest.TestCase):
    def test_initialization(self):
        """Test MemoryItem initialization"""
        # Test default initialization
        item = MemoryItem(content="test content")
        self.assertIsNotNone(item.id)
        self.assertEqual(item.content, "test content")
        self.assertEqual(item.type, "general")
        self.assertIsInstance(item.created_at, float)
        self.assertEqual(item.importance, 0.5)
        self.assertEqual(item.metadata, {})
        self.assertIsNone(item.embedding)

        # Test custom initialization
        custom_id = str(uuid.uuid4())
        custom_time = time.time()
        embedding = [0.1, 0.2, 0.3]
        metadata = {"source": "test"}

        item = MemoryItem(
            id=custom_id,
            content={"key": "value"},
            type="episodic",
            created_at=custom_time,
            importance=0.8,
            metadata=metadata,
            embedding=embedding
        )

        self.assertEqual(item.id, custom_id)
        self.assertEqual(item.content, {"key": "value"})
        self.assertEqual(item.type, "episodic")
        self.assertEqual(item.created_at, custom_time)
        self.assertEqual(item.importance, 0.8)
        self.assertEqual(item.metadata, metadata)
        self.assertEqual(item.embedding, embedding)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        item = MemoryItem(content="test")
        data = item.to_dict()

        self.assertEqual(data["id"], item.id)
        self.assertEqual(data["content"], "test")
        self.assertEqual(data["type"], "general")
        self.assertEqual(data["importance"], 0.5)

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "id": str(uuid.uuid4()),
            "content": "test",
            "type": "episodic",
            "created_at": time.time(),
            "importance": 0.9,
            "metadata": {"tag": "important"},
            "embedding": [0.1, 0.2]
        }

        item = MemoryItem.from_dict(data)

        self.assertEqual(item.id, data["id"])
        self.assertEqual(item.content, "test")
        self.assertEqual(item.type, "episodic")
        self.assertEqual(item.created_at, data["created_at"])
        self.assertEqual(item.importance, 0.9)
        self.assertEqual(item.metadata, {"tag": "important"})
        self.assertEqual(item.embedding, [0.1, 0.2])


class TestMemory(unittest.TestCase):
    def setUp(self):
        """Set up test memory system"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = Memory(storage_path=self.temp_dir, max_working_items=5)

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test memory system initialization"""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "memory.db")))
        self.assertEqual(self.memory.max_working_items, 5)
        self.assertEqual(len(self.memory.working_memory), 0)

    def test_add_memory(self):
        """Test adding memories"""
        # Add to working memory (low importance)
        id1 = self.memory.add("test1", importance=0.4)
        self.assertEqual(len(self.memory.working_memory), 1)
        self.assertEqual(self.memory.working_memory[0].id, id1)

        # Add to long-term memory (high importance)
        id2 = self.memory.add("test2", importance=0.8)

        # Working memory should have 2 items
        self.assertEqual(len(self.memory.working_memory), 2)

        # Verify persistence manually to ensure it's in DB
        import sqlite3
        conn = sqlite3.connect(os.path.join(self.temp_dir, "memory.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM memories")
        count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(count, 1) # Only the important one should be in DB

    def test_get_memory(self):
        """Test retrieving memories"""
        id1 = self.memory.add("working memory item", importance=0.4)
        id2 = self.memory.add("long-term memory item", importance=0.9)

        # Get from working memory
        item1 = self.memory.get(id1)
        self.assertIsNotNone(item1)
        self.assertEqual(item1.content, "working memory item")

        # Get from long-term memory
        # Clear working memory first to force DB fetch
        self.memory.working_memory = []
        item2 = self.memory.get(id2)
        self.assertIsNotNone(item2)
        self.assertEqual(item2.content, "long-term memory item")

        # Get non-existent
        self.assertIsNone(self.memory.get("non-existent"))

    def test_query_memory(self):
        """Test querying memories"""
        self.memory.add("python code", memory_type="code", importance=0.8)
        self.memory.add("javascript code", memory_type="code", importance=0.6)
        self.memory.add("random thought", memory_type="thought", importance=0.2)

        # Query by type
        results = self.memory.query(memory_type="code")
        self.assertEqual(len(results), 2)

        # Query by importance
        results = self.memory.query(min_importance=0.7)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "python code")

        # Search content
        results = self.memory.query(content_search="javascript")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "javascript code")

    def test_update_memory(self):
        """Test updating memories"""
        id1 = self.memory.add("original content", importance=0.8)

        # Update content
        success = self.memory.update(id1, content="updated content")
        self.assertTrue(success)

        # Check if updated in working memory (Memory.update updates working memory first)
        item = self.memory.get(id1)
        self.assertEqual(item.content, "updated content")

        # Check if updated in DB
        # Memory.update updates DB if item is found in DB?
        # Code: "Then try to update in long-term memory... if not cursor.fetchone(): return False"
        # Since importance was 0.8, it was in DB.

        # Let's verify DB content directly
        import sqlite3
        conn = sqlite3.connect(os.path.join(self.temp_dir, "memory.db"))
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM memories WHERE id = ?", (id1,))
        content = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(content, "updated content")

        # Update metadata
        self.memory.update(id1, metadata={"updated": True})
        item = self.memory.get(id1)
        self.assertEqual(item.metadata, {"updated": True})

    def test_forget_memory(self):
        """Test forgetting memories"""
        id1 = self.memory.add("forget me", importance=0.8)

        success = self.memory.forget(id1)
        self.assertTrue(success)

        self.assertIsNone(self.memory.get(id1))

        # Try to forget non-existent
        success = self.memory.forget("non-existent")
        self.assertFalse(success)

    def test_working_memory_limit(self):
        """Test working memory limits"""
        # Add more items than max_working_items (5)
        for i in range(10):
            # Vary importance to test sorting
            importance = 0.1 * i
            self.memory.add(f"item {i}", importance=importance)

        self.assertLessEqual(len(self.memory.working_memory), 5)

        # The most important items should remain (indices 5, 6, 7, 8, 9 have importance >= 0.5)
        # Sort key is (importance, created_at) descending.
        # So item 9 (0.9), item 8 (0.8), ... item 5 (0.5) should be in working memory.
        importances = [item.importance for item in self.memory.working_memory]
        self.assertTrue(all(imp >= 0.5 for imp in importances))

    def test_clear_working_memory(self):
        """Test clearing working memory"""
        self.memory.add("test", importance=0.5)
        self.assertGreater(len(self.memory.working_memory), 0)

        self.memory.clear_working_memory()
        self.assertEqual(len(self.memory.working_memory), 0)

    def test_summarize_memories(self):
        """Test memory summarization"""
        self.memory.add("Memory 1", importance=0.8)
        self.memory.add("Memory 2", importance=0.9)

        summary = self.memory.summarize_memories()

        self.assertIn("Memory Summary", summary)
        self.assertIn("Memory 1", summary)
        self.assertIn("Memory 2", summary)

    def test_complex_content(self):
        """Test storing complex content types"""
        complex_data = {"key": "value", "list": [1, 2, 3]}
        id1 = self.memory.add(complex_data, importance=0.8)

        # Retrieve from DB (clear working memory first)
        self.memory.clear_working_memory()
        item = self.memory.get(id1)

        self.assertEqual(item.content, complex_data)

if __name__ == '__main__':
    unittest.main()
