import sys
from unittest.mock import MagicMock

# Mock dotenv before importing AgentSystem
dotenv_mock = MagicMock()
sys.modules['dotenv'] = dotenv_mock

import unittest
import tempfile
import os
import json
import shutil
import time

from AgentSystem.core.state import AgentState, AgentStatus, TaskInfo

class TestAgentState(unittest.TestCase):
    def setUp(self):
        self.state = AgentState()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_agent_status_enum(self):
        """Test AgentStatus enum values"""
        self.assertEqual(AgentStatus.IDLE.name, "IDLE")
        self.assertEqual(AgentStatus.EXECUTING.name, "EXECUTING")
        self.assertEqual(AgentStatus.ERROR.name, "ERROR")

    def test_task_info_creation(self):
        """Test TaskInfo creation"""
        task = TaskInfo(name="Test Task", description="Test Description", priority=2)
        self.assertEqual(task.name, "Test Task")
        self.assertEqual(task.description, "Test Description")
        self.assertEqual(task.priority, 2)
        self.assertEqual(task.status, "pending")
        self.assertIsNotNone(task.id)
        self.assertIsNotNone(task.created_at)

    def test_task_info_to_dict(self):
        """Test TaskInfo to dictionary conversion"""
        task = TaskInfo(name="Test Task")
        data = task.to_dict()
        self.assertEqual(data["name"], "Test Task")
        self.assertEqual(data["status"], "pending")
        self.assertEqual(data["id"], task.id)

    def test_agent_state_init(self):
        """Test AgentState initialization"""
        self.assertEqual(self.state.status, AgentStatus.IDLE)
        self.assertIsNone(self.state.current_task_id)
        self.assertEqual(self.state.tasks, {})
        self.assertEqual(self.state.task_queue, [])
        self.assertEqual(self.state.context, {})

    def test_set_status(self):
        """Test setting agent status"""
        original_time = self.state.last_updated
        time.sleep(0.01)  # Ensure time advances
        self.state.set_status(AgentStatus.EXECUTING)
        self.assertEqual(self.state.status, AgentStatus.EXECUTING)
        self.assertNotEqual(self.state.last_updated, original_time)

    def test_create_task(self):
        """Test creating a task"""
        task_id = self.state.create_task(name="New Task", priority=5)
        self.assertIn(task_id, self.state.tasks)
        self.assertIn(task_id, self.state.task_queue)
        task = self.state.get_task(task_id)
        self.assertEqual(task.name, "New Task")
        self.assertEqual(task.priority, 5)

    def test_get_task(self):
        """Test getting a task"""
        task_id = self.state.create_task(name="Task 1")
        task = self.state.get_task(task_id)
        self.assertIsNotNone(task)
        self.assertEqual(task.id, task_id)

        # Test non-existent task
        self.assertIsNone(self.state.get_task("non_existent"))

    def test_update_task(self):
        """Test updating a task"""
        task_id = self.state.create_task(name="Task 1")

        # Successful update
        result = self.state.update_task(task_id, status="running", priority=10)
        self.assertTrue(result)
        task = self.state.get_task(task_id)
        self.assertEqual(task.status, "running")
        self.assertEqual(task.priority, 10)

        # Failed update (non-existent task)
        result = self.state.update_task("non_existent", status="running")
        self.assertFalse(result)

    def test_start_task(self):
        """Test starting a task"""
        task_id = self.state.create_task(name="Task 1")

        # Successful start
        result = self.state.start_task(task_id)
        self.assertTrue(result)
        task = self.state.get_task(task_id)
        self.assertEqual(task.status, "running")
        self.assertIsNotNone(task.started_at)
        self.assertEqual(self.state.current_task_id, task_id)

        # Failed start (non-existent task)
        result = self.state.start_task("non_existent")
        self.assertFalse(result)

    def test_complete_task(self):
        """Test completing a task"""
        task_id = self.state.create_task(name="Task 1")
        self.state.start_task(task_id)

        # Successful completion
        result = self.state.complete_task(task_id, success=True)
        self.assertTrue(result)
        task = self.state.get_task(task_id)
        self.assertEqual(task.status, "success")
        self.assertIsNotNone(task.completed_at)
        self.assertIsNone(self.state.current_task_id)
        self.assertNotIn(task_id, self.state.task_queue)

        # Failed completion (non-existent task)
        result = self.state.complete_task("non_existent")
        self.assertFalse(result)

    def test_get_next_task(self):
        """Test getting the next task based on priority"""
        # Empty queue
        self.assertIsNone(self.state.get_next_task())

        # Add tasks with different priorities
        id1 = self.state.create_task(name="Low Priority", priority=1)
        id2 = self.state.create_task(name="High Priority", priority=10)
        id3 = self.state.create_task(name="Medium Priority", priority=5)

        # Should get highest priority first
        self.assertEqual(self.state.get_next_task(), id2)

        # Remove highest and check next
        self.state.complete_task(id2)
        self.assertEqual(self.state.get_next_task(), id3)

        # Remove medium and check last
        self.state.complete_task(id3)
        self.assertEqual(self.state.get_next_task(), id1)

    def test_context_management(self):
        """Test context management"""
        self.state.set_context("user", "test_user")
        self.assertEqual(self.state.get_context("user"), "test_user")
        self.assertEqual(self.state.get_context("missing", "default"), "default")

    def test_save_load(self):
        """Test saving and loading state"""
        # Setup state
        self.state.set_status(AgentStatus.THINKING)
        self.state.set_context("key", "value")
        task_id = self.state.create_task(name="Persisted Task")

        # Save
        file_path = os.path.join(self.test_dir, "state.json")
        self.state.save(file_path)

        # Verify file exists
        self.assertTrue(os.path.exists(file_path))

        # Load
        new_state = AgentState.load(file_path)

        # Verify loaded state
        self.assertEqual(new_state.status, AgentStatus.THINKING)
        self.assertEqual(new_state.get_context("key"), "value")
        self.assertIn(task_id, new_state.tasks)
        self.assertEqual(new_state.tasks[task_id].name, "Persisted Task")

    def test_load_error(self):
        """Test loading from non-existent or invalid file"""
        # Non-existent file
        state = AgentState.load("non_existent_file.json")
        # Should return a default state instead of crashing
        self.assertIsInstance(state, AgentState)
        self.assertEqual(state.status, AgentStatus.IDLE)

if __name__ == '__main__':
    unittest.main()
