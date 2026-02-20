import unittest
import os
import json
import time
import shutil
from pathlib import Path

# Add project root to path if needed, though usually test runner handles it.
import sys
sys.path.append(os.getcwd())

from AgentSystem.core.agent import Agent, AgentConfig
from AgentSystem.core.state import AgentState, AgentStatus
from AgentSystem.core.memory import Memory

class TestAgentStateLoading(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data_agent_state"
        os.makedirs(self.test_dir, exist_ok=True)
        self.state_file = os.path.join(self.test_dir, "agent_state.json")
        self.memory_storage = os.path.join(self.test_dir, "memory")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load_state(self):
        # 1. Create and configure an agent with state and memory
        config = AgentConfig(name="TestAgent", version="1.0.0")
        state_manager = AgentState()
        memory_manager = Memory(storage_path=self.memory_storage)

        agent = Agent(
            config=config,
            state_manager=state_manager,
            memory_manager=memory_manager
        )

        # Modify state
        agent.state_manager.set_status(AgentStatus.THINKING)
        task_id = agent.state_manager.create_task("Test Task", "Description")
        agent.state_manager.start_task(task_id)

        # Modify memory
        memory_id = agent.memory_manager.add("Test memory content", importance=0.8)

        # Save original values for comparison
        original_id = agent.id
        original_created_at = agent.created_at
        original_status = agent.state_manager.status
        original_task_id = agent.state_manager.current_task_id
        original_memory_count = len(agent.memory_manager.working_memory)
        original_memory_content = agent.memory_manager.get(memory_id).content

        # 2. Save state
        agent.save_state(self.state_file)

        # Verify file exists
        self.assertTrue(os.path.exists(self.state_file))

        # 3. Create a new agent instance
        new_state_manager = AgentState()
        # Use a different storage path for the new agent to ensure we are testing working memory loading
        # and not accidentally just reading from DB (though DB reading is also valid, but working memory is transient)
        # Actually, if we use same storage path, long term memory is shared via DB.
        # But working memory is in-memory list. We want to test that is restored from JSON.
        new_memory_manager = Memory(storage_path=self.memory_storage)

        new_agent = Agent(
            state_manager=new_state_manager,
            memory_manager=new_memory_manager
        )

        # Verify initial state is different
        self.assertNotEqual(new_agent.id, original_id)
        self.assertEqual(new_agent.state_manager.status, AgentStatus.IDLE)
        self.assertIsNone(new_agent.state_manager.current_task_id)
        self.assertEqual(len(new_agent.memory_manager.working_memory), 0)

        # 4. Load state
        new_agent.load_state(self.state_file)

        # 5. Verify loaded state matches original
        self.assertEqual(new_agent.id, original_id)
        self.assertEqual(new_agent.created_at, original_created_at)
        self.assertEqual(new_agent.config.name, "TestAgent")
        self.assertEqual(new_agent.config.version, "1.0.0")

        self.assertEqual(new_agent.state_manager.status, original_status)
        self.assertEqual(new_agent.state_manager.current_task_id, original_task_id)
        self.assertTrue(original_task_id in new_agent.state_manager.tasks)

        self.assertEqual(len(new_agent.memory_manager.working_memory), original_memory_count)
        self.assertEqual(new_agent.memory_manager.working_memory[0].content, original_memory_content)

if __name__ == '__main__':
    unittest.main()
