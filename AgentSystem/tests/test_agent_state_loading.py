import unittest
import os
import json
import time
import shutil
import sys
from unittest.mock import MagicMock

# Mock dotenv before importing AgentSystem
mock_dotenv = MagicMock()
sys.modules['dotenv'] = mock_dotenv

# Add parent directory to path to import AgentSystem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now import the modules
from AgentSystem.core.agent import Agent, AgentConfig
from AgentSystem.core.state import AgentState, AgentStatus
from AgentSystem.core.memory import Memory, MemoryItem

class TestAgentStateLoading(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data_state_loading"
        os.makedirs(self.test_dir, exist_ok=True)
        self.state_file = os.path.join(self.test_dir, "agent_state.json")
        self.memory_path = os.path.join(self.test_dir, "memory")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load_state(self):
        # 1. Create and configure an agent with state and memory
        config = AgentConfig(name="test_agent")
        state_manager = AgentState()
        memory_manager = Memory(storage_path=self.memory_path)

        agent = Agent(config=config, state_manager=state_manager, memory_manager=memory_manager)

        # Modify state
        agent.state_manager.set_status(AgentStatus.THINKING)
        task_id = agent.state_manager.create_task("Test Task", "Description")
        agent.state_manager.start_task(task_id) # Set as current task and running

        # Modify memory (working memory)
        memory_id = agent.memory_manager.add("Test memory content", importance=0.6)

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

        # 3. Create a new agent and load state
        # Initialize with empty/default components
        new_state_manager = AgentState()
        new_memory_manager = Memory(storage_path=self.memory_path) # Use same storage path for DB consistency
        new_agent = Agent(state_manager=new_state_manager, memory_manager=new_memory_manager)

        # Ensure it's different initially
        self.assertNotEqual(new_agent.id, original_id)
        self.assertNotEqual(new_agent.config.name, "test_agent")
        self.assertEqual(new_agent.state_manager.status, AgentStatus.IDLE)
        self.assertEqual(len(new_agent.memory_manager.working_memory), 0)

        # Load state
        new_agent.load_state(self.state_file)

        # 4. Verify loaded state
        self.assertEqual(new_agent.id, original_id)
        self.assertEqual(new_agent.created_at, original_created_at)
        self.assertEqual(new_agent.config.name, "test_agent")

        # Verify State Manager
        self.assertEqual(new_agent.state_manager.status, original_status)
        self.assertEqual(new_agent.state_manager.current_task_id, original_task_id)
        self.assertIn(task_id, new_agent.state_manager.tasks)
        self.assertEqual(new_agent.state_manager.tasks[task_id].name, "Test Task")

        # Verify Memory Manager
        self.assertEqual(len(new_agent.memory_manager.working_memory), original_memory_count)
        self.assertEqual(new_agent.memory_manager.working_memory[0].content, original_memory_content)

if __name__ == '__main__':
    unittest.main()
