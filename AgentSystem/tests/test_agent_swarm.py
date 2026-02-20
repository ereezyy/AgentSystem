import sys
import os
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock missing dependencies
mock_dotenv = MagicMock()
sys.modules["dotenv"] = mock_dotenv

mock_requests = MagicMock()
sys.modules["requests"] = mock_requests

mock_pil = MagicMock()
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil.Image

mock_openai = MagicMock()
sys.modules["openai"] = mock_openai

mock_anthropic = MagicMock()
sys.modules["anthropic"] = mock_anthropic

import unittest
import asyncio
from datetime import datetime
from AgentSystem.core.agent_swarm import SpecializedAgent, SwarmTask, AgentRole

class FailingAgent(SpecializedAgent):
    """A specialized agent that always fails for testing purposes."""

    async def _execute_specialized_task(self, task: SwarmTask):
        """Simulate a task failure by raising an exception."""
        raise ValueError("Simulated failure")

class TestSpecializedAgent(unittest.IsolatedAsyncioTestCase):
    async def test_process_task_error_handling(self):
        """Test that process_task correctly handles exceptions from _execute_specialized_task."""
        # Initialize the failing agent
        agent = FailingAgent(
            agent_id="test_agent",
            role=AgentRole.RESEARCH,
            capabilities=["research"]
        )

        # Create a dummy task
        task = SwarmTask(
            id="task_1",
            description="Test task that will fail",
            complexity=1,
            required_capabilities=["research"],
            priority=1,
            created_at=datetime.now(),
            assigned_agents=["test_agent"]
        )

        # Execute the task
        result = await agent.process_task(task)

        # Verify the results
        self.assertFalse(result["success"], "Task should have failed")
        self.assertEqual(result["error"], "Simulated failure", "Error message mismatch")
        self.assertEqual(agent.status, "error", "Agent status should be 'error'")

if __name__ == "__main__":
    unittest.main()
