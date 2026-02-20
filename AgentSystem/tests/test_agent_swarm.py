import unittest
import unittest.mock
import asyncio
from datetime import datetime
from AgentSystem.core.agent_swarm import SwarmCoordinator, SwarmTask

class TestSwarmCoordinator(unittest.IsolatedAsyncioTestCase):
    async def test_assign_agents_no_suitable_agents(self):
        """
        Test that _assign_agents returns an empty list when no agent has the required capabilities.
        """
        coordinator = SwarmCoordinator()

        # Clear default agents and set up a controlled environment
        coordinator.agents = {}

        # Add a mock agent with specific capabilities
        mock_agent = unittest.mock.MagicMock()
        mock_agent.status = "idle"
        mock_agent.capabilities = ["basic_capability"]
        mock_agent.agent_id = "agent_basic"
        # We need to set performance metrics because the code sorts by it
        mock_agent.performance_metrics = {"success_rate": 0.9}

        coordinator.agents["agent_basic"] = mock_agent

        # Create a task requiring a capability that our agent does not have
        task = SwarmTask(
            id="task_impossible",
            description="Task with impossible capability",
            complexity=10,
            required_capabilities=["advanced_capability"],
            priority=1,
            created_at=datetime.now(),
            assigned_agents=[]
        )

        assigned_agents = await coordinator._assign_agents(task)

        # Verify that no agents were assigned
        self.assertEqual(assigned_agents, [])

    async def test_assign_agents_with_suitable_agents(self):
        """
        Test that _assign_agents returns the correct agent when capabilities match.
        """
        coordinator = SwarmCoordinator()
        coordinator.agents = {}

        mock_agent = unittest.mock.MagicMock()
        mock_agent.status = "idle"
        mock_agent.capabilities = ["required_capability"]
        mock_agent.agent_id = "agent_suitable"
        mock_agent.performance_metrics = {"success_rate": 0.9}

        coordinator.agents["agent_suitable"] = mock_agent

        task = SwarmTask(
            id="task_feasible",
            description="Task with feasible capability",
            complexity=5,
            required_capabilities=["required_capability"],
            priority=1,
            created_at=datetime.now(),
            assigned_agents=[]
        )

        assigned_agents = await coordinator._assign_agents(task)

        self.assertEqual(assigned_agents, ["agent_suitable"])

    async def test_execute_task_no_suitable_agents(self):
        """
        Test that _execute_task handles the case where no agents are assigned.
        """
        coordinator = SwarmCoordinator()

        task = SwarmTask(
            id="task_no_agents",
            description="Task with no agents",
            complexity=10,
            required_capabilities=["some_capability"],
            priority=1,
            created_at=datetime.now(),
            assigned_agents=[] # Explicitly empty
        )

        result = await coordinator._execute_task(task)

        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"], "No suitable agents available")
