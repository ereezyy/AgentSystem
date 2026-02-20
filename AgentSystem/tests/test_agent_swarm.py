import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import os

# Set dummy API key to prevent import-time errors
os.environ["OPENAI_API_KEY"] = "dummy"

import asyncio
from typing import Dict, Any, List

from AgentSystem.core.agent_swarm import (
    SwarmCoordinator,
    SwarmTask,
    AgentRole,
    SpecializedAgent,
    ResearchAgent,
    CodeAgent,
    SecurityAgent,
    VisionAgent
)

class TestAgentSwarm(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch multimodal_provider to avoid network calls and use of the (failed) provider
        self.patcher = patch("AgentSystem.core.agent_swarm.multimodal_provider")
        self.mock_provider = self.patcher.start()

        self.coordinator = SwarmCoordinator()

    async def asyncTearDown(self):
        self.patcher.stop()

    async def test_initialization(self):
        """Test SwarmCoordinator initialization"""
        self.assertEqual(len(self.coordinator.agents), 4)
        roles = {agent.role for agent in self.coordinator.agents.values()}
        expected_roles = {
            AgentRole.RESEARCH,
            AgentRole.CODE,
            AgentRole.SECURITY,
            AgentRole.VISION
        }
        self.assertEqual(roles, expected_roles)

    async def test_analyze_required_capabilities(self):
        """Test capability analysis logic"""
        # Test research capability
        caps = self.coordinator._analyze_required_capabilities("Research the history of AI")
        self.assertIn("research", caps)

        # Test code capability
        caps = self.coordinator._analyze_required_capabilities("Write a python script")
        self.assertIn("code_generation", caps)

        # Test security capability
        caps = self.coordinator._analyze_required_capabilities("Check for security vulnerabilities")
        self.assertIn("security_analysis", caps)

        # Test vision capability
        caps = self.coordinator._analyze_required_capabilities("Analyze this image")
        self.assertIn("vision", caps)

        # Test multiple capabilities
        caps = self.coordinator._analyze_required_capabilities("Research and write code")
        self.assertIn("research", caps)
        self.assertIn("code_generation", caps)

    async def test_research_task_execution(self):
        """Test execution of a research task"""
        task_desc = "Research the history of the internet"
        task_id = await self.coordinator.submit_task(task_desc)

        task = self.coordinator.tasks[task_id]
        self.assertEqual(task.status, "completed")
        self.assertTrue(task.results["success"])
        self.assertEqual(task.results["combined_output"][0]["agent_type"], "research")
        self.assertIn("findings", task.results["combined_output"][0])

    async def test_code_task_execution(self):
        """Test execution of a code generation task"""
        # Mock the multimodal provider response
        self.mock_provider.generate_code = AsyncMock(return_value={
            "code": "print('Hello World')",
            "success": True
        })

        task_desc = "Write a python script to print Hello World"
        task_id = await self.coordinator.submit_task(task_desc)

        task = self.coordinator.tasks[task_id]
        self.assertEqual(task.status, "completed")
        self.assertTrue(task.results["success"])

        # Verify the code agent was used
        code_result = task.results["combined_output"][0]
        self.assertEqual(code_result["agent_type"], "code")
        self.assertEqual(code_result["code"], "print('Hello World')")

        # Verify provider was called
        self.mock_provider.generate_code.assert_called_once()

    async def test_security_task_execution(self):
        """Test execution of a security task"""
        task_desc = "Perform security analysis on the system"
        task_id = await self.coordinator.submit_task(task_desc)

        task = self.coordinator.tasks[task_id]
        self.assertEqual(task.status, "completed")
        self.assertTrue(task.results["success"])

        security_result = task.results["combined_output"][0]
        self.assertEqual(security_result["agent_type"], "security")
        self.assertIn("vulnerabilities", security_result)

    async def test_vision_task_execution_generate(self):
        """Test execution of a vision generation task"""
        # Mock the multimodal provider response
        self.mock_provider.generate_image = AsyncMock(return_value={
            "image_url": "http://example.com/cat.jpg",
            "success": True
        })

        task_desc = "Generate an image of a cat"
        task_id = await self.coordinator.submit_task(task_desc)

        task = self.coordinator.tasks[task_id]
        self.assertEqual(task.status, "completed")
        self.assertTrue(task.results["success"])

        vision_result = task.results["combined_output"][0]
        self.assertEqual(vision_result["agent_type"], "vision")
        self.assertEqual(vision_result["image_url"], "http://example.com/cat.jpg")

        # Verify provider was called
        self.mock_provider.generate_image.assert_called_once()

    async def test_vision_task_execution_analyze(self):
        """Test execution of a vision analysis task"""
        task_desc = "Analyze this image for objects"
        task_id = await self.coordinator.submit_task(task_desc)

        task = self.coordinator.tasks[task_id]
        self.assertEqual(task.status, "completed")
        self.assertTrue(task.results["success"])

        vision_result = task.results["combined_output"][0]
        self.assertEqual(vision_result["agent_type"], "vision")
        self.assertIn("objects_detected", vision_result)

    async def test_agent_assignment_logic(self):
        """Test that busy agents are not assigned"""
        # Find the research agent
        research_agent_id = next(
            aid for aid, agent in self.coordinator.agents.items()
            if agent.role == AgentRole.RESEARCH
        )
        research_agent = self.coordinator.agents[research_agent_id]

        # Mark research agent as busy
        research_agent.status = "working"

        # Submit a research task
        task_desc = "Research the history of Python"

        # Override _assign_agents slightly to check logic without full execution
        # Or check the result of submit_task which should fail if no agents available

        # Since submit_task calls _execute_task immediately, and _execute_task returns failure if no agents assigned
        task_id = await self.coordinator.submit_task(task_desc)
        task = self.coordinator.tasks[task_id]

        # Task should fail or have empty results because the only capable agent was busy
        # Note: _assign_agents returns [] if no idle agents match
        # _execute_task returns error if assigned_agents is empty

        self.assertFalse(task.results["success"])
        self.assertEqual(task.results["error"], "No suitable agents available")

    async def test_all_agents_fail(self):
        """Test handling when all assigned agents fail"""
        # Mock the ResearchAgent's _execute_specialized_task to fail
        with patch.object(ResearchAgent, '_execute_specialized_task', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = Exception("Simulated failure")

            task_desc = "Research something complex"
            task_id = await self.coordinator.submit_task(task_desc)

            task = self.coordinator.tasks[task_id]

            # The task wrapper catches exception and returns {"success": False, "error": ...}
            # _combine_agent_results will see no successful results

            self.assertFalse(task.results["success"])
            self.assertEqual(task.results["error"], "All assigned agents failed")

if __name__ == '__main__':
    unittest.main()
