import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock dotenv module before importing AgentSystem
# This is necessary because the environment is missing python-dotenv
# and AgentSystem imports it at the top level.
mock_dotenv = MagicMock()
sys.modules['dotenv'] = mock_dotenv

# Import AgentSystem modules after mocking dotenv
from AgentSystem.core.agent_capabilities import AgentReasoner

class TestAgentCapabilities(unittest.TestCase):
    def setUp(self):
        self.agent_id = "test_agent"
        # Patch the logger instance on the module to prevent actual logging
        self.logger_patcher = patch('AgentSystem.core.agent_capabilities.logger')
        self.mock_logger = self.logger_patcher.start()

        self.reasoner = AgentReasoner(self.agent_id)

    def tearDown(self):
        self.logger_patcher.stop()

    @patch('AgentSystem.core.agent_capabilities.ai_service')
    def test_create_plan_exception_handling(self, mock_ai_service):
        """Test that create_plan returns a fallback plan when AI service fails."""
        # Setup the mock to raise an exception
        mock_ai_service.complete.side_effect = Exception("AI Service Failure")

        # Define the task
        task = "Test Task"

        # Execute create_plan
        plan = self.reasoner.create_plan(task)

        # Assertions
        expected_fallback = ["Analyze the task", "Gather necessary information", "Execute the task", "Verify results"]
        self.assertEqual(plan, expected_fallback)
        self.assertEqual(self.reasoner.thought.plan, expected_fallback)

        # Verify ai_service.complete was called
        mock_ai_service.complete.assert_called_once()

        # Verify logger.error was called with the exception message
        self.mock_logger.error.assert_called()
        call_args = self.mock_logger.error.call_args
        self.assertIn("Error creating plan", call_args[0][0])
        self.assertIn("AI Service Failure", call_args[0][0])

if __name__ == '__main__':
    unittest.main()
