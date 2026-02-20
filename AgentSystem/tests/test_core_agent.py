import unittest
import sys
import os
from unittest.mock import MagicMock

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock dotenv
sys.modules["dotenv"] = MagicMock()

from AgentSystem.core.agent import Agent, AgentConfig

class TestAgent(unittest.TestCase):
    """Test the Agent class"""

    def test_handle_event_exception_handling(self):
        """Test that handle_event gracefully handles exceptions in handlers."""
        agent = Agent()

        # Define a handler that raises an exception
        def failing_handler(data):
            raise ValueError("Intentional Failure")

        # Define a handler that succeeds
        def successful_handler(data):
            return f"Processed: {data}"

        event_type = "test_event"
        event_data = "test_data"

        # Register handlers
        # Register failing first to test ordering
        agent.register_handler(event_type, failing_handler)
        agent.register_handler(event_type, successful_handler)

        # Execute handle_event
        # We expect it to catch the ValueError and return [None, "Processed: test_data"]
        results = agent.handle_event(event_type, event_data)

        # Verify results
        self.assertEqual(len(results), 2, "Should return results for all handlers")
        self.assertIsNone(results[0], "First handler (failing) should return None")
        self.assertEqual(results[1], "Processed: test_data", "Second handler (successful) should return correct result")

if __name__ == '__main__':
    unittest.main()
