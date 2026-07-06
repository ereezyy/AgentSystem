import unittest
import json
import os
import tempfile
from AgentSystem.core.state import AgentState, AgentStatus, TaskInfo

class TestAgentState(unittest.TestCase):
    def setUp(self):
        self.state = AgentState()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = os.path.join(self.temp_dir.name, "state.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initial_state(self):
        self.assertEqual(self.state.status, AgentStatus.IDLE)
        self.assertEqual(len(self.state.tasks), 0)
        self.assertEqual(len(self.state.task_queue), 0)

    def test_save_and_load_success(self):
        # Setup state
        self.state.set_status(AgentStatus.EXECUTING)
        task_id = self.state.create_task("Test Task", "Description")
        self.state.set_context("key", "value")

        # Save state
        self.state.save(self.test_file)
        self.assertTrue(os.path.exists(self.test_file))

        # Load state
        new_state = AgentState.load(self.test_file)

        # Verify
        self.assertEqual(new_state.status, AgentStatus.EXECUTING)
        self.assertIn(task_id, new_state.tasks)
        self.assertEqual(new_state.tasks[task_id].name, "Test Task")
        self.assertEqual(new_state.context.get("key"), "value")
        self.assertEqual(new_state.task_queue, [task_id])

    def test_load_non_existent_file(self):
        """Test loading from a file that does not exist"""
        non_existent_file = os.path.join(self.temp_dir.name, "non_existent.json")

        with self.assertLogs('AgentSystem.core.state', level='ERROR') as cm:
            new_state = AgentState.load(non_existent_file)

        # Should return a default state
        self.assertEqual(new_state.status, AgentStatus.IDLE)
        self.assertEqual(len(new_state.tasks), 0)
        # Verify error was logged
        self.assertTrue(any("Error loading state" in output for output in cm.output))
        self.assertTrue(any("No such file or directory" in output for output in cm.output))

    def test_load_invalid_json(self):
        """Test loading from a file with invalid JSON"""
        with open(self.test_file, 'w') as f:
            f.write("invalid json content")

        with self.assertLogs('AgentSystem.core.state', level='ERROR') as cm:
            new_state = AgentState.load(self.test_file)

        # Should return a default state
        self.assertEqual(new_state.status, AgentStatus.IDLE)
        # Verify error was logged
        self.assertTrue(any("Error loading state" in output for output in cm.output))
        self.assertTrue(any("Expecting value" in output for output in cm.output))

    def test_load_missing_fields(self):
        """Test loading from a file with missing fields but valid JSON"""
        # Minimal JSON with only some fields
        minimal_state = {
            "status": "THINKING",
            "context": {"key": "value"}
        }
        with open(self.test_file, 'w') as f:
            json.dump(minimal_state, f)

        new_state = AgentState.load(self.test_file)

        # Should load available fields and use defaults for others
        self.assertEqual(new_state.status, AgentStatus.THINKING)
        self.assertEqual(new_state.context.get("key"), "value")
        self.assertEqual(len(new_state.tasks), 0)
        self.assertEqual(new_state.current_task_id, None)

    def test_load_invalid_status(self):
        """Test loading from a file with an invalid status name"""
        invalid_state = {
            "status": "NON_EXISTENT_STATUS"
        }
        with open(self.test_file, 'w') as f:
            json.dump(invalid_state, f)

        with self.assertLogs('AgentSystem.core.state', level='ERROR') as cm:
            new_state = AgentState.load(self.test_file)

        # Should return a default state due to KeyError in AgentStatus[state_dict.get("status", "IDLE")]
        self.assertEqual(new_state.status, AgentStatus.IDLE)
        # Verify error was logged
        self.assertTrue(any("Error loading state" in output for output in cm.output))
        self.assertTrue(any("NON_EXISTENT_STATUS" in output for output in cm.output))

if __name__ == '__main__':
    unittest.main()
