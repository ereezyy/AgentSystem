import unittest
from unittest.mock import MagicMock
from AgentSystem.core.task_orchestrator import TaskOrchestrator
from AgentSystem.core.state import AgentState, Task
from AgentSystem.core.agent_capabilities import AgentReasoner
from AgentSystem.modules.system_interface import SystemInterfaceModule

class TestTaskOrchestrator(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.reasoner = MagicMock(spec=AgentReasoner)
        self.agent_state = MagicMock(spec=AgentState)
        self.system_interface = MagicMock(spec=SystemInterfaceModule)
        
        # Initialize TaskOrchestrator with mocked dependencies
        self.orchestrator = TaskOrchestrator(self.reasoner, self.agent_state)
        self.orchestrator.system_interface = self.system_interface
        
        # Sample task for testing
        self.task = Task(
            id="task1",
            name="Test Task",
            description="A test task for orchestration",
            priority=5
        )
        
        # Setup mock returns
        self.agent_state.tasks = {}
        self.reasoner.decompose_task.return_value = [
            {"name": "Subtask 1", "description": "First subtask", "metadata": {"priority": 5, "step_number": 1}},
            {"name": "Subtask 2", "description": "Second subtask", "metadata": {"priority": 5, "step_number": 2}}
        ]
        self.reasoner.create_plan.return_value = ["Step 1", "Step 2"]
        self.reasoner.translate_to_command.return_value = "echo test command"
        self.system_interface.execute_command.return_value = {"success": True, "stdout": "Test output", "stderr": ""}

    def test_decompose_task(self):
        # Test task decomposition
        subtasks = self.orchestrator.decompose_task(self.task)
        self.assertEqual(len(subtasks), 2)
        self.assertTrue(self.task.is_decomposed)
        self.assertEqual(subtasks[0].name, "Subtask 1")
        self.assertEqual(subtasks[1].name, "Subtask 2")
        self.assertEqual(subtasks[0].parent_id, "task1")
        self.assertEqual(subtasks[1].parent_id, "task1")

    def test_decompose_task_already_decomposed(self):
        # Test decomposition of already decomposed task
        self.task.is_decomposed = True
        subtasks = self.orchestrator.decompose_task(self.task)
        self.assertEqual(len(subtasks), 0)

    def test_prioritize_tasks_empty(self):
        # Test prioritization with no tasks
        self.agent_state.tasks = {}
        prioritized = self.orchestrator.prioritize_tasks()
        self.assertEqual(len(prioritized), 0)

    def test_prioritize_tasks_single(self):
        # Test prioritization with a single task
        self.agent_state.tasks = {"task1": self.task}
        prioritized = self.orchestrator.prioritize_tasks()
        self.assertEqual(len(prioritized), 1)
        self.assertEqual(prioritized[0].name, "Test Task")

    def test_prioritize_tasks_multiple(self):
        # Test prioritization with multiple tasks
        task2 = Task(id="task2", name="High Priority Task", description="High priority", priority=8)
        self.agent_state.tasks = {"task1": self.task, "task2": task2}
        prioritized = self.orchestrator.prioritize_tasks()
        self.assertEqual(len(prioritized), 2)
        self.assertEqual(prioritized[0].name, "High Priority Task")  # Higher priority should come first

    def test_plan_execution_no_subtasks(self):
        # Test execution planning for a task with no subtasks
        self.agent_state.tasks = {"task1": self.task}
        plan = self.orchestrator.plan_execution("task1")
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan, ["Step 1", "Step 2"])

    def test_plan_execution_with_subtasks(self):
        # Test execution planning for a task with subtasks
        subtask1 = Task(id="task1_1", name="Subtask 1", description="First subtask", priority=5, parent_id="task1", metadata={"step_number": 1})
        subtask2 = Task(id="task1_2", name="Subtask 2", description="Second subtask", priority=5, parent_id="task1", metadata={"step_number": 2})
        self.agent_state.tasks = {"task1": self.task, "task1_1": subtask1, "task1_2": subtask2}
        plan = self.orchestrator.plan_execution("task1")
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan, ["First subtask", "Second subtask"])

    def test_predict_task_outcome_no_history(self):
        # Test outcome prediction with no historical data
        outcome = self.orchestrator.predict_task_outcome(self.task)
        self.assertIn("success_probability", outcome)
        self.assertIn("estimated_duration", outcome)
        self.assertGreaterEqual(outcome["success_probability"], 0.1)
        self.assertLessEqual(outcome["success_probability"], 0.9)

    def test_predict_task_outcome_with_history(self):
        # Test outcome prediction with historical data
        self.orchestrator.historical_data["generic"] = [0.8, 0.9, 0.7]
        outcome = self.orchestrator.predict_task_outcome(self.task)
        self.assertIn("success_probability", outcome)
        self.assertGreater(outcome["success_probability"], 0.5)

    def test_learn_from_execution(self):
        # Test learning from execution outcome
        self.agent_state.tasks = {"task1": self.task}
        self.orchestrator.learn_from_execution("task1", 0.8)
        self.assertIn("generic", self.orchestrator.historical_data)
        self.assertEqual(self.orchestrator.historical_data["generic"], [0.8])

    def test_adapt_decision_strategy_no_data(self):
        # Test decision strategy adaptation with insufficient data
        self.orchestrator.historical_data["generic"] = [0.8]
        initial_weights = self.orchestrator.decision_weights.copy()
        self.orchestrator.adapt_decision_strategy()
        self.assertEqual(self.orchestrator.decision_weights, initial_weights)

    def test_adapt_decision_strategy_poor_performance(self):
        # Test decision strategy adaptation with poor performance
        self.orchestrator.historical_data["generic"] = [0.2, 0.1, 0.3, 0.2, 0.1]
        initial_outcome_weight = self.orchestrator.decision_weights["outcome"]
        self.orchestrator.adapt_decision_strategy()
        self.assertGreater(self.orchestrator.decision_weights["outcome"], initial_outcome_weight)

    def test_execute_next_task_no_tasks(self):
        # Test task execution with no tasks
        self.agent_state.tasks = {}
        result = self.orchestrator.execute_next_task()
        self.assertIsNone(result)

    def test_execute_next_task_success(self):
        # Test successful task execution
        self.agent_state.tasks = {"task1": self.task}
        result = self.orchestrator.execute_next_task()
        self.assertEqual(result, "task1")
        self.assertEqual(self.task.status, "completed")

    def test_execute_next_task_failure(self):
        # Test task execution with failure
        self.system_interface.execute_command.return_value = {"success": False, "error": "Command failed"}
        self.agent_state.tasks = {"task1": self.task}
        result = self.orchestrator.execute_next_task()
        self.assertEqual(result, "task1")
        self.assertEqual(self.task.status, "failed")

if __name__ == '__main__':
    unittest.main()