import logging
from typing import Dict, List, Optional
from AgentSystem.core.agent_capabilities import AgentReasoner
from AgentSystem.core.state import AgentState, Task
from AgentSystem.modules.system_interface import SystemInterfaceModule
from AgentSystem.modules.safety_module import SafetyModule

logger = logging.getLogger(__name__)

class TaskOrchestrator:
    """
    A class responsible for orchestrating tasks in an autonomous agent system.
    It handles task decomposition, prioritization, execution planning, and adaptive decision-making
    using predictive analytics and reinforcement learning. Integrates with system interfaces
    for actual task execution.
    """
    def __init__(self, reasoner: AgentReasoner, agent_state: AgentState):
        """
        Initializes the TaskOrchestrator with necessary dependencies for task management.
        
        Args:
            reasoner (AgentReasoner): The reasoning component for AI-driven planning and decision-making.
            agent_state (AgentState): The state management component for tracking tasks and agent status.
        """
        self.reasoner = reasoner
        self.agent_state = agent_state
        # Historical data for predictive analytics, stores task outcomes by type for outcome prediction
        self.historical_data: Dict[str, List[float]] = {}
        # Decision policy weights for reinforcement learning, used in task prioritization
        self.decision_weights: Dict[str, float] = {
            "priority": 0.3,    # Weight for base task priority
            "urgency": 0.2,     # Weight for urgency factor
            "importance": 0.2,  # Weight for importance factor
            "outcome": 0.2,     # Weight for predicted outcome
            "dependency": 0.1   # Weight for dependency considerations
        }
        # Initialize system interface for task execution
        self.system_interface = SystemInterfaceModule()
        # Initialize safety module for safety and ethical checks
        self.safety_module = SafetyModule()
        self.safety_module.integrate_with_task_execution(self)
        logger.info("Initialized TaskOrchestrator with SystemInterfaceModule and SafetyModule")

    def decompose_task(self, task: Task) -> List[Task]:
        """
        Breaks down a complex task into smaller, manageable subtasks using AI-driven planning.
        Utilizes the AgentReasoner to generate subtasks with metadata for priority and execution order.
        
        Args:
            task (Task): The task to be decomposed.
            
        Returns:
            List[Task]: A list of subtasks created from the decomposition process.
        """
        if task.is_decomposed:
            logger.info(f"Task {task.name} is already decomposed.")
            return []

        # Prepare prompt for AI-driven task decomposition
        subtasks_prompt = (
            f"Decompose the following task into smaller subtasks:\n"
            f"Task: {task.name}\n"
            f"Description: {task.description}\n"
            f"Return a list of subtasks with their descriptions and any relevant metadata."
        )
        subtasks_response = self.reasoner.decompose_task(subtasks_prompt)
        subtasks = []
        for i, subtask_data in enumerate(subtasks_response):
            subtask_id = f"{task.id}_{i+1}"
            metadata = subtask_data.get("metadata", {})
            # Ensure priority is inherited or adjusted based on decomposition
            subtask_priority = metadata.get("priority", task.priority)
            # Add step number for execution order
            metadata["step_number"] = i + 1
            subtask = Task(
                id=subtask_id,
                name=subtask_data["name"],
                description=subtask_data["description"],
                priority=subtask_priority,
                parent_id=task.id,
                metadata=metadata
            )
            subtasks.append(subtask)
            self.agent_state.add_task(subtask)
            logger.debug(f"Added subtask {subtask.name} with priority {subtask.priority} and metadata {metadata}")
        task.is_decomposed = True
        logger.info(f"Decomposed task {task.name} into {len(subtasks)} subtasks.")
        return subtasks

    def prioritize_tasks(self) -> List[Task]:
        """
        Prioritizes tasks based on urgency, importance, dependencies, and predicted outcomes.
        Uses adaptive weights from reinforcement learning to calculate a score for each task,
        then sorts tasks by score to determine execution order.
        
        Returns:
            List[Task]: A sorted list of tasks based on priority scores.
        """
        tasks = list(self.agent_state.tasks.values())
        if not tasks:
            return []

        # Calculate priority scores with additional factors
        for task in tasks:
            base_priority = task.priority
            urgency_factor = task.metadata.get("urgency", 0.0) if task.metadata else 0.0
            importance_factor = task.metadata.get("importance", 0.0) if task.metadata else 0.0
            
            # Predict task outcome to adjust priority
            predicted_outcome = self.predict_task_outcome(task)
            outcome_factor = predicted_outcome.get("success_probability", 0.5)
            
            # Dependency factor: Lower priority if depends on other tasks
            dependency_factor = 0.0
            if task.parent_id:
                dependency_factor = -0.2  # Slightly lower priority for subtasks waiting on parent
            
            # Calculate final score using adaptive weights from reinforcement learning
            score = (base_priority * self.decision_weights["priority"]) + \
                    (urgency_factor * self.decision_weights["urgency"]) + \
                    (importance_factor * self.decision_weights["importance"]) + \
                    (outcome_factor * self.decision_weights["outcome"]) + \
                    (dependency_factor * self.decision_weights["dependency"])
            task.score = score
            logger.debug(f"Task {task.name} priority score: {score} (Base: {base_priority}, Urgency: {urgency_factor}, Importance: {importance_factor}, Outcome: {outcome_factor}, Dependency: {dependency_factor})")

        # Sort tasks by score in descending order (higher score = higher priority)
        prioritized_tasks = sorted(tasks, key=lambda t: t.score, reverse=True)
        
        # Update task priorities in state based on sorted order
        for idx, task in enumerate(prioritized_tasks):
            task.priority = len(tasks) - idx  # Higher number = higher priority
            self.agent_state.update_task(task)
        
        logger.info(f"Prioritized {len(tasks)} tasks. Top task: {prioritized_tasks[0].name} with score {prioritized_tasks[0].score}")
        return prioritized_tasks

    def plan_execution(self, task_id: str) -> List[str]:
        """
        Creates a detailed execution plan for a given task, considering dependencies and subtask order.
        If the task has no subtasks, it uses the reasoner to create a direct plan. If subtasks exist,
        it sorts them by metadata (e.g., step_number) to create an ordered plan.
        
        Args:
            task_id (str): The ID of the task to plan execution for.
            
        Returns:
            List[str]: A list of steps or subtask descriptions representing the execution plan.
        """
        task = self.agent_state.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for execution planning.")
            return []

        # Check if the task has subtasks (i.e., it has been decomposed)
        if not any(t.parent_id == task_id for t in self.agent_state.tasks.values()):
            # No subtasks, create a direct plan using the reasoner
            plan = self.reasoner.create_plan(f"Execute task: {task.name} - {task.description}")
            logger.debug(f"Created new execution plan with {len(plan)} steps for task {task.name}")
            return plan
        else:
            # Task has subtasks, retrieve and sort them based on metadata (e.g., step_number)
            subtasks = [t for t in self.agent_state.tasks.values() if t.parent_id == task_id]
            sorted_subtasks = sorted(
                subtasks,
                key=lambda t: t.metadata.get("step_number", float('inf')) if t.metadata else float('inf')
            )
            # Create a plan based on the sorted subtasks
            plan = [t.description for t in sorted_subtasks]
            logger.debug(f"Retrieved execution plan with {len(plan)} subtasks for task {task.name}")
            return plan

    def predict_task_outcome(self, task: Task) -> Dict[str, float]:
        """
        Predicts the outcome of a task based on historical data or simple heuristics.
        Uses historical data if available for the task type to calculate success probability
        and estimated duration. Falls back to heuristics based on priority and complexity if no data exists.
        
        Args:
            task (Task): The task to predict the outcome for.
            
        Returns:
            Dict[str, float]: A dictionary with 'success_probability' and 'estimated_duration' predictions.
        """
        task_type = task.metadata.get("type", "generic") if task.metadata else "generic"
        if task_type in self.historical_data:
            past_outcomes = self.historical_data[task_type]
            success_count = sum(1 for outcome in past_outcomes if outcome > 0.5)
            success_probability = success_count / len(past_outcomes) if past_outcomes else 0.5
            avg_duration = sum(past_outcomes) / len(past_outcomes) if past_outcomes else 1.0
        else:
            # Default heuristic if no historical data
            priority_factor = task.priority / 10.0  # Assuming max priority is 10
            complexity_factor = task.metadata.get("complexity", 0.5) if task.metadata else 0.5
            success_probability = max(0.1, min(0.9, priority_factor * (1.0 - complexity_factor)))
            avg_duration = 1.0 + complexity_factor * 2.0

        outcome = {
            "success_probability": success_probability,
            "estimated_duration": avg_duration
        }
        logger.debug(f"Predicted outcome for task {task.name}: {outcome}")
        return outcome

    def learn_from_execution(self, task_id: str, outcome: float):
        """
        Updates the decision-making model based on task execution outcomes.
        Records the outcome in historical data for the task type, which is used for predictive analytics
        and reinforcement learning adjustments.
        
        Args:
            task_id (str): The ID of the task to record the outcome for.
            outcome (float): The outcome of the task execution, typically between 0.0 (failure) and 1.0 (success).
        """
        task = self.agent_state.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for learning update.")
            return

        task_type = task.metadata.get("type", "generic") if task.metadata else "generic"
        if task_type not in self.historical_data:
            self.historical_data[task_type] = []
        self.historical_data[task_type].append(outcome)
        logger.info(f"Updated historical data for task type {task_type} with outcome {outcome}")

    def adapt_decision_strategy(self):
        """
        Adapts the decision-making strategy based on learned experiences using reinforcement learning.
        Adjusts the weights used in task prioritization based on historical outcomes for each task type.
        Increases weights for successful factors if performance is good, or adjusts for better prediction
        if performance is poor.
        """
        logger.info("Adapting decision strategy based on learned experiences.")
        # Simple reinforcement learning: Adjust weights based on historical outcomes
        for task_type, outcomes in self.historical_data.items():
            if len(outcomes) < 5:  # Need enough data points to adjust strategy
                continue
            
            avg_outcome = sum(outcomes) / len(outcomes)
            if avg_outcome < 0.3:  # Poor performance, adjust weights
                logger.info(f"Poor performance for task type {task_type}, adjusting decision weights.")
                # Increase weight for outcome prediction to prioritize tasks with higher success probability
                self.decision_weights["outcome"] = min(0.4, self.decision_weights["outcome"] + 0.05)
                # Decrease weight for base priority to rely less on static values
                self.decision_weights["priority"] = max(0.1, self.decision_weights["priority"] - 0.05)
            elif avg_outcome > 0.7:  # Good performance, reinforce current strategy
                logger.info(f"Good performance for task type {task_type}, reinforcing decision weights.")
                # Slightly increase weight for successful factors
                self.decision_weights["outcome"] = min(0.4, self.decision_weights["outcome"] + 0.02)
                self.decision_weights["urgency"] = min(0.3, self.decision_weights["urgency"] + 0.02)
            else:
                logger.info(f"Stable performance for task type {task_type}, maintaining decision weights.")
            
            # Ensure weights sum to 1.0 to maintain balanced scoring
            total = sum(self.decision_weights.values())
            if total != 1.0:
                for key in self.decision_weights:
                    self.decision_weights[key] = self.decision_weights[key] / total
        
        logger.debug(f"Updated decision weights: {self.decision_weights}")

    def execute_next_task(self) -> Optional[str]:
        """
        Identifies and executes the next task based on priority and dependencies using SystemInterfaceModule.
        Prioritizes tasks, selects the highest priority pending task, generates an execution plan,
        and uses the system interface to execute each step as a system command. Updates task status
        based on execution outcome and records learning data.
        
        Returns:
            Optional[str]: The ID of the executed task, or None if no tasks are ready for execution.
        """
        prioritized_tasks = self.prioritize_tasks()
        if not prioritized_tasks:
            logger.info("No tasks available for execution.")
            return None

        for task in prioritized_tasks:
            if task.status == "pending":
                plan = self.plan_execution(task.id)
                if plan:
                    task.status = "in_progress"
                    self.agent_state.update_task(task)
                    logger.info(f"Executing task {task.name} with plan: {plan}")
                    # Execute the plan using SystemInterfaceModule
                    for step in plan:
                        # Check if the step can be translated to a system command
                        command = self.reasoner.translate_to_command(step)
                        if command:
                            result = self.system_interface.execute_command(command)
                            if result.get("success", False):
                                logger.info(f"Successfully executed step: {step} with command: {command}")
                                logger.debug(f"Command output: {result.get('stdout', '')}")
                            else:
                                logger.error(f"Failed to execute step: {step} with command: {command}")
                                logger.error(f"Error: {result.get('error', 'Unknown error')}")
                                task.status = "failed"
                                self.agent_state.update_task(task)
                                self.learn_from_execution(task.id, 0.0)  # Record failure
                                return task.id
                        else:
                            logger.warning(f"No command translation for step: {step}")
                    task.status = "completed"
                    self.agent_state.update_task(task)
                    self.learn_from_execution(task.id, 1.0)  # Record success
                    return task.id
        logger.info("No pending tasks ready for execution.")
        return None