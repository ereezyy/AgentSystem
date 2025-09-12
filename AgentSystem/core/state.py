"""
Agent State Management
---------------------
Handles the agent's state, including current tasks, status, and context
"""

import time
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# Local imports
from AgentSystem.utils.logger import get_logger

# Get module logger
logger = get_logger("core.state")


class AgentStatus(Enum):
    """Possible statuses for an agent"""
    IDLE = auto()
    INITIALIZING = auto()
    THINKING = auto()
    EXECUTING = auto()
    WAITING = auto()
    SUCCESS = auto()
    ERROR = auto()
    TERMINATED = auto()


@dataclass
class TaskInfo:
    """Information about a task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    priority: int = 1
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class AgentState:
    """
    Manages the agent's state
    
    Tracks the agent's current status, active tasks, context, and other
    state information needed for operation.
    """
    
    def __init__(self):
        """Initialize the state manager"""
        self.status = AgentStatus.IDLE
        self.current_task_id: Optional[str] = None
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue: List[str] = []
        self.context: Dict[str, Any] = {}
        self.last_updated = time.time()
        self.execution_count = 0
        
        logger.debug("Agent state initialized")
    
    def set_status(self, status: AgentStatus) -> None:
        """
        Set the agent's status
        
        Args:
            status: New status
        """
        old_status = self.status
        self.status = status
        self.last_updated = time.time()
        
        logger.info(f"Agent status changed: {old_status.name} -> {status.name}")
    
    def create_task(
        self, 
        name: str, 
        description: str = "", 
        priority: int = 1,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new task
        
        Args:
            name: Task name
            description: Task description
            priority: Task priority (higher is more important)
            parent_id: Parent task ID
            metadata: Additional task metadata
            
        Returns:
            Task ID
        """
        task = TaskInfo(
            name=name,
            description=description,
            priority=priority,
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        self.tasks[task.id] = task
        self.task_queue.append(task.id)
        self.last_updated = time.time()
        
        logger.debug(f"Created task '{name}' with ID {task.id}")
        
        return task.id
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get a task by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            Task information
        """
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs) -> bool:
        """
        Update a task
        
        Args:
            task_id: Task ID
            **kwargs: Fields to update
            
        Returns:
            Success flag
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found for update")
            return False
        
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        self.last_updated = time.time()
        logger.debug(f"Updated task {task_id}")
        
        return True
    
    def start_task(self, task_id: str) -> bool:
        """
        Start a task
        
        Args:
            task_id: Task ID
            
        Returns:
            Success flag
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found to start")
            return False
        
        task.status = "running"
        task.started_at = time.time()
        self.current_task_id = task_id
        self.last_updated = time.time()
        
        logger.info(f"Started task '{task.name}' ({task_id})")
        
        return True
    
    def complete_task(self, task_id: str, success: bool = True) -> bool:
        """
        Mark a task as complete
        
        Args:
            task_id: Task ID
            success: Whether the task was successful
            
        Returns:
            Success flag
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found to complete")
            return False
        
        task.status = "success" if success else "failed"
        task.completed_at = time.time()
        
        if self.current_task_id == task_id:
            self.current_task_id = None
        
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        self.last_updated = time.time()
        
        logger.info(f"Completed task '{task.name}' ({task_id}) with status: {task.status}")
        
        return True
    
    def get_next_task(self) -> Optional[str]:
        """
        Get the next task ID from the queue
        
        Returns:
            Next task ID or None
        """
        if not self.task_queue:
            return None
        
        # Sort by priority
        self.task_queue.sort(
            key=lambda task_id: self.tasks[task_id].priority if task_id in self.tasks else 0,
            reverse=True
        )
        
        return self.task_queue[0]
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set a context value
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        self.last_updated = time.time()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a context value
        
        Args:
            key: Context key
            default: Default value
            
        Returns:
            Context value or default
        """
        return self.context.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary
        
        Returns:
            State dictionary
        """
        return {
            "status": self.status.name,
            "current_task_id": self.current_task_id,
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()},
            "task_queue": self.task_queue.copy(),
            "context": self.context.copy(),
            "last_updated": self.last_updated,
            "execution_count": self.execution_count
        }
    
    def save(self, file_path: str) -> None:
        """
        Save state to file
        
        Args:
            file_path: Path to save state
        """
        state_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.debug(f"Saved state to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'AgentState':
        """
        Load state from file
        
        Args:
            file_path: Path to load state from
            
        Returns:
            Agent state
        """
        state = cls()
        
        try:
            with open(file_path, 'r') as f:
                state_dict = json.load(f)
            
            state.status = AgentStatus[state_dict.get("status", "IDLE")]
            state.current_task_id = state_dict.get("current_task_id")
            
            # Load tasks
            for tid, task_dict in state_dict.get("tasks", {}).items():
                task = TaskInfo(**task_dict)
                state.tasks[tid] = task
            
            state.task_queue = state_dict.get("task_queue", [])
            state.context = state_dict.get("context", {})
            state.last_updated = state_dict.get("last_updated", time.time())
            state.execution_count = state_dict.get("execution_count", 0)
            
            logger.debug(f"Loaded state from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading state from {file_path}: {e}")
        
        return state
