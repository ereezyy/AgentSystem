"""
Base Agent Classes
-----------------
Defines the core Agent class and related configuration
"""

import time
import uuid
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

# Local imports
from AgentSystem.utils.logger import get_logger

# Get module logger
logger = get_logger("core.agent")


@dataclass
class AgentConfig:
    """Configuration for an Agent"""
    
    # Basic configuration
    name: str = "agent"
    description: str = "A general purpose autonomous agent"
    version: str = "0.1.0"
    
    # Capabilities configuration
    capabilities: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    
    # Performance configuration
    max_iterations: int = 100
    timeout_seconds: int = 300
    
    # Memory configuration
    memory_size: int = 10000
    
    # System configuration
    working_directory: str = "./data/workspace"
    cache_directory: str = "./data/cache"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentConfig':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    def save(self, file_path: str) -> None:
        """Save configuration to file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, file_path: str) -> 'AgentConfig':
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())


class Agent:
    """
    Base Agent class
    
    An agent is an autonomous entity that can perceive its environment,
    make decisions, and take actions to achieve goals.
    """
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None,
        state_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None
    ):
        """
        Initialize the agent
        
        Args:
            config: Agent configuration
            state_manager: State management system
            memory_manager: Memory management system
        """
        self.config = config or AgentConfig()
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        self.state_manager = state_manager
        self.memory_manager = memory_manager
        self._modules = {}
        self._handlers = {}
        
        logger.info(f"Agent '{self.config.name}' ({self.id}) initialized")
    
    def register_module(self, name: str, module: Any) -> None:
        """
        Register a module with the agent
        
        Args:
            name: Module name
            module: Module instance
        """
        self._modules[name] = module
        logger.debug(f"Registered module '{name}'")
    
    def get_module(self, name: str) -> Any:
        """
        Get a registered module
        
        Args:
            name: Module name
            
        Returns:
            Module instance
        """
        if name not in self._modules:
            logger.warning(f"Module '{name}' not found")
            return None
        
        return self._modules[name]
    
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register an event handler
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for '{event_type}'")
    
    def handle_event(self, event_type: str, event_data: Any) -> List[Any]:
        """
        Handle an event
        
        Args:
            event_type: Type of event
            event_data: Event data
            
        Returns:
            List of handler results
        """
        if event_type not in self._handlers:
            logger.warning(f"No handlers for event type '{event_type}'")
            return []
        
        results = []
        for handler in self._handlers[event_type]:
            try:
                result = handler(event_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in handler for '{event_type}': {e}")
                results.append(None)
        
        return results
    
    def run(self, task: Any = None) -> Any:
        """
        Run the agent on a task
        
        Args:
            task: Task to run
            
        Returns:
            Task result
        """
        logger.info(f"Agent '{self.config.name}' starting task")
        
        # Initialize tracking variables
        start_time = time.time()
        iterations = 0
        
        # Main agent loop
        while iterations < self.config.max_iterations:
            # Check for timeout
            if time.time() - start_time > self.config.timeout_seconds:
                logger.warning(f"Agent timed out after {iterations} iterations")
                break
            
            # TODO: Implement the agent's decision-making process
            
            iterations += 1
        
        logger.info(f"Agent '{self.config.name}' completed task after {iterations} iterations")
        
        # Return the result
        return None
    
    def save_state(self, file_path: str) -> None:
        """
        Save agent state to file
        
        Args:
            file_path: Path to save state
        """
        # TODO: Implement state saving
        pass
    
    def load_state(self, file_path: str) -> None:
        """
        Load agent state from file
        
        Args:
            file_path: Path to load state from
        """
        # TODO: Implement state loading
        pass
    
    def __str__(self) -> str:
        """String representation"""
        return f"Agent('{self.config.name}', {self.id})"
