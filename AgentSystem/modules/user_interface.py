"""
User Interface Module
---------------------
Handles user interaction and feedback for the autonomous agent system.
This module provides mechanisms for command input, status reporting,
customizable settings, and feedback loops to maintain transparency
and alignment with user intent.
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.core.state import AgentState
from AgentSystem.core.memory import Memory

# Get module logger
logger = get_logger("modules.user_interface")


class UserInterfaceModule:
    """Module for handling user interaction and feedback in the autonomous agent system"""
    
    def __init__(self, agent_state: Optional[AgentState] = None, memory_manager: Optional[Memory] = None):
        """
        Initialize the user interface module
        
        Args:
            agent_state: AgentState instance for tracking tasks and status
            memory_manager: Memory instance for storing user preferences and feedback
        """
        self.agent_state = agent_state
        self.memory_manager = memory_manager
        self.settings = self._load_default_settings()
        self.feedback_log = []
        self.status_callbacks: list[Callable[[str, Any], None]] = []
        logger.info("Initialized UserInterfaceModule")
    
    def _load_default_settings(self) -> Dict[str, Any]:
        """
        Load default user settings
        
        Returns:
            Dictionary containing default settings
        """
        return {
            "notification_level": "normal",  # Options: silent, normal, verbose
            "auto_refresh_interval": 5000,  # Milliseconds for UI auto-refresh
            "theme": "light",  # Options: light, dark
            "language": "en",  # Default language code
            "max_displayed_tasks": 10  # Maximum number of tasks to display in UI
        }
    
    def get_tools(self) -> Dict[str, Any]:
        """
        Get tools provided by this module
        
        Returns:
            Dictionary of available tools with descriptions and parameters
        """
        return {
            "submit_command": {
                "description": "Submit a command or task for the agent to execute",
                "function": self.submit_command,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command or task description to execute"
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Priority level for the task (1-10)",
                            "default": 5
                        }
                    },
                    "required": ["command"]
                }
            },
            "get_status": {
                "description": "Get the current status of the agent and active tasks",
                "function": self.get_status,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "update_settings": {
                "description": "Update user interface settings and preferences",
                "function": self.update_settings,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "settings": {
                            "type": "object",
                            "description": "Settings to update",
                            "properties": {
                                "notification_level": {
                                    "type": "string",
                                    "enum": ["silent", "normal", "verbose"]
                                },
                                "auto_refresh_interval": {
                                    "type": "integer",
                                    "description": "UI auto-refresh interval in milliseconds"
                                },
                                "theme": {
                                    "type": "string",
                                    "enum": ["light", "dark"]
                                },
                                "language": {
                                    "type": "string",
                                    "description": "Language code (e.g., 'en')"
                                },
                                "max_displayed_tasks": {
                                    "type": "integer",
                                    "description": "Maximum number of tasks to display"
                                }
                            }
                        }
                    },
                    "required": ["settings"]
                }
            },
            "provide_feedback": {
                "description": "Provide feedback on agent performance or task results",
                "function": self.provide_feedback,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task related to this feedback, if applicable"
                        },
                        "rating": {
                            "type": "integer",
                            "description": "Rating from 1 (poor) to 5 (excellent)",
                            "default": 3
                        },
                        "comment": {
                            "type": "string",
                            "description": "Detailed feedback or comments"
                        }
                    },
                    "required": ["comment"]
                }
            },
            "register_status_callback": {
                "description": "Register a callback function to receive status updates",
                "function": self.register_status_callback,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "callback": {
                            "type": "object",
                            "description": "Callback function to receive status updates"
                        }
                    },
                    "required": ["callback"]
                }
            }
        }
    
    def submit_command(self, command: str, priority: int = 5) -> Dict[str, Any]:
        """
        Submit a command or task for the agent to execute
        
        Args:
            command: The command or task description to execute
            priority: Priority level for the task (1-10)
            
        Returns:
            Dictionary with submission result
        """
        try:
            if not self.agent_state:
                logger.error("AgentState not initialized for UserInterfaceModule")
                return {
                    "success": False,
                    "reason": "Agent state not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Create a task from the command
            task_id = self.agent_state.create_task(
                name="User Command",
                description=command,
                priority=priority
            )
            
            # Start the task
            self.agent_state.start_task(task_id)
            
            logger.info(f"Submitted user command as task {task_id}: {command}")
            return {
                "success": True,
                "task_id": task_id,
                "command": command,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error submitting command '{command}': {e}")
            return {
                "success": False,
                "reason": str(e),
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent and active tasks
        
        Returns:
            Dictionary with status information
        """
        try:
            if not self.agent_state:
                logger.error("AgentState not initialized for UserInterfaceModule")
                return {
                    "success": False,
                    "reason": "Agent state not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            
            status = self.agent_state.status.name
            current_task_id = self.agent_state.current_task_id
            tasks = self.agent_state.get_all_tasks()
            
            # Convert tasks to a list of dictionaries for serialization
            tasks_list = [
                {
                    "task_id": task_id,
                    "name": task.name,
                    "description": task.description,
                    "status": task.status,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at
                }
                for task_id, task in tasks.items()
            ]
            
            # If memory manager is available, include memory stats
            memory_stats = {}
            if self.memory_manager:
                memory_stats = {
                    "working_memory_items": len(self.memory_manager.working_memory)
                }
            
            status_data = {
                "success": True,
                "agent_status": status,
                "current_task_id": current_task_id or "None",
                "tasks": tasks_list,
                "memory": memory_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            # Notify registered callbacks
            for callback in self.status_callbacks:
                try:
                    callback("status_update", status_data)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")
            
            return status_data
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                "success": False,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user interface settings and preferences
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            Dictionary with update result
        """
        try:
            # Validate settings
            if "notification_level" in settings and settings["notification_level"] not in ["silent", "normal", "verbose"]:
                return {
                    "success": False,
                    "reason": f"Invalid notification level: {settings['notification_level']}",
                    "timestamp": datetime.now().isoformat()
                }
            
            if "theme" in settings and settings["theme"] not in ["light", "dark"]:
                return {
                    "success": False,
                    "reason": f"Invalid theme: {settings['theme']}",
                    "timestamp": datetime.now().isoformat()
                }
            
            if "auto_refresh_interval" in settings and (settings["auto_refresh_interval"] < 1000 or settings["auto_refresh_interval"] > 60000):
                return {
                    "success": False,
                    "reason": f"Auto-refresh interval must be between 1000 and 60000 ms",
                    "timestamp": datetime.now().isoformat()
                }
            
            if "max_displayed_tasks" in settings and (settings["max_displayed_tasks"] < 1 or settings["max_displayed_tasks"] > 50):
                return {
                    "success": False,
                    "reason": f"Max displayed tasks must be between 1 and 50",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Update settings
            self.settings.update(settings)
            
            # Save to memory if available
            if self.memory_manager:
                self.memory_manager.add_to_long_term({
                    "type": "user_settings",
                    "settings": self.settings,
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"Updated user settings: {settings}")
            return {
                "success": True,
                "updated_settings": self.settings,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return {
                "success": False,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def provide_feedback(self, comment: str, task_id: Optional[str] = None, rating: int = 3) -> Dict[str, Any]:
        """
        Provide feedback on agent performance or task results
        
        Args:
            comment: Detailed feedback or comments
            task_id: ID of the task related to this feedback, if applicable
            rating: Rating from 1 (poor) to 5 (excellent)
            
        Returns:
            Dictionary with feedback submission result
        """
        try:
            if rating < 1 or rating > 5:
                return {
                    "success": False,
                    "reason": "Rating must be between 1 and 5",
                    "timestamp": datetime.now().isoformat()
                }
            
            feedback_entry = {
                "task_id": task_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            }
            self.feedback_log.append(feedback_entry)
            
            # Store in memory if available
            if self.memory_manager:
                self.memory_manager.add_to_long_term({
                    "type": "user_feedback",
                    "feedback": feedback_entry
                })
            
            logger.info(f"Received user feedback for task {task_id or 'general'}: Rating {rating}, Comment: {comment}")
            return {
                "success": True,
                "feedback_id": len(self.feedback_log),
                "task_id": task_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            return {
                "success": False,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def register_status_callback(self, callback: Callable[[str, Any], None]) -> Dict[str, Any]:
        """
        Register a callback function to receive status updates
        
        Args:
            callback: Callback function to receive status updates
            
        Returns:
            Dictionary with registration result
        """
        try:
            self.status_callbacks.append(callback)
            logger.info(f"Registered new status callback. Total callbacks: {len(self.status_callbacks)}")
            return {
                "success": True,
                "callback_id": len(self.status_callbacks),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error registering status callback: {e}")
            return {
                "success": False,
                "reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def notify_status_update(self, update_type: str, data: Any) -> None:
        """
        Notify all registered callbacks of a status update
        
        Args:
            update_type: Type of update (e.g., 'status_update', 'task_update')
            data: Data associated with the update
        """
        for callback in self.status_callbacks:
            try:
                callback(update_type, data)
            except Exception as e:
                logger.error(f"Error in status callback for update type {update_type}: {e}")