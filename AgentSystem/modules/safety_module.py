"""
Safety Module
-------------
Implements safety protocols and ethical considerations for an autonomous agent designed for full computer control.
This module provides error detection, fail-safes, permission controls, and ethical guidelines to ensure responsible operation.
It integrates with the SystemInterfaceModule and TaskOrchestrator to enforce safety and ethical checks during system interactions and task execution.
"""

import os
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import logging

# Local imports
from AgentSystem.utils.logger import get_logger

# Get module logger
logger = get_logger("modules.safety_module")


class SafetyModule:
    """Module for enforcing safety protocols and ethical considerations in autonomous agent operations"""
    
    def __init__(self):
        """Initialize the safety module with default configurations and rules"""
        self.restricted_paths: Set[str] = {"/etc", "/system", "C:\\Windows", "C:\\System32"}
        self.restricted_commands: Set[str] = {"rm -rf /", "format c:", "deltree", "shutdown -h now"}
        self.ethics_rules = self._load_ethics_rules()
        self.permission_levels = self._initialize_permission_levels()
        self.error_log = []
        logger.info("Initialized SafetyModule with default safety and ethical configurations")
    
    def _load_ethics_rules(self) -> List[Dict[str, Any]]:
        """
        Load ethical guidelines for agent operation
        
        Returns:
            List of ethical rules with conditions and actions
        """
        return [
            {
                "id": "privacy_protection",
                "description": "Protect user privacy by avoiding access to personal data without consent",
                "condition": lambda action: "personal_data" in action.get("context", "") or "user_files" in action.get("target", ""),
                "action": "request_consent",
                "priority": 1
            },
            {
                "id": "non_destructive",
                "description": "Prevent destructive actions that could harm the system or data",
                "condition": lambda action: action.get("type", "") in ["delete", "overwrite", "format"],
                "action": "confirm_intent",
                "priority": 2
            },
            {
                "id": "resource_limit",
                "description": "Limit resource usage to prevent system overload",
                "condition": lambda action: action.get("resource_usage", 0) > 80,
                "action": "scale_down",
                "priority": 3
            }
        ]
    
    def _initialize_permission_levels(self) -> Dict[str, int]:
        """
        Initialize permission levels for different types of operations
        
        Returns:
            Dictionary mapping operation types to required permission levels
        """
        return {
            "file_read": 1,
            "file_write": 2,
            "directory_create": 2,
            "directory_delete": 3,
            "command_execute": 3,
            "system_modify": 4,
            "network_access": 3
        }
    
    def get_tools(self) -> Dict[str, Any]:
        """
        Get tools provided by this module
        
        Returns:
            Dictionary of available tools with descriptions and parameters
        """
        return {
            "check_safety": {
                "description": "Check if an action complies with safety protocols",
                "function": self.check_safety,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "object",
                            "description": "The action to be evaluated for safety"
                        }
                    },
                    "required": ["action"]
                }
            },
            "validate_ethics": {
                "description": "Validate an action against ethical guidelines",
                "function": self.validate_ethics,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "object",
                            "description": "The action to be evaluated for ethical compliance"
                        }
                    },
                    "required": ["action"]
                }
            },
            "check_permissions": {
                "description": "Check if the agent has required permissions for an operation",
                "function": self.check_permissions,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "The type of operation to check permissions for"
                        },
                        "agent_level": {
                            "type": "integer",
                            "description": "The current permission level of the agent",
                            "default": 1
                        }
                    },
                    "required": ["operation"]
                }
            },
            "log_error": {
                "description": "Log an error or safety violation for monitoring",
                "function": self.log_error,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "error_type": {
                            "type": "string",
                            "description": "Type of error or violation"
                        },
                        "details": {
                            "type": "string",
                            "description": "Detailed description of the error"
                        }
                    },
                    "required": ["error_type", "details"]
                }
            },
            "trigger_failsafe": {
                "description": "Trigger a fail-safe mechanism in response to a critical issue",
                "function": self.trigger_failsafe,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for triggering the fail-safe"
                        }
                    },
                    "required": ["reason"]
                }
            }
        }
    
    def check_safety(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an action complies with safety protocols
        
        Args:
            action: The action to be evaluated for safety
            
        Returns:
            Dictionary with safety check result
        """
        try:
            action_type = action.get("type", "unknown")
            target = action.get("target", "")
            
            # Check for restricted paths in file operations
            if action_type in ["file_read", "file_write", "directory_delete"]:
                if any(target.startswith(path) for path in self.restricted_paths):
                    self.log_error("restricted_path_access", f"Attempted access to restricted path: {target}")
                    return {
                        "success": False,
                        "reason": f"Access to restricted path {target} is not allowed",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Check for restricted commands
            if action_type == "command_execute":
                command = action.get("command", "")
                if any(cmd in command for cmd in self.restricted_commands):
                    self.log_error("restricted_command", f"Attempted execution of restricted command: {command}")
                    return {
                        "success": False,
                        "reason": f"Execution of restricted command {command} is not allowed",
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "success": True,
                "reason": "Action complies with safety protocols",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in safety check for action {action}: {e}")
            self.log_error("safety_check_error", f"Error during safety check: {str(e)}")
            return {
                "success": False,
                "reason": f"Safety check failed due to error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_ethics(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an action against ethical guidelines
        
        Args:
            action: The action to be evaluated for ethical compliance
            
        Returns:
            Dictionary with ethics validation result
        """
        try:
            triggered_rules = []
            for rule in sorted(self.ethics_rules, key=lambda r: r["priority"]):
                if rule["condition"](action):
                    triggered_rules.append({
                        "rule_id": rule["id"],
                        "description": rule["description"],
                        "required_action": rule["action"]
                    })
            
            if triggered_rules:
                return {
                    "success": False,
                    "reason": "Action violates ethical guidelines",
                    "triggered_rules": triggered_rules,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": True,
                    "reason": "Action complies with ethical guidelines",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error in ethics validation for action {action}: {e}")
            self.log_error("ethics_validation_error", f"Error during ethics validation: {str(e)}")
            return {
                "success": False,
                "reason": f"Ethics validation failed due to error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def check_permissions(self, operation: str, agent_level: int = 1) -> Dict[str, Any]:
        """
        Check if the agent has required permissions for an operation
        
        Args:
            operation: The type of operation to check permissions for
            agent_level: The current permission level of the agent
            
        Returns:
            Dictionary with permission check result
        """
        try:
            required_level = self.permission_levels.get(operation, 5)  # Default to highest level if unknown operation
            if agent_level >= required_level:
                return {
                    "success": True,
                    "operation": operation,
                    "required_level": required_level,
                    "agent_level": agent_level,
                    "reason": f"Agent has sufficient permissions (level {agent_level} >= {required_level})",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                self.log_error("permission_denied", f"Agent level {agent_level} insufficient for operation {operation} (required level {required_level})")
                return {
                    "success": False,
                    "operation": operation,
                    "required_level": required_level,
                    "agent_level": agent_level,
                    "reason": f"Agent has insufficient permissions (level {agent_level} < {required_level})",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error in permission check for operation {operation}: {e}")
            self.log_error("permission_check_error", f"Error during permission check: {str(e)}")
            return {
                "success": False,
                "operation": operation,
                "reason": f"Permission check failed due to error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def log_error(self, error_type: str, details: str) -> Dict[str, Any]:
        """
        Log an error or safety violation for monitoring
        
        Args:
            error_type: Type of error or violation
            details: Detailed description of the error
            
        Returns:
            Dictionary with logging result
        """
        try:
            error_entry = {
                "type": error_type,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            self.error_log.append(error_entry)
            logger.error(f"Safety error logged: {error_type} - {details}")
            return {
                "success": True,
                "error_type": error_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error logging safety error {error_type}: {e}")
            return {
                "success": False,
                "error_type": error_type,
                "reason": f"Failed to log error due to: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def trigger_failsafe(self, reason: str) -> Dict[str, Any]:
        """
        Trigger a fail-safe mechanism in response to a critical issue
        
        Args:
            reason: Reason for triggering the fail-safe
            
        Returns:
            Dictionary with fail-safe activation result
        """
        try:
            logger.critical(f"Fail-safe triggered: {reason}")
            self.log_error("failsafe_triggered", reason)
            # Implementation of fail-safe could include halting operations,
            # notifying administrators, or rolling back recent actions
            return {
                "success": True,
                "action": "failsafe_activated",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error triggering fail-safe for reason {reason}: {e}")
            return {
                "success": False,
                "action": "failsafe_failed",
                "reason": f"Failed to trigger fail-safe due to: {str(e)}",
                "original_reason": reason,
                "timestamp": datetime.now().isoformat()
            }
    
    def integrate_with_task_execution(self, task_orchestrator) -> None:
        """
        Integrate safety checks into task execution workflow
        
        Args:
            task_orchestrator: The TaskOrchestrator instance to integrate with
        """
        logger.info("Integrating SafetyModule with TaskOrchestrator")
        # Wrap the execute_next_task method to include safety checks
        original_execute = task_orchestrator.execute_next_task
        
        def safe_execute_next_task():
            prioritized_tasks = task_orchestrator.prioritize_tasks()
            if not prioritized_tasks:
                logger.info("No tasks available for safe execution.")
                return None
            
            for task in prioritized_tasks:
                if task.status == "pending":
                    # Perform safety and ethics checks before execution
                    action = {
                        "type": "task_execution",
                        "target": task.name,
                        "context": task.description
                    }
                    safety_result = self.check_safety(action)
                    if not safety_result["success"]:
                        logger.error(f"Safety check failed for task {task.name}: {safety_result['reason']}")
                        task.status = "blocked_safety"
                        task_orchestrator.agent_state.update_task(task)
                        self.trigger_failsafe(safety_result["reason"])
                        return task.id
                    
                    ethics_result = self.validate_ethics(action)
                    if not ethics_result["success"]:
                        logger.error(f"Ethics validation failed for task {task.name}: {ethics_result['reason']}")
                        task.status = "blocked_ethics"
                        task_orchestrator.agent_state.update_task(task)
                        self.trigger_failsafe(ethics_result["reason"])
                        return task.id
                    
                    # If checks pass, proceed with execution
                    return original_execute()
            return None
        
        task_orchestrator.execute_next_task = safe_execute_next_task
        logger.info("Completed integration of SafetyModule with TaskOrchestrator")