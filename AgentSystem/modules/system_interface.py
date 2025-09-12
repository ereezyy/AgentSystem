"""
System Interface Module
----------------------
Provides low-level system access and control for the autonomous agent.
This module enables interaction with the operating system, file systems,
hardware, and application control through secure protocols.
"""

import os
import subprocess
import shutil
import platform
from typing import Dict, Any, Optional, List
from datetime import datetime

# Local imports
from AgentSystem.utils.logger import get_logger

# Get module logger
logger = get_logger("modules.system_interface")


class SystemInterfaceModule:
    """Module for system access and control"""
    
    def __init__(self):
        """Initialize the system interface module"""
        self.system_info = self._get_system_info()
        logger.info("Initialized SystemInterfaceModule")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """
        Gather basic system information
        
        Returns:
            Dictionary containing system information
        """
        try:
            return {
                "os_name": platform.system(),
                "os_version": platform.version(),
                "os_release": platform.release(),
                "architecture": platform.architecture()[0],
                "machine": platform.machine(),
                "processor": platform.processor(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error gathering system info: {e}")
            return {"error": str(e)}
    
    def get_tools(self) -> Dict[str, Any]:
        """
        Get tools provided by this module
        
        Returns:
            Dictionary of available tools with descriptions and parameters
        """
        return {
            "get_system_info": {
                "description": "Get detailed information about the system",
                "function": self.get_system_info,
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            "execute_command": {
                "description": "Execute a system command and return the output",
                "function": self.execute_command,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The system command to execute"
                        },
                        "shell": {
                            "type": "boolean",
                            "description": "Whether to use shell to execute the command",
                            "default": False
                        }
                    },
                    "required": ["command"]
                }
            },
            "read_file": {
                "description": "Read the contents of a file",
                "function": self.read_file,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding",
                            "default": "utf-8"
                        }
                    },
                    "required": ["path"]
                }
            },
            "write_file": {
                "description": "Write content to a file",
                "function": self.write_file,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        },
                        "encoding": {
                            "type": "string",
                            "description": "File encoding",
                            "default": "utf-8"
                        },
                        "append": {
                            "type": "boolean",
                            "description": "Whether to append to the file instead of overwriting",
                            "default": False
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            "list_directory": {
                "description": "List contents of a directory",
                "function": self.list_directory,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory to list",
                            "default": "."
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Whether to list contents recursively",
                            "default": False
                        }
                    }
                }
            },
            "create_directory": {
                "description": "Create a new directory",
                "function": self.create_directory,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path for the new directory"
                        },
                        "parents": {
                            "type": "boolean",
                            "description": "Whether to create parent directories if they don't exist",
                            "default": True
                        }
                    },
                    "required": ["path"]
                }
            },
            "delete_file_or_directory": {
                "description": "Delete a file or directory",
                "function": self.delete_file_or_directory,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file or directory to delete"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Whether to delete directories recursively",
                            "default": False
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the system
        
        Returns:
            Dictionary with system information
        """
        return {
            "success": True,
            "info": self.system_info
        }
    
    def execute_command(self, command: str, shell: bool = False) -> Dict[str, Any]:
        """
        Execute a system command and return the output
        
        Args:
            command: The system command to execute
            shell: Whether to use shell to execute the command
            
        Returns:
            Dictionary with execution result
        """
        try:
            # Security note: Using shell=True can be dangerous with untrusted input
            if shell:
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout to prevent hanging
                )
            else:
                # Split command into list if not using shell
                if isinstance(command, str):
                    command_list = command.split()
                else:
                    command_list = command
                    
                process = subprocess.run(
                    command_list,
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
            return {
                "success": True,
                "command": command,
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "timestamp": datetime.now().isoformat()
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return {
                "success": False,
                "error": "Command timed out after 30 seconds",
                "command": command
            }
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    def read_file(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Read the contents of a file
        
        Args:
            path: Path to the file to read
            encoding: File encoding
            
        Returns:
            Dictionary with file contents or error
        """
        try:
            # Security note: Ensure path is within allowed directories
            abs_path = os.path.abspath(path)
            with open(abs_path, 'r', encoding=encoding) as f:
                content = f.read()
                
            return {
                "success": True,
                "path": path,
                "content": content,
                "encoding": encoding,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error reading file '{path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "path": path
            }
    
    def write_file(self, path: str, content: str, encoding: str = "utf-8", append: bool = False) -> Dict[str, Any]:
        """
        Write content to a file
        
        Args:
            path: Path to the file to write
            content: Content to write to the file
            encoding: File encoding
            append: Whether to append to the file instead of overwriting
            
        Returns:
            Dictionary with result information
        """
        try:
            # Security note: Ensure path is within allowed directories
            abs_path = os.path.abspath(path)
            mode = 'a' if append else 'w'
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            
            with open(abs_path, mode, encoding=encoding) as f:
                f.write(content)
                
            return {
                "success": True,
                "path": path,
                "mode": "append" if append else "write",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error writing to file '{path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "path": path
            }
    
    def list_directory(self, path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        """
        List contents of a directory
        
        Args:
            path: Path to the directory to list
            recursive: Whether to list contents recursively
            
        Returns:
            Dictionary with directory contents
        """
        try:
            # Security note: Ensure path is within allowed directories
            abs_path = os.path.abspath(path)
            contents = []
            
            if recursive:
                for root, dirs, files in os.walk(abs_path):
                    for d in dirs:
                        contents.append({
                            "type": "directory",
                            "name": d,
                            "path": os.path.relpath(os.path.join(root, d), abs_path),
                            "full_path": os.path.join(root, d)
                        })
                    for f in files:
                        contents.append({
                            "type": "file",
                            "name": f,
                            "path": os.path.relpath(os.path.join(root, f), abs_path),
                            "full_path": os.path.join(root, f)
                        })
            else:
                with os.scandir(abs_path) as entries:
                    for entry in entries:
                        contents.append({
                            "type": "directory" if entry.is_dir() else "file",
                            "name": entry.name,
                            "path": entry.name,
                            "full_path": entry.path
                        })
                        
            return {
                "success": True,
                "path": path,
                "contents": contents,
                "count": len(contents),
                "recursive": recursive,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error listing directory '{path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "path": path
            }
    
    def create_directory(self, path: str, parents: bool = True) -> Dict[str, Any]:
        """
        Create a new directory
        
        Args:
            path: Path for the new directory
            parents: Whether to create parent directories if they don't exist
            
        Returns:
            Dictionary with result information
        """
        try:
            # Security note: Ensure path is within allowed directories
            abs_path = os.path.abspath(path)
            os.makedirs(abs_path, exist_ok=True)
            
            return {
                "success": True,
                "path": path,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating directory '{path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "path": path
            }
    
    def delete_file_or_directory(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Delete a file or directory
        
        Args:
            path: Path to the file or directory to delete
            recursive: Whether to delete directories recursively
            
        Returns:
            Dictionary with result information
        """
        try:
            # Security note: Ensure path is within allowed directories
            abs_path = os.path.abspath(path)
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": "Path does not exist",
                    "path": path
                }
                
            if os.path.isfile(abs_path):
                os.remove(abs_path)
            elif os.path.isdir(abs_path):
                if recursive:
                    shutil.rmtree(abs_path)
                else:
                    os.rmdir(abs_path)  # Will fail if directory is not empty
                    
            return {
                "success": True,
                "path": path,
                "type": "directory" if os.path.isdir(abs_path) else "file",
                "recursive": recursive,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error deleting '{path}': {e}")
            return {
                "success": False,
                "error": str(e),
                "path": path
            }