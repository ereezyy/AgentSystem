"""
Environment Variable Loader
---------------------------
Securely loads environment variables from .env file
Provides fallback mechanisms and validation
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class EnvLoader:
    """Loads and validates environment variables from .env file"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize the environment loader
        
        Args:
            env_file: Path to .env file (defaults to config/.env)
        """
        self.env_path = env_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', '.env')
        self._load_env_file()
        self.variables = {}
        
    def _load_env_file(self) -> None:
        """Load the environment file"""
        env_path = Path(self.env_path)
        if not env_path.exists():
            logger.warning(f"Environment file not found at {self.env_path}")
            return
        
        load_dotenv(dotenv_path=self.env_path)
        logger.info(f"Loaded environment from {self.env_path}")
        
    def get(self, key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get an environment variable
        
        Args:
            key: The environment variable key
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            The environment variable value or default
        
        Raises:
            ValueError: If the variable is required but not found
        """
        value = os.environ.get(key, default)
        
        # Check if the variable is required but not found
        if required and value is None:
            logger.error(f"Required environment variable {key} not found")
            raise ValueError(f"Required environment variable {key} not found")
            
        # Cache the result
        self.variables[key] = value
        
        return value
    
    def get_with_fallback(self, keys: list, default: Any = None) -> Any:
        """
        Try multiple keys and return the first one that exists
        
        Args:
            keys: List of keys to try
            default: Default value if none found
            
        Returns:
            The first environment variable value found or default
        """
        for key in keys:
            value = self.get(key)
            if value is not None:
                return value
        
        return default
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all environment variables that have been accessed
        
        Returns:
            Dictionary of all accessed environment variables
        """
        return self.variables.copy()
    
    def validate_required_vars(self, required_vars: list) -> bool:
        """
        Validate that all required variables are present
        
        Args:
            required_vars: List of required variable keys
            
        Returns:
            True if all required variables are present, False otherwise
        """
        missing = []
        for var in required_vars:
            try:
                self.get(var, required=True)
            except ValueError:
                missing.append(var)
        
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            return False
            
        return True


# Create a singleton instance for global use
env = EnvLoader()

def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Convenience function to get an environment variable
    
    Args:
        key: The environment variable key
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        The environment variable value or default
    """
    return env.get(key, default, required)
