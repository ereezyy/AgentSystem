"""
Logger Configuration
-------------------
Configures logging for the application
Provides different logging levels and formats
"""

import os
import sys
import logging
import datetime
from pathlib import Path
from typing import Optional

# Local imports
from AgentSystem.utils.env_loader import get_env

# Configure default logger
logger = logging.getLogger("AgentSystem")

# Log levels mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logging(
    level: str = "info",
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: Optional[str] = None,
    app_name: str = "agent"
) -> logging.Logger:
    """
    Configure logging for the application
    
    Args:
        level: Log level (debug, info, warning, error, critical)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        log_dir: Directory to store log files
        app_name: Name of the application for log files
        
    Returns:
        Configured logger
    """
    # Get log level
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # Reset existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure logger
    logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if log_to_file:
        # Use provided log directory or get from environment
        logs_dir = log_dir or get_env("LOGS_DIR", "./logs")
        
        # Create logs directory if it doesn't exist
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        
        # Create log filename with current date
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(logs_dir, f"{app_name}_{date_str}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Logging initialized at level {level.upper()}")
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        name: Name of the module
        
    Returns:
        Logger for the module
    """
    if name:
        return logging.getLogger(f"AgentSystem.{name}")
    
    return logger


# Setup default logger
setup_logging(
    level=get_env("LOG_LEVEL", "info"),
    log_to_console=get_env("LOG_TO_CONSOLE", "true").lower() == "true",
    log_to_file=get_env("LOG_TO_FILE", "true").lower() == "true"
)
