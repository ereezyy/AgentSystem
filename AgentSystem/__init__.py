"""
AgentSystem
-----------
An autonomous agent system that can perform various tasks
including coding, web browsing, email management, and self-improvement.

This package provides the foundation for creating AI-powered agents
that can interact with the world through various interfaces.
"""

__version__ = "0.1.0"
__author__ = "Agent Developer"

# Import key components for easy access
from AgentSystem.utils.env_loader import get_env
from AgentSystem.utils.logger import get_logger

# Setup logger
logger = get_logger(__name__)

# Log startup
logger.info(f"AgentSystem v{__version__} initializing")
