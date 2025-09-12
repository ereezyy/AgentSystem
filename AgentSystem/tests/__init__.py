"""
Test Package for AgentSystem
--------------------------
Contains test modules for verifying functionality of the AgentSystem components.

Usage:
    python -m unittest discover -s tests

Or run individual tests:
    python -m AgentSystem.tests.test_sensory_input
    python -m AgentSystem.tests.test_continuous_learning

Quick tests:
    python -m AgentSystem.tests.test_sensory_input --quick
    python -m AgentSystem.tests.test_continuous_learning --quick
"""

import logging
from AgentSystem.utils.logger import setup_logging

# Configure logging for tests
setup_logging(level="info")
