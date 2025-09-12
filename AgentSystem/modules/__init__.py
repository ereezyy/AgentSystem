"""
Modules Package
--------------
Contains optional modules that extend agent capabilities
"""

from .browser import BrowserModule
from .code_editor import CodeEditor
from .continuous_learning import ContinuousLearningModule
from .email import EmailModule
from .knowledge_graph import KnowledgeGraphModule
from .sensory_input import SensoryInputModule
from .system_interface import SystemInterfaceModule

__all__ = [
    'BrowserModule',
    'CodeEditor',
    'ContinuousLearningModule',
    'EmailModule',
    'KnowledgeGraphModule',
    'SensoryInputModule',
    'SystemInterfaceModule'
]
