"""
Modules Package
--------------
Contains optional modules that extend agent capabilities
"""

__all__ = [
    'AgentForgeRegistry',
    'AgentForgeSDK',
    'KnowledgeExchange',
    'ModuleDescriptor',
    'BrowserModule',
    'CodeEditor',
    'ContinuousLearningModule',
    'EmailModule',
    'KnowledgeGraphModule',
    'SensoryInputModule',
    'SystemInterfaceModule'
]


def __getattr__(name):  # pragma: no cover - thin import helper
    if name == 'AgentForgeRegistry':
        from .agent_forge import AgentForgeRegistry
        return AgentForgeRegistry
    if name == 'AgentForgeSDK':
        from .agent_forge import AgentForgeSDK
        return AgentForgeSDK
    if name == 'KnowledgeExchange':
        from .agent_forge import KnowledgeExchange
        return KnowledgeExchange
    if name == 'ModuleDescriptor':
        from .agent_forge import ModuleDescriptor
        return ModuleDescriptor
    if name == 'BrowserModule':
        from .browser import BrowserModule
        return BrowserModule
    if name == 'CodeEditor':
        from .code_editor import CodeEditor
        return CodeEditor
    if name == 'ContinuousLearningModule':
        from .continuous_learning import ContinuousLearningModule
        return ContinuousLearningModule
    if name == 'EmailModule':
        from .email import EmailModule
        return EmailModule
    if name == 'KnowledgeGraphModule':
        from .knowledge_graph import KnowledgeGraphModule
        return KnowledgeGraphModule
    if name == 'SensoryInputModule':
        from .sensory_input import SensoryInputModule
        return SensoryInputModule
    if name == 'SystemInterfaceModule':
        from .system_interface import SystemInterfaceModule
        return SystemInterfaceModule
    raise AttributeError(name)
