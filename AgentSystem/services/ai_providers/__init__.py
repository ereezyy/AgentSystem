"""
AI Providers Package for AgentSystem
Provides unified access to multiple AI providers with fallback capabilities
"""

from .provider_manager import (
    AIProviderManager,
    ProviderType,
    ProviderConfig,
    get_provider_manager,
    reset_provider_manager
)

from .openrouter_provider import (
    OpenRouterProvider,
    OpenRouterConfig,
    create_openrouter_provider
)

from .gemini_provider import (
    GeminiProvider,
    GeminiConfig,
    create_gemini_provider
)

from .xai_provider import (
    XAIProvider,
    XAIConfig,
    create_xai_provider
)

__all__ = [
    # Provider Manager
    'AIProviderManager',
    'ProviderType',
    'ProviderConfig',
    'get_provider_manager',
    'reset_provider_manager',
    
    # OpenRouter
    'OpenRouterProvider',
    'OpenRouterConfig',
    'create_openrouter_provider',
    
    # Gemini
    'GeminiProvider',
    'GeminiConfig',
    'create_gemini_provider',
    
    # xAI
    'XAIProvider',
    'XAIConfig',
    'create_xai_provider',
]

# Version information
__version__ = "1.0.0"
__author__ = "AgentSystem Team"
__description__ = "Multi-provider AI service integration for AgentSystem"
