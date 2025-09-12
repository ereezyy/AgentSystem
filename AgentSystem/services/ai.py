"""
AI Service Manager
-----------------
Manages AI model providers and provides a unified interface for AI operations
"""

import time
import json
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.utils.env_loader import get_env

# Get module logger
logger = get_logger("services.ai")


@dataclass
class AIMessage:
    """Represents a message in an AI conversation"""
    role: str  # 'system', 'user', 'assistant', 'function', etc.
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "role": self.role,
            "content": self.content
        }
        
        if self.name:
            result["name"] = self.name
            
        if self.function_call:
            result["function_call"] = self.function_call
            
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
            
        return result


@dataclass
class AIRequestOptions:
    """Options for an AI request"""
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    timeout: float = 60.0
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {}
        
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
                
        return result


@dataclass
class AIResponse:
    """Response from an AI model"""
    content: str
    model: str
    provider: str
    created_at: float = field(default_factory=time.time)
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    messages: Optional[List[AIMessage]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class AIProvider(ABC):
    """Abstract base class for AI model providers"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the provider name"""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Get available models"""
        pass
    
    @abstractmethod
    def complete(
        self,
        messages: List[AIMessage],
        model: str,
        options: Optional[AIRequestOptions] = None
    ) -> AIResponse:
        """
        Complete a conversation
        
        Args:
            messages: Conversation messages
            model: Model to use
            options: Request options
            
        Returns:
            AI response
        """
        pass
    
    @abstractmethod
    def stream_complete(
        self,
        messages: List[AIMessage],
        model: str,
        callback: Callable[[str], None],
        options: Optional[AIRequestOptions] = None
    ) -> AIResponse:
        """
        Stream a conversation completion
        
        Args:
            messages: Conversation messages
            model: Model to use
            callback: Callback function for each chunk
            options: Request options
            
        Returns:
            Final AI response
        """
        pass
    
    @abstractmethod
    def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text
        
        Args:
            text: Text to embed
            model: Model to use
            
        Returns:
            Text embeddings
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass


class AIServiceManager:
    """
    Manages multiple AI providers and routes requests
    
    Provides fallback mechanisms and load balancing between providers
    """
    
    def __init__(self):
        """Initialize the AI service manager"""
        self.providers: Dict[str, AIProvider] = {}
        self.default_provider: Optional[str] = None
        self.default_models: Dict[str, str] = {}
        self._load_providers()
        
        logger.info(f"AI Service Manager initialized with {len(self.providers)} providers")
    
    def _load_providers(self) -> None:
        """Load available AI providers"""
        # These will be dynamically imported based on available credentials
        
        # OpenAI Provider
        openai_key = get_env("OPENAI_API_KEY")
        if openai_key:
            try:
                from AgentSystem.services.ai_providers.openai_provider import OpenAIProvider
                self.providers["openai"] = OpenAIProvider()
                self.default_provider = "openai"
                self.default_models["chat"] = "gpt-4o"
                self.default_models["embedding"] = "text-embedding-3-small"
                logger.info("Loaded OpenAI provider")
            except ImportError:
                logger.warning("OpenAI provider not available - missing dependencies")
        
        # OpenRouter Provider
        openrouter_key = get_env("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                from AgentSystem.services.ai_providers.openrouter_provider import OpenRouterProvider
                self.providers["openrouter"] = OpenRouterProvider()
                if not self.default_provider:
                    self.default_provider = "openrouter"
                    self.default_models["chat"] = "openai/gpt-4o"
                logger.info("Loaded OpenRouter provider")
            except ImportError:
                logger.warning("OpenRouter provider not available - missing dependencies")
        
        # Google (Gemini) Provider
        gemini_key = get_env("GEMINI_API_KEY")
        if gemini_key:
            try:
                from AgentSystem.services.ai_providers.gemini_provider import GeminiProvider
                self.providers["gemini"] = GeminiProvider()
                if not self.default_provider:
                    self.default_provider = "gemini"
                    self.default_models["chat"] = "gemini-1.5-pro"
                logger.info("Loaded Gemini provider")
            except ImportError:
                logger.warning("Gemini provider not available - missing dependencies")
        
        # HuggingFace Provider
        hf_token = get_env("HUGGINGFACE_TOKEN")
        if hf_token:
            try:
                from AgentSystem.services.ai_providers.huggingface_provider import HuggingFaceProvider
                self.providers["huggingface"] = HuggingFaceProvider()
                logger.info("Loaded HuggingFace provider")
            except ImportError:
                logger.warning("HuggingFace provider not available - missing dependencies")
        
        # Anthropic Provider
        anthropic_key = get_env("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                from AgentSystem.services.ai_providers.anthropic_provider import AnthropicProvider
                self.providers["anthropic"] = AnthropicProvider()
                logger.info("Loaded Anthropic provider")
            except ImportError:
                logger.warning("Anthropic provider not available - missing dependencies")
        
        # Check if any providers were loaded
        if not self.providers:
            logger.error("No AI providers available - system will have limited functionality")
    
    def register_provider(self, provider: AIProvider) -> None:
        """
        Register a new provider
        
        Args:
            provider: Provider to register
        """
        self.providers[provider.name] = provider
        logger.info(f"Registered provider '{provider.name}'")
        
        # Set as default if no default exists
        if not self.default_provider:
            self.default_provider = provider.name
            logger.info(f"Set '{provider.name}' as default provider")
    
    def get_provider(self, name: Optional[str] = None) -> Optional[AIProvider]:
        """
        Get a provider by name
        
        Args:
            name: Provider name (or None for default)
            
        Returns:
            Provider instance
        """
        if not name:
            name = self.default_provider
            
        if not name or name not in self.providers:
            logger.warning(f"Provider '{name}' not found")
            return None
            
        return self.providers[name]
    
    def list_providers(self) -> List[str]:
        """
        Get list of available providers
        
        Returns:
            List of provider names
        """
        return list(self.providers.keys())
    
    def list_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get list of available models
        
        Args:
            provider: Provider name (or None for all)
            
        Returns:
            Dictionary of provider:models
        """
        if provider:
            p = self.get_provider(provider)
            if not p:
                return {}
            return {provider: p.available_models}
        
        return {name: p.available_models for name, p in self.providers.items()}
    
    def complete(
        self,
        messages: List[Union[Dict[str, Any], AIMessage]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        options: Optional[Union[Dict[str, Any], AIRequestOptions]] = None,
        fallback: bool = True
    ) -> AIResponse:
        """
        Complete a conversation
        
        Args:
            messages: Conversation messages
            model: Model to use
            provider: Provider to use
            options: Request options
            fallback: Whether to try fallback providers
            
        Returns:
            AI response
        """
        # Convert messages if they're dictionaries
        ai_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                ai_messages.append(AIMessage(**msg))
            else:
                ai_messages.append(msg)
        
        # Convert options if it's a dictionary
        ai_options = None
        if options:
            if isinstance(options, dict):
                ai_options = AIRequestOptions(**options)
            else:
                ai_options = options
        else:
            ai_options = AIRequestOptions()
        
        # Get provider
        p = self.get_provider(provider)
        if not p:
            if not fallback:
                raise ValueError(f"Provider '{provider}' not available")
            
            # Try to find any available provider
            for name, provider_instance in self.providers.items():
                if provider_instance.is_available():
                    p = provider_instance
                    logger.info(f"Using fallback provider '{name}'")
                    break
            
            if not p:
                raise ValueError("No AI providers available")
        
        # Get model
        if not model:
            model = self.default_models.get("chat")
            if not model:
                # Try to get the first available model from the provider
                models = p.available_models
                if models:
                    model = models[0]
                else:
                    raise ValueError("No models available")
        
        # Try to complete with the selected provider
        try:
            return p.complete(ai_messages, model, ai_options)
        except Exception as e:
            logger.error(f"Error with provider '{p.name}': {e}")
            
            if not fallback:
                raise
            
            # Try fallback providers
            for name, provider_instance in self.providers.items():
                if name != p.name and provider_instance.is_available():
                    try:
                        logger.info(f"Trying fallback provider '{name}'")
                        return provider_instance.complete(ai_messages, model, ai_options)
                    except Exception as e2:
                        logger.error(f"Error with fallback provider '{name}': {e2}")
            
            # If we get here, all providers failed
            raise ValueError("All AI providers failed")
    
    def stream_complete(
        self,
        messages: List[Union[Dict[str, Any], AIMessage]],
        callback: Callable[[str], None],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        options: Optional[Union[Dict[str, Any], AIRequestOptions]] = None,
        fallback: bool = True
    ) -> AIResponse:
        """
        Stream a conversation completion
        
        Args:
            messages: Conversation messages
            callback: Callback function for each chunk
            model: Model to use
            provider: Provider to use
            options: Request options
            fallback: Whether to try fallback providers
            
        Returns:
            Final AI response
        """
        # Convert messages if they're dictionaries
        ai_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                ai_messages.append(AIMessage(**msg))
            else:
                ai_messages.append(msg)
        
        # Convert options if it's a dictionary
        ai_options = None
        if options:
            if isinstance(options, dict):
                ai_options = AIRequestOptions(**options)
            else:
                ai_options = options
        else:
            ai_options = AIRequestOptions()
        
        # Ensure streaming is enabled
        ai_options.stream = True
        
        # Get provider
        p = self.get_provider(provider)
        if not p:
            if not fallback:
                raise ValueError(f"Provider '{provider}' not available")
            
            # Try to find any available provider
            for name, provider_instance in self.providers.items():
                if provider_instance.is_available():
                    p = provider_instance
                    logger.info(f"Using fallback provider '{name}'")
                    break
            
            if not p:
                raise ValueError("No AI providers available")
        
        # Get model
        if not model:
            model = self.default_models.get("chat")
            if not model:
                # Try to get the first available model from the provider
                models = p.available_models
                if models:
                    model = models[0]
                else:
                    raise ValueError("No models available")
        
        # Try to stream with the selected provider
        try:
            return p.stream_complete(ai_messages, model, callback, ai_options)
        except Exception as e:
            logger.error(f"Error with provider '{p.name}': {e}")
            
            if not fallback:
                raise
            
            # Try fallback providers
            for name, provider_instance in self.providers.items():
                if name != p.name and provider_instance.is_available():
                    try:
                        logger.info(f"Trying fallback provider '{name}'")
                        return provider_instance.stream_complete(ai_messages, model, callback, ai_options)
                    except Exception as e2:
                        logger.error(f"Error with fallback provider '{name}': {e2}")
            
            # If we get here, all providers failed
            raise ValueError("All AI providers failed")
    
    def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        fallback: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text
        
        Args:
            text: Text to embed
            model: Model to use
            provider: Provider to use
            fallback: Whether to try fallback providers
            
        Returns:
            Text embeddings
        """
        # Get provider
        p = self.get_provider(provider)
        if not p:
            if not fallback:
                raise ValueError(f"Provider '{provider}' not available")
            
            # Try to find any available provider
            for name, provider_instance in self.providers.items():
                if provider_instance.is_available():
                    p = provider_instance
                    logger.info(f"Using fallback provider '{name}'")
                    break
            
            if not p:
                raise ValueError("No AI providers available")
        
        # Get model
        if not model:
            model = self.default_models.get("embedding")
        
        # Try to embed with the selected provider
        try:
            return p.embed(text, model)
        except Exception as e:
            logger.error(f"Error with provider '{p.name}': {e}")
            
            if not fallback:
                raise
            
            # Try fallback providers
            for name, provider_instance in self.providers.items():
                if name != p.name and provider_instance.is_available():
                    try:
                        logger.info(f"Trying fallback provider '{name}'")
                        return provider_instance.embed(text, model)
                    except Exception as e2:
                        logger.error(f"Error with fallback provider '{name}': {e2}")
            
            # If we get here, all providers failed
            raise ValueError("All AI providers failed")


# Create a singleton instance for global use
ai_service = AIServiceManager()
