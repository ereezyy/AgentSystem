"""
AI Provider Manager for AgentSystem
Manages multiple AI providers with fallback capabilities
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Type
from dataclasses import dataclass
from enum import Enum
import time

from .openrouter_provider import OpenRouterProvider, create_openrouter_provider
from .gemini_provider import GeminiProvider, create_gemini_provider
from .xai_provider import XAIProvider, create_xai_provider

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Available AI provider types"""
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    XAI = "xai"
    OPENAI = "openai"  # For future implementation

@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    provider_type: ProviderType
    api_key: str
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 30

class AIProviderManager:
    """Manages multiple AI providers with automatic fallback"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the provider manager"""
        self.providers: Dict[ProviderType, Any] = {}
        self.provider_configs: Dict[ProviderType, ProviderConfig] = {}
        self.provider_health: Dict[ProviderType, Dict[str, Any]] = {}
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        
        # Load configuration
        self._load_configuration(config_file)
        
        # Initialize providers
        self._initialize_providers()
    
    def _load_configuration(self, config_file: Optional[str] = None):
        """Load provider configuration from environment or file"""
        # Load from environment variables
        configs = []
        
        # OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            configs.append(ProviderConfig(
                provider_type=ProviderType.OPENROUTER,
                api_key=openrouter_key,
                priority=int(os.getenv("OPENROUTER_PRIORITY", "1")),
                enabled=os.getenv("OPENROUTER_ENABLED", "true").lower() == "true"
            ))
        
        # Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            configs.append(ProviderConfig(
                provider_type=ProviderType.GEMINI,
                api_key=gemini_key,
                priority=int(os.getenv("GEMINI_PRIORITY", "2")),
                enabled=os.getenv("GEMINI_ENABLED", "true").lower() == "true"
            ))
        
        # xAI
        xai_key = os.getenv("XAI_API_KEY")
        if xai_key:
            configs.append(ProviderConfig(
                provider_type=ProviderType.XAI,
                api_key=xai_key,
                priority=int(os.getenv("XAI_PRIORITY", "3")),
                enabled=os.getenv("XAI_ENABLED", "true").lower() == "true"
            ))
        
        # Store configurations
        for config in configs:
            self.provider_configs[config.provider_type] = config
        
        # Sort by priority
        self.provider_configs = dict(
            sorted(self.provider_configs.items(), key=lambda x: x[1].priority)
        )
        
        logger.info(f"Loaded {len(self.provider_configs)} provider configurations")
    
    def _initialize_providers(self):
        """Initialize all configured providers"""
        for provider_type, config in self.provider_configs.items():
            if not config.enabled:
                continue
                
            try:
                if provider_type == ProviderType.OPENROUTER:
                    self.providers[provider_type] = create_openrouter_provider(config.api_key)
                elif provider_type == ProviderType.GEMINI:
                    self.providers[provider_type] = create_gemini_provider(config.api_key)
                elif provider_type == ProviderType.XAI:
                    self.providers[provider_type] = create_xai_provider(config.api_key)
                
                logger.info(f"Initialized {provider_type.value} provider")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_type.value} provider: {e}")
                config.enabled = False
    
    def get_available_providers(self) -> List[ProviderType]:
        """Get list of available and enabled providers"""
        return [
            provider_type for provider_type, config in self.provider_configs.items()
            if config.enabled and provider_type in self.providers
        ]
    
    def get_provider_by_type(self, provider_type: ProviderType) -> Optional[Any]:
        """Get a specific provider by type"""
        return self.providers.get(provider_type)
    
    def check_provider_health(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Check health of a specific provider"""
        if provider_type not in self.providers:
            return {"status": "unavailable", "error": "Provider not initialized"}
        
        try:
            provider = self.providers[provider_type]
            health = provider.health_check()
            self.provider_health[provider_type] = health
            return health
        except Exception as e:
            health = {"status": "unhealthy", "error": str(e)}
            self.provider_health[provider_type] = health
            return health
    
    def check_all_providers_health(self) -> Dict[ProviderType, Dict[str, Any]]:
        """Check health of all providers"""
        current_time = time.time()
        
        # Only check if enough time has passed
        if current_time - self.last_health_check < self.health_check_interval:
            return self.provider_health
        
        for provider_type in self.get_available_providers():
            self.check_provider_health(provider_type)
        
        self.last_health_check = current_time
        return self.provider_health
    
    def get_healthy_providers(self) -> List[ProviderType]:
        """Get list of healthy providers in priority order"""
        self.check_all_providers_health()
        
        healthy_providers = []
        for provider_type in self.get_available_providers():
            health = self.provider_health.get(provider_type, {})
            if health.get("status") == "healthy":
                healthy_providers.append(provider_type)
        
        return healthy_providers
    
    def generate_text(self, 
                     prompt: str,
                     preferred_provider: Optional[ProviderType] = None,
                     **kwargs) -> Dict[str, Any]:
        """Generate text using the best available provider"""
        # Get providers to try
        providers_to_try = []
        
        # Add preferred provider first if specified and healthy
        if preferred_provider and preferred_provider in self.get_healthy_providers():
            providers_to_try.append(preferred_provider)
        
        # Add other healthy providers
        for provider_type in self.get_healthy_providers():
            if provider_type not in providers_to_try:
                providers_to_try.append(provider_type)
        
        if not providers_to_try:
            raise Exception("No healthy AI providers available")
        
        last_error = None
        
        for provider_type in providers_to_try:
            try:
                provider = self.providers[provider_type]
                logger.debug(f"Attempting text generation with {provider_type.value}")
                
                result = provider.generate_text(prompt, **kwargs)
                result["provider_used"] = provider_type.value
                
                logger.info(f"Successfully generated text using {provider_type.value}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Text generation failed with {provider_type.value}: {e}")
                
                # Mark provider as potentially unhealthy
                self.provider_health[provider_type] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": time.time()
                }
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def analyze_code(self, 
                    code: str,
                    language: str = "python",
                    analysis_type: str = "general",
                    preferred_provider: Optional[ProviderType] = None) -> Dict[str, Any]:
        """Analyze code using the best available provider"""
        providers_to_try = []
        
        if preferred_provider and preferred_provider in self.get_healthy_providers():
            providers_to_try.append(preferred_provider)
        
        for provider_type in self.get_healthy_providers():
            if provider_type not in providers_to_try:
                providers_to_try.append(provider_type)
        
        if not providers_to_try:
            raise Exception("No healthy AI providers available")
        
        last_error = None
        
        for provider_type in providers_to_try:
            try:
                provider = self.providers[provider_type]
                logger.debug(f"Attempting code analysis with {provider_type.value}")
                
                result = provider.analyze_code(code, language, analysis_type)
                result["provider_used"] = provider_type.value
                
                logger.info(f"Successfully analyzed code using {provider_type.value}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Code analysis failed with {provider_type.value}: {e}")
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def improve_code(self, 
                    code: str,
                    language: str = "python",
                    improvement_focus: str = "general",
                    preferred_provider: Optional[ProviderType] = None) -> Dict[str, Any]:
        """Improve code using the best available provider"""
        providers_to_try = []
        
        if preferred_provider and preferred_provider in self.get_healthy_providers():
            providers_to_try.append(preferred_provider)
        
        for provider_type in self.get_healthy_providers():
            if provider_type not in providers_to_try:
                providers_to_try.append(provider_type)
        
        if not providers_to_try:
            raise Exception("No healthy AI providers available")
        
        last_error = None
        
        for provider_type in providers_to_try:
            try:
                provider = self.providers[provider_type]
                logger.debug(f"Attempting code improvement with {provider_type.value}")
                
                result = provider.improve_code(code, language, improvement_focus)
                result["provider_used"] = provider_type.value
                
                logger.info(f"Successfully improved code using {provider_type.value}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Code improvement failed with {provider_type.value}: {e}")
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def classify_vulnerability(self, 
                             vulnerability_data: Dict[str, Any],
                             preferred_provider: Optional[ProviderType] = None) -> Dict[str, Any]:
        """Classify vulnerability using the best available provider"""
        providers_to_try = []
        
        if preferred_provider and preferred_provider in self.get_healthy_providers():
            providers_to_try.append(preferred_provider)
        
        for provider_type in self.get_healthy_providers():
            if provider_type not in providers_to_try:
                providers_to_try.append(provider_type)
        
        if not providers_to_try:
            raise Exception("No healthy AI providers available")
        
        last_error = None
        
        for provider_type in providers_to_try:
            try:
                provider = self.providers[provider_type]
                logger.debug(f"Attempting vulnerability classification with {provider_type.value}")
                
                result = provider.classify_vulnerability(vulnerability_data)
                result["provider_used"] = provider_type.value
                
                logger.info(f"Successfully classified vulnerability using {provider_type.value}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Vulnerability classification failed with {provider_type.value}: {e}")
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def generate_phishing_content(self, 
                                 target_info: Dict[str, Any],
                                 campaign_type: str = "email",
                                 preferred_provider: Optional[ProviderType] = None) -> Dict[str, Any]:
        """Generate phishing content using the best available provider"""
        providers_to_try = []
        
        if preferred_provider and preferred_provider in self.get_healthy_providers():
            providers_to_try.append(preferred_provider)
        
        for provider_type in self.get_healthy_providers():
            if provider_type not in providers_to_try:
                providers_to_try.append(provider_type)
        
        if not providers_to_try:
            raise Exception("No healthy AI providers available")
        
        last_error = None
        
        for provider_type in providers_to_try:
            try:
                provider = self.providers[provider_type]
                logger.debug(f"Attempting phishing content generation with {provider_type.value}")
                
                result = provider.generate_phishing_content(target_info, campaign_type)
                result["provider_used"] = provider_type.value
                
                logger.info(f"Successfully generated phishing content using {provider_type.value}")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Phishing content generation failed with {provider_type.value}: {e}")
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of all providers"""
        self.check_all_providers_health()
        
        report = {
            "total_providers": len(self.provider_configs),
            "enabled_providers": len(self.get_available_providers()),
            "healthy_providers": len(self.get_healthy_providers()),
            "provider_details": {}
        }
        
        for provider_type, config in self.provider_configs.items():
            health = self.provider_health.get(provider_type, {"status": "unknown"})
            
            report["provider_details"][provider_type.value] = {
                "enabled": config.enabled,
                "priority": config.priority,
                "health": health,
                "initialized": provider_type in self.providers
            }
        
        return report

# Global instance
_provider_manager = None

def get_provider_manager() -> AIProviderManager:
    """Get the global provider manager instance"""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = AIProviderManager()
    return _provider_manager

def reset_provider_manager():
    """Reset the global provider manager (useful for testing)"""
    global _provider_manager
    _provider_manager = None

# Example usage
if __name__ == "__main__":
    # Test the provider manager
    try:
        manager = AIProviderManager()
        
        # Get status report
        status = manager.get_status_report()
        print(f"Provider status: {status}")
        
        # Test text generation
        if manager.get_healthy_providers():
            result = manager.generate_text("What is artificial intelligence?", max_tokens=100)
            print(f"Generated text: {result['text'][:200]}...")
            print(f"Used provider: {result['provider_used']}")
        else:
            print("No healthy providers available for testing")
            
    except Exception as e:
        print(f"Error testing provider manager: {e}")