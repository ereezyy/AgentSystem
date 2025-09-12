"""
xAI (Grok) Provider for AgentSystem
Provides access to xAI's Grok models
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class XAIConfig:
    """Configuration for xAI provider"""
    api_key: str
    base_url: str = "https://api.x.ai/v1"
    default_model: str = "grok-beta"
    fallback_model: str = "grok-beta"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class XAIProvider:
    """xAI (Grok) AI provider implementation"""
    
    def __init__(self, config: Optional[XAIConfig] = None):
        """Initialize xAI provider"""
        self.config = config or self._load_config()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        })
        
        # Available models
        self.available_models = [
            "grok-beta",
            "grok-vision-beta"
        ]
        
    def _load_config(self) -> XAIConfig:
        """Load configuration from environment"""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
            
        return XAIConfig(
            api_key=api_key,
            default_model=os.getenv("XAI_DEFAULT_MODEL", "grok-beta"),
            fallback_model=os.getenv("XAI_FALLBACK_MODEL", "grok-beta"),
            timeout=int(os.getenv("XAI_TIMEOUT", "30")),
            max_retries=int(os.getenv("XAI_MAX_RETRIES", "3"))
        )
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available xAI models"""
        return [
            {
                "id": "grok-beta",
                "name": "Grok Beta",
                "provider": "xai",
                "context_length": 131072,
                "description": "xAI's flagship conversational AI model"
            },
            {
                "id": "grok-vision-beta",
                "name": "Grok Vision Beta",
                "provider": "xai",
                "context_length": 131072,
                "description": "xAI's multimodal model with vision capabilities"
            }
        ]
    
    def generate_text(self, 
                     prompt: str, 
                     model: Optional[str] = None,
                     max_tokens: int = 1000,
                     temperature: float = 0.7,
                     system_prompt: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """Generate text using xAI API"""
        model = model or self.config.default_model
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request data
        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs
        }
        
        # Try with primary model first, then fallback
        models_to_try = [model]
        if model != self.config.fallback_model and model != "grok-beta":
            models_to_try.append(self.config.fallback_model)
            
        last_error = None
        
        for attempt_model in models_to_try:
            request_data["model"] = attempt_model
            
            for attempt in range(self.config.max_retries):
                try:
                    logger.debug(f"Attempting generation with model {attempt_model}, attempt {attempt + 1}")
                    
                    response = self.session.post(
                        f"{self.config.base_url}/chat/completions",
                        json=request_data,
                        timeout=self.config.timeout
                    )
                    
                    if response.status_code == 429:  # Rate limited
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract generated text
                    if "choices" in result and result["choices"]:
                        generated_text = result["choices"][0]["message"]["content"]
                        
                        return {
                            "text": generated_text,
                            "model": attempt_model,
                            "usage": result.get("usage", {}),
                            "finish_reason": result["choices"][0].get("finish_reason"),
                            "provider": "xai"
                        }
                    else:
                        raise ValueError("No choices in response")
                        
                except requests.exceptions.RequestException as e:
                    last_error = e
                    logger.warning(f"Request failed for model {attempt_model}, attempt {attempt + 1}: {e}")
                    
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        time.sleep(wait_time)
                    continue
                    
                except Exception as e:
                    last_error = e
                    logger.error(f"Unexpected error with model {attempt_model}: {e}")
                    break
        
        # If all attempts failed
        raise Exception(f"All generation attempts failed. Last error: {last_error}")
    
    def analyze_code(self, 
                    code: str, 
                    language: str = "python",
                    analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze code using xAI"""
        system_prompt = f"""You are an expert code analyzer with deep knowledge of {language}. Analyze the provided code and provide:
1. Code quality assessment
2. Potential bugs or issues
3. Security vulnerabilities
4. Performance improvements
5. Best practices recommendations

Focus on {analysis_type} analysis. Be thorough and specific in your recommendations."""

        prompt = f"""Please analyze this {language} code:

```{language}
{code}
```

Provide a detailed analysis with specific recommendations."""

        try:
            result = self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            return {
                "analysis": result["text"],
                "language": language,
                "analysis_type": analysis_type,
                "model": result["model"],
                "provider": "xai"
            }
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            raise
    
    def improve_code(self, 
                    code: str, 
                    language: str = "python",
                    improvement_focus: str = "general") -> Dict[str, Any]:
        """Improve code using xAI"""
        system_prompt = f"""You are an expert {language} developer with years of experience. Improve the provided code by:
1. Fixing bugs and issues
2. Enhancing performance
3. Improving readability and maintainability
4. Following best practices and conventions
5. Adding proper error handling and documentation

Focus on {improvement_focus} improvements. Return only the improved code with clear comments explaining the changes made."""

        prompt = f"""Please improve this {language} code:

```{language}
{code}
```

Return the improved code with explanatory comments about the changes."""

        try:
            result = self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=3000,
                temperature=0.2
            )
            
            return {
                "improved_code": result["text"],
                "original_code": code,
                "language": language,
                "improvement_focus": improvement_focus,
                "model": result["model"],
                "provider": "xai"
            }
            
        except Exception as e:
            logger.error(f"Code improvement failed: {e}")
            raise
    
    def classify_vulnerability(self, 
                             vulnerability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify vulnerability using xAI"""
        system_prompt = """You are a cybersecurity expert with extensive knowledge of vulnerability assessment. Classify the provided vulnerability data and provide:
1. Severity level (Critical, High, Medium, Low) with justification
2. CVSS score estimation with breakdown
3. Attack vector analysis and exploitation scenarios
4. Potential impact assessment on confidentiality, integrity, and availability
5. Detailed remediation recommendations with priority levels"""

        prompt = f"""Please classify this vulnerability:

{json.dumps(vulnerability_data, indent=2)}

Provide a comprehensive classification and assessment with specific technical details."""

        try:
            result = self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.1
            )
            
            return {
                "classification": result["text"],
                "vulnerability_data": vulnerability_data,
                "model": result["model"],
                "provider": "xai"
            }
            
        except Exception as e:
            logger.error(f"Vulnerability classification failed: {e}")
            raise
    
    def generate_phishing_content(self, 
                                 target_info: Dict[str, Any],
                                 campaign_type: str = "email") -> Dict[str, Any]:
        """Generate phishing content for testing (ethical use only)"""
        system_prompt = """You are a cybersecurity professional creating phishing simulation content for authorized security testing and awareness training. 
Generate realistic but clearly marked test content that includes obvious indicators it's a simulation.
Always include clear disclaimers that this is for authorized testing only and educational purposes.
Focus on creating content that helps users learn to identify phishing attempts."""

        prompt = f"""Create a {campaign_type} phishing simulation for authorized security testing with these parameters:

{json.dumps(target_info, indent=2)}

Requirements:
- Include clear "SIMULATION" or "TRAINING" markers
- Add educational content about phishing indicators
- Make it realistic but obviously a test
- Include tips for identifying real phishing attempts"""

        try:
            result = self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000,
                temperature=0.4
            )
            
            return {
                "content": result["text"],
                "campaign_type": campaign_type,
                "target_info": target_info,
                "model": result["model"],
                "provider": "xai",
                "disclaimer": "FOR AUTHORIZED SECURITY TESTING AND TRAINING ONLY"
            }
            
        except Exception as e:
            logger.error(f"Phishing content generation failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check provider health and connectivity"""
        try:
            # Test with a simple generation
            result = self.generate_text(
                prompt="Hello, please respond with 'OK' if you're working correctly.",
                max_tokens=10,
                temperature=0.1
            )
            
            return {
                "status": "healthy",
                "provider": "xai",
                "model": result["model"],
                "response_time": "< 1s",
                "available_models": len(self.get_available_models())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "xai",
                "error": str(e)
            }

# Factory function for easy instantiation
def create_xai_provider(api_key: Optional[str] = None) -> XAIProvider:
    """Create xAI provider instance"""
    if api_key:
        config = XAIConfig(api_key=api_key)
        return XAIProvider(config)
    return XAIProvider()

# Example usage
if __name__ == "__main__":
    # Test the provider
    try:
        provider = create_xai_provider()
        
        # Health check
        health = provider.health_check()
        print(f"Health check: {health}")
        
        # Test generation
        result = provider.generate_text("What is machine learning?", max_tokens=100)
        print(f"Generated text: {result['text'][:200]}...")
        
    except Exception as e:
        print(f"Error testing xAI provider: {e}")