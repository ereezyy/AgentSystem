"""
Google Gemini AI Provider for AgentSystem
Provides access to Google's Gemini models
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
class GeminiConfig:
    """Configuration for Gemini provider"""
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    default_model: str = "gemini-1.5-pro"
    fallback_model: str = "gemini-1.5-flash"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class GeminiProvider:
    """Google Gemini AI provider implementation"""
    
    def __init__(self, config: Optional[GeminiConfig] = None):
        """Initialize Gemini provider"""
        self.config = config or self._load_config()
        self.session = requests.Session()
        
        # Available models
        self.available_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro"
        ]
        
    def _load_config(self) -> GeminiConfig:
        """Load configuration from environment"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
            
        return GeminiConfig(
            api_key=api_key,
            default_model=os.getenv("GEMINI_DEFAULT_MODEL", "gemini-1.5-pro"),
            fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash"),
            timeout=int(os.getenv("GEMINI_TIMEOUT", "30")),
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        )
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Gemini models"""
        return [
            {
                "id": model,
                "name": model,
                "provider": "google",
                "context_length": 1000000 if "1.5" in model else 30720
            }
            for model in self.available_models
        ]
    
    def _prepare_content(self, text: str) -> List[Dict[str, Any]]:
        """Prepare content for Gemini API format"""
        return [{"text": text}]
    
    def generate_text(self, 
                     prompt: str, 
                     model: Optional[str] = None,
                     max_tokens: int = 1000,
                     temperature: float = 0.7,
                     system_prompt: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """Generate text using Gemini API"""
        model = model or self.config.default_model
        
        # Prepare the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Prepare request data
        request_data = {
            "contents": [
                {
                    "parts": self._prepare_content(full_prompt)
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        # Try with primary model first, then fallback
        models_to_try = [model]
        if model != self.config.fallback_model:
            models_to_try.append(self.config.fallback_model)
            
        last_error = None
        
        for attempt_model in models_to_try:
            for attempt in range(self.config.max_retries):
                try:
                    logger.debug(f"Attempting generation with model {attempt_model}, attempt {attempt + 1}")
                    
                    url = f"{self.config.base_url}/models/{attempt_model}:generateContent"
                    params = {"key": self.config.api_key}
                    
                    response = self.session.post(
                        url,
                        params=params,
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
                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            generated_text = candidate["content"]["parts"][0]["text"]
                            
                            return {
                                "text": generated_text,
                                "model": attempt_model,
                                "usage": result.get("usageMetadata", {}),
                                "finish_reason": candidate.get("finishReason"),
                                "provider": "gemini"
                            }
                    
                    raise ValueError("No valid candidates in response")
                        
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
        """Analyze code using Gemini AI"""
        system_prompt = f"""You are an expert code analyzer. Analyze the provided {language} code and provide:
1. Code quality assessment
2. Potential bugs or issues
3. Security vulnerabilities
4. Performance improvements
5. Best practices recommendations

Focus on {analysis_type} analysis."""

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
                "provider": "gemini"
            }
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            raise
    
    def improve_code(self, 
                    code: str, 
                    language: str = "python",
                    improvement_focus: str = "general") -> Dict[str, Any]:
        """Improve code using Gemini AI"""
        system_prompt = f"""You are an expert {language} developer. Improve the provided code by:
1. Fixing bugs and issues
2. Enhancing performance
3. Improving readability
4. Following best practices
5. Adding proper error handling

Focus on {improvement_focus} improvements. Return only the improved code with comments explaining changes."""

        prompt = f"""Please improve this {language} code:

```{language}
{code}
```

Return the improved code with explanatory comments."""

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
                "provider": "gemini"
            }
            
        except Exception as e:
            logger.error(f"Code improvement failed: {e}")
            raise
    
    def classify_vulnerability(self, 
                             vulnerability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify vulnerability using Gemini AI"""
        system_prompt = """You are a cybersecurity expert. Classify the provided vulnerability data and provide:
1. Severity level (Critical, High, Medium, Low)
2. CVSS score estimation
3. Attack vector analysis
4. Potential impact assessment
5. Remediation recommendations"""

        prompt = f"""Please classify this vulnerability:

{json.dumps(vulnerability_data, indent=2)}

Provide a comprehensive classification and assessment."""

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
                "provider": "gemini"
            }
            
        except Exception as e:
            logger.error(f"Vulnerability classification failed: {e}")
            raise
    
    def generate_phishing_content(self, 
                                 target_info: Dict[str, Any],
                                 campaign_type: str = "email") -> Dict[str, Any]:
        """Generate phishing content for testing (ethical use only)"""
        system_prompt = """You are a cybersecurity professional creating phishing simulation content for authorized security testing. 
Generate realistic but clearly marked test content that includes obvious indicators it's a simulation.
Always include clear disclaimers that this is for authorized testing only."""

        prompt = f"""Create a {campaign_type} phishing simulation for authorized security testing with these parameters:

{json.dumps(target_info, indent=2)}

Include clear "SIMULATION" markers and educational content about phishing indicators."""

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
                "provider": "gemini",
                "disclaimer": "FOR AUTHORIZED SECURITY TESTING ONLY"
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
                "provider": "gemini",
                "model": result["model"],
                "response_time": "< 1s",
                "available_models": len(self.get_available_models())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "gemini",
                "error": str(e)
            }

# Factory function for easy instantiation
def create_gemini_provider(api_key: Optional[str] = None) -> GeminiProvider:
    """Create Gemini provider instance"""
    if api_key:
        config = GeminiConfig(api_key=api_key)
        return GeminiProvider(config)
    return GeminiProvider()

# Example usage
if __name__ == "__main__":
    # Test the provider
    try:
        provider = create_gemini_provider()
        
        # Health check
        health = provider.health_check()
        print(f"Health check: {health}")
        
        # Test generation
        result = provider.generate_text("What is machine learning?", max_tokens=100)
        print(f"Generated text: {result['text'][:200]}...")
        
    except Exception as e:
        print(f"Error testing Gemini provider: {e}")