"""
Local/Open-Source AI Provider for AgentSystem
Handles local model inference as a fallback option
"""

import logging
import os
from typing import Dict, Any, Optional
import subprocess
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class LocalProvider:
    """Provider for local or open-source AI models"""

    def __init__(self, model_path: str = "", model_name: str = "llama3.1"):
        """Initialize the local provider"""
        self.model_path = model_path or os.getenv("LOCAL_MODEL_PATH", "")
        self.model_name = model_name
        self.is_available = False
        self.server_process = None
        self.api_endpoint = os.getenv("LOCAL_API_ENDPOINT", "http://localhost:11434")

        # Check if model path exists
        if self.model_path and os.path.exists(self.model_path):
            self.is_available = True
            logger.info(f"Local model found at {self.model_path}")
        elif self.api_endpoint:
            # Check if API endpoint is reachable
            try:
                import requests
                response = requests.get(f"{self.api_endpoint}/api/tags", timeout=5)
                if response.status_code == 200:
                    self.is_available = True
                    logger.info(f"Local API endpoint available at {self.api_endpoint}")
            except Exception as e:
                logger.warning(f"Local API endpoint check failed: {e}")
        else:
            logger.warning("No local model path or API endpoint configured")

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the local provider"""
        if not self.is_available:
            return {
                "status": "unhealthy",
                "error": "Local provider not configured or model not found",
                "last_check": datetime.now().isoformat()
            }

        try:
            import requests
            response = requests.get(f"{self.api_endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                return {
                    "status": "healthy",
                    "model": self.model_name,
                    "available_models": model_names,
                    "endpoint": self.api_endpoint,
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"API endpoint returned status {response.status_code}",
                    "last_check": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using the local model"""
        if not self.is_available:
            raise Exception("Local provider not available")

        try:
            import requests

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            # Add optional parameters
            if "max_tokens" in kwargs:
                payload["num_predict"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                payload["temperature"] = kwargs["temperature"]

            response = requests.post(
                f"{self.api_endpoint}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get("response", ""),
                    "usage": {
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    },
                    "model": result.get("model", self.model_name),
                    "cost": 0.0  # Local inference has no API cost
                }
            else:
                raise Exception(f"Local API request failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Local text generation failed: {e}")
            raise Exception(f"Local provider failed: {e}")

    def analyze_code(self, code: str, language: str = "python", analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze code using the local model"""
        prompt = f"""Analyze the following {language} code:

{code}

Please provide a {analysis_type} analysis of this code, including:
1. Code structure and organization
2. Potential issues or bugs
3. Suggestions for improvement
"""
        return self.generate_text(prompt)

    def improve_code(self, code: str, language: str = "python", improvement_focus: str = "general") -> Dict[str, Any]:
        """Improve code using the local model"""
        prompt = f"""Improve the following {language} code with a focus on {improvement_focus}:

{code}

Please provide:
1. The improved code
2. An explanation of the changes made
3. Any additional recommendations
"""
        return self.generate_text(prompt)

    def classify_vulnerability(self, vulnerability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a vulnerability using the local model"""
        prompt = f"""Classify the following vulnerability data:

{json.dumps(vulnerability_data, indent=2)}

Please provide:
1. Vulnerability type and severity
2. Potential impact
3. Recommended mitigation steps
"""
        return self.generate_text(prompt)

    def generate_phishing_content(self, target_info: Dict[str, Any], campaign_type: str = "email") -> Dict[str, Any]:
        """Generate phishing content using the local model"""
        prompt = f"""Generate {campaign_type} phishing content based on the following target information:

{json.dumps(target_info, indent=2)}

Please provide:
1. The phishing content
2. Key psychological triggers used
3. Recommendations for delivery timing and method
"""
        return self.generate_text(prompt)

def create_local_provider(model_path: str = "", model_name: str = "llama3.1") -> LocalProvider:
    """Factory function to create a local provider"""
    return LocalProvider(model_path, model_name)
