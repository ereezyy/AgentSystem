"""
Local LLM Provider
----------------
Implementation of AI provider for local LLM models using llama-cpp-python
"""

import time
import json
import os
import threading
from typing import Dict, List, Any, Optional, Union, Callable

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.utils.env_loader import get_env
from AgentSystem.services.ai import AIProvider, AIMessage, AIRequestOptions, AIResponse

# Get module logger
logger = get_logger("services.ai_providers.local")

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    logger.warning("llama-cpp-python package not installed. Local LLM provider will not be available.")
    LLAMA_CPP_AVAILABLE = False


class LocalProvider(AIProvider):
    """Provider for local LLM models using llama-cpp-python"""
    
    def __init__(self):
        """Initialize the Local LLM provider"""
        self._models_dir = get_env("MODELS_DIR", "./models")
        self._models = {}
        self._llm_instances = {}
        self._scan_available_models()
        
        logger.debug("Local LLM provider initialized")
    
    def _scan_available_models(self) -> None:
        """Scan for available local models"""
        if not os.path.exists(self._models_dir):
            logger.warning(f"Models directory {self._models_dir} does not exist")
            return
            
        # Look for .gguf files in the models directory
        for file in os.listdir(self._models_dir):
            if file.endswith(".gguf"):
                model_path = os.path.join(self._models_dir, file)
                model_name = file.replace(".gguf", "")
                
                # Extract model details from filename
                # Expect format like "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
                parts = model_name.split(".")
                quantization = parts[-1] if len(parts) > 1 else "unknown"
                
                # Try to determine model details from name
                is_instruct = "instruct" in model_name.lower()
                
                self._models[model_name] = {
                    "path": model_path,
                    "type": "chat" if is_instruct else "completion",
                    "quantization": quantization,
                    "description": f"Local {model_name} model"
                }
                
                logger.info(f"Found local model: {model_name}")
    
    @property
    def name(self) -> str:
        """Get the provider name"""
        return "local"
    
    @property
    def available_models(self) -> List[str]:
        """Get available models"""
        return list(self._models.keys())
    
    def is_available(self) -> bool:
        """Check if the provider is available"""
        return LLAMA_CPP_AVAILABLE and len(self._models) > 0
    
    def _get_llm_instance(self, model_name: str) -> Optional[Llama]:
        """Get or create a Llama instance for the model"""
        if not LLAMA_CPP_AVAILABLE:
            return None
            
        if model_name not in self._models:
            logger.error(f"Model {model_name} not found in available models")
            return None
            
        # Return existing instance if available
        if model_name in self._llm_instances:
            return self._llm_instances[model_name]
            
        # Create new instance
        try:
            model_path = self._models[model_name]["path"]
            
            # Create Llama instance with reasonable defaults
            # Users can adjust these in their own implementations
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,          # Context window size
                n_batch=512,         # Batch size for prompt processing
                n_gpu_layers=0,      # Default to CPU; can be overridden
                verbose=False        # Silence logging from llama-cpp
            )
            
            self._llm_instances[model_name] = llm
            logger.info(f"Created LLM instance for model {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Error creating LLM instance for model {model_name}: {e}")
            return None
    
    def _format_prompt(self, messages: List[AIMessage]) -> str:
        """Format messages into a prompt for the model"""
        prompt = ""
        
        for msg in messages:
            if msg.role == "system":
                prompt += f"<|system|>\n{msg.content}\n"
            elif msg.role == "user":
                prompt += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"<|assistant|>\n{msg.content}\n"
            elif msg.role == "function":
                # Format function messages as user messages with clear labels
                prompt += f"<|user|>\n[Function Result: {msg.name}]\n{msg.content}\n"
        
        # Add final assistant prefix for generation
        prompt += "<|assistant|>\n"
        
        return prompt
    
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
        if not self.is_available():
            raise ValueError("Local LLM provider not available")
            
        # Get LLM instance
        llm = self._get_llm_instance(model)
        if not llm:
            raise ValueError(f"Failed to load model {model}")
        
        # Use default options if none provided
        if not options:
            options = AIRequestOptions()
        
        # Format messages into prompt
        prompt = self._format_prompt(messages)
        
        # Prepare parameters
        params = {
            "prompt": prompt,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "repeat_penalty": 1.1,  # Helps prevent repetition
            "echo": False,          # Don't include prompt in the response
        }
        
        # Add max_tokens if specified
        if options.max_tokens:
            params["max_tokens"] = options.max_tokens
        else:
            params["max_tokens"] = 2048  # Default to reasonable value
            
        # Add stop sequences if specified
        if options.stop:
            params["stop"] = options.stop
        
        try:
            # Measure start time
            start_time = time.time()
            
            # Call the model
            response = llm(**params)
            
            # Measure end time
            end_time = time.time()
            
            # Extract the content
            content = response.get("choices", [{}])[0].get("text", "").strip()
            
            # Create the response
            ai_response = AIResponse(
                content=content,
                model=model,
                provider=self.name,
                created_at=start_time,
                finish_reason="stop",
                usage={
                    "prompt_tokens": llm.n_tokens(prompt),
                    "completion_tokens": llm.n_tokens(content),
                    "total_tokens": llm.n_tokens(prompt) + llm.n_tokens(content)
                }
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in local LLM completion: {e}")
            raise
    
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
        if not self.is_available():
            raise ValueError("Local LLM provider not available")
            
        # Get LLM instance
        llm = self._get_llm_instance(model)
        if not llm:
            raise ValueError(f"Failed to load model {model}")
        
        # Use default options if none provided
        if not options:
            options = AIRequestOptions()
        
        # Format messages into prompt
        prompt = self._format_prompt(messages)
        
        # Prepare parameters
        params = {
            "prompt": prompt,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "repeat_penalty": 1.1,  # Helps prevent repetition
            "echo": False,          # Don't include prompt in the response
            "stream": True          # Enable streaming
        }
        
        # Add max_tokens if specified
        if options.max_tokens:
            params["max_tokens"] = options.max_tokens
        else:
            params["max_tokens"] = 2048  # Default to reasonable value
            
        # Add stop sequences if specified
        if options.stop:
            params["stop"] = options.stop
        
        try:
            # Measure start time
            start_time = time.time()
            
            # Variables to collect the complete response
            full_content = ""
            
            # Stream from the model
            for chunk in llm.create_completion(**params):
                text = chunk.get("choices", [{}])[0].get("text", "")
                if text:
                    full_content += text
                    callback(text)
            
            # Measure end time
            end_time = time.time()
            
            # Create the final response
            ai_response = AIResponse(
                content=full_content,
                model=model,
                provider=self.name,
                created_at=start_time,
                finish_reason="stop",
                usage={
                    "prompt_tokens": llm.n_tokens(prompt),
                    "completion_tokens": llm.n_tokens(full_content),
                    "total_tokens": llm.n_tokens(prompt) + llm.n_tokens(full_content)
                }
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in local LLM streaming completion: {e}")
            raise
    
    def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text
        
        Note: Most local models don't support embeddings directly through llama-cpp.
        This implementation is a placeholder. For real embeddings, consider
        using a specialized package like sentence-transformers.
        
        Args:
            text: Text to embed
            model: Model to use
            
        Returns:
            Text embeddings
        """
        logger.warning("Local embeddings are not fully supported. Using a placeholder implementation.")
        
        # Convert to list if a single string
        input_texts = [text] if isinstance(text, str) else text
        
        # Create simple placeholder embeddings (1536 dimensions to match OpenAI)
        results = []
        for _ in input_texts:
            # Generate a pseudo-random vector (not useful for actual similarity)
            embedding = [0.0] * 1536
            results.append(embedding)
        
        # Return as a single list if input was a string
        if isinstance(text, str):
            return results[0]
        
        return results
