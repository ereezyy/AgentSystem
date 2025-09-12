"""
OpenAI Provider
--------------
Implementation of AI provider for OpenAI API
"""

import time
import json
from typing import Dict, List, Any, Optional, Union, Callable

# Local imports
from AgentSystem.utils.logger import get_logger
from AgentSystem.utils.env_loader import get_env
from AgentSystem.services.ai import AIProvider, AIMessage, AIRequestOptions, AIResponse

# Get module logger
logger = get_logger("services.ai_providers.openai")

# Try to import OpenAI
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not installed. OpenAI provider will not be available.")
    OPENAI_AVAILABLE = False


class OpenAIProvider(AIProvider):
    """Provider for OpenAI API"""
    
    def __init__(self):
        """Initialize the OpenAI provider"""
        self._models = {
            "gpt-4o": {
                "type": "chat",
                "context_window": 128000,
                "description": "Most capable model for a wide range of tasks"
            },
            "gpt-4o-mini": {
                "type": "chat",
                "context_window": 128000,
                "description": "Smaller, faster and more affordable GPT-4o model"
            },
            "gpt-4": {
                "type": "chat",
                "context_window": 8192,
                "description": "Powerful model for various tasks"
            },
            "gpt-4-turbo": {
                "type": "chat",
                "context_window": 128000,
                "description": "More capable and cost-effective version of GPT-4"
            },
            "gpt-3.5-turbo": {
                "type": "chat",
                "context_window": 16385,
                "description": "Fast and cost-effective model for most tasks"
            },
            "text-embedding-3-small": {
                "type": "embedding",
                "dimensions": 1536,
                "description": "Efficient embedding model"
            },
            "text-embedding-3-large": {
                "type": "embedding",
                "dimensions": 3072,
                "description": "High-quality embedding model"
            }
        }
        
        self._client = None
        self._init_client()
        
        logger.debug("OpenAI provider initialized")
    
    def _init_client(self) -> None:
        """Initialize the OpenAI client"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not installed. Cannot initialize client.")
            return
        
        api_key = get_env("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not found in environment variables.")
            return
        
        try:
            self._client = OpenAI(api_key=api_key)
            
            # Test the client with a simple request - REMOVE limit parameter for compatibility
            logger.debug("Testing OpenAI client connection...")
            models_response = self._client.models.list()
            logger.debug(f"OpenAI client test successful - found {len(models_response.data)} models")
            
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            logger.exception("OpenAI client initialization error details:")
            self._client = None
    
    @property
    def name(self) -> str:
        """Get the provider name"""
        return "openai"
    
    @property
    def available_models(self) -> List[str]:
        """Get available models"""
        return list(self._models.keys())
    
    def is_available(self) -> bool:
        """Check if the provider is available"""
        return OPENAI_AVAILABLE and self._client is not None
    
    def _convert_messages(self, messages: List[AIMessage]) -> List[Dict[str, Any]]:
        """Convert AIMessages to OpenAI format"""
        openai_messages = []
        
        for msg in messages:
            openai_msg = {
                "role": msg.role,
                "content": msg.content
            }
            
            if msg.name:
                openai_msg["name"] = msg.name
                
            if msg.function_call:
                openai_msg["function_call"] = msg.function_call
                
            if msg.tool_calls:
                openai_msg["tool_calls"] = msg.tool_calls
                
            openai_messages.append(openai_msg)
        
        return openai_messages
    
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
            raise ValueError("OpenAI provider not available")
        
        # Use default options if none provided
        if not options:
            options = AIRequestOptions()
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": openai_messages,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "frequency_penalty": options.frequency_penalty,
            "presence_penalty": options.presence_penalty,
        }
        
        # Add optional parameters
        if options.max_tokens:
            params["max_tokens"] = options.max_tokens
            
        if options.stop:
            params["stop"] = options.stop
            
        if options.tools:
            params["tools"] = options.tools
            
        if options.tool_choice:
            params["tool_choice"] = options.tool_choice
            
        if options.response_format:
            params["response_format"] = options.response_format
            
        if options.seed:
            params["seed"] = options.seed
        
        try:
            # Make the request
            response = self._client.chat.completions.create(**params)
            
            # Extract the content
            content = response.choices[0].message.content or ""
            
            # Extract tool calls if present
            tool_calls = None
            if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            
            # Create the response
            ai_response = AIResponse(
                content=content,
                model=model,
                provider=self.name,
                finish_reason=response.choices[0].finish_reason,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                tool_calls=tool_calls
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in OpenAI completion: {e}")
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
            raise ValueError("OpenAI provider not available")
        
        # Use default options if none provided
        if not options:
            options = AIRequestOptions()
        
        # Force streaming
        options.stream = True
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": openai_messages,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "frequency_penalty": options.frequency_penalty,
            "presence_penalty": options.presence_penalty,
            "stream": True
        }
        
        # Add optional parameters
        if options.max_tokens:
            params["max_tokens"] = options.max_tokens
            
        if options.stop:
            params["stop"] = options.stop
            
        if options.tools:
            params["tools"] = options.tools
            
        if options.tool_choice:
            params["tool_choice"] = options.tool_choice
            
        if options.response_format:
            params["response_format"] = options.response_format
            
        if options.seed:
            params["seed"] = options.seed
        
        try:
            # Make the streaming request
            response_stream = self._client.chat.completions.create(**params)
            
            # Variables to collect the complete response
            full_content = ""
            finish_reason = None
            usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            tool_calls_data = None
            
            # Process the stream
            for chunk in response_stream:
                if not chunk.choices:
                    continue
                
                # Get the delta
                delta = chunk.choices[0].delta
                
                # Process content
                if delta.content:
                    full_content += delta.content
                    callback(delta.content)
                
                # Process tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    # Initialize tool_calls_data if needed
                    if tool_calls_data is None:
                        tool_calls_data = []
                    
                    # Process each tool call in the delta
                    for tc in delta.tool_calls:
                        # Find or create the tool call
                        tc_id = tc.index
                        if tc_id >= len(tool_calls_data):
                            # Extend the list
                            tool_calls_data.extend([None] * (tc_id - len(tool_calls_data) + 1))
                        
                        if tool_calls_data[tc_id] is None:
                            tool_calls_data[tc_id] = {
                                "id": tc.id or "",
                                "type": tc.type or "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        
                        # Update the tool call
                        if tc.id:
                            tool_calls_data[tc_id]["id"] = tc.id
                            
                        if tc.type:
                            tool_calls_data[tc_id]["type"] = tc.type
                            
                        if tc.function:
                            if tc.function.name:
                                tool_calls_data[tc_id]["function"]["name"] = tc.function.name
                                
                            if tc.function.arguments:
                                tool_calls_data[tc_id]["function"]["arguments"] += tc.function.arguments
                
                # Update finish reason
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            
            # Create the final response
            ai_response = AIResponse(
                content=full_content,
                model=model,
                provider=self.name,
                finish_reason=finish_reason,
                usage=usage_data,
                tool_calls=tool_calls_data
            )
            
            return ai_response
            
        except Exception as e:
            logger.error(f"Error in OpenAI streaming completion: {e}")
            raise
    
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
        if not self.is_available():
            raise ValueError("OpenAI provider not available")
        
        # Use default model if none provided
        if not model:
            model = "text-embedding-3-small"
        
        # Convert to list if a single string
        input_texts = [text] if isinstance(text, str) else text
        
        try:
            # Make the request
            response = self._client.embeddings.create(
                input=input_texts,
                model=model
            )
            
            # Extract the embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Return as a single list if input was a string
            if isinstance(text, str):
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in OpenAI embedding: {e}")
            raise
