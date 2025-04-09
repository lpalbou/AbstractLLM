"""
Anthropic API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List
import os
import logging
import time
import asyncio

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import (
    log_request, 
    log_response, 
    log_api_key_from_env, 
    log_api_key_missing,
    log_request_url
)
from abstractllm.utils.image import preprocess_image_inputs
from abstractllm.utils.config import ConfigurationManager

# Configure logger
logger = logging.getLogger("abstractllm.providers.anthropic.AnthropicProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "claude-3-opus", 
    "claude-3-sonnet", 
    "claude-3-haiku",
    "claude-3.5-sonnet",
    "claude-3.5-haiku",
    "claude-3-7-sonnet"
]

# Try to import Anthropic API
try:
    import anthropic
    from anthropic import (
        AI_PROMPT,
        HUMAN_PROMPT,
        APIError,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(AbstractLLMInterface):
    """
    Anthropic API implementation.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Anthropic API provider with given configuration.

        Args:
            config: Configuration dictionary with required parameters.
        """
        super().__init__(config)
        
        # Check if Anthropic package is available
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not found. Install it with: pip install anthropic")
        
        # Initialize provider-specific configuration
        self.config = ConfigurationManager.initialize_provider_config("anthropic", self.config)
        
        # Extract the model and API key
        self.model = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "claude-3-5-haiku-20241022")
        api_key = ConfigurationManager.get_param(self.config, ModelParameter.API_KEY)
        
        # Create client
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Initialized {self.__class__.__name__} with model {self.model}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using Anthropic API.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
                - image: A single image (URL, file path, base64 string, or dict)
                - images: List of images (URLs, file paths, base64 strings, or dicts)
            
        Returns:
            The generated response or a generator if streaming
            
        Raises:
            Exception: If the API call fails or no API key is provided
        """
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "anthropic", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        api_key = params.get("api_key")
        model = params.get("model", self.model)
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 2048)
        system_prompt = params.get("system_prompt")
        
        # Check for API key
        if not api_key:
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Log at INFO level
        logger.info(f"Generating response with Anthropic model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}")
        if system_prompt:
            logger.debug("Using system prompt")
        
        # Check if model supports vision
        has_vision = any(vision_model in model.lower() for vision_model in [vm.lower() for vm in VISION_CAPABLE_MODELS])
        
        # Process image inputs if any, and if model supports vision
        image_request = False
        if has_vision and ("image" in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = preprocess_image_inputs(params, "anthropic")
        
        # Prepare message parameters
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": []
        }
        
        # Add system message if provided
        if system_prompt:
            message_params["system"] = system_prompt
        
        # Handle image content
        if image_request:
            # The image content should already be formatted by preprocess_image_inputs
            message_params["messages"] = params.get("messages", [])
            if not message_params["messages"]:
                # Fallback if messages not set by preprocess_image_inputs
                message_params["messages"] = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            # Text-only request
            message_params["messages"] = [{"role": "user", "content": prompt}]
        
        # Log the request
        log_request("anthropic", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request
        })
        
        # Log API request (without exposing API key)
        log_request_url("anthropic", f"Anthropic API (model: {model})")
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting streaming generation")
            
            def response_generator():
                # Create a copy of message_params for streaming (without the 'stream' parameter)
                streaming_params = message_params.copy()
                
                # The Anthropic API's messages.stream() method handles streaming automatically,
                # so we don't need to add stream=True to the parameters
                
                # Call the API
                with self.client.messages.stream(**streaming_params) as stream:
                    for chunk in stream:
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            yield chunk.delta.text
            
            return response_generator()
        else:
            # Standard non-streaming response
            response = self.client.messages.create(**message_params)
            result = response.content[0].text
            log_response("anthropic", result)
            logger.info("Generation completed successfully")
            
            return result
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Asynchronously generate a response using Anthropic API.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
                - image: A single image (URL, file path, base64 string, or dict)
                - images: List of images (URLs, file paths, base64 strings, or dicts)
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: An async generator yielding response chunks
            
        Raises:
            Exception: If the API call fails or no API key is provided
        """
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "anthropic", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        api_key = params.get("api_key")
        model = params.get("model", self.model)
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 2048)
        system_prompt = params.get("system_prompt")
        
        # Check for API key
        if not api_key:
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Log at INFO level
        logger.info(f"Generating async response with Anthropic model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}")
        if system_prompt:
            logger.debug("Using system prompt")
        
        # Check if model supports vision
        has_vision = any(vision_model in model.lower() for vision_model in [vm.lower() for vm in VISION_CAPABLE_MODELS])
        
        # Process image inputs if any, and if model supports vision
        image_request = False
        if has_vision and ("image" in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = preprocess_image_inputs(params, "anthropic")
        
        # Prepare message parameters
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": []
        }
        
        # Add system message if provided
        if system_prompt:
            message_params["system"] = system_prompt
        
        # Handle image content
        if image_request:
            # The image content should already be formatted by preprocess_image_inputs
            message_params["messages"] = params.get("messages", [])
            if not message_params["messages"]:
                # Fallback if messages not set by preprocess_image_inputs
                message_params["messages"] = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        else:
            # Text-only request
            message_params["messages"] = [{"role": "user", "content": prompt}]
        
        # Log the request
        log_request("anthropic", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request
        })
        
        # Log API request (without exposing API key)
        log_request_url("anthropic", f"Anthropic API (model: {model})")
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting async streaming generation")
            
            async def async_generator():
                # For async streaming, we need to use the synchronous API through asyncio.to_thread
                # Create a copy of message_params for streaming
                streaming_params = message_params.copy()
                
                # Implement a wrapper that runs the synchronous streaming in a thread
                async def stream_wrapper():
                    def sync_stream():
                        chunks = []
                        with self.client.messages.stream(**streaming_params) as stream:
                            for chunk in stream:
                                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                                    chunks.append(chunk.delta.text)
                        return chunks
                    
                    return await asyncio.to_thread(sync_stream)
                
                # Get all chunks from the stream
                chunks = await stream_wrapper()
                
                # Yield each chunk individually to maintain streaming behavior
                for chunk in chunks:
                    yield chunk
            
            return async_generator()
        else:
            # Use a wrapper function to execute the synchronous API call
            async def async_wrapper():
                # Standard non-streaming response
                response = await asyncio.to_thread(
                    self.client.messages.create, **message_params
                )
                result = response.content[0].text
                log_response("anthropic", result)
                logger.info("Async generation completed successfully")
                return result
            
            return await async_wrapper()
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the Anthropic provider.
        
        Returns:
            Dictionary of capabilities
        """
        # Default base capabilities
        capabilities = {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 100000,  # Anthropic models support large outputs
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,  # Add this when Anthropic supports tool use
            ModelCapability.VISION: False
        }
        
        # Check if the current model supports vision
        model = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, self.model)
        has_vision = any(vision_model in model.lower() for vision_model in [vm.lower() for vm in VISION_CAPABLE_MODELS])
        
        # Update vision capability
        if has_vision:
            capabilities[ModelCapability.VISION] = True
            
        return capabilities 