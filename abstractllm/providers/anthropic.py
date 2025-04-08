"""
Anthropic API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator
import os
import logging

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability, create_config
from abstractllm.utils.logging import (
    log_request, 
    log_response, 
    log_api_key_from_env, 
    log_api_key_missing,
    log_request_url
)

# Configure logger with specific class path
logger = logging.getLogger("abstractllm.providers.anthropic.AnthropicProvider")


class AnthropicProvider(AbstractLLMInterface):
    """
    Anthropic API implementation.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if ModelParameter.API_KEY not in self.config and "api_key" not in self.config:
            env_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if env_api_key:
                self.config[ModelParameter.API_KEY] = env_api_key
                log_api_key_from_env("Anthropic", "ANTHROPIC_API_KEY")
        
        if ModelParameter.MODEL not in self.config and "model" not in self.config:
            self.config[ModelParameter.MODEL] = "claude-3-opus-20240229"
    
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
            
        Returns:
            The generated response or a generator if streaming
            
        Raises:
            Exception: If the API call fails or no API key is provided
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Install it with: pip install anthropic"
            )
        
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters (using both string and enum keys for backwards compatibility)
        api_key = params.get(ModelParameter.API_KEY, params.get("api_key"))
        model = params.get(ModelParameter.MODEL, params.get("model", "claude-3-opus-20240229"))
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens", 1024))
        system_prompt_from_config = params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        system_prompt = system_prompt or system_prompt_from_config
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Log at INFO level
        logger.info(f"Generating response with Anthropic model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
        if system_prompt:
            logger.debug("Using system prompt")
        if stop:
            logger.debug(f"Using stop sequences: {stop}")
        
        # Check for API key
        if not api_key:
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Log the request
        log_request("anthropic", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream
        })
        
        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Log API request (without exposing API key)
        log_request_url("anthropic", f"https://api.anthropic.com/v1/messages (model: {model})")
        
        # Prepare message
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
        
        if system_prompt:
            message_params["system"] = system_prompt
            
        if stop:
            message_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting streaming generation")
            
            def response_generator():
                # Create a copy of message_params for streaming (without the 'stream' parameter)
                streaming_params = message_params.copy()
                if 'stream' in streaming_params:
                    # Remove the 'stream' parameter as it's not needed for client.messages.stream()
                    del streaming_params['stream']
                
                with client.messages.stream(**streaming_params) as stream:
                    for text in stream.text_stream:
                        yield text
            
            return response_generator()
        else:
            # Call API
            response = client.messages.create(**message_params)
            
            # Extract and log the response
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
            
        Returns:
            The generated response or an async generator if streaming
            
        Raises:
            Exception: If the API call fails
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Install it with: pip install anthropic"
            )
        
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters (using both string and enum keys for backwards compatibility)
        api_key = params.get(ModelParameter.API_KEY, params.get("api_key"))
        model = params.get(ModelParameter.MODEL, params.get("model", "claude-3-opus-20240229"))
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens", 1024))
        system_prompt_from_config = params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        system_prompt = system_prompt or system_prompt_from_config
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Log at INFO level
        logger.info(f"Generating async response with Anthropic model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
        if system_prompt:
            logger.debug("Using system prompt")
        if stop:
            logger.debug(f"Using stop sequences: {stop}")
        
        # Check for API key
        if not api_key:
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Log the request
        log_request("anthropic", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "async": True
        })
        
        # Initialize client
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Log API request (without exposing API key)
        log_request_url("anthropic", f"https://api.anthropic.com/v1/messages (model: {model})")
        
        # Prepare message
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }
        
        if system_prompt:
            message_params["system"] = system_prompt
            
        if stop:
            message_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting async streaming generation")
            
            async def async_generator():
                # Create a copy of message_params for streaming (without the 'stream' parameter)
                streaming_params = message_params.copy()
                if 'stream' in streaming_params:
                    # Remove the 'stream' parameter as it's not needed for client.messages.stream()
                    del streaming_params['stream']
                
                async with client.messages.stream(**streaming_params) as stream:
                    async for text in stream.text_stream:
                        yield text
            
            return async_generator()
        else:
            # Call API
            response = await client.messages.create(**message_params)
            
            # Extract and log the response
            result = response.content[0].text
            log_response("anthropic", result)
            logger.info("Async generation completed successfully")
            
            return result
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the Anthropic provider.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 100000,  # This varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: True,  # Claude 3 supports tool use
            ModelCapability.VISION: True  # Claude 3 supports vision
        } 