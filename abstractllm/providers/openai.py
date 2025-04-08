"""
OpenAI API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator
import os
import asyncio
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
logger = logging.getLogger("abstractllm.providers.openai.OpenAIProvider")


class OpenAIProvider(AbstractLLMInterface):
    """
    OpenAI API implementation.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if ModelParameter.API_KEY not in self.config and "api_key" not in self.config:
            env_api_key = os.environ.get("OPENAI_API_KEY")
            if env_api_key:
                self.config[ModelParameter.API_KEY] = env_api_key
                log_api_key_from_env("OpenAI", "OPENAI_API_KEY")
        
        if ModelParameter.MODEL not in self.config and "model" not in self.config:
            self.config[ModelParameter.MODEL] = "gpt-3.5-turbo"
        
        # Log provider initialization
        model = self.config.get(ModelParameter.MODEL, self.config.get("model", "gpt-3.5-turbo"))
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using OpenAI API.
        
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
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install it with: pip install openai"
            )
        
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters (using both string and enum keys for backwards compatibility)
        api_key = params.get(ModelParameter.API_KEY, params.get("api_key"))
        
        # Try getting the API key from environment if not in config
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                # Update the config with the API key from environment for future use
                self.config[ModelParameter.API_KEY] = api_key
                log_api_key_from_env("OpenAI", "OPENAI_API_KEY")
        
        # Ensure model is set and has a valid value
        model = params.get(ModelParameter.MODEL, params.get("model"))
        if not model:
            model = "gpt-3.5-turbo"  # Default model
            # Update the config for future use
            self.config[ModelParameter.MODEL] = model
        
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens"))
        system_prompt = system_prompt or params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        frequency_penalty = params.get(ModelParameter.FREQUENCY_PENALTY, params.get("frequency_penalty", 0.0))
        presence_penalty = params.get(ModelParameter.PRESENCE_PENALTY, params.get("presence_penalty", 0.0))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Log at INFO level
        logger.info(f"Generating response with OpenAI model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
        if system_prompt:
            logger.debug("Using system prompt")
        if stop:
            logger.debug(f"Using stop sequences: {stop}")
        
        # Check for API key
        if not api_key:
            log_api_key_missing("OpenAI", "OPENAI_API_KEY")
            raise ValueError(
                "OpenAI API key not provided. Pass it as a parameter in config or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Log the request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream
        })
        
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Log API request (without exposing API key)
        log_request_url("openai", f"https://api.openai.com/v1/chat/completions (model: {model})")
        
        # Prepare completion parameters
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        if max_tokens:
            completion_params["max_tokens"] = max_tokens
            
        if stop:
            completion_params["stop"] = stop
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting streaming generation")
            
            def response_generator():
                stream_resp = client.chat.completions.create(**completion_params)
                for chunk in stream_resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield content
            
            return response_generator()
        else:
            # Standard non-streaming response
            response = client.chat.completions.create(**completion_params)
            result = response.choices[0].message.content
            log_response("openai", result)
            logger.info("Generation completed successfully")
            
            return result
    
    async def generate_async(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None, 
                           stream: bool = False, 
                           **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Asynchronously generate a response using OpenAI API.
        
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
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install it with: pip install openai"
            )
        
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters (using both string and enum keys for backwards compatibility)
        api_key = params.get(ModelParameter.API_KEY, params.get("api_key"))
        
        # Try getting the API key from environment if not in config
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                # Update the config with the API key from environment for future use
                self.config[ModelParameter.API_KEY] = api_key
                log_api_key_from_env("OpenAI", "OPENAI_API_KEY")
        
        # Ensure model is set and has a valid value
        model = params.get(ModelParameter.MODEL, params.get("model"))
        if not model:
            model = "gpt-3.5-turbo"  # Default model
            # Update the config for future use
            self.config[ModelParameter.MODEL] = model
        
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens"))
        system_prompt = system_prompt or params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        frequency_penalty = params.get(ModelParameter.FREQUENCY_PENALTY, params.get("frequency_penalty", 0.0))
        presence_penalty = params.get(ModelParameter.PRESENCE_PENALTY, params.get("presence_penalty", 0.0))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Log at INFO level
        logger.info(f"Generating async response with OpenAI model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
        if system_prompt:
            logger.debug("Using system prompt")
        if stop:
            logger.debug(f"Using stop sequences: {stop}")
        
        # Check for API key
        if not api_key:
            log_api_key_missing("OpenAI", "OPENAI_API_KEY")
            raise ValueError(
                "OpenAI API key not provided. Pass it as a parameter in config or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Log the request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "async": True
        })
        
        # Initialize async client
        client = AsyncOpenAI(api_key=api_key)
        
        # Log API request (without exposing API key)
        log_request_url("openai", f"https://api.openai.com/v1/chat/completions (model: {model})")
        
        # Prepare completion parameters
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        if max_tokens:
            completion_params["max_tokens"] = max_tokens
            
        if stop:
            completion_params["stop"] = stop
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting async streaming generation")
            
            async def async_generator():
                stream_resp = await client.chat.completions.create(**completion_params)
                async for chunk in stream_resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield content
            
            return async_generator()
        else:
            # Standard non-streaming response
            response = await client.chat.completions.create(**completion_params)
            result = response.choices[0].message.content
            log_response("openai", result)
            logger.info("Async generation completed successfully")
            
            return result
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the OpenAI provider.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 4096,  # This varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: True,
            ModelCapability.VISION: True  # For models that support it like GPT-4 Vision
        } 