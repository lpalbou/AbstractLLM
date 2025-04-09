"""
OpenAI API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import (
    log_request, 
    log_response, 
    log_api_key_from_env, 
    log_api_key_missing,
    log_request_url
)
from abstractllm.media.processor import MediaProcessor
from abstractllm.utils.config import ConfigurationManager

# Configure logger
logger = logging.getLogger("abstractllm.providers.openai.OpenAIProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "gpt-4-vision",
    "gpt-4-vision-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-vision-preview",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-2024-05-13"
]

class OpenAIProvider(AbstractLLMInterface):
    """
    OpenAI API implementation.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the OpenAI API provider with given configuration.

        Args:
            config: Configuration dictionary with required parameters.
        """
        super().__init__(config)
        
        # Initialize provider-specific configuration
        self.config = ConfigurationManager.initialize_provider_config("openai", self.config)
        
        # Log provider initialization
        model = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "gpt-3.5-turbo")
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
                - image: A single image (URL, file path, base64 string, or dict)
                - images: List of images (URLs, file paths, base64 strings, or dicts)
                - image_detail: Detail level for image analysis ('low', 'high', 'auto')
            
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
        
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "openai", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        api_key = params.get("api_key")
        model = params.get("model", "gpt-3.5-turbo")
        
        # Ensure model is not None
        if model is None:
            model = "gpt-3.5-turbo"
            logger.warning(f"Model was None, defaulting to {model}")
            
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens")
        system_prompt = params.get("system_prompt")
        top_p = params.get("top_p", 1.0)
        frequency_penalty = params.get("frequency_penalty", 0.0)
        presence_penalty = params.get("presence_penalty", 0.0)
        stop = params.get("stop")
        image_detail = params.get("image_detail", "auto")
        
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
        
        # Check if model supports vision
        has_vision = any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS)
        
        # Process image inputs if any, and if model supports vision
        image_request = False
        if has_vision and ("image" in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = MediaProcessor.process_inputs(params, "openai")
        
        # Prepare messages
        messages = params.get("messages", [])
        
        # Add system message if not already in messages
        if system_prompt and not any(msg.get("role") == "system" for msg in messages):
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message if not already in messages and not an image request
        # For image requests, the preprocess_image_inputs function already adds the user message
        if not image_request and not any(msg.get("role") == "user" for msg in messages):
            messages.append({"role": "user", "content": prompt})
        
        # For text-only requests, make sure messages exist
        if not messages:
            messages.append({"role": "user", "content": prompt})
        
        # Log the request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request
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
                - image: A single image (URL, file path, base64 string, or dict)
                - images: List of images (URLs, file paths, base64 strings, or dicts)
                - image_detail: Detail level for image analysis ('low', 'high', 'auto')
            
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
            logger.warning(f"Model was None, defaulting to {model}")
            # Update the config for future use
            self.config[ModelParameter.MODEL] = model
        
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens"))
        system_prompt = system_prompt or params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        frequency_penalty = params.get(ModelParameter.FREQUENCY_PENALTY, params.get("frequency_penalty", 0.0))
        presence_penalty = params.get(ModelParameter.PRESENCE_PENALTY, params.get("presence_penalty", 0.0))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Handle image detail parameter for vision models
        image_detail = params.get(ModelParameter.IMAGE_DETAIL, params.get("image_detail", "auto"))
        
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
        
        # Check if model supports vision
        has_vision = any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS)
        
        # Process image inputs if any
        image_request = False
        if has_vision and (ModelParameter.IMAGE in params or "image" in params or 
                          ModelParameter.IMAGES in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = MediaProcessor.process_inputs(params, "openai")
        
        # Prepare messages
        messages = params.get("messages", [])
        
        # Add system message if not already in messages
        if system_prompt and not any(msg.get("role") == "system" for msg in messages):
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message if not already in messages and not an image request
        # For image requests, the preprocess_image_inputs function already adds the user message
        if not image_request and not any(msg.get("role") == "user" for msg in messages):
            messages.append({"role": "user", "content": prompt})
        
        # For text-only requests, make sure messages exist
        if not messages:
            messages.append({"role": "user", "content": prompt})
        
        # Log the request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request
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
        # Get current model
        model = self.config.get(ModelParameter.MODEL, self.config.get("model", "gpt-3.5-turbo"))
        
        # Check if model is vision-capable - ensure model is not None before checking
        has_vision = False
        if model:
            has_vision = any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS)
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 4096,  # This varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: True,
            ModelCapability.VISION: has_vision,  # Dynamic based on model
            ModelCapability.JSON_MODE: True  # OpenAI supports JSON mode
        } 

# Add a wrapper class for backward compatibility with the test suite
class OpenAILLM:
    """
    Wrapper around OpenAIProvider for backward compatibility with the test suite.
    """
    
    def __init__(self, model="gpt-4o", api_key=None):
        """
        Initialize an OpenAI LLM instance.
        
        Args:
            model: The model to use
            api_key: Optional API key (will use environment variable if not provided)
        """
        config = {
            ModelParameter.MODEL: model,
        }
        
        if api_key:
            config[ModelParameter.API_KEY] = api_key
            
        self.provider = OpenAIProvider(config)
        
    def generate(self, prompt, image=None, images=None, **kwargs):
        """
        Generate a response using the OpenAI provider.
        
        Args:
            prompt: The prompt to send
            image: Optional single image
            images: Optional list of images
            return_format: Format to return the response in
            **kwargs: Additional parameters
            
        Returns:
            The generated response
        """
        # Add images to kwargs if provided
        if image:
            kwargs["image"] = image
        if images:
            kwargs["images"] = images
            
        response = self.provider.generate(prompt, **kwargs)

        return response
