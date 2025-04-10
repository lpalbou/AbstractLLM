"""
OpenAI API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List
import os
import logging
import asyncio
from pathlib import Path

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import (
    log_request, 
    log_response, 
    log_api_key_from_env, 
    log_api_key_missing,
    log_request_url
)
from abstractllm.media.factory import MediaFactory
from abstractllm.media.image import ImageInput
from abstractllm.exceptions import (
    UnsupportedFeatureError,
    FileProcessingError,
    ProviderAPIError
)

# Configure logger
logger = logging.getLogger("abstractllm.providers.openai.OpenAIProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini"
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
        
        # Set default configuration for OpenAI
        default_config = {
            ModelParameter.MODEL: "gpt-4o",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 2048,
            ModelParameter.TOP_P: 1.0,
            ModelParameter.FREQUENCY_PENALTY: 0.0,
            ModelParameter.PRESENCE_PENALTY: 0.0
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Get necessary parameters from config
        model = self.config_manager.get_param(ModelParameter.MODEL)
        temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE)
        max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS)
        api_key = self.config_manager.get_param(ModelParameter.API_KEY)
        
        # Check for API key
        if not api_key:
            log_api_key_missing("OpenAI", "OPENAI_API_KEY")
            raise ValueError(
                "OpenAI API key not provided. Pass it as a parameter in config or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Process files if any
        processed_files = []
        if files:
            for file_path in files:
                try:
                    media_input = MediaFactory.from_source(file_path)
                    processed_files.append(media_input)
                except Exception as e:
                    raise FileProcessingError(
                        f"Failed to process file {file_path}: {str(e)}",
                        provider="openai",
                        original_exception=e
                    )
        
        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="openai"
            )
        
        # Prepare messages
        messages = []
        
        # Add system message if provided (either from config or parameter)
        system_prompt = system_prompt or self.config_manager.get_param(ModelParameter.SYSTEM_PROMPT)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Prepare user message with files if any
        if processed_files:
            content = [{"type": "text", "text": prompt}]
            for media_input in processed_files:
                content.append(media_input.to_provider_format("openai"))
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Log request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files)
        })
        
        # Make API call
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                def response_generator():
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return response_generator()
            else:
                response = completion.choices[0].message.content
                log_response("openai", response)
                return response
                
        except Exception as e:
            raise ProviderAPIError(
                f"OpenAI API error: {str(e)}",
                provider="openai",
                original_exception=e
            )
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response using OpenAI API."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Get necessary parameters from config
        model = self.config_manager.get_param(ModelParameter.MODEL)
        temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE)
        max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS)
        api_key = self.config_manager.get_param(ModelParameter.API_KEY)
        
        # Check for API key
        if not api_key:
            log_api_key_missing("OpenAI", "OPENAI_API_KEY")
            raise ValueError(
                "OpenAI API key not provided. Pass it as a parameter in config or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Process files if any
        processed_files = []
        if files:
            for file_path in files:
                try:
                    media_input = MediaFactory.from_source(file_path)
                    processed_files.append(media_input)
                except Exception as e:
                    raise FileProcessingError(
                        f"Failed to process file {file_path}: {str(e)}",
                        provider="openai",
                        original_exception=e
                    )
        
        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="openai"
            )
        
        # Prepare messages
        messages = []
        
        # Add system message if provided (either from config or parameter)
        system_prompt = system_prompt or self.config_manager.get_param(ModelParameter.SYSTEM_PROMPT)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Prepare user message with files if any
        if processed_files:
            content = [{"type": "text", "text": prompt}]
            for media_input in processed_files:
                content.append(media_input.to_provider_format("openai"))
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Initialize async OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Log request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files)
        })
        
        # Make API call
        try:
            if stream:
                async def async_generator():
                    stream_resp = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )
                    async for chunk in stream_resp:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                return async_generator()
            else:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                response = completion.choices[0].message.content
                log_response("openai", response)
                return response
                
        except Exception as e:
            raise ProviderAPIError(
                f"OpenAI API error: {str(e)}",
                provider="openai",
                original_exception=e
            )
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of the OpenAI provider."""
        # Get current model
        model = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check if model is vision-capable
        has_vision = any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS)
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 4096,  # This varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: True,
            ModelCapability.VISION: has_vision,
            ModelCapability.JSON_MODE: True
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
