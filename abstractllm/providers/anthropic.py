"""
Anthropic API implementation for AbstractLLM.
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
logger = logging.getLogger("abstractllm.providers.anthropic.AnthropicProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "claude-3-haiku-20240307"
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229", 
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022", # work on images
    "claude-3-5-haiku-20241022", # did not work on images somehow...
    "claude-3-7-sonnet-20250219"
]

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
        
        # Set default configuration for Anthropic
        default_config = {
            ModelParameter.MODEL: "claude-3-5-haiku-20241022",
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
        logger.info(f"Initialized Anthropic provider with model: {model}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic package not found. Install it with: pip install anthropic")
        
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
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
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
                        provider="anthropic",
                        original_exception=e
                    )
        
        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="anthropic"
            )
        
        # Initialize Anthropic client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare messages
        messages = []
        
        # Add system message if provided (either from config or parameter)
        system_prompt = system_prompt or self.config_manager.get_param(ModelParameter.SYSTEM_PROMPT)
        
        # Prepare user message content
        content = []
        
        # Add files first if any
        if processed_files:
            for media_input in processed_files:
                content.append(media_input.to_provider_format("anthropic"))
        
        # Add text prompt after files
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Add the user message with the complete content array
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Log request
        log_request("anthropic", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files)
        })
        
        # Make API call
        try:
            # Create message with system prompt if provided
            message_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            
            if system_prompt:
                message_params["system"] = system_prompt
            
            if stream:
                def response_generator():
                    with client.messages.stream(**message_params) as stream:
                        for chunk in stream:
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                                yield chunk.delta.text
                return response_generator()
            else:
                response = client.messages.create(**message_params)
                result = response.content[0].text
                log_response("anthropic", result)
                return result
                
        except Exception as e:
            raise ProviderAPIError(
                f"Anthropic API error: {str(e)}",
                provider="anthropic",
                original_exception=e
            )
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response using Anthropic API."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("Anthropic package not found. Install it with: pip install anthropic")
        
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
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
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
                        provider="anthropic",
                        original_exception=e
                    )
        
        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="anthropic"
            )
        
        # Initialize async Anthropic client
        client = AsyncAnthropic(api_key=api_key)
        
        # Prepare messages
        messages = []
        
        # Add system message if provided (either from config or parameter)
        system_prompt = system_prompt or self.config_manager.get_param(ModelParameter.SYSTEM_PROMPT)
        
        # Prepare user message with files if any
        if processed_files:
            content = [{"type": "text", "text": prompt}]
            for media_input in processed_files:
                content.append(media_input.to_provider_format("anthropic"))
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Log request
        log_request("anthropic", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files)
        })
        
        # Make API call
        try:
            # Create message with system prompt if provided
            message_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            
            if system_prompt:
                message_params["system"] = system_prompt
            
            if stream:
                async def async_generator():
                    async with client.messages.stream(**message_params) as stream:
                        async for chunk in stream:
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                                yield chunk.delta.text
                return async_generator()
            else:
                response = await client.messages.create(**message_params)
                result = response.content[0].text
                log_response("anthropic", result)
                return result
                
        except Exception as e:
            raise ProviderAPIError(
                f"Anthropic API error: {str(e)}",
                provider="anthropic",
                original_exception=e
            )
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of the Anthropic provider."""
        # Get current model
        model = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check if model is vision-capable
        has_vision = any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS)
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 100000,  # Claude 3 models support large outputs
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,  # Not yet supported
            ModelCapability.VISION: has_vision,
            ModelCapability.JSON_MODE: True
        }

# Add a wrapper class for backward compatibility with the test suite
class AnthropicLLM:
    """
    Simple adapter around AnthropicProvider for test compatibility.
    """
    
    def __init__(self, model="claude-3-haiku", api_key=None):
        """
        Initialize an Anthropic LLM instance.
        
        Args:
            model: The model to use
            api_key: Optional API key (will use environment variable if not provided)
        """
        config = {
            ModelParameter.MODEL: model,
        }
        
        if api_key:
            config[ModelParameter.API_KEY] = api_key
            
        self.provider = AnthropicProvider(config)
        
    def generate(self, prompt, image=None, images=None, **kwargs):
        """
        Generate a response using the provider.
        
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