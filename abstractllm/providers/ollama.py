"""
Ollama API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List
from pathlib import Path
import os
import json
import asyncio
import aiohttp
import requests
import logging
import copy

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import (
    log_request, 
    log_response,
    log_request_url,
    truncate_base64
)
from abstractllm.media.processor import MediaProcessor
from abstractllm.exceptions import ImageProcessingError, FileProcessingError, UnsupportedFeatureError, ProviderAPIError
from abstractllm.media.factory import MediaFactory
from abstractllm.media.image import ImageInput

# Configure logger
logger = logging.getLogger("abstractllm.providers.ollama.OllamaProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "llama3.2-vision:latest",
    "deepseek-janus-pro",
    "erwan2/DeepSeek-Janus-Pro-7B",
    "llava",
    "llama2-vision",
    "bakllava",
    "cogvlm",
    "moondream",
    "multimodal",
    "vision"
]

class OllamaProvider(AbstractLLMInterface):
    """
    Ollama API implementation.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration for Ollama
        default_config = {
            ModelParameter.MODEL: "phi4-mini:latest",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 2048,
            ModelParameter.BASE_URL: "http://localhost:11434"
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        base_url = self.config_manager.get_param(ModelParameter.BASE_URL)
        logger.info(f"Initialized Ollama provider with model: {model}, base URL: {base_url}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using Ollama API.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (paths or URLs)
                  Supported types: images (for vision models), text, markdown, CSV, TSV
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: A generator yielding response chunks
            
        Raises:
            Exception: If the generation fails
        """
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Get necessary parameters from config
        model = self.config_manager.get_param(ModelParameter.MODEL)
        temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE)
        max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS)
        base_url = self.config_manager.get_param(ModelParameter.BASE_URL)
        
        # Process files if any
        processed_files = []
        file_contents = ""
        if files:
            for file_path in files:
                try:
                    media_input = MediaFactory.from_source(file_path)
                    processed_files.append(media_input)
                except Exception as e:
                    raise FileProcessingError(
                        f"Failed to process file {file_path}: {str(e)}",
                        provider="ollama",
                        original_exception=e
                    )
        
        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not any(model.lower().startswith(vm.lower()) for vm in VISION_CAPABLE_MODELS):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="ollama"
            )
        
        # Prepare request data
        request_data = {
            "model": model,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Handle files
        if processed_files:
            # For Ollama, we need to handle files differently:
            # - Images go into the images array
            # - Text/tabular content gets appended to the prompt
            images = []
            file_contents = ""
            
            for media_input in processed_files:
                if isinstance(media_input, ImageInput):
                    images.append(media_input.to_provider_format("ollama"))
                else:
                    # For text and tabular data, append to prompt
                    file_contents += media_input.to_provider_format("ollama")
            
            if images:
                request_data["images"] = images
        
        # Add prompt with file contents
        request_data["prompt"] = prompt + file_contents
        
        # Log request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files)
        })
        
        # Make API call
        try:
            endpoint = f"{base_url.rstrip('/')}/api/generate"
            
            # Log API request URL
            log_request_url("ollama", endpoint)
            
            if stream:
                def response_generator():
                    response = requests.post(endpoint, json=request_data, stream=True)
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                                elif "done" in data and data["done"]:
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON from Ollama response: {line}")
                return response_generator()
            else:
                response = requests.post(endpoint, json=request_data)
                response.raise_for_status()
                data = response.json()
                
                if "response" in data:
                    result = data["response"]
                    log_response("ollama", result)
                    return result
                else:
                    logger.error(f"Unexpected response format: {data}")
                    raise ValueError("Unexpected response format from Ollama API")
                    
        except requests.RequestException as e:
            raise ProviderAPIError(
                f"Ollama API request failed: {str(e)}",
                provider="ollama",
                original_exception=e
            )
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        files: Optional[List[Union[str, Path]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Asynchronously generate a response from the Ollama model.
        
        Args:
            prompt (str): The prompt to generate from
            system_prompt (Optional[str]): System prompt to use
            files (Optional[List[str]]): List of file paths to process
            stream (bool): Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Union[str, AsyncGenerator[str, None]]: Generated text or async generator for streaming
            
        Raises:
            FileProcessingError: If there's an error processing input files
            UnsupportedFeatureError: If vision features are requested but not supported
            ProviderAPIError: If the API request fails
        """
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Get necessary parameters from config
        model = self.config_manager.get_param(ModelParameter.MODEL)
        temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE)
        base_url = self.config_manager.get_param(ModelParameter.BASE_URL)

        # Process files if provided
        processed_files = []
        if files:
            for file_path in files:
                try:
                    media_input = MediaFactory.from_source(file_path)
                    processed_files.append(media_input)
                except Exception as e:
                    raise FileProcessingError(
                        f"Failed to process file {file_path}: {str(e)}",
                        provider="ollama",
                        original_exception=e
                    )

        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not any(model.lower().startswith(vm.lower()) for vm in VISION_CAPABLE_MODELS):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="ollama"
            )

        # Prepare request data
        request_data = {
            "model": model,
            "stream": stream,
            "options": {
                "temperature": temperature
            }
        }

        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt

        # Handle images for vision-capable models
        if processed_files:
            # For Ollama, we need to add images as a list of base64 strings or URLs
            images = []
            for media_input in processed_files:
                if isinstance(media_input, ImageInput):
                    images.append(media_input.to_provider_format("ollama"))
            if images:
                request_data["images"] = images

        # Add prompt
        request_data["prompt"] = prompt

        # Log request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files)
        })

        # Construct API endpoint
        api_endpoint = f"{base_url.rstrip('/')}/api/generate"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_endpoint, json=request_data) as response:
                    if not response.ok:
                        error_msg = await response.text()
                        logger.error(f"Ollama API error: {error_msg}")
                        raise ProviderAPIError(f"Ollama API request failed: {error_msg}")

                    if stream:
                        async def response_generator():
                            async for line in response.content:
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                    elif "done" in data and data["done"]:
                                        break
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse streaming response: {e}")
                                    continue
                        return response_generator()
                    else:
                        data = await response.json()
                        if "response" in data:
                            result = data["response"]
                            log_response("ollama", result)
                            return result
                        else:
                            logger.error(f"Unexpected response format: {data}")
                            raise ValueError("Unexpected response format from Ollama API")

        except aiohttp.ClientError as e:
            logger.error(f"Network error during Ollama API request: {str(e)}")
            raise ProviderAPIError(f"Failed to connect to Ollama API: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Ollama API request: {str(e)}")
            raise ProviderAPIError(f"Unexpected error: {str(e)}")
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the Ollama provider.
        
        Returns:
            Dictionary of capabilities
        """
        # Default base capabilities
        capabilities = {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: None,  # Varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.VISION: False
        }
        
        # Check if the current model supports vision
        model = self.config_manager.get_param(ModelParameter.MODEL)
        has_vision = any(vision_model in model.lower() for vision_model in [vm.lower() for vm in VISION_CAPABLE_MODELS])
        
        # Update vision capability
        if has_vision:
            capabilities[ModelCapability.VISION] = True
            
        return capabilities

# Simple adapter class for tests
class OllamaLLM:
    """
    Simple adapter around OllamaProvider for test compatibility.
    """
    
    def __init__(self, model="llava", api_key=None):
        """
        Initialize an Ollama LLM instance.
        
        Args:
            model: The model to use
            api_key: Not used for Ollama but included for API consistency
        """
        config = {
            ModelParameter.MODEL: model,
        }
            
        self.provider = OllamaProvider(config)
        
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