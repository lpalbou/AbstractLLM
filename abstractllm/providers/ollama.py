"""
Ollama API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator
import os
import json
import asyncio
import aiohttp
import requests
import logging

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability, create_config
from abstractllm.utils.logging import (
    log_request, 
    log_response,
    log_request_url
)
from abstractllm.utils.image import preprocess_image_inputs

# Configure logger with specific class path
logger = logging.getLogger("abstractllm.providers.ollama.OllamaProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "llama3.2-vision",
    "deepseek-janus-pro",
    "erwan2/DeepSeek-Janus-Pro-7B"
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
        
        # Set default configuration
        if ModelParameter.BASE_URL not in self.config and "base_url" not in self.config:
            self.config[ModelParameter.BASE_URL] = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
        
        if ModelParameter.MODEL not in self.config and "model" not in self.config:
            self.config[ModelParameter.MODEL] = "phi4-mini:latest"
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using Ollama API.
        
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
            Exception: If the API call fails
        """
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters (using both string and enum keys for backwards compatibility)
        base_url = params.get(ModelParameter.BASE_URL, params.get("base_url", "http://localhost:11434"))
        model = params.get(ModelParameter.MODEL, params.get("model", "phi4-mini:latest"))
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        system_prompt_from_config = params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        system_prompt = system_prompt or system_prompt_from_config
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens"))
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        top_k = params.get(ModelParameter.TOP_K, params.get("top_k", 40))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Log at INFO level
        logger.info(f"Generating response with Ollama model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, top_k={top_k}")
        logger.debug(f"Using Ollama instance at: {base_url}")
        if system_prompt:
            logger.debug("Using system prompt")
        if stop:
            logger.debug(f"Using stop sequences: {stop}")
        
        # Check if model supports vision
        has_vision = any(vm in model.lower() for vm in VISION_CAPABLE_MODELS)
        
        # Process image inputs if any
        image_request = False
        if has_vision and (ModelParameter.IMAGE in params or "image" in params or 
                          ModelParameter.IMAGES in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = preprocess_image_inputs(params, "ollama")
        
        # Build the request
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
            "options": {}  # Additional options can go here
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
            
        # Add max tokens if provided
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
            
        # Add stop sequences if provided
        if stop:
            request_data["options"]["stop"] = stop if isinstance(stop, list) else [stop]
            
        # Include images if provided through preprocess_image_inputs
        if "images" in params:
            request_data["images"] = params["images"]
        
        # Log the request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request
        })
        
        # Log API request URL
        log_request_url("ollama", f"{base_url}/api/generate (model: {model})")
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting streaming generation")
            
            def response_generator():
                try:
                    with requests.post(
                        f"{base_url}/api/generate",
                        json=request_data,
                        stream=True
                    ) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                except requests.exceptions.RequestException as e:
                    logger.error(f"Ollama API request failed: {e}")
                    raise Exception(f"Ollama API request failed: {e}")
            
            return response_generator()
        else:
            # Make the standard API request
            try:
                response = requests.post(
                    f"{base_url}/api/generate",
                    json=request_data
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama API request failed: {e}")
                raise Exception(f"Ollama API request failed: {e}")
            
            # Extract and log the response
            result = response.json().get("response", "")
            log_response("ollama", result)
            logger.info("Generation completed successfully")
            
            return result
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Asynchronously generate a response using Ollama API.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
                - image: A single image (URL, file path, base64 string, or dict)
                - images: List of images (URLs, file paths, base64 strings, or dicts)
            
        Returns:
            The generated response or an async generator if streaming
            
        Raises:
            Exception: If the API call fails
        """
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters (using both string and enum keys for backwards compatibility)
        base_url = params.get(ModelParameter.BASE_URL, params.get("base_url", "http://localhost:11434"))
        model = params.get(ModelParameter.MODEL, params.get("model", "phi4-mini:latest"))
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        system_prompt_from_config = params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        system_prompt = system_prompt or system_prompt_from_config
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens"))
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        top_k = params.get(ModelParameter.TOP_K, params.get("top_k", 40))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Log at INFO level
        logger.info(f"Generating async response with Ollama model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}, top_k={top_k}")
        logger.debug(f"Using Ollama instance at: {base_url}")
        if system_prompt:
            logger.debug("Using system prompt")
        if stop:
            logger.debug(f"Using stop sequences: {stop}")
        
        # Check if model supports vision
        has_vision = any(vm in model.lower() for vm in VISION_CAPABLE_MODELS)
        
        # Process image inputs if any
        image_request = False
        if has_vision and (ModelParameter.IMAGE in params or "image" in params or 
                          ModelParameter.IMAGES in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = preprocess_image_inputs(params, "ollama")
        
        # Build the request
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
            "options": {}  # Additional options can go here
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
            
        # Add max tokens if provided
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
            
        # Add stop sequences if provided
        if stop:
            request_data["options"]["stop"] = stop if isinstance(stop, list) else [stop]
            
        # Include images if provided through preprocess_image_inputs
        if "images" in params:
            request_data["images"] = params["images"]
        
        # Log the request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request
        })
        
        # Log API request URL
        log_request_url("ollama", f"{base_url}/api/generate (model: {model})")
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting async streaming generation")
            
            async def async_generator():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{base_url}/api/generate",
                            json=request_data
                        ) as response:
                            response.raise_for_status()
                            async for line in response.content:
                                if line:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                except aiohttp.ClientError as e:
                    logger.error(f"Ollama API request failed: {e}")
                    raise Exception(f"Ollama API request failed: {e}")
            
            return async_generator()
        else:
            # Make the standard API request
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url}/api/generate",
                        json=request_data
                    ) as response:
                        response.raise_for_status()
                        response_json = await response.json()
            except aiohttp.ClientError as e:
                logger.error(f"Ollama API request failed: {e}")
                raise Exception(f"Ollama API request failed: {e}")
            
            # Extract and log the response
            result = response_json.get("response", "")
            log_response("ollama", result)
            logger.info("Async generation completed successfully")
            
            return result
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the Ollama provider.
        
        Returns:
            Dictionary of capabilities
        """
        # Get current model
        model = self.config.get(ModelParameter.MODEL, self.config.get("model", "llama2"))
        
        # Check if model is vision-capable - ensure model is not None before checking
        has_vision = False
        if model:
            has_vision = any(vm in model.lower() for vm in VISION_CAPABLE_MODELS)
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: None,  # Varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,  # Not supported natively
            ModelCapability.VISION: has_vision  # Depends on model
        } 