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
import copy

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import (
    log_request, 
    log_response,
    log_request_url,
    truncate_base64
)
from abstractllm.utils.image import preprocess_image_inputs
from abstractllm.utils.config import ConfigurationManager

# Configure logger
logger = logging.getLogger("abstractllm.providers.ollama.OllamaProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "llama3.2-vision",
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
        
        # Initialize provider-specific configuration
        self.config = ConfigurationManager.initialize_provider_config("ollama", self.config)
        
        # Log provider initialization
        model = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "phi4-mini:latest")
        base_url = ConfigurationManager.get_param(self.config, ModelParameter.BASE_URL, "http://localhost:11434")
        logger.info(f"Initialized Ollama provider with model: {model}, base URL: {base_url}")
    
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
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "ollama", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        base_url = params.get("base_url", "http://localhost:11434")
        model = params.get("model", "phi4-mini:latest")
        temperature = params.get("temperature", 0.7)
        system_prompt = params.get("system_prompt")
        
        # Log at INFO level
        logger.info(f"Generating response with Ollama model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Using Ollama instance at: {base_url}")
        logger.debug(f"Generation parameters: temperature={temperature}")
        if system_prompt:
            logger.debug("Using system prompt")
        
        # Check if model supports vision
        has_vision = any(vision_model in model.lower() for vision_model in [vm.lower() for vm in VISION_CAPABLE_MODELS])
        
        # Process image inputs if any, and if model supports vision
        image_request = False
        if has_vision and ("image" in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = preprocess_image_inputs(params, "ollama")
        
        # Determine endpoint based on whether this is a chat-format request or completion request
        format_type = params.pop("format", "chat")
        
        if format_type == "chat":
            # Chat API format
            if "messages" in params:
                # If messages are already formatted (e.g., by image preprocessing)
                messages = params.pop("messages", [])
            else:
                # Format message array
                messages = []
                
                # Add system message if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                # Add user message
                messages.append({"role": "user", "content": prompt})
            
            # Build the request
            request_data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            # Add images parameter at the root level if present (needed for vision capabilities)
            if "images" in params:
                request_data["images"] = params.pop("images")
            
            # Add any additional options (removing any that might be duplicated)
            for key, value in params.items():
                if key not in ["model", "messages", "stream"] and key not in request_data["options"]:
                    request_data["options"][key] = value
            
            endpoint = f"{base_url.rstrip('/')}/api/chat"
        else:
            # Completion API format (legacy format)
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_data["system"] = system_prompt
            
            # Add any additional options
            for key, value in params.items():
                if key not in ["model", "prompt", "stream", "system"] and key not in request_data["options"]:
                    request_data["options"][key] = value
            
            endpoint = f"{base_url.rstrip('/')}/api/generate"
        
        # Log the request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request,
            "format": format_type
        })
        
        # Log API request URL
        log_request_url("ollama", endpoint)
        
        # For image requests, create a sanitized version of request data for debug logging
        if image_request and logger.isEnabledFor(logging.DEBUG):
            sanitized_request = copy.deepcopy(request_data)
            if "images" in sanitized_request:
                image_count = len(sanitized_request["images"])
                sanitized_request["images"] = f"[{image_count} image(s), truncated for logging]"
            logger.debug(f"Ollama request data (sanitized): {sanitized_request}")
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting streaming generation")
            
            if format_type == "chat":
                def chat_response_generator():
                    try:
                        response = requests.post(endpoint, json=request_data, stream=True)
                        response.raise_for_status()
                        
                        # Process the streamed response
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "message" in data and "content" in data["message"]:
                                        yield data["message"]["content"]
                                    elif "done" in data and data["done"]:
                                        # End of stream
                                        break
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON from Ollama response: {line}")
                    except requests.RequestException as e:
                        logger.error(f"Ollama API request failed: {str(e)}")
                        raise
                
                return chat_response_generator()
            else:
                def response_generator():
                    try:
                        response = requests.post(endpoint, json=request_data, stream=True)
                        response.raise_for_status()
                        
                        # Process the streamed response
                        for line in response.iter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                    elif "done" in data and data["done"]:
                                        # End of stream
                                        break
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON from Ollama response: {line}")
                    except requests.RequestException as e:
                        logger.error(f"Ollama API request failed: {str(e)}")
                        raise
                
                return response_generator()
        else:
            # Standard non-streaming response
            try:
                # Set stream to False for non-streaming request
                request_data["stream"] = False
                
                # Make the API request
                response = requests.post(endpoint, json=request_data)
                response.raise_for_status()
                data = response.json()
                
                # Extract result based on format
                if format_type == "chat" and "message" in data and "content" in data["message"]:
                    result = data["message"]["content"]
                elif "response" in data:
                    result = data["response"]
                else:
                    logger.error(f"Unexpected response format from Ollama: {data}")
                    raise ValueError(f"Unexpected response format from Ollama: {data}")
                
                log_response("ollama", result)
                logger.info("Generation completed successfully")
                
                return result
            except requests.RequestException as e:
                logger.error(f"Ollama API request failed: {str(e)}")
                raise
    
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
            If stream=False: The complete generated response as a string
            If stream=True: An async generator yielding response chunks
            
        Raises:
            Exception: If the API call fails
        """
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "ollama", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        base_url = params.get("base_url", "http://localhost:11434")
        model = params.get("model", "phi4-mini:latest")
        temperature = params.get("temperature", 0.7)
        system_prompt = params.get("system_prompt")
        
        # Log at INFO level
        logger.info(f"Generating async response with Ollama model: {model}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}")
        if system_prompt:
            logger.debug("Using system prompt")
        
        # Check if model supports vision
        has_vision = any(vision_model in model.lower() for vision_model in [vm.lower() for vm in VISION_CAPABLE_MODELS])
        
        # Process image inputs if any, and if model supports vision
        image_request = False
        if has_vision and ("image" in params or "images" in params):
            logger.info("Processing image inputs for vision request")
            image_request = True
            params = preprocess_image_inputs(params, "ollama")
        
        # Determine endpoint based on whether this is a chat-format request or completion request
        format_type = params.pop("format", "chat")
        
        if format_type == "chat":
            # Chat API format
            if "messages" in params:
                # If messages are already formatted (e.g., by image preprocessing)
                messages = params.pop("messages", [])
            else:
                # Format message array
                messages = []
                
                # Add system message if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                # Add user message
                messages.append({"role": "user", "content": prompt})
            
            # Build the request
            request_data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            # Add images parameter at the root level if present (needed for vision capabilities)
            if "images" in params:
                request_data["images"] = params.pop("images")
            
            # Add any additional options (removing any that might be duplicated)
            for key, value in params.items():
                if key not in ["model", "messages", "stream"] and key not in request_data["options"]:
                    request_data["options"][key] = value
            
            endpoint = f"{base_url.rstrip('/')}/api/chat"
        else:
            # Completion API format (legacy format)
            request_data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_data["system"] = system_prompt
            
            # Add any additional options
            for key, value in params.items():
                if key not in ["model", "prompt", "stream", "system"] and key not in request_data["options"]:
                    request_data["options"][key] = value
            
            endpoint = f"{base_url.rstrip('/')}/api/generate"
        
        # Log the request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": image_request,
            "format": format_type
        })
        
        # Log API request URL
        log_request_url("ollama", endpoint)
        
        # For image requests, create a sanitized version of request data for debug logging
        if image_request and logger.isEnabledFor(logging.DEBUG):
            sanitized_request = copy.deepcopy(request_data)
            if "images" in sanitized_request:
                image_count = len(sanitized_request["images"])
                sanitized_request["images"] = f"[{image_count} image(s), truncated for logging]"
            logger.debug(f"Ollama async request data (sanitized): {sanitized_request}")
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting async streaming generation")
            
            if format_type == "chat":
                async def async_chat_generator():
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(endpoint, json=request_data) as response:
                                response.raise_for_status()
                                
                                # Process the streamed response
                                async for line in response.content.iter_any():
                                    if line:
                                        line_str = line.decode('utf-8').strip()
                                        if line_str:
                                            for sub_line in line_str.split('\n'):
                                                if sub_line:
                                                    try:
                                                        data = json.loads(sub_line)
                                                        if "message" in data and "content" in data["message"]:
                                                            yield data["message"]["content"]
                                                        elif "done" in data and data["done"]:
                                                            # End of stream
                                                            return
                                                    except json.JSONDecodeError:
                                                        logger.warning(f"Failed to parse JSON from Ollama response: {sub_line}")
                    except aiohttp.ClientError as e:
                        logger.error(f"Ollama API request failed: {str(e)}")
                        raise
                
                return async_chat_generator()
            else:
                async def async_generator():
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(endpoint, json=request_data) as response:
                                response.raise_for_status()
                                
                                # Process the streamed response
                                async for line in response.content.iter_any():
                                    if line:
                                        line_str = line.decode('utf-8').strip()
                                        if line_str:
                                            for sub_line in line_str.split('\n'):
                                                if sub_line:
                                                    try:
                                                        data = json.loads(sub_line)
                                                        if "response" in data:
                                                            yield data["response"]
                                                        elif "done" in data and data["done"]:
                                                            # End of stream
                                                            return
                                                    except json.JSONDecodeError:
                                                        logger.warning(f"Failed to parse JSON from Ollama response: {sub_line}")
                    except aiohttp.ClientError as e:
                        logger.error(f"Ollama API request failed: {str(e)}")
                        raise
                
                return async_generator()
        else:
            # Standard non-streaming response
            try:
                # Set stream to False for non-streaming request
                request_data["stream"] = False
                
                # Make the API request
                async with aiohttp.ClientSession() as session:
                    async with session.post(endpoint, json=request_data) as response:
                        response.raise_for_status()
                        data = await response.json()
                
                # Extract result based on format
                if format_type == "chat" and "message" in data and "content" in data["message"]:
                    result = data["message"]["content"]
                elif "response" in data:
                    result = data["response"]
                else:
                    logger.error(f"Unexpected response format from Ollama: {data}")
                    raise ValueError(f"Unexpected response format from Ollama: {data}")
                
                log_response("ollama", result)
                logger.info("Async generation completed successfully")
                
                return result
            except aiohttp.ClientError as e:
                logger.error(f"Ollama API request failed: {str(e)}")
                raise
    
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
        model = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "phi4-mini:latest")
        has_vision = any(vision_model in model.lower() for vision_model in [vm.lower() for vm in VISION_CAPABLE_MODELS])
        
        # Update vision capability
        if has_vision:
            capabilities[ModelCapability.VISION] = True
            
        return capabilities 