"""
Ollama API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List, TYPE_CHECKING
from pathlib import Path
import os
import json
import asyncio
import logging
import copy

# Check for required packages
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from abstractllm.interface import ModelParameter, ModelCapability
from abstractllm.providers.base import BaseProvider
from abstractllm.utils.logging import (
    log_request, 
    log_response,
    log_request_url,
    truncate_base64
)
from abstractllm.utils.model_capabilities import supports_tool_calls, supports_vision
from abstractllm.media.processor import MediaProcessor
from abstractllm.exceptions import ImageProcessingError, FileProcessingError, UnsupportedFeatureError, ProviderAPIError
from abstractllm.media.factory import MediaFactory
from abstractllm.media.image import ImageInput

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools.types import ToolCallRequest, ToolDefinition

# Try importing tools package directly
try:
    from abstractllm.tools import (
        ToolDefinition,
        ToolCallRequest,
        ToolCall,
        ToolCallResponse,
        function_to_tool_definition,
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    if not TYPE_CHECKING:
        class ToolDefinition:
            pass
        class ToolCallRequest:
            pass
        class ToolCall:
            pass
        class ToolCallResponse:
            pass

# Configure logger
logger = logging.getLogger("abstractllm.providers.ollama.OllamaProvider")

class OllamaProvider(BaseProvider):
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
        
        # Check if required dependencies are available
        if not REQUESTS_AVAILABLE:
            raise ImportError("The 'requests' package is required for OllamaProvider. Install with: pip install abstractllm[ollama]")
        
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
    
        
    def _check_for_tool_calls(self, response: Dict[str, Any]) -> bool:
        """
        Check if an Ollama response contains tool calls.
        
        Args:
            response: The raw response from the provider
            
        Returns:
            True if the response contains tool calls, False otherwise
        """
        # Check for tool_calls in the message field
        if isinstance(response.get("message", {}), dict):
            return "tool_calls" in response["message"] and response["message"]["tool_calls"]
        return False
    
    def _extract_tool_calls(self, response: Dict[str, Any]) -> Optional["ToolCallRequest"]:
        """
        Extract tool calls from an Ollama response.
        
        Args:
            response: Raw Ollama response
            
        Returns:
            ToolCallRequest object if tool calls are present, None otherwise
        """
        if not TOOLS_AVAILABLE or not self._check_for_tool_calls(response):
            return None
            
        # Extract content from response
        content = ""
        if isinstance(response.get("message", {}), dict):
            content = response["message"].get("content", "")
            
        # Extract tool calls from the response
        tool_calls = []
        for tc in response["message"].get("tool_calls", []):
            # Get function data - Ollama uses the OpenAI format with function nested
            function_data = tc.get("function", {})
            
            # Get name from function object (Ollama format) or directly (fallback)
            name = function_data.get("name", tc.get("name", ""))
            
            # Get arguments from function object (Ollama format) or directly (fallback)
            args = function_data.get("arguments", tc.get("parameters", tc.get("arguments", {})))
            
            # Parse arguments if needed
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call arguments: {args}")
                    args = {"_raw": args}
            
            # Create a tool call object
            tool_call_obj = ToolCall(
                id=tc.get("id", f"call_{len(tool_calls)}"),
                name=name,
                arguments=args
            )
            tool_calls.append(tool_call_obj)
            
        # Return a ToolCallRequest object
        return ToolCallRequest(
            content=content,
            tool_calls=tool_calls
        )
    
    def _supports_tool_calls(self) -> bool:
        """
        Check if the configured model supports tool calls.
        
        Returns:
            True if the current model supports tool calls, False otherwise
        """
        model = self.config_manager.get_param(ModelParameter.MODEL)
        return supports_tool_calls(model)
        
    def _prepare_request_for_chat(self, 
                                 model: str,
                                 prompt: str,
                                 system_prompt: Optional[str],
                                 processed_files: List[Any],
                                 processed_tools: Optional[List[Dict[str, Any]]],
                                 temperature: float,
                                 max_tokens: int,
                                 stream: bool) -> Dict[str, Any]:
        """
        Prepare request data for the Ollama chat API endpoint.
        
        Args:
            model: The model to use
            prompt: The user prompt
            system_prompt: Optional system prompt
            processed_files: List of processed media files
            processed_tools: List of processed tools
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary of request data for the chat API endpoint
        """
        # Base request structure
        request_data = {
            "model": model,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif processed_tools:
            # If tools are provided but no system prompt, add a tool-encouraging system prompt
            messages.append({
                "role": "system", 
                "content": "You are a helpful assistant. When you need to access information or perform operations, use the available tools."
            })
            
        # Prepare user message content
        images = []
        for media_input in processed_files:
            if isinstance(media_input, ImageInput):
                images.append(media_input.to_provider_format("ollama"))
                
        # Create user message
        user_message = {"role": "user", "content": prompt}
        if images:
            user_message["images"] = images
            
        messages.append(user_message)
        
        # Add messages to request data
        request_data["messages"] = messages
        
        # Add tools if provided
        if processed_tools:
            request_data["tools"] = processed_tools
            
        return request_data
    
    def _prepare_request_for_generate(self,
                                    model: str,
                                    prompt: str,
                                    system_prompt: Optional[str],
                                    processed_files: List[Any],
                                    temperature: float,
                                    max_tokens: int,
                                    stream: bool) -> Dict[str, Any]:
        """
        Prepare request data for the Ollama generate API endpoint.
        
        Args:
            model: The model to use
            prompt: The user prompt
            system_prompt: Optional system prompt
            processed_files: List of processed media files
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary of request data for the generate API endpoint
        """
        # Base request structure
        request_data = {
            "model": model,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Handle files
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
        
        return request_data
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                **kwargs) -> Union[str, Generator[str, None, None], Generator[Dict[str, Any], None, None]]:
        """
        Generate a response using Ollama API.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (paths or URLs)
                  Supported types: images (for vision models), text, markdown, CSV, TSV
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
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
        
        # Validate if tools are provided but not supported
        if tools:
            if not self._supports_tool_calls():
                raise UnsupportedFeatureError(
                    "function_calling",
                    "Current model does not support function calling",
                    provider="ollama"
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
                        provider="ollama",
                        original_exception=e
                    )
        
        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not supports_vision(model):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="ollama"
            )
        
        # Log request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files),
            "has_tools": bool(tools)
        })
        
        # Handle tools using base class methods
        enhanced_system_prompt = system_prompt
        formatted_tools = None
        tool_mode = "none"
        
        if tools:
            # Use base class method to prepare tool context
            enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, system_prompt)
            
            # For Ollama, we check if the model supports native tools
            # Tool defs are already formatted by _prepare_tool_context
            if tool_mode == "native" and tool_defs:
                handler = self._get_tool_handler()
                if handler:
                    formatted_tools = tool_defs  # Already in correct format
        
        # Determine if we should use the chat endpoint
        use_chat_endpoint = formatted_tools is not None

        # Select endpoint and prepare request
        if use_chat_endpoint:
            endpoint = f"{base_url.rstrip('/')}/api/chat"
            request_data = self._prepare_request_for_chat(
                model=model,
                prompt=prompt,
                system_prompt=enhanced_system_prompt,
                processed_files=processed_files,
                processed_tools=formatted_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        else:
            endpoint = f"{base_url.rstrip('/')}/api/generate"
            request_data = self._prepare_request_for_generate(
                model=model,
                prompt=prompt,
                system_prompt=enhanced_system_prompt,
                processed_files=processed_files,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        # Log API request URL
        log_request_url("ollama", endpoint)
        
        # Make API call
        try:
            if stream:
                def response_generator():
                    # Initialize variables for tool call collection
                    collecting_tool_call = False
                    current_tool_calls = []
                    current_content = ""
                    
                    response = requests.post(endpoint, json=request_data, stream=True)
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                
                                # Handle generate endpoint response
                                if "response" in data:
                                    current_content += data["response"]
                                    yield data["response"]
                                # Handle chat endpoint response with tool calls
                                elif "message" in data and isinstance(data["message"], dict):
                                    # Extract content if available
                                    if "content" in data["message"]:
                                        content_chunk = data["message"]["content"]
                                        current_content += content_chunk
                                        yield content_chunk
                                        
                                    # Collect tool calls if present
                                    if "tool_calls" in data["message"] and data["message"]["tool_calls"]:
                                        collecting_tool_call = True
                                        
                                        # Add or update tool calls
                                        for tool_call in data["message"]["tool_calls"]:
                                            current_tool_calls.append(tool_call)
                                # Check for completion
                                elif "done" in data and data["done"]:
                                    # At the end of streaming, yield tool calls if any
                                    if collecting_tool_call and current_tool_calls:
                                        # Create a proper ToolCallRequest object
                                        tool_calls = []
                                        for tc in current_tool_calls:
                                            # Parse arguments if needed
                                            args = tc.get("parameters", tc.get("arguments", {}))
                                            # Standardize argument handling
                                            if isinstance(args, str):
                                                try:
                                                    args = json.loads(args)
                                                except json.JSONDecodeError:
                                                    logger.warning(f"Failed to parse tool call arguments: {args}")
                                                    args = {"_raw": args}
                                            
                                            tool_calls.append(ToolCall(
                                                id=tc.get("id", f"call_{len(tool_calls)}"),
                                                name=tc.get("name", ""),
                                                arguments=args
                                            ))
                                        
                                        # Yield the ToolCallRequest object
                                        yield ToolCallRequest(
                                            content=current_content,
                                            tool_calls=tool_calls
                                        )
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON from Ollama response: {line}")
                                
                    return response_generator()
                
                return response_generator()
            else:
                response = requests.post(endpoint, json=request_data)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract content from response
                content = None
                if "response" in data:
                    content = data["response"]
                elif "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                    content = data["message"]["content"]
                else:
                    logger.error(f"Unexpected response format: {data}")
                    raise ValueError("Unexpected response format from Ollama API")
                
                # Extract tool calls using the handler
                handler = self._get_tool_handler()
                if handler:
                    if tool_mode == "native" and formatted_tools and self._check_for_tool_calls(data):
                        # For native mode, parse the response
                        tool_response = handler.parse_response(data, mode="native")
                    else:
                        # Use prompted extraction
                        tool_response = handler.parse_response(content, mode="prompted") if content else None
                else:
                    tool_response = None
                
                # Return appropriate response
                if tool_response and tool_response.has_tool_calls():
                    from abstractllm.types import GenerateResponse
                    return GenerateResponse(
                        content=content,
                        tool_calls=tool_response,
                        model=model
                    )
                else:
                    log_response("ollama", content)
                    return content
                    
        except requests.RequestException as e:
            logger.error(f"Network error during Ollama API request: {str(e)}")
            raise ProviderAPIError(
                f"Failed to connect to Ollama API: {str(e)}",
                provider="ollama",
                original_exception=e
            )
    
    async def generate_async(self,
        prompt: str,
        system_prompt: Optional[str] = None,
        files: Optional[List[Union[str, Path]]] = None,
        stream: bool = False,
        tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None], AsyncGenerator[Dict[str, Any], None]]:
        """
        Asynchronously generate a response using Ollama API.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (paths or URLs)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            **kwargs: Additional parameters to override configuration
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: An async generator yielding response chunks
            
        Raises:
            Exception: If the generation fails
        """
        # Check if aiohttp is available for async operations
        if not AIOHTTP_AVAILABLE:
            raise ImportError("The 'aiohttp' package is required for async operations. Install with: pip install abstractllm[ollama]")
            
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Get necessary parameters from config
        model = self.config_manager.get_param(ModelParameter.MODEL)
        temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE)
        max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS)
        base_url = self.config_manager.get_param(ModelParameter.BASE_URL)
        
        # Validate if tools are provided but not supported
        if tools:
            if not self._supports_tool_calls():
                raise UnsupportedFeatureError(
                    "function_calling",
                    "Current model does not support function calling",
                    provider="ollama"
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
                        provider="ollama",
                        original_exception=e
                    )
        
        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not supports_vision(model):
            raise UnsupportedFeatureError(
                "vision",
                "Current model does not support vision input",
                provider="ollama"
            )
        
        # Log request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files),
            "has_tools": bool(tools)
        })
        
        # Handle tools using base class methods
        enhanced_system_prompt = system_prompt
        formatted_tools = None
        tool_mode = "none"
        
        if tools:
            # Use base class method to prepare tool context
            enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, system_prompt)
            
            # For Ollama, we check if the model supports native tools
            # Tool defs are already formatted by _prepare_tool_context
            if tool_mode == "native" and tool_defs:
                handler = self._get_tool_handler()
                if handler:
                    formatted_tools = tool_defs  # Already in correct format
        
        # Determine if we should use the chat endpoint
        use_chat_endpoint = formatted_tools is not None

        # Select endpoint and prepare request
        if use_chat_endpoint:
            endpoint = f"{base_url.rstrip('/')}/api/chat"
            request_data = self._prepare_request_for_chat(
                model=model,
                prompt=prompt,
                system_prompt=enhanced_system_prompt,
                processed_files=processed_files,
                processed_tools=formatted_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        else:
            endpoint = f"{base_url.rstrip('/')}/api/generate"
            request_data = self._prepare_request_for_generate(
                model=model,
                prompt=prompt,
                system_prompt=enhanced_system_prompt,
                processed_files=processed_files,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        # Log API request URL
        log_request_url("ollama", endpoint)
        
        try:
            async with aiohttp.ClientSession() as session:
                if stream:
                    async with session.post(endpoint, json=request_data) as response:
                        response.raise_for_status()
                        
                        async def response_generator():
                            # Initialize variables for tool call collection
                            collecting_tool_call = False
                            current_tool_calls = []
                            current_content = ""
                            
                            async for line in response.content:
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    
                                    # Handle generate endpoint response
                                    if "response" in data:
                                        current_content += data["response"]
                                        yield data["response"]
                                    # Handle chat endpoint response with tool calls
                                    elif "message" in data and isinstance(data["message"], dict):
                                        # Extract content if available
                                        if "content" in data["message"]:
                                            content_chunk = data["message"]["content"]
                                            current_content += content_chunk
                                            yield content_chunk
                                            
                                        # Collect tool calls if present
                                        if "tool_calls" in data["message"] and data["message"]["tool_calls"]:
                                            collecting_tool_call = True
                                            
                                            # Add or update tool calls
                                            for tool_call in data["message"]["tool_calls"]:
                                                current_tool_calls.append(tool_call)
                                    # Check for completion
                                    elif "done" in data and data["done"]:
                                        # At the end of streaming, yield tool calls if any
                                        if collecting_tool_call and current_tool_calls:
                                            # Create a proper ToolCallRequest object
                                            tool_calls = []
                                            for tc in current_tool_calls:
                                                # Parse arguments if needed
                                                args = tc.get("parameters", tc.get("arguments", {}))
                                                # Standardize argument handling
                                                if isinstance(args, str):
                                                    try:
                                                        args = json.loads(args)
                                                    except json.JSONDecodeError:
                                                        logger.warning(f"Failed to parse tool call arguments: {args}")
                                                        args = {"_raw": args}
                                                
                                                tool_calls.append(ToolCall(
                                                    id=tc.get("id", f"call_{len(tool_calls)}"),
                                                    name=tc.get("name", ""),
                                                    arguments=args
                                                ))
                                            
                                            # Yield the ToolCallRequest object
                                            yield ToolCallRequest(
                                                content=current_content,
                                                tool_calls=tool_calls
                                            )
                                        break
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse streaming response: {e}")
                                    continue
                                    
                        return response_generator()
                else:
                    async with session.post(endpoint, json=request_data) as response:
                        response.raise_for_status()
                        data = await response.json()
                        
                        # Extract content from response
                        content = None
                        if "response" in data:
                            content = data["response"]
                        elif "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                            content = data["message"]["content"]
                        else:
                            logger.error(f"Unexpected response format: {data}")
                            raise ValueError("Unexpected response format from Ollama API")
                        
                        # Extract tool calls based on mode
                        if tool_mode == "native" and formatted_tools:
                            # For native mode, check if response has tool calls
                            if self._check_for_tool_calls(data):
                                # Use the existing _extract_tool_calls which returns ToolCallRequest
                                tool_response_old = self._extract_tool_calls(data)
                                if tool_response_old:
                                    # Convert to new format
                                    from abstractllm.tools import ToolCallResponse
                                    tool_response = ToolCallResponse(
                                        content=tool_response_old.content,
                                        tool_calls=tool_response_old.tool_calls
                                    )
                                else:
                                    tool_response = None
                            else:
                                tool_response = None
                        else:
                            # Use prompted extraction
                            tool_response = self._extract_tool_calls(content) if content else None
                        
                        # Return appropriate response
                        if tool_response and tool_response.has_tool_calls():
                            from abstractllm.types import GenerateResponse
                            return GenerateResponse(
                                content=content,
                                tool_calls=tool_response,
                                model=model
                            )
                        else:
                            log_response("ollama", content)
                            return content

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
            ModelCapability.TOOL_USE: False,
            ModelCapability.VISION: False
        }
        
        # Get current model
        model = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check if the current model supports vision
        has_vision = supports_vision(model)
        
        # Check if the current model supports tool calls
        has_tool_calls = supports_tool_calls(model)
        
        # Update capabilities
        if has_vision:
            capabilities[ModelCapability.VISION] = True
            
        if has_tool_calls:
            capabilities[ModelCapability.FUNCTION_CALLING] = True
            capabilities[ModelCapability.TOOL_USE] = True
            
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