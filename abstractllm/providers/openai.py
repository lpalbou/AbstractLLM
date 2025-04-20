"""
OpenAI API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List, TYPE_CHECKING, AsyncIterator
import os
import logging
import asyncio
import json
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

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools.types import ToolCallRequest, ToolDefinition

# Try importing tools package directly
try:
    from abstractllm.tools import (
        ToolDefinition,
        ToolCallRequest,
        ToolCall,
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
    
    def _process_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert AbstractLLM tool definitions to OpenAI function format.
        
        Args:
            tools: List of tool definitions or callables
            
        Returns:
            List of OpenAI-formatted function definitions
        """
        if not TOOLS_AVAILABLE:
            raise ValueError("Tool support is not available. Install the required dependencies.")
            
        processed_tools = []
        
        for tool in tools:
            # If it's a callable, convert it to a tool definition
            if callable(tool):
                tool = function_to_tool_definition(tool)
                
            # If it's already a ToolDefinition, convert to dict format
            if isinstance(tool, ToolDefinition):
                tool = tool.to_dict()
                
            # If it's a dictionary, convert to OpenAI format
            if isinstance(tool, dict):
                # Validate that required fields are present
                if not all(k in tool for k in ['name', 'description', 'input_schema']):
                    logger.warning(f"Skipping invalid tool definition: {tool}")
                    continue
                    
                # Convert to OpenAI format
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool['name'],
                        "description": tool['description'],
                        "parameters": tool['input_schema']
                    }
                }
                processed_tools.append(openai_tool)
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
                
        return processed_tools
    
    def _check_for_tool_calls(self, response: Any) -> bool:
        """
        Check if a provider response contains tool calls.
        
        Args:
            response: The raw response from the provider
            
        Returns:
            True if the response contains tool calls, False otherwise
        """
        if not hasattr(response, "choices") or not response.choices:
            return False
            
        if not hasattr(response.choices[0], "message"):
            return False
            
        return hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls
    
    def _extract_tool_calls(self, response: Any) -> Optional["ToolCallRequest"]:
        """
        Extract tool calls from an OpenAI response.
        
        Args:
            response: Raw OpenAI response
            
        Returns:
            ToolCallRequest object if tool calls are present, None otherwise
        """
        if not TOOLS_AVAILABLE or not self._check_for_tool_calls(response):
            return None
            
        # Extract content from the response
        content = response.choices[0].message.content
        
        # Extract tool calls from the response
        tool_calls = []
        for tool_call in response.choices[0].message.tool_calls:
            # Parse arguments
            args = tool_call.function.arguments
            # Standardize argument handling
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call arguments: {args}")
                    args = {"_raw": args}
                
            # Create a tool call object
            tool_call_obj = ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=args
            )
            tool_calls.append(tool_call_obj)
            
        # Return a ToolCallRequest object
        return ToolCallRequest(
            content=content,
            tool_calls=tool_calls
        )
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                **kwargs) -> Union[str, Generator[str, None, None], Generator[Dict[str, Any], None, None]]:
        """
        Generate a response using OpenAI API.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (paths or URLs)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            **kwargs: Additional parameters to override configuration
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: A generator yielding response chunks
            
        Raises:
            Exception: If the generation fails
        """
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
        
        # Validate if tools are provided but not supported
        if tools:
            capabilities = self.get_capabilities()
            if not capabilities.get(ModelCapability.FUNCTION_CALLING, False):
                raise UnsupportedFeatureError(
                    "function_calling",
                    "Current model does not support function calling",
                    provider="openai"
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
            "has_files": bool(files),
            "has_tools": bool(tools)
        })
        
        # Process tools if provided
        processed_tools = None
        if tools:
            processed_tools = self._process_tools(tools)
        
        # Make API call
        try:
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # Add tools if available
            if processed_tools:
                api_params["tools"] = processed_tools
            
            # Make the API call
            completion = client.chat.completions.create(**api_params)
            
            if stream:
                def response_generator():
                    # Initialize variables to collect tool call information from chunks
                    collecting_tool_call = False
                    current_tool_calls = []
                    current_content = ""
                    
                    for chunk in completion:
                        # Extract content from delta if available
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            current_content += chunk.choices[0].delta.content
                            yield chunk.choices[0].delta.content
                            
                        # Check for tool calls in the delta
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            collecting_tool_call = True
                            
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                # Find if this tool_call is already in our list
                                existing_call = next((c for c in current_tool_calls if c.index == tool_call.index), None)
                                
                                if existing_call:
                                    # Append to existing tool call
                                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name') and tool_call.function.name:
                                        existing_call.function.name = tool_call.function.name
                                        
                                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                        if not hasattr(existing_call.function, 'arguments'):
                                            existing_call.function.arguments = ""
                                        existing_call.function.arguments += tool_call.function.arguments
                                else:
                                    # Add new tool call to our list
                                    current_tool_calls.append(tool_call)
                    
                    # At the end of streaming, if we have tool calls, yield a final response with them
                    if collecting_tool_call and current_tool_calls:
                        # Construct a final response object with the tool calls
                        final_response_obj = {
                            "content": current_content,
                            "tool_calls": [
                                {
                                    "id": f"call_{t.index}",
                                    "function": {
                                        "name": t.function.name,
                                        "arguments": t.function.arguments
                                    }
                                } for t in current_tool_calls if hasattr(t, 'function')
                            ]
                        }
                        # Create a proper ToolCallRequest object
                        from .types import ToolCallRequest, ToolCall
                        tool_calls = []
                        for tc in final_response_obj["tool_calls"]:
                            args = tc["function"]["arguments"]
                            # Standardize argument handling
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse tool call arguments: {args}")
                                    args = {"_raw": args}
                            
                            tool_calls.append(ToolCall(
                                id=tc["id"],
                                name=tc["function"]["name"],
                                arguments=args
                            ))
                        
                        yield ToolCallRequest(
                            content=current_content,
                            tool_calls=tool_calls
                        )
                
                return response_generator()
            else:
                # For non-streaming responses
                response_text = completion.choices[0].message.content
                
                # Check if response has tool calls
                if self._check_for_tool_calls(completion):
                    # Return the ToolCallRequest object directly
                    return self._extract_tool_calls(completion)
                else:
                    log_response("openai", response_text)
                    return response_text
                
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
                          tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                          **kwargs) -> Union[str, AsyncGenerator[str, None], AsyncGenerator[Dict[str, Any], None]]:
        """
        Asynchronously generate a response using OpenAI API.
        
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
        
        # Validate if tools are provided but not supported
        if tools:
            capabilities = self.get_capabilities()
            if not capabilities.get(ModelCapability.FUNCTION_CALLING, False):
                raise UnsupportedFeatureError(
                    "function_calling",
                    "Current model does not support function calling",
                    provider="openai"
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
        
        # Initialize AsyncOpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Log request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files),
            "has_tools": bool(tools)
        })
        
        # Process tools if provided
        processed_tools = None
        if tools:
            processed_tools = self._process_tools(tools)
        
        # Make API call
        try:
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # Add tools if available
            if processed_tools:
                api_params["tools"] = processed_tools
            
            # Make the API call
            completion = await client.chat.completions.create(**api_params)
            
            if stream:
                async def async_generator():
                    # Initialize variables to collect tool call information from chunks
                    collecting_tool_call = False
                    current_tool_calls = []
                    current_content = ""
                    
                    async for chunk in completion:
                        # Extract content from delta if available
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            current_content += chunk.choices[0].delta.content
                            yield chunk.choices[0].delta.content
                            
                        # Check for tool calls in the delta
                        if chunk.choices and hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            collecting_tool_call = True
                            
                            for tool_call in chunk.choices[0].delta.tool_calls:
                                # Find if this tool_call is already in our list
                                existing_call = next((c for c in current_tool_calls if c.index == tool_call.index), None)
                                
                                if existing_call:
                                    # Append to existing tool call
                                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name') and tool_call.function.name:
                                        existing_call.function.name = tool_call.function.name
                                        
                                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                        if not hasattr(existing_call.function, 'arguments'):
                                            existing_call.function.arguments = ""
                                        existing_call.function.arguments += tool_call.function.arguments
                                else:
                                    # Add new tool call to our list
                                    current_tool_calls.append(tool_call)
                    
                    # At the end of streaming, if we have tool calls, yield a final response with them
                    if collecting_tool_call and current_tool_calls:
                        # Construct a final response object with the tool calls
                        final_response_obj = {
                            "content": current_content,
                            "tool_calls": [
                                {
                                    "id": f"call_{t.index}",
                                    "function": {
                                        "name": t.function.name,
                                        "arguments": t.function.arguments
                                    }
                                } for t in current_tool_calls if hasattr(t, 'function')
                            ]
                        }
                        # Create a proper ToolCallRequest object
                        from .types import ToolCallRequest, ToolCall
                        tool_calls = []
                        for tc in final_response_obj["tool_calls"]:
                            args = tc["function"]["arguments"]
                            # Standardize argument handling
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse tool call arguments: {args}")
                                    args = {"_raw": args}
                            
                            tool_calls.append(ToolCall(
                                id=tc["id"],
                                name=tc["function"]["name"],
                                arguments=args
                            ))
                        
                        yield ToolCallRequest(
                            content=current_content,
                            tool_calls=tool_calls
                        )
                
                return async_generator()
            else:
                # For non-streaming responses
                response_text = completion.choices[0].message.content
                
                # Check if response has tool calls
                if self._check_for_tool_calls(completion):
                    # Return the ToolCallRequest object directly
                    return self._extract_tool_calls(completion)
                else:
                    log_response("openai", response_text)
                    return response_text
                
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
            ModelCapability.TOOL_USE: True,
            ModelCapability.VISION: has_vision,
            ModelCapability.JSON_MODE: True
        }

    async def _process_openai_stream(
        self, response: AsyncIterator
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process an OpenAI stream response asynchronously."""
        
        if not TOOLS_AVAILABLE:
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield {"text": chunk.choices[0].delta.content}
            return

        # Extract content and tool_calls incrementally
        content = ""
        tool_calls_by_id = {}  # Store tool calls by their ID
        
        async for chunk in response:
            # Extract the delta if it exists
            delta = chunk.choices[0].delta
            
            # Build content
            if hasattr(delta, "content") and delta.content is not None:
                content += delta.content
                yield ToolCallRequest(content=content)
            
            # Process tool calls if present
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tool_call in delta.tool_calls:
                    # Get or create the tool call in our tracking dict
                    if tool_call.index not in tool_calls_by_id:
                        tool_calls_by_id[tool_call.index] = {
                            "id": "",
                            "name": "",
                            "arguments": ""
                        }
                    
                    # Update the tool call properties
                    if hasattr(tool_call, "id") and tool_call.id:
                        tool_calls_by_id[tool_call.index]["id"] = tool_call.id
                    
                    if hasattr(tool_call, "function"):
                        function = tool_call.function
                        if hasattr(function, "name") and function.name:
                            tool_calls_by_id[tool_call.index]["name"] = function.name
                        
                        if hasattr(function, "arguments") and function.arguments:
                            tool_calls_by_id[tool_call.index]["arguments"] += function.arguments
                
                # Create tool call objects for completed ones
                completed_tool_calls = []
                for tool_call_data in tool_calls_by_id.values():
                    # Only include complete tool calls
                    if tool_call_data["id"] and tool_call_data["name"]:
                        # Parse arguments
                        args = tool_call_data["arguments"]
                        # Standardize argument handling
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tool call arguments: {args}")
                                args = {"_raw": args}
                        
                        completed_tool_calls.append(
                            ToolCall(
                                id=tool_call_data["id"],
                                name=tool_call_data["name"],
                                arguments=args
                            )
                        )
                
                # Yield a ToolCallRequest if we have any completed tool calls
                if completed_tool_calls:
                    yield ToolCallRequest(
                        content=content,
                        tool_calls=completed_tool_calls
                    )

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
