"""
Anthropic API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List, TYPE_CHECKING
import os
import logging
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from abstractllm.interface import ModelParameter, ModelCapability
from abstractllm.providers.base import BaseProvider
from abstractllm.types import GenerateResponse
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
from abstractllm.architectures.detection import supports_tools as supports_tool_calls

# Check if Anthropic package is available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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
logger = logging.getLogger("abstractllm.providers.anthropic.AnthropicProvider")

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    # Claude 3 models
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    
    # Claude 3.5 models
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
    
    # Claude 3.7 models
    "claude-3-7-sonnet-20250219"
]

class AnthropicProvider(BaseProvider):
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
        
        # Check if required dependencies are available
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package is required for AnthropicProvider. Install with: pip install abstractllm[anthropic]")
        
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
    
    
    def _format_tools_for_provider(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for Anthropic's specific API requirements.
        
        Anthropic expects tools in the format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {...}  # parameters with JSON schema
        }
        """
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"]
            })
        return formatted_tools
    
    def _check_for_tool_calls(self, response: Any) -> bool:
        """
        Check if an Anthropic response contains tool calls.
        
        Args:
            response: The raw response from the provider
            
        Returns:
            True if the response contains tool calls, False otherwise
        """
        if not hasattr(response, "content"):
            return False
            
        # Check for tool_use blocks in the content
        return any(block.type == "tool_use" for block in response.content if hasattr(block, "type"))
    
    def _extract_tool_calls(self, response: Any) -> Optional["ToolCallRequest"]:
        """
        Extract tool calls from an Anthropic response.
        
        Args:
            response: Raw Anthropic response
            
        Returns:
            ToolCallRequest object if tool calls are present, None otherwise
        """
        if not TOOLS_AVAILABLE or not self._check_for_tool_calls(response):
            return None
            
        # Extract content blocks by type
        text_content = []
        tool_calls = []
        
        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
            elif block.type == "tool_use":
                # Create a tool call object
                tool_call_obj = ToolCall(
                    id=getattr(block, "id", f"call_{len(tool_calls)}"),
                    name=block.name,
                    arguments=block.input
                )
                tool_calls.append(tool_call_obj)
        
        # Join text content
        content = "".join(text_content)
        
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
        logger.debug(f"Checking tool call support for model: {model}")
        
        # If model is None, return False to avoid 'NoneType' has no attribute 'startswith' error
        if model is None:
            logger.warning("Model is None, cannot determine tool call support")
            return False
            
        # Use centralized capability checking
        has_support = supports_tool_calls(model)
        logger.debug(f"Tool call support for {model}: {has_support}")
        return has_support
    
    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize messages to avoid trailing whitespace errors from Anthropic API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Sanitized message list
        """
        sanitized_messages = []
        
        for message in messages:
            sanitized_message = message.copy()
            
            # Handle different content formats
            if isinstance(message.get('content'), str):
                # Simple string content
                sanitized_message['content'] = message['content'].strip()
            elif isinstance(message.get('content'), list):
                # Content list with different types of blocks
                sanitized_content = []
                for content_block in message['content']:
                    if isinstance(content_block, dict):
                        sanitized_block = content_block.copy()
                        # Handle text blocks
                        if content_block.get('type') == 'text' and 'text' in content_block:
                            sanitized_block['text'] = content_block['text'].strip()
                        sanitized_content.append(sanitized_block)
                    else:
                        # Non-dict content (shouldn't happen in normal usage)
                        sanitized_content.append(content_block)
                sanitized_message['content'] = sanitized_content
            
            sanitized_messages.append(sanitized_message)
            
        return sanitized_messages
    
    def _generate_impl(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None, 
                      files: Optional[List[Union[str, Path]]] = None,
                      stream: bool = False,
                      tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                      **kwargs) -> Union[str, Generator[str, None, None], Generator[Dict[str, Any], None, None]]:
        """
        Generate a response using Anthropic API.
        
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
        
        # Get SOTA parameters (Anthropic supports subset of OpenAI parameters)
        top_p = self.config_manager.get_param(ModelParameter.TOP_P)
        stop = self.config_manager.get_param(ModelParameter.STOP)
        
        # Check for API key
        if not api_key:
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Validate if tools are provided but not supported
        if tools:
            if not self._supports_tool_calls():
                raise UnsupportedFeatureError(
                    "function_calling",
                    "Current model does not support function calling",
                    provider="anthropic"
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
        
        # Handle tools using base class methods
        enhanced_system_prompt = system_prompt or self.config_manager.get_param(ModelParameter.SYSTEM_PROMPT)
        formatted_tools = None
        tool_mode = "none"
        
        if tools:
            # Use base class method to prepare tool context
            enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, enhanced_system_prompt)
            
            # If native mode, format tools for Anthropic API
            if tool_mode == "native" and tool_defs:
                handler = self._get_tool_handler()
                if handler:
                    formatted_tools = tool_defs  # Tool defs are already formatted by _prepare_tool_context
        
        # Prepare messages
        messages = []
        
        # Check if messages are provided in kwargs
        if 'messages' in kwargs and kwargs['messages']:
            messages = kwargs['messages']
        else:
            # Prepare user message content
            content = []
            
            # Add files first if any
            if processed_files:
                for media_input in processed_files:
                    content.append(media_input.to_provider_format("anthropic"))
            
            # Add text prompt after files, only if not empty
            if prompt and prompt.strip():
                content.append({
                    "type": "text",
                    "text": prompt
                })
            
            # Ensure there's at least one content item with non-whitespace text
            if not content:
                content.append({
                    "type": "text",
                    "text": "Hello"  # Use a minimal valid non-whitespace text
                })
            
            # Add the user message with the content array
            messages.append({
                "role": "user",
                "content": content
            })
        
        # Log request using base class method
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt,
            enhanced_system_prompt=enhanced_system_prompt,
            messages=messages,
            tools=tools,
            formatted_messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            has_files=bool(files),
            tool_mode=tool_mode,
            endpoint="https://api.anthropic.com/v1/messages"
        )
        
        # Sanitize messages to avoid trailing whitespace errors
        messages = self._sanitize_messages(messages)
        
        # Make API call
        try:
            # Create message with system prompt if provided
            message_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens if max_tokens is not None else 2048,
                "temperature": temperature if temperature is not None else 0.7,
                "stream": stream
            }
            
            # Add SOTA parameters if specified
            if top_p is not None:
                message_params["top_p"] = top_p
            if stop is not None:
                message_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
            
            if enhanced_system_prompt:
                message_params["system"] = enhanced_system_prompt
                
            # Add tools if available
            if formatted_tools:
                logger.debug(f"Adding tools to message params: {formatted_tools}")
                message_params["tools"] = formatted_tools
            
            if stream:
                def response_generator():
                    # Initialize variables to collect content and tool calls
                    collecting_tool_call = False
                    current_tool_calls = []
                    current_content = ""
                
                    # Remove 'stream' flag before calling the stream method
                    sync_params = message_params.copy()
                    sync_params.pop("stream", None)

                    # Capture verbatim context sent to LLM
                    import json
                    verbatim_context = json.dumps(sync_params, indent=2, ensure_ascii=False)
                    self._capture_verbatim_context(f"ANTHROPIC API ENDPOINT: messages (streaming)\n\nREQUEST PAYLOAD:\n{verbatim_context}")

                    with client.messages.stream(**sync_params) as stream:
                        for chunk in stream:
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                                current_content += chunk.delta.text
                                yield chunk.delta.text
                            
                            # For tool calls, collect them but don't yield until the end
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'tool_use'):
                                collecting_tool_call = True
                                tool_use = chunk.delta.tool_use
                                
                                # Extract tool call ID
                                tool_call_id = getattr(tool_use, 'id', None)
                                if not tool_call_id:
                                    # Generate a unique ID if none provided
                                    tool_call_id = f"call_{len(current_tool_calls)}"
                                    
                                # Find existing tool call or create a new one
                                existing_call_idx = next((i for i, c in enumerate(current_tool_calls) 
                                                         if getattr(c, 'id', None) == tool_call_id), None)
                                
                                if existing_call_idx is not None:
                                    # Update existing call with new information
                                    existing_call = current_tool_calls[existing_call_idx]
                                    
                                    if hasattr(tool_use, 'name') and tool_use.name:
                                        existing_call.name = tool_use.name
                                    
                                    if hasattr(tool_use, 'input') and tool_use.input:
                                        existing_call.input = tool_use.input
                                else:
                                    # Create a new placeholder for this tool call
                                    tool_call_obj = SimpleNamespace()
                                    tool_call_obj.id = tool_call_id
                                    
                                    if hasattr(tool_use, 'name'):
                                        tool_call_obj.name = tool_use.name
                                    else:
                                        tool_call_obj.name = ""
                                        
                                    if hasattr(tool_use, 'input'):
                                        tool_call_obj.input = tool_use.input
                                    else:
                                        tool_call_obj.input = {}
                                        
                                    current_tool_calls.append(tool_call_obj)
                    
                    # At the end of streaming, yield tool calls if any
                    if collecting_tool_call and current_tool_calls:
                        # Create a standardized tool call list
                        tool_calls = []
                        for tc in current_tool_calls:
                            # Skip incomplete tool calls
                            if not hasattr(tc, 'name') or not tc.name:
                                logger.warning(f"Skipping incomplete tool call: {tc}")
                                continue
                                
                            try:
                                # Create a proper ToolCall object
                                tool_call_obj = ToolCall(
                                    id=tc.id,
                                    name=tc.name,
                                    arguments=tc.input
                                )
                                tool_calls.append(tool_call_obj)
                            except Exception as e:
                                logger.error(f"Error creating tool call: {e}")
                        
                        if tool_calls:
                            # Yield the ToolCallRequest with complete tool calls
                            logger.debug(f"Yielding tool call request with {len(tool_calls)} tool calls")
                            yield ToolCallRequest(
                                content=current_content,
                                tool_calls=tool_calls
                            )
                
                return response_generator()
            else:
                # Capture verbatim context sent to LLM
                import json
                verbatim_context = json.dumps(message_params, indent=2, ensure_ascii=False)
                self._capture_verbatim_context(f"ANTHROPIC API ENDPOINT: messages\n\nREQUEST PAYLOAD:\n{verbatim_context}")

                response = client.messages.create(**message_params)
                
                # Extract content from response
                if response.content and response.content[0].type == "text":
                    content = response.content[0].text
                else:
                    content = ""
                
                # Extract tool calls based on mode
                if tool_mode == "native":
                    # Extract tool calls using handler for native mode
                    handler = self._get_tool_handler()
                    if handler and self._check_for_tool_calls(response):
                        # For Anthropic, create a proper response structure
                        tool_response = handler.parse_response({
                            "content": content,
                            "tool_calls": [
                                {
                                    "id": getattr(block, "id", f"call_{i}"),
                                    "name": block.name,
                                    "arguments": block.input
                                }
                                for i, block in enumerate(response.content)
                                if hasattr(block, "type") and block.type == "tool_use"
                            ] if hasattr(response, "content") else []
                        }, mode="native")
                    else:
                        tool_response = None
                else:
                    # Use prompted extraction
                    handler = self._get_tool_handler()
                    if handler:
                        tool_response = handler.parse_response(content, mode="prompted")
                    else:
                        tool_response = None
                
                # Return appropriate response
                if tool_response and tool_response.has_tool_calls():
                    return GenerateResponse(
                        content=content,
                        tool_calls=tool_response,
                        model=model,
                        usage=response.usage.model_dump() if hasattr(response, 'usage') else None
                    )
                else:
                    # Log response using base class method
                    self._log_response_details(
                        response,
                        content,
                        has_tool_calls=False,
                        model=model,
                        usage=response.usage.model_dump() if hasattr(response, 'usage') else None
                    )
                    # Return a GenerateResponse object for consistency
                    return GenerateResponse(
                        content=content,
                        raw_response=response,
                        model=model
                    )
                
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
                          tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                          **kwargs) -> Union[str, AsyncGenerator[str, None], AsyncGenerator[Dict[str, Any], None]]:
        """
        Asynchronously generate a response using Anthropic API.
        
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
        
        # Get SOTA parameters (Anthropic supports subset of OpenAI parameters)
        top_p = self.config_manager.get_param(ModelParameter.TOP_P)
        stop = self.config_manager.get_param(ModelParameter.STOP)
        
        # Check for API key
        if not api_key:
            log_api_key_missing("Anthropic", "ANTHROPIC_API_KEY")
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter in config or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Validate if tools are provided but not supported
        if tools:
            if not self._supports_tool_calls():
                raise UnsupportedFeatureError(
                    "function_calling",
                    "Current model does not support function calling",
                    provider="anthropic"
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
        
        # Handle tools using base class methods
        enhanced_system_prompt = system_prompt or self.config_manager.get_param(ModelParameter.SYSTEM_PROMPT)
        formatted_tools = None
        tool_mode = "none"
        
        if tools:
            # Use base class method to prepare tool context
            enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, enhanced_system_prompt)
            
            # If native mode, format tools for Anthropic API
            if tool_mode == "native" and tool_defs:
                handler = self._get_tool_handler()
                if handler:
                    formatted_tools = tool_defs  # Tool defs are already formatted by _prepare_tool_context
        
        # Prepare messages
        messages = []
        
        # Check if messages are provided in kwargs
        if 'messages' in kwargs and kwargs['messages']:
            messages = kwargs['messages']
        else:
            # Prepare user message content
            content = []
            
            # Add files first if any
            if processed_files:
                for media_input in processed_files:
                    content.append(media_input.to_provider_format("anthropic"))
            
            # Add text prompt after files, only if not empty
            if prompt and prompt.strip():
                content.append({
                    "type": "text",
                    "text": prompt
                })
            
            # Ensure there's at least one content item with non-whitespace text
            if not content:
                content.append({
                    "type": "text",
                    "text": "Hello"  # Use a minimal valid non-whitespace text
                })
            
            # Add the user message with the content array
            messages.append({
                "role": "user",
                "content": content
            })
        
        # Log request using base class method
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt,
            enhanced_system_prompt=enhanced_system_prompt,
            messages=messages,
            tools=tools,
            formatted_messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            has_files=bool(files),
            tool_mode=tool_mode,
            endpoint="https://api.anthropic.com/v1/messages"
        )
        
        # Sanitize messages to avoid trailing whitespace errors
        messages = self._sanitize_messages(messages)
        
        # Make API call
        try:
            # Create message with system prompt if provided
            message_params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens if max_tokens is not None else 2048,
                "temperature": temperature if temperature is not None else 0.7,
                "stream": stream
            }
            
            # Add SOTA parameters if specified
            if top_p is not None:
                message_params["top_p"] = top_p
            if stop is not None:
                message_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
            
            if enhanced_system_prompt:
                message_params["system"] = enhanced_system_prompt
                
            # Add tools if available
            if formatted_tools:
                logger.debug(f"Adding tools to message params: {formatted_tools}")
                message_params["tools"] = formatted_tools
            
            if stream:
                async def async_generator():
                    # Initialize variables to collect content and tool calls
                    collecting_tool_call = False
                    current_tool_calls = []
                    current_content = ""
                
                    # Remove 'stream' flag before calling the stream method
                    async_params = message_params.copy()
                    async_params.pop("stream", None)
                    async with client.messages.stream(**async_params) as stream:
                        async for chunk in stream:
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                                current_content += chunk.delta.text
                                yield chunk.delta.text
                            
                            # For tool calls, collect them but don't yield until the end
                            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'tool_use'):
                                collecting_tool_call = True
                                tool_use = chunk.delta.tool_use
                                
                                # Extract tool call ID
                                tool_call_id = getattr(tool_use, 'id', None)
                                if not tool_call_id:
                                    # Generate a unique ID if none provided
                                    tool_call_id = f"call_{len(current_tool_calls)}"
                                    
                                # Find existing tool call or create a new one
                                existing_call_idx = next((i for i, c in enumerate(current_tool_calls) 
                                                         if getattr(c, 'id', None) == tool_call_id), None)
                                
                                if existing_call_idx is not None:
                                    # Update existing call with new information
                                    existing_call = current_tool_calls[existing_call_idx]
                                    
                                    if hasattr(tool_use, 'name') and tool_use.name:
                                        existing_call.name = tool_use.name
                                    
                                    if hasattr(tool_use, 'input') and tool_use.input:
                                        existing_call.input = tool_use.input
                                else:
                                    # Create a new placeholder for this tool call
                                    tool_call_obj = SimpleNamespace()
                                    tool_call_obj.id = tool_call_id
                                    
                                    if hasattr(tool_use, 'name'):
                                        tool_call_obj.name = tool_use.name
                                    else:
                                        tool_call_obj.name = ""
                                        
                                    if hasattr(tool_use, 'input'):
                                        tool_call_obj.input = tool_use.input
                                    else:
                                        tool_call_obj.input = {}
                                        
                                    current_tool_calls.append(tool_call_obj)
                    
                    # At the end of streaming, yield tool calls if any
                    if collecting_tool_call and current_tool_calls:
                        # Create a standardized tool call list
                        tool_calls = []
                        for tc in current_tool_calls:
                            # Skip incomplete tool calls
                            if not hasattr(tc, 'name') or not tc.name:
                                logger.warning(f"Skipping incomplete tool call: {tc}")
                                continue
                                
                            try:
                                # Create a proper ToolCall object
                                tool_call_obj = ToolCall(
                                    id=tc.id,
                                    name=tc.name,
                                    arguments=tc.input
                                )
                                tool_calls.append(tool_call_obj)
                            except Exception as e:
                                logger.error(f"Error creating tool call: {e}")
                        
                        if tool_calls:
                            # Yield the ToolCallRequest with complete tool calls
                            logger.debug(f"Yielding tool call request with {len(tool_calls)} tool calls")
                            yield ToolCallRequest(
                                content=current_content,
                                tool_calls=tool_calls
                            )
                
                return async_generator()
            else:
                # Capture verbatim context sent to LLM
                import json
                verbatim_context = json.dumps(message_params, indent=2, ensure_ascii=False)
                self._capture_verbatim_context(f"ANTHROPIC API ENDPOINT: messages (async)\n\nREQUEST PAYLOAD:\n{verbatim_context}")

                response = await client.messages.create(**message_params)
                
                # Extract content from response
                if response.content and response.content[0].type == "text":
                    content = response.content[0].text
                else:
                    content = ""
                
                # Extract tool calls based on mode
                if tool_mode == "native":
                    # Extract tool calls using handler for native mode
                    handler = self._get_tool_handler()
                    if handler and self._check_for_tool_calls(response):
                        # For Anthropic, create a proper response structure
                        tool_response = handler.parse_response({
                            "content": content,
                            "tool_calls": [
                                {
                                    "id": getattr(block, "id", f"call_{i}"),
                                    "name": block.name,
                                    "arguments": block.input
                                }
                                for i, block in enumerate(response.content)
                                if hasattr(block, "type") and block.type == "tool_use"
                            ] if hasattr(response, "content") else []
                        }, mode="native")
                    else:
                        tool_response = None
                else:
                    # Use prompted extraction
                    handler = self._get_tool_handler()
                    if handler:
                        tool_response = handler.parse_response(content, mode="prompted")
                    else:
                        tool_response = None
                
                # Return appropriate response
                if tool_response and tool_response.has_tool_calls():
                    return GenerateResponse(
                        content=content,
                        tool_calls=tool_response,
                        model=model,
                        usage=response.usage.model_dump() if hasattr(response, 'usage') else None
                    )
                else:
                    # Log response using base class method
                    self._log_response_details(
                        response,
                        content,
                        has_tool_calls=False,
                        model=model,
                        usage=response.usage.model_dump() if hasattr(response, 'usage') else None
                    )
                    # Return a GenerateResponse object for consistency
                    return GenerateResponse(
                        content=content,
                        raw_response=response,
                        model=model
                    )
                
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
        
        # Check if model supports tool calls
        has_tool_calls = self._supports_tool_calls()
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 100000,  # Claude 3 models support large outputs
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: has_tool_calls,
            ModelCapability.TOOL_USE: has_tool_calls,
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