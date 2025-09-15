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
import time
import re

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
from abstractllm.architectures.detection import supports_tools as supports_tool_calls, supports_vision
from abstractllm.media.processor import MediaProcessor
from abstractllm.exceptions import ImageProcessingError, FileProcessingError, UnsupportedFeatureError, ProviderAPIError
from abstractllm.media.factory import MediaFactory
from abstractllm.media.image import ImageInput

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools.types import ToolCallResponse, ToolDefinition

# Try importing tools package directly
try:
    from abstractllm.tools import (
        ToolDefinition,
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
        class ToolCall:
            pass
        class ToolCallResponse:
            pass

# Configure logger
logger = logging.getLogger("abstractllm.providers.ollama.OllamaProvider")

def _approximate_token_count(text: str) -> int:
    """
    Approximate token count for text.
    Uses a simple heuristic: ~4 characters per token for most models.
    This provides reasonable estimates when exact token counts aren't available.
    """
    if not text:
        return 0
    
    # Remove extra whitespace and count characters
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    
    # Rough approximation: 4 characters per token
    # This is conservative but reasonable for most languages
    return max(1, len(cleaned_text) // 4)


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
    
    def _extract_tool_calls(self, response: Dict[str, Any]) -> Optional["ToolCallResponse"]:
        """
        Extract tool calls from an Ollama response.
        
        Args:
            response: Raw Ollama response
            
        Returns:
            ToolCallResponse object if tool calls are present, None otherwise
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
            
        # Create the response object
        response = ToolCallResponse(
            content=content,
            tool_calls=tool_calls
        )
        
        # Note: Tool call logging is handled at session level for universal coverage
        return response
    
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
                                 stream: bool,
                                 provided_messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
        # Check for user-configured max input tokens first
        user_max_tokens = self.config_manager.get_param(ModelParameter.MAX_INPUT_TOKENS)

        if user_max_tokens:
            # User has set a specific context limit
            context_length = user_max_tokens
            logger.info(f"Using user-configured context size: {context_length:,} tokens for {model}")
        else:
            # Fall back to model's default context length
            from abstractllm.architectures.detection import get_context_length
            context_length = get_context_length(model)
            logger.info(f"Using model default context size: {context_length:,} tokens for {model}")

        # Apply reasonable maximum to prevent memory issues
        MAX_SAFE_CONTEXT = 1_000_000  # 1M tokens max
        if context_length > MAX_SAFE_CONTEXT:
            logger.warning(f"Context length of {context_length:,} tokens exceeds safe limit, capping at {MAX_SAFE_CONTEXT:,}")
            context_length = MAX_SAFE_CONTEXT
        
        # Base request structure
        request_data = {
            "model": model,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": context_length  # Set context size to model's full capacity
            }
        }

        # Add seed for deterministic generation if specified
        seed = self.config_manager.get_param(ModelParameter.SEED)
        if seed is not None:
            request_data["options"]["seed"] = seed
            logger.info(f"Set Ollama seed to {seed} for deterministic generation")
        
        # Prepare messages
        if provided_messages:
            # Use provided messages - convert Message objects to dicts if needed
            messages = []
            
            # Check if we need to inject/update system prompt for tools
            has_system_in_provided = any(
                (msg.get('role') if isinstance(msg, dict) else getattr(msg, 'role', None)) == 'system' 
                for msg in provided_messages
            )
            
            for i, msg in enumerate(provided_messages):
                if isinstance(msg, dict):
                    role = msg.get('role')
                    content = msg.get('content')
                else:
                    # Handle Message objects
                    role = getattr(msg, 'role', 'user')
                    content = getattr(msg, 'content', str(msg))
                
                # If this is the system message and we have an enhanced system prompt, use it
                if role == 'system' and system_prompt and i == 0:
                    messages.append({"role": "system", "content": system_prompt})
                else:
                    messages.append({"role": role, "content": content})
            
            # If no system message was provided but we have an enhanced system prompt, prepend it
            if not has_system_in_provided and system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
                
            # Add the current user prompt if not already in messages
            # (This handles the case where prompt is a new message not in provided_messages)
            if prompt and not any(
                (msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)) == prompt 
                for msg in provided_messages
            ):
                # Prepare user message content
                images = []
                for media_input in processed_files:
                    if isinstance(media_input, ImageInput):
                        images.append(media_input.to_provider_format("ollama"))
                        
                user_message = {"role": "user", "content": prompt}
                if images:
                    user_message["images"] = images
                    
                messages.append(user_message)
            
            # OLLAMA FIX: Handle consecutive assistant messages properly
            # The issue is that tool calls create two consecutive assistant messages:
            # 1. Assistant message with tool call
            # 2. Assistant message with tool output (formatted as "TOOL OUTPUT [tool_name]: ...")
            # 
            # The model gets confused by consecutive assistant messages. The fix is to convert
            # tool output messages to user messages, since tool outputs are information
            # being provided TO the assistant, not FROM the assistant.
            fixed_messages = []
            for msg in messages:
                if (fixed_messages and 
                    fixed_messages[-1]["role"] == "assistant" and 
                    msg["role"] == "assistant" and
                    msg["content"].startswith("TOOL OUTPUT [")):
                    # Convert tool output to user message
                    tool_output_msg = {
                        "role": "user",
                        "content": f"Tool result: {msg['content'][msg['content'].find(']:') + 2:].strip()}"
                    }
                    fixed_messages.append(tool_output_msg)
                    logger.info(f"[OLLAMA FIX] Converted tool output to user message: {len(tool_output_msg['content'])} chars")
                else:
                    fixed_messages.append(msg)
            
            # Log the final message structure for debugging
            logger.debug(f"[OLLAMA FIX] Final message count: {len(fixed_messages)}")
            for i, msg in enumerate(fixed_messages):
                logger.debug(f"[OLLAMA FIX] Message {i}: role={msg['role']}, content_length={len(msg['content'])}")
            
            messages = fixed_messages
        else:
            # Create new messages from prompt/system_prompt
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
        # Check for user-configured max input tokens first
        user_max_tokens = self.config_manager.get_param(ModelParameter.MAX_INPUT_TOKENS)

        if user_max_tokens:
            # User has set a specific context limit
            context_length = user_max_tokens
            logger.info(f"Using user-configured context size: {context_length:,} tokens for {model}")
        else:
            # Fall back to model's default context length
            from abstractllm.architectures.detection import get_context_length
            context_length = get_context_length(model)
            logger.info(f"Using model default context size: {context_length:,} tokens for {model}")

        # Apply reasonable maximum to prevent memory issues
        MAX_SAFE_CONTEXT = 1_000_000  # 1M tokens max
        if context_length > MAX_SAFE_CONTEXT:
            logger.warning(f"Context length of {context_length:,} tokens exceeds safe limit, capping at {MAX_SAFE_CONTEXT:,}")
            context_length = MAX_SAFE_CONTEXT
        
        # Base request structure
        request_data = {
            "model": model,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": context_length  # Set context size to model's full capacity
            }
        }

        # Add seed for deterministic generation if specified
        seed = self.config_manager.get_param(ModelParameter.SEED)
        if seed is not None:
            request_data["options"]["seed"] = seed
            logger.info(f"Set Ollama seed to {seed} for deterministic generation")
        
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
    
    def _generate_impl(self, 
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
        # Extract messages if provided (for conversation history)
        messages = kwargs.pop('messages', None)
        
        # Update config with any remaining kwargs (filter out None values)
        if kwargs:
            # Filter out None values to avoid overriding existing config
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if filtered_kwargs:
                self.config_manager.update_config(filtered_kwargs)
        
        # Track timing for metrics
        start_time = time.time()
        
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
        
        # Log request using shared method - before processing tools
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            has_files=bool(files),
            files_count=len(files) if files else 0
        )
        
        # Handle tools using base class methods
        enhanced_system_prompt = system_prompt
        formatted_tools = None
        tool_mode = "none"
        
        if tools:
            # Use base class method to prepare tool context
            enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, system_prompt)
            
            # Ollama-specific override: Some models claim native tool support but work better with prompted mode
            # This is because while the model supports native tools, Ollama's implementation may not be optimal
            model_name = self.config_manager.get_param(ModelParameter.MODEL, "").lower()
            
            # List of models that should use prompted mode despite claiming native support
            force_prompted_models = [
                "qwen3:4b", "qwen3-4b", "qwen3:7b", "qwen3-7b",  # Qwen3 models
                "qwen2.5:7b", "qwen2.5-7b", "qwen2.5:14b", "qwen2.5-14b",  # Some Qwen2.5 models
            ]
            
            if tool_mode == "native" and any(model_pattern in model_name for model_pattern in force_prompted_models):
                logger.info(f"[OLLAMA OVERRIDE] Model {model_name} claims native tool support but forcing prompted mode for better compatibility")
                # Re-prepare with forced prompted mode
                handler = self._get_tool_handler()
                if handler and handler.supports_prompted:
                    # Force prompted mode by preparing tools for prompted format
                    processed_tools = self._process_tools(tools)
                    tool_prompt = handler.format_tools_prompt(processed_tools)
                    
                    # Combine with existing system prompt
                    if system_prompt:
                        enhanced_system_prompt = f"{system_prompt}\n\n{tool_prompt}"
                    else:
                        enhanced_system_prompt = tool_prompt
                    
                    tool_mode = "prompted"
                    tool_defs = None  # No native tool definitions needed for prompted mode
                    logger.info(f"[OLLAMA OVERRIDE] Successfully switched to prompted mode, enhanced system prompt length: {len(enhanced_system_prompt)}")
            
            # Log the tool preparation results
            logger.info(f"Tool context prepared: mode={tool_mode}, tools={len(tool_defs) if tool_defs else len(tools)}")
            if tool_mode == "prompted":
                logger.info(f"[TOOL SETUP] Using PROMPTED mode for tools")
                logger.info(f"[TOOL SETUP] Enhanced system prompt length: {len(enhanced_system_prompt) if enhanced_system_prompt else 0}")
                if enhanced_system_prompt:
                    logger.debug(f"[TOOL SETUP] Full enhanced system prompt:\n{enhanced_system_prompt}")
            
            # For Ollama, we check if the model supports native tools
            # Tool defs are already formatted by _prepare_tool_context
            if tool_mode == "native" and tool_defs:
                handler = self._get_tool_handler()
                if handler:
                    formatted_tools = tool_defs  # Already in correct format
        
        # Determine if we should use the chat endpoint
        # Use chat endpoint if we have tools (either native or prompted) or messages
        use_chat_endpoint = (formatted_tools is not None) or (tools and tool_mode in ["native", "prompted"]) or messages is not None

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
                stream=stream,
                provided_messages=messages
            )
            
            # Debug logging for tool support
            logger.info(f"Using chat endpoint with tool_mode={tool_mode}")
            if request_data.get('messages'):
                for i, msg in enumerate(request_data['messages']):
                    if msg.get('role') == 'system':
                        logger.debug(f"System message {i}: {msg['content'][:200]}...")
                    elif msg.get('role') == 'user':
                        logger.debug(f"User message {i}: {msg['content'][:100]}...")
            if request_data.get('tools'):
                logger.debug(f"Native tools in request: {len(request_data['tools'])} tools")
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
        
        # Log the complete request details after all processing
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt if not tools else None,  # Avoid duplicate system prompt logging
            messages=messages,
            tools=tools,
            formatted_messages=request_data.get("messages", []),
            request_data=request_data,
            endpoint=endpoint,
            enhanced_system_prompt=enhanced_system_prompt if tools else None,
            tool_mode=tool_mode,
            formatted_tools=formatted_tools,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config_manager.get_param(ModelParameter.TOP_P),
            formatted_prompt=request_data.get("prompt") if endpoint.endswith("/generate") else None
        )
        
        # Make API call
        try:
            if stream:
                def response_generator():
                    # Initialize variables for tool call collection
                    collecting_tool_call = False
                    current_tool_calls = []
                    current_content = ""
                    
                    # Capture verbatim context sent to LLM
                    verbatim_context = json.dumps(request_data, indent=2, ensure_ascii=False)
                    self._capture_verbatim_context(f"ENDPOINT: {endpoint}\n\nREQUEST PAYLOAD:\n{verbatim_context}")

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
                                    # At the end of streaming, check for tool calls
                                    tool_response = None
                                    
                                    # First check if we collected native tool calls
                                    if collecting_tool_call and current_tool_calls:
                                        # Create a proper ToolCallResponse object
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
                                        
                                        tool_response = ToolCallResponse(
                                            content=current_content,
                                            tool_calls=tool_calls
                                        )
                                    
                                    # If no native tool calls, check for prompted tool calls in content
                                    if not tool_response and current_content and tool_mode in ["native", "prompted"]:
                                        handler = self._get_tool_handler()
                                        if handler:
                                            logger.debug(f"Checking for prompted tool calls in streamed content")
                                            prompted_response = handler.parse_response(current_content, mode="prompted")
                                            if prompted_response and prompted_response.has_tool_calls():
                                                logger.debug(f"Found {len(prompted_response.tool_calls)} prompted tool calls")
                                                tool_response = prompted_response
                                    
                                    # Yield the tool response if we found any tool calls
                                    if tool_response:
                                        yield tool_response
                                    
                                    break
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON from Ollama response: {line}")
                                
                    return response_generator()
                
                return response_generator()
            else:
                # Capture verbatim context sent to LLM
                verbatim_context = json.dumps(request_data, indent=2, ensure_ascii=False)
                self._capture_verbatim_context(f"ENDPOINT: {endpoint}\n\nREQUEST PAYLOAD:\n{verbatim_context}")

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
                
                # Extract tool calls using the appropriate method
                tool_response = None
                
                # First try native tool call extraction if in native mode
                if tool_mode == "native" and formatted_tools and self._check_for_tool_calls(data):
                    logger.debug(f"Parsing native tool response: {data}")
                    tool_response = self._extract_tool_calls(data)
                    if tool_response and tool_response.has_tool_calls():
                        logger.info(f"[NATIVE TOOL PARSING] SUCCESS: Found {len(tool_response.tool_calls)} tool calls")
                        for tc in tool_response.tool_calls:
                            logger.info(f"[NATIVE TOOL PARSING] Tool call: {tc.name}({tc.arguments})")
                
                # If no tool calls found in native mode, try prompted parsing
                # This handles cases where models output tool calls in text even when using native API
                if not tool_response or not tool_response.has_tool_calls():
                    handler = self._get_tool_handler()
                    if handler and content:
                        logger.info(f"[TOOL PARSING] Checking content for prompted tool calls")
                        logger.debug(f"[TOOL PARSING] Full content to parse: {content}")
                        prompted_response = handler.parse_response(content, mode="prompted")
                        if prompted_response and prompted_response.has_tool_calls():
                            logger.info(f"[TOOL PARSING] SUCCESS: Found {len(prompted_response.tool_calls)} tool calls")
                            for tc in prompted_response.tool_calls:
                                logger.info(f"[TOOL PARSING] Tool call: {tc.name}({tc.arguments})")
                            tool_response = prompted_response
                        else:
                            logger.info("[TOOL PARSING] NO TOOL CALLS FOUND in content")
                    else:
                        logger.debug(f"No prompted tool parsing: handler={bool(handler)}, has_content={bool(content)}")
                
                # Return appropriate response
                if tool_response and tool_response.has_tool_calls():
                    logger.debug(f"Tool response has tool calls: {tool_response.tool_calls}")
                    # Calculate approximate token counts and timing
                    prompt_tokens = _approximate_token_count(prompt)
                    completion_tokens = _approximate_token_count(content)
                    total_time = time.time() - start_time
                    
                    # Log response with tool calls
                    self._log_response_details(
                        data, 
                        content, 
                        has_tool_calls=True, 
                        tool_calls=tool_response.tool_calls,
                        model=model,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens, 
                            "total_tokens": prompt_tokens + completion_tokens,
                            "total_time": total_time
                        }
                    )
                    from abstractllm.types import GenerateResponse
                    return GenerateResponse(
                        content=content,
                        tool_calls=tool_response,
                        model=model,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens, 
                            "total_tokens": prompt_tokens + completion_tokens,
                            "total_time": total_time
                        }
                    )
                else:
                    logger.debug(f"No tool calls detected in response. tool_response={tool_response}")
                    # Calculate approximate token counts and timing
                    prompt_tokens = _approximate_token_count(prompt)
                    completion_tokens = _approximate_token_count(content)
                    total_time = time.time() - start_time
                    
                    # Log response without tool calls
                    self._log_response_details(
                        data, 
                        content, 
                        has_tool_calls=False,
                        model=model,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens, 
                            "total_time": total_time
                        }
                    )
                    # Always return a GenerateResponse for consistency
                    from abstractllm.types import GenerateResponse
                    return GenerateResponse(
                        content=content,
                        tool_calls=None,
                        model=model,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens, 
                            "total_time": total_time
                        }
                    )
                    
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
            
        # Extract messages if provided (for conversation history)
        messages = kwargs.pop('messages', None)
        
        # Update config with any remaining kwargs (filter out None values)
        if kwargs:
            # Filter out None values to avoid overriding existing config
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            if filtered_kwargs:
                self.config_manager.update_config(filtered_kwargs)
        
        # Track timing for metrics
        start_time = time.time()
        
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
        
        # Log request using shared method - before processing tools
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            has_files=bool(files),
            files_count=len(files) if files else 0
        )
        
        # Handle tools using base class methods
        enhanced_system_prompt = system_prompt
        formatted_tools = None
        tool_mode = "none"
        
        if tools:
            # Use base class method to prepare tool context
            enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, system_prompt)
            
            # Ollama-specific override: Some models claim native tool support but work better with prompted mode
            # This is because while the model supports native tools, Ollama's implementation may not be optimal
            model_name = self.config_manager.get_param(ModelParameter.MODEL, "").lower()
            
            # List of models that should use prompted mode despite claiming native support
            force_prompted_models = [
                "qwen3:4b", "qwen3-4b", "qwen3:7b", "qwen3-7b",  # Qwen3 models
                "qwen2.5:7b", "qwen2.5-7b", "qwen2.5:14b", "qwen2.5-14b",  # Some Qwen2.5 models
            ]
            
            if tool_mode == "native" and any(model_pattern in model_name for model_pattern in force_prompted_models):
                logger.info(f"[OLLAMA OVERRIDE] Model {model_name} claims native tool support but forcing prompted mode for better compatibility")
                # Re-prepare with forced prompted mode
                handler = self._get_tool_handler()
                if handler and handler.supports_prompted:
                    # Force prompted mode by preparing tools for prompted format
                    processed_tools = self._process_tools(tools)
                    tool_prompt = handler.format_tools_prompt(processed_tools)
                    
                    # Combine with existing system prompt
                    if system_prompt:
                        enhanced_system_prompt = f"{system_prompt}\n\n{tool_prompt}"
                    else:
                        enhanced_system_prompt = tool_prompt
                    
                    tool_mode = "prompted"
                    tool_defs = None  # No native tool definitions needed for prompted mode
                    logger.info(f"[OLLAMA OVERRIDE] Successfully switched to prompted mode, enhanced system prompt length: {len(enhanced_system_prompt)}")
            
            # Log the tool preparation results
            logger.info(f"Tool context prepared: mode={tool_mode}, tools={len(tool_defs) if tool_defs else len(tools)}")
            if tool_mode == "prompted":
                logger.info(f"[TOOL SETUP] Using PROMPTED mode for tools")
                logger.info(f"[TOOL SETUP] Enhanced system prompt length: {len(enhanced_system_prompt) if enhanced_system_prompt else 0}")
                if enhanced_system_prompt:
                    logger.debug(f"[TOOL SETUP] Full enhanced system prompt:\n{enhanced_system_prompt}")
            
            # For Ollama, we check if the model supports native tools
            # Tool defs are already formatted by _prepare_tool_context
            if tool_mode == "native" and tool_defs:
                handler = self._get_tool_handler()
                if handler:
                    formatted_tools = tool_defs  # Already in correct format
        
        # Determine if we should use the chat endpoint
        # Use chat endpoint if we have tools (either native or prompted) or messages
        use_chat_endpoint = (formatted_tools is not None) or (tools and tool_mode in ["native", "prompted"]) or messages is not None

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
                stream=stream,
                provided_messages=messages
            )
            
            # Debug logging for tool support
            logger.info(f"Using chat endpoint with tool_mode={tool_mode}")
            if request_data.get('messages'):
                for i, msg in enumerate(request_data['messages']):
                    if msg.get('role') == 'system':
                        logger.debug(f"System message {i}: {msg['content'][:200]}...")
                    elif msg.get('role') == 'user':
                        logger.debug(f"User message {i}: {msg['content'][:100]}...")
            if request_data.get('tools'):
                logger.debug(f"Native tools in request: {len(request_data['tools'])} tools")
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
        
        
        # Capture verbatim context sent to LLM
        verbatim_context = json.dumps(request_data, indent=2, ensure_ascii=False)
        self._capture_verbatim_context(f"ENDPOINT: {endpoint}\n\nREQUEST PAYLOAD:\n{verbatim_context}")

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
                                            # Create a proper ToolCallResponse object
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
                                tool_response = self._extract_tool_calls(data)
                            else:
                                tool_response = None
                        else:
                            # Use prompted extraction
                            tool_response = self._extract_tool_calls(content) if content else None
                        
                        # Return appropriate response
                        if tool_response and tool_response.has_tool_calls():
                            # Calculate approximate token counts and timing
                            prompt_tokens = _approximate_token_count(prompt)
                            completion_tokens = _approximate_token_count(content)
                            total_time = time.time() - start_time
                            
                            # Log response with tool calls
                            self._log_response_details(
                                data, 
                                content, 
                                has_tool_calls=True, 
                                tool_calls=tool_response.tool_calls,
                                model=model,
                                usage={
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                    "total_time": total_time
                                }
                            )
                            from abstractllm.types import GenerateResponse
                            return GenerateResponse(
                                content=content,
                                tool_calls=tool_response,
                                model=model,
                                usage={
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                    "total_time": total_time
                                }
                            )
                        else:
                            # Calculate approximate token counts and timing
                            prompt_tokens = _approximate_token_count(prompt)
                            completion_tokens = _approximate_token_count(content)
                            total_time = time.time() - start_time
                            
                            # Log response without tool calls
                            self._log_response_details(
                                data, 
                                content, 
                                has_tool_calls=False,
                                model=model,
                                usage={
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                    "total_time": total_time
                                }
                            )
                            from abstractllm.types import GenerateResponse
                            return GenerateResponse(
                                content=content,
                                tool_calls=None,
                                model=model,
                                usage={
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens,
                                    "total_time": total_time
                                }
                            )

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