"""
LM Studio API implementation for AbstractLLM.

This provider connects to LM Studio's OpenAI-compatible API server,
providing seamless integration with local models while maintaining
full compatibility with AbstractLLM's unified memory system,
cognitive enhancements, and tool orchestration.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List, TYPE_CHECKING
from pathlib import Path
import os
import json
import asyncio
import logging
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
from abstractllm.types import GenerateResponse
from abstractllm.utils.logging import (
    log_request,
    log_response,
    log_request_url
)
from abstractllm.media.factory import MediaFactory
from abstractllm.media.image import ImageInput
from abstractllm.exceptions import (
    UnsupportedFeatureError,
    FileProcessingError,
    ProviderAPIError,
    ModelLoadingError
)

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
logger = logging.getLogger("abstractllm.providers.lmstudio.LMStudioProvider")

def _approximate_token_count(text: str) -> int:
    """
    Approximate token count for text using LM Studio patterns.
    Uses a conservative heuristic similar to Ollama provider.
    """
    if not text:
        return 0

    # Remove extra whitespace and count characters
    cleaned_text = re.sub(r'\s+', ' ', text.strip())

    # Conservative approximation: 4 characters per token
    return max(1, len(cleaned_text) // 4)

# Vision-capable model patterns (common ones supported by LM Studio)
VISION_MODEL_PATTERNS = [
    "llava", "vision", "visual", "multimodal", "vit", "clip",
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "phi-vision",
    "pixtral", "molmo", "paligemma", "idefics", "blip"
]

class LMStudioProvider(BaseProvider):
    """
    LM Studio API implementation using OpenAI-compatible endpoints.

    This provider connects to a local LM Studio server, providing
    full integration with AbstractLLM's cognitive architecture while
    leveraging LM Studio's model management and inference capabilities.
    """

    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the LM Studio provider.

        Args:
            config: Configuration dictionary with LM Studio-specific parameters
        """
        super().__init__(config)

        # Check if required dependencies are available
        if not REQUESTS_AVAILABLE:
            raise ImportError("The 'requests' package is required for LMStudioProvider. Install with: pip install abstractllm[lmstudio]")

        # Start with provider-specific base defaults
        default_config = {
            ModelParameter.MODEL: "qwen/qwen3-next-80b",  # Reasonable default
            ModelParameter.BASE_URL: "http://localhost:1234/v1",  # LM Studio OpenAI-compatible endpoint
            ModelParameter.API_KEY: "lm-studio",  # LM Studio accepts any API key
            ModelParameter.TRUNCATION_STRATEGY: "stopAtLimit",  # LM Studio's context overflow policy
        }

        # Get model name (could be from config or default)
        model_name = (config or {}).get(ModelParameter.MODEL) or default_config[ModelParameter.MODEL]

        # Use architecture detection for intelligent model-specific defaults
        try:
            from abstractllm.architectures.detection import get_model_capabilities
            capabilities = get_model_capabilities(model_name)

            if capabilities:
                # Use model-specific capabilities as intelligent defaults
                context_length = capabilities.get('context_length')
                max_output = capabilities.get('max_output_tokens')

                if context_length:
                    default_config[ModelParameter.MAX_INPUT_TOKENS] = context_length
                    logger.debug(f"Using model-specific context length: {context_length}")

                if max_output:
                    # For LM Studio, prefer higher output tokens if model supports it
                    # Use the higher of model capability or LM Studio optimized default
                    lmstudio_optimized = 8192
                    default_config[ModelParameter.MAX_TOKENS] = max(max_output, lmstudio_optimized)
                    logger.debug(f"Using optimized output tokens: {default_config[ModelParameter.MAX_TOKENS]}")
                else:
                    # Fallback to LM Studio optimized default
                    default_config[ModelParameter.MAX_TOKENS] = 8192
            else:
                # No model capabilities found - use LM Studio optimized defaults
                default_config[ModelParameter.MAX_TOKENS] = 8192
                default_config[ModelParameter.MAX_INPUT_TOKENS] = 32768
                logger.debug(f"Using LM Studio optimized defaults for unknown model: {model_name}")

        except Exception as e:
            # Error in architecture detection - use safe defaults
            logger.warning(f"Architecture detection failed for {model_name}: {e}")
            default_config[ModelParameter.MAX_TOKENS] = 8192
            default_config[ModelParameter.MAX_INPUT_TOKENS] = 32768

        # Set remaining generation parameter defaults
        default_config.update({
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.TOP_P: 0.95,  # Good nucleus sampling default
        })

        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)

        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        base_url = self.config_manager.get_param(ModelParameter.BASE_URL)
        logger.info(f"Initialized LM Studio provider with model: {model}, base URL: {base_url}")

        # Test server connectivity on initialization
        self._verify_server_connection()

    def _verify_server_connection(self) -> None:
        """
        Verify that LM Studio server is accessible.
        Provides helpful error messages if connection fails.
        """
        base_url = self.config_manager.get_param(ModelParameter.BASE_URL)

        try:
            # Test connection with a simple models endpoint call
            models_url = f"{base_url}/models"
            response = requests.get(models_url, timeout=5)

            if response.status_code == 200:
                logger.info("✅ Successfully connected to LM Studio server")
                # Log available models for debugging
                try:
                    models_data = response.json()
                    model_count = len(models_data.get('data', []))
                    logger.info(f"Found {model_count} models available on LM Studio server")
                except:
                    pass  # Don't fail if we can't parse models
            else:
                logger.warning(f"LM Studio server responded with status {response.status_code}")

        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ Cannot connect to LM Studio server")
            logger.warning("Make sure LM Studio is running and server is started on port 1234")
            logger.warning("In LM Studio: Go to 'Local Server' tab and click 'Start Server'")
        except requests.exceptions.Timeout:
            logger.warning("⚠️ LM Studio server connection timed out")
        except Exception as e:
            logger.warning(f"⚠️ LM Studio server check failed: {e}")

    def _is_vision_model(self, model_name: str) -> bool:
        """
        Check if the specified model supports vision capabilities.

        Args:
            model_name: Model name to check

        Returns:
            True if model supports vision, False otherwise
        """
        model_name_lower = model_name.lower()

        # Check for vision indicators in model name
        return any(pattern in model_name_lower for pattern in VISION_MODEL_PATTERNS)

    def _format_tools_for_provider(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for LM Studio's OpenAI-compatible API.

        LM Studio uses the same format as OpenAI:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }
        """
        formatted_tools = []
        for tool in tools:
            formatted_tools.append({
                "type": "function",
                "function": tool
            })
        return formatted_tools

    def _check_for_tool_calls(self, response: Any) -> bool:
        """
        Check if an LM Studio response contains tool calls.

        Args:
            response: The raw response from LM Studio

        Returns:
            True if the response contains tool calls, False otherwise
        """
        # Handle different response formats
        if isinstance(response, dict):
            # Check for choices array (OpenAI format)
            if "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                return bool(message.get("tool_calls"))

        # Handle direct message objects
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            return bool(getattr(message, "tool_calls", None))

        return False

    def _extract_tool_calls(self, response: Any) -> Optional["ToolCallResponse"]:
        """
        Extract tool calls from an LM Studio response.

        Args:
            response: Raw LM Studio response (OpenAI-compatible format)

        Returns:
            ToolCallResponse object if tool calls are present, None otherwise
        """
        if not TOOLS_AVAILABLE or not self._check_for_tool_calls(response):
            return None

        # Extract content and tool calls from response
        content = ""
        tool_calls = []

        # Handle dict response format
        if isinstance(response, dict) and "choices" in response:
            message = response["choices"][0].get("message", {})
            content = message.get("content", "")
            raw_tool_calls = message.get("tool_calls", [])

        # Handle object response format (requests response)
        elif hasattr(response, "choices"):
            message = response.choices[0].message
            content = getattr(message, "content", "")
            raw_tool_calls = getattr(message, "tool_calls", [])
        else:
            return None

        # Process tool calls
        for tc in raw_tool_calls:
            # Handle both dict and object formats
            if isinstance(tc, dict):
                tool_id = tc.get("id", f"call_{len(tool_calls)}")
                function = tc.get("function", {})
                name = function.get("name", "")
                args = function.get("arguments", {})
            else:
                tool_id = getattr(tc, "id", f"call_{len(tool_calls)}")
                function = getattr(tc, "function", None)
                name = getattr(function, "name", "") if function else ""
                args = getattr(function, "arguments", {}) if function else {}

            # Parse arguments if they're a string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call arguments: {args}")
                    args = {"_raw": args}

            # Create tool call object
            tool_call_obj = ToolCall(
                id=tool_id,
                name=name,
                arguments=args
            )
            tool_calls.append(tool_call_obj)

        # Return tool call response
        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls
        )

    def _prepare_openai_request(self,
                              model: str,
                              messages: List[Dict[str, Any]],
                              temperature: float,
                              max_tokens: int,
                              stream: bool,
                              tools: Optional[List[Dict[str, Any]]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Prepare request data for LM Studio's OpenAI-compatible API with full parameter support.

        Args:
            model: The model to use
            messages: OpenAI-format messages
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: Optional formatted tools
            **kwargs: Additional parameters

        Returns:
            Dictionary of request data for the API
        """
        request_data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        # Core generation parameters (filter None values)
        if temperature is not None:
            request_data["temperature"] = temperature

        # Handle max_tokens - LM Studio supports -1 for unlimited
        if max_tokens is not None:
            # LM Studio can handle large token counts or -1 for unlimited
            request_data["max_tokens"] = max_tokens

        # SOTA parameters from AbstractLLM config
        seed = self.config_manager.get_param(ModelParameter.SEED)
        if seed is not None:
            request_data["seed"] = seed
            logger.debug(f"LM Studio using seed: {seed}")

        top_p = self.config_manager.get_param(ModelParameter.TOP_P)
        if top_p is not None:
            request_data["top_p"] = top_p

        # OpenAI-compatible penalty parameters
        frequency_penalty = self.config_manager.get_param(ModelParameter.FREQUENCY_PENALTY)
        if frequency_penalty is not None:
            request_data["frequency_penalty"] = frequency_penalty

        presence_penalty = self.config_manager.get_param(ModelParameter.PRESENCE_PENALTY)
        if presence_penalty is not None:
            request_data["presence_penalty"] = presence_penalty

        # Stop sequences
        stop = self.config_manager.get_param(ModelParameter.STOP)
        if stop is not None:
            request_data["stop"] = stop

        # Additional local model parameters that LM Studio may support
        top_k = self.config_manager.get_param(ModelParameter.TOP_K)
        if top_k is not None:
            # Note: OpenAI API doesn't have top_k, but some local models do
            # LM Studio may support this as an extension
            request_data["top_k"] = top_k

        repetition_penalty = self.config_manager.get_param(ModelParameter.REPETITION_PENALTY)
        if repetition_penalty is not None:
            # Some local APIs support repetition_penalty
            request_data["repetition_penalty"] = repetition_penalty

        # LM Studio-specific parameters
        max_input_tokens = self.config_manager.get_param(ModelParameter.MAX_INPUT_TOKENS)
        if max_input_tokens is not None:
            # LM Studio uses context_length or handles this automatically
            # We'll let LM Studio manage context based on the model
            logger.debug(f"Max input tokens configured: {max_input_tokens}")

        truncation_strategy = self.config_manager.get_param(ModelParameter.TRUNCATION_STRATEGY)
        if truncation_strategy is not None:
            # This could map to LM Studio's contextOverflowPolicy
            # For now, we'll log it but not send as OpenAI API doesn't support it
            logger.debug(f"Truncation strategy: {truncation_strategy}")

        # Add tools if provided
        if tools:
            request_data["tools"] = tools

        # Log the final request for debugging
        logger.debug(f"LM Studio request parameters: {list(request_data.keys())}")

        return request_data

    def _generate_impl(self,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      files: Optional[List[Union[str, Path]]] = None,
                      stream: bool = False,
                      tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                      messages: Optional[List[Dict[str, Any]]] = None,
                      **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """
        Generate a response using LM Studio's OpenAI-compatible API.

        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (paths or URLs)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            messages: Optional conversation history
            **kwargs: Additional parameters to override configuration

        Returns:
            GenerateResponse or generator for streaming responses
        """
        # Update config with any provided kwargs (filter out None values)
        if kwargs:
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
        api_key = self.config_manager.get_param(ModelParameter.API_KEY)

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
                        provider="lmstudio",
                        original_exception=e
                    )

        # Check for images and model compatibility
        has_images = any(isinstance(f, ImageInput) for f in processed_files)
        if has_images and not self._is_vision_model(model):
            raise UnsupportedFeatureError(
                "vision",
                f"Model {model} does not support vision input. Try a vision-capable model like llava.",
                provider="lmstudio"
            )

        # Handle tools using base class methods
        enhanced_system_prompt = system_prompt
        formatted_tools = None
        tool_mode = "none"

        if tools:
            # Use base class method to prepare tool context
            enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, system_prompt)

            # For LM Studio, format tools for OpenAI-compatible API if in native mode
            if tool_mode == "native" and tool_defs:
                formatted_tools = self._format_tools_for_provider(tool_defs)

        # Prepare messages in OpenAI format with proper deduplication
        openai_messages = []

        # Handle conversation history first
        if messages:
            system_found = False
            current_prompt_found = False

            for msg in messages:
                if isinstance(msg, dict):
                    msg_role = msg.get("role")
                    msg_content = msg.get("content", "")
                else:
                    # Handle Message objects
                    msg_role = getattr(msg, 'role', 'user')
                    msg_content = getattr(msg, 'content', str(msg))

                # Check for duplicates
                if msg_role == "system":
                    if not system_found:
                        # Use enhanced system prompt if we have tools, otherwise use original
                        if enhanced_system_prompt and tools:
                            openai_messages.append({"role": "system", "content": enhanced_system_prompt})
                        else:
                            openai_messages.append({"role": msg_role, "content": msg_content})
                        system_found = True
                    # Skip duplicate system messages
                elif msg_role == "user" and msg_content.strip() == prompt.strip():
                    # Mark that current prompt is already in conversation history
                    current_prompt_found = True
                    openai_messages.append({"role": msg_role, "content": msg_content})
                else:
                    # Add non-duplicate messages
                    openai_messages.append({"role": msg_role, "content": msg_content})

            # Add system prompt if not found in conversation history
            if not system_found and enhanced_system_prompt:
                openai_messages.insert(0, {"role": "system", "content": enhanced_system_prompt})

            # Add current prompt if not already in conversation history
            if not current_prompt_found:
                if processed_files:
                    # Multimodal content for vision models
                    content = [{"type": "text", "text": prompt}]
                    for media_input in processed_files:
                        if isinstance(media_input, ImageInput):
                            # Convert to OpenAI format
                            content.append(media_input.to_provider_format("openai"))
                    openai_messages.append({"role": "user", "content": content})
                else:
                    openai_messages.append({"role": "user", "content": prompt})

        else:
            # No conversation history - simple case
            if enhanced_system_prompt:
                openai_messages.append({"role": "system", "content": enhanced_system_prompt})

            # Add current user prompt
            if processed_files:
                # Multimodal content for vision models
                content = [{"type": "text", "text": prompt}]
                for media_input in processed_files:
                    if isinstance(media_input, ImageInput):
                        # Convert to OpenAI format
                        content.append(media_input.to_provider_format("openai"))
                openai_messages.append({"role": "user", "content": content})
            else:
                openai_messages.append({"role": "user", "content": prompt})

        # Prepare request data - filter out conflicting kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['model', 'messages', 'temperature', 'max_tokens', 'stream', 'tools']}

        request_data = self._prepare_openai_request(
            model=model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=formatted_tools,
            **filtered_kwargs
        )

        # Add streaming enhancements for better performance
        if stream:
            if "stream_options" not in request_data:
                request_data["stream_options"] = {}
            request_data["stream_options"]["include_usage"] = True  # Get token usage during streaming

        # API endpoint
        endpoint = f"{base_url}/chat/completions"

        # Log API request URL
        log_request_url("lmstudio", endpoint)

        # Log the complete request details
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            formatted_messages=openai_messages,
            request_data=request_data,
            endpoint=endpoint,
            enhanced_system_prompt=enhanced_system_prompt if tools else None,
            tool_mode=tool_mode,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config_manager.get_param(ModelParameter.TOP_P),
            has_files=bool(files)
        )

        # Headers for authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Make API call
        try:
            if stream:
                return self._generate_streaming(endpoint, request_data, headers, start_time, model, tool_mode)
            else:
                return self._generate_non_streaming(endpoint, request_data, headers, start_time, model, tool_mode)

        except requests.exceptions.ConnectionError as e:
            error_msg = "Cannot connect to LM Studio server. Make sure LM Studio is running and server is started."
            logger.error(f"LM Studio connection error: {e}")
            raise ProviderAPIError(error_msg, provider="lmstudio", original_exception=e)
        except requests.exceptions.Timeout as e:
            error_msg = "LM Studio server request timed out."
            logger.error(f"LM Studio timeout error: {e}")
            raise ProviderAPIError(error_msg, provider="lmstudio", original_exception=e)
        except Exception as e:
            logger.error(f"LM Studio API error: {e}")
            raise ProviderAPIError(f"LM Studio API error: {str(e)}", provider="lmstudio", original_exception=e)

    def _generate_streaming(self, endpoint: str, request_data: Dict, headers: Dict,
                          start_time: float, model: str, tool_mode: str) -> Generator[GenerateResponse, None, None]:
        """Handle streaming response generation."""

        def response_generator():
            current_content = ""
            tool_calls_data = []
            final_usage = None  # Collect usage data from streaming

            # Capture verbatim context sent to LLM
            verbatim_context = json.dumps(request_data, indent=2, ensure_ascii=False)
            self._capture_verbatim_context(f"ENDPOINT: {endpoint}\n\nREQUEST PAYLOAD:\n{verbatim_context}")

            try:
                response = requests.post(endpoint, json=request_data, headers=headers, stream=True)
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix

                    if line.strip() == '[DONE]':
                        break

                    try:
                        chunk_data = json.loads(line)
                        if "choices" in chunk_data and chunk_data["choices"]:
                            delta = chunk_data["choices"][0].get("delta", {})

                            # Handle content - yield only the delta, not cumulative
                            if "content" in delta and delta["content"]:
                                current_content += delta["content"]
                                yield GenerateResponse(
                                    content=delta["content"],  # FIXED: Yield only the delta content
                                    model=model,
                                    usage=None  # Usage provided at end
                                )

                            # Handle tool calls (collect for final response)
                            if "tool_calls" in delta:
                                tool_calls_data.extend(delta["tool_calls"])

                        # Collect usage data if available (from stream_options.include_usage)
                        if "usage" in chunk_data:
                            final_usage = chunk_data["usage"]

                    except json.JSONDecodeError:
                        continue  # Skip malformed chunks

                # Process final tool calls if any
                if tool_calls_data and tool_mode == "native":
                    # Mock a complete response for tool extraction
                    mock_response = {
                        "choices": [{
                            "message": {
                                "content": current_content,
                                "tool_calls": tool_calls_data
                            }
                        }]
                    }
                    tool_response = self._extract_tool_calls(mock_response)
                    if tool_response:
                        yield GenerateResponse(
                            content="",  # FIXED: Don't repeat content in streaming mode
                            tool_calls=tool_response,
                            model=model,
                            usage=final_usage  # Include streaming usage data
                        )

                # Check for prompted tool calls if no native ones found
                if not tool_calls_data and current_content:
                    handler = self._get_tool_handler()
                    if handler:
                        prompted_response = handler.parse_response(current_content, mode="prompted")
                        if prompted_response and prompted_response.has_tool_calls():
                            yield GenerateResponse(
                                content="",  # FIXED: Don't repeat content in streaming mode
                                tool_calls=prompted_response,
                                model=model,
                                usage=final_usage  # Include streaming usage data
                            )

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                raise

        return response_generator()

    def _generate_non_streaming(self, endpoint: str, request_data: Dict, headers: Dict,
                              start_time: float, model: str, tool_mode: str) -> GenerateResponse:
        """Handle non-streaming response generation."""

        # Capture verbatim context sent to LLM
        verbatim_context = json.dumps(request_data, indent=2, ensure_ascii=False)
        self._capture_verbatim_context(f"ENDPOINT: {endpoint}\n\nREQUEST PAYLOAD:\n{verbatim_context}")

        response = requests.post(endpoint, json=request_data, headers=headers)
        response.raise_for_status()

        data = response.json()

        # Extract content from response
        if "choices" not in data or not data["choices"]:
            raise ProviderAPIError("Invalid response format from LM Studio", provider="lmstudio")

        message = data["choices"][0]["message"]
        content = message.get("content", "")

        # Calculate timing and usage
        total_time = time.time() - start_time
        usage_data = data.get("usage", {})

        # If no usage data, estimate
        if not usage_data:
            prompt_tokens = _approximate_token_count(str(request_data.get("messages", "")))
            completion_tokens = _approximate_token_count(content)
            usage_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }

        usage_data["total_time"] = total_time

        # Handle tool calls
        tool_response = None

        # First try native tool extraction
        if tool_mode == "native" and self._check_for_tool_calls(data):
            tool_response = self._extract_tool_calls(data)

        # If no native tool calls, try prompted parsing
        if not tool_response or not tool_response.has_tool_calls():
            handler = self._get_tool_handler()
            if handler and content:
                prompted_response = handler.parse_response(content, mode="prompted")
                if prompted_response and prompted_response.has_tool_calls():
                    tool_response = prompted_response

        # Log response
        self._log_response_details(
            data,
            content,
            has_tool_calls=bool(tool_response and tool_response.has_tool_calls()),
            tool_calls=tool_response.tool_calls if tool_response else None,
            model=model,
            usage=usage_data
        )

        # Return response
        return GenerateResponse(
            content=content,
            tool_calls=tool_response,
            model=model,
            usage=usage_data,
            raw_response=data
        )

    async def generate_async(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           files: Optional[List[Union[str, Path]]] = None,
                           stream: bool = False,
                           tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                           messages: Optional[List[Dict[str, Any]]] = None,
                           **kwargs) -> Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]:
        """
        Asynchronously generate a response using LM Studio's API.

        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (paths or URLs)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            messages: Optional conversation history
            **kwargs: Additional parameters to override configuration

        Returns:
            GenerateResponse or async generator for streaming responses
        """
        # Check if aiohttp is available for async operations
        if not AIOHTTP_AVAILABLE:
            raise ImportError("The 'aiohttp' package is required for async operations. Install with: pip install abstractllm[lmstudio]")

        # For now, use asyncio.run_in_executor to wrap the sync implementation
        # This ensures compatibility while providing async interface
        loop = asyncio.get_event_loop()

        if stream:
            # For streaming, we need to wrap the generator in an async generator
            async def async_stream():
                sync_generator = await loop.run_in_executor(
                    None,
                    lambda: self._generate_impl(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        files=files,
                        stream=True,
                        tools=tools,
                        messages=messages,
                        **kwargs
                    )
                )
                for response in sync_generator:
                    yield response
            return async_stream()
        else:
            # For non-streaming, run in executor
            return await loop.run_in_executor(
                None,
                lambda: self._generate_impl(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    files=files,
                    stream=False,
                    tools=tools,
                    messages=messages,
                    **kwargs
                )
            )

    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the LM Studio provider.

        Returns:
            Dictionary of capabilities
        """
        model = self.config_manager.get_param(ModelParameter.MODEL)

        capabilities = {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: None,  # Varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: True,  # Via OpenAI-compatible API
            ModelCapability.TOOL_USE: True,
            ModelCapability.VISION: self._is_vision_model(model),
            ModelCapability.JSON_MODE: True  # LM Studio supports structured output
        }

        return capabilities


