"""
HuggingFace provider for AbstractLLM.

This provider uses HuggingFace Transformers for local inference,
following the simple and effective patterns from the MLX provider.
"""

import time
import logging
import platform
from pathlib import Path
import os
from typing import Dict, List, Any, Optional, Generator, Union, Callable, ClassVar, Tuple
import copy

# Import the interface class
from abstractllm.interface import (
    ModelParameter, 
    ModelCapability,
    GenerateResponse
)
from abstractllm.providers.base import BaseProvider
from abstractllm.exceptions import (
    ModelLoadingError,
    GenerationError,
    UnsupportedFeatureError,
    ImageProcessingError,
    FileProcessingError,
    MemoryExceededError
)
from abstractllm.utils.utilities import TokenCounter
from abstractllm.utils.logging import log_request, log_response
from abstractllm.architectures.detection import detect_architecture

# Set up logging
logger = logging.getLogger("abstractllm.providers.huggingface")

# Check for required dependencies
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import transformers
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        pipeline, TextStreamer
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    TORCH_AVAILABLE = True
    logger.debug("HuggingFace Transformers and PyTorch available")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    TORCH_AVAILABLE = False
    logger.warning(f"HuggingFace Transformers or PyTorch not available: {e}")


def torch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE


class HuggingFaceProvider(BaseProvider):
    """
    HuggingFace implementation following MLX provider patterns for simplicity and speed.
    
    Key principles:
    - Keep it simple and let the library handle complexity
    - Use high-level functions (pipeline) when possible
    - Minimal parameter configuration to avoid conflicts
    - Fast loading and generation
    - Follow MLX patterns for tool support
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the HuggingFace provider."""
        super().__init__(config)
        
        # Check for dependencies
        if not TRANSFORMERS_AVAILABLE:
            logger.error("HuggingFace Transformers package not found")
            raise ImportError("HuggingFace Transformers is required. Install with: pip install transformers")
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch package not found")
            raise ImportError("PyTorch is required. Install with: pip install torch")
        
        # Set default configuration - follow MLX patterns
        default_config = {
            ModelParameter.MODEL: "microsoft/DialoGPT-small",  # Fast, small model for testing
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 100,  # Conservative default like MLX
            ModelParameter.TOP_P: 0.9,
            "device": "auto",  # Let transformers decide
            "trust_remote_code": False,
            "use_fast": True
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize components - keep it simple
        self._pipeline = None
        self._tokenizer = None
        self._tool_handler = None
        self._is_loaded = False
        
        # Log initialization
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized HuggingFace provider with model: {model_name}")
        
        # Automatically load the model during initialization
        self.load_model()

    def load_model(self) -> None:
        """Load the HuggingFace model using pipeline for maximum simplicity and speed."""
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check if model is already loaded
        if self._is_loaded and self._pipeline is not None:
            logger.debug(f"Model {model_name} already loaded")
            return
        
        try:
            logger.info(f"Loading HuggingFace model: {model_name}")
            
            # Get configuration
            device = self.config_manager.get_param("device", "auto")
            trust_remote_code = self.config_manager.get_param("trust_remote_code", False)
            
            # Use pipeline for maximum simplicity and reliability
            # This is the equivalent of mlx_lm.load() - high-level and well-tested
            self._pipeline = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None,
                trust_remote_code=trust_remote_code,
                return_full_text=False  # Only return generated text
            )
            
            # Also load tokenizer for token counting
            self._tokenizer = self._pipeline.tokenizer
            
            # Set flags to indicate model is loaded
            self._is_loaded = True
            logger.info(f"Successfully loaded HuggingFace model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadingError(f"Failed to load model {model_name}: {str(e)}")

    def _get_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get generation parameters - keep it simple like MLX.
        """
        # Filter out None values
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Simple parameters like MLX
        params = {
            "max_new_tokens": filtered_kwargs.get("max_tokens", self.config_manager.get_param(ModelParameter.MAX_TOKENS)),
            "temperature": filtered_kwargs.get("temperature", self.config_manager.get_param(ModelParameter.TEMPERATURE)),
            "top_p": filtered_kwargs.get("top_p", self.config_manager.get_param(ModelParameter.TOP_P)),
            "do_sample": True,
            "pad_token_id": self._tokenizer.eos_token_id,  # Simple padding
        }
        
        return params

    def _ensure_chat_template_compatibility(self, messages: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """
        Ensure chat template compatibility by fixing role alternation issues.
        Similar to MLX provider's implementation but simplified.
        """
        if not messages:
            return messages
            
        # Use architecture detection to determine if strict alternation is needed
        architecture = detect_architecture(model_name)
        
        # Check if this architecture requires strict role alternation
        strict_alternation_models = ["gemma", "llama"]
        requires_strict_alternation = architecture in strict_alternation_models
        
        if not requires_strict_alternation:
            return messages
            
        logger.debug(f"Applying chat template compatibility fixes for {architecture} architecture")
        
        # Step 1: Collect all system messages and merge them
        fixed_messages = []
        system_contents = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_contents.append(msg["content"])
            else:
                # If we have collected system messages, add them before the first non-system message
                if system_contents and not fixed_messages:
                    merged_system_content = "\n\n".join(system_contents)
                    
                    # For Gemma models, integrate system prompt into first user message
                    if architecture == "gemma" and msg["role"] == "user":
                        enhanced_content = f"System: {merged_system_content}\n\nUser: {msg['content']}"
                        fixed_messages.append({"role": "user", "content": enhanced_content})
                    else:
                        # For other models, keep as separate system message but ensure it's first
                        fixed_messages.append({"role": "system", "content": merged_system_content})
                        fixed_messages.append(msg)
                    
                    system_contents = []  # Clear collected system messages
                else:
                    fixed_messages.append(msg)
        
        # Step 2: Handle tool messages by converting them to assistant messages
        final_messages = []
        for msg in fixed_messages:
            if msg["role"] == "tool":
                tool_name = msg.get("name", "unknown_tool")
                tool_content = msg.get("content", "")
                # Convert tool response to assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": f"I executed the {tool_name} tool and received: {tool_content}"
                }
                final_messages.append(assistant_msg)
            else:
                final_messages.append(msg)
        
        # Step 3: Ensure proper user/assistant alternation
        if len(final_messages) > 1:
            validated_messages = []
            last_role = None
            
            for msg in final_messages:
                current_role = msg["role"]
                
                # Skip consecutive messages with the same role (except system)
                if current_role == last_role and current_role != "system":
                    if current_role == "user":
                        # Merge consecutive user messages
                        if validated_messages:
                            validated_messages[-1]["content"] += f"\n\n{msg['content']}"
                        else:
                            validated_messages.append(msg)
                    elif current_role == "assistant":
                        # Merge consecutive assistant messages
                        if validated_messages:
                            validated_messages[-1]["content"] += f"\n\n{msg['content']}"
                        else:
                            validated_messages.append(msg)
                else:
                    validated_messages.append(msg)
                    last_role = current_role
            
            final_messages = validated_messages
        
        logger.debug(f"Fixed messages: {len(messages)} -> {len(final_messages)} messages")
        return final_messages

    def _get_tool_handler(self) -> Optional["UniversalToolHandler"]:
        """Get or create the tool handler for this provider - override to enable prompted tools for all HF models."""
        # Import TOOLS_AVAILABLE from base
        from abstractllm.providers.base import TOOLS_AVAILABLE
        
        if not TOOLS_AVAILABLE:
            return None
            
        if self._tool_handler is None:
            # Import here to avoid circular imports
            from abstractllm.tools.handler import UniversalToolHandler
            
            # Get the model from config
            model = self.config_manager.get_param(ModelParameter.MODEL)
            if model:
                # Create handler and force prompted mode for HuggingFace models
                self._tool_handler = UniversalToolHandler(model)
                
                # Override the capabilities to enable prompted tool support
                # This allows any HuggingFace model to use tools via system prompt enhancement
                self._tool_handler.capabilities = {
                    "tool_support": "prompted",  # Force prompted mode
                    "structured_output": "prompted",
                    "parallel_tools": True,
                    "max_tools": -1,  # Unlimited
                    "tool_template": "special_token"  # Use <|tool_call|> format
                }
                
                # Update support flags
                self._tool_handler.tool_support = "prompted"
                self._tool_handler.supports_native = False
                self._tool_handler.supports_prompted = True
                
                logger.info(f"Enabled prompted tool support for HuggingFace model: {model}")
                
        return self._tool_handler

    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                messages: Optional[List[Dict[str, Any]]] = None,
                **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate a response using the HuggingFace model with tool support following MLX patterns."""
        
        logger.info(f"Generation request: model={self.config_manager.get_param(ModelParameter.MODEL)}, "
                   f"stream={stream}, prompt_len={len(prompt)}, tools={len(tools) if tools else 0}")
        
        try:
            # For now, ignore files - focus on text generation and tools
            if files:
                raise UnsupportedFeatureError("vision", "Vision models not yet supported", provider="huggingface")
            
            # Use standard text generation with tool support
            if stream:
                return self._generate_text_stream(prompt, system_prompt=system_prompt, tools=tools, messages=messages, **kwargs)
            else:
                return self._generate_text(prompt, system_prompt=system_prompt, tools=tools, messages=messages, **kwargs)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Generation failed: {str(e)}")

    def _generate_text(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Any]] = None, messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> GenerateResponse:
        """Generate text using HuggingFace pipeline with tool support following MLX patterns."""
        
        # Make sure model is loaded
        if not self._is_loaded:
            logger.info("Model not loaded, loading now")
            self.load_model()
        
        # Get simple parameters
        params = self._get_generation_params(**kwargs)
        
        try:
            start_time = time.time()
            
            # Prepare messages for chat template - follow MLX pattern
            if messages is not None:
                chat_messages = messages.copy()
            else:
                chat_messages = []
                if system_prompt:
                    chat_messages.append({"role": "system", "content": system_prompt})
                chat_messages.append({"role": "user", "content": prompt})
            
            # Get model name for architecture detection
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            
            # Use base class method to prepare tool context - EXACTLY like MLX
            enhanced_system_prompt = system_prompt
            hf_tools = None
            if tools:
                # Use base class method for tool preparation
                enhanced_system_prompt, tool_defs, mode = self._prepare_tool_context(tools, system_prompt)
                logger.info(f"Prepared {len(tools)} tools in {mode} mode")
                
                # HuggingFace uses prompted mode like MLX, so hf_tools is mainly for tracking
                hf_tools = tool_defs if mode == "native" else tools
                
                # Update messages with enhanced system prompt - follow MLX pattern
                if messages is not None:
                    # When using conversation history, preserve it but enhance system prompt
                    formatted_messages = []
                    system_added = False
                    for msg in chat_messages:
                        if msg["role"] == "system":
                            if not system_added:
                                # Replace first system prompt with enhanced version
                                formatted_messages.append({"role": "system", "content": enhanced_system_prompt})
                                system_added = True
                            # Skip additional system messages (they'll be merged by compatibility fix)
                        elif msg["role"] == "tool":
                            # Convert tool messages to assistant messages with clear formatting
                            tool_name = msg.get("name", "unknown_tool")
                            tool_output = msg.get("content", "")
                            formatted_messages.append({
                                "role": "assistant", 
                                "content": f"I called the {tool_name} tool and received: {tool_output}"
                            })
                        else:
                            formatted_messages.append(msg)
                    
                    # If no system message was found, add enhanced system prompt at the beginning
                    if not system_added:
                        formatted_messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
                else:
                    # For new conversations, just use enhanced system prompt
                    formatted_messages = []
                    formatted_messages.append({"role": "system", "content": enhanced_system_prompt})
                    formatted_messages.append({"role": "user", "content": prompt})
            else:
                # Standard chat template without tools - use chat_messages directly
                formatted_messages = chat_messages
            
            # Apply chat template compatibility fixes for Gemma/Llama models
            compatible_messages = self._ensure_chat_template_compatibility(formatted_messages, model_name)
            
            # Try to apply chat template with simple fallback - follow MLX pattern
            try:
                formatted_prompt = self._tokenizer.apply_chat_template(
                    compatible_messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except Exception as template_error:
                logger.warning(f"Chat template failed: {template_error}")
                
                # Use simple fallback template
                logger.info(f"Using simple fallback template")
                formatted_prompt = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in compatible_messages
                ])
                if compatible_messages and compatible_messages[-1]['role'] != 'assistant':
                    formatted_prompt += "\nassistant:"
            
            # The tool instructions are already included in the enhanced_system_prompt
            # No need to add them again to the formatted prompt
            
            # Log the request using shared method - EXACTLY like MLX
            self._log_request_details(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                formatted_messages=compatible_messages,
                enhanced_system_prompt=enhanced_system_prompt if tools else system_prompt,
                stream=False,
                formatted_prompt=formatted_prompt,
                **params
            )
            
            logger.info(f"HuggingFace generation starting - prompt: {len(formatted_prompt)} chars, max_tokens: {params['max_new_tokens']}")
            
            # Generate using pipeline - this is the equivalent of mlx_lm.generate()
            # Simple, fast, and reliable
            result = self._pipeline(
                formatted_prompt,
                **params
            )
            
            # Extract the generated text
            output = result[0]['generated_text']
            
            # Log response using shared method
            self._log_response_details(output, output)
            
            logger.info(f"HuggingFace generation completed - response length: {len(output)} chars, time: {time.time() - start_time:.2f}s")
            
            # Use base class tool extraction method - EXACTLY like MLX
            if hf_tools:
                # Log the output to help with debugging
                logger.debug(f"Checking for tool calls in output: {output[:200]}...")
                
                # Try to extract tool calls from the response
                tool_response = self._extract_tool_calls(output)
                
                # Log whether tool calls were found
                if tool_response and tool_response.has_tool_calls():
                    logger.info(f"Found {len(tool_response.tool_calls)} tool calls in response")
                    for tc in tool_response.tool_calls:
                        logger.info(f"Tool call: {tc.name} with args: {tc.arguments}")
                    
                    # Return a GenerateResponse with tool calls
                    return GenerateResponse(
                        content=output,
                        tool_calls=tool_response,
                        model=model_name,
                        usage={
                            "prompt_tokens": len(self._tokenizer.encode(formatted_prompt)),
                            "completion_tokens": len(self._tokenizer.encode(output)),
                            "total_tokens": len(self._tokenizer.encode(formatted_prompt)) + len(self._tokenizer.encode(output))
                        }
                    )
                else:
                    logger.warning(f"No tool calls found in response despite tools being provided")
            
            # Calculate token usage
            prompt_tokens = len(self._tokenizer.encode(formatted_prompt))
            completion_tokens = len(self._tokenizer.encode(output))
            
            return GenerateResponse(
                content=output,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "time": time.time() - start_time
                }
            )
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise GenerationError(f"Text generation failed: {str(e)}")

    def _generate_text_stream(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Any]] = None, messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[GenerateResponse, None, None]:
        """Stream text generation - not implemented yet, return full response."""
        
        # For now, just return the full response as a single chunk
        # TODO: Implement proper streaming later
        response = self._generate_text(prompt, system_prompt=system_prompt, tools=tools, messages=messages, **kwargs)
        yield response

    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM provider."""
        return {
            ModelCapability.STREAMING: False,  # Not implemented yet
            ModelCapability.MAX_TOKENS: self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048),
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: False,  # Not implemented yet
            ModelCapability.FUNCTION_CALLING: True,  # Now implemented following MLX patterns
            ModelCapability.TOOL_USE: True,  # Now implemented following MLX patterns
            ModelCapability.VISION: False,  # Not implemented yet
        }

    async def generate_async(self, *args, **kwargs):
        """Async generation not implemented yet."""
        raise UnsupportedFeatureError("async_generation", "Async generation not implemented yet", provider="huggingface")


# Legacy compatibility class
class HuggingFaceLLM:
    """Legacy compatibility wrapper."""
    
    def __init__(self, model="microsoft/DialoGPT-medium", api_key=None):
        self.provider = HuggingFaceProvider({ModelParameter.MODEL: model})
    
    def generate(self, prompt, image=None, images=None, **kwargs):
        return self.provider.generate(prompt, **kwargs) 