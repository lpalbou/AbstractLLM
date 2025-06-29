"""
HuggingFace provider for AbstractLLM.

This provider uses HuggingFace transformers for local inference.
"""

import time
import logging
import platform
from pathlib import Path
import os
from typing import Dict, List, Any, Optional, Generator, Union, Callable, Tuple
import warnings

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
    ProviderAPIError
)
from abstractllm.utils.logging import log_request, log_response
from abstractllm.utils.utilities import TokenCounter

# Set up logging
logger = logging.getLogger("abstractllm.providers.huggingface")

# Check for required dependencies
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
LLAMA_CPP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. HuggingFace provider will be disabled.")

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TextStreamer,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. HuggingFace provider will be disabled.")

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
    logger.info("llama-cpp-python available - GGUF support enabled")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.info("llama-cpp-python not available - GGUF support disabled")


def torch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE


class HuggingFaceProvider(BaseProvider):
    """
    HuggingFace implementation for AbstractLLM.
    
    This provider leverages HuggingFace transformers for local inference
    and supports both regular models and GGUF quantized models.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the HuggingFace provider."""
        super().__init__(config)
        
        # Check for required dependencies
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HuggingFaceProvider. Install with: pip install torch")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers is required for HuggingFaceProvider. Install with: pip install transformers")
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "microsoft/DialoGPT-medium",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 50,
            ModelParameter.TOP_P: 0.9,
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._is_loaded = False
        self._is_gguf_model = False
        self._gguf_model_path = None
        self._llama_model = None
        
        # Log initialization
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized HuggingFace provider with model: {model_name}")
        
        # Check if this is a GGUF model
        self._is_gguf_model = self._check_gguf_model(model_name)
        
        # Automatically load the model during initialization
        self.load_model()

    def _check_gguf_model(self, model_name: str) -> bool:
        """
        Check if the specified model is a GGUF model.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is GGUF, False otherwise
        """
        # Check for GGUF indicators in model name
        gguf_indicators = [
            "gguf", "GGUF", ".gguf", 
            "Q4_K_M", "Q5_0", "Q5_K_M", "Q6_K", "Q8_0",  # Common quantization formats
            "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0"   # Lowercase variants
        ]
        
        is_gguf = any(indicator in model_name for indicator in gguf_indicators)
        
        if is_gguf:
            logger.info(f"Model {model_name} identified as a GGUF model")
        
        return is_gguf

    def _download_gguf_file(self, model_name: str) -> str:
        """
        Download a specific GGUF file from HuggingFace.
        
        Args:
            model_name: Model name in format like "Qwen/Qwen3-4B-GGUF:Q4_K_M"
            
        Returns:
            Path to downloaded GGUF file
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # Parse model name and file specification
            if ":" in model_name:
                repo_id, file_spec = model_name.split(":", 1)
            else:
                repo_id = model_name
                file_spec = "Q4_K_M"  # Default to Q4_K_M
            
            logger.info(f"Downloading GGUF model: {repo_id} with quantization: {file_spec}")
            
            # List available files
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            # Find the specific quantization file
            target_file = None
            for f in gguf_files:
                if file_spec.upper() in f.upper():
                    target_file = f
                    break
            
            if not target_file:
                # If specific quantization not found, try to find any Q4_K_M file
                for f in gguf_files:
                    if "Q4_K_M" in f.upper():
                        target_file = f
                        break
                        
            if not target_file:
                # If still not found, use the first GGUF file
                if gguf_files:
                    target_file = gguf_files[0]
                else:
                    raise ValueError(f"No GGUF files found in {repo_id}")
            
            logger.info(f"Downloading GGUF file: {target_file}")
            
            # Download the file
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=target_file,
                cache_dir=None  # Use default cache
            )
            
            logger.info(f"GGUF file downloaded to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download GGUF file: {e}")
            raise ModelLoadingError(f"Failed to download GGUF file: {str(e)}")

    def load_model(self) -> None:
        """Load the HuggingFace model or GGUF model."""
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check if model is already loaded
        if self._is_loaded:
            logger.debug(f"Model {model_name} already loaded")
            return
        
        try:
            if self._is_gguf_model:
                self._load_gguf_model(model_name)
            else:
                self._load_transformers_model(model_name)
                
            self._is_loaded = True
            logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadingError(f"Failed to load model {model_name}: {str(e)}")

    def _load_gguf_model(self, model_name: str) -> None:
        """Load a GGUF model using llama-cpp-python."""
        if not LLAMA_CPP_AVAILABLE:
            error_msg = (
                f"GGUF model '{model_name}' requires llama-cpp-python. "
                f"Install it with: pip install llama-cpp-python\n\n"
                f"Alternatively, you can:\n"
                f"1. Use the Ollama provider: --provider ollama --model 'hf.co/{model_name}'\n"
                f"2. Use the regular (unquantized) model: --model 'Qwen/Qwen3-4B'"
            )
            logger.error(error_msg)
            raise UnsupportedFeatureError("gguf_models", error_msg, provider="huggingface")
        
        logger.info(f"Loading GGUF model: {model_name}")
        
        # Check if this is a Qwen3 model
        if self._is_qwen3_model(model_name):
            self._handle_qwen3_model(model_name)
            return
        
        # Download the GGUF file
        self._gguf_model_path = self._download_gguf_file(model_name)
        
        # Check if we need to apply architecture mapping for unsupported models
        model_params = {}
        
        # Architecture mapping for models not natively supported by llama-cpp-python
        if "qwen3" in model_name.lower() or "qwen-3" in model_name.lower():
            logger.info("Detected Qwen3 model - applying architecture mapping")
            # Map Qwen3 to a supported architecture (llama is the most compatible)
            model_params["model_architecture"] = "llama"
            logger.info("Mapped Qwen3 architecture to 'llama' for compatibility")
        
        # Load with llama-cpp-python
        try:
            self._llama_model = llama_cpp.Llama(
                model_path=self._gguf_model_path,
                n_ctx=4096,  # Context length
                n_threads=None,  # Use all available threads
                verbose=False,
                **model_params  # Apply any architecture mappings
            )
            logger.info(f"Successfully loaded GGUF model with llama-cpp-python")
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            if "unknown model architecture" in str(e).lower():
                architecture_name = str(e).split("'")[-2] if "'" in str(e) else "unknown"
                raise ModelLoadingError(
                    f"Unsupported model architecture '{architecture_name}'. "
                    f"This version of llama-cpp-python doesn't support this architecture. "
                    f"Try using a different model or update llama-cpp-python."
                )
            raise
        
        # For GGUF models, we don't have a separate tokenizer
        # llama-cpp-python handles tokenization internally
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def _is_qwen3_model(self, model_name: str) -> bool:
        """
        Check if the model is a Qwen3 model.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if the model is a Qwen3 model, False otherwise
        """
        qwen3_indicators = ["qwen3", "qwen-3", "qwen/qwen3"]
        return any(indicator.lower() in model_name.lower() for indicator in qwen3_indicators)
        
    def _handle_qwen3_model(self, model_name: str) -> None:
        """
        Handle Qwen3 GGUF models which are not compatible with the current version of llama-cpp-python.
        
        Args:
            model_name: Model name
            
        Raises:
            ModelLoadingError: With helpful instructions
        """
        llama_cpp_version = getattr(llama_cpp, "__version__", "unknown")
        
        error_msg = (
            f"Qwen3 GGUF models are not supported by llama-cpp-python v{llama_cpp_version}.\n\n"
            f"Alternatives:\n"
            f"1. Use a different model architecture: --model \"TheBloke/Llama-2-7b-Chat-GGUF:Q4_K_M\"\n"
            f"2. Use Ollama provider: --provider ollama --model \"qwen3\"\n"
            f"3. Use the regular HuggingFace model: --model \"Qwen/Qwen3-4B\"\n"
            f"4. Use MLX provider (Apple Silicon only): --provider mlx --model \"mlx-community/Qwen3-4B-4bit\"\n\n"
            f"Technical details: llama-cpp-python v{llama_cpp_version} doesn't support the 'qwen3' architecture."
        )
        
        logger.error(error_msg)
        raise ModelLoadingError(error_msg)

    def _load_transformers_model(self, model_name: str) -> None:
        """Load a regular transformers model."""
        logger.info(f"Loading transformers model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with pipeline for simplicity
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

    def _get_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get generation parameters by merging config defaults with kwargs.
        
        Args:
            **kwargs: Generation parameters provided to the method
            
        Returns:
            Dictionary of generation parameters
        """
        # Filter out None values from kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Get parameters from config with filtered kwargs override
        params = {
            "max_tokens": filtered_kwargs.get("max_tokens", self.config_manager.get_param(ModelParameter.MAX_TOKENS)),
            "temperature": filtered_kwargs.get("temperature", self.config_manager.get_param(ModelParameter.TEMPERATURE)),
            "top_p": filtered_kwargs.get("top_p", self.config_manager.get_param(ModelParameter.TOP_P, 0.95))
        }
        
        return params

    def _ensure_chat_template_compatibility(self, messages: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """
        Ensure chat template compatibility for different models.
        
        Args:
            messages: List of message dictionaries
            model_name: Name of the model
            
        Returns:
            Fixed list of messages
        """
        if not messages:
            return messages
        
        # Simple compatibility fixes
        fixed_messages = []
        
        for msg in messages:
            if msg["role"] == "tool":
                # Convert tool messages to assistant messages
                tool_name = msg.get("name", "unknown_tool")
                tool_content = msg.get("content", "")
                fixed_messages.append({
                    "role": "assistant",
                    "content": f"Tool '{tool_name}' output: {tool_content}"
                })
            else:
                fixed_messages.append(msg)
        
        return fixed_messages

    def _get_tool_handler(self) -> Optional["UniversalToolHandler"]:
        """
        Get or create the tool handler for this provider.
        
        This method overrides the base class to always enable prompted tool support
        for HuggingFace models, since most HF models don't have native tool calling.
        """
        if not hasattr(self, '_tool_handler') or self._tool_handler is None:
            try:
                from abstractllm.tools.handler import UniversalToolHandler
                
                # Get the model name
                model = self.config_manager.get_param(ModelParameter.MODEL)
                if model:
                    self._tool_handler = UniversalToolHandler(model)
                    
                    # Force prompted mode for HuggingFace models
                    # Most HF models don't have native tool support
                    self._tool_handler.supports_native = False
                    self._tool_handler.supports_prompted = True
                    
                    logger.info(f"Created tool handler for HF model {model} in prompted mode")
                else:
                    logger.warning("No model specified for tool handler")
                    return None
                    
            except ImportError:
                logger.warning("Tool handler not available - tools package not installed")
                return None
            except Exception as e:
                logger.error(f"Failed to create tool handler: {e}")
                return None
                
        return self._tool_handler

    def _generate_impl(self,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      files: Optional[List[Union[str, Path]]] = None,
                      stream: bool = False,
                      tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                      messages: Optional[List[Dict[str, Any]]] = None,
                      **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate a response using the HuggingFace model."""
        
        logger.info(f"Generation request: model={self.config_manager.get_param(ModelParameter.MODEL)}, "
                   f"stream={stream}, has_system_prompt={system_prompt is not None}")
        
        if files:
            raise UnsupportedFeatureError("vision", "Vision capabilities not implemented yet", provider="huggingface")
        
        try:
            if stream:
                return self._generate_text_stream(prompt, system_prompt=system_prompt, tools=tools, messages=messages, **kwargs)
            else:
                return self._generate_text(prompt, system_prompt=system_prompt, tools=tools, messages=messages, **kwargs)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Generation failed: {str(e)}")

    def _generate_text(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Any]] = None, messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> GenerateResponse:
        """Generate text using HuggingFace transformers or llama-cpp-python."""
        
        if not self._is_loaded:
            self.load_model()
        
        # Get parameters
        params = self._get_generation_params(**kwargs)
        max_tokens = params["max_tokens"]
        temperature = params["temperature"]
        
        try:
            start_time = time.time()
            
            if self._is_gguf_model:
                # Remove parameters that are already passed as positional arguments
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                return self._generate_with_llama_cpp(prompt, system_prompt, tools, messages, max_tokens, temperature, **filtered_kwargs)
            else:
                # Remove parameters that are already passed as positional arguments
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                return self._generate_with_transformers(prompt, system_prompt, tools, messages, max_tokens, temperature, **filtered_kwargs)
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise GenerationError(f"Text generation failed: {str(e)}")

    def _generate_with_llama_cpp(self, prompt: str, system_prompt: Optional[str], tools: Optional[List[Any]], messages: Optional[List[Dict[str, Any]]], max_tokens: int, temperature: float, **kwargs) -> GenerateResponse:
        """Generate text using llama-cpp-python for GGUF models."""
        
        # Prepare the full prompt
        if messages:
            # Use conversation history
            full_prompt = self._format_messages_for_llama_cpp(messages, system_prompt)
        else:
            # Simple prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
        
        # Handle tools if provided
        enhanced_prompt = full_prompt
        if tools:
            enhanced_system_prompt, tool_defs, mode = self._prepare_tool_context(tools, system_prompt)
            if mode == "prompted":
                # Replace system prompt in the full prompt
                if system_prompt and enhanced_system_prompt != system_prompt:
                    enhanced_prompt = full_prompt.replace(f"System: {system_prompt}", f"System: {enhanced_system_prompt}")
                elif not system_prompt:
                    enhanced_prompt = f"System: {enhanced_system_prompt}\n\n{full_prompt}"
        
        # Log request
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt,
            enhanced_system_prompt=enhanced_prompt if tools else system_prompt,
            tools=tools,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            model_type="GGUF"
        )
        
        logger.info(f"Generating with llama-cpp-python - prompt length: {len(enhanced_prompt)} chars")
        
        # Generate with llama-cpp-python
        start_time = time.time()
        
        # Ensure all parameters have valid values
        top_p_value = kwargs.get("top_p", 0.9)
        if top_p_value is None:
            top_p_value = 0.9
            
        result = self._llama_model(
            enhanced_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p_value,
            echo=False,
            stop=["User:", "Human:", "\n\n"]
        )
        
        # Extract the generated text
        output = result["choices"][0]["text"]
        
        # Log response
        self._log_response_details(output, output)
        
        logger.info(f"GGUF generation completed - response length: {len(output)} chars")
        
        # Check for tool calls if tools were provided
        if tools:
            tool_response = self._extract_tool_calls(output)
            if tool_response and tool_response.has_tool_calls():
                return GenerateResponse(
                    content=output,
                    tool_calls=tool_response,
                    model=self.config_manager.get_param(ModelParameter.MODEL),
                    usage={
                        "prompt_tokens": result["usage"]["prompt_tokens"],
                        "completion_tokens": result["usage"]["completion_tokens"],
                        "total_tokens": result["usage"]["total_tokens"],
                        "time": time.time() - start_time
                    }
                )
        
        return GenerateResponse(
            content=output,
            model=self.config_manager.get_param(ModelParameter.MODEL),
            usage={
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "total_tokens": result["usage"]["total_tokens"],
                "time": time.time() - start_time
            }
        )

    def _generate_with_transformers(self, prompt: str, system_prompt: Optional[str], tools: Optional[List[Any]], messages: Optional[List[Dict[str, Any]]], max_tokens: int, temperature: float, **kwargs) -> GenerateResponse:
        """Generate text using HuggingFace transformers."""
        
        # Prepare messages
        if messages:
            chat_messages = messages.copy()
        else:
            chat_messages = []
            if system_prompt:
                chat_messages.append({"role": "system", "content": system_prompt})
            chat_messages.append({"role": "user", "content": prompt})
        
        # Handle tools if provided
        enhanced_system_prompt = system_prompt
        if tools:
            enhanced_system_prompt, tool_defs, mode = self._prepare_tool_context(tools, system_prompt)
            
            # Update system prompt in messages
            if messages:
                # Replace first system message or add one
                system_added = False
                for i, msg in enumerate(chat_messages):
                    if msg["role"] == "system":
                        chat_messages[i]["content"] = enhanced_system_prompt
                        system_added = True
                        break
                if not system_added:
                    chat_messages.insert(0, {"role": "system", "content": enhanced_system_prompt})
            else:
                # Update the system prompt for new conversations
                if chat_messages and chat_messages[0]["role"] == "system":
                    chat_messages[0]["content"] = enhanced_system_prompt
        
        # Apply compatibility fixes
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        compatible_messages = self._ensure_chat_template_compatibility(chat_messages, model_name)
        
        # Format messages for the model
        try:
            if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    compatible_messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                # Fallback formatting
                formatted_prompt = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in compatible_messages
                ])
                formatted_prompt += "\nassistant:"
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, using fallback")
            formatted_prompt = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in compatible_messages
            ])
            formatted_prompt += "\nassistant:"
        
        # Log request
        self._log_request_details(
            prompt=prompt,
            system_prompt=system_prompt,
            enhanced_system_prompt=enhanced_system_prompt if tools else system_prompt,
            tools=tools,
            messages=messages,
            formatted_messages=compatible_messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            model_type="Transformers"
        )
        
        logger.info(f"Generating with transformers pipeline - prompt length: {len(formatted_prompt)} chars")
        
        # Generate with pipeline
        start_time = time.time()
        results = self.pipeline(
            formatted_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=kwargs.get("top_p", 0.9),
            do_sample=True,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract the generated text
        output = results[0]["generated_text"]
        
        # Log response
        self._log_response_details(output, output)
        
        logger.info(f"Transformers generation completed - response length: {len(output)} chars")
        
        # Check for tool calls if tools were provided
        if tools:
            tool_response = self._extract_tool_calls(output)
            if tool_response and tool_response.has_tool_calls():
                return GenerateResponse(
                    content=output,
                    tool_calls=tool_response,
                    model=model_name,
                    usage={
                        "prompt_tokens": len(self.tokenizer.encode(formatted_prompt)),
                        "completion_tokens": len(self.tokenizer.encode(output)),
                        "total_tokens": len(self.tokenizer.encode(formatted_prompt)) + len(self.tokenizer.encode(output)),
                        "time": time.time() - start_time
                    }
                )
        
        return GenerateResponse(
            content=output,
            model=model_name,
            usage={
                "prompt_tokens": len(self.tokenizer.encode(formatted_prompt)) if self.tokenizer else 0,
                "completion_tokens": len(self.tokenizer.encode(output)) if self.tokenizer else 0,
                "total_tokens": (len(self.tokenizer.encode(formatted_prompt)) + len(self.tokenizer.encode(output))) if self.tokenizer else 0,
                "time": time.time() - start_time
            }
        )

    def _format_messages_for_llama_cpp(self, messages: List[Dict[str, Any]], system_prompt: Optional[str]) -> str:
        """Format messages for llama-cpp-python."""
        formatted_lines = []
        
        # Add system prompt if provided and not in messages
        has_system = any(msg["role"] == "system" for msg in messages)
        if system_prompt and not has_system:
            formatted_lines.append(f"System: {system_prompt}")
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted_lines.append(f"System: {content}")
            elif role == "user":
                formatted_lines.append(f"User: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                formatted_lines.append(f"Tool {tool_name}: {content}")
        
        formatted_lines.append("Assistant:")
        return "\n\n".join(formatted_lines)

    def _generate_text_stream(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Any]] = None, messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[GenerateResponse, None, None]:
        """Generate streaming text - not implemented yet."""
        # For now, just generate normally and yield the result
        result = self._generate_text(prompt, system_prompt, tools, messages, **kwargs)
        yield result

    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM provider."""
        return {
            ModelCapability.STREAMING: False,  # Not implemented yet
            ModelCapability.MAX_TOKENS: self.config_manager.get_param(ModelParameter.MAX_TOKENS, 50),
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.MULTI_TURN: True,
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
    