"""
MLX provider for AbstractLLM.

This provider leverages Apple's MLX framework for efficient
inference on Apple Silicon devices.
"""

import os
import time
import logging
import platform
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union, Callable, Tuple, ClassVar, AsyncGenerator

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import mlx_lm
    MLXLM_AVAILABLE = True
except ImportError:
    MLXLM_AVAILABLE = False

from abstractllm.interface import AbstractLLMInterface
from abstractllm.enums import ModelParameter, ModelCapability
from abstractllm.types import GenerateResponse
from abstractllm.exceptions import UnsupportedFeatureError, FileProcessingError
from abstractllm.utils.logging import log_request, log_response, log_api_key_missing
from abstractllm.media.factory import MediaFactory

# Set up logger
logger = logging.getLogger("abstractllm.providers.mlx")

class MLXProvider(AbstractLLMInterface):
    """
    MLX implementation for AbstractLLM.
    
    This provider leverages Apple's MLX framework for efficient
    inference on Apple Silicon devices.
    """
    
    # Class-level model cache
    _model_cache: ClassVar[Dict[str, Tuple[Any, Any, float]]] = {}
    _max_cached_models = 2  # Default to 2 models in memory
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the MLX provider."""
        super().__init__(config)
        
        # Check for MLX availability
        if not MLX_AVAILABLE:
            logger.error("MLX package not found")
            raise ImportError("MLX is required for MLXProvider. Install with: pip install mlx")
        
        if not MLXLM_AVAILABLE:
            logger.error("MLX-LM package not found")
            raise ImportError("MLX-LM is required for MLXProvider. Install with: pip install mlx-lm")
        
        # Check if running on Apple Silicon
        self._check_apple_silicon()
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "mlx-community/qwen2.5-coder-14b-instruct-abliterated",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 4096,
            ModelParameter.TOP_P: 0.9,
            "cache_dir": None,  # Use default HuggingFace cache
            "quantize": True    # Use quantized models by default
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize components
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._is_vision_model = False
        
        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized MLX provider with model: {model}")
        
    def _check_apple_silicon(self) -> None:
        """Check if running on Apple Silicon."""
        is_macos = platform.system().lower() == "darwin"
        if not is_macos:
            logger.warning(f"MLX requires macOS, current platform: {platform.system()}")
            raise UnsupportedFeatureError(
                feature="mlx",
                message="MLX provider is only available on macOS with Apple Silicon",
                provider="mlx"
            )
        
        # Check processor architecture
        is_arm = platform.processor() == "arm"
        if not is_arm:
            logger.warning(f"MLX requires Apple Silicon, current processor: {platform.processor()}")
            raise UnsupportedFeatureError(
                feature="mlx",
                message="MLX provider requires Apple Silicon (M1/M2/M3) hardware",
                provider="mlx"
            )
        
        logger.info(f"Platform check successful: macOS with Apple Silicon detected")
        
    def load_model(self) -> None:
        """
        Load the MLX model and tokenizer.
        
        This method will check the cache first before loading from the source.
        """
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check in-memory cache first
        if model_name in self._model_cache:
            logger.info(f"Loading model {model_name} from in-memory cache")
            self._model, self._tokenizer, _ = self._model_cache[model_name]
            # Update last access time
            self._model_cache[model_name] = (self._model, self._tokenizer, time.time())
            self._is_loaded = True
            
            # Check if this is a vision model
            self._is_vision_model = self._check_vision_capability(model_name)
            return
        
        # If not in memory cache, load from disk/HF
        logger.info(f"Loading model {model_name}")
        
        try:
            # Import MLX-LM utilities
            from mlx_lm.utils import load
            
            # Check if this is a vision model
            self._is_vision_model = self._check_vision_capability(model_name)
            if self._is_vision_model:
                logger.info(f"Detected vision capabilities in model {model_name}")
            
            # Log loading parameters
            logger.debug(f"Loading MLX model: {model_name}")
            
            # Load the model using MLX-LM (removed cache_dir parameter)
            self._model, self._tokenizer = load(model_name)
            self._is_loaded = True
            
            # Add to in-memory cache
            self._update_model_cache(model_name)
            
            logger.info(f"Successfully loaded model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Failed to load MLX model: {str(e)}")

    def _update_model_cache(self, model_name: str) -> None:
        """Update the model cache with the current model."""
        self._model_cache[model_name] = (self._model, self._tokenizer, time.time())
        
        # Prune cache if needed
        if len(self._model_cache) > self._max_cached_models:
            # Find oldest model by last access time
            oldest_key = min(self._model_cache.keys(), 
                            key=lambda k: self._model_cache[k][2])
            logger.info(f"Removing {oldest_key} from model cache")
            del self._model_cache[oldest_key]
            
    def _check_vision_capability(self, model_name: str) -> bool:
        """
        Check if a model has vision capabilities based on its name.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            True if the model likely supports vision, False otherwise
        """
        # List of keywords that indicate vision capabilities
        vision_keywords = ["llava", "clip", "vision", "blip", "image", "vit", "visual", "multimodal"]
        
        # Check if any vision keyword is in the model name (case insensitive)
        model_name_lower = model_name.lower()
        for keyword in vision_keywords:
            if keyword in model_name_lower:
                logger.debug(f"Vision capability detected for model {model_name} (matched '{keyword}')")
                return True
            
        return False

    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate a response using the MLX model."""
        # Load model if not already loaded
        if not self._is_loaded:
            self.load_model()
        
        # Process system prompt if provided
        formatted_prompt = prompt
        if system_prompt:
            # Use model's chat template if available
            if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
                # Construct messages in the expected format
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                try:
                    # Try to use HF's template application
                    from transformers import AutoTokenizer
                    formatted_prompt = AutoTokenizer.apply_chat_template(
                        messages, 
                        chat_template=self._tokenizer.chat_template,
                        tokenize=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply chat template: {e}")
                    formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                # Simple concatenation fallback
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Process files if provided
        if files and len(files) > 0:
            try:
                formatted_prompt = self._process_files(formatted_prompt, files)
            except (UnsupportedFeatureError, FileProcessingError) as e:
                # Pass through our custom exceptions
                raise e
            except Exception as e:
                # Wrap unknown exceptions
                logger.error(f"Unexpected error processing files: {e}")
                raise FileProcessingError(f"Failed to process input files: {str(e)}", provider="mlx")
        
        # Tools are not supported
        if tools:
            raise UnsupportedFeatureError(
                "tool_use",
                "MLX provider does not support tool use or function calling",
                provider="mlx"
            )
        
        # Get generation parameters
        temperature = kwargs.get("temperature", 
                               self.config_manager.get_param(ModelParameter.TEMPERATURE))
        max_tokens = kwargs.get("max_tokens", 
                              self.config_manager.get_param(ModelParameter.MAX_TOKENS))
        top_p = kwargs.get("top_p", 
                         self.config_manager.get_param(ModelParameter.TOP_P))
        
        # Validate parameters
        if temperature is not None and (temperature < 0 or temperature > 2):
            logger.warning(f"Temperature {temperature} out of recommended range [0, 2], clamping")
            temperature = max(0, min(temperature, 2))
        
        if max_tokens is not None and max_tokens <= 0:
            logger.warning(f"Invalid max_tokens {max_tokens}, using default")
            max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS)
        
        if top_p is not None and (top_p <= 0 or top_p > 1):
            logger.warning(f"Top_p {top_p} out of valid range (0, 1], clamping")
            top_p = max(0.001, min(top_p, 1.0))
        
        # Log request parameters
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        log_request("mlx", prompt, {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "has_files": bool(files)
        })
        
        # Encode prompt
        try:
            prompt_tokens = self._tokenizer.encode(formatted_prompt)
        except Exception as e:
            logger.error(f"Failed to encode prompt: {e}")
            raise RuntimeError(f"Failed to encode prompt: {str(e)}")
        
        # Import MLX-LM generation utilities
        from mlx_lm.utils import generate
        
        # Handle streaming vs non-streaming
        if stream:
            return self._generate_stream(
                prompt_tokens, 
                temperature, 
                max_tokens, 
                top_p
            )
        else:
            # Generate text (non-streaming)
            try:
                output = generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt_tokens,
                    temp=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                
                # Create response
                completion_tokens = len(self._tokenizer.encode(output)) if hasattr(self._tokenizer, "encode") else len(output.split())
                
                # Log the response
                log_response("mlx", output)
                
                return GenerateResponse(
                    text=output,
                    model=self.config_manager.get_param(ModelParameter.MODEL),
                    prompt_tokens=len(prompt_tokens),
                    completion_tokens=completion_tokens,
                    total_tokens=len(prompt_tokens) + completion_tokens
                )
            except Exception as e:
                logger.error(f"Error generating text: {e}")
                raise RuntimeError(f"Error generating text: {str(e)}")
    
    def _generate_stream(self, 
                       prompt_tokens, 
                       temperature: float, 
                       max_tokens: int, 
                       top_p: float) -> Generator[GenerateResponse, None, None]:
        """Generate a streaming response."""
        import mlx.core as mx
        from mlx_lm.utils import generate_step
        
        # Convert to MLX array if not already
        if not isinstance(prompt_tokens, mx.array):
            prompt_tokens = mx.array(prompt_tokens)
        
        # Initial state
        tokens = prompt_tokens
        finish_reason = None
        current_text = ""
        
        # Generate tokens one by one
        for _ in range(max_tokens):
            next_token, _ = generate_step(
                self._model,
                tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Add token to sequence
            tokens = mx.concatenate([tokens, next_token[None]])
            
            # Convert to text
            current_text = self._tokenizer.decode(tokens.tolist()[len(prompt_tokens):])
            
            # Check for EOS token
            if hasattr(self._tokenizer, "eos_token") and self._tokenizer.eos_token in current_text:
                current_text = current_text.replace(self._tokenizer.eos_token, "")
                finish_reason = "stop"
            
            # Create response chunk
            yield GenerateResponse(
                text=current_text,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(tokens) - len(prompt_tokens),
                total_tokens=len(tokens),
                finish_reason=finish_reason
            )
            
            # Stop if we reached the end
            if finish_reason:
                break
        
        # Log the final response for streaming generation
        logger.debug(f"Streaming generation completed: {len(tokens) - len(prompt_tokens)} tokens generated")
        log_response("mlx", current_text)
                
    def _process_files(self, prompt: str, files: List[Union[str, Path]]) -> str:
        """Process input files and append to prompt as needed."""
        processed_prompt = prompt
        has_images = False
        
        # Convert all file paths to Path objects for consistent handling
        file_paths = [Path(f) if isinstance(f, str) else f for f in files]
        
        # Verify files exist before processing
        for file_path in file_paths:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                raise FileProcessingError(f"File not found: {file_path}", provider="mlx", file_path=str(file_path))
        
        logger.info(f"Processing {len(files)} file(s) for MLX model")
        
        # Process each file
        for file_path in file_paths:
            try:
                logger.debug(f"Processing file: {file_path}")
                media_input = MediaFactory.from_source(file_path)
                
                if media_input.media_type == "image":
                    logger.debug(f"Detected image file: {file_path}")
                    has_images = True
                    # Actual image processing will be implemented in a future task
                    # Simply flagging for now to check model compatibility
                elif media_input.media_type == "text":
                    logger.debug(f"Processing text file: {file_path}")
                    # Append text content to prompt with clear formatting
                    processed_prompt += f"\n\n### Content from file '{file_path.name}':\n{media_input.content}\n###\n"
                    logger.debug(f"Added {len(media_input.content)} chars of text content from {file_path.name}")
                else:
                    logger.warning(f"Unsupported media type: {media_input.media_type} for file {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                raise FileProcessingError(f"Failed to process file {file_path}: {str(e)}", provider="mlx", file_path=str(file_path))
        
        # Check if this is a vision model if images are present
        if has_images:
            logger.debug("Images detected, checking if model supports vision")
            if not self._is_vision_model:
                logger.warning("Model does not support vision but images were provided")
                raise UnsupportedFeatureError(
                    "vision",
                    "This model does not support vision inputs. Try using a vision-capable model like 'mlx-community/llava-1.5-7b-mlx'",
                    provider="mlx"
                )
            else:
                logger.debug("Vision-capable model confirmed for image processing")
                
        return processed_prompt
        
    async def generate_async(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None, 
                       files: Optional[List[Union[str, Path]]] = None,
                       stream: bool = False, 
                       tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                       **kwargs) -> Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]:
        """
        Asynchronously generate a response using the MLX model.
        
        This is currently a wrapper around the synchronous method as MLX doesn't provide
        native async support, but follows the required interface.
        """
        import asyncio
        from typing import AsyncGenerator
        
        # Use the current event loop
        loop = asyncio.get_running_loop()
        
        if stream:
            # For streaming, we need to convert the synchronous generator to an async one
            async def async_gen() -> AsyncGenerator[GenerateResponse, None]:
                # Run the sync generate in an executor to avoid blocking
                sync_gen = await loop.run_in_executor(
                    None,
                    lambda: self.generate(
                        prompt, system_prompt, files, stream=True, tools=tools, **kwargs
                    )
                )
                
                # Yield items from the sync generator
                for item in sync_gen:
                    yield item
                    # Small delay to allow other tasks to run, but not too long to maintain responsiveness
                    await asyncio.sleep(0.001)
            
            # Return the async generator directly
            return async_gen()
        else:
            # For non-streaming, we can just run the synchronous method in the executor
            return await loop.run_in_executor(
                None, 
                lambda: self.generate(
                    prompt, system_prompt, files, stream=False, tools=tools, **kwargs
                )
            )
            
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM provider."""
        capabilities = {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: self.config_manager.get_param(ModelParameter.MAX_TOKENS, 4096),
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.TOOL_USE: False,
            ModelCapability.VISION: self._is_vision_capable(),
        }
        
        return capabilities

    def _is_vision_capable(self) -> bool:
        """Check if the current model supports vision capabilities."""
        return self._is_vision_model 