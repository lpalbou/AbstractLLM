"""
MLX provider for AbstractLLM.

This provider leverages Apple's MLX framework for efficient
inference on Apple Silicon devices.
"""

import os
import time
import logging
import platform
import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union, Callable, Tuple, ClassVar, AsyncGenerator

try:
    import mlx.core as mx
    import mlx_lm
    import mlx_vlm
    from mlx_vlm.utils import (
        process_image,
        process_inputs,
        generate,
        stream_generate
    )
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import numpy as np
    MLX_AVAILABLE = True
    MLXLM_AVAILABLE = True
    MLXVLM_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    MLXLM_AVAILABLE = False
    MLXVLM_AVAILABLE = False
    logging.warning(f"MLX not available: {e}")

from abstractllm.interface import AbstractLLMInterface
from abstractllm.enums import ModelParameter, ModelCapability
from abstractllm.types import GenerateResponse
from abstractllm.exceptions import (
    UnsupportedFeatureError,
    ImageProcessingError,
    FileProcessingError,
    MemoryExceededError,
    ModelLoadingError,
    GenerationError
)
from abstractllm.utils.logging import log_request, log_response, log_api_key_missing
from abstractllm.media.factory import MediaFactory
from abstractllm.media.image import ImageInput

# Import the MLX model factory
from abstractllm.providers.mlx_model_factory import MLXModelFactory

# Set up logger
logger = logging.getLogger(__name__)

class MLXProvider(AbstractLLMInterface):
    """
    MLX implementation for AbstractLLM.
    
    This provider leverages Apple's MLX framework for efficient
    inference on Apple Silicon devices.
    """
    
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
            
        if not MLXVLM_AVAILABLE:
            logger.error("MLX-VLM package not found")
            raise ImportError("MLX-VLM is required for MLXProvider. Install with: pip install mlx-vlm")
        
        # Apply tensor type patches
        try:
            from abstractllm.providers.tensor_type_patch import apply_all_patches
            apply_all_patches()
        except ImportError:
            logger.warning("Could not import tensor_type_patch, MLX vision capabilities may be limited")
        
        # Check if running on Apple Silicon
        self._check_apple_silicon()
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
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
        self._processor = None
        self._config = None
        self._is_loaded = False
        
        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized MLX provider with model: {model}")
        
        # Check if the model name indicates vision capabilities
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        logger.debug(f"Checking if model name indicates vision capabilities: {model_name}")
        self._is_vision_model = MLXModelFactory.is_vision_model(model_name)
        if self._is_vision_model:
            logger.debug(f"Model name indicates vision capabilities: {model_name}")
            self._model_type = MLXModelFactory.determine_model_type(model_name)
            logger.info(f"Detected vision model type: {self._model_type}")
        else:
            self._model_type = None

    @staticmethod
    def list_cached_models(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all models cached by this implementation.
        
        Args:
            cache_dir: Custom cache directory path (uses default if None)
            
        Returns:
            List of cached model information
        """
        # For now, just leverage HF's cache scanning
        try:
            from huggingface_hub import scan_cache_dir
            
            # Use default HF cache if not specified
            if cache_dir is None:
                cache_dir = "~/.cache/huggingface/hub"
                
            if cache_dir and '~' in cache_dir:
                cache_dir = os.path.expanduser(cache_dir)
            
            logger.info(f"Scanning cache directory: {cache_dir}")
                
            # Scan the cache
            cache_info = scan_cache_dir(cache_dir)
            
            # Filter to only include MLX models
            mlx_models = []
            for repo in cache_info.repos:
                # Look for MLX models by name or content
                if "mlx" in repo.repo_id.lower():
                    mlx_models.append({
                        "name": repo.repo_id,
                        "size": repo.size_on_disk,
                        "last_used": repo.last_accessed,
                        "implementation": "mlx"
                    })
            
            logger.info(f"Found {len(mlx_models)} MLX models in cache")
            return mlx_models
        except ImportError:
            logger.warning("huggingface_hub not available for cache scanning")
            return []

    @staticmethod
    def clear_model_cache(model_name: Optional[str] = None) -> None:
        """
        Clear model cache for this implementation.
        
        Args:
            model_name: Specific model to clear (or all if None)
        """
        # Use the factory's cache clearing method
        MLXModelFactory.clear_cache(model_name)
        
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

    def _get_model_config(self) -> Dict[str, Any]:
        """Get the configuration for the current model type."""
        return MLXModelFactory.get_model_config(self._model_type)

    def _check_memory_requirements(self, image_size: Tuple[int, int], num_images: int = 1) -> None:
        """Check if processing these images might exceed memory limits."""
        try:
            # Get model-specific configuration
            config = self._get_model_config()
            target_size = config["image_size"]
            
            # Calculate memory for image processing pipeline
            # 1. Original image (RGB uint8)
            orig_mem = image_size[0] * image_size[1] * 3
            
            # 2. Resized image (float32 + original)
            resize_mem = target_size[0] * target_size[1] * 3 * 4 + orig_mem
            
            # 3. Normalized image (float32 + resized)
            norm_mem = resize_mem * 2
            
            # 4. Model processing buffer (estimate 2x normalized)
            proc_mem = norm_mem * 2
            
            # 5. Model output buffer (estimate based on target size)
            output_buffer = target_size[0] * target_size[1] * 16  # Conservative estimate
            
            # Total per image
            total_per_image = proc_mem + output_buffer
            total_bytes = total_per_image * num_images
            
            # Add safety margin (50%)
            total_bytes = int(total_bytes * 1.5)
            
            # Get system memory info
            import psutil
            mem = psutil.virtual_memory()
            available = mem.available
            
            # Check against Metal buffer size limit (80% of 77GB to be safe)
            # Metal has a hard limit of ~77GB per buffer
            METAL_BUFFER_LIMIT = int(77309411328 * 0.8)  # ~62GB
            
            if total_bytes > METAL_BUFFER_LIMIT:
                raise MemoryExceededError(
                    f"Processing {num_images} image(s) of size {image_size} would require "
                    f"{total_bytes/1e9:.2f}GB, exceeding Metal buffer limit of {METAL_BUFFER_LIMIT/1e9:.2f}GB. "
                    f"Try using a smaller image or resizing it before processing.",
                    provider="mlx",
                    required_memory=total_bytes,
                    available_memory=METAL_BUFFER_LIMIT
                )
            
            # Check against available system memory (with 20% safety margin)
            if total_bytes > available * 0.8:
                raise MemoryExceededError(
                    f"Processing {num_images} image(s) of size {image_size} would require "
                    f"{total_bytes/1e9:.2f}GB with only {available/1e9:.2f}GB available. "
                    f"Try closing other applications to free up memory.",
                    provider="mlx",
                    required_memory=total_bytes,
                    available_memory=available
                )
            
            logger.debug(
                f"Memory check passed: {total_bytes/1e9:.2f}GB required for {num_images} image(s), "
                f"{available/1e9:.2f}GB available"
            )
            
        except Exception as e:
            if isinstance(e, MemoryExceededError):
                raise
            logger.error(f"Error checking memory requirements: {e}")
            raise RuntimeError(f"Failed to check memory requirements: {str(e)}")

    def _process_image_for_model(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Process image for a specific model with precise dimensions."""
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            # Check if resize is needed
            if image.size[0] != target_size[0] or image.size[1] != target_size[1]:
                # Create a new empty image with the exact target dimensions
                new_image = Image.new("RGB", target_size, (0, 0, 0))
                
                # Resize while maintaining aspect ratio
                ratio = min(target_size[0]/image.width, target_size[1]/image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                resized = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Calculate position to center
                pos_x = (target_size[0] - new_size[0]) // 2
                pos_y = (target_size[1] - new_size[1]) // 2
                
                # Paste the resized image onto the black canvas
                new_image.paste(resized, (pos_x, pos_y))
                return new_image
            else:
                # Image is already the right size
                return image
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}", provider="mlx")

    def _process_image(self, image_input: ImageInput) -> mx.array:
        """Process image input into MLX format with robust error handling."""
        try:
            # Get image content
            image_content = image_input.get_content()
            
            # Convert to PIL Image with robust error handling
            try:
                if isinstance(image_content, Image.Image):
                    image = image_content
                elif isinstance(image_content, (str, Path)):
                    image = Image.open(image_content)
                else:
                    from io import BytesIO
                    image = Image.open(BytesIO(image_content))
            except Exception as e:
                raise ImageProcessingError(f"Failed to load image: {str(e)}", provider="mlx")

            # Get original size for memory check
            orig_size = image.size
            
            # Pre-downsample very large images before memory check to avoid excessive memory use
            MAX_DIMENSION = 1024  # Maximum dimension for initial downsample
            if orig_size[0] > MAX_DIMENSION or orig_size[1] > MAX_DIMENSION:
                # Calculate new size maintaining aspect ratio
                if orig_size[0] > orig_size[1]:
                    new_size = (MAX_DIMENSION, int(orig_size[1] * (MAX_DIMENSION / orig_size[0])))
                else:
                    new_size = (int(orig_size[0] * (MAX_DIMENSION / orig_size[1])), MAX_DIMENSION)
                
                logger.info(f"Pre-downsampling large image from {orig_size} to {new_size}")
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                orig_size = new_size
            
            # Now check memory requirements with potentially downsampled image
            self._check_memory_requirements(orig_size)
            
            # Get model-specific configuration
            config = self._get_model_config()
            target_size = config["image_size"]
            
            logger.debug(f"Processing image for model type {self._model_type} with target size {target_size}")
            
            # Process the image with exact dimensions needed for the model
            image = self._process_image_for_model(image, target_size)
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Normalize using model-specific values
            mean = np.array(config["mean"], dtype=np.float32).reshape(1, 1, 3)
            std = np.array(config["std"], dtype=np.float32).reshape(1, 1, 3)
            image_array = (image_array - mean) / std
            
            # Convert to CHW format (channel, height, width)
            image_array = np.transpose(image_array, (2, 0, 1))
                
            # Convert to MLX array
            return mx.array(image_array)
                
        except Exception as e:
            if isinstance(e, (ImageProcessingError, MemoryExceededError)):
                raise
            raise ImageProcessingError(f"Image processing failed: {str(e)}", provider="mlx")

    def _format_prompt(self, prompt: str, num_images: int) -> str:
        """Format prompt with image tokens based on model configuration."""
        return MLXModelFactory.format_prompt(self._model_type, prompt, num_images)

    def _format_system_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format system and user prompts together based on model type."""
        return MLXModelFactory.format_system_prompt(self._model_type, system_prompt, user_prompt)

    def load_model(self) -> None:
        """Load the MLX model and processor."""
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check if model is already loaded
        if self._is_loaded and self._model is not None and self._processor is not None:
            logger.debug(f"Model {model_name} already loaded")
            return
        
        try:
            # Use the factory to load the model
            self._model, self._processor, self._config = MLXModelFactory.load_model(
                model_name, 
                self._is_vision_model
            )
            self._is_loaded = True
            
        except Exception as e:
            if isinstance(e, ModelLoadingError):
                raise
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadingError(
                f"Failed to load MLX model {model_name}: {str(e)}",
                provider="mlx",
                model_name=model_name
            )

    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate a response using the MLX model."""
        try:
            # Log generation request
            logger.info(f"Generation request: model={self.config_manager.get_param(ModelParameter.MODEL)}, "
                       f"stream={stream}, has_system_prompt={system_prompt is not None}, "
                       f"files={len(files) if files else 0}")
            
            # Load model if not already loaded
            if not self._is_loaded:
                logger.info("Model not loaded, loading now")
                self.load_model()
            
            # Process files if provided
            images = []
            image_paths = []  # Store original image paths
            if files:
                logger.info(f"Processing {len(files)} files")
                for file_path in files:
                    try:
                        logger.debug(f"Processing file: {file_path}")
                        media_input = MediaFactory.from_source(file_path)
                        logger.debug(f"Media type: {media_input.media_type}")
                        
                        if media_input.media_type == "image":
                            if not self._is_vision_model:
                                logger.warning(f"Image provided but model {self.config_manager.get_param(ModelParameter.MODEL)} is not a vision model")
                                raise UnsupportedFeatureError(
                                    "vision",
                                    "This model does not support vision inputs",
                                    provider="mlx"
                                )
                            logger.debug(f"Processing image: {file_path}")
                            images.append(self._process_image(media_input))
                            image_paths.append(str(file_path))  # Store the original path
                            logger.debug(f"Image processed successfully: {file_path}")
                        elif media_input.media_type == "text":
                            # Append text content to prompt
                            logger.debug(f"Appending text content from {file_path}")
                            prompt += f"\n\nFile content from {file_path}:\n{media_input.get_content()}"
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                        if isinstance(e, (UnsupportedFeatureError, ImageProcessingError, MemoryExceededError)):
                            raise
                        raise FileProcessingError(
                            f"Failed to process file {file_path}: {str(e)}",
                            provider="mlx",
                            original_exception=e
                        )
            
            # Get generation parameters
            temperature = kwargs.get("temperature",
                                   self.config_manager.get_param(ModelParameter.TEMPERATURE))
            max_tokens = kwargs.get("max_tokens",
                                  self.config_manager.get_param(ModelParameter.MAX_TOKENS))
            top_p = kwargs.get("top_p",
                             self.config_manager.get_param(ModelParameter.TOP_P))
            
            logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
            
            # Prepare generation kwargs
            gen_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            # Handle vision model generation
            if self._is_vision_model and images:
                logger.info(f"Using vision model with {len(images)} images")
                try:
                    # Format prompt with image tokens
                    formatted_prompt = self._format_prompt(prompt, len(images))
                    logger.debug(f"Formatted prompt with image tokens: {formatted_prompt[:100]}...")
                    
                    if system_prompt:
                        logger.debug(f"Adding system prompt: {system_prompt[:100]}...")
                        formatted_prompt = self._format_system_prompt(system_prompt, formatted_prompt)
                    
                    # Generate response
                    if stream:
                        logger.info("Starting vision streaming generation")
                        return self._generate_vision_stream(formatted_prompt, images, image_paths, **gen_kwargs)
                    else:
                        logger.info("Starting vision generation")
                        return self._generate_vision(formatted_prompt, images, image_paths, **gen_kwargs)
                except Exception as e:
                    logger.error(f"Vision generation failed: {e}", exc_info=True)
                    raise GenerationError(f"Vision generation failed: {str(e)}")
                    
            # Handle text-only generation
            logger.info("Using text-only generation")
            formatted_prompt = prompt
            if system_prompt:
                logger.debug(f"Adding system prompt: {system_prompt[:100]}...")
                formatted_prompt = self._format_system_prompt(system_prompt, prompt)
                
            # Generate response
            if stream:
                logger.info("Starting text streaming generation")
                return self._generate_text_stream(formatted_prompt, **gen_kwargs)
            else:
                logger.info("Starting text generation")
                return self._generate_text(formatted_prompt, **gen_kwargs)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            if isinstance(e, (UnsupportedFeatureError, ImageProcessingError, FileProcessingError, 
                            MemoryExceededError, GenerationError)):
                raise
            raise GenerationError(f"Generation failed: {str(e)}")

    def _generate_vision(self, prompt: str, images: List[mx.array], image_paths: List[str], **kwargs) -> GenerateResponse:
        """Generate text from images using MLX-VLM."""
        try:
            # We can only handle one image at a time with current MLX-VLM implementations
            if len(images) > 1:
                logger.warning(f"Multiple images provided ({len(images)}), but only using the first one due to MLX-VLM limitations")
            
            # Get the first image path
            image_path = image_paths[0] if image_paths else None
            
            if image_path is None:
                raise ValueError("No valid image provided for vision generation")
            
            # Log image path for debugging
            logger.debug(f"Using image at path {image_path} for vision generation")
            
            # Format prompt for vision model
            formatted_prompt = self._format_prompt_for_vision(prompt)
            
            # Get max tokens
            max_tokens = kwargs.get("max_tokens", 100)
            
            # Get tokenizer
            if hasattr(self._processor, 'tokenizer'):
                tokenizer = self._processor.tokenizer
            else:
                tokenizer = self._processor
            
            # Use our direct generation approach
            logger.info("Using direct vision generation approach")
            text_content = self._generate_vision_directly(
                self._model,
                tokenizer,
                image_path,
                formatted_prompt,
                max_tokens
            )
            
            # Get the tokenizer for token counting
            tokenizer = self._processor.tokenizer if hasattr(self._processor, "tokenizer") else self._processor
            
            # Create response
            try:
                # Try to count tokens
                prompt_tokens = len(tokenizer.encode(formatted_prompt))
                completion_tokens = len(tokenizer.encode(text_content))
            except Exception:
                # Fallback to rough estimates
                prompt_tokens = len(formatted_prompt.split())
                completion_tokens = len(text_content.split())
            
            return GenerateResponse(
                content=text_content,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )
        except Exception as e:
            logger.error(f"Vision generation failed: {e}")
            raise RuntimeError(f"Vision generation failed: {str(e)}")

    def _generate_vision_stream(self, prompt: str, images: List[mx.array], image_paths: List[str], **kwargs) -> Generator[GenerateResponse, None, None]:
        """Stream generate text from images using MLX-VLM."""
        try:
            # We can only handle one image at a time with current MLX-VLM implementations
            if len(images) > 1:
                logger.warning(f"Multiple images provided ({len(images)}), but only using the first one due to MLX-VLM limitations")
            
            # Get the first image path
            image_path = image_paths[0] if image_paths else None
            
            if image_path is None:
                raise ValueError("No valid image provided for vision generation")
            
            # Log image path for debugging
            logger.debug(f"Using image at path {image_path} for vision generation")
            
            # Get model config and check dimensions
            config = self._get_model_config()
            target_size = config["image_size"]
            
            # Debug image existence and dimensions
            try:
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    logger.debug(f"Image loaded successfully: {image_path}, dimensions: {img.size}, mode: {img.mode}")
                    
                    # If dimensions don't match, resize the image
                    if img.size[0] != target_size[0] or img.size[1] != target_size[1]:
                        logger.warning(f"Image dimensions {img.size} don't match expected {target_size}, resizing temporarily")
                        img_resized = self._process_image_for_model(img, target_size)
                        temp_path = f"{image_path}.resized.jpg"
                        img_resized.save(temp_path)
                        image_path = temp_path
                        logger.debug(f"Using resized image at {temp_path}")
                else:
                    logger.error(f"Image file not found at {image_path}")
            except Exception as img_debug_e:
                logger.warning(f"Failed to debug check image: {img_debug_e}")
            
            # Only use max_tokens parameter which is supported
            max_tokens = kwargs.get("max_tokens", 100)
            
            logger.debug(f"Streaming vision with max_tokens={max_tokens}")
            
            # Use the MLXModelFactory to stream generate with the image
            for response in MLXModelFactory.stream_generate_vision(
                self._model,
                self._processor,
                prompt=prompt,
                image_path=image_path,
                max_tokens=max_tokens
            ):
                yield GenerateResponse(
                    content=response.text if hasattr(response, "text") else response,
                    model=self.config_manager.get_param(ModelParameter.MODEL),
                    usage={
                        "prompt_tokens": response.prompt_tokens if hasattr(response, "prompt_tokens") else 0,
                        "completion_tokens": response.generation_tokens if hasattr(response, "generation_tokens") else 0,
                        "total_tokens": (response.prompt_tokens + response.generation_tokens) if hasattr(response, "prompt_tokens") else 0
                    },
                    image_paths=image_paths
                )
                
            # Clean up temporary file if we created one
            if image_path.endswith('.resized.jpg') and os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    logger.debug(f"Removed temporary resized image: {image_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {image_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Vision generation streaming failed: {e}", exc_info=True)
            raise RuntimeError(f"Vision generation streaming failed: {str(e)}")

    def _generate_text(self, prompt: str, **kwargs) -> GenerateResponse:
        """Generate text using MLX-LM."""
        try:
            # Only use max_tokens parameter which is supported
            max_tokens = kwargs.get("max_tokens", 100)
            
            logger.debug(f"Generating text with max_tokens={max_tokens}")
            
            # Use a direct approach for text generation
            tokenizer = self._processor
            model = self._model
            
            # Tokenize the prompt
            prompt_ids = tokenizer.encode(prompt)
            
            # Initial input
            y = mx.array(prompt_ids)
            
            # Generation loop
            generated_tokens = []
            for _ in range(max_tokens):
                # Get next token
                outputs = model(y)
                
                # Extract logits
                logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs
                
                # Get next token (simple greedy approach)
                next_token = mx.argmax(logits[-1])
                token_id = next_token.item()
                
                # Check for end of sequence
                if hasattr(tokenizer, "eos_token_id") and token_id == tokenizer.eos_token_id:
                    break
                    
                # Add token to generated text
                generated_tokens.append(token_id)
                
                # Update input for next iteration
                y = mx.concatenate([y, next_token.reshape(1)])
            
            # Decode generated tokens
            output = tokenizer.decode(generated_tokens)

            return GenerateResponse(
                content=output,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                usage={
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": len(prompt_ids) + len(generated_tokens)
                }
            )
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def _generate_text_stream(self, prompt: str, **kwargs) -> Generator[GenerateResponse, None, None]:
        """Stream text using MLX-LM."""
        try:
            # Only use max_tokens parameter which is supported
            max_tokens = kwargs.get("max_tokens", 100)
            
            logger.debug(f"Streaming text with max_tokens={max_tokens}")
            
            start_time = time.time()
            for chunk in MLXModelFactory.generate_text(
                self._model,
                self._processor,
                prompt=prompt,
                max_tokens=max_tokens,
                stream=True
            ):
                yield GenerateResponse(
                    content=chunk,
                    model=self.config_manager.get_param(ModelParameter.MODEL),
                    usage={
                        "prompt_tokens": len(self._processor.encode(prompt)),
                        "completion_tokens": len(self._processor.encode(chunk)),
                        "total_tokens": len(self._processor.encode(prompt)) + len(self._processor.encode(chunk)),
                        "time": time.time() - start_time
                    }
                )
        except Exception as e:
            logger.error(f"Text streaming failed: {e}")
            raise RuntimeError(f"Text streaming failed: {str(e)}")

    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM provider."""
        capabilities = {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: self.config_manager.get_param(ModelParameter.MAX_TOKENS, 4096),
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.TOOL_USE: False,
            ModelCapability.VISION: self._is_vision_model,
        }
        
        return capabilities

    async def generate_async(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           files: Optional[List[Union[str, Path]]] = None,
                           stream: bool = False,
                           tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                           **kwargs) -> Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]:
        """Generate a response asynchronously using the MLX model."""
        import asyncio
        loop = asyncio.get_event_loop()

        if stream:
            # For streaming, we need to wrap the generator in an async generator
            async def async_stream():
                for response in await loop.run_in_executor(
                    None,
                    lambda: self.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        files=files,
                        stream=True,
                        tools=tools,
                        **kwargs
                    )
                ):
                    yield response
            return async_stream()
        else:
            # For non-streaming, we can just run the synchronous function in the executor
            return await loop.run_in_executor(
                None,
                lambda: self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    files=files,
                    stream=False,
                    tools=tools,
                    **kwargs
                )
            ) 

    def _generate_vision_directly(self, model, tokenizer, image_path, prompt, max_tokens=100):
        """
        Generate text from an image using direct model calls without relying on libraries.
        
        Args:
            model: The MLX model to use
            tokenizer: The tokenizer to use
            image_path: Path to the image file
            prompt: The text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text from the image
        """
        try:
            # Load and preprocess image
            from PIL import Image
            import mlx.core as mx
            import numpy as np
            
            # Get model config for image size
            config = self._get_model_config()
            target_size = config["image_size"]
            
            # Open and resize image
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Resize image to target size
            img = self._process_image_for_model(img, target_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Apply normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            img_array = (img_array - mean) / std
            
            # Convert to CHW format (channels first)
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # Add batch dimension and convert to mx array
            pixel_values = mx.array(img_array).reshape(1, 3, target_size[0], target_size[1])
            
            # Try using the standard mlx_lm library for generation
            try:
                import mlx_lm
                
                # Format the prompt for the model type
                if "<image>" not in prompt and "llava" in self._model_type.lower():
                    formatted_prompt = f"<image>\n{prompt}"
                else:
                    formatted_prompt = prompt
                
                # Use the mlx_lm generate function
                logger.info("Using mlx_lm.generate for text generation")
                
                # Create a simple wrapper function to handle the image
                def model_with_image(x, **kwargs):
                    """Wrapper function to include the image in the model call."""
                    return model(x, pixel_values=pixel_values, mask=mx.ones_like(x), **kwargs)
                
                # Generate text using the wrapper
                text = mlx_lm.generate.generate(
                    model_with_image,
                    tokenizer,
                    prompt=formatted_prompt,
                    temp=0.1,
                    max_tokens=max_tokens,
                    verbose=False
                )
                
                return text
            except Exception as e:
                logger.warning(f"mlx_lm.generate failed: {e}, falling back to manual generation")
                
                # Handle tokenization
                try:
                    # Try to encode prompt
                    prompt_ids = tokenizer.encode(prompt) if hasattr(tokenizer, "encode") else []
                    
                    if len(prompt_ids) == 0 and hasattr(tokenizer, "convert_tokens_to_ids"):
                        # Try tokenize then convert
                        tokens = tokenizer.tokenize(prompt)
                        prompt_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
                    # If still empty, use a default prompt ID sequence
                    if len(prompt_ids) == 0:
                        logger.warning("Could not encode prompt, using default token IDs")
                        prompt_ids = [1, 2, 3]  # Generic starter tokens
                except Exception as e:
                    logger.error(f"Tokenization failed: {e}")
                    prompt_ids = [1, 2, 3]  # Generic starter tokens
                
                # Start generation
                logger.info(f"Starting direct generation with prompt IDs: {prompt_ids[:10]}...")
                
                # Initial tokens
                y = mx.array(prompt_ids)
                
                # Generation loop
                generated_tokens = []
                
                # Try different model input signatures to handle different MLX models
                try:
                    # For LLaVA models, we know the exact signature needed
                    if "llava" in self._model_type.lower():
                        # Create mask for attention
                        mask = mx.ones(len(y))
                        
                        # Call model with the required parameters for LLaVA
                        output = model(y, pixel_values=pixel_values, mask=mask)
                        
                        # Debug output structure
                        logger.info(f"Model output type: {type(output)}")
                        if isinstance(output, tuple):
                            logger.info(f"Output tuple length: {len(output)}")
                            for i, item in enumerate(output):
                                logger.info(f"Output[{i}] type: {type(item)}")
                        elif isinstance(output, dict):
                            logger.info(f"Output dict keys: {output.keys()}")
                        
                        # Extract logits
                        logits = None
                        if isinstance(output, dict) and "logits" in output:
                            logits = output["logits"]
                        elif isinstance(output, tuple) and len(output) > 0:
                            # Try to get logits from first element if it's a tuple
                            if isinstance(output[0], dict) and "logits" in output[0]:
                                logits = output[0]["logits"]
                            else:
                                # Assume first element is logits directly
                                logits = output[0]
                        
                        if logits is None:
                            logger.error("Could not extract logits from model output")
                            return "Could not generate text from this image."
                        
                        # Generate tokens
                        for _ in range(max_tokens):
                            # Get next token (simple greedy approach)
                            next_token = mx.argmax(logits[-1], axis=0)
                            token_id = next_token.item()
                            
                            # Check for EOS token
                            if hasattr(tokenizer, "eos_token_id") and token_id == tokenizer.eos_token_id:
                                break
                                
                            # Add token and update
                            generated_tokens.append(token_id)
                            
                            # Update input sequence
                            y = mx.concatenate([y, next_token.reshape(1)])
                            mask = mx.ones(len(y))
                            
                            # Get next logits
                            output = model(y, pixel_values=pixel_values, mask=mask)
                            logits = output["logits"]
                    else:
                        # For other model types, we'll need to try different approaches
                        logger.error(f"Unsupported model type: {self._model_type}")
                        return "Model type not supported for direct generation."
                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    return "Could not generate text from this image due to an error."
                
                # Decode generated tokens
                if generated_tokens:
                    try:
                        text = tokenizer.decode(generated_tokens)
                        return text
                    except Exception as decode_e:
                        logger.error(f"Decoding failed: {decode_e}")
                        return f"Generated {len(generated_tokens)} tokens but couldn't decode them."
                else:
                    return "Could not generate text from this image."
                
        except Exception as e:
            logger.error(f"Direct vision generation failed: {e}")
            return "Image processing failed. Could not generate a description." 