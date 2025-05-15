"""
MLX provider for AbstractLLM.

This provider uses Apple's MLX framework for local inference on Apple Silicon hardware.
"""

import time
import logging
import platform
from pathlib import Path
import os
import base64
from typing import Dict, List, Any, Optional, Generator, Union, Callable, AsyncGenerator, Tuple

# Import the interface class
from abstractllm.interface import (
    AbstractLLMInterface, 
    ModelParameter, 
    ModelCapability,
    GenerateResponse
)
from abstractllm.exceptions import (
    ModelLoadingError,
    GenerationError,
    UnsupportedFeatureError,
    ImageProcessingError,
    FileProcessingError,
    MemoryExceededError
)
from abstractllm.providers.mlx_model_configs import ModelConfigFactory, MLXModelConfig
from abstractllm.media.factory import MediaFactory
from abstractllm.providers.tensor_type_patch import apply_all_patches as apply_tensor_patches

# Set up logging
logger = logging.getLogger("abstractllm.providers.mlx_provider")

# Check for required dependencies
MLX_AVAILABLE = False
MLXLM_AVAILABLE = False
MLXVLM_AVAILABLE = False

try:
    # Import MLX core libraries
    import mlx.core as mx
    import mlx_lm
    MLX_AVAILABLE = True
    MLXLM_AVAILABLE = True
    
    # Try to import MLX vision libraries
    try:
        import mlx_vlm
        from mlx_vlm import load as load_vlm
        from mlx_vlm import generate as generate_vlm
        from mlx_vlm.utils import load_config
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image
        MLXVLM_AVAILABLE = True
    except ImportError:
        MLXVLM_AVAILABLE = False
        logging.warning("MLX-VLM not available. Vision capabilities will be disabled.")
    
except ImportError as e:
    MLX_AVAILABLE = False
    MLXLM_AVAILABLE = False
    logging.warning(f"MLX libraries not available: {e}")
    
# If module is loaded directly without dependencies, prevent provider initialization
if not MLX_AVAILABLE or not MLXLM_AVAILABLE:
    logging.warning("MLX provider disabled due to missing dependencies.")

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
        
        # Apply tensor type and vision patches
        apply_tensor_patches()
        
        # Check if running on Apple Silicon
        self._check_apple_silicon()
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 4096,
            ModelParameter.TOP_P: 0.9,
            "cache_dir": None  # Use default HuggingFace cache
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize components
        self._model = None
        self._processor = None
        self._config = None
        self._is_loaded = False
        self._model_config = None  # Will hold the model-specific config
        
        # Log initialization
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized MLX provider with model: {model_name}")
        
        # Determine if model has vision capabilities
        self._is_vision_model = self._check_vision_model(model_name)
        
    def _check_vision_model(self, model_name: str) -> bool:
        """
        Check if the specified model has vision capabilities.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model supports vision, False otherwise
        """
        # If MLX-VLM is not available, no vision models can be used
        if not MLXVLM_AVAILABLE:
            return False
            
        model_name_lower = model_name.lower()
        
        # Look for vision indicators in model name
        vision_indicators = [
            "vlm", "vision", "visual", "llava", "clip", "multimodal", "vit", 
            "blip", "vqa", "image", "qwen-vl", "idefics", "phi-3-vision", 
            "phi3-vision", "phi-vision", "bakllava", "paligemma"  # Added paligemma
        ]
        
        # Also directly check for known vision models
        known_vision_models = [
            "mlx-community/paligemma-3b-mix-448-8bit",
            "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        ]
        
        # Check if model name directly matches a known vision model
        if any(known_model in model_name for known_model in known_vision_models):
            return True
        
        # Check for vision indicators in model name
        return any(indicator in model_name_lower for indicator in vision_indicators)

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
        """Load the MLX model and processor."""
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Check if model is already loaded
        if self._is_loaded and self._model is not None and self._processor is not None:
            logger.debug(f"Model {model_name} already loaded")
            return
        
        try:
            if self._is_vision_model and MLXVLM_AVAILABLE:
                logger.info(f"Loading vision model: {model_name}")
                # For vision models, use MLX-VLM instead of MLX-LM
                try:
                    self._model, self._processor = load_vlm(model_name, trust_remote_code=True)
                    self._config = load_config(model_name, trust_remote_code=True)
                    logger.info(f"Vision model {model_name} loaded successfully with MLX-VLM")
                except Exception as e:
                    logger.error(f"Failed to load vision model with MLX-VLM: {e}")
                    raise ModelLoadingError(
                        f"Failed to load vision model {model_name} with MLX-VLM: {str(e)}"
                    )
            else:
                logger.info(f"Loading language model: {model_name}")
                self._model, self._processor = mlx_lm.load(model_name)
                self._config = {}  # Language models don't need special config
                logger.info(f"Language model {model_name} loaded successfully")
            
            self._is_loaded = True
            
            # Fix processor if needed (immediately after loading)
            if self._processor is not None:
                if hasattr(self._processor, 'patch_size') and self._processor.patch_size is None:
                    self._processor.patch_size = 14
                    logger.info("Fixed missing patch_size in processor after loading")
                
                if hasattr(self._processor, 'image_processor'):
                    if hasattr(self._processor.image_processor, 'patch_size') and self._processor.image_processor.patch_size is None:
                        self._processor.image_processor.patch_size = 14
                        logger.info("Fixed missing patch_size in image_processor after loading")
            
            # Set up model config
            self._set_model_config(model_name)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadingError(
                f"[mlx] Failed to load MLX model {model_name}: {str(e)}"
            )

    def _set_model_config(self, model_name: str) -> None:
        """
        Set up the model-specific configuration.
        
        Args:
            model_name: The name of the model
        """
        # Get the appropriate configuration for this model
        self._model_config = ModelConfigFactory.get_for_model(model_name)
        
        # Apply the configuration to the tokenizer
        if self._processor is not None:
            try:
                # Check if the processor is a PaliGemmaProcessor or similar that doesn't have add_eos_token
                has_add_eos_token = hasattr(self._processor, 'add_eos_token')
                
                # If this is a PaliGemma processor, create a dummy add_eos_token that does nothing
                # This avoids the warning when the model config tries to add EOS token
                if not has_add_eos_token and 'paligemma' in model_name.lower():
                    def dummy_add_eos_token(token):
                        logger.debug(f"Ignored request to add EOS token for PaliGemma processor: {token}")
                        return
                    
                    # Add the dummy method to suppress warnings
                    self._processor.add_eos_token = dummy_add_eos_token
                    logger.debug("Added dummy add_eos_token method to PaliGemma processor")
                
                # Apply the model-specific configuration
                self._model_config.apply_to_tokenizer(self._processor)
                logger.info(f"Successfully applied model-specific configuration for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to fully apply model configuration: {e}. Some features may not work correctly.")

    def _process_image(self, image_input: Union[str, Path, Any]) -> Image.Image:
        """
        Process an image for vision models.
        
        This method handles both file paths and PIL Image objects, performs
        automatic resizing if needed, and ensures all processing happens in
        memory without creating temporary files.
        
        Args:
            image_input: Path to image file, PIL Image, or image-like object
            
        Returns:
            Processed PIL image ready for model input
        """
        try:
            # Import PIL.Image at the beginning to avoid reference issues
            from PIL import Image
            
            # Handle different input types
            if isinstance(image_input, str) or isinstance(image_input, Path):
                # It's a file path
                img = Image.open(str(image_input))
            elif hasattr(image_input, 'mode') and hasattr(image_input, 'size'):
                # It's already a PIL Image
                img = image_input
            else:
                # Try to handle other image-like objects (numpy arrays, etc.)
                try:
                    import numpy as np
                    if isinstance(image_input, np.ndarray):
                        img = Image.fromarray(image_input)
                    else:
                        raise ImageProcessingError(f"Unsupported image type: {type(image_input)}", provider="mlx")
                except ImportError:
                    raise ImageProcessingError("NumPy required for array-based image processing", provider="mlx")
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Auto-resize if needed - most vision models expect specific dimensions
            if self._processor is not None and hasattr(self._processor, 'image_processor'):
                # Get target size from processor if available
                target_size = None
                
                # PaliGemma and some other models use different ways to specify size
                if hasattr(self._processor.image_processor, 'size'):
                    processor_size = self._processor.image_processor.size
                    # Handle dict case (e.g., {"height": 448, "width": 448})
                    if isinstance(processor_size, dict) and "height" in processor_size and "width" in processor_size:
                        target_size = (processor_size["width"], processor_size["height"])
                    # Handle tuple/list case
                    elif isinstance(processor_size, (tuple, list)) and len(processor_size) == 2:
                        target_size = processor_size
                    # Handle int case
                    elif isinstance(processor_size, int):
                        target_size = (processor_size, processor_size)
                    else:
                        logger.warning(f"Unrecognized size format: {processor_size}. Skipping resize.")
                
                # Some processors use crop_size instead
                elif hasattr(self._processor.image_processor, 'crop_size'):
                    processor_size = self._processor.image_processor.crop_size
                    if isinstance(processor_size, (tuple, list)) and len(processor_size) == 2:
                        target_size = processor_size
                    elif isinstance(processor_size, int):
                        target_size = (processor_size, processor_size)
                
                # If we found a target size, resize the image
                if target_size:
                    # Only resize if current size doesn't match
                    if img.size != target_size:
                        logger.debug(f"Resizing image from {img.size} to {target_size}")
                        img = img.resize(target_size, Image.LANCZOS)
            
            # Fix processor if needed
            if self._processor is not None:
                # Fix patch_size if missing
                if hasattr(self._processor, 'patch_size') and self._processor.patch_size is None:
                    self._processor.patch_size = 14  # Standard patch size for CLIP-like models
                    logger.info("Fixed missing patch_size in processor")
                
                # Also check image_processor if it exists
                if hasattr(self._processor, 'image_processor'):
                    if hasattr(self._processor.image_processor, 'patch_size') and self._processor.image_processor.patch_size is None:
                        self._processor.image_processor.patch_size = 14
                        logger.info("Fixed missing patch_size in image_processor")
                
                # Make sure we can handle PIL image objects directly
                if hasattr(self._processor, 'process_images'):
                    original_process_images = self._processor.process_images
                    
                    def patched_process_images(images, **kwargs):
                        """Ensures image processing can handle PIL images directly."""
                        try:
                            return original_process_images(images, **kwargs)
                        except Exception as e:
                            logger.warning(f"Error in original process_images: {e}. Trying alternate approach.")
                            # Try to convert the image to RGB if it's not already
                            try:
                                if isinstance(images, list):
                                    processed_images = []
                                    for img in images:
                                        if hasattr(img, 'convert'):
                                            img = img.convert('RGB')
                                        processed_images.append(img)
                                    return original_process_images(processed_images, **kwargs)
                                elif hasattr(images, 'convert'):
                                    images = images.convert('RGB')
                                    return original_process_images(images, **kwargs)
                            except Exception as inner_e:
                                logger.error(f"Failed to process image in alternate way: {inner_e}")
                                raise
                    
                    # Apply the fix to process_images
                    if hasattr(self._processor, 'process_images'):
                        self._processor.process_images = patched_process_images
                        logger.info("Fixed processor.process_images to handle PIL images directly")
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to process image {image_input}: {e}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}", provider="mlx")

    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """
        Generate a response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            files: Optional list of files to process
            stream: Whether to stream the response
            tools: Optional list of tools
            **kwargs: Additional parameters

        Returns:
            Generated response or stream of responses
        """
        # Log generation request
        logger.info(f"Generation request: model={self.config_manager.get_param(ModelParameter.MODEL)}, "
                    f"stream={stream}, has_system_prompt={system_prompt is not None}, "
                    f"files={len(files) if files else 0}")
        
        # Make sure model is loaded
        if not self._is_loaded:
            logger.info("Model not loaded, loading now")
            self.load_model()
        
        # Check for tool usage (not supported)
        if tools:
            logger.warning("Tool usage not supported by MLX provider")
            raise UnsupportedFeatureError("tool_use", "Tool usage not supported by MLX provider", "mlx")
        
        # Prepare image paths for vision model if images provided
        image_paths = []
        
        # Handle files if provided
        if files:
            for file_path in files:
                file_path_str = str(file_path)
                file_ext = os.path.splitext(file_path_str)[1].lower()
                
                # Check if it's an image file
                if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
                    image_paths.append(file_path_str)
                else:
                    logger.warning(f"Unsupported file type: {file_ext}")
                    
        # Also check for images passed directly as kwargs
        images = kwargs.get("images", [])
        if images:
            image_paths.extend(images)
        
        # If not a vision model but images provided, warn the user
        if not self._is_vision_model:
            if image_paths:
                logger.warning(f"Image provided but model {self.config_manager.get_param(ModelParameter.MODEL)} is not a vision model")
            
            # Use standard text generation
            if stream:
                return self._generate_text_stream(prompt, **kwargs)
            else:
                return self._generate_text(prompt, **kwargs)
        
        # Vision model processing
        if self._is_vision_model and MLXVLM_AVAILABLE and image_paths:
            # For vision models with images
            if stream:
                return self._generate_vision_stream(prompt, image_paths, **kwargs)
            else:
                return self._generate_vision(prompt, image_paths, **kwargs)
        else:
            # Use text-only generation if no images or vision not available
            logger.info("Using text-only generation")
            if stream:
                return self._generate_text_stream(prompt, **kwargs)
            else:
                return self._generate_text(prompt, **kwargs)

    def _generate_vision(self, prompt: str, image_paths: List[str], **kwargs) -> GenerateResponse:
        """
        Generate a response for a vision prompt using MLX-VLM.
        
        Args:
            prompt: Text prompt
            image_paths: List of image paths, PIL Images, or image-like objects
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        if not self._is_loaded:
            self.load_model()
        
        # Process parameters
        max_tokens = kwargs.get("max_tokens", self.config_manager.get_param(ModelParameter.MAX_TOKENS, 100))
        temperature = kwargs.get("temperature", self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.1))
        
        images = []
        # Process images
        for img_path in image_paths:
            try:
                # Process the image using our enhanced _process_image method
                # This handles both file paths and PIL Images
                img = self._process_image(img_path)
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {e}")
                raise ImageProcessingError(f"Failed to process image: {str(e)}")
        
        if not images:
            raise GenerationError("No valid images provided for vision generation")
        
        try:
            # Use the first image for now (multi-image support could be added later)
            image = images[0]
            
            # Add image token to prompt if not already present
            # This prevents the warning from PaliGemmaProcessor
            if "<image>" not in prompt:
                # For most MLX vision models, image token goes at the beginning
                prompt = "<image> " + prompt
                logger.debug(f"Added <image> token to prompt: {prompt}")
            
            # Generate using MLX-VLM
            logger.info(f"Generating vision response with MLX-VLM")
            result = generate_vlm(
                model=self._model,
                processor=self._processor,
                image=image,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature
            )
            
            # Extract the text from the result (handle different return types)
            if isinstance(result, tuple):
                text = result[0]
                metadata = result[1] if len(result) > 1 else {}
            else:
                text = result
                metadata = {}
            
            # Create and return the response
            return GenerateResponse(
                content=text,
                raw_response=result,
                usage={
                    "completion_tokens": metadata.get("output_tokens", max_tokens),
                    "total_tokens": metadata.get("total_tokens", len(prompt) + max_tokens)
                },
                model=self.config_manager.get_param(ModelParameter.MODEL)
            )
        except Exception as e:
            logger.error(f"Vision generation failed: {e}")
            raise GenerationError(f"Vision generation failed: {str(e)}")

    def _generate_vision_stream(self, prompt: str, image_paths: List[str], **kwargs) -> Generator[GenerateResponse, None, None]:
        """
        Generate streaming response for a vision prompt using MLX-VLM.
        
        Note: True token-by-token streaming is not available in MLX-VLM yet.
        This method generates the full response first and then simulates streaming
        by chunking the response and yielding with small delays.
        
        Args:
            prompt: Text prompt
            image_paths: List of image paths, PIL Images, or image-like objects
            **kwargs: Additional parameters
            
        Yields:
            Stream of generated responses
        """
        if not self._is_loaded:
            self.load_model()
        
        # Process parameters
        max_tokens = kwargs.get("max_tokens", self.config_manager.get_param(ModelParameter.MAX_TOKENS, 100))
        temperature = kwargs.get("temperature", self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.1))
        
        images = []
        # Process images
        for img_path in image_paths:
            try:
                # Process the image using our enhanced _process_image method
                # This handles both file paths and PIL Images
                img = self._process_image(img_path)
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to process image {img_path}: {e}")
                raise ImageProcessingError(f"Failed to process image: {str(e)}")
        
        if not images:
            raise GenerationError("No valid images provided for vision generation")
        
        try:
            # Use the first image for now (multi-image support could be added later)
            image = images[0]
            
            # Add image token to prompt if not already present
            # This prevents the warning from PaliGemmaProcessor
            if "<image>" not in prompt:
                # For most MLX vision models, image token goes at the beginning
                prompt = "<image> " + prompt
                logger.debug(f"Added <image> token to prompt: {prompt}")
            
            # MLX-VLM doesn't have a native streaming implementation yet
            # So we generate the full response first and then simulate streaming
            logger.info(f"Generating vision response with MLX-VLM (simulated streaming)")
            result = generate_vlm(
                model=self._model,
                processor=self._processor,
                image=image,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature
            )
            
            # Extract the text from the result
            if isinstance(result, tuple):
                text = result[0]
                metadata = result[1] if len(result) > 1 else {}
            else:
                text = result
                metadata = {}
            
            # Simulate streaming by yielding chunks of text
            full_text = text
            chunk_size = 8  # Simulate streaming with small chunks
            
            # To make it look more natural, use larger chunks at the beginning and smaller at the end
            total_length = len(full_text)
            tokens_generated = 0
            
            # Stream the response in chunks
            for i in range(0, total_length, chunk_size):
                end_idx = min(i + chunk_size, total_length)
                chunk = full_text[i:end_idx]
                tokens_generated += len(chunk.split())
                
                # Yield a response for this chunk
                yield GenerateResponse(
                    content=full_text[:end_idx],  # Include all text up to this point
                    raw_response=None,
                    usage={
                        "completion_tokens": tokens_generated,
                        "total_tokens": len(prompt.split()) + tokens_generated  # Approximation
                    },
                    model=self.config_manager.get_param(ModelParameter.MODEL)
                )
                
                # Add a small delay to simulate real streaming
                time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Vision stream generation failed: {e}")
            raise GenerationError(f"Vision stream generation failed: {str(e)}")

    def _generate_text(self, prompt: str, **kwargs) -> GenerateResponse:
        """
        Generate text using MLX-LM.
        
        Args:
            prompt: The prompt text
            
        Returns:
            GenerateResponse with the generated content
        """
        try:
            # Get generation parameters
            temperature = kwargs.pop("temperature", 
                                    self.config_manager.get_param(ModelParameter.TEMPERATURE))
            
            # Ensure temperature is valid (not None)
            if temperature is None:
                temperature = 0.7  # Default temperature
                logger.info(f"Temperature was None, using default: {temperature}")
            
            max_tokens = kwargs.pop("max_tokens", 100)
            
            # Get generation parameters from the model config
            try:
                generation_params = self._model_config.get_generation_params(temperature)
            except Exception as e:
                logger.warning(f"Error getting model-specific generation parameters: {e}. Using defaults.")
                generation_params = {}
                # Ensure we at least have a valid sampler
                try:
                    # Use the mlx_lm that was imported at the module level
                    sampler = mlx_lm.sample_utils.make_sampler(temp=float(max(0.01, temperature)))
                    generation_params["sampler"] = sampler
                except Exception as sampler_error:
                    logger.warning(f"Could not create sampler: {sampler_error}")
                    # If we can't create a sampler, we'll let mlx_lm use its defaults
            
            # Use mlx_lm's generate function with model-specific parameters
            generate_kwargs = {
                "model": self._model,
                "tokenizer": self._processor,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "verbose": False,
            }
            
            # Add model-specific generation parameters
            generate_kwargs.update(generation_params)
            
            # Generate text
            output = mlx_lm.generate(**generate_kwargs)
            
            # Count tokens
            try:
                prompt_tokens = len(self._processor.encode(prompt))
                completion_tokens = len(self._processor.encode(output)) - prompt_tokens
                completion_tokens = max(0, completion_tokens)  # Ensure non-negative
            except:
                # Fallback to rough estimates
                prompt_tokens = len(prompt.split())
                completion_tokens = len(output.split()) - prompt_tokens
                completion_tokens = max(0, completion_tokens)  # Ensure non-negative
            
            return GenerateResponse(
                content=output,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise GenerationError(f"Text generation failed: {str(e)}")

    def _generate_text_stream(self, prompt: str, **kwargs) -> Generator[GenerateResponse, None, None]:
        """
        Stream text generation using MLX-LM.
        
        Args:
            prompt: The prompt text
            
        Yields:
            GenerateResponse objects with incremental content
        """
        try:
            # Use mlx_lm's streaming functionality
            current_text = ""
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            
            # Get generation parameters
            temperature = kwargs.pop("temperature", 
                                    self.config_manager.get_param(ModelParameter.TEMPERATURE))
            
            # Ensure temperature is valid (not None)
            if temperature is None:
                temperature = 0.7  # Default temperature
                logger.info(f"Temperature was None, using default: {temperature}")
            
            max_tokens = kwargs.pop("max_tokens", 100)
            
            # Get generation parameters from the model config
            try:
                generation_params = self._model_config.get_generation_params(temperature)
            except Exception as e:
                logger.warning(f"Error getting model-specific generation parameters: {e}. Using defaults.")
                generation_params = {}
                # Ensure we at least have a valid sampler
                try:
                    # Use the mlx_lm that was imported at the module level
                    sampler = mlx_lm.sample_utils.make_sampler(temp=float(max(0.01, temperature)))
                    generation_params["sampler"] = sampler
                except Exception as sampler_error:
                    logger.warning(f"Could not create sampler: {sampler_error}")
                    # If we can't create a sampler, we'll let mlx_lm use its defaults
            
            # Stream tokens from the model
            generate_kwargs = {
                "model": self._model,
                "tokenizer": self._processor,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
                "verbose": False,
            }
            
            # Add model-specific generation parameters
            generate_kwargs.update(generation_params)
            
            start_time = time.time()
            
            # Stream tokens from the model
            for token in mlx_lm.generate(**generate_kwargs):
                current_text += token
                
                # Calculate a rough estimate of token count
                try:
                    prompt_tokens = len(self._processor.encode(prompt))
                    completion_tokens = len(self._processor.encode(current_text))
                except:
                    # Fallback to rough estimates
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(current_text.split())
                
                yield GenerateResponse(
                    content=current_text,
                    model=model_name,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "time": time.time() - start_time
                    }
                )
                
        except Exception as e:
            logger.error(f"Text streaming failed: {e}")
            raise GenerationError(f"Text streaming failed: {str(e)}")

    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM provider."""
        capabilities = {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: self.config_manager.get_param(ModelParameter.MAX_TOKENS, 4096),
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.TOOL_USE: False,
            ModelCapability.VISION: self._is_vision_model and MLXVLM_AVAILABLE,
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