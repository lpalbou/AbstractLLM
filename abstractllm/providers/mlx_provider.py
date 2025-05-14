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
        load as load_vlm,
        load_config,
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

# Set up logger
logger = logging.getLogger(__name__)

# Load vision model information from JSON
def _load_vision_model_data() -> Dict[str, Any]:
    """Load vision model information from the JSON file."""
    try:
        json_path = os.path.join(os.path.dirname(__file__), "mlx_vision_models.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Vision model data file not found at {json_path}")
            return {"model_families": {}, "detection_patterns": {"model_name_indicators": [], "family_patterns": {}}}
    except Exception as e:
        logger.error(f"Error loading vision model data: {e}")
        return {"model_families": {}, "detection_patterns": {"model_name_indicators": [], "family_patterns": {}}}

# Load the vision model data
VISION_MODEL_DATA = _load_vision_model_data()

# Vision model indicators from the JSON file
VISION_MODEL_INDICATORS = VISION_MODEL_DATA["detection_patterns"]["model_name_indicators"]

# Model family patterns for detection
MODEL_FAMILY_PATTERNS = VISION_MODEL_DATA["detection_patterns"]["family_patterns"]

# Model configurations from the JSON file
MODEL_CONFIGS = {}
for family_id, family_data in VISION_MODEL_DATA["model_families"].items():
    MODEL_CONFIGS[family_id] = {
        "image_size": tuple(family_data["image_size"]),
        "mean": [0.485, 0.456, 0.406],  # Default normalization values
        "std": [0.229, 0.224, 0.225],   # Default normalization values
        "prompt_format": family_data["prompt_format"],
        "patterns": MODEL_FAMILY_PATTERNS.get(family_id, [family_id])
    }

# Add default configuration
if "default" not in MODEL_CONFIGS:
    MODEL_CONFIGS["default"] = {
        "image_size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "prompt_format": "<image>{prompt}",
        "patterns": []
    }

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
            
        if not MLXVLM_AVAILABLE:
            logger.error("MLX-VLM package not found")
            raise ImportError("MLX-VLM is required for MLXProvider. Install with: pip install mlx-vlm")
        
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
        self._is_vision_model = False
        self._model_type = None
        
        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized MLX provider with model: {model}")
        
        # Check if the model name indicates vision capabilities
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        logger.debug(f"Checking if model name indicates vision capabilities: {model_name}")
        if self._check_vision_capability(model_name):
            logger.debug(f"Model name indicates vision capabilities: {model_name}")
            self._is_vision_model = True
            self._model_type = self._determine_model_type(model_name)
            logger.info(f"Detected vision model type: {self._model_type}")

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
        # Clear the in-memory cache
        if model_name:
            # Remove specific model from cache
            if model_name in MLXProvider._model_cache:
                del MLXProvider._model_cache[model_name]
                logger.info(f"Removed {model_name} from in-memory cache")
            else:
                logger.info(f"Model {model_name} not found in in-memory cache")
        else:
            # Clear all in-memory cache
            model_count = len(MLXProvider._model_cache)
            MLXProvider._model_cache.clear()
            logger.info(f"Cleared {model_count} models from in-memory cache")
        
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
        
    def _check_vision_capability(self, model_name: str) -> bool:
        """Check if the model name indicates vision capabilities."""
        # First check if the model is in our known models list
        model_name_lower = model_name.lower()
        
        # Check if it's in any of the model families
        for family_id, family_data in VISION_MODEL_DATA["model_families"].items():
            if any(model.lower() == model_name_lower for model in family_data["models"]):
                logger.debug(f"Model {model_name} found in {family_id} family")
                return True
        
        # Then check for specific vision indicators in the name
        # Use a more restrictive set of indicators to avoid false positives
        vision_specific_indicators = [
            "-vl", "vision-", "-vision", "llava", "blip", "clip", 
            "multimodal", "image-to-text", "img2text"
        ]
        
        if any(indicator in model_name_lower for indicator in vision_specific_indicators):
            logger.debug(f"Vision indicator found in model name: {model_name}")
            return True
        
        # For models not in our list or without clear indicators, try to check the model config
        try:
            config = load_config(model_name, trust_remote_code=True)
            if config:
                # Check for vision-related fields in the config
                vision_config_indicators = [
                    "vision_config", "image_size", "vision_tower", 
                    "is_vision_model", "modality", "image_processor"
                ]
                
                for indicator in vision_config_indicators:
                    if indicator in config:
                        logger.debug(f"Vision indicator '{indicator}' found in model config for {model_name}")
                        return True
                        
                # Check model type for vision-related types
                if "model_type" in config:
                    vision_model_types = ["llava", "blip", "git", "paligemma", "idefics", "flamingo", "qwen_vl", "qwen2_vl"]
                    if any(vmt in config["model_type"].lower() for vmt in vision_model_types):
                        logger.debug(f"Vision model type found in config: {config['model_type']}")
                        return True
        except Exception as e:
            # Log but don't fail - just use name-based detection
            logger.debug(f"Could not load model config for {model_name}: {e}")
        
        # If we reach here, it's not a vision model
        return False

    def _determine_model_type(self, model_name: str) -> str:
        """Determine the specific vision model type using a robust approach."""
        model_name_lower = model_name.lower()
        
        # Special case for Qwen2.5-VL
        if "qwen2.5-vl" in model_name_lower or "qwen2-5-vl" in model_name_lower:
            logger.debug(f"Detected Qwen2.5-VL model: {model_name}")
            return "qwen2.5-vl"
        
        # First check if the model is in our known models list
        for family_id, family_data in VISION_MODEL_DATA["model_families"].items():
            if any(model.lower() == model_name_lower for model in family_data["models"]):
                logger.debug(f"Model {model_name} found in {family_id} family")
                return family_id
        
        # Try to load model config from HF if available
        try:
            config = load_config(model_name, trust_remote_code=True)
            if "model_type" in config:
                model_type = config["model_type"].lower()
                
                # Map common model types to our family IDs
                type_to_family = {
                    "llava": "llava",
                    "qwen2_vl": "qwen-vl",
                    "qwen2_5_vl": "qwen2.5-vl",
                    "qwen_vl": "qwen-vl",
                    "blip": "blip",
                    "git": "git",
                    "idefics": "idefics",
                    "paligemma": "paligemma",
                    "flamingo": "flamingo"
                }
                
                if model_type in type_to_family:
                    logger.debug(f"Found model type {model_type} in HF config, mapping to {type_to_family[model_type]}")
                    return type_to_family[model_type]
                    
                # For other model types, check if they match any of our family patterns
                for family_id, patterns in MODEL_FAMILY_PATTERNS.items():
                    if any(pattern in model_type for pattern in patterns):
                        logger.debug(f"Matched model type {model_type} to family {family_id} using pattern")
                        return family_id
        except Exception as e:
            logger.warning(f"Could not load model config from HF: {e}")
        
        # Next, try to match using the patterns in our MODEL_CONFIGS
        for family_id, patterns in MODEL_FAMILY_PATTERNS.items():
            if any(pattern in model_name_lower for pattern in patterns):
                logger.debug(f"Matched model {model_name} to family {family_id} using pattern")
                return family_id
        
        # If no match found, try to extract from model name components
        name_components = model_name_lower.split("/")[-1].replace("-", "_").split("_")
        
        # Check for common vision model identifiers in the name components
        vision_model_identifiers = {
            "llava": "llava",
            "qwen": "qwen-vl",
            "gemma": "gemma",
            "blip": "blip",
            "git": "git",
            "idefics": "idefics",
            "paligemma": "paligemma",
            "vision": "default",
            "vl": "default",
            "visual": "default"
        }
        
        for component in name_components:
            if component in vision_model_identifiers:
                family_id = vision_model_identifiers[component]
                logger.debug(f"Matched model {model_name} to family {family_id} using name component {component}")
                return family_id
        
        # Last resort: use default configuration
        logger.warning(f"Could not determine model type for {model_name}, using default")
        return "default"

    def _get_model_config(self) -> Dict[str, Any]:
        """Get the configuration for the current model type."""
        if self._model_type in MODEL_CONFIGS:
            return MODEL_CONFIGS[self._model_type]
        return MODEL_CONFIGS["default"]

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

    def _resize_with_aspect_ratio(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image maintaining aspect ratio and padding if necessary."""
        target_w, target_h = target_size
        orig_w, orig_h = image.size
        
        # Calculate aspect ratios
        target_aspect = target_w / target_h
        orig_aspect = orig_w / orig_h
        
        if orig_aspect > target_aspect:
            # Image is wider than target
            new_w = target_w
            new_h = int(target_w / orig_aspect)
        else:
            # Image is taller than target
            new_h = target_h
            new_w = int(target_h * orig_aspect)
        
        # Resize maintaining aspect ratio
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image

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
            MAX_DIMENSION = 512  # Maximum dimension for initial downsample
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
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize with proper aspect ratio handling
            image = self._resize_with_aspect_ratio(image, target_size)
            
            try:
                # Convert to numpy array with explicit dtype
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                # Normalize using model-specific values (with broadcasting)
                mean = np.array(config["mean"], dtype=np.float32).reshape(1, 1, 3)
                std = np.array(config["std"], dtype=np.float32).reshape(1, 1, 3)
                image_array = (image_array - mean) / std
                
                # Convert to CHW format
                image_array = np.transpose(image_array, (2, 0, 1))
                
                # Convert to MLX array with explicit device placement
                return mx.array(image_array)
                
            except Exception as e:
                raise ImageProcessingError(
                    f"Failed to process image array: {str(e)}", 
                    provider="mlx"
                )
                
        except Exception as e:
            if isinstance(e, (ImageProcessingError, MemoryExceededError)):
                raise
            raise ImageProcessingError(f"Image processing failed: {str(e)}", provider="mlx")

    def _format_prompt(self, prompt: str, num_images: int) -> str:
        """Format prompt with image tokens based on model configuration."""
        if num_images == 0:
            return prompt
        
        config = self._get_model_config()
        base_format = config["prompt_format"]
        
        logger.debug(f"Formatting prompt for model type {self._model_type} with format {base_format}")
        
        # Handle special cases for different models
        if self._model_type == "llava" or "llava" in self._model_type.lower():
            # LLaVA and LLaVA-Next use a single image token regardless of number of images
            return f"<image>\n{prompt}"
        elif "phi-3-vision" in self._model_type.lower() or "phi3-vision" in self._model_type.lower():
            # Phi-3-Vision format
            return f"Image: <image>\nUser: {prompt}\nAssistant:"
        elif "qwen2.5-vl" in self._model_type.lower() or "qwen2-5-vl" in self._model_type.lower():
            # Qwen2.5-VL uses a specific image token
            return f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
        elif self._model_type == "qwen-vl" or "qwen" in self._model_type.lower():
            # Qwen-VL uses a different image token format
            if num_images == 1:
                return f"<img>{prompt}"
            formatted = prompt
            for i in range(num_images):
                formatted = f"<image{i+1}>{formatted}"
            return formatted
        elif "florence" in self._model_type.lower():
            # Florence 2 format
            return f"<image>\nUser: {prompt}\nAssistant:"
        elif "kimi-vl" in self._model_type.lower():
            # Kimi-VL uses a special token
            return f"<|image|>{prompt}"
        elif "gemma-vision" in self._model_type.lower() or "gemma-vl" in self._model_type.lower():
            # Gemma Vision format
            return f"<image>\n{prompt}"
        elif "idefics" in self._model_type.lower():
            # Idefics format
            return f"<image>\nUser: {prompt}\nAssistant:"
        elif "smolvlm" in self._model_type.lower():
            # SmoLVLM format
            return f"<image>\nUser: {prompt}\nAssistant:"
        elif "deepseek-vl" in self._model_type.lower():
            # DeepSeek VL format
            return f"<img>{prompt}"
        
        # Default format (repeat token for multiple images)
        formatted = prompt
        for _ in range(num_images):
            formatted = base_format.format(prompt=formatted)
        return formatted

    def load_model(self) -> None:
        """Load the MLX model and processor."""
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        
        try:
            if self._is_vision_model:
                # Load vision model using MLX-VLM
                logger.info(f"Loading vision model {model_name}")
                self._model, self._processor = load_vlm(
                    model_name,
                    trust_remote_code=True
                )
                self._config = load_config(model_name, trust_remote_code=True)
            else:
                # Load language model using MLX-LM
                logger.info(f"Loading language model {model_name}")
                self._model, self._processor = mlx_lm.load(model_name)
                self._config = load_config(model_name)
                
            self._is_loaded = True
            logger.info(f"Successfully loaded model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load MLX model: {str(e)}")

    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate a response using the MLX model."""
        try:
            # Load model if not already loaded
            if not self._is_loaded:
                self.load_model()
            
            # Process files if provided
            images = []
            image_paths = []  # Store original image paths
            if files:
                for file_path in files:
                    try:
                        media_input = MediaFactory.from_source(file_path)
                        if media_input.media_type == "image":
                            if not self._is_vision_model:
                                raise UnsupportedFeatureError(
                                    "vision",
                                    "This model does not support vision inputs",
                                    provider="mlx"
                                )
                            images.append(self._process_image(media_input))
                            image_paths.append(str(file_path))  # Store the original path
                        elif media_input.media_type == "text":
                            # Append text content to prompt
                            prompt += f"\n\nFile content from {file_path}:\n{media_input.get_content()}"
                    except Exception as e:
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
            
            # Prepare generation kwargs
            gen_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }
            
            # Handle vision model generation
            if self._is_vision_model and images:
                try:
                    # Format prompt with image tokens
                    formatted_prompt = self._format_prompt(prompt, len(images))
                    if system_prompt:
                        formatted_prompt = f"{system_prompt}\n\n{formatted_prompt}"
                    
                    # Generate response
                    if stream:
                        return self._generate_vision_stream(formatted_prompt, images, image_paths, **gen_kwargs)
                    else:
                        return self._generate_vision(formatted_prompt, images, image_paths, **gen_kwargs)
                except Exception as e:
                    logger.error(f"Vision generation failed: {e}")
                    raise GenerationError(f"Vision generation failed: {str(e)}")
                    
            # Handle text-only generation
            formatted_prompt = prompt
            if system_prompt:
                formatted_prompt = self._format_system_prompt(system_prompt, prompt)
                
            # Generate response
            if stream:
                return self._generate_text_stream(formatted_prompt, **gen_kwargs)
            else:
                return self._generate_text(formatted_prompt, **gen_kwargs)
            
        except Exception as e:
            if isinstance(e, (UnsupportedFeatureError, ImageProcessingError, FileProcessingError, 
                            MemoryExceededError, GenerationError)):
                raise
            logger.error(f"Generation failed: {e}")
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
            
            # Use the generate function from mlx_vlm.utils with the original image path
            # This allows the function to handle the image loading and processing
            output = generate(
                self._model,
                self._processor,
                prompt=prompt,
                image=image_path,  # Use the original image path
                **kwargs
            )
            
            # Extract the text content from the output
            text_content = output[0] if isinstance(output, tuple) else output
            
            # Get the tokenizer for token counting
            tokenizer = self._processor.tokenizer if hasattr(self._processor, "tokenizer") else self._processor
            
            return GenerateResponse(
                content=text_content,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                usage={
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "completion_tokens": len(tokenizer.encode(text_content)),
                    "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(text_content))
                },
                image_paths=image_paths
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
            
            # Use the stream_generate function from mlx_vlm.utils with the original image path
            # This allows the function to handle the image loading and processing
            for response in stream_generate(
                self._model,
                self._processor,
                prompt=prompt,
                image=image_path,  # Use the original image path instead of passing the processed image
                **kwargs
            ):
                yield GenerateResponse(
                    content=response.text,
                    model=self.config_manager.get_param(ModelParameter.MODEL),
                    usage={
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.generation_tokens,
                        "total_tokens": response.prompt_tokens + response.generation_tokens
                    },
                    image_paths=image_paths
                )
        except Exception as e:
            logger.error(f"Vision generation streaming failed: {e}")
            raise RuntimeError(f"Vision generation streaming failed: {str(e)}")

    def _generate_text(self, prompt: str, **kwargs) -> GenerateResponse:
        """Generate text using MLX-LM."""
        try:
            output = mlx_lm.generate(
                self._model,
                self._processor,
                prompt=prompt,
                **kwargs
            )

            return GenerateResponse(
                content=output,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                usage={
                    "prompt_tokens": len(self._processor.encode(prompt)),
                    "completion_tokens": len(self._processor.encode(output)),
                    "total_tokens": len(self._processor.encode(prompt)) + len(self._processor.encode(output))
                }
            )
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {str(e)}")

    def _generate_text_stream(self, prompt: str, **kwargs) -> Generator[GenerateResponse, None, None]:
        """Stream text using MLX-LM."""
        try:
            start_time = time.time()
            for chunk in mlx_lm.generate(
                self._model,
                self._processor,
                prompt=prompt,
                stream=True,
                **kwargs
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