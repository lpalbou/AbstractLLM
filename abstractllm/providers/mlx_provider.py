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
import copy
import io
import numpy as np
import re
import json
from datetime import datetime

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
from abstractllm.providers.mlx_model_configs import ModelConfigFactory, MLXModelConfig
from abstractllm.media.factory import MediaFactory
from abstractllm.tools import ToolCall, ToolCallResponse
from abstractllm.utils.utilities import TokenCounter, is_apple_silicon
from abstractllm.utils.logging import log_request, log_response
from abstractllm.architectures.detection import detect_architecture
# Architecture capabilities and templates removed during refactoring

# PIL is required for image processing
from PIL import Image

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
    from mlx_lm import load as load_model, generate as generate_text, stream_generate
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
    
except (ImportError, OSError) as e:
    # Handle both import errors and dynamic library loading errors
    MLX_AVAILABLE = False
    MLXLM_AVAILABLE = False
    logging.warning(f"MLX libraries not available: {e}")
    logging.warning("To fix MLX issues: pip uninstall mlx mlx-lm && pip install mlx mlx-lm")
    
# If module is loaded directly without dependencies, prevent provider initialization
if not MLX_AVAILABLE or not MLXLM_AVAILABLE:
    logging.warning("MLX provider disabled due to missing dependencies.")

class MLXProvider(BaseProvider):
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
        
        if not is_apple_silicon():
            logger.error("MLX requires Apple Silicon hardware")
            raise EnvironmentError("MLX requires Apple Silicon hardware (M1/M2/M3/M4)")
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
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
        
        # Automatically load the model during initialization
        self.load_model()
        
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
            "blip", "vqa", "image", "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen2-5-vl", "idefics", "phi-vision",
            "phi3-vision", "llava", "fuyu", "clip", "florence", "paligemma", "gemini-vision",
            "pixtral", "molmo"
        ]
        
        # Check for vision indicators in the model name
        has_vision_indicator = any(indicator in model_name_lower for indicator in vision_indicators)
        
        # Also directly check for known vision models
        known_vision_models = [
            "mlx-community/paligemma-3b-mix-448-8bit",
            "mlx-community/Qwen2-VL-2B-Instruct-4bit",
            "mlx-community/Qwen2-VL-7B-Instruct-4bit",
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
            "mlx-community/Qwen2.5-VL-72B-Instruct-4bit",
            "mlx-community/qwen-vl-chat",
            "Qwen/Qwen-VL-Chat",
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct"
        ]
        
        # Check if model name is in known vision models list
        is_known_vision_model = any(known_model in model_name for known_model in known_vision_models)
        
        # Additionally check if the model name directly contains Qwen2.5-VL
        contains_qwen25_vl = "qwen2.5-vl" in model_name_lower or "qwen2-5-vl" in model_name_lower
        
        # Return True if any of the checks pass
        is_vision_model = has_vision_indicator or is_known_vision_model or contains_qwen25_vl
        
        if is_vision_model:
            logger.info(f"Model {model_name} identified as a vision model")
        else:
            logger.info(f"Model {model_name} does not appear to be a vision model")
            
        return is_vision_model

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
                    self._config = ModelConfigFactory.create_config(model_name)
                except Exception as e:
                    # Fallback for older versions without load_vlm or using renamed function
                    logger.warning(f"Error using standard load_vlm: {e}. Trying alternate loading method...")
                    try:
                        # Import function from mlx_vlm directly
                        from mlx_vlm import load
                        self._model, self._processor = load(model_name, trust_remote_code=True)
                        logger.info("Successfully loaded model with alternate method")
                        self._config = ModelConfigFactory.create_config(model_name)
                    except ImportError:
                        logger.error("Could not import mlx_vlm. Please install it with 'pip install mlx-vlm'")
                        raise
                    except Exception as e2:
                        logger.error(f"Failed to load vision model with alternate method: {e2}")
                        raise
            else:
                # For text-only models, use MLX-LM
                logger.info(f"Loading text model: {model_name}")
                
                # Load model with simplest possible call
                # MLX-LM load function only requires the model path
                self._model, self._processor = load_model(model_name)
                
                # Ensure proper EOS token configuration for different model architectures
                self._configure_eos_tokens(model_name)
                
                # Get model config from factory
                self._config = ModelConfigFactory.create_config(model_name)
            
            # Set flags to indicate model is loaded
            self._is_loaded = True
            self._model_id = model_name
            logger.info(f"Successfully loaded model: {model_name}")
            
            # Check if model supports chat completion
            self._supports_chat = self._check_chat_support()
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadingError(f"Failed to load model {model_name}: {str(e)}")
            
    def _configure_eos_tokens(self, model_name: str) -> None:
        """
        Configure EOS tokens properly for different model architectures.
        This helps ensure models know when to stop generating.
        """
        if not self._processor:
            return
            
        try:
            model_lower = model_name.lower()
            
            # Configure EOS tokens based on model architecture
            if "gemma" in model_lower:
                # Gemma models use specific EOS tokens
                if hasattr(self._processor, 'eos_token') and self._processor.eos_token is None:
                    # Try to set the EOS token if it's not already set
                    possible_eos_tokens = ["<end_of_turn>", "<eos>", "</s>"]
                    for eos_token in possible_eos_tokens:
                        if eos_token in getattr(self._processor, 'vocab', {}):
                            self._processor.eos_token = eos_token
                            logger.info(f"Set EOS token for Gemma model: {eos_token}")
                            break
            
            elif "llama" in model_lower:
                # Llama models typically use </s>
                if hasattr(self._processor, 'eos_token') and self._processor.eos_token is None:
                    if "</s>" in getattr(self._processor, 'vocab', {}):
                        self._processor.eos_token = "</s>"
                        logger.info("Set EOS token for Llama model: </s>")
            
            elif "qwen" in model_lower:
                # Qwen models use specific tokens
                if hasattr(self._processor, 'eos_token') and self._processor.eos_token is None:
                    possible_eos_tokens = ["<|im_end|>", "<|endoftext|>", "</s>"]
                    for eos_token in possible_eos_tokens:
                        if eos_token in getattr(self._processor, 'vocab', {}):
                            self._processor.eos_token = eos_token
                            logger.info(f"Set EOS token for Qwen model: {eos_token}")
                            break
            
            # Log current EOS token configuration
            eos_token = getattr(self._processor, 'eos_token', None)
            if eos_token:
                logger.debug(f"Model {model_name} EOS token: {eos_token}")
            else:
                logger.warning(f"Model {model_name} has no EOS token configured")
                
        except Exception as e:
            logger.warning(f"Failed to configure EOS tokens for {model_name}: {e}")
            
    def _check_chat_support(self) -> bool:
        """
        Check if the model supports chat completion.
        
        Returns:
            True if the model supports chat completion, False otherwise.
        """
        if self._config is None:
            self._config = ModelConfigFactory.create_config(self.config_manager.get_param(ModelParameter.MODEL))
            
        # If model has a chat template or specific chat format, it supports chat
        chat_indicators = [
            "chat",
            "instruct",
            "conversation"
        ]
        
        model_name = self.config_manager.get_param(ModelParameter.MODEL).lower()
        
        # Check for chat indicators in model name
        for indicator in chat_indicators:
            if indicator in model_name:
                logger.debug(f"Model {model_name} supports chat based on name")
                return True
                
        # Check if processor has chat template
        has_chat_template = (
            hasattr(self._processor, "apply_chat_template") or
            hasattr(self._processor, "tokenizer") and hasattr(self._processor.tokenizer, "apply_chat_template")
        )
        
        if has_chat_template:
            logger.debug(f"Model {model_name} supports chat based on chat template")
            return True
            
        # Check if config has format_chat_prompt method
        if hasattr(self._config, "format_system_prompt"):
            logger.debug(f"Model {model_name} supports chat based on config")
            return True
            
        logger.debug(f"Model {model_name} does not appear to support chat")
        return False

    def _process_image(self, image_input: Union[str, Path, Any]) -> Image.Image:
        """
        Process an image from various input types.
        
        Args:
            image_input: Path to an image, PIL Image object, or a numpy array
            
        Returns:
            Processed PIL Image
        """
        try:
            # Image is already imported at module level
            import numpy as np
            
            # Case 1: image_input is a string path
            if isinstance(image_input, (str, Path)):
                # Handle data URLs
                if isinstance(image_input, str) and image_input.startswith("data:image"):
                    header, encoded = image_input.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    image = Image.open(io.BytesIO(image_data))
                    logger.debug(f"Loaded image from data URL, format: {image.format}, size: {image.size}")
                else:
                    # Handle direct file paths
                    try:
                        image = Image.open(image_input)
                        logger.debug(f"Loaded image from path: {image_input}, format: {image.format}, size: {image.size}")
                    except FileNotFoundError:
                        logger.error(f"Image file not found: {image_input}")
                        raise
                    except Exception as e:
                        logger.error(f"Error opening image from path {image_input}: {e}")
                        raise
                        
            # Case 2: image_input is already a PIL Image
            elif isinstance(image_input, Image.Image):
                image = image_input
                logger.debug(f"Using provided PIL Image, format: {getattr(image, 'format', 'N/A')}, size: {image.size}")
                
            # Case 3: image_input is a numpy array
            elif hasattr(image_input, "shape") and hasattr(image_input, "dtype"):
                # Convert numpy array to PIL Image
                if hasattr(image_input, "squeeze"):
                    # Handle PyTorch and TF tensors
                    try:
                        if "torch" in str(type(image_input)):
                            # Convert PyTorch tensor to numpy
                            import torch
                            if hasattr(image_input, "detach"):
                                image_input = image_input.detach().cpu().numpy()
                            else:
                                image_input = image_input.cpu().numpy()
                        image_input = image_input.squeeze()
                    except Exception as e:
                        logger.warning(f"Error squeezing tensor: {e}, continuing with original")
                
                # Create PIL image from numpy array
                if len(image_input.shape) == 3 and image_input.shape[0] in (1, 3, 4):
                    # Handle CHW format (convert to HWC)
                    image_input = np.transpose(image_input, (1, 2, 0))
                
                # Convert to 8-bit for PIL
                if image_input.dtype != np.uint8 and np.issubdtype(image_input.dtype, np.floating):
                    image_input = (image_input * 255).astype(np.uint8)
                
                image = Image.fromarray(image_input)
                logger.debug(f"Converted array to PIL Image, size: {image.size}")
                
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
            
            # Auto-resize if needed - most vision models expect specific dimensions
            if self._processor is not None and hasattr(self._processor, 'image_processor'):
                # Get target size from processor if available
                target_size = None
                
                # Check if we have a model config that specifies preferred image size
                if self._config and hasattr(self._config, 'get_image_size'):
                    try:
                        model_name = self.config_manager.get_param(ModelParameter.MODEL)
                        target_size = self._config.get_image_size(self._processor, model_name)
                        logger.debug(f"Using model-specific target size from config: {target_size}")
                    except Exception as e:
                        logger.debug(f"Could not get model-specific target size: {e}")
                
                # PaliGemma and some other models use different ways to specify size
                if target_size is None and hasattr(self._processor.image_processor, 'size'):
                    processor_size = self._processor.image_processor.size
                    # Handle dict case (e.g., {"height": 448, "width": 448})
                    if isinstance(processor_size, dict) and "height" in processor_size and "width" in processor_size:
                        target_size = (processor_size["width"], processor_size["height"])
                    # Handle int case (e.g., 448)
                    elif isinstance(processor_size, int):
                        target_size = (processor_size, processor_size)
                    logger.debug(f"Using processor-defined target size: {target_size}")
                
                # If we determined a target size, resize the image
                if target_size:
                    # Only resize if the current size is very different
                    current_w, current_h = image.size
                    target_w, target_h = target_size
                    
                    # Check if resizing is needed (if dimensions differ by more than 20%)
                    resize_needed = (abs(current_w - target_w) / target_w > 0.2 or 
                                    abs(current_h - target_h) / target_h > 0.2)
                    
                    if resize_needed:
                        logger.info(f"Resizing image from {image.size} to {target_size}")
                        # Use LANCZOS for high-quality downsampling
                        image = image.resize(target_size, Image.LANCZOS)
            
            # Ensure the image is in RGB mode (MLX models typically expect RGB)
            if image.mode != "RGB":
                image = image.convert("RGB")
                logger.debug(f"Converted image to RGB mode")
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def _generate_impl(self,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      files: Optional[List[Union[str, Path]]] = None,
                      stream: bool = False,
                      tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                      messages: Optional[List[Dict[str, Any]]] = None,
                      **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate a response using the MLX model with architecture-based tool calling support."""
        
        logger.info(f"Generation request: model={self.config_manager.get_param(ModelParameter.MODEL)}, "
                   f"stream={stream}, has_system_prompt={system_prompt is not None}, files={len(files) if files else 0}")
        
        try:
            # Check for vision model usage
            if files and self._is_vision_model:
                if tools:
                    raise GenerationError("Vision models do not currently support tool calling")
                
                # Process image files
                image_paths = []
                for file in files:
                    if isinstance(file, (str, Path)):
                        image_paths.append(str(file))
                    else:
                        # Assume it's already an image
                        temp_path = f"/tmp/temp_image_{int(time.time())}.jpg"
                        if hasattr(file, 'save'):
                            file.save(temp_path)
                            image_paths.append(temp_path)
                        else:
                            raise ValueError(f"Unsupported file type: {type(file)}")
                
                # Use vision generation
                if stream:
                    return self._generate_vision_stream(prompt, image_paths, **kwargs)
                else:
                    return self._generate_vision(prompt, image_paths, **kwargs)
            
            # Use standard text generation with proper tool calling
            if stream:
                return self._generate_text_stream(prompt, system_prompt=system_prompt, tools=tools, messages=messages, **kwargs)
            else:
                return self._generate_text(prompt, system_prompt=system_prompt, tools=tools, messages=messages, **kwargs)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

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
        
        # Get parameters using the new helper method
        params = self._get_generation_params(**kwargs)
        max_tokens = params["max_tokens"]
        temperature = params["temperature"]
        
        images = []
        # Process images
        for image_path in image_paths:
            if isinstance(image_path, str):
                logger.debug(f"Loading image from path: {image_path}")
                image = self._process_image(image_path)
            else:
                logger.debug("Processing image-like object")
                image = self._process_image(image_path)
            
            if image is not None:
                images.append(image)
        
        if not images:
            return GenerateResponse(text="Error: No valid images found")
        
        processor_kwargs = {}
        stream = kwargs.get("stream", False)
        
        try:
            logger.info("Generating vision response with MLX-VLM")
            
            # Determine if we should use conversation format
            use_conversation = "qwen" in self._model_id.lower() and "<|im_start|>" in prompt
            
            if use_conversation:
                processor_kwargs["add_generation_prompt"] = True
                # For Qwen models that use conversation format, we need to use chat template
                try:
                    # For Qwen models, try to use their chat template
                    prompt = self._processor.apply_chat_template(
                        [{"role": "user", "content": prompt}], 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply chat template: {e}")
            
            # Set up generation parameters - generate_vlm expects specific parameter names
            generate_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "verbose": False
            }
            
            # Add any other valid generation parameters from kwargs
            # Remove parameters that generate_vlm doesn't expect
            for key, value in kwargs.items():
                if key not in ["max_tokens", "temperature", "stream", "images"]:
                    generate_kwargs[key] = value
            
            # Log vision request using shared method
            self._log_request_details(
                prompt=prompt,
                stream=False,
                num_images=len(images),
                image_sizes=[f"{img.size[0]}x{img.size[1]}" for img in images],
                use_conversation=use_conversation,
                **generate_kwargs
            )
            logger.info(f"MLX vision generation starting - prompt: '{prompt}', images: {len(images)}")
            
            # Try multiple approaches to handle potential broadcasting errors
            try:
                # First approach - standard generation
                # generate_vlm expects: model, processor, prompt, image (single), **kwargs
                # Returns: (text, usage_stats)
                result = generate_vlm(
                    self._model,
                    self._processor,
                    prompt,
                    image=images[0] if images else None,  # Use first image only
                    **generate_kwargs
                )
                
                # Handle return value - generate_vlm returns (text, usage_stats)
                if isinstance(result, tuple) and len(result) == 2:
                    output, usage_stats = result
                else:
                    # Fallback for older versions that might return just text
                    output = result
                    usage_stats = {}
                    
            except RuntimeError as e:
                if "broadcast_shapes" in str(e):
                    logger.warning(f"Broadcast error in standard generation: {e}. Trying alternative approach...")
                    
                    # Try with single image processing to avoid broadcasting issues
                    try:
                        # Special handling for Qwen models that have broadcasting issues
                        if len(images) == 1:
                            # Try with different tensor format
                            try:
                                # Use generate_vlm with corrected parameters
                                result = generate_vlm(
                                    self._model,
                                    self._processor,
                                    prompt,
                                    image=images[0],
                                    **generate_kwargs
                                )
                                
                                # Handle return value
                                if isinstance(result, tuple) and len(result) == 2:
                                    output, usage_stats = result
                                else:
                                    output = result
                                    usage_stats = {}
                                    
                            except Exception as inner_e:
                                logger.warning(f"Alternative approach failed: {inner_e}. Trying one more approach...")
                                
                                # Try another approach with format conversion
                                try:
                                    # Create a temporary processor with image processor settings that work
                                    temp_processor = copy.deepcopy(self._processor)
                                    if hasattr(temp_processor, "image_processor"):
                                        # Adjust image processor settings
                                        temp_processor.image_processor.size = {"height": 336, "width": 336}
                                    
                                    # Use the temporary processor
                                    result = generate_vlm(
                                        self._model,
                                        temp_processor,
                                        prompt,
                                        image=images[0],
                                        **generate_kwargs
                                    )
                                    
                                    # Handle return value
                                    if isinstance(result, tuple) and len(result) == 2:
                                        output, usage_stats = result
                                    else:
                                        output = result
                                        usage_stats = {}
                                        
                                except Exception as e3:
                                    logger.error(f"All approaches failed for vision generation: {e3}")
                                    return GenerateResponse(text=f"Error generating vision response: {e3}")
                        else:
                            # For multiple images, try a different approach or return an error
                            return GenerateResponse(text=f"Error: Multiple image processing with this model is not supported yet: {e}")
                    except Exception as inner_e:
                        logger.error(f"Alternative approach also failed: {inner_e}")
                        return GenerateResponse(text=f"Error generating vision response: {inner_e}")
                else:
                    # Re-raise if it's not a broadcasting error
                    logger.error(f"Error generating vision response: {e}")
                    return GenerateResponse(text=f"Error generating vision response: {e}")
                        
            # Log vision response using shared method
            self._log_response_details(output, output)
            logger.info(f"MLX vision generation completed - response length: {len(output)} chars")
            
            # Return the output as a GenerateResponse with proper usage stats
            return GenerateResponse(
                content=output,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                usage={
                    "prompt_tokens": usage_stats.get("input_tokens", 0),
                    "completion_tokens": usage_stats.get("output_tokens", 0),
                    "total_tokens": usage_stats.get("total_tokens", 0),
                    "prompt_tps": usage_stats.get("prompt_tps", 0.0),
                    "generation_tps": usage_stats.get("generation_tps", 0.0),
                    "peak_memory": usage_stats.get("peak_memory", 0.0)
                }
            )
        
        except Exception as e:
            logger.error(f"Error generating vision response: {e}")
            return GenerateResponse(text=f"Error generating vision response: {e}")

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
        
        # Get parameters using the new helper method
        params = self._get_generation_params(**kwargs)
        max_tokens = params["max_tokens"]
        temperature = params["temperature"]
        top_p = params["top_p"]
        
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
                self._model,
                self._processor,
                prompt,
                image=image,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Handle return value - generate_vlm returns (text, usage_stats)
            if isinstance(result, tuple) and len(result) == 2:
                text, usage_stats = result
            else:
                # Fallback for older versions that might return just text
                text = result
                usage_stats = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "prompt_tps": 0.0,
                    "generation_tps": 0.0,
                    "peak_memory": 0.0
                }
            
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
                    model=self.config_manager.get_param(ModelParameter.MODEL),
                    usage={
                        "prompt_tokens": usage_stats.get("input_tokens", 0),
                        "completion_tokens": int(tokens_generated * usage_stats.get("output_tokens", 0) / len(full_text.split())),
                        "total_tokens": usage_stats.get("input_tokens", 0) + int(tokens_generated * usage_stats.get("output_tokens", 0) / len(full_text.split())),
                        "prompt_tps": usage_stats.get("prompt_tps", 0.0),
                        "generation_tps": usage_stats.get("generation_tps", 0.0),
                        "peak_memory": usage_stats.get("peak_memory", 0.0)
                    }
                )
                
                # Add a small delay to simulate real streaming
                time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Vision stream generation failed: {e}")
            raise GenerationError(f"Vision stream generation failed: {str(e)}")

    def _get_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get generation parameters by merging config defaults with kwargs, filtering out None values.
        
        Args:
            **kwargs: Generation parameters provided to the method
            
        Returns:
            Dictionary of generation parameters with None values filtered out
        """
        # Filter out None values from kwargs to prevent them from overriding config defaults
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Get parameters from config with filtered kwargs override
        params = {
            "max_tokens": filtered_kwargs.get("max_tokens", self.config_manager.get_param(ModelParameter.MAX_TOKENS)),
            "temperature": filtered_kwargs.get("temperature", self.config_manager.get_param(ModelParameter.TEMPERATURE)),
            "top_p": filtered_kwargs.get("top_p", self.config_manager.get_param(ModelParameter.TOP_P, 0.95))
        }
        
        # Add SOTA parameters if provided
        if "repetition_penalty" in filtered_kwargs:
            params["repetition_penalty"] = filtered_kwargs["repetition_penalty"]
        
        # Add seed for reproducible generation
        seed = self.config_manager.get_param(ModelParameter.SEED)
        if seed is not None:
            params["seed"] = seed
        
        # Add other local model parameters
        top_k = self.config_manager.get_param(ModelParameter.TOP_K)
        if top_k is not None:
            params["top_k"] = top_k
        
        return params

    def _get_tool_handler(self) -> Optional["UniversalToolHandler"]:
        """
        Get or create the tool handler for this provider.
        
        This method overrides the base class to ensure MLX models always have
        tool support via prompting, since MLX can run any model with tools
        through careful prompt engineering.
        """
        if not hasattr(self, '_tool_handler') or self._tool_handler is None:
            try:
                from abstractllm.tools.handler import UniversalToolHandler
                
                # Get the model name
                model = self.config_manager.get_param(ModelParameter.MODEL)
                if model:
                    self._tool_handler = UniversalToolHandler(model)
                    
                    # For MLX, ensure we always have at least prompted tool support
                    # MLX can run any instruct model with tools via prompting
                    if not self._tool_handler.supports_prompted and not self._tool_handler.supports_native:
                        # Force prompted mode for instruct models
                        from abstractllm.architectures.detection import detect_model_type
                        if detect_model_type(model) == "instruct":
                            self._tool_handler.supports_prompted = True
                            logger.info(f"Enabled prompted tool support for MLX model {model}")
                    
                    logger.debug(f"Created tool handler for MLX model {model}: "
                               f"native={self._tool_handler.supports_native}, "
                               f"prompted={self._tool_handler.supports_prompted}")
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

    def _ensure_chat_template_compatibility(self, messages: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """
        Ensure chat template compatibility by fixing role alternation issues.
        
        This method addresses the "Conversation roles must alternate user/assistant" error
        that occurs with Gemma and Llama models when tools or multiple system messages are used.
        
        Args:
            messages: List of message dictionaries
            model_name: Name of the model to check compatibility for
            
        Returns:
            Fixed list of messages that satisfy chat template requirements
        """
        if not messages:
            return messages
            
        # Use architecture detection to determine if strict alternation is needed
        architecture = detect_architecture(model_name)
        
        # Check if this architecture requires strict role alternation
        requires_strict_alternation = False
        # TODO: Add proper capability checking when architecture capabilities are restored
        
        # Fallback check for known problematic models
        if not requires_strict_alternation:
            strict_alternation_models = ["gemma", "llama"]
            requires_strict_alternation = any(model.lower() in model_name.lower() for model in strict_alternation_models)
        
        if not requires_strict_alternation:
            return messages
            
        logger.debug(f"Applying chat template compatibility fixes for {architecture} architecture")
        
        # Use the existing logic but simplified
        fixed_messages = []
        system_contents = []
        
        # Step 1: Collect all system messages and merge them
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
                # Convert tool response to assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": f"I called the {msg.get('name', 'unknown_tool')} tool and received: {msg.get('content', '')}"
                }
                final_messages.append(assistant_msg)
            else:
                final_messages.append(msg)
        
        # Step 3: Ensure proper user/assistant alternation
        if len(final_messages) > 1:
            validated_messages = []
            last_role = None
            
            for i, msg in enumerate(final_messages):
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

    def _generate_text(self, prompt: str, system_prompt: Optional[str] = None, tools: Optional[List[Any]] = None, messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> GenerateResponse:
        """Generate text using MLX with architecture-based tool calling support."""
        
        # Make sure model is loaded
        if not self._is_loaded:
            logger.info("Model not loaded, loading now")
            self.load_model()
        
        # Get parameters using the new helper method
        params = self._get_generation_params(**kwargs)
        max_tokens = params["max_tokens"]
        temperature = params["temperature"]
        top_p = params["top_p"]
        
        try:
            start_time = time.time()
            
            # Get model-specific generation parameters
            if self._model_config is None:
                self._model_config = ModelConfigFactory.create_config(
                    self.config_manager.get_param(ModelParameter.MODEL)
                )
            
            # Create a copy of kwargs without temperature to avoid conflict
            config_kwargs = kwargs.copy()
            config_kwargs.pop("temperature", None)  # Remove temperature to avoid duplicate argument
            generation_params = self._model_config.get_generation_params(temperature, **config_kwargs)
            
            # Get model name for architecture detection
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            
            # Prepare tool context BEFORE constructing messages
            enhanced_system_prompt = system_prompt
            mlx_tools = None
            if tools:
                # Use base class method for tool preparation
                enhanced_system_prompt, tool_defs, mode = self._prepare_tool_context(tools, system_prompt)
                logger.info(f"Prepared {len(tools)} tools in {mode} mode")
            
            # NOW prepare messages for MLX chat template with enhanced system prompt
            if messages is not None:
                # Use provided messages (includes conversation history and tool results)
                chat_messages = messages.copy()
            else:
                # Construct messages from prompt and enhanced_system_prompt (with tools)
                chat_messages = []
                if enhanced_system_prompt:  # Use enhanced prompt that includes tool instructions
                    chat_messages.append({"role": "system", "content": enhanced_system_prompt})
                    # Debug logging to see what system prompt is being used
                    if tools:
                        logger.warning(f"System prompt with tools (first 500 chars): {enhanced_system_prompt[:500]}")
                chat_messages.append({"role": "user", "content": prompt})
            
            if tools:
                
                # MLX uses prompted mode, so mlx_tools is mainly for tracking
                mlx_tools = tool_defs if mode == "native" else tools
                
                # Update messages with enhanced system prompt
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
                            
                            # Use clearer formatting for smaller models to understand tool execution
                            formatted_tool_message = (
                                f"I have successfully executed the {tool_name} tool. "
                                f"Here is the result:\n\n{tool_output}\n\n"
                                f"Based on this information, I can now provide an answer."
                            )
                            
                            formatted_messages.append({
                                "role": "assistant", 
                                "content": formatted_tool_message
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
            
            # Try to apply chat template with fallback mechanism
            try:
                formatted_prompt = self._processor.apply_chat_template(
                    compatible_messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except Exception as template_error:
                logger.warning(f"Chat template failed: {template_error}")
                
                # Use architecture-based fallback template handling
                logger.info(f"Using architecture-based fallback template")
                # Simple fallback template
                formatted_prompt = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in compatible_messages
                ])
                if compatible_messages and compatible_messages[-1]['role'] != 'assistant':
                    formatted_prompt += "\nassistant:"
            
            # Log the request using shared method with all details
            generation_params_log = {}
            for key, value in generation_params.items():
                if key == "sampler":
                    # Extract sampler parameters if available
                    generation_params_log["sampler_type"] = value.__name__ if hasattr(value, '__name__') else str(value)
                    generation_params_log["sampler_temperature"] = temperature
                elif key == "logits_processors":
                    # Extract logits processor info
                    if isinstance(value, list):
                        processor_info = []
                        for processor in value:
                            if hasattr(processor, '__name__'):
                                processor_info.append(processor.__name__)
                            else:
                                processor_info.append(str(type(processor).__name__))
                        generation_params_log["logits_processors"] = processor_info
                        generation_params_log["repetition_penalty"] = self._model_config.default_repetition_penalty if self._model_config else 1.0
                else:
                    generation_params_log[key] = value
            
            # Use shared logging method
            self._log_request_details(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                formatted_messages=formatted_messages,
                enhanced_system_prompt=enhanced_system_prompt if tools else system_prompt,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                formatted_prompt=formatted_prompt,
                **generation_params_log
            )
            
            logger.info(f"MLX generation starting - prompt length: {len(formatted_prompt)} chars")
            
            # ===== EXACT PROMPT SENT TO MLX MODEL =====
#            print("\n" + "="*80)
#            print("🔥 EXACT PROMPT SENT TO MLX MODEL:")
#            print("="*80)
#            print(formatted_prompt)
#            print("="*80)
#            print(f"📊 Length: {len(formatted_prompt)} characters")
#            print("="*80 + "\n")
            
            # Set seed for deterministic generation (similar to ForgeLLM approach)
            seed = self.config_manager.get_param(ModelParameter.SEED)
            if seed is not None:
                mx.random.seed(seed)
                logger.info(f"Set MLX random seed to {seed} for deterministic generation")

            # Generate with MLX
            generate_kwargs = {
                "model": self._model,
                "tokenizer": self._processor,
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "verbose": False,
                **generation_params
            }
            
            # For Gemma models, ensure proper EOS token configuration
            if "gemma" in model_name.lower():
                # Gemma models need proper repetition penalty to avoid looping
                # Note: MLX-LM handles repetition penalty through the model config, not generation parameters
                logger.info(f"Using Gemma model - repetition penalty handled by model config")
            
            # Capture verbatim context sent to LLM
            import json
            verbatim_context = json.dumps(generate_kwargs, indent=2, ensure_ascii=False, default=str)
            self._capture_verbatim_context(f"MLX LOCAL MODEL\n\nGENERATION PARAMETERS:\n{verbatim_context}")

            # Generate text
            output = generate_text(**generate_kwargs)
            
            # Log response using shared method
            self._log_response_details(output, output)
            
            logger.info(f"MLX generation completed - response length: {len(output)} chars")
            
            # Use base class tool extraction method
            if mlx_tools:
                # Log the output to help with debugging
                logger.debug(f"Checking for tool calls in output: {output[:200]}...")
                
                # Try to extract tool calls from the response
                tool_response = self._extract_tool_calls(output)
                
                # Check if tool calls were found and return appropriate response
                if tool_response and tool_response.has_tool_calls():
                    # Note: Tool call logging is handled at session level for universal coverage
                    # Return a GenerateResponse with tool calls
                    return GenerateResponse(
                        content=output,
                        tool_calls=tool_response,
                        model=model_name,
                        usage={
                            "prompt_tokens": len(self._processor.encode(formatted_prompt)),
                            "completion_tokens": len(self._processor.encode(output)),
                            "total_tokens": len(self._processor.encode(formatted_prompt)) + len(self._processor.encode(output))
                        }
                    )
                # No warning when no tools are called - this is normal behavior
            
            # Calculate token usage
            prompt_tokens = TokenCounter.count_tokens(formatted_prompt, model_name)
            completion_tokens = TokenCounter.count_tokens(output, model_name)
            
            return GenerateResponse(
                content=output,
                model=model_name,
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
        """Stream text generation using MLX with architecture-based tool calling support."""
        
        # Make sure model is loaded
        if not self._is_loaded:
            logger.info("Model not loaded, loading now")
            self.load_model()
        
        # Get parameters using the new helper method
        params = self._get_generation_params(**kwargs)
        max_tokens = params["max_tokens"]
        temperature = params["temperature"]
        top_p = params["top_p"]
        
        try:
            start_time = time.time()
            
            # Get model-specific generation parameters
            if self._model_config is None:
                self._model_config = ModelConfigFactory.create_config(
                    self.config_manager.get_param(ModelParameter.MODEL)
                )
            
            # Create a copy of kwargs without temperature to avoid conflict
            config_kwargs = kwargs.copy()
            config_kwargs.pop("temperature", None)  # Remove temperature to avoid duplicate argument
            generation_params = self._model_config.get_generation_params(temperature, **config_kwargs)
            
            # Get model name for architecture detection
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            
            # Prepare tool context BEFORE constructing messages
            enhanced_system_prompt = system_prompt
            mlx_tools = None
            if tools:
                # Use base class method for tool preparation
                enhanced_system_prompt, tool_defs, mode = self._prepare_tool_context(tools, system_prompt)
                logger.info(f"Prepared {len(tools)} tools in {mode} mode")
            
            # NOW prepare messages for MLX chat template with enhanced system prompt
            if messages is not None:
                # Use provided messages (includes conversation history and tool results)
                chat_messages = messages.copy()
            else:
                # Construct messages from prompt and enhanced_system_prompt (with tools)
                chat_messages = []
                if enhanced_system_prompt:  # Use enhanced prompt that includes tool instructions
                    chat_messages.append({"role": "system", "content": enhanced_system_prompt})
                    # Debug logging to see what system prompt is being used
                    if tools:
                        logger.warning(f"System prompt with tools (first 500 chars): {enhanced_system_prompt[:500]}")
                chat_messages.append({"role": "user", "content": prompt})
            
            if tools:
                
                # MLX uses prompted mode, so mlx_tools is mainly for tracking
                mlx_tools = tool_defs if mode == "native" else tools
                
                # Update messages with enhanced system prompt
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
                            
                            # Use clearer formatting for smaller models to understand tool execution
                            formatted_tool_message = (
                                f"I have successfully executed the {tool_name} tool. "
                                f"Here is the result:\n\n{tool_output}\n\n"
                                f"Based on this information, I can now provide an answer."
                            )
                            
                            formatted_messages.append({
                                "role": "assistant", 
                                "content": formatted_tool_message
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
            
            # Try to apply chat template with fallback mechanism
            try:
                formatted_prompt = self._processor.apply_chat_template(
                    compatible_messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except Exception as template_error:
                logger.warning(f"Chat template failed: {template_error}")
                
                # Use architecture-based fallback template handling
                logger.info(f"Using architecture-based fallback template")
                # Simple fallback template
                formatted_prompt = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in compatible_messages
                ])
                if compatible_messages and compatible_messages[-1]['role'] != 'assistant':
                    formatted_prompt += "\nassistant:"
            
            # Log the request using shared method with all details
            generation_params_log = {}
            for key, value in generation_params.items():
                if key == "sampler":
                    # Extract sampler parameters if available
                    generation_params_log["sampler_type"] = value.__name__ if hasattr(value, '__name__') else str(value)
                    generation_params_log["sampler_temperature"] = temperature
                elif key == "logits_processors":
                    # Extract logits processor info
                    if isinstance(value, list):
                        processor_info = []
                        for processor in value:
                            if hasattr(processor, '__name__'):
                                processor_info.append(processor.__name__)
                            else:
                                processor_info.append(str(type(processor).__name__))
                        generation_params_log["logits_processors"] = processor_info
                        generation_params_log["repetition_penalty"] = self._model_config.default_repetition_penalty if self._model_config else 1.0
                else:
                    generation_params_log[key] = value
            
            # Use shared logging method
            self._log_request_details(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
                formatted_messages=formatted_messages,
                enhanced_system_prompt=enhanced_system_prompt if tools else system_prompt,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                formatted_prompt=formatted_prompt,
                **generation_params_log
            )
            
            logger.info(f"MLX generation starting - prompt length: {len(formatted_prompt)} chars")
            
            # ===== EXACT PROMPT SENT TO MLX MODEL (STREAMING) =====
            print("\n" + "="*80)
            print("🔥 EXACT PROMPT SENT TO MLX MODEL (STREAMING):")
            print("="*80)
            print(formatted_prompt)
            print("="*80)
            print(f"📊 Length: {len(formatted_prompt)} characters")
            print("="*80 + "\n")

            # Set seed for deterministic generation (similar to ForgeLLM approach)
            seed = self.config_manager.get_param(ModelParameter.SEED)
            if seed is not None:
                mx.random.seed(seed)
                logger.info(f"Set MLX random seed to {seed} for deterministic generation")

            # Stream tokens from the model using stream_generate
            stream_kwargs = {
                "model": self._model,
                "tokenizer": self._processor,
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                **generation_params
            }
            
            # For Gemma models, ensure proper EOS token configuration  
            if "gemma" in model_name.lower():
                # Gemma models need proper repetition penalty to avoid looping
                # Note: MLX-LM handles repetition penalty through the model config, not generation parameters
                logger.info(f"Using Gemma model - repetition penalty handled by model config")
            
            # Capture verbatim context sent to LLM
            import json
            verbatim_context = json.dumps(stream_kwargs, indent=2, ensure_ascii=False, default=str)
            self._capture_verbatim_context(f"MLX LOCAL MODEL (streaming)\n\nGENERATION PARAMETERS:\n{verbatim_context}")

            current_text = ""

            # Stream tokens from the model using stream_generate
            for response in stream_generate(**stream_kwargs):
                current_text = response.text  # stream_generate yields response objects with .text attribute
                
                # Calculate token usage
                prompt_tokens = TokenCounter.count_tokens(formatted_prompt, model_name)
                completion_tokens = TokenCounter.count_tokens(current_text, model_name)
                
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
            logger.error(f"Text stream generation failed: {e}")
            raise GenerationError(f"Text stream generation failed: {str(e)}")

    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM provider."""
        capabilities = {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: self.config_manager.get_param(ModelParameter.MAX_TOKENS, 4096),
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: True,  # Now uses architecture-based tool calling
            ModelCapability.TOOL_USE: True,  # Now uses architecture-based tool calling
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