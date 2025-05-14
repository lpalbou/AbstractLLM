"""
MLX model factory for AbstractLLM.

This module provides a factory for loading MLX models,
abstracting away direct implementation details.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar, Generator
import numpy as np

try:
    import mlx.core as mx
    import mlx_lm
    import mlx_vlm
    from mlx_vlm.utils import load as load_vlm, load_config, generate, stream_generate
    from huggingface_hub import hf_hub_download
    MLX_AVAILABLE = True
    MLXLM_AVAILABLE = True
    MLXVLM_AVAILABLE = True
except ImportError as e:
    MLX_AVAILABLE = False
    MLXLM_AVAILABLE = False
    MLXVLM_AVAILABLE = False
    logging.warning(f"MLX not available: {e}")

from abstractllm.exceptions import ModelLoadingError, UnsupportedFeatureError

# Set up logger
logger = logging.getLogger(__name__)

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

class MLXModelFactory:
    """
    Factory for loading and managing MLX models.
    
    This class abstracts the loading of language and vision models from MLX,
    handling different model types, caching, and configuration.
    """
    
    # Class-level model cache
    _model_cache: ClassVar[Dict[str, Tuple[Any, Any, float]]] = {}
    _max_cached_models = 2  # Default to 2 models in memory
    
    @classmethod
    def is_vision_model(cls, model_name: str) -> bool:
        """
        Check if the model name indicates vision capabilities.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            True if the model has vision capabilities, False otherwise
        """
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
        
    @classmethod
    def determine_model_type(cls, model_name: str) -> str:
        """
        Determine the model type from the model name.
        
        Args:
            model_name: The name of the model
            
        Returns:
            The determined model type
        """
        model_name_lower = model_name.lower()
        
        # Check if this is an exact match for a known model in our database
        for family_id, family_data in VISION_MODEL_DATA["model_families"].items():
            if any(model.lower() == model_name_lower for model in family_data["models"]):
                logger.debug(f"Found exact match for model {model_name} in family {family_id}")
                return family_id
        
        # Otherwise try to determine from name
        name_components = model_name_lower.replace("-", " ").replace("_", " ").split()
        
        # First check for specific architectures
        if "llava" in model_name_lower:
            logger.debug(f"Detected LLaVA model: {model_name}")
            return "llava"
            
        elif "phi-3-vision" in model_name_lower or "phi3-vision" in model_name_lower or "phi-vision" in model_name_lower:
            logger.debug(f"Detected Phi-3-Vision model: {model_name}")
            return "phi-vision"
            
        elif "bakllava" in model_name_lower:
            logger.debug(f"Detected BakLLaVA model: {model_name}")
            return "llava"  # BakLLaVA uses LLaVA format
            
        elif "qwen2-vl" in model_name_lower or "qwen2.5-vl" in model_name_lower or "qwen-2-vl" in model_name_lower:
            logger.debug(f"Detected Qwen2 VL model: {model_name}")
            return "qwen2-vl"
            
        elif "qwen-vl" in model_name_lower:
            logger.debug(f"Detected Qwen VL model: {model_name}")
            return "qwen-vl"
            
        elif "idefics" in model_name_lower:
            logger.debug(f"Detected Idefics model: {model_name}")
            return "idefics"
        
        # Check for model family patterns
        for family_id, patterns in MODEL_FAMILY_PATTERNS.items():
            for pattern in patterns:
                if pattern in model_name_lower:
                    logger.debug(f"Matched model {model_name} to family {family_id} using pattern {pattern}")
                    return family_id
        
        # Check for common vision model identifiers in the name components
        vision_model_identifiers = {
            "llava": "llava",
            "qwen": "qwen-vl",
            "gemma": "gemma-it",
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
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific model type.
        
        Args:
            model_type: The type of model
            
        Returns:
            The model configuration
        """
        if model_type in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_type]
        return MODEL_CONFIGS["default"]
    
    @classmethod
    def clear_cache(cls, model_name: Optional[str] = None) -> None:
        """
        Clear model cache.
        
        Args:
            model_name: Specific model to clear (or all if None)
        """
        if model_name:
            if model_name in cls._model_cache:
                logger.info(f"Clearing cached model {model_name}")
                del cls._model_cache[model_name]
        else:
            logger.info("Clearing all cached models")
            cls._model_cache.clear()
    
    @classmethod
    def _fix_processor_patch_size(cls, processor: Any, model_name: str) -> None:
        """
        Fix processor patch_size which might be None for some models.
        
        Args:
            processor: The model processor
            model_name: The model name for logging
        """
        # The default patch size for most vision models
        DEFAULT_PATCH_SIZE = 14
        
        # Set patch_size at various locations where it might be needed
        try:
            # First check if we have an image processor
            if hasattr(processor, 'image_processor'):
                # Check if image_processor has patch_size attribute
                if not hasattr(processor.image_processor, 'patch_size') or processor.image_processor.patch_size is None:
                    logger.warning(f"Processor patch_size is None for {model_name}, setting to default")
                    processor.image_processor.patch_size = DEFAULT_PATCH_SIZE
                    logger.info(f"Set processor.image_processor.patch_size to {DEFAULT_PATCH_SIZE} for {model_name}")
                
                # Fix return_tensors to use "mlx" instead of "ml"
                if hasattr(processor.image_processor, 'return_tensors') and processor.image_processor.return_tensors == "ml":
                    processor.image_processor.return_tensors = "mlx"
                    logger.info(f"Fixed image_processor.return_tensors to use 'mlx' instead of 'ml'")
                
            # Set patch_size on processor directly if it has the attribute
            if hasattr(processor, 'patch_size'):
                if processor.patch_size is None:
                    processor.patch_size = DEFAULT_PATCH_SIZE
                    logger.info(f"Set processor.patch_size to {DEFAULT_PATCH_SIZE} for {model_name}")
                
            # Special handling for LLaVA models - set patch_size on the processor instance
            if "llava" in model_name.lower():
                # Force patch_size to be set for LLaVA processor
                processor.patch_size = DEFAULT_PATCH_SIZE
                logger.info(f"Set processor.patch_size for LLaVA model {model_name}")
                
            # Set on the processor instance for any model
            if not hasattr(processor, 'patch_size') or processor.patch_size is None:
                processor.patch_size = DEFAULT_PATCH_SIZE
                
            # Special handling for transformers LLaVAProcessor
            if hasattr(processor, 'current_processor'):
                if hasattr(processor.current_processor, 'patch_size'):
                    if processor.current_processor.patch_size is None:
                        processor.current_processor.patch_size = DEFAULT_PATCH_SIZE
                
            # Set the patch_size on the model's vision_tower if available
            if hasattr(processor, 'model') and hasattr(processor.model, 'vision_tower'):
                if hasattr(processor.model.vision_tower, 'patch_size') and processor.model.vision_tower.patch_size is None:
                    processor.model.vision_tower.patch_size = DEFAULT_PATCH_SIZE
                    
            logger.info(f"Successfully set patch_size for {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to fix processor patch_size: {e}")
            # Don't fail on patch_size setting errors - continuing with a warning
    
    @classmethod
    def load_model(cls, model_name: str, is_vision_model: bool) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load a model from the cache or source.
        
        Args:
            model_name: Name of the model to load
            is_vision_model: Whether the model is a vision model
            
        Returns:
            Tuple of (model, processor, config)
            
        Raises:
            ModelLoadingError: If the model fails to load
        """
        # Check if model is cached
        if model_name in cls._model_cache:
            logger.info(f"Using cached model {model_name}")
            model, processor, _ = cls._model_cache[model_name]
            # Update the timestamp
            cls._model_cache[model_name] = (model, processor, time.time())
            
            # Get configuration
            try:
                config = load_config(model_name, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Could not load config for cached model {model_name}: {e}")
                config = {}
            
            return model, processor, config
        
        # Limit cache size by removing oldest model if needed
        if len(cls._model_cache) >= cls._max_cached_models:
            oldest_model = None
            oldest_time = float('inf')
            for model_key, (_, _, timestamp) in cls._model_cache.items():
                if timestamp < oldest_time:
                    oldest_time = timestamp
                    oldest_model = model_key
            
            if oldest_model:
                logger.info(f"Removing oldest model from cache: {oldest_model}")
                del cls._model_cache[oldest_model]
        
        try:
            logger.info(f"Loading model {model_name} from source")
            
            if is_vision_model:
                # Load vision model using MLX-VLM
                logger.info(f"Loading vision model {model_name}")
                try:
                    model, processor = load_vlm(
                        model_name,
                        trust_remote_code=True
                    )
                    config = load_config(model_name, trust_remote_code=True)
                    
                    # Fix processor patch_size which might be None
                    cls._fix_processor_patch_size(processor, model_name)
                    
                    logger.info(f"Successfully loaded vision model {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load vision model: {e}")
                    raise ModelLoadingError(
                        f"Failed to load vision model {model_name}: {str(e)}",
                        provider="mlx",
                        model_name=model_name
                    )
            else:
                # Load language model using MLX-LM
                logger.info(f"Loading language model {model_name}")
                try:
                    model, processor = mlx_lm.load(model_name)
                    config = load_config(model_name)
                    logger.info(f"Successfully loaded language model {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load language model: {e}")
                    raise ModelLoadingError(
                        f"Failed to load language model {model_name}: {str(e)}",
                        provider="mlx",
                        model_name=model_name
                    )
            
            # Add to in-memory cache
            cls._model_cache[model_name] = (model, processor, time.time())
            
            return model, processor, config
            
        except Exception as e:
            if isinstance(e, ModelLoadingError):
                raise
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadingError(
                f"Failed to load model {model_name}: {str(e)}",
                provider="mlx",
                model_name=model_name
            )
    
    @classmethod
    def generate_text(cls, model: Any, processor: Any, prompt: str, 
                    max_tokens: int = 100, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate text using MLX-LM.
        
        Args:
            model: The loaded MLX model
            processor: The model's processor/tokenizer
            prompt: The text prompt to use
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            The generated text content or generator
            
        Raises:
            RuntimeError: If generation fails
        """
        if not model or not processor:
            raise RuntimeError("Model or processor not provided")
            
        try:
            if stream:
                # Use MLX-LM's streaming functionality
                def response_generator():
                    current_response = ""
                    for token in mlx_lm.generate(
                        model=model,
                        tokenizer=processor,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temp=0.1,  # Use a low temperature for deterministic outputs
                        stream=True
                    ):
                        current_response += token
                        yield current_response
                
                return response_generator()
            else:
                # Use MLX-LM's generate function for non-streaming
                output = mlx_lm.generate(
                    model=model,
                    tokenizer=processor,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=0.1,  # Use a low temperature for deterministic outputs
                    stream=False
                )
                
                return output
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    @classmethod
    def generate_vision(cls, model: Any, processor: Any, prompt: str, image_path: str, 
                       max_tokens: int = 100, fallback_to_direct: bool = True) -> str:
        """
        Generate text from an image using a vision model.
        
        Args:
            model: The loaded MLX model
            processor: The model's processor/tokenizer
            prompt: The text prompt to use
            image_path: Path to the image file
            max_tokens: Maximum number of tokens to generate
            fallback_to_direct: Whether to fallback to direct API calls if standard generation fails
            
        Returns:
            The generated text content
            
        Raises:
            RuntimeError: If generation fails
        """
        if not model or not processor:
            raise RuntimeError("Model or processor not provided")
            
        try:
            # Debug the processor configuration
            logger.debug(f"Processor type: {type(processor)}")
            logger.debug(f"Processor attributes: {dir(processor)}")
            
            # Check for patch_size attribute and configuration
            if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "patch_size"):
                logger.debug(f"Image processor patch size: {processor.image_processor.patch_size}")
            elif hasattr(processor, "patch_size"):
                logger.debug(f"Processor patch size: {processor.patch_size}")
                
            try:
                # First try with standard mlx_vlm interface
                logger.info("Attempting standard mlx_vlm generation")
                
                # Ensure patch_size is set again before generation
                cls._fix_processor_patch_size(processor, "runtime_model")
                
                # Double-check image token in prompt for LLaVA-like models
                if "<image>" not in prompt and "llava" in str(processor.__class__).lower():
                    logger.info("Adding <image> tag for LLaVA model")
                    prompt = f"<image>\n{prompt}"
                
                try:
                    # Use MLX-VLM's generate function with the image path
                    from mlx_vlm.utils import generate as mlx_generate
                    output = mlx_generate(
                        model,
                        processor,
                        prompt=prompt,
                        image=image_path,
                        max_tokens=max_tokens
                    )
                    
                    # Extract the text content from the output
                    text_content = output[0] if isinstance(output, tuple) else output
                    return text_content
                except KeyError as e:
                    if "'input_ids'" in str(e):
                        logger.warning(f"KeyError for input_ids, trying manual tokenization approach")
                        
                        # Try manual tokenization and input preparation
                        if hasattr(processor, "tokenizer"):
                            tokenizer = processor.tokenizer
                        else:
                            tokenizer = processor
                            
                        from PIL import Image as PILImage
                        import mlx.core as mx
                        
                        # Manually tokenize the prompt
                        prompt_ids = tokenizer.encode(prompt)
                        attention_mask = [1] * len(prompt_ids)
                        
                        # Load the image
                        img = PILImage.open(image_path)
                        
                        # Process the image
                        if hasattr(processor, "image_processor"):
                            image_processor = processor.image_processor
                            # Use numpy which is compatible with our patch
                            pixel_values_np = image_processor(img, return_tensors="np")["pixel_values"]
                            pixel_values = mx.array(pixel_values_np)
                        else:
                            # Fallback to simpler processing
                            # Resize the image to expected dimensions
                            config = cls.get_model_config("default")
                            target_size = config["image_size"]
                            img = img.resize(target_size)
                            
                            # Process image data
                            img_data = np.array(img).astype(np.float32) / 255.0
                            # Convert to CHW format
                            img_data = np.transpose(img_data, (2, 0, 1))
                            pixel_values = mx.array(img_data).reshape(1, 3, target_size[0], target_size[1])
                            
                        # Create model inputs
                        model_inputs = {
                            "input_ids": mx.array(prompt_ids),
                            "attention_mask": mx.array(attention_mask),
                            "pixel_values": pixel_values
                        }
                        
                        # Generate with manual inputs 
                        # Use mlx_lm instead of generate_step which might not be available
                        try:
                            import mlx_lm
                            
                            # Create a simple prompt to start from
                            output = mlx_lm.generate(
                                model,
                                tokenizer,
                                prompt=prompt,
                                image=pixel_values,  # Add image directly
                                max_tokens=max_tokens,
                                temp=0.1,
                                verbose=False
                            )
                            
                            # Return result
                            return output
                        except Exception as e:
                            logger.error(f"mlx_lm generation failed: {e}")
                            
                            # Fall back to a very basic approach
                            # Start with prompt tokens
                            y = mx.array(prompt_ids)
                            
                            # Run generation loop
                            generated = []
                            try:
                                # Create mask for attention
                                mask = mx.ones(len(prompt_ids))
                                
                                for i in range(max_tokens):
                                    # Get next token using model with mask parameter
                                    logits = model(y, pixel_values=pixel_values, mask=mask)["logits"]
                                    
                                    # Simple greedy decoding - just take argmax
                                    next_token = mx.argmax(logits[-1])
                                    
                                    # Stop if we hit end token
                                    token_id = next_token.item()
                                    if hasattr(tokenizer, "eos_token_id") and token_id == tokenizer.eos_token_id:
                                        break
                                        
                                    # Append token to generated text
                                    generated.append(token_id)
                                    
                                    # Update input for next iteration
                                    y = mx.concatenate([y, next_token.reshape(1)])
                                    
                                    # Update mask
                                    mask = mx.ones(len(y))
                            except Exception as direct_e:
                                logger.error(f"Basic generation failed: {direct_e}")
                                # Just return an error message
                                return "Cannot process this image due to model compatibility issues."
                            
                            # Decode generated tokens
                            if generated:
                                generated_text = tokenizer.decode(generated)
                                return generated_text
                            else:
                                return "Failed to generate any tokens from the image."
                    else:
                        raise 
                    
            except Exception as e:
                logger.error(f"Standard generation failed: {e}")
                if not fallback_to_direct:
                    raise RuntimeError(f"Vision generation failed: {str(e)}")
                    
                # Try with direct API calls to mlx_vlm components
                logger.info("Falling back to direct mlx_vlm API calls")
                
                # Import what we need from mlx_vlm
                try:
                    from mlx_vlm.utils import prepare_inputs, generate_step, stream_generate, process_inputs_with_fallback
                    from PIL import Image as PILImage
                    import mlx.core as mx
                    
                    # Tokenize the prompt directly
                    if hasattr(processor, 'tokenizer'):
                        tokenizer = processor.tokenizer
                    else:
                        tokenizer = processor
                        
                    logger.debug("Tokenizing prompt directly")
                    
                    try:
                        # Try to encode with prompt
                        prompt_ids = tokenizer.encode(prompt)
                    except:
                        # Fallback to simpler encoding
                        if hasattr(tokenizer, "convert_tokens_to_ids"):
                            tokens = tokenizer.tokenize(prompt)
                            prompt_ids = tokenizer.convert_tokens_to_ids(tokens)
                        else:
                            # Create a basic prompt - just try something simple
                            prompt_ids = list(range(10))  # Placeholder
                            
                    # Load the image directly 
                    logger.debug(f"Loading image from {image_path}")
                    img = PILImage.open(image_path)
                    
                    # Create a simpler direct approach that doesn't rely on existing functions
                    try:
                        # Create inputs directly
                        inputs_ids = mx.array(prompt_ids)
                        attention_mask = mx.ones(len(prompt_ids))
                        
                        # Handle image preparation
                        # Use a standard size if available from config, otherwise default
                        img_size = (224, 224)
                        if hasattr(cls, "get_model_config"):
                            config = cls.get_model_config("default")
                            if "image_size" in config:
                                img_size = config["image_size"]
                        
                        # Resize and normalize the image
                        img_resized = img.resize(img_size)
                        img_array = np.array(img_resized).astype(np.float32) / 255.0
                        
                        # Handle different input formats
                        if img_array.ndim == 2:
                            # Convert grayscale to RGB
                            img_array = np.stack([img_array, img_array, img_array], axis=2)
                            
                        # Convert to CHW format for model
                        img_array = np.transpose(img_array, (2, 0, 1))
                        pixel_values = mx.array(img_array).reshape(1, 3, img_size[0], img_size[1])
                        
                        # Run a fixed number of generation steps
                        y = inputs_ids
                        generated_tokens = []
                        
                        # Generate a fixed number of tokens
                        for _ in range(max_tokens):
                            try:
                                # Create mask for current sequence
                                mask = mx.ones(len(y))
                                
                                # Call model with required parameters
                                outputs = model(
                                    y,
                                    pixel_values=pixel_values,
                                    mask=mask
                                )
                                
                                # Handle different output formats
                                if isinstance(outputs, dict) and "logits" in outputs:
                                    logits = outputs["logits"]
                                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                                    logits = outputs[0]
                                else:
                                    # Cannot get logits, stop generation
                                    break
                                
                                # Get next token (simple greedy approach)
                                next_token = mx.argmax(logits[-1], axis=0)
                                token_id = next_token.item()
                                
                                # Check for end of sequence
                                if hasattr(tokenizer, "eos_token_id") and token_id == tokenizer.eos_token_id:
                                    break
                                    
                                generated_tokens.append(token_id)
                                y = mx.concatenate([y, next_token.reshape(1)])
                            except Exception as gen_e:
                                logger.error(f"Generation step failed: {gen_e}")
                                break
                        
                        # Generate output
                        if generated_tokens:
                            try:
                                text_content = tokenizer.decode(generated_tokens)
                            except Exception:
                                # If decode fails, just return the token IDs
                                text_content = f"Generated {len(generated_tokens)} tokens but couldn't decode them."
                            return text_content
                        else:
                            return "No text could be generated for this image."
                            
                    except Exception as direct_e:
                        logger.error(f"Direct generation also failed: {direct_e}")
                        raise RuntimeError(f"All vision generation approaches failed: {direct_e}")
                        
                except Exception as fallback_e:
                    logger.error(f"Vision generation failed: {fallback_e}")
                    raise RuntimeError(f"All vision generation approaches failed: {fallback_e}")
                
        except Exception as e:
            logger.error(f"Vision generation failed: {e}")
            raise RuntimeError(f"Vision generation failed: {str(e)}")
            
    @classmethod
    def stream_generate_vision(cls, model: Any, processor: Any, prompt: str, image_path: str, 
                             max_tokens: int = 100) -> Generator[Any, None, None]:
        """
        Stream generate text from an image using a vision model.
        
        Args:
            model: The loaded MLX model
            processor: The model's processor/tokenizer
            prompt: The text prompt to use
            image_path: Path to the image file
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            A generator yielding response chunks
            
        Raises:
            RuntimeError: If generation fails
        """
        if not model or not processor:
            raise RuntimeError("Model or processor not provided")
            
        try:
            # Use MLX-VLM's stream_generate function with the image path
            for response in stream_generate(
                model,
                processor,
                prompt=prompt,
                image=image_path,
                max_tokens=max_tokens
            ):
                yield response
        except Exception as e:
            logger.error(f"Vision generation streaming failed: {e}")
            raise RuntimeError(f"Vision generation streaming failed: {str(e)}")
            
    @classmethod
    def format_prompt(cls, model_type: str, prompt: str, num_images: int = 0) -> str:
        """
        Format a prompt with image tokens based on model type.
        
        Args:
            model_type: The type of model
            prompt: The prompt text
            num_images: The number of images to include
            
        Returns:
            The formatted prompt
        """
        if num_images == 0:
            return prompt
        
        # No model type specified, use default
        if model_type is None:
            model_type = "default"
            
        # Get configuration
        config = cls.get_model_config(model_type)
        base_format = config["prompt_format"]
        
        logger.debug(f"Formatting prompt for model type {model_type} with format {base_format}")
        
        # Format the prompt according to model-specific conventions
        # Model-specific formatting for different vision models
        model_type_lower = model_type.lower()
        
        # LLaVA models (including bakllava)
        if "llava" in model_type_lower:
            # LLaVA uses a simple <image> token
            return f"<image>\n{prompt}"
            
        # Phi-3-Vision and similar
        elif "phi" in model_type_lower and ("vision" in model_type_lower or "visual" in model_type_lower):
            # Phi uses a specific format
            return f"Image: <image>\nUser: {prompt}\nAssistant:"
            
        # Qwen2-VL models
        elif "qwen2" in model_type_lower or "qwen2.5" in model_type_lower:
            # Qwen2 uses a specific vision token format
            return f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
            
        # Qwen-VL (original)
        elif "qwen" in model_type_lower:
            # Original Qwen-VL uses a different format
            return f"<img>{prompt}"
            
        # Gemma (including Gemma IT) models
        elif "gemma" in model_type_lower:
            # Gemma uses a simple format
            return f"<image>\n{prompt}"
            
        # Idefics models
        elif "idefics" in model_type_lower:
            # Idefics format
            return f"<image>\nUser: {prompt}\nAssistant:"
            
        # Default case - use the config's prompt format
        try:
            return base_format.format(prompt=prompt)
        except Exception as e:
            logger.warning(f"Failed to format prompt using template {base_format}: {e}")
            return f"<image>\n{prompt}"
    
    @classmethod
    def format_system_prompt(cls, model_type: str, system_prompt: str, user_prompt: str) -> str:
        """
        Format system and user prompts together based on model type.
        
        Args:
            model_type: The type of model
            system_prompt: The system prompt
            user_prompt: The user prompt
            
        Returns:
            The formatted prompt
        """
        # Model-specific system prompt formatting
        model_type_lower = model_type.lower() if model_type else ""
        
        if "llava" in model_type_lower:
            # LLaVA models typically use a specific system prompt format
            return f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n<|assistant|>"
        elif "phi" in model_type_lower and ("vision" in model_type_lower or "visual" in model_type_lower):
            # Phi-3-Vision format
            return f"System: {system_prompt}\n\nImage: <image>\nUser: {user_prompt}\nAssistant:"
        elif "qwen" in model_type_lower:
            # Qwen format
            return f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n<|assistant|>"
        elif "gemma" in model_type_lower:
            # Gemma format
            return f"<system>\n{system_prompt}\n</system>\n\n{user_prompt}"
            
        # Default simple concatenation for most models
        return f"{system_prompt}\n\n{user_prompt}" 