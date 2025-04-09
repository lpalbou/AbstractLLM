"""
Hugging Face implementation for AbstractLLM using Transformers.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, Tuple, ClassVar, List
import os
import asyncio
import logging
import time
import platform
import sys
import gc
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from pathlib import Path

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import (
    log_request, 
    log_response,
    log_request_url
)
from abstractllm.media.processor import MediaProcessor
from abstractllm.utils.config import ConfigurationManager

# Configure logger with specific class path
logger = logging.getLogger("abstractllm.providers.huggingface.HuggingFaceProvider")

# Default model - small, usable by most systems
DEFAULT_MODEL = "bartowski/microsoft_Phi-4-mini-instruct-GGUF"

# Default timeout in seconds for generation (preventing infinite loops)
DEFAULT_GENERATION_TIMEOUT = 60

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "openai/clip-vit-base-patch32",
    "facebook/dinov2-small",
    "Salesforce/blip2-opt-2.7b",
#    "deepseek-ai/deepseek-vl-7b-chat",  # DeepSeek Omni vision model
#    "Qwen/Qwen2.5-Omni-7B", 
#    "Qwen/Qwen2.5-VL-32B-Instruct", 
#    "Qwen/Qwen2.5-VL-7B-Instruct", 
#    "Qwen/Qwen2.5-VL-3B-Instruct", 
    "llava-hf/llava-v1.6-mistral-7b-hf", 
#    "microsoft/Phi-4-multimodal-instruct"
]

# Constants for model selection
TINY_TEST_PROMPT = "Hello, what is the capital of France?"  # Used for warmup"

def _get_optimal_device() -> str:
    """
    Determine the optimal device for model loading based on system capabilities.
    
    Returns:
        str: The optimal device to use ('cuda', 'mps', or 'cpu')
        
    Priority order:
    1. CUDA (if available)
    2. MPS (Apple Silicon, if available)
    3. CPU (fallback)
    """
    try:
        # Check for CUDA availability (NVIDIA GPUs)
        if torch.cuda.is_available():
            logger.info(f"CUDA detected with {torch.cuda.device_count()} device(s)")
            cuda_info = []
            for i in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_properties(i)
                cuda_info.append(f"{device.name} ({device.total_memory / (1024**3):.2f} GB)")
            logger.info(f"CUDA devices: {', '.join(cuda_info)}")
            return "cuda"
        
        # Check for MPS availability (Apple Silicon M-series)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) detected")
            return "mps"
    
    except ImportError:
        logger.warning("PyTorch not available for device detection")
    except Exception as e:
        logger.warning(f"Error detecting optimal device: {str(e)}")
    
    # Fallback to CPU
    logger.info("Using CPU for model inference (no GPU detected)")
    return "cpu"

class HuggingFaceProvider(AbstractLLMInterface):
    """
    Hugging Face implementation using Transformers.
    """
    
    # Default cache directory
    DEFAULT_CACHE_DIR = "~/.cache/abstractllm/models"
    
    # Class-level cache for sharing models between instances
    # Format: {(model_name, device_map, quantization): (model, tokenizer, last_used_time)}
    _model_cache: ClassVar[Dict[Tuple[str, str, bool, bool], Tuple[Any, Any, float]]] = {}
    
    # Maximum number of models to keep in the cache
    _max_cached_models = 3
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the Hugging Face provider.
        
        Args:
            config (dict): Configuration for the provider.
                Required keys:
                - model: The model ID or path to load
                
                Optional keys:
                - device: The device to load the model on (e.g., 'cpu', 'cuda', 'cuda:0')
                - load_in_8bit/load_in_4bit: Whether to quantize the model
                - max_tokens: Maximum number of tokens to generate
                - temperature: Temperature for sampling
                - top_p: Top-p sampling parameter
                - system_prompt: Default system prompt to use
                - trust_remote_code: Whether to trust remote code when loading the model
                - auto_load: Whether to load the model during initialization
                - auto_warmup: Whether to run a warmup pass after loading
        """
        super().__init__(config)
        
        # Initialize provider-specific configuration
        self.config = ConfigurationManager.initialize_provider_config("huggingface", self.config)
        
        # Store device preference (will be used later when loading model)
        self._device = ConfigurationManager.get_param(self.config, ModelParameter.DEVICE, _get_optimal_device())
        
        # Initialize model and tokenizer objects to None (will be loaded on demand)
        self._model = None
        self._tokenizer = None
        self._processor = None  # For vision models
        self._model_loaded = False
        self._warmup_completed = False
        
        # Log provider initialization
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, DEFAULT_MODEL)
        logger.info(f"Initialized HuggingFace provider with model: {model_name}")
        
        # Preload the model if auto_load is set
        if self.config.get("auto_load", False):
            self.load_model()
            
            # Run warmup if auto_warmup is set
            if self.config.get("auto_warmup", False):
                self.warmup()
    
    def preload(self) -> None:
        """
        Preload the model (alias for load_model for compatibility with tests).
        This ensures the model is loaded and ready for inference.
        
        Returns:
            None
        """
        self.load_model()
    
    def warmup(self) -> None:
        """
        Run a simple inference pass to ensure the model is fully loaded and optimized.
        This helps avoid the first inference being slow due to lazy initialization.
        
        Returns:
            None
        """
        if not self._model_loaded:
            self.load_model()
            
        logger.info("Warming up model with small inference pass...")
        try:
            # Create a simple input
            inputs = self._tokenizer(TINY_TEST_PROMPT, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Run inference with a small max_tokens to keep it fast
            generation_config = {
                "max_new_tokens": 5,
                "do_sample": False
            }
            
            # Add pad_token_id if available
            if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
                generation_config["pad_token_id"] = self._tokenizer.pad_token_id
            
            # Warmup inference
            start_time = time.time()
            with torch.no_grad():
                _ = self._model.generate(**inputs, **generation_config)
            end_time = time.time()
            
            self._warmup_completed = True
            logger.info(f"Warmup completed in {end_time - start_time:.2f}s")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def _get_cache_key(self) -> Tuple[str, str, bool, bool]:
        """
        Get a cache key for the current model configuration.
        
        Returns:
            A tuple that can be used as a dictionary key for the model cache
        """
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, DEFAULT_MODEL)
        device_map = self.config.get("device_map", "auto")
        load_in_8bit = self.config.get("load_in_8bit", False)
        load_in_4bit = self.config.get("load_in_4bit", False)
        return (model_name, str(device_map), load_in_8bit, load_in_4bit)
    
    def _clean_model_cache_if_needed(self) -> None:
        """
        Clean up the model cache if it exceeds the maximum size.
        Removes the least recently used models.
        
        Returns:
            None
        """
        if len(HuggingFaceProvider._model_cache) <= self._max_cached_models:
            return
            
        # Sort by last used time (oldest first)
        sorted_keys = sorted(
            HuggingFaceProvider._model_cache.keys(),
            key=lambda k: HuggingFaceProvider._model_cache[k][2]
        )
        
        # Remove oldest models
        models_to_remove = len(HuggingFaceProvider._model_cache) - self._max_cached_models
        for i in range(models_to_remove):
            key = sorted_keys[i]
            logger.info(f"Removing model from cache: {key[0]}")
            model, tokenizer, _ = HuggingFaceProvider._model_cache[key]
            
            # Set models to None to help with garbage collection
            model = None
            tokenizer = None
            
            # Remove from cache
            del HuggingFaceProvider._model_cache[key]
        
        # Explicitly run garbage collection to free memory
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_model(self) -> None:
        """
        Load the model and tokenizer based on the configuration.
        
        Returns:
            None
            
        Raises:
            ImportError: If required dependencies are not available
            Exception: For other loading errors
        """
        # Skip if model already loaded
        if self._model_loaded:
            logger.debug("Model already loaded, skipping load")
            return
            
        # Check if PyTorch is available first
        if not torch_available():
            raise ImportError("PyTorch is required for HuggingFace models")
            
        # Import here to avoid dependency issues for users not using HF
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        import os
        from pathlib import Path
            
        # Get model name
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, DEFAULT_MODEL)
        if not model_name:
            raise ValueError("Model name must be provided in configuration")
        
        # Check if this is a vision-capable model
        is_vision_capable = any(vision_model in model_name for vision_model in VISION_CAPABLE_MODELS)
        
        # Create a cache key for this model configuration
        cache_key = self._get_cache_key()
        
        # Check if model is already in cache
        if cache_key in HuggingFaceProvider._model_cache:
            logger.info(f"Loading model from cache: {model_name}")
            self._model, self._tokenizer, _ = HuggingFaceProvider._model_cache[cache_key]
            # Update last access time
            HuggingFaceProvider._model_cache[cache_key] = (self._model, self._tokenizer, time.time())
            self._model_loaded = True
            return
        
        # Log model loading at INFO level
        logger.info(f"Loading HuggingFace model: {model_name}{' (vision-capable)' if is_vision_capable else ''}")
        
        # Extract parameters for model loading
        load_in_8bit = self.config.get("load_in_8bit", False)
        load_in_4bit = self.config.get("load_in_4bit", False)
        
        # Let the user override the device_map if needed
        device_map = self.config.get("device_map", None)
        # If no device_map is specified, use the optimal device
        if device_map is None:
            device = self._device
            device_map = device
            logger.debug(f"Using device: {device_map}")
            
        # Other loading parameters
        cache_dir = self.config.get("cache_dir", HuggingFaceProvider.DEFAULT_CACHE_DIR)
        # Expand user directory in path if it exists
        if cache_dir and '~' in cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
        timeout = self.config.get("load_timeout", 300)  # 5 minutes default
        trust_remote_code = self.config.get("trust_remote_code", False)
        
        # Check if FlashAttention should be disabled
        disable_flash_attn = self.config.get("disable_flash_attn", True)
        
        # Add to model kwargs if needed 
        attn_implementation = "eager" if disable_flash_attn else "flash_attention_2"
        
        # Log detailed configuration at DEBUG level
        logger.debug(f"Model loading configuration: load_in_8bit={load_in_8bit}, load_in_4bit={load_in_4bit}, device_map={device_map}")
        logger.debug(f"FlashAttention disabled: {disable_flash_attn}, using attention implementation: {attn_implementation}")
        if cache_dir:
            logger.debug(f"Using custom cache directory: {cache_dir}")
        
        # Start a timer for timeout tracking
        start_time = time.time()
        
        try:
            # First check if the model exists on Hugging Face Hub
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                # Try to get model info to verify it exists and is accessible
                model_info = api.model_info(model_name)
                logger.debug(f"Model exists on HF Hub: {model_name}")
            except Exception as e:
                logger.warning(f"Could not verify model existence on HF Hub: {e}")
                # Continue anyway, as the model might be local or use a custom format
            
            # For vision models, try to load processor first
            if is_vision_capable:
                try:
                    logger.debug(f"Loading processor for vision model: {model_name}")
                    self._processor = AutoProcessor.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        trust_remote_code=trust_remote_code
                    )
                    logger.debug(f"Successfully loaded processor for {model_name}")
                except Exception as processor_error:
                    logger.warning(f"Failed to load processor: {processor_error}. Will try to use tokenizer only.")
                    self._processor = None
            
            # Load tokenizer with graceful fallback options
            logger.debug(f"Loading tokenizer for model: {model_name}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    use_fast=True
                )
            except Exception as tokenizer_error:
                logger.warning(f"Failed to load fast tokenizer: {tokenizer_error}. Trying alternative options...")
                
                # Try with use_fast=False
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        trust_remote_code=trust_remote_code,
                        use_fast=False
                    )
                except Exception:
                    # Last resort: try to load the GPT2 tokenizer as fallback
                    logger.warning("Trying GPT2 tokenizer as fallback")
                    self._tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
            
            # Handle pad token for tokenizers that don't have one
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    logger.debug("Setting pad_token to eos_token for tokenizer")
                elif self._tokenizer.bos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.bos_token
                    logger.debug("Setting pad_token to bos_token for tokenizer")
                elif self._tokenizer.unk_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.unk_token
                    logger.debug("Setting pad_token to unk_token for tokenizer")
                else:
                    # If all else fails, set a default pad token
                    self._tokenizer.pad_token = "[PAD]"
                    logger.debug("Setting default pad_token '[PAD]'")
            
            # Load model with appropriate settings
            model_kwargs = {
                "device_map": device_map,
                "cache_dir": cache_dir,
                "trust_remote_code": trust_remote_code,
                "attn_implementation": attn_implementation
            }
            
            if load_in_8bit:
                try:
                    import bitsandbytes
                    model_kwargs["load_in_8bit"] = True
                    logger.debug("Using 8-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not installed. Falling back to default precision.")
            elif load_in_4bit:
                try:
                    import bitsandbytes
                    model_kwargs["load_in_4bit"] = True
                    logger.debug("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not installed. Falling back to default precision.")
            
            logger.debug(f"Starting to load model with kwargs: {model_kwargs}")
            
            # Check for timeout during model loading
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timed out loading model {model_name} after {timeout} seconds")
            
            # For vision models, may need to use a different model class
            if is_vision_capable:
                try:
                    # Vision models require different handling based on model type
                    logger.debug(f"Loading vision-capable model: {model_name}")
                    
                    # Try model-specific approaches based on known architectures
                    if "deepseek-vl" in model_name.lower():
                        try:
                            # For DeepSeek vision models 
                            logger.debug("Loading DeepSeek vision model with AutoModelForCausalLM")
                            from transformers.models.auto import modeling_auto
                            
                            # Set torch dtype to load in lower precision if needed
                            if device_map not in ["cpu", "auto"]:
                                try:
                                    import torch
                                    model_kwargs["torch_dtype"] = torch.float16  # Use FP16 for GPU
                                    logger.debug("Using torch_dtype=float16 for DeepSeek model")
                                except Exception as dtype_error:
                                    logger.warning(f"Failed to set torch_dtype: {dtype_error}")
                                
                            self._model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                **model_kwargs
                            )
                        except Exception as e:
                            logger.error(f"Failed to load DeepSeek vision model: {e}")
                            raise
                    elif "llava" in model_name.lower():
                        try:
                            from transformers import LlavaForConditionalGeneration
                            logger.debug("Loading LlavaForConditionalGeneration model")
                            self._model = LlavaForConditionalGeneration.from_pretrained(
                                model_name,
                                **model_kwargs
                            )
                        except (ImportError, Exception) as e:
                            logger.debug(f"Failed to load LlavaForConditionalGeneration: {e}")
                            # Fall back to standard approach
                            self._model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                **model_kwargs
                            )
                    elif "qwen" in model_name.lower():
                        try:
                            if "qwen2" in model_name.lower():
                                try:
                                    from transformers import Qwen2VLForConditionalGeneration
                                    logger.debug("Loading Qwen2VLForConditionalGeneration model")
                                    self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                                        model_name,
                                        **model_kwargs
                                    )
                                except ImportError as e:
                                    logger.debug(f"Qwen2VLForConditionalGeneration not available: {e}")
                                    # Fall back to standard approach
                                    self._model = AutoModelForCausalLM.from_pretrained(
                                        model_name,
                                        **model_kwargs
                                    )
                            else:
                                from transformers import AutoModelForCausalLM
                                logger.debug("Loading QwenVL model with AutoModelForCausalLM")
                                self._model = AutoModelForCausalLM.from_pretrained(
                                    model_name,
                                    **model_kwargs
                                )
                        except Exception as e:
                            logger.debug(f"Failed to load Qwen vision model: {e}")
                            # Fall back to standard approach
                            self._model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                **model_kwargs
                            )
                    elif "phi-4" in model_name.lower():
                        try:
                            # For Phi-4 multimodal models
                            logger.debug("Loading Phi-4 multimodal model with AutoModelForCausalLM")
                            self._model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                **model_kwargs
                            )
                        except Exception as e:
                            logger.error(f"Failed to load Phi-4 multimodal model: {e}")
                            raise
                    else:
                        # Default to causal LM for other models
                        logger.debug("Using AutoModelForCausalLM for vision model")
                        self._model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            **model_kwargs
                        )
                except Exception as vision_error:
                    logger.error(f"All attempts to load vision model failed: {vision_error}")
                    raise
            else:
                # Load standard causal LM model
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            
            # Set pad_token_id in the model's config to match the tokenizer
            if hasattr(self._model, 'config') and self._tokenizer.pad_token_id is not None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id
                logger.debug(f"Set model's pad_token_id to tokenizer's pad_token_id: {self._tokenizer.pad_token_id}")
            
            # Add to cache
            HuggingFaceProvider._model_cache[cache_key] = (self._model, self._tokenizer, time.time())
            
            # Clean up cache if needed
            self._clean_model_cache_if_needed()
            
            logger.info(f"Successfully loaded model {model_name}")
            self._model_loaded = True
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using HuggingFace model.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
                - image: A single image (URL, file path, base64 string, or dict)
                - images: List of images (URLs, file paths, base64 strings, or dicts)
            
        Returns:
            The generated response or a generator if streaming
            
        Raises:
            Exception: If the generation fails
        """
        # Load model if not already loaded
        if not self._model_loaded:
            self.load_model()
        
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "huggingface", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, DEFAULT_MODEL)
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 2048)
        system_prompt = params.get("system_prompt")
        top_p = params.get("top_p", 1.0)
        repetition_penalty = params.get("repetition_penalty", 1.0)
        timeout = params.get("generation_timeout", DEFAULT_GENERATION_TIMEOUT)
        
        # Process image inputs if any
        if "image" in params or "images" in params:
            logger.info("Processing image inputs for vision request")
            params = MediaProcessor.process_inputs(params, "huggingface")
        
        # Log the request
        log_request("huggingface", prompt, {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream,
            "image_request": "image" in params or "images" in params
        })
        
        # Handle special case for multimodal generation
        if "image" in params or "images" in params:
            # Check if the model is vision-capable
            if not any(vision_model in model_name for vision_model in VISION_CAPABLE_MODELS):
                raise ValueError(f"Model {model_name} does not support vision. Choose a vision-capable model.")
            
            # Process image(s)
            image_data = None
            if "image" in params:
                image_data = params["image"]
            elif "images" in params and params["images"]:
                # Take the first image if multiple are provided
                image_data = params["images"][0]
                if len(params["images"]) > 1:
                    logger.warning("Multiple images provided, but only the first one will be used.")
            
            # Detect phi4-multimodal models which need special handling
            is_phi4_vision = "phi-4-multimodal" in model_name.lower() or "phi4-multimodal" in model_name.lower()
            
            # Handle phi4-multimodal models
            if is_phi4_vision:
                return self._generate_phi4_vision(prompt, image_data, system_prompt, stream, max_tokens, temperature, top_p, repetition_penalty, timeout)
            
            # Handle other vision models (placeholder for different model-specific implementations)
            else:
                # Return this for now
                raise NotImplementedError(f"Vision generation not implemented for {model_name}. Currently only Phi-4-multimodal is supported.")
        
        # Create the input
        input_text = prompt
        if system_prompt:
            # Try to follow a common format for system instruction
            input_text = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
            
        # Define generation config
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0.0001,  # Only use sampling if temperature is non-zero
        }
        
        # Add pad_token_id if available
        if hasattr(self._tokenizer, 'pad_token_id') and self._tokenizer.pad_token_id is not None:
            generation_config["pad_token_id"] = self._tokenizer.pad_token_id
            
        # Add stopping criteria if requested
        stop_sequences = params.get("stop", None)
        if stop_sequences:
            stopping_criteria = self._create_stopping_criteria(stop_sequences, input_text)
            generation_config["stopping_criteria"] = stopping_criteria
            
        # Encode input text
        inputs = self._tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Handle streaming if requested
        if stream:
            logger.info("Starting streaming generation")
            
            # Use the phi4 multimodal generator if it's the phi4 model (without image)
            is_phi4 = "phi-4" in model_name.lower() or "phi4" in model_name.lower()
            if is_phi4:
                generator = self._phi4_multimodal_stream_generator(inputs, generation_config, timeout)
            else:
                generator = self._stream_generator(inputs, generation_config, timeout)
                
            return generator
        else:
            # Standard non-streaming generation
            try:
                # Set up timeout
                import signal
                
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutException("Generation timed out")
                
                # Set the timeout
                original_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                
                try:
                    # Generate output
                    with torch.no_grad():
                        outputs = self._model.generate(**inputs, **generation_config)
                    
                    # Decode and return result
                    output_text = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    
                    # Clean the output text
                    output_text = output_text.strip()
                    
                    # Log the result
                    log_response("huggingface", output_text)
                    logger.info("Generation completed successfully")
                    
                    return output_text
                except TimeoutException:
                    logger.error(f"Generation timed out after {timeout} seconds")
                    raise RuntimeError(f"Generation timed out after {timeout} seconds")
                finally:
                    # Restore the original alarm handler
                    signal.signal(signal.SIGALRM, original_handler)
                    signal.alarm(0)
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                raise
    
    async def generate_async(self, 
                        prompt: str, 
                        system_prompt: Optional[str] = None, 
                        stream: bool = False, 
                        **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate a response using HuggingFace model asynchronously.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
                - image: A single image (URL, file path, base64 string, or dict)
                - images: List of images (URLs, file paths, base64 strings, or dicts)
                
        Returns:
            The generated response or an async generator if streaming
        """
        # Import required modules for async execution
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Load model if not already loaded
        if not self._model_loaded:
            # Load model in a thread to avoid blocking the event loop
            with ThreadPoolExecutor() as executor:
                await asyncio.get_event_loop().run_in_executor(
                    executor, self.load_model
                )
        
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "huggingface", self.config, kwargs, system_prompt
        )
        
        # Process stream parameter
        if stream:
            # Define an async generator that wraps the sync generator
            async def async_stream_wrapper():
                sync_gen = self.generate(prompt, system_prompt, stream=True, **kwargs)
                for chunk in sync_gen:
                    yield chunk
                    # Add a small sleep to allow other async tasks to run
                    await asyncio.sleep(0.01)
            
            return async_stream_wrapper()
        else:
            # Run the synchronous generate method in a thread pool
            with ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, 
                    lambda: self.generate(prompt, system_prompt, stream=False, **kwargs)
                )
                return result
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the Hugging Face provider.
        
        Returns:
            Dictionary of capabilities
        """
        # Get model name from config
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, DEFAULT_MODEL)
        
        # Check if this is a vision-capable model
        is_vision_capable = any(vision_model in model_name for vision_model in VISION_CAPABLE_MODELS) if model_name else False
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: None,  # Varies by model and hardware
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,  # Not typically supported natively
            ModelCapability.VISION: is_vision_capable  # Check if model is in vision capable list
        }
    
    @staticmethod
    def list_cached_models(cache_dir: Optional[str] = None) -> list:
        """
        List all models cached locally.
        
        Args:
            cache_dir: Custom cache directory (uses DEFAULT_CACHE_DIR if None)
            
        Returns:
            List of dictionaries with model information
        """
        try:
            from huggingface_hub import scan_cache_dir
            import os
            
            # Use default cache directory if none provided
            if cache_dir is None:
                cache_dir = HuggingFaceProvider.DEFAULT_CACHE_DIR
            
            # Expand user directory in path if it exists
            if cache_dir and '~' in cache_dir:
                cache_dir = os.path.expanduser(cache_dir)
                # Create cache directory if it doesn't exist
                os.makedirs(cache_dir, exist_ok=True)
                
            # Check if directory exists before scanning
            if not os.path.exists(cache_dir):
                logger.warning(f"Cache directory {cache_dir} does not exist")
                return []
                
            cache_info = scan_cache_dir(cache_dir)
            models = []
            
            for repo in cache_info.repos:
                models.append({
                    "name": repo.repo_id,
                    "size": repo.size_on_disk,
                    "last_used": repo.last_accessed
                })
                
            return models
        except ImportError:
            raise ImportError("huggingface_hub package is required for this feature")
    
    @staticmethod
    def clear_model_cache(model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        """
        Clear cached models.
        
        Args:
            model_name: Specific model to clear (None for all)
            cache_dir: Custom cache directory (uses DEFAULT_CACHE_DIR if None)
            
        Returns:
            None
        """
        try:
            from huggingface_hub import delete_cache_folder, scan_cache_dir
            import os
            
            # Use default cache directory if none provided
            if cache_dir is None:
                cache_dir = HuggingFaceProvider.DEFAULT_CACHE_DIR
            
            # Expand user directory in path if it exists
            if cache_dir and '~' in cache_dir:
                cache_dir = os.path.expanduser(cache_dir)
                
            # Check if directory exists before scanning
            if not os.path.exists(cache_dir):
                logger.warning(f"Cache directory {cache_dir} does not exist, nothing to clear")
                return
                
            if model_name:
                # Delete specific model
                cache_info = scan_cache_dir(cache_dir)
                for repo in cache_info.repos:
                    if repo.repo_id == model_name:
                        delete_cache_folder(repo_id=model_name, cache_dir=cache_dir)
                        return
                raise ValueError(f"Model {model_name} not found in cache")
            else:
                # Delete entire cache
                delete_cache_folder(cache_dir=cache_dir)
        except ImportError:
            raise ImportError("huggingface_hub package is required for this feature")

    def _stream_generator(self, inputs, generation_config, timeout):
        """
        Helper method to generate streaming text using Hugging Face models.
        
        Args:
            inputs: Processed inputs for the model
            generation_config: Generation parameters
            timeout: Generation timeout in seconds
            
        Returns:
            Generator yielding text chunks
        """
        import torch
        
        def response_generator():
            start_time = time.time()
            
            # Get the tokenizer (for decoding)
            tokenizer = self._tokenizer
            
            # Set up the generation parameters
            gen_config = generation_config.copy()
            
            # Stream token by token
            generated_ids = inputs["input_ids"]
            past_key_values = None
            token_cache = []
            past = None
            
            for _ in range(gen_config.pop("max_new_tokens", 100)):
                # Check timeout
                if timeout and time.time() - start_time > timeout:
                    logger.warning("Generation timeout reached")
                    yield "\n[Generation timeout reached]"
                    break
                
                try:
                    # Set up model inputs for the next token
                    model_inputs = {
                        "input_ids": generated_ids,
                        "past_key_values": past_key_values if past_key_values is not None else past
                    }
                    
                    # Generate next token
                    with torch.no_grad():
                        outputs = self._model(**model_inputs, use_cache=True)
                    
                    # Get probabilities for the next token
                    next_token_logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    
                    # Apply temperature
                    if "temperature" in gen_config and gen_config["temperature"] > 0:
                        next_token_logits = next_token_logits / gen_config["temperature"]
                    
                    # Apply top-p
                    if "top_p" in gen_config and gen_config["top_p"] < 1.0:
                        next_token_logits = self._top_p_filtering(next_token_logits, 
                                                                 top_p=gen_config["top_p"])
                    
                    # Sample from the filtered distribution
                    if gen_config.get("do_sample", False):
                        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Append to the sequence
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Check if we're at an EOS token
                    if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                        break
                    
                    # Decode the token
                    token = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    token_cache.append(token)
                    
                    # Yield the token if it's not empty
                    if token:
                        yield token
                    
                    # For efficiency, yield accumulated tokens and clear cache periodically
                    if len(token_cache) > 5:
                        token_cache = []
                        
                except Exception as e:
                    logger.error(f"Error during streaming: {e}")
                    yield f"\n[Error: {str(e)}]"
                    break
            
            # Final check for any remaining tokens
            if token_cache:
                remaining = "".join(token_cache)
                if remaining:
                    yield remaining
        
        return response_generator()
    
    def _phi4_multimodal_stream_generator(self, inputs, generation_config, timeout):
        """
        Helper method to generate streaming text specifically for Phi-4 multimodal model.
        
        Args:
            inputs: Processed inputs for the model
            generation_config: Generation parameters
            timeout: Generation timeout in seconds
            
        Returns:
            Generator yielding text chunks
        """
        import torch
        
        def response_generator():
            start_time = time.time()
            
            try:
                # Get the processor for decoding
                processor = self._processor
                
                # Set up streamer
                from transformers import TextIteratorStreamer
                from threading import Thread
                
                streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
                
                # Copy the generation config
                gen_config = generation_config.copy()
                
                # Set up the generation parameters
                max_new_tokens = gen_config.pop("max_new_tokens", 100)
                
                # Set up model inputs for generation
                inputs_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Start generation in a separate thread
                generation_kwargs = dict(
                    **inputs_copy,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    **gen_config
                )
                
                thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Yield from the streamer
                for text_chunk in streamer:
                    # Check timeout
                    if timeout and time.time() - start_time > timeout:
                        logger.warning("Generation timeout reached")
                        yield "\n[Generation timeout reached]"
                        break
                    
                    yield text_chunk
                
                # Wait for the generation to complete
                thread.join()
                
            except Exception as e:
                logger.error(f"Error during Phi-4 streaming: {e}")
                yield f"\n[Error: {str(e)}]"
        
        return response_generator()
    
    def _top_p_filtering(self, logits, top_p=0.9):
        """
        Filter logits using nucleus (top-p) filtering.
        
        Args:
            logits: Logits to filter
            top_p: Cumulative probability threshold
            
        Returns:
            Filtered logits
        """
        import torch
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted indices to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("Inf")
        return logits

    def _create_stopping_criteria(self, stop_sequences, prompt=""):
        """
        Create a stopping criteria for text generation based on stop sequences.
        
        Args:
            stop_sequences: List of sequences that should stop generation when encountered
            prompt: The input prompt (used to determine prompt length)
            
        Returns:
            StoppingCriteriaList object
        """
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopSequenceCriteria(StoppingCriteria):
                """Criteria to stop generation when any of the specified sequences is generated."""
                def __init__(self, tokenizer, stop_sequences, prompt_length):
                    self.tokenizer = tokenizer
                    self.stop_sequences = stop_sequences
                    self.prompt_length = prompt_length
                    
                    # Pre-compute token IDs for stop sequences when possible
                    self.stop_sequence_ids = []
                    for seq in stop_sequences:
                        # Only use sequences that can be tokenized
                        try:
                            if seq:
                                ids = self.tokenizer.encode(seq, add_special_tokens=False)
                                if ids:
                                    self.stop_sequence_ids.append(ids)
                        except:
                            pass  # Skip sequences that can't be tokenized
                
                def __call__(self, input_ids, scores, **kwargs):
                    # Only check the part after the prompt
                    generated_ids = input_ids[0, self.prompt_length:]
                    
                    # If nothing generated yet, continue
                    if len(generated_ids) == 0:
                        return False
                    
                    # Check if any stop sequence is present in the generated text
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    
                    # Check for any stop sequence in the generated text
                    for seq in self.stop_sequences:
                        if seq and seq in generated_text:
                            return True
                    
                    # Check for token-level matches (more efficient for longer sequences)
                    if len(generated_ids) >= 1:
                        for stop_ids in self.stop_sequence_ids:
                            if len(stop_ids) <= len(generated_ids):
                                # Check if the end of generated_ids matches any stop_ids
                                if stop_ids == generated_ids[-len(stop_ids):].tolist():
                                    return True
                    
                    return False
            
            # Create stopping criteria with the prompt length for proper context
            prompt_length = len(self._tokenizer.encode(prompt, add_special_tokens=True))
            criteria = StopSequenceCriteria(self._tokenizer, stop_sequences, prompt_length)
            
            return StoppingCriteriaList([criteria])
            
        except ImportError:
            logger.warning("transformers.StoppingCriteria not available. Stop sequences will be ignored.")
            return None

def torch_available() -> bool:
    """
    Check if PyTorch is available.
    
    Returns:
        bool: True if PyTorch is available
    """
    try:
        import torch
        return True
    except ImportError:
        return False

# Simple adapter class for tests
class HuggingFaceLLM:
    """
    Simple adapter around HuggingFaceProvider for test compatibility.
    """
    
    def __init__(self, model="llava-hf/llava-1.5-7b-hf", api_key=None):
        """
        Initialize a HuggingFace LLM instance.
        
        Args:
            model: The model to use
            api_key: Optional API key (will use environment variable if not provided)
        """
        config = {
            ModelParameter.MODEL: model,
        }
        
        if api_key:
            config[ModelParameter.API_KEY] = api_key
            
        self.provider = HuggingFaceProvider(config)
        
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