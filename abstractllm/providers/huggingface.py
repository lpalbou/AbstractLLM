"""
HuggingFace provider for AbstractLLM.

This module provides direct integration with HuggingFace models using the transformers library.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, List, ClassVar, Tuple
import os
import asyncio
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    TextIteratorStreamer,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    LlavaForConditionalGeneration,
    LlavaProcessor
)
from pathlib import Path
from PIL import Image
import psutil
import requests
from urllib.parse import urlparse

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import log_request, log_response, log_request_url
from abstractllm.media.factory import MediaFactory
from abstractllm.exceptions import (
    UnsupportedOperationError, 
    ModelNotFoundError, 
    FileProcessingError, 
    UnsupportedFeatureError, 
    ImageProcessingError,
    GenerationError
)
from abstractllm.media.image import ImageInput
from abstractllm.utils.config import ConfigurationManager

# Configure logger
logger = logging.getLogger("abstractllm.providers.huggingface")

# Default timeout in seconds for generation
DEFAULT_GENERATION_TIMEOUT = 60

# Models that support vision capabilities with their specific architectures
VISION_CAPABLE_MODELS = {
    "Salesforce/blip-image-captioning-base": "vision_seq2seq",
    "Salesforce/blip-image-captioning-large": "vision_seq2seq",
    "liuhaotian/llava-v1.5-7b": "llava",
    "llava-hf/llava-1.5-7b-hf": "llava",
    "llava-hf/llava-v1.6-mistral-7b-hf": "llava",
    "microsoft/git-base": "vision_encoder",
    "microsoft/git-large": "vision_encoder"
}

# Model architecture to class mapping
MODEL_CLASSES = {
    "vision_seq2seq": (BlipProcessor, BlipForConditionalGeneration),
    "vision_causal_lm": (AutoProcessor, AutoModelForCausalLM),
    "vision_encoder": (AutoProcessor, AutoModelForVision2Seq),
    "causal_lm": (AutoTokenizer, AutoModelForCausalLM),
    "llava": (LlavaProcessor, LlavaForConditionalGeneration)
}

# Add after the existing MODEL_CLASSES definition
QUANTIZATION_VARIANTS = {
    "4bit": ["Q4_K_S", "Q4_K_M", "Q4_K_L"],  # Ordered by size/quality
    "5bit": ["Q5_K_S", "Q5_K_M", "Q5_K_L"],
    "6bit": ["Q6_K", "Q6_K_L"],
    "8bit": ["Q8_0"]
}

class HuggingFaceProvider(AbstractLLMInterface):
    """
    HuggingFace implementation using Transformers.
    """
    
    # Class-level model cache
    _model_cache: ClassVar[Dict[Tuple[str, str, bool, bool], Tuple[Any, Any, float]]] = {}
    _max_cached_models = 3
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the HuggingFace provider."""
        super().__init__(config)
        
        # Set default configuration for HuggingFace
        default_config = {
            ModelParameter.MODEL: "microsoft/Phi-4-mini-instruct",
            # ModelParameter.MODEL: "ibm-granite/granite-3.2-2b-instruct",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 1024,
            ModelParameter.DEVICE: self._get_optimal_device(),
            "trust_remote_code": True,
            "load_in_8bit": False,  # Enable 8-bit quantization by default
            "load_in_4bit": True,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",  # More memory efficient attention
            "load_timeout": 300,
            "generation_timeout": DEFAULT_GENERATION_TIMEOUT,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            # Add new quantization parameters with defaults
            "quantized_model": False,
            "quantization_type": None
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize model components
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._model_loaded = False
        self._model_type = "causal"  # Default model type
        
        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized HuggingFace provider with model: {model}")
    
    @staticmethod
    def _get_optimal_device() -> str:
        """Determine the optimal device for model loading."""
        try:
            if torch.cuda.is_available():
                logger.info(f"CUDA detected with {torch.cuda.device_count()} device(s)")
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS (Apple Silicon) detected")
                return "mps"
        except Exception as e:
            logger.warning(f"Error detecting optimal device: {e}")
        
        logger.info("Using CPU for model inference")
        return "cpu"
    
    def _get_model_architecture(self, model_name: str) -> str:
        """Determine the model architecture type based on the model name."""
        # Check exact matches first
        if model_name in VISION_CAPABLE_MODELS:
            return VISION_CAPABLE_MODELS[model_name]
            
        # Then check patterns
        if "llava" in model_name.lower():
            return "llava"
        if any(vision_model in model_name.lower() for vision_model in ["blip", "git"]):
            return "vision_seq2seq"
        return "causal_lm"
    
    def _get_model_classes(self, model_type: str) -> Tuple[Any, Any]:
        """Get the appropriate processor and model classes based on model type."""
        if model_type not in MODEL_CLASSES:
            logger.warning(f"Unknown model type {model_type}, falling back to causal_lm")
            model_type = "causal_lm"
            
        return MODEL_CLASSES[model_type]
    
    def _get_quantized_model_name(self, base_model: str, quant_type: str) -> str:
        """
        Convert a base model name to its quantized variant name.
        Example: microsoft/Phi-4-mini-instruct -> bartowski/microsoft_Phi-4-mini-instruct-GGUF/Q4_K_L
        """
        try:
            # Extract model name without organization
            base_name = base_model.split('/')[-1]
            # Convert to GGUF format
            return f"bartowski/{base_name}-GGUF/{quant_type}"
        except Exception as e:
            logger.error(f"Failed to construct quantized model name: {e}")
            return base_model

    def _is_direct_url(self, model_name: str) -> bool:
        """
        Check if the model name is a direct URL.
        
        Args:
            model_name: Model name or URL
            
        Returns:
            bool: True if it's a URL, False if it's a model ID
        """
        try:
            result = urlparse(model_name)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _download_model(self, url: str, cache_dir: Optional[str] = None) -> str:
        """
        Download a model from a direct URL.
        
        Args:
            url: URL to download from
            cache_dir: Optional cache directory
            
        Returns:
            str: Path to the downloaded model
            
        Raises:
            RuntimeError: If download fails or file size is incorrect
        """
        try:
            # Create cache directory if it doesn't exist
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            else:
                cache_dir = os.path.expanduser("~/.cache/abstractllm/models")
                os.makedirs(cache_dir, exist_ok=True)

            # Extract filename from URL
            filename = os.path.basename(urlparse(url).path)
            local_path = os.path.join(cache_dir, filename)

            # First, make a HEAD request to get the expected file size
            head_response = requests.head(url, allow_redirects=True)
            head_response.raise_for_status()
            expected_size = int(head_response.headers.get('content-length', 0))
            if expected_size == 0:
                raise RuntimeError("Could not determine expected file size from server")
            
            logger.info(f"Expected model size: {expected_size / (1024*1024*1024):.2f} GB")

            # Check if file already exists and has correct size
            if os.path.exists(local_path):
                actual_size = os.path.getsize(local_path)
                if actual_size == expected_size:
                    logger.info(f"Model already exists at {local_path} with correct size")
                    return local_path
                else:
                    logger.warning(f"Existing model file has incorrect size. Expected: {expected_size}, Found: {actual_size}")
                    os.remove(local_path)

            # Download the file
            logger.info(f"Downloading model from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Write the file with progress tracking
            downloaded = 0
            last_log_time = time.time()
            log_interval = 5  # Log every 5 seconds
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192*1024):  # 8MB chunks
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    f.write(chunk)
                    
                    # Log progress periodically
                    current_time = time.time()
                    if current_time - last_log_time >= log_interval:
                        progress = (downloaded / expected_size) * 100
                        speed = downloaded / (current_time - last_log_time) / (1024*1024)  # MB/s
                        logger.info(f"Download progress: {progress:.1f}% ({speed:.1f} MB/s)")
                        last_log_time = current_time

            # Verify final file size
            actual_size = os.path.getsize(local_path)
            if actual_size != expected_size:
                os.remove(local_path)
                raise RuntimeError(
                    f"Downloaded file size ({actual_size}) does not match expected size ({expected_size})"
                )

            logger.info(f"Model successfully downloaded to {local_path}")
            return local_path

        except Exception as e:
            # Clean up partial download if it exists
            if 'local_path' in locals() and os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(f"Failed to download model: {str(e)}") from e

    def _load_gguf_model(self, model_path: str) -> None:
        """
        Load a GGUF model using llama-cpp-python.
        
        Args:
            model_path: Path to the GGUF model file
            
        Raises:
            RuntimeError: If loading fails or required libraries are not available
        """
        try:
            from llama_cpp import Llama
            logger.info("Using llama-cpp-python for GGUF model loading")
            
            # Get device configuration
            device = self.config_manager.get_param("device", "cpu")
            n_gpu_layers = 0  # Default to CPU
            
            # Configure GPU acceleration based on platform
            if device != "cpu":
                import platform
                system = platform.system().lower()
                
                if system == "darwin" and device == "mps":
                    # Metal support for macOS
                    n_gpu_layers = -1  # Use all layers
                    logger.info("Enabling Metal acceleration for macOS")
                elif (system in ["linux", "windows"]) and device == "cuda":
                    # CUDA support for Linux/Windows
                    n_gpu_layers = -1  # Use all layers
                    logger.info("Enabling CUDA acceleration")
                else:
                    logger.warning(f"Unsupported device {device} for {system}, falling back to CPU")
            
            # Initialize model with appropriate parameters
            self._model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=os.cpu_count(),  # Use all available CPU threads
                n_gpu_layers=n_gpu_layers,  # GPU acceleration if enabled
                seed=self.config_manager.get_param("seed", 0),  # Random seed
                verbose=self.config_manager.get_param("verbose", False)  # Logging
            )
            
            # Create a simple tokenizer wrapper to match HF interface
            class GGUFTokenizer:
                def __init__(self, model):
                    self.model = model
                    self.eos_token = "</s>"
                    self.pad_token = "</s>"
                    self.bos_token = "<s>"
                
                def encode(self, text, **kwargs):
                    return self.model.tokenize(text.encode())
                
                def decode(self, tokens, **kwargs):
                    return self.model.detokenize(tokens).decode()
            
            self._tokenizer = GGUFTokenizer(self._model)
            logger.info("Successfully loaded GGUF model")
            
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for loading GGUF models. "
                "Install it with: pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model: {str(e)}")

    def load_model(self) -> None:
        """Load the model and tokenizer/processor."""
        try:
            # Get configuration parameters
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            cache_dir = self.config_manager.get_param("cache_dir")
            device = self.config_manager.get_param("device", "cpu")
            trust_remote_code = self.config_manager.get_param("trust_remote_code", True)
            use_auth_token = self.config_manager.get_param(ModelParameter.API_KEY)
            
            # Check for quantized model preference
            use_quantized = self.config_manager.get_param("quantized_model", False)
            quant_type = self.config_manager.get_param("quantization_type")

            # Handle HuggingFace Hub authentication
            if use_auth_token:
                import huggingface_hub
                huggingface_hub.login(token=use_auth_token)

            # Check if model_name is a direct URL
            is_url = self._is_direct_url(model_name)
            if is_url:
                logger.info(f"Loading model from direct URL: {model_name}")
                try:
                    # Download the model
                    local_path = self._download_model(model_name, cache_dir)
                    # Update model_name to use local path
                    model_name = local_path
                    # Force quantized model handling for GGUF files
                    if model_name.endswith('.gguf'):
                        use_quantized = True
                        self._load_gguf_model(model_name)
                        self._model_loaded = True
                        return
                except Exception as e:
                    raise RuntimeError(f"Failed to download model from URL: {str(e)}") from e

            # Determine model architecture and get appropriate classes
            self._model_type = self._get_model_architecture(model_name)
            processor_class, model_class = self._get_model_classes(self._model_type)
            
            # Log loading strategy
            if use_quantized:
                if is_url:
                    logger.info(f"Loading pre-quantized model from local path: {model_name}")
                else:
                    if quant_type:
                        logger.info(f"Loading pre-quantized model variant: {quant_type}")
                        model_name = self._get_quantized_model_name(model_name, quant_type)
                        logger.info(f"Quantized model path: {model_name}")
            else:
                logger.info(f"Loading {model_name} as {self._model_type} architecture")
                if self.config_manager.get_param("load_in_4bit"):
                    logger.warning("Using on-the-fly 4-bit quantization. This process can take several minutes depending on model size and hardware.")
                    start_time = time.time()
                elif self.config_manager.get_param("load_in_8bit"):
                    logger.warning("Using on-the-fly 8-bit quantization. This process can take several minutes depending on model size and hardware.")
                    start_time = time.time()

            # Load processor/tokenizer first for vision models
            if self._model_type in ["vision_seq2seq", "llava"]:
                self._processor = processor_class.from_pretrained(
                    model_name if not use_quantized else model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    use_auth_token=use_auth_token
                )
                if self._model_type == "llava":
                    self._tokenizer = self._processor.tokenizer
                    if self._tokenizer.pad_token is None:
                        self._tokenizer.pad_token = self._tokenizer.eos_token
                    if self._tokenizer.bos_token is None:
                        self._tokenizer.bos_token = self._tokenizer.eos_token
            else:
                self._tokenizer = processor_class.from_pretrained(
                    model_name if not use_quantized else model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    use_auth_token=use_auth_token
                )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                if self._tokenizer.bos_token is None:
                    self._tokenizer.bos_token = self._tokenizer.eos_token

            # Load the model
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Prepare model loading parameters
            model_params = {
                "cache_dir": cache_dir,
                "trust_remote_code": trust_remote_code,
                "torch_dtype": torch_dtype,
                "device_map": device_map,
                "use_auth_token": use_auth_token
            }

            # Add quantization parameters if not using pre-quantized model
            if not use_quantized:
                model_params.update({
                    "load_in_4bit": self.config_manager.get_param("load_in_4bit", False),
                    "load_in_8bit": self.config_manager.get_param("load_in_8bit", False)
                })

            # Load the model with appropriate parameters
            self._model = model_class.from_pretrained(
                model_name if not use_quantized else model_name,
                **model_params
            )

            # Log quantization completion time if applicable
            if not use_quantized and (self.config_manager.get_param("load_in_4bit") or self.config_manager.get_param("load_in_8bit")):
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"Quantization completed in {duration:.2f} seconds")

            # Move model to device if not using device_map="auto"
            if device_map is None:
                self._model.to(device)

            # Resize token embeddings if needed
            if len(self._tokenizer) > self._model.config.vocab_size:
                self._model.resize_token_embeddings(len(self._tokenizer))

            self._model_loaded = True
            logger.info(f"Successfully loaded model")
        
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _move_inputs_to_device(self, inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Move input tensors to the specified device."""
        if device == "cpu":
            return inputs
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _get_model_prompt_format(self, model_name: str) -> Dict[str, Any]:
        """
        Get the model's prompt format configuration.
        
        This method tries multiple approaches to determine the correct prompt format:
        1. Check model config for chat_template
        2. Check tokenizer for chat_template
        3. Use known model patterns
        4. Fall back to default format
        """
        format_info = {
            "is_chat_model": False,
            "template": None,
            "roles": {
                "system": "System: ",
                "user": "Human: ",
                "assistant": "Assistant: "
            }
        }
        
        try:
            # Try to get config-based template
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, "chat_template"):
                format_info["template"] = config.chat_template
                format_info["is_chat_model"] = True
                return format_info
            
            # Check tokenizer for chat template
            if hasattr(self._tokenizer, "chat_template"):
                format_info["template"] = self._tokenizer.chat_template
                format_info["is_chat_model"] = True
                return format_info
            
            # Check for known model patterns
            model_name_lower = model_name.lower()
            if "phi" in model_name_lower:
                format_info["roles"] = {
                    "system": "System: ",
                    "user": "Human: ",
                    "assistant": "Assistant: "
                }
            elif "llama" in model_name_lower:
                format_info["roles"] = {
                    "system": "<s>[INST] ",
                    "user": "[INST] ",
                    "assistant": " [/INST]"
                }
            elif "mistral" in model_name_lower:
                format_info["roles"] = {
                    "system": "<|system|>\n",
                    "user": "<|user|>\n",
                    "assistant": "<|assistant|>\n"
                }
        
        except Exception as e:
            logger.warning(f"Failed to get model format config: {e}, using default format")
        
        return format_info

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format the prompt according to the model's requirements.
        """
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        format_info = self._get_model_prompt_format(model_name)
        
        # If model has a template, use it
        if format_info["template"]:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return self._tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Otherwise use role-based formatting
        roles = format_info["roles"]
        if system_prompt:
            formatted = f"{roles['system']}{system_prompt}\n\n{roles['user']}{prompt}\n\n{roles['assistant']}"
        else:
            formatted = f"{roles['user']}{prompt}\n\n{roles['assistant']}"
        
        return formatted
    
    def _verify_model_state(self) -> None:
        """Verify that the model is in a valid state for generation."""
        try:
            if not self._model_loaded:
                raise RuntimeError("Model not loaded")
            
            # For GGUF models using llama-cpp-python
            if hasattr(self._model, 'model_path'):
                # GGUF models are always ready if loaded
                return
            
            # For PyTorch models
            if hasattr(self._model, 'parameters'):
                model_device = next(self._model.parameters()).device
                config_device = self.config_manager.get_param("device", "cpu")
                
                if str(model_device) != config_device and config_device != "auto":
                    logger.warning(
                        f"Model is on device {model_device} but config specifies {config_device}. "
                        "This may cause issues."
                    )
            else:
                raise RuntimeError("Unknown model type - neither GGUF nor PyTorch")
            
        except Exception as e:
            raise RuntimeError(f"Model state verification failed: {e}")

    def _get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration for the model."""
        try:
            # Start with base parameters
            params = {
                "temperature": self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7),
                "top_p": self.config_manager.get_param(ModelParameter.TOP_P, 0.9),
                "top_k": 50,
                "num_return_sequences": 1,
                "do_sample": True,
                "min_new_tokens": 1,
                "max_new_tokens": self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048),
            }
            
            # Get model's generation config if available
            if hasattr(self._model, "generation_config"):
                gen_config = self._model.generation_config
                logger.debug(f"Using model's generation config: {gen_config}")
                
                # Update with model's defaults while preserving our parameters
                for k, v in gen_config.to_dict().items():
                    if k not in params:
                        params[k] = v
            
            # Ensure we have token IDs
            if self._tokenizer.pad_token_id is not None:
                params["pad_token_id"] = self._tokenizer.pad_token_id
            if self._tokenizer.eos_token_id is not None:
                params["eos_token_id"] = self._tokenizer.eos_token_id
            
            logger.debug(f"Final generation parameters: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Failed to get generation config: {e}")
            raise RuntimeError(f"Failed to get generation config: {e}")

    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text based on the prompt and optional files."""
        logger.debug("Starting generation with prompt: %s", prompt)
        if not self._model_loaded:
            logger.debug("Model not loaded, loading now...")
            self.load_model()
        
        # Get device from config
        device = self.config_manager.get_param("device", "cpu")
        logger.debug("Using device: %s", device)
            
        # Process files if provided
        if files:
            logger.debug("Processing %d files", len(files))
            try:
                processed_files = []
                for file_path in files:
                    media_input = MediaFactory.from_source(file_path)
                    processed_files.append(media_input)
                
                # For vision models, we only support one image at a time currently
                if self._model_type in ["vision_seq2seq", "llava"]:
                    if len(processed_files) > 1:
                        raise ValueError("Vision models currently support only one image at a time")
                    if not any(isinstance(f, ImageInput) for f in processed_files):
                        raise ValueError("No valid image file found in the provided files")
                    image_file = next(f for f in processed_files if isinstance(f, ImageInput))
                    image = Image.open(image_file.source)
                else:
                    # For text models, we append file contents to the prompt
                    file_contents = []
                    for file in processed_files:
                        if file.media_type != "text":
                            logger.warning(f"Skipping non-text file {file.source} for text model")
                            continue
                        with open(file.source, 'r') as f:
                            file_contents.append("\n===== " + file.source + " =========\n" + f.read() + "\n")
                    if file_contents:
                        prompt = prompt + "\n\n===== JOINT FILES ======\n" + "\n".join(file_contents)
                        
            except Exception as e:
                error_msg = f"Error processing files: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        try:
            # Verify model state
            self._verify_model_state()
            
            # Format prompt using model-specific formatting
            logger.debug("Formatting prompt...")
            formatted_prompt = self._format_prompt(prompt, system_prompt)
            logger.debug("Formatted prompt: %s", formatted_prompt)
            
            # Handle GGUF models differently
            if hasattr(self._model, 'model_path'):
                # Get generation parameters
                temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7)
                max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048)
                
                # Generate using llama-cpp-python
                completion = self._model.create_completion(
                    formatted_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
                
                if stream:
                    def response_generator():
                        for chunk in completion:
                            if 'choices' in chunk and chunk['choices']:
                                yield chunk['choices'][0]['text']
                    return response_generator()
                else:
                    result = completion['choices'][0]['text']
                    log_response("huggingface", result)
                    return result
            else:
                # Get generation parameters for PyTorch models
                params = self._get_generation_config()
                
                # Prepare inputs based on model type
                logger.debug("Preparing inputs for model type: %s", self._model_type)
                if self._model_type in ["vision_seq2seq", "llava"]:
                    inputs = self._processor(images=image, text=formatted_prompt, return_tensors="pt")
                else:
                    inputs = self._tokenizer(formatted_prompt, return_tensors="pt", padding=True)
                    logger.debug("Input shape: %s", {k: v.shape for k, v in inputs.items()})
                
                # Move inputs to the correct device
                logger.debug("Moving inputs to device: %s", device)
                inputs = self._move_inputs_to_device(inputs, device)
                
                # Generate
                logger.debug("Starting model.generate()...")
                try:
                    with torch.no_grad():
                        outputs = self._model.generate(
                            **inputs,
                            **params
                        )
                    logger.debug("Generation completed. Output shape: %s", outputs.shape)
                except Exception as gen_error:
                    logger.error("Generation failed with error: %s", str(gen_error))
                    logger.error("Model config: %s", self._model.config)
                    logger.error("Tokenizer config: %s", self._tokenizer.init_kwargs)
                    raise
                
                # Decode outputs
                logger.debug("Decoding outputs...")
                if self._model_type in ["vision_seq2seq", "llava"]:
                    generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)
                else:
                    # Only decode the new tokens, not the input
                    if hasattr(self._model.config, "max_position_embeddings"):
                        generated_text = self._tokenizer.batch_decode(
                            outputs[:, inputs["input_ids"].shape[1]:], 
                            skip_special_tokens=True
                        )
                    else:
                        generated_text = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Return first sequence if only one was requested
                result = generated_text[0] if params["num_return_sequences"] == 1 else generated_text
                
                # Log the response
                logger.debug("Generation successful. Result: %s", result)
                log_response("huggingface", result)
                return result
                
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)  # Include full traceback
            raise GenerationError(error_msg) from e
    
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None,
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False, **kwargs) -> str:
        """Generate text asynchronously."""
        # Run the synchronous generate method in a thread pool
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, 
                self.generate,
                prompt,
                files,
                **kwargs
            )
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this implementation."""
        model = self.config_manager.get_param(ModelParameter.MODEL)
        is_vision_capable = any(vm in model for vm in VISION_CAPABLE_MODELS)
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: None,  # Varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.VISION: is_vision_capable
        }
    
    @staticmethod
    def list_cached_models(cache_dir: Optional[str] = None) -> list:
        """List all models cached by this implementation."""
        if cache_dir is None:
            cache_dir = HuggingFaceProvider.DEFAULT_CACHE_DIR
            
        if cache_dir and '~' in cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            
        if not os.path.exists(cache_dir):
            return []
            
        try:
            from huggingface_hub import scan_cache_dir
                
            cache_info = scan_cache_dir(cache_dir)
            return [{
                    "name": repo.repo_id,
                    "size": repo.size_on_disk,
                "last_used": repo.last_accessed,
                "implementation": "transformers"
            } for repo in cache_info.repos]
        except ImportError:
            logger.warning("huggingface_hub not available for cache scanning")
            return []
    
    @staticmethod
    def clear_model_cache(model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        """Clear model cache for this implementation."""
        if cache_dir is None:
            cache_dir = HuggingFaceProvider.DEFAULT_CACHE_DIR
        
        if cache_dir and '~' in cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
            
        if not os.path.exists(cache_dir):
            return
                
        try:
            from huggingface_hub import delete_cache_folder
            
            if model_name:
                        delete_cache_folder(repo_id=model_name, cache_dir=cache_dir)
            else:
                delete_cache_folder(cache_dir=cache_dir)
        except ImportError:
            logger.warning("huggingface_hub not available for cache clearing")

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