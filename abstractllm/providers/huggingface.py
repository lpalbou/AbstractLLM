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
import json

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

from abstractllm.media.interface import MediaInput

# Configure logger
logger = logging.getLogger("abstractllm.providers.huggingface")

# Default timeout in seconds for generation
DEFAULT_GENERATION_TIMEOUT = 60

# Models that support vision capabilities with their specific architectures
VISION_CAPABLE_MODELS = {
    "Salesforce/blip-image-captioning-base": "vision_seq2seq",
    "Salesforce/blip-image-captioning-large": "vision_seq2seq",
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
            ModelParameter.MODEL: "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_L.gguf",
            # ModelParameter.MODEL: "microsoft/Phi-4-mini-instruct",
            # ModelParameter.MODEL: "ibm-granite/granite-3.2-2b-instruct",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 2048,
            ModelParameter.DEVICE: self._get_optimal_device(),
            ModelParameter.TIMEOUT: 60,
            "generation_timeout": DEFAULT_GENERATION_TIMEOUT,
            "load_timeout": 300,
            "trust_remote_code": True,
            "load_in_8bit": False,  # Enable 8-bit quantization by default
            "load_in_4bit": True,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",  # More memory efficient attention
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            # Add new quantization parameters with defaults
            "quantized_model": False,
            "quantization_type": None
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize model components with clear state
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._model_loaded = False
        self._model_type = None  # Will be set during loading
        
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
            
            # Create a temporary file first
            temp_path = local_path + ".tmp"
            downloaded = 0
            last_log_time = time.time()
            log_interval = 5  # Log every 5 seconds
            
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
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
            actual_size = os.path.getsize(temp_path)
            if actual_size != expected_size:
                os.remove(temp_path)
                raise RuntimeError(
                    f"Downloaded file size ({actual_size}) does not match expected size ({expected_size})"
                )

            # Move temporary file to final location
            os.replace(temp_path, local_path)
            logger.info(f"Model successfully downloaded to {local_path}")
            return local_path

        except Exception as e:
            # Clean up temporary file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
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
            
            # Get model parameters from config
            context_size = self.config_manager.get_param("context_size", 2048)
            n_threads = self.config_manager.get_param("n_threads", os.cpu_count())
            
            # Initialize model with appropriate parameters
            model = Llama(
                model_path=model_path,
                n_ctx=context_size,  # Context window
                n_threads=n_threads,  # CPU threads
                n_gpu_layers=n_gpu_layers,  # GPU acceleration if enabled
                seed=self.config_manager.get_param("seed", 0),  # Random seed
                verbose=self.config_manager.get_param("verbose", False)  # Logging
            )
            
            # Extract model metadata
            metadata = {}
            if hasattr(model, 'metadata'):
                metadata = model.metadata  # Access as property, not method
                logger.debug(f"GGUF model metadata: {metadata}")
                
                # Update configuration based on metadata
                if "general.name" in metadata:
                    logger.info(f"Model name from metadata: {metadata['general.name']}")
                if "phi3.context_length" in metadata:
                    context_size = int(metadata["phi3.context_length"])
                    logger.info(f"Context length from metadata: {context_size}")
            
            # Create a simple tokenizer wrapper to match HF interface
            class GGUFTokenizer:
                def __init__(self, model, metadata=None):
                    self.model = model
                    self.metadata = metadata or {}
                    
                    # Get tokens from metadata if available
                    self.eos_token = self.metadata.get("tokenizer.ggml.eos_token", "</s>")
                    self.pad_token = self.metadata.get("tokenizer.ggml.pad_token", "</s>")
                    self.bos_token = self.metadata.get("tokenizer.ggml.bos_token", "<s>")
                    
                    # Get template from metadata if available
                    self.chat_template = self.metadata.get("tokenizer.chat_template")
                
                def encode(self, text, **kwargs):
                    return self.model.tokenize(text.encode())
                
                def decode(self, tokens, **kwargs):
                    return self.model.detokenize(tokens).decode()
            
            # Only set model and tokenizer after successful initialization
            self._model = model
            self._tokenizer = GGUFTokenizer(model, metadata)
            self._model_type = "gguf"  # Set model type for GGUF models
            self._model_loaded = True  # Mark as successfully loaded
            
            logger.info("Successfully loaded GGUF model")
            
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is required for loading GGUF models. "
                "Install it with: pip install llama-cpp-python"
            )
        except Exception as e:
            # Reset state on failure
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
            raise RuntimeError(f"Failed to load GGUF model: {str(e)}")

    def load_model(self) -> None:
        """
        Load a model from HuggingFace or a direct URL.
        
        Args:
            model_name: Model name or URL
            
        Raises:
            RuntimeError: If loading fails
        """
        try:
            # Reset state before loading
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
            
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            device = self.config_manager.get_param("device", "cpu")
            
            # Check if this is a direct URL
            if self._is_direct_url(model_name):
                # For GGUF models from URLs, download and use local path
                if model_name.endswith('.gguf'):
                    local_path = self._download_model(model_name)
                    logger.info(f"Loading GGUF model from {local_path}")
                    self._load_gguf_model(local_path)
                    return
                
                # For other URLs, let HuggingFace handle it
                logger.warning(
                    "Loading models directly from URLs will be deprecated in a future version. "
                    "Please use model IDs or local paths instead."
                )
            
            # For local GGUF files
            if model_name.endswith('.gguf'):
                if os.path.exists(model_name):
                    logger.info(f"Loading local GGUF model from {model_name}")
                    self._load_gguf_model(model_name)
                    return
                else:
                    raise RuntimeError(f"GGUF model file not found: {model_name}")
            
            # For HuggingFace models
            logger.info(f"Loading model {model_name} using transformers")
            
            # Get quantization parameters
            load_in_4bit = self.config_manager.get_param("load_in_4bit", False)
            load_in_8bit = self.config_manager.get_param("load_in_8bit", False)
            
            if load_in_4bit or load_in_8bit:
                logger.warning(
                    "On-the-fly quantization can take several minutes. "
                    "Consider using a pre-quantized model for faster loading."
                )
                start_time = time.time()
            
            try:
                # Prepare model loading parameters
                model_kwargs = {
                    "trust_remote_code": self.config_manager.get_param("trust_remote_code", True),
                    "torch_dtype": self.config_manager.get_param("torch_dtype", "auto"),
                    "low_cpu_mem_usage": self.config_manager.get_param("low_cpu_mem_usage", True)
                }
                
                # Add device mapping if not CPU
                if device != "cpu":
                    model_kwargs["device_map"] = "auto"
                
                # Add quantization parameters if specified
                if load_in_4bit:
                    model_kwargs["load_in_4bit"] = True
                elif load_in_8bit:
                    model_kwargs["load_in_8bit"] = True
                
                # Load the model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                
                # Move model to device if not using device_map="auto"
                if device != "cpu" and "device_map" not in model_kwargs:
                    model = model.to(device)
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Only set model and tokenizer after successful loading
                self._model = model
                self._tokenizer = tokenizer
                self._model_type = "causal_lm"  # Set default type for HF models
                self._model_loaded = True  # Mark as successfully loaded
                
                if load_in_4bit or load_in_8bit:
                    elapsed = time.time() - start_time
                    logger.info(f"Quantization completed in {elapsed:.1f} seconds")
                
                logger.info("Successfully loaded model and tokenizer")
                
            except Exception as e:
                # Reset state on failure
                self._model = None
                self._tokenizer = None
                self._model_loaded = False
                raise RuntimeError(f"Failed to load HuggingFace model: {str(e)}")
            
        except Exception as e:
            # Reset state on any failure
            self._model = None
            self._tokenizer = None
            self._model_loaded = False
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _move_inputs_to_device(self, inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Move input tensors to the specified device."""
        if device == "cpu":
            return inputs
        return {k: v.to(device) for k, v in inputs.items()}
    
    def _get_model_prompt_format(self, model_name: str) -> Dict[str, Any]:
        """
        Get the model's prompt format configuration.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Dict containing prompt format configuration
        """
        # Default format configuration
        format_info = {
            "template": None,
            "roles": {
                "system": "System: ",
                "user": "Human: ",
                "assistant": "Assistant: "
            }
        }
        
        try:
            # Check if this is a GGUF model
            if model_name.endswith('.gguf'):
                # For GGUF models, check if we have metadata with a chat template
                if hasattr(self._model, 'metadata'):
                    metadata = self._model.metadata  # Access as property, not method
                    if 'tokenizer.chat_template' in metadata:
                        format_info["template"] = metadata['tokenizer.chat_template']
                        return format_info
                    
                    # If no template in metadata, try to infer from model name
                    model_name_lower = model_name.lower()
                    if 'phi' in model_name_lower:
                        format_info["roles"] = {
                            "system": "System: ",
                            "user": "Instruct: ",
                            "assistant": "Output: "
                        }
                    elif 'llama' in model_name_lower:
                        format_info["roles"] = {
                            "system": "<s>[INST] ",
                            "user": "[INST] ",
                            "assistant": "[/INST]"
                        }
                    elif 'mistral' in model_name_lower:
                        format_info["roles"] = {
                            "system": "<s>[INST] ",
                            "user": "[INST] ",
                            "assistant": "[/INST]"
                        }
                return format_info
            
            # For HuggingFace models, try to get config
            try:
                config_path = os.path.join(
                    os.path.dirname(model_name), "config.json"
                )
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if 'prompt_format' in config:
                            format_info["template"] = config['prompt_format']
                            return format_info
            except Exception as e:
                logger.debug(f"Could not load model config: {e}")
            
            # If no config found or no prompt format in config,
            # try to infer from model name
            model_name_lower = model_name.lower()
            if 'phi' in model_name_lower:
                format_info["roles"] = {
                    "system": "System: ",
                    "user": "Instruct: ",
                    "assistant": "Output: "
                }
            elif 'llama' in model_name_lower:
                format_info["roles"] = {
                    "system": "<s>[INST] ",
                    "user": "[INST] ",
                    "assistant": "[/INST]"
                }
            elif 'mistral' in model_name_lower:
                format_info["roles"] = {
                    "system": "<s>[INST] ",
                    "user": "[INST] ",
                    "assistant": "[/INST]"
                }
            
            return format_info
            
        except Exception as e:
            logger.warning(f"Error getting prompt format: {e}, using default format")
            return format_info

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Format the prompt according to the model's requirements.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        try:
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            format_info = self._get_model_prompt_format(model_name)
            
            # If model has a chat template and we have a tokenizer that supports it
            if (format_info["template"] and hasattr(self._tokenizer, "apply_chat_template")):
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                return self._tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Otherwise use role-based formatting
            roles = format_info["roles"]
            if system_prompt:
                formatted = f"{roles['system']}{system_prompt}\n\n{roles['user']}{prompt}\n{roles['assistant']}"
            else:
                formatted = f"{roles['user']}{prompt}\n{roles['assistant']}"
            
            return formatted.strip()
            
        except Exception as e:
            logger.warning(f"Error formatting prompt: {e}, using basic format")
            # Fallback to basic format
            if system_prompt:
                return f"System: {system_prompt}\n\nHuman: {prompt}\nAssistant:"
            return f"Human: {prompt}\nAssistant:"
    
    def _verify_model_state(self) -> None:
        """Verify that the model is in a valid state for generation."""
        try:
            if not self._model_loaded or self._model is None:
                raise RuntimeError("Model not loaded or initialization incomplete")
            
            # For GGUF models
            if self._model_type == "gguf":
                if not hasattr(self._model, 'model_path'):
                    raise RuntimeError("GGUF model not properly initialized")
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
                raise RuntimeError(f"Unknown model type: {self._model_type}")
            
        except Exception as e:
            # Reset state on verification failure
            self._model_loaded = False
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
                "max_new_tokens": self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048),
            }
            
            # Get model's generation config if available
            if hasattr(self._model, "generation_config"):
                gen_config = self._model.generation_config
                logger.debug(f"Using model's generation config: {gen_config}")
                
                # Update with model's defaults while preserving our parameters
                for k, v in gen_config.to_dict().items():
                    if k not in params and k not in ["max_length", "min_length"]:  # Skip length params
                        params[k] = v
            
            # Ensure we have token IDs
            if hasattr(self._tokenizer, "pad_token_id") and self._tokenizer.pad_token_id is not None:
                params["pad_token_id"] = self._tokenizer.pad_token_id
            if hasattr(self._tokenizer, "eos_token_id") and self._tokenizer.eos_token_id is not None:
                params["eos_token_id"] = self._tokenizer.eos_token_id
            
            logger.debug(f"Final generation parameters: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Failed to get generation config: {e}")
            raise RuntimeError(f"Failed to get generation config: {e}")

    def _process_media_input(self, media_input: MediaInput) -> Any:
        """
        Process a media input for HuggingFace models.
        
        Args:
            media_input: MediaInput instance to process
            
        Returns:
            Processed input suitable for the model
            
        Raises:
            ImageProcessingError: If there's an error processing an image
            FileProcessingError: If there's an error processing a file
        """
        try:
            formatted = media_input.to_provider_format("huggingface")
            
            if formatted["type"] == "text":
                logger.debug(f"Processing text input with MIME type: {formatted['mime_type']}")
                # For text, we'll return the raw content to be combined with the prompt
                return {
                    "type": "text",
                    "content": formatted["content"]
                }
            
            elif formatted["type"] == "tabular":
                logger.debug(f"Processing tabular input with MIME type: {formatted['mime_type']}")
                # For tabular data, return the formatted table
                return {
                    "type": "text",
                    "content": formatted["content"]
                }
            
            elif formatted["type"] == "image":
                logger.debug(f"Processing image input with source type: {formatted['source_type']}")
                try:
                    from PIL import Image
                    import requests
                    from io import BytesIO
                    
                    # Get image data based on source type
                    if formatted["source_type"] == "path":
                        image = Image.open(formatted["content"])
                    elif formatted["source_type"] == "url":
                        response = requests.get(formatted["content"])
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                    else:  # binary
                        image = Image.open(BytesIO(formatted["content"]))
                    
                    # Process image based on model type
                    if self._model_type in ["vision_seq2seq", "llava"]:
                        return {
                            "type": "image",
                            "pixel_values": self._processor(images=image, return_tensors="pt")["pixel_values"]
                        }
                    else:
                        raise UnsupportedFeatureError(
                            "vision",
                            "Current model does not support vision input",
                            provider="huggingface"
                        )
                        
                except ImportError as e:
                    raise ImageProcessingError(
                        "Required packages not available. Install with: pip install Pillow requests",
                        provider="huggingface",
                        original_exception=e
                    )
                except Exception as e:
                    raise ImageProcessingError(
                        f"Failed to process image: {str(e)}",
                        provider="huggingface",
                        original_exception=e
                    )
            
            else:
                raise ValueError(f"Unsupported media type: {formatted['type']}")
            
        except Exception as e:
            if isinstance(e, (ImageProcessingError, UnsupportedFeatureError)):
                raise
            raise FileProcessingError(
                f"Failed to process media input: {str(e)}",
                provider="huggingface",
                original_exception=e
            )

    def _format_file_content(self, text_inputs: List[Dict[str, str]], prompt: str) -> str:
        """
        Format file content with the prompt in a way that's clear for the model.
        
        Args:
            text_inputs: List of text inputs to format
            prompt: The original prompt
            
        Returns:
            Formatted prompt with file content
        """
        # Start with the user's question
        formatted = prompt.strip()
        
        # If we have file content, add it in a clear structured way
        if text_inputs:
            formatted += "\n\nHere is the content of the file(s):\n"
            formatted += "```\n"  # Use markdown-style code blocks
            for text_input in text_inputs:
                formatted += text_input["content"].strip()
            formatted += "\n```\n\n"
            # Repeat the question to make it clear what we want
            formatted += f"Based on this content, {prompt}"
        
        return formatted

    def _combine_inputs(self, processed_inputs: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
        """
        Combine processed inputs with the prompt.
        
        Args:
            processed_inputs: List of processed media inputs
            prompt: The text prompt
            
        Returns:
            Combined input suitable for the model
            
        Raises:
            ValueError: If inputs cannot be combined
        """
        # Handle different cases based on input types
        text_inputs = [p for p in processed_inputs if p["type"] == "text"]
        image_inputs = [p for p in processed_inputs if p["type"] == "image"]
        
        # Format the prompt with file content
        formatted_prompt = self._format_file_content(text_inputs, prompt)
        logger.debug(f"Formatted prompt with file content: {formatted_prompt}")
        
        # For GGUF models, handle differently
        if hasattr(self._model, 'model_path'):
            return formatted_prompt
        
        # For vision models
        if image_inputs:
            if len(image_inputs) > 1:
                raise ValueError("Multiple images are not supported for this model")
            if self._model_type in ["vision_seq2seq", "llava"]:
                # For vision models, we need both the image and the prompt
                return {
                    "pixel_values": image_inputs[0]["pixel_values"],
                    "input_ids": self._tokenizer(formatted_prompt, return_tensors="pt", 
                                              truncation=True, max_length=512)["input_ids"]
                }
        
        # For text-only inputs with regular HF models
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": True,
            "max_length": self.config_manager.get_param("max_length", 2048),
            "add_special_tokens": True  # Ensure special tokens are added
        }
        
        # Add model-specific handling
        if hasattr(self._tokenizer, "pad_token") and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Tokenize with attention mask
        inputs = self._tokenizer(formatted_prompt, **tokenizer_kwargs)
        
        # Ensure we have attention mask
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        return inputs

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
        processed_inputs = []
        if files:
            logger.debug("Processing %d files", len(files))
            try:
                for file_path in files:
                    media_input = MediaFactory.from_source(file_path)
                    processed = self._process_media_input(media_input)
                    processed_inputs.append(processed)
                    logger.debug(f"Processed file {file_path}")
            except Exception as e:
                raise FileProcessingError(
                    f"Failed to process files: {str(e)}",
                    provider="huggingface",
                    original_exception=e
                )

        try:
            # Verify model state
            self._verify_model_state()
            
            # Format prompt using model-specific formatting
            logger.debug("Formatting prompt...")
            formatted_prompt = self._format_prompt(prompt, system_prompt)
            logger.debug("Formatted prompt: %s", formatted_prompt)
            
            # Get model name and temperature for logging
            model = self.config_manager.get_param(ModelParameter.MODEL)
            temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7)
            max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048)
            
            # Log request before generation
            log_request("huggingface", prompt, {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "has_system_prompt": system_prompt is not None,
                "stream": stream,
                "has_files": bool(files),
                "model_type": self._model_type
            })
            
            # Handle GGUF models differently
            if hasattr(self._model, 'model_path'):
                # Combine prompt with processed inputs
                if processed_inputs:
                    formatted_prompt = self._combine_inputs(processed_inputs, formatted_prompt)
                    logger.debug(f"Combined prompt for GGUF: {formatted_prompt}")
                
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
                
                # Prepare inputs based on model type and processed files
                logger.debug("Preparing inputs for model type: %s", self._model_type)
                if processed_inputs:
                    # Combine prompt with processed inputs
                    inputs = self._combine_inputs(processed_inputs, formatted_prompt)
                else:
                    # Just tokenize the prompt
                    inputs = self._tokenizer(formatted_prompt, return_tensors="pt", padding=True)
                
                logger.debug("Input shape: %s", {k: v.shape for k, v in inputs.items()})
                
                # Move inputs to the same device as model
                model_device = next(self._model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                logger.debug(f"Moved inputs to model device: {model_device}")
                
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