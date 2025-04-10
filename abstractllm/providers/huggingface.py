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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer
from pathlib import Path

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import log_request, log_response, log_request_url
from abstractllm.media.factory import MediaFactory
from abstractllm.exceptions import UnsupportedOperationError, ModelNotFoundError

# Configure logger
logger = logging.getLogger("abstractllm.providers.huggingface")

# Default model - small, usable by most systems
DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"  # Small but capable model

# Default timeout in seconds for generation
DEFAULT_GENERATION_TIMEOUT = 60

# Models that support vision capabilities
VISION_CAPABLE_MODELS = [
    "openai/clip-vit-base-patch32",
    "facebook/dinov2-small",
    "Salesforce/blip2-opt-2.7b",
    "llava-hf/llava-1.5-7b-hf"
]

class HuggingFaceProvider(AbstractLLMInterface):
    """
    HuggingFace implementation using the transformers library directly.
    """
    
    # Default cache directory
    DEFAULT_CACHE_DIR = "~/.cache/abstractllm/models/huggingface"
    
    # Class-level cache for sharing models between instances
    _model_cache: ClassVar[Dict[Tuple[str, str, bool, bool], Tuple[Any, Any, float]]] = {}
    _max_cached_models = 3
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the HuggingFace provider."""
        super().__init__(config)
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: DEFAULT_MODEL,
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 1024,
            ModelParameter.DEVICE: self._get_optimal_device(),
            ModelParameter.CACHE_DIR: self.DEFAULT_CACHE_DIR,
            "trust_remote_code": True,
            "load_in_8bit": True,  # Enable 8-bit quantization by default
            "load_in_4bit": False,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",  # More memory efficient attention
            "load_timeout": 300,
            "generation_timeout": DEFAULT_GENERATION_TIMEOUT,
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize model components
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._model_loaded = False
        
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
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._model_loaded:
            return
            
        try:
            # Get configuration
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            cache_dir = self.config_manager.get_param(ModelParameter.CACHE_DIR)
            device = self.config_manager.get_param(ModelParameter.DEVICE)
            trust_remote_code = self.config_manager.get_param("trust_remote_code", True)
            load_in_8bit = self.config_manager.get_param("load_in_8bit", True)
            load_in_4bit = self.config_manager.get_param("load_in_4bit", False)
            device_map = self.config_manager.get_param("device_map", "auto")
            
            # Expand cache directory
            if cache_dir and '~' in cache_dir:
                cache_dir = os.path.expanduser(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"Loading model {model_name} from HuggingFace...")
            
            # Load tokenizer first
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code
            )
            
            # Ensure the tokenizer has the necessary special tokens
            special_tokens = {
                "pad_token": "<|pad|>",
                "eos_token": "<|endoftext|>",
                "bos_token": "<|startoftext|>"
            }
            self._tokenizer.add_special_tokens(special_tokens)
            
            # Load model with appropriate configuration
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                device_map=device_map if device == "cuda" else None,
                load_in_8bit=load_in_8bit if device == "cuda" else False,
                load_in_4bit=load_in_4bit if device == "cuda" else False,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32
            )
            
            # Move model to appropriate device if not using device_map
            if device_map is None and device != "cpu":
                self._model.to(device)
            
            # Resize token embeddings if needed
            self._model.resize_token_embeddings(len(self._tokenizer))
            
            self._model_loaded = True
            logger.info(f"Successfully loaded model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelNotFoundError(
                model_name=model_name,
                reason=str(e),
                provider="huggingface"
            )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None, stream: bool = False,
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text using the model."""
        if not self._model_loaded:
            self.load_model()
        
        try:
            # Handle system prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Get generation parameters
            temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE)
            max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS)
            
            # Log request
            log_request("huggingface", prompt, {
                "model": self.config_manager.get_param(ModelParameter.MODEL),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "has_system_prompt": system_prompt is not None,
                "stream": stream
            })
            
            # Generate with standard HF model
            inputs = self._tokenizer(full_prompt, return_tensors="pt", padding=True)
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generation_config = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "bos_token_id": self._tokenizer.bos_token_id,
                "repetition_penalty": 1.3,
                "no_repeat_ngram_size": 5,
                "top_p": 0.85,
                "top_k": 40,
                "num_beams": 3,
                "early_stopping": True,
                "length_penalty": 1.0,
                "do_sample": temperature > 0.1
            }
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_config
                )
            
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            if response.startswith(full_prompt):
                response = response[len(full_prompt):].strip()
            
            # Log response
            log_response("huggingface", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
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
                system_prompt,
                files,
                stream,
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