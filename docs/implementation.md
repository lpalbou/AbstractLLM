# AbstractLLM Implementation Guide

This document provides a detailed explanation of how AbstractLLM is implemented, with practical code examples, implementation patterns, and technical details.

## Core Components Implementation

### Interface and Enums

The core of AbstractLLM is the abstract interface that all provider implementations must follow. The interface is defined in `interface.py` and includes:

```python
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator

class ModelParameter(str, Enum):
    """Model parameters that can be configured."""
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "system_prompt"
    TOP_P = "top_p"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    STOP = "stop"
    MODEL = "model"
    API_KEY = "api_key"
    BASE_URL = "base_url"
    # ... more parameters
    
class ModelCapability(str, Enum):
    """Capabilities that a model may support."""
    STREAMING = "streaming"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "supports_system_prompt"
    ASYNC = "supports_async" 
    FUNCTION_CALLING = "supports_function_calling"
    VISION = "supports_vision"
    # ... more capabilities

class AbstractLLMInterface(ABC):
    """Abstract interface for LLM providers."""
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the LLM provider."""
        from abstractllm.utils.config import ConfigurationManager
        self.config = config or ConfigurationManager.create_base_config()
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response to the prompt using the LLM."""
        pass

    @abstractmethod
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response to the prompt using the LLM."""
        pass
        
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM."""
        return {
            ModelCapability.STREAMING: False,
            ModelCapability.MAX_TOKENS: 2048,
            ModelCapability.SYSTEM_PROMPT: False,
            ModelCapability.ASYNC: False,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.VISION: False,
        }
        
    def set_config(self, **kwargs) -> None:
        """Update the configuration with individual parameters."""
        self.config.update(kwargs)
        
    def update_config(self, config: Dict[Union[str, ModelParameter], Any]) -> None:
        """Update the configuration with a dictionary of parameters."""
        self.config.update(config)
        
    def get_config(self) -> Dict[Union[str, ModelParameter], Any]:
        """Get the current configuration."""
        return self.config.copy()
```

### Configuration Management

The centralized configuration management is implemented in `config.py`, providing a unified approach to parameter handling:

```python
from typing import Dict, Any, Optional, Union, TypeVar, Generic, List, Set
from enum import Enum
import os
import logging

from abstractllm.interface import ModelParameter

# Configure logger
logger = logging.getLogger("abstractllm.utils.config")

# Define provider-specific default models
DEFAULT_MODELS = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-5-haiku-20241022",
    "ollama": "phi4-mini:latest",
    "huggingface": "distilgpt2"
}

# Environment variable mapping for API keys
ENV_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,  # No API key needed for Ollama
    "huggingface": "HUGGINGFACE_API_KEY"
}

# Generic value type for configuration
T = TypeVar('T')

class ConfigurationManager:
    """
    Centralized configuration management for AbstractLLM providers.
    """

    @staticmethod
    def create_base_config(**kwargs) -> Dict[Union[str, ModelParameter], Any]:
        """
        Create a configuration dictionary with default values.
        """
        # Default configuration
        config = {
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 2048,
            ModelParameter.SYSTEM_PROMPT: None,
            ModelParameter.TOP_P: 1.0,
            ModelParameter.FREQUENCY_PENALTY: 0.0,
            ModelParameter.PRESENCE_PENALTY: 0.0,
            ModelParameter.STOP: None,
            ModelParameter.MODEL: None,
            ModelParameter.API_KEY: None,
            ModelParameter.BASE_URL: None,
            ModelParameter.TIMEOUT: 120,
            ModelParameter.RETRY_COUNT: 3,
            ModelParameter.SEED: None,
            ModelParameter.LOGGING_ENABLED: True,
        }
        # Update with provided values
        config.update(kwargs)
        return config

    @staticmethod
    def initialize_provider_config(
        provider_name: str, 
        config: Optional[Dict[Union[str, ModelParameter], Any]] = None
    ) -> Dict[Union[str, ModelParameter], Any]:
        """
        Initialize provider-specific configuration with appropriate defaults.
        """
        if config is None:
            config = {}
            
        # Create a copy to avoid modifying the input
        config_copy = config.copy()
        
        # Set provider-specific defaults
        
        # 1. Default model
        if ModelParameter.MODEL not in config_copy and "model" not in config_copy:
            default_model = DEFAULT_MODELS.get(provider_name)
            if default_model:
                config_copy[ModelParameter.MODEL] = default_model
                logger.debug(f"Using default model for {provider_name}: {default_model}")
        
        # 2. API Key
        api_key = config_copy.get(ModelParameter.API_KEY, config_copy.get("api_key"))
        if not api_key:
            env_var = ENV_API_KEYS.get(provider_name)
            if env_var:
                env_api_key = os.environ.get(env_var)
                if env_api_key:
                    config_copy[ModelParameter.API_KEY] = env_api_key
                    # Import here to avoid circular import
                    from abstractllm.utils.logging import log_api_key_from_env
                    log_api_key_from_env(provider_name, env_var)
        
        # ... more provider-specific initialization
        
        return config_copy

    @staticmethod
    def get_param(
        config: Dict[Union[str, ModelParameter], Any],
        param: ModelParameter,
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Get a parameter value from configuration, supporting both enum and string keys.
        """
        # Try with enum first, then with string key
        return config.get(param, config.get(param.value, default))

    @staticmethod
    def extract_generation_params(
        provider: str,
        config: Dict[Union[str, ModelParameter], Any],
        kwargs: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract and combine parameters for generation from config and kwargs.
        """
        # Create a copy of the config and update with kwargs
        params = config.copy()
        params.update(kwargs)
        
        # Handle system prompt override
        if system_prompt is not None:
            params[ModelParameter.SYSTEM_PROMPT] = system_prompt
        
        # Extract common parameters
        result = {
            "model": ConfigurationManager.get_param(params, ModelParameter.MODEL),
            "temperature": ConfigurationManager.get_param(params, ModelParameter.TEMPERATURE, 0.7),
            "max_tokens": ConfigurationManager.get_param(params, ModelParameter.MAX_TOKENS, 2048),
            "system_prompt": ConfigurationManager.get_param(params, ModelParameter.SYSTEM_PROMPT),
            # ... more common parameters
        }
        
        # Process provider-specific parameters
        if provider == "openai":
            # Add OpenAI-specific parameters
            organization = ConfigurationManager.get_param(params, ModelParameter.ORGANIZATION)
            if organization:
                result["organization"] = organization
                
        elif provider == "huggingface":
            # Add HuggingFace-specific parameters
            result["device"] = ConfigurationManager.get_param(params, ModelParameter.DEVICE, "cpu")
            result["load_in_8bit"] = ConfigurationManager.get_param(params, "load_in_8bit", False)
            
        # ... more provider-specific parameter extraction
            
        return result
```

### Factory Pattern Implementation

The factory pattern is implemented in `factory.py`, providing a consistent way to create provider instances:

```python
from typing import Dict, Any, Optional
import importlib
from abstractllm.interface import AbstractLLMInterface, ModelParameter
from abstractllm.utils.config import ConfigurationManager

# Provider mapping
_PROVIDERS = {
    "openai": "abstractllm.providers.openai.OpenAIProvider",
    "anthropic": "abstractllm.providers.anthropic.AnthropicProvider",
    "ollama": "abstractllm.providers.ollama.OllamaProvider",
    "huggingface": "abstractllm.providers.huggingface.HuggingFaceProvider",
}

def create_llm(provider: str, **config) -> AbstractLLMInterface:
    """Create an LLM provider instance."""
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            f"Available providers: {', '.join(_PROVIDERS.keys())}"
        )
    
    # Import the provider class
    module_path, class_name = _PROVIDERS[provider].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import provider {provider}: {e}")
    
    # Create configuration with defaults and provider-specific setup
    base_config = ConfigurationManager.create_base_config(**config)
    provider_config = ConfigurationManager.initialize_provider_config(provider, base_config)
    
    # Instantiate and return the provider
    return provider_class(config=provider_config)
```

## Provider Implementations

Let's examine how specific providers are implemented:

### OpenAI Provider

The OpenAI provider (`openai.py`) implements the interface for the OpenAI API using the centralized configuration management:

```python
import os
import requests
import json
import logging
from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import log_request, log_response
from abstractllm.utils.image import preprocess_image_inputs
from abstractllm.utils.config import ConfigurationManager

logger = logging.getLogger("abstractllm.providers.openai.OpenAIProvider")

class OpenAIProvider(AbstractLLMInterface):
    """OpenAI API implementation."""
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the OpenAI provider."""
        super().__init__(config)
        
        # Initialize provider-specific configuration
        self.config = ConfigurationManager.initialize_provider_config("openai", self.config)
        
        # Log provider initialization
        model = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "gpt-3.5-turbo")
        logger.info(f"Initialized OpenAI provider with model: {model}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using the OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install it with: pip install openai"
            )
        
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "openai", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        api_key = params.get("api_key")
        model = params.get("model", "gpt-3.5-turbo")
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens")
        system_prompt = params.get("system_prompt")
        
        # Check for image inputs
        has_vision = any(model.startswith(vm) for vm in ["gpt-4-vision", "gpt-4o"])
        
        if has_vision and ("image" in params or "images" in params):
            params = preprocess_image_inputs(params, "openai")
        
        # Log the request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream
        })
        
        # Make API call and return response
        # ... implementation details
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response using the OpenAI API."""
        # Uses same configuration extraction approach as synchronous method
        params = ConfigurationManager.extract_generation_params(
            "openai", self.config, kwargs, system_prompt
        )
        
        # ... implementation details
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of the OpenAI provider."""
        # Get model name to determine capabilities
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "gpt-3.5-turbo")
        
        # Determine if the model supports vision
        supports_vision = any(model in model_name for model in [
            "gpt-4-vision-preview", "gpt-4-turbo", "gpt-4o"
        ])
        
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 4096,  # Varies by model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: True,
            ModelCapability.VISION: supports_vision
        }
```

### HuggingFace Provider

The HuggingFace provider (`huggingface.py`) implements the interface for local models using the centralized configuration management:

```python
import os
import gc
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, Tuple, ClassVar

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.config import ConfigurationManager
from abstractllm.utils.logging import log_request, log_response

logger = logging.getLogger("abstractllm.providers.huggingface.HuggingFaceProvider")

class HuggingFaceProvider(AbstractLLMInterface):
    """HuggingFace implementation using Transformers."""
    
    # Class-level model cache
    _model_cache: ClassVar[Dict[Tuple[str, str, bool, bool], Tuple[Any, Any, float]]] = {}
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the HuggingFace provider."""
        super().__init__(config)
        
        # Initialize provider-specific configuration
        self.config = ConfigurationManager.initialize_provider_config("huggingface", self.config)
        
        # Store device preference
        self._device = ConfigurationManager.get_param(self.config, ModelParameter.DEVICE, _get_optimal_device())
        
        # Initialize model and tokenizer objects to None (will be loaded on demand)
        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        
        # Log provider initialization
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "distilgpt2")
        logger.info(f"Initialized HuggingFace provider with model: {model_name}")
        
        # Preload the model if auto_load is set
        if self.config.get("auto_load", False):
            self.load_model()
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        # Check if model already loaded
        if self._model_loaded:
            return
            
        # Get model name
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "distilgpt2")
        
        # Create a cache key and check if already cached
        cache_key = self._get_cache_key()
        
        if cache_key in self._model_cache:
            logger.info(f"Loading model from cache: {model_name}")
            self._model, self._tokenizer, _ = self._model_cache[cache_key]
            # Update last access time
            self._model_cache[cache_key] = (self._model, self._tokenizer, time.time())
            self._model_loaded = True
            return
        
        # Load model from HuggingFace
        # ... implementation details
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using a HuggingFace model."""
        # Load model if not already loaded
        if not self._model_loaded:
            self.load_model()
        
        # Extract and combine parameters using the configuration manager
        params = ConfigurationManager.extract_generation_params(
            "huggingface", self.config, kwargs, system_prompt
        )
        
        # Extract key parameters
        model_name = ConfigurationManager.get_param(self.config, ModelParameter.MODEL, "distilgpt2")
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 2048)
        
        # ... implementation details
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response using a HuggingFace model."""
        # Load model if not already loaded
        if not self._model_loaded:
            # Load model in a thread to avoid blocking the event loop
            with ThreadPoolExecutor() as executor:
                await asyncio.get_event_loop().run_in_executor(
                    executor, self.load_model
                )
        
        # Run the synchronous generate method in a thread pool
        if stream:
            # Define an async generator that wraps the sync generator
            async def async_stream_wrapper():
                sync_gen = self.generate(prompt, system_prompt, stream=True, **kwargs)
                for chunk in sync_gen:
                    yield chunk
                    await asyncio.sleep(0.01)
            
            return async_stream_wrapper()
        else:
            with ThreadPoolExecutor() as executor:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, 
                    lambda: self.generate(prompt, system_prompt, stream=False, **kwargs)
                )
                return result
```

## Putting It All Together

Let's see how all of these components work together in a complete flow:

```python
from abstractllm import create_llm, ModelParameter

# 1. Create a provider instance with configuration
llm = create_llm("openai", **{
    ModelParameter.TEMPERATURE: 0.7,
    ModelParameter.MAX_TOKENS: 1000,
    ModelParameter.MODEL: "gpt-4o"
})

# 2. Generate a response
response = llm.generate(
    "Write a poem about artificial intelligence.",
    system_prompt="You are a poet who specializes in modern free verse."
)

# The internal flow:
# - Factory creates base configuration
# - Provider-specific initialization occurs
# - Generate method extracts parameters
# - API request is made
# - Response is returned
```

This design provides a clean, consistent interface while handling provider-specific details behind the scenes through centralized configuration management. 