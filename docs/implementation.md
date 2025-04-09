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

def create_config(**kwargs) -> Dict[str, Any]:
    """Create a configuration dictionary with default values."""
    # Default configuration
    config = {
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 2048,
        ModelParameter.SYSTEM_PROMPT: None,
        # ... more defaults
    }
    # Update with provided values
    config.update(kwargs)
    return config

class AbstractLLMInterface(ABC):
    """Abstract interface for LLM providers."""
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the LLM provider."""
        self.config = config or create_config()
    
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

### Factory Pattern Implementation

The factory pattern is implemented in `factory.py`, providing a consistent way to create provider instances:

```python
from typing import Dict, Any, Optional
import importlib
from abstractllm.interface import AbstractLLMInterface, ModelParameter, create_config

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
    
    # Create configuration with defaults
    provider_config = create_config(**config)
    
    # Instantiate and return the provider
    return provider_class(config=provider_config)
```

## Provider Implementations

Let's examine how specific providers are implemented:

### OpenAI Provider

The OpenAI provider (`openai.py`) implements the interface for the OpenAI API:

```python
import os
import requests
import json
import logging
from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import log_request, log_response, log_api_key_missing, log_api_key_from_env

logger = logging.getLogger("abstractllm.providers.openai.OpenAIProvider")

class OpenAIProvider(AbstractLLMInterface):
    """OpenAI API implementation."""
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the OpenAI provider."""
        super().__init__(config)
        
        # Get API key from config or environment
        self._api_key = self.config.get(ModelParameter.API_KEY, self.config.get("api_key"))
        if not self._api_key:
            # Try to get from environment
            self._api_key = os.environ.get("OPENAI_API_KEY")
            if self._api_key:
                log_api_key_from_env("OpenAI", "OPENAI_API_KEY")
            else:
                log_api_key_missing("OpenAI", "OPENAI_API_KEY")
                
        # Set default model if not specified
        if not self.config.get(ModelParameter.MODEL) and not self.config.get("model"):
            self.config[ModelParameter.MODEL] = "gpt-3.5-turbo"
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using the OpenAI API."""
        # Implementation includes:
        # 1. Combine config with kwargs
        # 2. Extract parameters (model, temperature, etc.)
        # 3. Process image inputs if present
        # 4. Prepare the API request
        # 5. Handle streaming vs. non-streaming
        # 6. Make the API call
        # 7. Process and return the response
        
        # Example of streaming implementation:
        if stream:
            def response_generator():
                # Streaming implementation
                # Yield chunks of the response as they're received
                pass
            return response_generator()
        else:
            # Non-streaming implementation
            # Return the complete response
            pass
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response using the OpenAI API."""
        # Async implementation follows similar pattern to synchronous
        # but uses aiohttp for async HTTP requests
        pass
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of the OpenAI provider."""
        # Get model name to determine capabilities
        model_name = self.config.get(ModelParameter.MODEL, self.config.get("model", "gpt-3.5-turbo"))
        
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

The HuggingFace provider (`huggingface.py`) has unique features for managing local models:

```python
import os
import gc
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator, Tuple, ClassVar, List
from pathlib import Path

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.utils.logging import log_request, log_response

logger = logging.getLogger("abstractllm.providers.huggingface.HuggingFaceProvider")

# Default model to use
DEFAULT_MODEL = "distilgpt2"

# List of vision-capable models
VISION_CAPABLE_MODELS = [
    "microsoft/Phi-4-multimodal-instruct",
    "liuhaotian/llava-phi-1.5",
    "Qwen/Qwen2-VL",
    "internlm/internlm-xcomposer2-vl",
    "deepseek-ai/deepseek-vl-7b",
    # ... more vision models
]

def _get_optimal_device() -> str:
    """Determine the best available device for model loading."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU support
        else:
            return "cpu"
    except ImportError:
        return "cpu"

class HuggingFaceProvider(AbstractLLMInterface):
    """Hugging Face implementation using Transformers."""
    
    # Default cache directory
    DEFAULT_CACHE_DIR = "~/.cache/abstractllm/models"
    
    # Class-level cache for sharing models between instances
    _model_cache: ClassVar[Dict[Tuple[str, str, bool, bool], Tuple[Any, Any, float]]] = {}
    
    # Maximum number of models to keep in the cache
    _max_cached_models = 3
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the Hugging Face provider."""
        super().__init__(config)
        
        # Set default model if not specified
        if ModelParameter.MODEL not in self.config and "model" not in self.config:
            self.config[ModelParameter.MODEL] = DEFAULT_MODEL
        
        # Determine optimal device if not specified
        self._device = self.config.get(ModelParameter.DEVICE, self.config.get("device", _get_optimal_device()))
        
        # Initialize model state
        self._model = None
        self._tokenizer = None
        self._processor = None  # For vision models
        self._model_loaded = False
        self._warmup_completed = False
        
        # Preload model if requested
        if self.config.get("auto_load", False):
            self.load_model()
            
            # Run warmup if requested
            if self.config.get("auto_warmup", False):
                self.warmup()
    
    def load_model(self) -> None:
        """Load the model and tokenizer based on the configuration."""
        # Implementation includes:
        # 1. Check if model already loaded
        # 2. Check class-level cache
        # 3. Load tokenizer and model with appropriate settings
        # 4. Handle vision-capable models specially
        # 5. Store model in cache for reuse
        
        # Example of cache key generation and lookup:
        cache_key = self._get_cache_key()
        if cache_key in HuggingFaceProvider._model_cache:
            self._model, self._tokenizer, _ = HuggingFaceProvider._model_cache[cache_key]
            # Update last access time
            HuggingFaceProvider._model_cache[cache_key] = (self._model, self._tokenizer, time.time())
            self._model_loaded = True
            return
        
        # Example of cache cleanup:
        self._clean_model_cache_if_needed()
        
        # Example of model loading logic:
        model_name = self.config.get(ModelParameter.MODEL, self.config.get("model"))
        is_vision_capable = any(vision_model in model_name for vision_model in VISION_CAPABLE_MODELS)
        
        # ... (model loading implementation)
        
        # Store in cache
        HuggingFaceProvider._model_cache[cache_key] = (self._model, self._tokenizer, time.time())
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using Hugging Face model."""
        # Implementation includes:
        # 1. Load model if not already loaded
        # 2. Process parameters and inputs
        # 3. Handle image inputs for vision models
        # 4. Set up generation configuration
        # 5. Generate response (streaming or non-streaming)
        pass
    
    # Other methods...
```

## Configuration Management

Configuration is managed through a dictionary-based approach with support for both string keys and enumerated types for backward compatibility and type safety.

Example of parameter extraction in a provider:

```python
# Combine configuration with kwargs
params = self.config.copy()
params.update(kwargs)

# Extract parameters with fallbacks for both string and enum keys
model_name = params.get(ModelParameter.MODEL, params.get("model", "default-model"))
temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_tokens", 2048))
system_prompt_from_config = params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
system_prompt = system_prompt or system_prompt_from_config
```

## Capability Inspection

Capability inspection allows clients to adapt their behavior based on provider capabilities:

```python
llm = create_llm("openai")
capabilities = llm.get_capabilities()

if capabilities.get(ModelCapability.VISION):
    # Use vision features
    response = llm.generate("What's in this image?", image="path/to/image.jpg")
else:
    # Fall back to text-only
    response = llm.generate("Please describe what might be in an image.")
```

Providers implement `get_capabilities()` to report what features they support, based on their implementation and the configured model.

## Streaming Implementation

Streaming is implemented using Python generators (synchronous) and async generators (asynchronous):

### Synchronous Streaming

```python
def generate(self, prompt: str, stream: bool = False, **kwargs):
    # ... parameter processing ...
    
    if stream:
        def response_generator():
            # Make streaming API call
            with requests.post(url, json=payload, stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        yield data["choices"][0]["text"]
        
        return response_generator()
    else:
        # Make standard API call
        response = requests.post(url, json=payload)
        return response.json()["choices"][0]["text"]
```

### Asynchronous Streaming

```python
async def generate_async(self, prompt: str, stream: bool = False, **kwargs):
    # ... parameter processing ...
    
    if stream:
        async def async_generator():
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    async for line in response.content:
                        if line:
                            data = json.loads(line)
                            yield data["choices"][0]["text"]
        
        return async_generator()
    else:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                result = await response.json()
                return result["choices"][0]["text"]
```

## Vision Capability Implementation

Vision capability is implemented by detecting image inputs and processing them appropriately for each provider:

### Image Processing Utilities

```python
def format_image_for_provider(image_input, provider):
    """Format an image for a specific provider's API format."""
    # Handle different input types (URL, file path, base64)
    # Format according to provider requirements
    if provider == "openai":
        return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
    elif provider == "anthropic":
        return {"type": "image", "source": {"type": "url", "url": url}}
    # ... etc.

def preprocess_image_inputs(params, provider):
    """Preprocess image inputs in the params dictionary for a specific provider."""
    # Extract and format image inputs
    # Add to request in provider-specific format
    # Return updated parameters
```

### Using Vision Capabilities

In provider implementation:

```python
# Check for image inputs
has_image = "image" in params or "images" in params
is_vision_capable = any(vision_model in model_name for vision_model in VISION_CAPABLE_MODELS)

if has_image and not is_vision_capable:
    logger.warning(f"Model {model_name} does not support vision inputs. Ignoring image input.")
    # Remove image inputs
elif has_image and is_vision_capable:
    # Process image inputs for vision-capable models
    params = preprocess_image_inputs(params, "provider_name")
    # Prepare vision-specific request
```

## Memory Management

Memory management is particularly important for the HuggingFace provider, which loads models into memory:

### Model Cache Implementation

```python
def _clean_model_cache_if_needed(self) -> None:
    """Clean up the model cache if it exceeds the maximum size."""
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
        model, tokenizer, _ = HuggingFaceProvider._model_cache[key]
        
        # Set models to None to help with garbage collection
        model = None
        tokenizer = None
        
        # Remove from cache
        del HuggingFaceProvider._model_cache[key]
    
    # Explicitly run garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Error Handling

Error handling is implemented throughout the codebase to provide clear feedback on failures:

```python
try:
    # Make API call
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    # Process response
    result = response.json()
    log_response("provider_name", result)
    return result
except requests.exceptions.RequestException as e:
    # Handle network or API errors
    logger.error(f"API request failed: {e}")
    if "timeout" in str(e).lower():
        raise TimeoutError(f"API request timed out after {timeout} seconds")
    else:
        raise Exception(f"API request failed: {e}")
except (ValueError, KeyError) as e:
    # Handle JSON parsing or response format errors
    logger.error(f"Error processing response: {e}")
    raise Exception(f"Error processing response: {e}")
except Exception as e:
    # Handle any other errors
    logger.error(f"Unexpected error: {e}")
    raise
```

## Logging System

The logging system is designed to provide visibility into the operation of the library while respecting security concerns:

```python
def log_request(provider: str, prompt: str, parameters: Dict[str, Any]) -> None:
    """Log an LLM request."""
    # Log basic request info at INFO level
    logger.info(f"LLM request to {provider} provider")
    
    # Log detailed request information at DEBUG level
    logger.debug(f"REQUEST [{provider}]: {datetime.now().isoformat()}")
    logger.debug(f"Parameters: {parameters}")
    logger.debug(f"Prompt: {prompt}")

def log_response(provider: str, response: str) -> None:
    """Log an LLM response."""
    # Log basic response info at INFO level
    logger.info(f"LLM response received from {provider} provider")
    
    # Log detailed response at DEBUG level
    logger.debug(f"RESPONSE [{provider}]: {datetime.now().isoformat()}")
    logger.debug(f"Response: {response}")
```

## Cross-Platform Compatibility

The library is designed to work across different platforms:

1. **Automatic device detection** based on available hardware
2. **Flexible model loading options** for different hardware constraints
3. **Dependency management** to avoid unnecessary dependencies
4. **Path handling** that works across operating systems

For example, with HuggingFace provider on Apple Silicon:

```python
def _get_optimal_device() -> str:
    """Determine the best available device for model loading."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU support
        else:
            return "cpu"
    except ImportError:
        return "cpu"
```

## Performance Considerations

1. **Lazy imports** to avoid loading unnecessary dependencies
2. **Model caching** to avoid reloading models
3. **Warmup passes** to optimize model performance
4. **Timeout handling** to prevent hanging requests
5. **Async support** for concurrent operations

## Testing Strategy

Testing is implemented with a focus on real-world behavior:

1. **Unit tests** for individual components
2. **Integration tests** for provider implementations
3. **Visual tests** for multimodal capabilities
4. **Cross-provider tests** for consistency

For example, a vision test:

```python
@pytest.mark.vision
def test_vision_capability():
    """Test vision capability detection and basic functionality."""
    # Test with a vision-capable model
    llm = create_llm("openai", **{
        ModelParameter.MODEL: "gpt-4o"
    })
    
    capabilities = llm.get_capabilities()
    assert capabilities.get(ModelCapability.VISION) is True
    
    # Test with a non-vision model
    llm = create_llm("openai", **{
        ModelParameter.MODEL: "gpt-3.5-turbo"
    })
    
    capabilities = llm.get_capabilities()
    assert capabilities.get(ModelCapability.VISION) is False
``` 