# MLX Provider Implementation Guide

This document provides detailed implementation guidance for the MLX provider in AbstractLLM. It outlines the necessary components, considerations, and best practices for creating an efficient provider that leverages Apple's MLX framework for optimized inference on Apple Silicon devices.

## Provider Structure

The MLX provider should be implemented as a new file in the providers directory:

```
abstractllm/providers/mlx_provider.py
```

The provider will follow AbstractLLM's existing provider interface pattern while implementing MLX-specific optimizations.

## Required Dependencies

Add MLX-related dependencies to `pyproject.toml` or `setup.py` as optional dependencies:

```python
# In setup.py
extras_require={
    # Existing extras
    "mlx": ["mlx>=0.0.25", "mlx-lm>=0.0.7"],
}
```

## Provider Class Implementation

### Basic Structure

```python
"""
MLX provider for AbstractLLM.

This provider leverages Apple's MLX framework for efficient
inference on Apple Silicon devices.
"""

import os
import time
import logging
import platform
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union, Callable, Tuple, ClassVar

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import mlx_lm
    MLXLM_AVAILABLE = True
except ImportError:
    MLXLM_AVAILABLE = False

from abstractllm.interface import AbstractLLMInterface
from abstractllm.enums import ModelParameter, ModelCapability
from abstractllm.types import GenerateResponse
from abstractllm.exceptions import UnsupportedFeatureError

# Set up logger
logger = logging.getLogger("abstractllm.providers.mlx")

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
            raise ImportError("MLX is required for MLXProvider. Install with: pip install mlx")
        
        if not MLXLM_AVAILABLE:
            raise ImportError("MLX-LM is required for MLXProvider. Install with: pip install mlx-lm")
        
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
        
        # Initialize MLX components
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._is_vision_model = False
```

### Platform Detection

Implement Apple Silicon detection to ensure the provider only runs on compatible hardware:

```python
def _check_apple_silicon(self) -> None:
    """Check if running on Apple Silicon."""
    is_macos = platform.system().lower() == "darwin"
    if not is_macos:
        raise ImportError("MLX provider is only available on macOS with Apple Silicon")
    
    # Check processor architecture
    is_arm = platform.processor() == "arm"
    if not is_arm:
        raise ImportError("MLX provider requires Apple Silicon (M1/M2/M3) hardware")
```

### Model Loading with Caching

Implement model loading with both in-memory caching and HuggingFace caching:

```python
def load_model(self) -> None:
    """
    Load the MLX model and tokenizer.
    
    This method will check the cache first before loading from the source.
    """
    model_name = self.config_manager.get_param(ModelParameter.MODEL)
    cache_dir = self.config_manager.get_param("cache_dir")  # Uses HF default if None
    
    # Check in-memory cache first
    if model_name in self._model_cache:
        logger.info(f"Loading model {model_name} from in-memory cache")
        self._model, self._tokenizer, _ = self._model_cache[model_name]
        # Update last access time
        self._model_cache[model_name] = (self._model, self._tokenizer, time.time())
        self._is_loaded = True
        return
    
    # If not in memory cache, load from disk/HF
    logger.info(f"Loading model {model_name}")
    
    try:
        # Import MLX-LM utilities
        from mlx_lm.utils import load
        
        # Check if this is a vision model
        if any(x in model_name.lower() for x in ["llava", "clip", "vision"]):
            self._is_vision_model = True
            # Note: When MLX-VLM becomes more mature, add vision-specific loading here
        
        # Load the model using MLX-LM
        self._model, self._tokenizer = load(model_name, cache_dir=cache_dir)
        self._is_loaded = True
        
        # Add to in-memory cache
        self._update_model_cache(model_name)
        
        logger.info(f"Successfully loaded model {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Failed to load MLX model: {str(e)}")

def _update_model_cache(self, model_name: str) -> None:
    """Update the model cache with the current model."""
    self._model_cache[model_name] = (self._model, self._tokenizer, time.time())
    
    # Prune cache if needed
    if len(self._model_cache) > self._max_cached_models:
        # Find oldest model by last access time
        oldest_key = min(self._model_cache.keys(), 
                         key=lambda k: self._model_cache[k][2])
        logger.info(f"Removing {oldest_key} from model cache")
        del self._model_cache[oldest_key]
```

### Text Generation Implementation

Implement the core generate method to handle text generation:

```python
def generate(self, 
            prompt: str, 
            system_prompt: Optional[str] = None, 
            files: Optional[List[Union[str, Path]]] = None,
            stream: bool = False, 
            tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
            **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
    """Generate a response using the MLX model."""
    # Load model if not already loaded
    if not self._is_loaded:
        self.load_model()
    
    # Process system prompt if provided
    formatted_prompt = prompt
    if system_prompt:
        # Use model's chat template if available
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            # Construct messages in the expected format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            try:
                # Try to use HF's template application
                from transformers import AutoTokenizer
                formatted_prompt = AutoTokenizer.apply_chat_template(
                    messages, 
                    chat_template=self._tokenizer.chat_template,
                    tokenize=False
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            # Simple concatenation fallback
            formatted_prompt = f"{system_prompt}\n\n{prompt}"
    
    # Process files if provided
    if files:
        formatted_prompt = self._process_files(formatted_prompt, files)
    
    # Tools are not supported
    if tools:
        raise UnsupportedFeatureError(
            "tool_use",
            "MLX provider does not support tool use or function calling",
            provider="mlx"
        )
    
    # Get generation parameters
    temperature = kwargs.get("temperature", 
                           self.config_manager.get_param(ModelParameter.TEMPERATURE))
    max_tokens = kwargs.get("max_tokens", 
                          self.config_manager.get_param(ModelParameter.MAX_TOKENS))
    top_p = kwargs.get("top_p", 
                     self.config_manager.get_param(ModelParameter.TOP_P))
    
    # Encode prompt
    prompt_tokens = self._tokenizer.encode(formatted_prompt)
    
    # Import MLX-LM generation utilities
    from mlx_lm.utils import generate, generate_step
    
    # Handle streaming vs non-streaming
    if stream:
        return self._generate_stream(prompt_tokens, temperature, max_tokens, top_p)
    else:
        # Generate text (non-streaming)
        output = generate(
            self._model,
            self._tokenizer,
            prompt=prompt_tokens,
            temp=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        # Create response
        return GenerateResponse(
            text=output,
            model=self.config_manager.get_param(ModelParameter.MODEL),
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(self._tokenizer.encode(output)) if hasattr(self._tokenizer, "encode") else len(output.split()),
            total_tokens=len(prompt_tokens) + (len(self._tokenizer.encode(output)) if hasattr(self._tokenizer, "encode") else len(output.split()))
        )
```

### Streaming Implementation

Implement streaming support for real-time generation:

```python
def _generate_stream(self, 
                    prompt_tokens, 
                    temperature: float, 
                    max_tokens: int, 
                    top_p: float) -> Generator[GenerateResponse, None, None]:
    """Generate a streaming response."""
    import mlx.core as mx
    from mlx_lm.utils import generate_step
    
    # Convert to MLX array if not already
    if not isinstance(prompt_tokens, mx.array):
        prompt_tokens = mx.array(prompt_tokens)
    
    # Initial state
    tokens = prompt_tokens
    finish_reason = None
    
    # Generate tokens one by one
    for _ in range(max_tokens):
        next_token, _ = generate_step(
            self._model,
            tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add token to sequence
        tokens = mx.concatenate([tokens, next_token[None]])
        
        # Convert to text
        current_text = self._tokenizer.decode(tokens.tolist()[len(prompt_tokens):])
        
        # Check for EOS token
        if hasattr(self._tokenizer, "eos_token") and self._tokenizer.eos_token in current_text:
            current_text = current_text.replace(self._tokenizer.eos_token, "")
            finish_reason = "stop"
        
        # Create response chunk
        yield GenerateResponse(
            text=current_text,
            model=self.config_manager.get_param(ModelParameter.MODEL),
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(tokens) - len(prompt_tokens),
            total_tokens=len(tokens),
            finish_reason=finish_reason
        )
        
        # Stop if we reached the end
        if finish_reason:
            break
```

### Async Implementation

Implement async support by wrapping the synchronous API, since MLX doesn't have native async:

```python
async def generate_async(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None, 
                       files: Optional[List[Union[str, Path]]] = None,
                       stream: bool = False, 
                       tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                       **kwargs) -> Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]:
    """
    Asynchronously generate a response using the MLX model.
    
    This is currently a wrapper around the synchronous method as MLX doesn't provide
    native async support, but follows the required interface.
    """
    import asyncio
    
    loop = asyncio.get_event_loop()
    
    if stream:
        # For streaming, we need to convert the synchronous generator to an async one
        sync_gen = self.generate(
            prompt, system_prompt, files, stream=True, tools=tools, **kwargs
        )
        
        async def async_gen():
            for item in sync_gen:
                yield item
                # Small delay to allow other tasks to run
                await asyncio.sleep(0.001)
                
        return async_gen()
    else:
        # For non-streaming, we can just run the synchronous method in the executor
        return await loop.run_in_executor(
            None, 
            lambda: self.generate(
                prompt, system_prompt, files, stream=False, tools=tools, **kwargs
            )
        )
```

### File Processing for Vision Support

Implement basic vision capability detection and file processing:

```python
def _process_files(self, prompt: str, files: List[Union[str, Path]]) -> str:
    """Process input files and append to prompt as needed."""
    from abstractllm.media.factory import MediaFactory
    
    processed_prompt = prompt
    has_images = False
    
    for file_path in files:
        try:
            media_input = MediaFactory.from_source(file_path)
            
            if media_input.media_type == "image":
                # For now, just set flag for vision check
                has_images = True
                # Actual image will be handled later when MLX-VLM is more mature
            elif media_input.media_type == "text":
                # Append text content to prompt
                processed_prompt += f"\n\nFile content from {file_path}:\n{media_input.content}"
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Check if this is a vision model if images are present
    if has_images and not self._is_vision_model:
        raise UnsupportedFeatureError(
            "vision",
            "This model does not support vision inputs",
            provider="mlx"
        )
            
    return processed_prompt

def _is_vision_capable(self) -> bool:
    """Check if the current model supports vision."""
    return self._is_vision_model
```

### Capabilities Reporting

Implement proper capabilities reporting to indicate what the provider supports:

```python
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
```

### Cache Management Methods

Add methods for listing and clearing cached models:

```python
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
            
        # Scan the cache
        cache_info = scan_cache_dir(cache_dir)
        
        # Filter to only include MLX models
        mlx_models = [{
            "name": repo.repo_id,
            "size": repo.size_on_disk,
            "last_used": repo.last_accessed,
            "implementation": "mlx"
        } for repo in cache_info.repos if "mlx" in repo.repo_id.lower()]
        
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
    else:
        # Clear all in-memory cache
        MLXProvider._model_cache.clear()
```

## Factory Registration

Update the factory registration in `abstractllm/factory.py`:

```python
# In abstractllm/factory.py or similar location

def register_mlx_provider():
    """Register the MLX provider if available."""
    try:
        from abstractllm.providers.mlx_provider import MLXProvider
        register_provider("mlx", MLXProvider)
        return True
    except ImportError:
        return False

# Call this during initialization
register_mlx_provider()
```

## Package Updates

Add MLX to the optional dependencies in your package config:

```toml
# In pyproject.toml

[project.optional-dependencies]
# Existing dependencies...
mlx = ["mlx>=0.0.25", "mlx-lm>=0.0.7"]
```

## Testing

Create basic tests in `tests/providers/test_mlx_provider.py`:

```python
"""
Tests for the MLX provider.

These tests will only run on Apple Silicon hardware.
"""

import pytest
import platform

# Skip all tests if not on macOS with Apple Silicon
is_macos = platform.system().lower() == "darwin"
is_arm = platform.processor() == "arm" 
pytestmark = pytest.mark.skipif(
    not (is_macos and is_arm),
    reason="MLX tests require macOS with Apple Silicon"
)

# Try to import MLX, skip if not available
try:
    import mlx.core
    import mlx_lm
except ImportError:
    pytestmark = pytest.mark.skip(reason="MLX dependencies not available")

from abstractllm import create_llm, ModelParameter

def test_mlx_provider_initialization():
    """Test MLX provider initialization."""
    llm = create_llm("mlx", **{
        ModelParameter.MODEL: "mlx-community/phi-2",  # Small model for quick testing
        ModelParameter.MAX_TOKENS: 100  # Small limit for tests
    })
    
    # Check initialization
    assert llm is not None
    
    # Check model loading
    if hasattr(llm, 'load_model'):
        llm.load_model()
    
    # Check basic generation
    response = llm.generate("Hello, world!")
    assert response.text is not None
    assert len(response.text) > 0
```

## Documentation

Add documentation for the MLX provider in `docs/mlx/`:

1. Architecture overview
2. Implementation details
3. Usage examples
4. Performance considerations

## Implementation Notes

1. **Availability Checks**: Always check for MLX availability and Apple Silicon before using MLX-specific features.
2. **HuggingFace Cache Integration**: Use HuggingFace's caching mechanisms instead of implementing a separate system.
3. **Focus on Core Functionality**: Start with text generation support and add vision capabilities as MLX support matures.
4. **Error Handling**: Provide clear error messages when running on unsupported hardware or when dependencies are missing.
5. **Performance Optimization**: Use MLX's unified memory architecture for efficient inference.

This implementation guide provides the foundation for a functional MLX provider in AbstractLLM. As the MLX ecosystem evolves, additional capabilities can be added to enhance the provider's functionality. 