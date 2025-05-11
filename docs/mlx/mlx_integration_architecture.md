# MLX Integration Architecture for AbstractLLM

## Executive Summary

This document outlines the architectural approach for integrating Apple's MLX framework with AbstractLLM. It focuses on providing efficient, optimized LLM inference on Apple Silicon devices while maintaining the core AbstractLLM interfaces and design principles.

The architecture ensures that users with Apple Silicon devices can seamlessly benefit from MLX's performance optimizations without changes to their existing code, while also leveraging the Hugging Face caching infrastructure for efficient model management.

## Architectural Principles

### 1. Separation of Concerns

The integration follows these key principles:

- **Core Functionality Focus**: AbstractLLM should maintain its focus on providing a unified interface for text-based LLM interactions.
- **Specialized Libraries for Specialized Tasks**: Speech recognition (VoiceLLM) and advanced image generation (VisionLLM) should remain separate dedicated libraries.
- **Minimal Dependencies**: The core AbstractLLM package should have minimal required dependencies, with MLX support available as an optional extension.

### 2. Modular Design

The architecture is designed to be modular at multiple levels:

- **Provider Level**: MLX becomes a provider within AbstractLLM, similar to OpenAI, Anthropic, etc.
- **Media Handling**: The existing media factory pattern is leveraged for handling different media types.
- **Hugging Face Integration**: MLX models from Hugging Face utilize the existing caching infrastructure.

## Core Architecture Components

### 1. MLX Provider Implementation

The MLX provider will be implemented as a standard AbstractLLM provider, focusing on text generation capabilities with optional vision support:

```
abstractllm/
└── providers/
    ├── anthropic.py
    ├── base.py
    ├── huggingface.py
    ├── ollama.py
    ├── openai.py
    ├── registry.py
    └── mlx_provider.py  # New MLX provider
```

The provider will follow AbstractLLM's standard pattern:

1. **Extend AbstractLLMInterface**: Implement the required methods including `generate()` and `generate_async()`
2. **Conditional Imports**: Import MLX dependencies conditionally to prevent errors on non-Apple platforms
3. **Platform Detection**: Check for Apple Silicon before activating MLX-specific features
4. **Error Handling**: Use AbstractLLM's exception hierarchy for consistent error reporting

The MLX provider will support:
- Text generation with various MLX-optimized models
- Streaming responses
- System prompts
- Basic vision capabilities through media handling
- Seamless fallback to CPU when not on Apple Silicon

### 2. Media Handling Integration

The MLX provider will leverage AbstractLLM's existing media handling system:

```python
def _process_files(self, prompt: str, files: List[Union[str, Path]]) -> str:
    """Process input files and append to prompt as needed."""
    from abstractllm.media.factory import MediaFactory
    
    processed_prompt = prompt
    has_images = False
    
    for file_path in files:
        media_input = MediaFactory.from_source(file_path)
        
        if media_input.media_type == "image":
            # For vision-capable models
            has_images = True
            # Process image as needed for MLX models
        elif media_input.media_type == "text":
            # Append text content
            processed_prompt += f"\n\nFile content: {media_input.content}"
    
    # Check if this is a vision model if images are present
    if has_images and not self._is_vision_capable():
        raise UnsupportedFeatureError(
            "vision",
            "This model does not support vision inputs",
            provider="mlx"
        )
            
    return processed_prompt
```

This integration leverages the existing media handling system without requiring MLX-specific extensions.

### 3. Factory Registration

The MLX provider will be registered with AbstractLLM's factory system following the pattern established in the codebase:

```python
# In abstractllm/providers/__init__.py or a registration module
try:
    from abstractllm.providers.mlx_provider import MLXProvider
    from abstractllm.providers.registry import register_provider
    
    # Register the provider
    register_provider("mlx", "abstractllm.providers.mlx_provider", "MLXProvider")
except ImportError:
    # MLX is not available, skip registration
    pass
```

This approach ensures the provider is only registered when MLX dependencies are available.

## Model Management

### 1. Hugging Face Cache Integration

The MLX provider will leverage the existing Hugging Face caching infrastructure rather than implementing a separate caching system:

```python
# MLX provider will use HuggingFace cache
DEFAULT_CACHE_DIR = "~/.cache/huggingface"

def _load_model(self) -> None:
    """Load the MLX model and tokenizer."""
    from mlx_lm.utils import load
    
    model_name = self.config_manager.get_param(ModelParameter.MODEL)
    cache_dir = self.config_manager.get_param("cache_dir", None)  # Will use HF default if None
    
    self._model, self._tokenizer = load(model_name, cache_dir=cache_dir)
    self._is_loaded = True
```

Benefits of this approach:
- Consistency with existing caching mechanisms
- Reduced disk space usage by avoiding duplicate caches
- Simplified model management

### 2. In-Memory Model Caching

For efficient model switching, the MLX provider will implement lightweight in-memory caching following the pattern observed in other AbstractLLM providers:

```python
# Class-level model cache
_model_cache: ClassVar[Dict[str, Tuple[Any, Any, float]]] = {}
_max_cached_models = 2  # Conservative default to manage memory

def _update_model_cache(self, model_name: str) -> None:
    """Update the model cache with the current model."""
    self._model_cache[model_name] = (self._model, self._tokenizer, time.time())
    
    # Prune cache if needed
    if len(self._model_cache) > self._max_cached_models:
        # Find oldest model by last access time
        oldest_key = min(self._model_cache.keys(), 
                         key=lambda k: self._model_cache[k][2])
        del self._model_cache[oldest_key]
```

This allows multiple models to remain loaded simultaneously, reducing latency when switching between models.

## Implementation Details

### 1. Configuration Management

The MLX provider will use AbstractLLM's `ConfigurationManager` for consistent parameter handling:

```python
def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
    """Initialize the MLX provider."""
    super().__init__(config)
    
    # Set default configuration
    default_config = {
        ModelParameter.MODEL: "mlx-community/qwen2.5-coder-14b-instruct-abliterated",
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 4096,
        ModelParameter.TOP_P: 0.9,
        "cache_dir": None,  # Use default HuggingFace cache
        "quantize": True    # Use quantized models by default
    }
    
    # Merge defaults with provided config
    self.config_manager.merge_with_defaults(default_config)
```

### 2. Error Handling

The MLX provider will use AbstractLLM's exception hierarchy for consistent error reporting:

```python
def _check_apple_silicon(self) -> None:
    """Check if running on Apple Silicon."""
    is_macos = platform.system().lower() == "darwin"
    if not is_macos:
        raise UnsupportedFeatureError(
            feature="mlx",
            message="MLX provider is only available on macOS with Apple Silicon",
            provider="mlx"
        )
    
    # Check processor architecture
    is_arm = platform.processor() == "arm"
    if not is_arm:
        raise UnsupportedFeatureError(
            feature="mlx",
            message="MLX provider requires Apple Silicon (M1/M2/M3) hardware",
            provider="mlx"
        )
```

### 3. Capability Reporting

The MLX provider will accurately report its capabilities through the standard AbstractLLM mechanism:

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
        ModelCapability.VISION: self._is_vision_capable,
    }
    
    return capabilities
```

### 4. Optional Dependencies

MLX-related dependencies will be added as optional dependencies in the package configuration:

```python
# In setup.py or pyproject.toml
extras_require={
    # Existing extras
    "mlx": ["mlx>=0.25.0", "mlx-lm>=0.0.7"],
}
```

## Vision Capabilities

For vision capabilities within MLX models:

1. **Use Existing Media Factory**: The current AbstractLLM media factory will process images for vision-capable MLX models, ensuring consistent behavior across providers.

2. **Capability Detection**: Models that support vision will report this capability through the standard AbstractLLM capability reporting system.

3. **Vision Model Detection**: The provider will detect vision-capable models based on model name patterns:

```python
def _is_vision_capable(self) -> bool:
    """Check if the current model supports vision."""
    model_name = self.config_manager.get_param(ModelParameter.MODEL, "").lower()
    return any(name in model_name for name in ["llava", "vision", "clip", "blip"])
```

## Auto-Detection and Fallback

The MLX provider will include intelligent handling for various environments:

1. **Apple Silicon Detection**: Automatically detect if running on Apple Silicon and provide helpful error messages if not.

2. **Model Availability**: Check for models optimized for MLX and fall back to compatible alternatives if needed.

3. **Graceful Degradation**: When MLX dependencies aren't available, provide clear guidance on installation.

## Integration Examples

### Basic Text Generation

```python
from abstractllm import create_llm

# Create MLX-based LLM (automatically uses Apple Silicon if available)
llm = create_llm("mlx", model="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX")

# Generate text (identical API to other providers)
response = llm.generate("Explain the benefits of MLX on Apple Silicon")
print(response.text)
```

### Vision Model Usage

```python
from abstractllm import create_llm, ModelCapability

# Create MLX-based vision-capable LLM
llm = create_llm("mlx", model="mlx-community/llava-1.5-7b-mlx")

# Check if vision is supported
if llm.get_capabilities().get(ModelCapability.VISION):
    # Use existing AbstractLLM image handling
    response = llm.generate(
        "What's in this image?", 
        files=["path/to/image.jpg"]
    )
    print(response.text)
```

## Rationale for Design Decisions

### 1. Using Hugging Face Cache

By leveraging the existing Hugging Face cache infrastructure, we:
- Avoid redundant downloads and storage
- Benefit from the maturity of HF's caching system
- Ensure consistent behavior across AbstractLLM providers

### 2. Focusing on Core LLM Functionality

By keeping the focus on optimized text and vision model inference:
- The implementation remains manageable
- We avoid scope creep into areas better handled by specialized libraries
- Users benefit from performance gains without architectural complexity

### 3. Maintaining Provider Interface Consistency

By following the established AbstractLLM provider interface:
- Users can switch between providers with minimal code changes
- Existing features (streaming, system prompts, etc.) work as expected
- Integration with other AbstractLLM features remains seamless 