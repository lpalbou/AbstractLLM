# Migration Guide: HuggingFace Provider

This guide helps you migrate from the old HuggingFace provider implementation to the new pipeline-based system.

## Overview of Changes

The new HuggingFace provider implementation introduces several improvements:
1. Pipeline-based architecture for better model handling
2. Improved configuration system
3. Enhanced capability detection
4. Better resource management
5. Model recommendations
6. Comprehensive system requirement checks

## Key Changes

### 1. Configuration System

Old implementation:
```python
provider = HuggingFaceProvider({
    "model": "microsoft/phi-2",
    "temperature": 0.7,
    "max_tokens": 2048
})
```

New implementation:
```python
from abstractllm.enums import ModelParameter

provider = HuggingFaceProvider({
    ModelParameter.MODEL: "microsoft/phi-2",
    ModelParameter.TEMPERATURE: 0.7,
    ModelParameter.MAX_TOKENS: 2048,
    "device_map": "auto",  # Device configuration
    "torch_dtype": "auto",  # Data type optimization
    "use_flash_attention": True  # Performance optimization
})
```

### 2. Model Loading

Old implementation:
```python
provider = HuggingFaceProvider(model="microsoft/phi-2")
provider.load()  # Manual loading
```

New implementation:
```python
provider = HuggingFaceProvider({
    ModelParameter.MODEL: "microsoft/phi-2"
})
# Model loads automatically on first use
response = provider.generate("Hello")
```

### 3. Resource Management

Old implementation:
```python
provider = HuggingFaceProvider({
    "model": "microsoft/phi-2",
    "device": "cuda"
})
```

New implementation:
```python
provider = HuggingFaceProvider({
    ModelParameter.MODEL: "microsoft/phi-2",
    "device_map": "cuda",
    "max_memory": {
        "cuda:0": "4GiB",  # GPU memory limit
        "cpu": "8GiB"      # CPU memory limit
    }
})
```

### 4. Media Handling

Old implementation:
```python
provider.generate("Describe this image", image="image.jpg")
```

New implementation:
```python
provider.generate(
    "Describe this image",
    files=["image.jpg"]  # Supports multiple files
)
```

### 5. Capability Detection

Old implementation:
```python
if provider.supports_vision:
    # Handle vision tasks
```

New implementation:
```python
capabilities = provider.get_capabilities()
if "image_to_text" in capabilities:
    # Handle vision tasks with confidence scores
    confidence = capabilities["image_to_text"].confidence
```

### 6. Model Recommendations

New feature:
```python
# Get recommended models for a task
recommendations = provider.get_model_recommendations("text-generation")
for rec in recommendations:
    print(f"Model: {rec['model']}")
    print(f"Description: {rec['description']}")

# Update recommendations
provider.update_model_recommendations("text-generation", [
    ("my-model", "Custom model description")
])
```

## Breaking Changes

1. **Configuration Parameters**:
   - Use `ModelParameter` enum for standard parameters
   - Device configuration moved to `device_map`
   - Added type validation for parameters

2. **Model Loading**:
   - Removed explicit `load()` method
   - Automatic loading on first use
   - Added system requirement checks

3. **Media Handling**:
   - Replaced individual media parameters with `files` list
   - Added support for multiple media inputs
   - Improved media type detection

4. **Error Handling**:
   - More specific error types
   - Added resource-related errors
   - Better error messages with details

## Migration Steps

1. **Update Dependencies**:
   ```bash
   pip install --upgrade abstractllm
   pip install torch>=2.1.0  # For Flash Attention 2 support
   ```

2. **Update Configuration**:
   - Replace string parameters with `ModelParameter` enum
   - Add resource limits if needed
   - Configure device mapping

3. **Update Media Handling**:
   - Replace `image`, `document` parameters with `files`
   - Update media processing code

4. **Update Error Handling**:
   - Catch new error types
   - Handle resource errors
   - Add cleanup in finally blocks

5. **Optional Improvements**:
   - Add model recommendations
   - Use capability detection
   - Implement resource limits

## Example: Complete Migration

Old code:
```python
from abstractllm import HuggingFaceProvider

provider = HuggingFaceProvider({
    "model": "microsoft/phi-2",
    "temperature": 0.7,
    "device": "cuda"
})

provider.load()

try:
    response = provider.generate(
        "Describe this image",
        image="image.jpg"
    )
    print(response)
finally:
    provider.cleanup()
```

New code:
```python
from abstractllm import HuggingFaceProvider
from abstractllm.enums import ModelParameter
from abstractllm.exceptions import ResourceError

provider = HuggingFaceProvider({
    ModelParameter.MODEL: "microsoft/phi-2",
    ModelParameter.TEMPERATURE: 0.7,
    "device_map": "cuda",
    "max_memory": {"cuda:0": "4GiB"},
    "use_flash_attention": True
})

try:
    # Get recommendations
    recs = provider.get_model_recommendations("vision")
    print(f"Recommended model: {recs[0]['model']}")
    
    # Check capabilities
    caps = provider.get_capabilities()
    if "image_to_text" not in caps:
        raise ValueError("Model does not support vision tasks")
    
    # Generate with improved media handling
    response = provider.generate(
        "Describe this image",
        files=["image.jpg"]
    )
    print(response)
    
except ResourceError as e:
    print(f"Resource error: {e.details}")
except Exception as e:
    print(f"Error: {e}")
finally:
    provider.cleanup()
```

## Best Practices

1. **Resource Management**:
   - Always set memory limits
   - Use `cleanup()` in finally blocks
   - Monitor resource usage

2. **Configuration**:
   - Use `ModelParameter` enum
   - Set appropriate device mapping
   - Enable optimizations when possible

3. **Error Handling**:
   - Catch specific error types
   - Handle resource errors
   - Log error details

4. **Media Handling**:
   - Use `files` parameter
   - Check media type support
   - Handle multiple inputs properly

5. **Capability Detection**:
   - Check capabilities before use
   - Consider confidence scores
   - Handle missing capabilities

## Getting Help

- Check the [documentation](docs/architecture.md)
- Review the [examples](docs/examples.md)
- File issues on GitHub
- Contact support for assistance 