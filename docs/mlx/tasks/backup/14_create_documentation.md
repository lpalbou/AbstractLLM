# Task 14: Create Documentation

## Description
Create comprehensive documentation for the MLX provider, including detailed coverage of vision capabilities.

## Requirements
1. Document provider setup and configuration
2. Document vision model support and capabilities
3. Document image processing and memory requirements
4. Provide usage examples and best practices
5. Document error handling and troubleshooting

## Implementation Details

### Provider Documentation

Create documentation in `docs/mlx/provider.md`:

```markdown
# MLX Provider Documentation

## Overview
The MLX provider enables running MLX models on Apple Silicon hardware, including support for vision-language models. This provider is optimized for performance on Apple devices and includes robust support for image processing and multi-modal interactions.

## Features
- Text generation with MLX models
- Vision-language model support
- Streaming generation
- System prompt support
- Memory-safe image processing
- Multi-image support
- Async/await support

## Requirements
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.0 or later
- Python 3.8 or later
- MLX and MLX-VLM packages

## Installation
```bash
pip install abstractllm[mlx]
```

## Configuration
The MLX provider accepts the following configuration parameters:

```python
from abstractllm import create_llm
from abstractllm.enums import ModelParameter

config = {
    ModelParameter.MODEL: "mlx-community/Qwen2.5-VL-32B-Instruct-6bit",  # Model name
    ModelParameter.TEMPERATURE: 0.7,  # Generation temperature
    ModelParameter.MAX_TOKENS: 1024,  # Maximum tokens to generate
    "quantize": True,  # Enable quantization for reduced memory usage
}

llm = create_llm("mlx", **config)
```

## Supported Models

### Text Models
- Mistral 7B
- Phi-2
- Nous-Hermes
- And more...

### Vision Models
- Qwen2.5-VL-32B-Instruct-6bit
- LLaVA v1.6 34B
- Kimi-VL 70B
- And more...

## Vision Capabilities

### Image Processing
The provider automatically handles image processing based on model requirements:

- Automatic resizing while preserving aspect ratio
- Proper normalization and tensor formatting
- Memory-safe processing with size limits
- Support for multiple input formats:
  - File paths
  - PIL Images
  - NumPy arrays
  - URLs (with automatic download)

### Memory Requirements
Memory usage varies by model and image size:

| Model | Quantized Size | Max Image Resolution | Peak Memory |
|-------|---------------|---------------------|-------------|
| Qwen2.5-VL | ~16GB | 448x448 | ~20GB |
| LLaVA v1.6 | ~12GB | 336x336 | ~16GB |
| Kimi-VL | ~24GB | 448x448 | ~28GB |

### Usage Examples

#### Basic Image Analysis
```python
from abstractllm import create_llm

# Create provider
llm = create_llm("mlx", model="mlx-community/Qwen2.5-VL-32B-Instruct-6bit")

# Generate description
response = llm.generate(
    prompt="What's in this image?",
    files=["path/to/image.jpg"]
)
print(response.content)
```

#### Multiple Images
```python
# Compare two images
response = llm.generate(
    prompt="Compare these two images.",
    files=["image1.jpg", "image2.jpg"]
)
```

#### Streaming Generation
```python
# Stream response chunks
for chunk in llm.generate(
    prompt="Describe this image in detail.",
    files=["image.jpg"],
    stream=True
):
    print(chunk.content, end="")
```

#### System Prompts
```python
# Use system prompt for specialized analysis
response = llm.generate(
    prompt="Analyze this image.",
    system_prompt="You are an expert art critic.",
    files=["artwork.jpg"]
)
```

#### Async Usage
```python
# Async generation
response = await llm.generate_async(
    prompt="What's in this image?",
    files=["image.jpg"]
)
```

## Error Handling

### Common Errors

#### ImageProcessingError
Raised when image processing fails:
- Invalid image format
- Corrupted image data
- Unsupported image type

```python
try:
    response = llm.generate(prompt="Describe this.", files=["invalid.jpg"])
except ImageProcessingError as e:
    print(f"Image processing failed: {e}")
```

#### MemoryError
Raised when image processing would exceed memory limits:
- Image too large
- Insufficient system memory
- Multiple large images

```python
try:
    response = llm.generate(prompt="Describe this.", files=["large.jpg"])
except MemoryError as e:
    print(f"Memory limit exceeded: {e}")
```

#### UnsupportedFeatureError
Raised when attempting vision tasks with non-vision models:
```python
try:
    response = llm.generate(prompt="Describe this.", files=["image.jpg"])
except UnsupportedFeatureError as e:
    print(f"Vision not supported: {e}")
```

## Best Practices

### Image Processing
1. **Optimize Image Size**: Pre-resize large images to model's required size
2. **Memory Management**: Monitor system memory when processing multiple images
3. **Format Selection**: Use appropriate image formats (JPEG for photos, PNG for graphics)
4. **Error Handling**: Always implement proper error handling for image processing

### Model Selection
1. **Choose Appropriate Model**: Select model based on task requirements
2. **Consider Memory**: Account for model and image memory requirements
3. **Quantization**: Enable quantization for reduced memory usage
4. **Batch Processing**: Implement rate limiting for multiple images

### Performance Optimization
1. **Async Processing**: Use async methods for concurrent processing
2. **Memory Cleanup**: Implement proper cleanup after processing
3. **Streaming**: Use streaming for large responses
4. **Resource Monitoring**: Monitor system resources during processing

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce image size
   - Enable quantization
   - Process fewer images simultaneously
   - Close unnecessary applications

2. **Slow Processing**
   - Check system resource usage
   - Optimize image size
   - Use async processing for multiple images
   - Consider using a smaller model

3. **Image Format Issues**
   - Verify image format support
   - Convert to supported format
   - Check image corruption
   - Validate image dimensions

4. **Model Loading Failures**
   - Verify model availability
   - Check internet connection
   - Validate model compatibility
   - Monitor system resources

## References
- [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- [MLX-VLM Documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx-vlm)
- [AbstractLLM Documentation](https://abstractllm.readthedocs.io/)
```

### API Documentation

Update `docs/api/providers/mlx.md`:

```markdown
# MLX Provider API Reference

## MLXProvider

### Constructor
```python
def __init__(self, config: Dict[str, Any]) -> None
```

#### Parameters
- `config`: Configuration dictionary with the following keys:
  - `ModelParameter.MODEL`: Model name/path (required)
  - `ModelParameter.TEMPERATURE`: Generation temperature (default: 0.7)
  - `ModelParameter.MAX_TOKENS`: Maximum tokens to generate (default: 1024)
  - `quantize`: Enable model quantization (default: True)

### Methods

#### generate
```python
def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    files: Optional[List[Union[str, Path, Image.Image]]] = None,
    stream: bool = False,
    **kwargs
) -> Union[Response, Generator[Response, None, None]]
```

Generate text or analyze images with the model.

##### Parameters
- `prompt`: The generation prompt
- `system_prompt`: Optional system prompt
- `files`: Optional list of image inputs
- `stream`: Whether to stream the response
- `**kwargs`: Additional generation parameters

##### Returns
- Single response or generator of response chunks

#### generate_async
```python
async def generate_async(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    files: Optional[List[Union[str, Path, Image.Image]]] = None,
    stream: bool = False,
    **kwargs
) -> Union[Response, AsyncGenerator[Response, None]]
```

Async version of generate method.

### Internal Methods

#### _process_image
```python
def _process_image(self, image: MediaInput) -> np.ndarray
```

Process image input for model consumption.

#### _check_memory_requirements
```python
def _check_memory_requirements(self, image_size: Tuple[int, int], num_images: int) -> None
```

Verify system can handle image processing.

#### _format_prompt
```python
def _format_prompt(self, prompt: str, num_images: int) -> str
```

Format prompt for vision model input.

## Exceptions

### ImageProcessingError
Raised for image processing failures.

### MemoryError
Raised when exceeding memory limits.

### UnsupportedFeatureError
Raised for unsupported operations.

### ModelLoadError
Raised when model loading fails.
```

## References
- See `docs/mlx/vision-upgrade.md` for vision implementation details
- See `docs/mlx/deepsearch-mlx-vlm.md` for MLX-VLM insights
- See MLX-VLM documentation for model-specific requirements

## Testing
Verify documentation:
1. Check all code examples run successfully
2. Verify all links are valid
3. Test documentation search functionality
4. Validate API reference accuracy

## Success Criteria
1. Documentation is comprehensive and accurate
2. All code examples are tested and working
3. Vision capabilities are clearly explained
4. Error handling is well documented
5. Best practices are clearly outlined
6. API reference is complete and accurate 