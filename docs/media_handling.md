# Media Handling in AbstractLLM

## Overview

AbstractLLM provides a robust system for handling various media inputs, particularly images, when interacting with LLM providers. This document explains how media handling works in AbstractLLM and how to use it in your applications.

## Architecture

The media handling system consists of several key components:

### MediaInput Interface

The `MediaInput` abstract base class defines the interface for all media inputs:

```python
class MediaInput(ABC):
    """Abstract base class for all media inputs."""
    
    @abstractmethod
    def to_provider_format(self, provider: str) -> Any:
        """Convert the media to a format suitable for the specified provider."""
        pass
    
    @property
    @abstractmethod
    def media_type(self) -> str:
        """Return the type of media (image, pdf, text, etc.)."""
        pass
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the media."""
        return {}
```

### ImageInput Implementation

The `ImageInput` class implements the `MediaInput` interface for image files:

```python
class ImageInput(MediaInput):
    """Class representing an image input."""
    
    def __init__(self, source: Union[str, Path], detail_level: str = "auto"):
        """Initialize an image input."""
        self.source = source
        self.detail_level = detail_level
        # ...
```

### MediaFactory

The `MediaFactory` class provides factory methods for creating media input objects:

```python
class MediaFactory:
    """Factory for creating media input objects."""
    
    @classmethod
    def from_source(cls, source: Union[str, Path, Dict], media_type: Optional[str] = None) -> MediaInput:
        """Create a media input object from a source."""
        # ...
```

### MediaProcessor

The `MediaProcessor` class handles the processing of media inputs for different providers:

```python
class MediaProcessor:
    """Process media inputs for LLM providers."""
    
    @classmethod
    def process_inputs(cls, params: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """Process all media inputs in params for the specified provider."""
        # ...
```

## Provider-Specific Handling

Each provider has unique requirements for how media inputs should be formatted:

### OpenAI

- Images are included in the content array of a message
- Each item in the content array has a type ("text" or "image_url")
- Images can be provided as URLs or base64-encoded data URIs

### Anthropic

- Images are included in the content array of a message
- Each item in the content array has a type ("text" or "image")
- Images can be provided as URLs or base64-encoded data

### Ollama

- Images are provided in a separate "images" array in the request
- Images can be URLs or base64-encoded strings
- Only certain models support image inputs (e.g., llava, bakllava)

### HuggingFace

- Images are typically provided as a single parameter ("image")
- Most models only support one image at a time
- Some advanced models like LLaVA-NeXT and IDEFICS support multiple images

## Usage Examples

### Basic Usage

The simplest way to use media inputs is directly through the `generate` method:

```python
from abstractllm import create_llm

# Create an LLM instance with a vision-capable model
llm = create_llm("openai", model="gpt-4o")

# Generate a response with an image from a URL
response = llm.generate(
    "What's in this image?",
    image="https://example.com/image.jpg"
)

# Or with a local image file
response = llm.generate(
    "Describe this image in detail.",
    image="path/to/local/image.jpg"
)

# Or with multiple images
response = llm.generate(
    "Compare these two images.",
    images=["https://example.com/image1.jpg", "path/to/local/image2.jpg"]
)
```

### Advanced Usage with ImageInput

For more control, you can use the `ImageInput` class directly:

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create an image input with specific options
image = ImageInput("path/to/image.jpg", detail_level="high")

# Use it with the LLM
llm = create_llm("anthropic", model="claude-3-opus-20240229")
response = llm.generate("Analyze this image in detail:", image=image)
```

### Handling Multiple Images

Different providers have different levels of support for multiple images:

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create image inputs
image1 = ImageInput("path/to/image1.jpg")
image2 = ImageInput("path/to/image2.jpg")

# For providers that support multiple images (OpenAI, Anthropic, Ollama)
llm = create_llm("openai", model="gpt-4o")
response = llm.generate("Compare these images:", images=[image1, image2])

# HuggingFace typically only supports one image at a time
llm = create_llm("huggingface", model="microsoft/Phi-4-multimodal-instruct")
response = llm.generate("Describe this image:", image=image1)
```

## Error Handling

The media handling system includes robust error handling:

```python
from abstractllm import create_llm
from abstractllm.exceptions import ImageProcessingError

llm = create_llm("openai", model="gpt-4o")

try:
    response = llm.generate("What's in this image?", image="nonexistent/image.jpg")
except ImageProcessingError as e:
    print(f"Error processing image: {e}")
    # Fallback to text-only prompt
    response = llm.generate("Sorry, I couldn't process your image. Please describe what you wanted to show.")
```

## Capability Checking

You can check if a model supports vision capabilities:

```python
from abstractllm import create_llm, ModelCapability

llm = create_llm("openai", model="gpt-3.5-turbo")
capabilities = llm.get_capabilities()

if capabilities.get(ModelCapability.VISION):
    print("This model supports vision inputs")
else:
    print("This model does not support vision inputs")
```

## Best Practices

1. **Check Vision Capability**: Always check if a model supports vision before sending image inputs.

2. **Handle Large Images**: Large images may exceed context limits; consider resizing or compressing images.

3. **Use Appropriate Detail Level**: For high-resolution images, consider using "low" or "medium" detail level to reduce token usage.

4. **Provider Awareness**: Be aware of provider-specific limitations and capabilities (e.g., HuggingFace generally supports only one image).

5. **Error Handling**: Implement proper error handling for image processing failures.

6. **Caching Consideration**: The `ImageInput` class caches formatted images, which can help performance but might increase memory usage for large images.

## Limitations

- Currently, only image files are supported. Future versions may add support for other media types like audio or video.
- Some providers have limitations on the number of images they can process simultaneously.
- Token usage increases significantly with image inputs, especially with high detail levels. 