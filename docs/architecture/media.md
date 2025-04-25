# Media System

This document explains AbstractLLM's media handling system, which enables working with different types of media inputs like images, documents, and other non-text content.

## Overview

The media system in AbstractLLM provides a unified interface for handling different media types across various providers. It allows users to pass media inputs to LLMs without worrying about provider-specific formats.

## Core Components

### MediaInput Interface

The foundation of the media system is the `MediaInput` abstract base class:

```python
# From abstractllm/media/interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class MediaInput(ABC):
    """
    Abstract base class for all media inputs.
    
    This class defines the interface that all media input types must implement.
    """
    
    @abstractmethod
    def to_provider_format(self, provider: str) -> Any:
        """Convert the media to a format suitable for the specified provider."""
        pass
    
    @property
    @abstractmethod
    def media_type(self) -> str:
        """Return the type of media (image, document, etc.)."""
        pass
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the media."""
        return {}
```

### MediaFactory

The `MediaFactory` class provides factory methods for creating media inputs from various sources:

```python
# From abstractllm/media/factory.py
class MediaFactory:
    """
    Factory class for creating media inputs.
    
    Provides methods for creating media inputs from various sources.
    """
    
    @classmethod
    def from_source(cls, source: Union[str, Path, Dict[str, Any]], media_type: Optional[str] = None) -> MediaInput:
        """
        Create a media input object from a source.
        
        Args:
            source: File path, URL, base64 string, or provider-specific dict
            media_type: Explicit media type (optional, auto-detected if not provided)
            
        Returns:
            Appropriate MediaInput instance
        """
        # Detect source type
        if isinstance(source, str):
            if os.path.exists(source):
                return cls._from_file(source, media_type)
            elif source.startswith(("http://", "https://")):
                return cls._from_url(source, media_type)
            elif source.startswith("data:"):
                return cls._from_base64(source, media_type)
        elif isinstance(source, dict):
            return cls._from_dict(source, media_type)
        
        raise ValueError("Unsupported source type")
```

## Media Types

### ImageInput

The most commonly used media type is `ImageInput`, for working with images:

```python
# From abstractllm/media/image.py
class ImageInput(MediaInput):
    """
    Image input for LLMs.
    
    Supports loading images from files, URLs, and base64 data.
    """
    
    def __init__(self, content: bytes = None, url: str = None, file_path: str = None, 
                base64_data: str = None, mime_type: str = None):
        """Initialize the image input."""
        # Implementation details...
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ImageInput":
        """Create an ImageInput from a file."""
        # Implementation details...
    
    @classmethod
    def from_url(cls, url: str) -> "ImageInput":
        """Create an ImageInput from a URL."""
        # Implementation details...
    
    @classmethod
    def from_base64(cls, base64_data: str, mime_type: str = None) -> "ImageInput":
        """Create an ImageInput from base64 data."""
        # Implementation details...
    
    def to_provider_format(self, provider: str) -> Any:
        """Convert the image to a format suitable for the specified provider."""
        # Provider-specific formatting
        if provider == "openai":
            return self._format_for_openai()
        elif provider == "anthropic":
            return self._format_for_anthropic()
        elif provider == "ollama":
            return self._format_for_ollama()
        elif provider == "huggingface":
            return self._format_for_huggingface()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
```

### TextInput

`TextInput` is used for text-based documents:

```python
# From abstractllm/media/text.py
class TextInput(MediaInput):
    """
    Text input for LLMs.
    
    Supports loading text from strings, files, and URLs.
    """
    
    def __init__(self, content: str = None, file_path: str = None, url: str = None):
        """Initialize the text input."""
        # Implementation details...
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "TextInput":
        """Create a TextInput from a file."""
        # Implementation details...
    
    @classmethod
    def from_url(cls, url: str) -> "TextInput":
        """Create a TextInput from a URL."""
        # Implementation details...
    
    def to_provider_format(self, provider: str) -> Any:
        """Convert the text to a format suitable for the specified provider."""
        # Provider-specific formatting
        # Most providers just need the text content
        return self.content
```

## Provider-Specific Handling

Each provider has different requirements for media inputs:

1. **OpenAI**: Requires images in base64 format or URLs, with optional detail level
2. **Anthropic**: Accepts base64-encoded images with MIME type
3. **Ollama**: Requires base64-encoded images for compatible models
4. **HuggingFace**: Variable requirements based on the specific model

The media system handles these differences through provider-specific formatting methods.

## Usage Patterns

### Basic Usage

```python
from abstractllm import create_llm
from abstractllm.media import MediaFactory

# Create a provider
llm = create_llm("openai", model="gpt-4o")

# Create media input
image = MediaFactory.from_source("path/to/image.jpg")

# Use in generation
response = llm.generate(
    prompt="What's in this image?",
    image=image
)
```

### Direct Provider Interface

```python
# Use provider-specific format
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    prompt="Describe this image.",
    image={
        "url": "https://example.com/image.jpg",
        "detail": "high"
    }
)
```

## Error Handling

The media system includes robust error handling for various scenarios:

```python
# From abstractllm/media/image.py
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    content = response.content
except (requests.RequestException, ValueError) as e:
    raise ImageProcessingError(f"Failed to fetch image from URL: {str(e)}")
```

## Future Extensions

The media system is designed to be extensible for future media types:

1. **Audio Inputs**: For speech and audio processing
2. **Video Inputs**: For video analysis and processing
3. **Document Inputs**: For structured documents like PDFs and spreadsheets

## Next Steps

- [Tool System](tools.md): How AbstractLLM implements tool calling
- [Vision Capabilities](../user-guide/vision.md): User guide for working with images 