# Vision and File Handling in AbstractLLM

## Overview

This document outlines a comprehensive approach to handling vision/image and other file inputs in the AbstractLLM library. The goal is to extend the current interface to support different types of files in a way that's consistent with AbstractLLM's minimalist design philosophy, while ensuring robust cross-provider compatibility.

## Current Status and Limitations

### Vision Implementation Status

The current vision capabilities in AbstractLLM have the following limitations:

1. **Inconsistent Provider Handling**: Each provider requires different formats for images (base64, URLs, file paths), leading to complex preprocessing logic.
2. **Limited File Types**: Only image files are currently supported, not text files, PDFs, CSVs, etc.
3. **Mixed Responsibilities**: The image preprocessing code handles both media format detection and provider-specific transformation.
4. **Limited Error Handling**: The current implementation lacks robust error handling for media loading failures.
5. **Maintenance Challenges**: The complex conditional logic makes it difficult to maintain and extend the code.

### Image Processing Flow

Currently, image processing works by:
1. Detecting if an image input is present in the parameters
2. Converting the image from its source (URL, file path, or base64) to a provider-specific format
3. Integrating the formatted image into the API request

## Proposed Architecture

We propose a more modular, extensible architecture for handling media inputs. The key components are:

### 1. Media Input Interface

Define a clear interface for different types of media inputs:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional

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

### 2. Concrete Media Implementations

Implement concrete classes for different media types:

```python
class ImageInput(MediaInput):
    """Class representing an image input."""
    
    def __init__(self, source: Union[str, Path], detail_level: str = "auto"):
        """
        Initialize an image input.
        
        Args:
            source: File path, URL, or base64 string
            detail_level: Detail level for image processing (high, medium, low, auto)
        """
        self.source = source
        self.detail_level = detail_level
        self._cached_formats = {}  # Cache provider-specific formats
    
    @property
    def media_type(self) -> str:
        return "image"
    
    def to_provider_format(self, provider: str) -> Any:
        """Convert to provider-specific format."""
        if provider in self._cached_formats:
            return self._cached_formats[provider]
        
        result = self._convert_for_provider(provider)
        self._cached_formats[provider] = result
        return result
    
    def _convert_for_provider(self, provider: str) -> Any:
        """Provider-specific conversion logic."""
        # Implementation moved from format_image_for_provider
        # with improved error handling
```

### 3. File Handling Factory

Create a factory for handling file detection and loading:

```python
class MediaFactory:
    """Factory for creating media input objects."""
    
    @staticmethod
    def from_source(source: Union[str, Path, Dict], media_type: Optional[str] = None) -> MediaInput:
        """
        Create a media input object from a source.
        
        Args:
            source: File path, URL, base64 string, or provider-specific dict
            media_type: Explicit media type (optional, auto-detected if not provided)
            
        Returns:
            Appropriate MediaInput instance
            
        Raises:
            ValueError: If the media type cannot be determined or is unsupported
        """
        # Implementation details
```

### 4. Media Processor Module

Create a dedicated module for handling media processing:

```python
class MediaProcessor:
    """Process media inputs for LLM providers."""
    
    @staticmethod
    def process_inputs(params: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Process all media inputs in params for the specified provider.
        
        Args:
            params: Parameters that may include media inputs
            provider: Provider name
            
        Returns:
            Updated parameters with media inputs formatted for the provider
        """
        # Implementation details
```

## Implementation Plan

### Phase 1: Refactor Current Image Handling

1. **Create Media Interface**: Implement the `MediaInput` abstract base class.
2. **Implement Image Handling**: Create `ImageInput` class that encapsulates the current image handling logic.
3. **Implement Factory**: Create the `MediaFactory` for instantiating appropriate media objects.
4. **Add Robust Error Handling**: Implement proper error handling with specialized exceptions.

### Phase 2: Extend to Other File Types

1. **Implement Document Input**: Add support for text documents, PDFs, etc.
2. **Add MIME Type Detection**: Improve media type detection based on file extensions and content.
3. **Implement Provider-Specific Handlers**: Create specialized handlers for providers with unique requirements.

### Phase 3: Integration with Provider Implementations

1. **Update Provider Classes**: Modify provider implementations to use the new media handling architecture.
2. **Add Capability Reporting**: Ensure providers accurately report their media handling capabilities.
3. **Implement Validation**: Add validation to ensure media inputs match provider capabilities.

## Best Practices and Implementation Guidelines

### Media Type Detection

Use a combination of methods to reliably detect media types:

1. **File Extension**: Use the file extension as the first hint.
2. **MIME Type Library**: Use Python's `mimetypes` module for robust detection.
3. **Content Analysis**: For ambiguous cases, analyze file headers or content patterns.
4. **Explicit Type Hints**: Allow users to explicitly specify media types.

### Error Handling

Implement robust error handling specific to media processing:

1. **Specialized Exceptions**: Use the existing `ImageProcessingError` and add other specialized exceptions.
2. **Detailed Error Messages**: Provide clear error messages that help users fix issues.
3. **Early Validation**: Validate media inputs as early as possible to fail fast.
4. **Graceful Degradation**: Fall back to simpler formats when complex ones are unavailable.

### Caching Strategy

Implement efficient caching to avoid redundant processing:

1. **Memory Caching**: Cache processed media in memory to avoid repeated processing.
2. **Provider Format Caching**: Cache provider-specific formats separately.
3. **Lazy Loading**: Only process media when needed by the provider.
4. **Cache Invalidation**: Clear caches when inputs change.

### Provider-Specific Considerations

Address unique requirements for each provider:

1. **OpenAI**: Supports images via URL or base64 data URLs in specific formats.
2. **Anthropic**: Requires base64-encoded images with specific structure.
3. **Ollama**: Accepts base64-encoded images directly in the request.
4. **HuggingFace**: May require direct file paths for loading with PIL.

## API Usage Examples

### Basic Image Usage

```python
from abstractllm import create_llm

# Using a file path
llm = create_llm("openai", model="gpt-4o")
response = llm.generate("What's in this image?", image="path/to/image.jpg")

# Using a URL
response = llm.generate("Describe this image:", image="https://example.com/image.jpg")

# Using multiple images
response = llm.generate(
    "Compare these two images:",
    images=["path/to/image1.jpg", "https://example.com/image2.jpg"]
)
```

### Advanced Usage with Options

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Creating an image input with options
image = ImageInput("path/to/image.jpg", detail_level="high")

# Using the image input
llm = create_llm("anthropic", model="claude-3-opus-20240229")
response = llm.generate("Analyze this image in detail:", image=image)
```

## Conclusion

This proposed architecture provides a robust, extensible framework for handling various media types in AbstractLLM. By following these guidelines, we can ensure consistent behavior across providers while maintaining AbstractLLM's philosophy of simplicity and minimalism.

The implementation separates concerns appropriately, allowing for easier maintenance and extension. It also provides a clear, intuitive API for users while handling the complex provider-specific details behind the scenes.

Future improvements could include support for audio files, video files, and other specialized media types as provider capabilities expand. 