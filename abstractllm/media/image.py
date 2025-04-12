"""
Image input implementation for AbstractLLM.

This module provides the ImageInput class for handling image inputs.
"""

import os
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, Union, Optional

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ImageProcessingError

# Configure logger
logger = logging.getLogger("abstractllm.media.image")


class ImageInput(MediaInput):
    """
    Class representing an image input.
    
    This class handles different image sources (file paths, URLs, base64 data) and
    converts them to provider-specific formats.
    """
    
    def __init__(
        self, 
        source: Union[str, Path], 
        detail_level: str = "auto",
        mime_type: Optional[str] = None
    ):
        """
        Initialize an image input.
        
        Args:
            source: File path, URL, or base64 string
            detail_level: Detail level for image processing ("high", "medium", "low", "auto")
            mime_type: Optional explicit MIME type (auto-detected if not provided)
        """
        self.source = source
        self.detail_level = detail_level
        self._mime_type = mime_type
        self._cached_formats = {}  # Cache provider-specific formats
        self._cached_content = None  # Cached binary content
        
        # Validate inputs
        if not isinstance(source, (str, Path)):
            raise ValueError(f"Image source must be a string or Path, got {type(source)}")
        
        if detail_level not in ("auto", "high", "medium", "low"):
            logger.warning(f"Invalid detail_level '{detail_level}', defaulting to 'auto'")
            self.detail_level = "auto"
    
    @property
    def media_type(self) -> str:
        """Return the type of media."""
        return "image"
    
    @property
    def mime_type(self) -> str:
        """
        Get the MIME type of the image.
        
        Returns:
            MIME type string (e.g., 'image/jpeg')
        """
        if self._mime_type:
            return self._mime_type
        
        # Convert Path to string
        source_str = str(self.source)
        
        # Handle URLs or file paths
        if source_str.startswith(('http://', 'https://')):
            # Extract MIME type from URL extension
            guessed_type, _ = mimetypes.guess_type(source_str)
            if guessed_type and guessed_type.startswith('image/'):
                return guessed_type
            return 'image/jpeg'  # Default for URLs
            
        elif os.path.exists(source_str):
            # Use file extension to determine MIME type
            guessed_type, _ = mimetypes.guess_type(source_str)
            if guessed_type and guessed_type.startswith('image/'):
                return guessed_type
                
            # Fallback to extension mapping
            ext = os.path.splitext(source_str)[-1].lower()
            mime_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }
            return mime_map.get(ext, 'image/jpeg')
            
        elif source_str.startswith('data:'):
            # Extract MIME type from data URL
            try:
                return source_str.split(';')[0].split(':')[1]
            except (IndexError, ValueError):
                return 'image/jpeg'
                
        # Default for base64 or other string content
        return 'image/jpeg'
    
    def get_content(self) -> bytes:
        """
        Get the binary content of the image.
        
        Returns:
            Binary image data
            
        Raises:
            ImageProcessingError: If the image cannot be loaded
        """
        # Return cached content if available
        if self._cached_content is not None:
            return self._cached_content
            
        source_str = str(self.source)
        
        try:
            # Handle URLs
            if source_str.startswith(('http://', 'https://')):
                try:
                    import requests
                    # Add a proper User-Agent header to comply with API policies
                    headers = {
                        'User-Agent': 'AbstractLLM/0.1.0 (https://github.com/lpalbou/abstractllm; contact@example.com)'
                    }
                    response = requests.get(source_str, headers=headers, timeout=10)
                    response.raise_for_status()
                    self._cached_content = response.content
                    return self._cached_content
                except ImportError:
                    raise ImageProcessingError(
                        "Requests library not available. Install with: pip install requests",
                        provider=None
                    )
                except Exception as e:
                    raise ImageProcessingError(
                        f"Failed to download image from URL: {e}",
                        provider=None,
                        original_exception=e
                    )
                    
            # Handle file paths
            elif os.path.exists(source_str):
                try:
                    with open(source_str, 'rb') as f:
                        self._cached_content = f.read()
                    return self._cached_content
                except Exception as e:
                    raise ImageProcessingError(
                        f"Failed to read image file: {e}",
                        provider=None,
                        original_exception=e
                    )
                    
            # Handle data URLs
            elif source_str.startswith('data:'):
                try:
                    # Extract the base64 data
                    base64_data = source_str.split(',')[1]
                    self._cached_content = base64.b64decode(base64_data)
                    return self._cached_content
                except Exception as e:
                    raise ImageProcessingError(
                        f"Failed to decode data URL: {e}",
                        provider=None,
                        original_exception=e
                    )
            
            # Check for non-existent files before trying to decode as base64
            elif not source_str.startswith(('http://', 'https://')) and not os.path.exists(source_str):
                raise ImageProcessingError(
                    f"Invalid image source: File not found '{source_str}'",
                    provider=None
                )
                    
            # Handle raw base64 strings
            else:
                try:
                    # Try to decode as base64
                    self._cached_content = base64.b64decode(source_str)
                    return self._cached_content
                except Exception as e:
                    raise ImageProcessingError(
                        f"Failed to decode base64 data: {e}",
                        provider=None,
                        original_exception=e
                    )
                    
        except ImageProcessingError:
            # Re-raise ImageProcessingError exceptions
            raise
        except Exception as e:
            # Catch any other exceptions and wrap them
            raise ImageProcessingError(
                f"Unexpected error processing image: {e}",
                provider=None,
                original_exception=e
            )
    
    def get_base64(self) -> str:
        """
        Get the base64-encoded content of the image.
        
        Returns:
            Base64-encoded string (without MIME type prefix)
            
        Raises:
            ImageProcessingError: If the image cannot be encoded
        """
        # Handle data URLs directly
        source_str = str(self.source)
        if source_str.startswith('data:'):
            try:
                return source_str.split(',')[1]
            except (IndexError, ValueError) as e:
                raise ImageProcessingError(
                    f"Invalid data URL format: {e}",
                    provider=None,
                    original_exception=e
                )
                
        # Handle raw base64 strings
        if len(source_str) > 100 and not source_str.startswith(('http://', 'https://')) and not os.path.exists(source_str):
            try:
                # Validate it's actually base64
                base64.b64decode(source_str)
                return source_str
            except Exception:
                # Not valid base64, continue to normal processing
                pass
                
        # Otherwise, get binary content and encode to base64
        try:
            content = self.get_content()
            return base64.b64encode(content).decode('utf-8')
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to encode image to base64: {e}",
                provider=None,
                original_exception=e
            )
    
    def to_provider_format(self, provider: str) -> Any:
        """
        Convert the image to a format suitable for the specified provider.
        
        Args:
            provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface')
            
        Returns:
            Provider-specific format for the image
            
        Raises:
            ValueError: If the provider is not supported
            ImageProcessingError: If there's an error processing the image
        """
        # Return cached format if available
        if provider in self._cached_formats:
            return self._cached_formats[provider]
        
        # Convert to provider-specific format
        try:
            if provider == "openai":
                format_result = self._format_for_openai()
            elif provider == "anthropic":
                format_result = self._format_for_anthropic()
            elif provider == "ollama":
                format_result = self._format_for_ollama()
            elif provider == "huggingface":
                source_str = str(self.source)
                
                # Determine source type and content
                if os.path.exists(source_str):
                    source_type = "path"
                    content = source_str
                elif source_str.startswith(('http://', 'https://')):
                    source_type = "url"
                    content = source_str
                else:
                    source_type = "binary"
                    content = self.get_content()
                
                format_result = {
                    "type": "image",
                    "content": content,
                    "mime_type": self.mime_type,
                    "source_type": source_type
                }
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Cache the result
            self._cached_formats[provider] = format_result
            return format_result
            
        except ImageProcessingError:
            # Re-raise ImageProcessingError exceptions
            raise
        except Exception as e:
            # Wrap other exceptions
            raise ImageProcessingError(
                f"Error formatting image for {provider}: {e}",
                provider=provider,
                original_exception=e
            )
    
    def _format_for_openai(self) -> Dict[str, Any]:
        """
        Format the image for OpenAI.
        
        Returns:
            Dictionary in OpenAI's image format
            
        Raises:
            ImageProcessingError: If there's an error processing the image
        """
        source_str = str(self.source)
        
        # For image URLs, use them directly
        if source_str.startswith(('http://', 'https://')):
            return {
                "type": "image_url",
                "image_url": {
                    "url": source_str,
                    "detail": self.detail_level
                }
            }
        
        # For data URLs, use them directly
        elif source_str.startswith('data:'):
            return {
                "type": "image_url",
                "image_url": {
                    "url": source_str,
                    "detail": self.detail_level
                }
            }
        
        # Otherwise, encode to base64 and create a data URL
        else:
            base64_data = self.get_base64()
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self.mime_type};base64,{base64_data}",
                    "detail": self.detail_level
                }
            }
    
    def _format_for_anthropic(self) -> Dict[str, Any]:
        """
        Format the image for Anthropic.
        
        Returns:
            Dictionary in Anthropic's image format
            
        Raises:
            ImageProcessingError: If there's an error processing the image
        """
        source_str = str(self.source)
        
        # For image URLs, use the URL source format
        if source_str.startswith(('http://', 'https://')):
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": source_str
                }
            }
        
        # For local files, read and encode
        try:
            import mimetypes
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(source_str)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'image/jpeg'  # default to JPEG
            
            # Read file content
            if os.path.exists(source_str):
                with open(source_str, 'rb') as f:
                    file_content = f.read()
            else:
                file_content = self.get_content()
            
            # Base64 encode without any additional processing
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            
            # Return in Anthropic's expected format
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": encoded_content
                }
            }
            
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to format image for Anthropic: {e}",
                provider="anthropic",
                original_exception=e
            )
    
    def _format_for_ollama(self) -> str:
        """
        Format the image for Ollama.
        
        Returns:
            Base64-encoded image string or URL
            
        Raises:
            ImageProcessingError: If there's an error processing the image
        """
        source_str = str(self.source)
        
        # For image URLs, return the URL directly
        if source_str.startswith(('http://', 'https://')):
            return source_str
        
        # Otherwise, return base64-encoded data
        return self.get_base64()
    
    def _format_for_huggingface(self) -> Union[str, bytes]:
        """
        Format the image for HuggingFace.
        
        Returns:
            File path, URL, or binary content
            
        Raises:
            ImageProcessingError: If there's an error processing the image
        """
        source_str = str(self.source)
        
        # For file paths, return the path directly
        if os.path.exists(source_str):
            return source_str
        
        # For URLs, return the URL
        if source_str.startswith(('http://', 'https://')):
            return source_str
        
        # Otherwise, return binary content
        return self.get_content()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the image.
        
        Returns:
            Dictionary with image metadata
        """
        metadata = {
            "media_type": "image",
            "mime_type": self.mime_type,
            "detail_level": self.detail_level
        }
        
        # Try to get image dimensions if possible
        try:
            from PIL import Image
            import io
            
            # For file paths
            source_str = str(self.source)
            if os.path.exists(source_str):
                with Image.open(source_str) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["format"] = img.format
            else:
                # For URLs or base64 data
                content = self.get_content()
                with Image.open(io.BytesIO(content)) as img:
                    metadata["width"] = img.width
                    metadata["height"] = img.height
                    metadata["format"] = img.format
        except Exception:
            # Skip dimension detection if PIL is not available or there's an error
            pass
            
        return metadata 