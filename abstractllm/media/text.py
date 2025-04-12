"""
Text input implementation for AbstractLLM.

This module provides the TextInput class for handling text file inputs.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Union, Optional

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ImageProcessingError

# Configure logger
logger = logging.getLogger("abstractllm.media.text")

class TextInput(MediaInput):
    """
    Class representing a text input.
    
    This class handles different text sources (file paths, URLs, raw text)
    and converts them to provider-specific formats.
    """
    
    def __init__(self, 
                source: Union[str, Path], 
                encoding: str = 'utf-8',
                mime_type: Optional[str] = None):
        """
        Initialize a text input.
        
        Args:
            source: File path, URL, or raw text content
            encoding: Text encoding (default: utf-8)
            mime_type: Optional explicit MIME type
        """
        self.source = source
        self.encoding = encoding
        self._mime_type = mime_type
        self._cached_content = None
        
        # Validate inputs
        if not isinstance(source, (str, Path)):
            raise ValueError(f"Text source must be a string or Path, got {type(source)}")
    
    @property
    def media_type(self) -> str:
        """Return the type of media."""
        return "text"
    
    @property
    def mime_type(self) -> str:
        """
        Get the MIME type of the text.
        
        Returns:
            MIME type string (e.g., 'text/plain', 'text/csv')
        """
        if self._mime_type:
            return self._mime_type
            
        # Convert Path to string
        source_str = str(self.source)
        
        # Handle URLs
        if source_str.startswith(('http://', 'https://')):
            # Extract MIME type from URL extension
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(source_str)
            if guessed_type and guessed_type.startswith('text/'):
                return guessed_type
            return 'text/plain'  # Default for URLs
            
        # Handle file paths
        elif os.path.exists(source_str):
            # Use file extension to determine MIME type
            import mimetypes
            guessed_type, _ = mimetypes.guess_type(source_str)
            if guessed_type and guessed_type.startswith('text/'):
                return guessed_type
                
            # Fallback to extension mapping
            ext = os.path.splitext(source_str)[-1].lower()
            mime_map = {
                '.txt': 'text/plain',
                '.csv': 'text/csv',
                '.tsv': 'text/tab-separated-values',
                '.md': 'text/markdown',
                '.json': 'application/json'
            }
            return mime_map.get(ext, 'text/plain')
            
        # Default for raw text content
        return 'text/plain'
    
    def get_content(self) -> str:
        """
        Get the text content.
        
        Returns:
            Text content as string
            
        Raises:
            Exception: If the text cannot be loaded
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
                    # Add a proper User-Agent header
                    headers = {
                        'User-Agent': 'AbstractLLM/0.1.0 (https://github.com/lpalbou/abstractllm)'
                    }
                    response = requests.get(source_str, headers=headers, timeout=10)
                    response.raise_for_status()
                    self._cached_content = response.text
                    return self._cached_content
                except ImportError:
                    raise ImportError(
                        "Requests library not available. Install with: pip install requests"
                    )
                except Exception as e:
                    raise Exception(f"Failed to download text from URL: {e}")
                    
            # Handle file paths
            elif os.path.exists(source_str):
                try:
                    with open(source_str, 'r', encoding=self.encoding) as f:
                        self._cached_content = f.read()
                    return self._cached_content
                except Exception as e:
                    raise Exception(f"Failed to read text file: {e}")
                    
            # Handle raw text content
            else:
                self._cached_content = source_str
                return self._cached_content
                
        except Exception as e:
            raise Exception(f"Failed to get text content: {e}")
    
    def to_provider_format(self, provider: str) -> Any:
        """
        Convert the text to a format suitable for the specified provider.
        
        Args:
            provider: The provider name
            
        Returns:
            Provider-specific format for the text
            
        Raises:
            ValueError: If the provider is not supported
        """
        # Get the text content
        content = self.get_content()
        
        # Format for each provider
        if provider == "openai":
            return {
                "type": "text",
                "text": content
            }
        elif provider == "anthropic":
            return {
                "type": "text",
                "text": content
            }
        elif provider == "ollama":
            # For Ollama, return a format that can be appended to the prompt
            source_name = str(self.source)
            if isinstance(self.source, Path):
                source_name = self.source.name
            
            return f"\n===== JOINT FILES ======\n\n===== {source_name} =========\n{content}\n"
        elif provider == "huggingface":
            return {
                "type": "text",
                "content": content,
                "mime_type": self.mime_type
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the text.
        
        Returns:
            Dictionary with text metadata
        """
        metadata = {
            "media_type": "text",
            "mime_type": self.mime_type,
            "encoding": self.encoding
        }
        
        # Add file-specific metadata if it's a file
        source_str = str(self.source)
        if os.path.exists(source_str):
            metadata.update({
                "file_size": os.path.getsize(source_str),
                "last_modified": os.path.getmtime(source_str)
            })
            
        # Add content length if we have cached content
        if self._cached_content is not None:
            metadata["content_length"] = len(self._cached_content)
            
        return metadata 