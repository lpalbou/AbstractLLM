"""
Media input interface for AbstractLLM.

This module defines the abstract interface for all media inputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional


class MediaInput(ABC):
    """
    Abstract base class for all media inputs.
    
    This class defines the interface that all media input types must implement.
    It provides methods for converting media to provider-specific formats and
    retrieving metadata about the media.
    """
    
    @abstractmethod
    def to_provider_format(self, provider: str) -> Any:
        """
        Convert the media to a format suitable for the specified provider.
        
        Args:
            provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface')
            
        Returns:
            Provider-specific format for the media (varies by provider and media type)
            
        Raises:
            ValueError: If the provider is not supported
            ImageProcessingError: If there's an error processing the media
        """
        pass
    
    @property
    @abstractmethod
    def media_type(self) -> str:
        """
        Return the type of media (image, document, etc.).
        
        Returns:
            String identifier for the media type
        """
        pass
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the media.
        
        This can include information like dimensions for images, page count for documents,
        file size, creation date, etc.
        
        Returns:
            Dictionary of metadata about the media
        """
        return {} 