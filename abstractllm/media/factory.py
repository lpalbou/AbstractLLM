"""
Media factory for AbstractLLM.

This module provides a factory class for creating media input objects.
"""

import os
import mimetypes
from pathlib import Path
from typing import Any, Dict, Union, Optional, List, Type

from abstractllm.media.interface import MediaInput
from abstractllm.media.image import ImageInput
from abstractllm.media.text import TextInput
from abstractllm.media.tabular import TabularInput
from abstractllm.exceptions import ImageProcessingError


class MediaFactory:
    """
    Factory for creating media input objects.
    
    This class provides methods for creating the appropriate MediaInput object
    based on the source and type of media.
    """
    
    # Media type mapping
    _MEDIA_HANDLERS = {
        "image": ImageInput,
        "text": TextInput,
        "tabular": TabularInput
    }
    
    # MIME type to media type mapping
    _MIME_TYPE_MAPPING = {
        'image/jpeg': 'image',
        'image/png': 'image',
        'image/gif': 'image',
        'image/webp': 'image',
        'image/bmp': 'image',
        'text/plain': 'text',
        'text/markdown': 'text',
        'text/csv': 'tabular',
        'text/tab-separated-values': 'tabular',
        'application/json': 'text'
    }
    
    @classmethod
    def from_source(
        cls, 
        source: Union[str, Path, Dict[str, Any]], 
        media_type: Optional[str] = None
    ) -> MediaInput:
        """
        Create a media input object from a source.
        
        Args:
            source: File path, URL, base64 string, or provider-specific dict
            media_type: Explicit media type (optional, auto-detected if not provided)
            
        Returns:
            Appropriate MediaInput instance
            
        Raises:
            ValueError: If the media type cannot be determined or is unsupported
            ImageProcessingError: If there's an error processing the media
        """
        # If already a MediaInput instance, return it
        if isinstance(source, MediaInput):
            return source
            
        # If media_type is provided, use it directly
        if media_type:
            return cls._create_media_input(source, media_type)
        
        # If source is a dictionary with a type key, use it    
        if isinstance(source, dict) and "type" in source:
            dict_type = source["type"]
            if dict_type in cls._MEDIA_HANDLERS:
                return cls._create_media_input(source, dict_type)
            else:
                raise ValueError(f"Unsupported media type in dictionary: {dict_type}")
            
        # Otherwise, try to detect the media type
        detected_type = cls._detect_media_type(source)
        if not detected_type:
            raise ValueError(
                f"Could not determine media type for source: {source}. "
                "Please specify media_type explicitly."
            )
            
        return cls._create_media_input(source, detected_type)
    
    @classmethod
    def from_sources(
        cls,
        sources: List[Union[str, Path, Dict[str, Any], MediaInput]],
        media_type: Optional[str] = None
    ) -> List[MediaInput]:
        """
        Create media input objects from multiple sources.
        
        Args:
            sources: List of file paths, URLs, base64 strings, or provider-specific dicts
            media_type: Explicit media type for all sources (optional)
            
        Returns:
            List of MediaInput instances
            
        Raises:
            ValueError: If a media type cannot be determined or is unsupported
            ImageProcessingError: If there's an error processing any media
        """
        return [cls.from_source(source, media_type) for source in sources]
    
    @classmethod
    def _create_media_input(
        cls, 
        source: Union[str, Path, Dict[str, Any]], 
        media_type: str
    ) -> MediaInput:
        """
        Create a media input object of the specified type.
        
        Args:
            source: File path, URL, base64 string, or provider-specific dict
            media_type: Media type (e.g., 'image', 'text', 'tabular')
            
        Returns:
            MediaInput instance
            
        Raises:
            ValueError: If the media type is unsupported
        """
        if media_type not in cls._MEDIA_HANDLERS:
            raise ValueError(f"Unsupported media type: {media_type}")
            
        handler_class = cls._MEDIA_HANDLERS[media_type]
        
        # Handle dictionary sources
        if isinstance(source, dict):
            # Handle image dictionaries
            if media_type == "image":
                # Extract optional parameters for the handler
                detail_level = source.get("detail_level", "auto")
                
                # Ensure the dictionary has a valid source
                if "source" not in source and "url" not in source and "image_url" not in source:
                    raise ValueError(f"Image dictionary must contain 'source', 'url', or 'image_url': {source}")
                
                # For image dicts, extract the source and create an ImageInput
                if "url" in source:
                    return handler_class(source["url"], detail_level=detail_level)
                elif "image_url" in source and isinstance(source["image_url"], dict):
                    return handler_class(source["image_url"]["url"], detail_level=detail_level)
                elif "source" in source:
                    if isinstance(source["source"], dict) and source["source"]["type"] == "url":
                        return handler_class(source["source"]["url"], detail_level=detail_level)
                    elif isinstance(source["source"], dict) and source["source"]["type"] == "base64":
                        return handler_class(
                            f"data:{source['source'].get('media_type', 'image/jpeg')};base64,{source['source']['data']}",
                            detail_level=detail_level
                        )
                    elif isinstance(source["source"], str):
                        # Handle direct string URLs in the "source" key
                        return handler_class(source["source"], detail_level=detail_level)
                        
            # Handle tabular data dictionaries
            elif media_type == "tabular":
                delimiter = source.get("delimiter", ",")
                encoding = source.get("encoding", "utf-8")
                return handler_class(source.get("source", source), delimiter=delimiter, encoding=encoding)
                
            # Handle text dictionaries
            elif media_type == "text":
                encoding = source.get("encoding", "utf-8")
                return handler_class(source.get("source", source), encoding=encoding)
        
        # Handle string or Path sources
        source_str = str(source)
        
        # For tabular data, detect delimiter based on extension
        if media_type == "tabular":
            delimiter = "\t" if source_str.endswith(".tsv") else ","
            return handler_class(source, delimiter=delimiter)
            
        # For other types, create with default parameters
        return handler_class(source)
    
    @staticmethod
    def _detect_media_type(source: Union[str, Path, Dict[str, Any]]) -> Optional[str]:
        """
        Detect the media type from the source.
        
        Args:
            source: File path, URL, base64 string, or provider-specific dict
            
        Returns:
            Media type string or None if the type cannot be determined
            
        Raises:
            ValueError: If a dictionary is provided without a type field
        """
        # Handle dictionary sources
        if isinstance(source, dict):
            # For dictionaries, we require a "type" field
            if "type" not in source:
                raise ValueError(f"Dictionary source must include a 'type' field: {source}")
                
            if any(key in source for key in ["image_url", "source"]):
                return "image"
            return None
        
        # Convert Path to string
        source_str = str(source)
        
        # Handle URLs
        if source_str.startswith(('http://', 'https://')):
            # Check extension
            guessed_type, _ = mimetypes.guess_type(source_str)
            if guessed_type:
                # Map MIME type to media type
                for mime_pattern, media_type in MediaFactory._MIME_TYPE_MAPPING.items():
                    if guessed_type.startswith(mime_pattern):
                        return media_type
                
            # Default for URLs without clear type
            if any(domain in source_str for domain in [
                "imgur.com", "flickr.com", "unsplash.com", "pexels.com",
                ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"
            ]):
                return "image"
                
            return None
            
        # Handle file paths
        elif os.path.exists(source_str):
            # Use file extension to determine MIME type
            guessed_type, _ = mimetypes.guess_type(source_str)
            if guessed_type:
                # Map MIME type to media type
                for mime_pattern, media_type in MediaFactory._MIME_TYPE_MAPPING.items():
                    if guessed_type.startswith(mime_pattern):
                        return media_type
                
            # Fallback to extension mapping
            ext = os.path.splitext(source_str)[-1].lower()
            ext_map = {
                '.txt': 'text',
                '.md': 'text',
                '.csv': 'tabular',
                '.tsv': 'tabular',
                '.json': 'text',
                '.jpg': 'image',
                '.jpeg': 'image',
                '.png': 'image',
                '.gif': 'image',
                '.webp': 'image',
                '.bmp': 'image'
            }
            return ext_map.get(ext)
            
        # For other strings, harder to determine
        # If it's a long string, it might be base64
        elif len(source_str) > 100:
            # We'll default to image for now if it's a long string that isn't a file path
            try:
                import base64
                base64.b64decode(source_str)
                return "image"  # Assume base64 is an image
            except Exception:
                return None
                
        return None 