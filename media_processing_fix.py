#!/usr/bin/env python3
"""
Media Processing Fix for AbstractLLM

This fix addresses the media type detection error by improving the factory's
ability to handle various file types and providing better fallback behavior.
"""

from pathlib import Path
from typing import Union, Optional, Dict, Any
import mimetypes
import logging

logger = logging.getLogger(__name__)


class EnhancedMediaFactory:
    """
    Enhanced media factory with improved type detection and error handling.

    Key improvements:
    - Better extension-based detection
    - Graceful handling of unsupported files
    - Content inspection fallbacks
    - Programming language file detection
    """

    # Enhanced MIME type mapping
    _ENHANCED_MIME_MAPPING = {
        # Standard media types
        'image/jpeg': 'image',
        'image/png': 'image',
        'image/gif': 'image',
        'image/webp': 'image',
        'image/bmp': 'image',
        'image/svg+xml': 'image',

        # Text types
        'text/plain': 'text',
        'text/markdown': 'text',
        'text/csv': 'tabular',
        'text/tab-separated-values': 'tabular',
        'application/json': 'text',
        'application/xml': 'text',
        'text/xml': 'text',
        'text/html': 'text',
        'text/css': 'text',
        'text/javascript': 'text',

        # Programming languages (should be treated as text)
        'text/x-python': 'text',
        'text/x-c': 'text',
        'text/x-java': 'text',
        'application/x-python': 'text',
    }

    # File extension to media type mapping (fallback)
    _EXTENSION_MAPPING = {
        # Images
        '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.gif': 'image',
        '.webp': 'image', '.bmp': 'image', '.svg': 'image', '.tiff': 'image',

        # Text files
        '.txt': 'text', '.md': 'text', '.markdown': 'text', '.rst': 'text',
        '.json': 'text', '.xml': 'text', '.html': 'text', '.htm': 'text',
        '.css': 'text', '.js': 'text', '.ts': 'text', '.yaml': 'text', '.yml': 'text',
        '.log': 'text', '.conf': 'text', '.cfg': 'text', '.ini': 'text',

        # Programming languages (should be treated as text)
        '.py': 'text', '.pyx': 'text', '.pyi': 'text',
        '.c': 'text', '.cpp': 'text', '.cc': 'text', '.cxx': 'text', '.h': 'text', '.hpp': 'text',
        '.java': 'text', '.scala': 'text', '.kt': 'text',
        '.js': 'text', '.ts': 'text', '.jsx': 'text', '.tsx': 'text',
        '.php': 'text', '.rb': 'text', '.go': 'text', '.rs': 'text',
        '.sh': 'text', '.bash': 'text', '.zsh': 'text', '.fish': 'text',
        '.sql': 'text', '.r': 'text', '.m': 'text', '.swift': 'text',
        '.lua': 'text', '.pl': 'text', '.ps1': 'text',

        # Tabular data
        '.csv': 'tabular', '.tsv': 'tabular', '.tab': 'tabular',

        # Documentation
        '.pdf': 'text',  # If PDF processing is available
        '.doc': 'text', '.docx': 'text',  # If Word processing is available
    }

    @classmethod
    def detect_media_type_enhanced(cls, source: Union[str, Path, Dict[str, Any]]) -> Optional[str]:
        """
        Enhanced media type detection with better fallback behavior.

        Args:
            source: File path, URL, base64 string, or provider-specific dict

        Returns:
            Media type string or None if unsupported (but won't raise errors)
        """
        # Handle dictionary sources
        if isinstance(source, dict):
            if "type" in source:
                return source["type"]
            if any(key in source for key in ["image_url", "source"]):
                return "image"
            return None

        # Convert Path to string
        source_str = str(source)

        # Handle base64 data URLs
        if source_str.startswith('data:'):
            if source_str.startswith('data:image/'):
                return "image"
            elif source_str.startswith('data:text/'):
                return "text"
            return None

        # Handle URLs
        if source_str.startswith(('http://', 'https://')):
            # Try to get extension from URL
            try:
                from urllib.parse import urlparse
                parsed = urlparse(source_str)
                path = Path(parsed.path)
                if path.suffix:
                    return cls._get_type_from_extension(path.suffix.lower())
            except:
                pass
            return None

        # Handle file paths
        try:
            path = Path(source_str)

            # Check if file exists and is readable
            if path.exists() and path.is_file():
                # Try MIME type detection first
                mime_type, _ = mimetypes.guess_type(str(path))
                if mime_type and mime_type in cls._ENHANCED_MIME_MAPPING:
                    detected_type = cls._ENHANCED_MIME_MAPPING[mime_type]
                    logger.debug(f"Detected media type '{detected_type}' for {source_str} via MIME type {mime_type}")
                    return detected_type

                # Fallback to extension-based detection
                extension = path.suffix.lower()
                if extension in cls._EXTENSION_MAPPING:
                    detected_type = cls._EXTENSION_MAPPING[extension]
                    logger.debug(f"Detected media type '{detected_type}' for {source_str} via extension {extension}")
                    return detected_type

                # If it's a programming file that we didn't recognize, default to text
                if extension in ['.py', '.pyx', '.pyi']:  # Specifically for Python files
                    logger.debug(f"Treating Python file {source_str} as text")
                    return 'text'

                # Content inspection for files without extensions
                if not extension:
                    try:
                        with open(path, 'rb') as f:
                            header = f.read(1024)

                        # Check for image headers
                        if header.startswith(b'\\xff\\xd8\\xff'):  # JPEG
                            return 'image'
                        elif header.startswith(b'\\x89PNG\\r\\n\\x1a\\n'):  # PNG
                            return 'image'
                        elif header.startswith(b'GIF'):  # GIF
                            return 'image'

                        # Check if it's text (UTF-8 decodable)
                        try:
                            header.decode('utf-8')
                            return 'text'
                        except UnicodeDecodeError:
                            pass

                    except (OSError, PermissionError):
                        logger.warning(f"Cannot read file {source_str} for content inspection")

                logger.debug(f"Could not determine media type for {source_str}")
                return None
            else:
                logger.debug(f"File {source_str} does not exist or is not readable")
                return None

        except Exception as e:
            logger.debug(f"Error processing file path {source_str}: {e}")
            return None

    @classmethod
    def _get_type_from_extension(cls, extension: str) -> Optional[str]:
        """Get media type from file extension."""
        return cls._EXTENSION_MAPPING.get(extension.lower())

    @classmethod
    def from_source_safe(cls, source: Union[str, Path, Dict[str, Any]],
                        media_type: Optional[str] = None) -> Optional['MediaInput']:
        """
        Safe version of from_source that won't raise errors for unsupported types.

        Args:
            source: File path, URL, base64 string, or provider-specific dict
            media_type: Explicit media type (optional, auto-detected if not provided)

        Returns:
            MediaInput instance or None if type cannot be determined/supported
        """
        try:
            # If media_type is provided, try using it directly
            if media_type:
                return cls._create_media_input_safe(source, media_type)

            # Try to detect the media type
            detected_type = cls.detect_media_type_enhanced(source)
            if detected_type:
                return cls._create_media_input_safe(source, detected_type)
            else:
                logger.info(f"Unsupported or unrecognized media type for source: {source}")
                return None

        except Exception as e:
            logger.warning(f"Failed to create media input for {source}: {e}")
            return None

    @classmethod
    def _create_media_input_safe(cls, source, media_type: str):
        """Safely create media input with error handling."""
        # This would integrate with the actual MediaFactory
        # For now, just a placeholder
        logger.debug(f"Would create {media_type} media input for {source}")
        return None


def apply_media_processing_fix():
    """
    Apply the media processing fix to the existing MediaFactory.

    This function patches the existing factory to use enhanced detection
    and provide better error handling for unsupported file types.
    """
    try:
        from abstractllm.media.factory import MediaFactory

        # Patch the detection method
        MediaFactory._detect_media_type_original = MediaFactory._detect_media_type
        MediaFactory._detect_media_type = EnhancedMediaFactory.detect_media_type_enhanced

        # Add safe creation method
        MediaFactory.from_source_safe = EnhancedMediaFactory.from_source_safe

        logger.info("Applied media processing enhancement patch")

    except ImportError:
        logger.warning("Could not import MediaFactory - fix not applied")


# Usage example for the LM Studio provider
def safe_file_processing(files):
    """
    Example of how to safely process files in providers.

    This prevents the "Could not determine media type" errors by
    filtering out unsupported files and providing user feedback.
    """
    processed_files = []
    unsupported_files = []

    for file_path in files:
        try:
            # Try enhanced detection first
            media_type = EnhancedMediaFactory.detect_media_type_enhanced(file_path)

            if media_type:
                # Create media input safely
                media_input = EnhancedMediaFactory.from_source_safe(file_path, media_type)
                if media_input:
                    processed_files.append(media_input)
                else:
                    unsupported_files.append(file_path)
            else:
                unsupported_files.append(file_path)

        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            unsupported_files.append(file_path)

    # Provide user feedback about unsupported files
    if unsupported_files:
        logger.info(f"Skipped {len(unsupported_files)} unsupported files: {unsupported_files}")

    return processed_files


if __name__ == "__main__":
    # Test the enhanced detection
    test_files = [
        "abstractllm/providers/mlx_provider.py",  # The problematic file from the error
        "image.jpg",
        "data.csv",
        "README.md",
        "unknown_file"
    ]

    for file_path in test_files:
        media_type = EnhancedMediaFactory.detect_media_type_enhanced(file_path)
        print(f"{file_path} -> {media_type}")