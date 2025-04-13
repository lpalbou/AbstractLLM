"""
Text input implementation for AbstractLLM.

This module provides the TextInput class for handling text and document inputs.
"""

import os
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Union, Optional

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import MediaProcessingError

import mimetypes

# Configure logger
logger = logging.getLogger("abstractllm.media.text")

class TextInput(MediaInput):
    """
    Class representing a text input.
    
    This class handles:
    1. Plain text files (txt, md, json)
    2. Document files (pdf, docx) with text extraction
    3. URLs and file paths
    4. Raw text content
    """
    
    # Supported formats and their MIME types
    SUPPORTED_FORMATS = {
        # Plain text formats
        "txt": "text/plain",
        "md": "text/markdown",
        "json": "application/json",
        
        # Document formats
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "rtf": "application/rtf"
    }
    
    # Providers with native document handling
    NATIVE_HANDLERS = {
        "openai": {"pdf", "docx"},  # GPT-4 Vision handles these natively
        "anthropic": {"pdf", "docx"}  # Claude handles these natively
    }
    
    def __init__(
        self, 
        source: Union[str, Path],
        encoding: str = 'utf-8',
        mime_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize a text input.
        
        Args:
            source: File path, URL, or raw text content
            encoding: Text encoding (default: utf-8)
            mime_type: Optional explicit MIME type
            **kwargs: Additional processing options
        """
        self.source = source
        self.encoding = encoding
        self._mime_type = mime_type
        self._options = kwargs
        
        # Caching
        self._cached_raw = None  # Raw content (bytes)
        self._cached_text = None  # Processed text content
        self._cached_formats = {}  # Provider-specific formats
        
        # Validate inputs
        if not isinstance(source, (str, Path)):
            raise ValueError(f"Text source must be a string or Path, got {type(source)}")
        
        # Detect format
        self._format = self._detect_format()
    
    @property
    def media_type(self) -> str:
        """Return the type of media."""
        return "text"
    
    @property
    def mime_type(self) -> str:
        """Get the MIME type of the content."""
        if self._mime_type:
            return self._mime_type
            
        return self.SUPPORTED_FORMATS.get(self._format, "text/plain")
    
    @property
    def needs_processing(self) -> bool:
        """Check if content needs special processing."""
        return self._format in {"pdf", "docx", "doc", "rtf"}
    
    def _detect_format(self) -> str:
        """
        Detect format from source.
        
        Returns:
            Format string (e.g., 'txt', 'pdf')
        """
        source_str = str(self.source)
        
        # Handle URLs
        if source_str.startswith(('http://', 'https://')):
            ext = os.path.splitext(source_str)[-1].lower().lstrip('.')
            if ext in self.SUPPORTED_FORMATS:
                return ext
            return "txt"  # Default for URLs
        
        # Handle file paths
        elif os.path.exists(source_str):
            # Use file extension
            ext = os.path.splitext(source_str)[-1].lower().lstrip('.')
            if ext in self.SUPPORTED_FORMATS:
                return ext
                
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(source_str)
            if mime_type:
                for fmt, mime in self.SUPPORTED_FORMATS.items():
                    if mime_type == mime:
                        return fmt
            
            return "txt"  # Default for files
            
        # Raw content
        return "txt"
    
    def get_raw_content(self) -> bytes:
        """
        Get raw content as bytes.
        
        Returns:
            Raw content
            
        Raises:
            MediaProcessingError: If content cannot be loaded
        """
        if self._cached_raw is not None:
            return self._cached_raw
            
        source_str = str(self.source)
        
        try:
            # Handle URLs
            if source_str.startswith(('http://', 'https://')):
                try:
                    import requests
                    headers = {
                        'User-Agent': 'AbstractLLM/0.1.0 (https://github.com/lpalbou/abstractllm)'
                    }
                    response = requests.get(source_str, headers=headers, timeout=10)
                    response.raise_for_status()
                    self._cached_raw = response.content
                except ImportError:
                    raise MediaProcessingError(
                        "Requests library not available. Install with: pip install requests"
                    )
                except Exception as e:
                    raise MediaProcessingError(f"Failed to download content: {e}")
            
            # Handle file paths
            elif os.path.exists(source_str):
                try:
                    with open(source_str, 'rb') as f:
                        self._cached_raw = f.read()
                except Exception as e:
                    raise MediaProcessingError(f"Failed to read file: {e}")
            
            # Handle raw content
            else:
                self._cached_raw = source_str.encode(self.encoding)
            
            return self._cached_raw
            
        except MediaProcessingError:
            raise
        except Exception as e:
            raise MediaProcessingError(f"Unexpected error loading content: {e}")
    
    def get_text(self) -> str:
        """
        Get content as text, processing if needed.
        
        Returns:
            Text content
            
        Raises:
            MediaProcessingError: If text extraction fails
        """
        if self._cached_text is not None:
            return self._cached_text
            
        try:
            if not self.needs_processing:
                # Simple text decoding
                raw = self.get_raw_content()
                text = raw.decode(self.encoding)
            else:
                # Process based on format
                if self._format == "pdf":
                    text = self._extract_pdf_text()
                elif self._format in ["docx", "doc"]:
                    text = self._extract_doc_text()
                else:
                    raise MediaProcessingError(f"Unsupported format for text extraction: {self._format}")
            
            self._cached_text = text
            return text
            
        except MediaProcessingError:
            raise
        except Exception as e:
            raise MediaProcessingError(f"Failed to extract text: {e}")
    
    def _extract_pdf_text(self) -> str:
        """Extract text from PDF content."""
        try:
            import PyPDF2
            raw = self.get_raw_content()
            
            with BytesIO(raw) as stream:
                reader = PyPDF2.PdfReader(stream)
                text = "\n".join(
                    page.extract_text() 
                    for page in reader.pages
                )
            return text
            
        except ImportError:
            raise MediaProcessingError(
                "PyPDF2 library not available. Install with: pip install PyPDF2"
            )
        except Exception as e:
            raise MediaProcessingError(f"Failed to extract PDF text: {e}")
    
    def _extract_doc_text(self) -> str:
        """Extract text from DOC/DOCX content."""
        try:
            from docx import Document  # python-docx package
            raw = self.get_raw_content()
            
            with BytesIO(raw) as stream:
                doc = Document(stream)
                text = "\n".join(
                    paragraph.text 
                    for paragraph in doc.paragraphs
                )
            return text
            
        except ImportError:
            raise MediaProcessingError(
                "python-docx library not available. Install with: pip install python-docx"
            )
        except Exception as e:
            raise MediaProcessingError(f"Failed to extract DOC/DOCX text: {e}")
    
    def to_provider_format(self, provider: str) -> Any:
        """
        Convert the content to provider-specific format.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider-specific format
            
        Raises:
            ValueError: If provider not supported
            MediaProcessingError: If formatting fails
        """
        # Return cached format if available
        if provider in self._cached_formats:
            return self._cached_formats[provider]
        
        try:
            # For providers with native document handling
            if (provider in self.NATIVE_HANDLERS and 
                self._format in self.NATIVE_HANDLERS[provider]):
                format_result = self._format_for_native_handler(provider)
            else:
                # For text-based handling
                format_result = self._format_for_text_handler(provider)
            
            # Cache and return
            self._cached_formats[provider] = format_result
            return format_result
            
        except Exception as e:
            raise MediaProcessingError(f"Failed to format for {provider}: {e}")
    
    def _format_for_native_handler(self, provider: str) -> Dict[str, Any]:
        """Format for providers with native document handling."""
        source_str = str(self.source)
        
        # For URLs, use them directly
        if source_str.startswith(('http://', 'https://')):
            return {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": source_str
                },
                "mime_type": self.mime_type
            }
        
        # For files, provide the content
        raw = self.get_raw_content()
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": raw
            }
        }
    
    def _format_for_text_handler(self, provider: str) -> Union[str, Dict[str, Any]]:
        """Format for text-based handling."""
        text = self.get_text()
        
        if provider == "openai":
            return {
                "type": "text",
                "text": text
            }
        elif provider == "anthropic":
            return {
                "type": "text",
                "text": text
            }
        elif provider == "ollama":
            source_name = Path(str(self.source)).name
            return f"\n===== {source_name} =====\n{text}\n"
        elif provider == "huggingface":
            return {
                "type": "text",
                "content": text,
                "mime_type": self.mime_type
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the content."""
        metadata = {
            "media_type": "text",
            "mime_type": self.mime_type,
            "format": self._format,
            "encoding": self.encoding,
            "needs_processing": self.needs_processing
        }
        
        # Add file metadata if available
        source_str = str(self.source)
        if os.path.exists(source_str):
            metadata.update({
                "file_size": os.path.getsize(source_str),
                "last_modified": os.path.getmtime(source_str)
            })
        
        # Add content metadata if available
        if self._cached_raw is not None:
            metadata["raw_size"] = len(self._cached_raw)
        if self._cached_text is not None:
            metadata["text_length"] = len(self._cached_text)
        
        return metadata 