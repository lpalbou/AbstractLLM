"""
Tabular data input implementation for AbstractLLM.

This module provides the TabularInput class for handling CSV and TSV files.
"""

import os
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Union, Optional, List
import io

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ImageProcessingError

# Configure logger
logger = logging.getLogger("abstractllm.media.tabular")

class TabularInput(MediaInput):
    """
    Class representing a tabular data input (CSV, TSV).
    
    This class handles different tabular data sources (file paths, URLs, raw content)
    and converts them to provider-specific formats.
    """
    
    def __init__(self, 
                source: Union[str, Path], 
                delimiter: str = ',',
                encoding: str = 'utf-8',
                mime_type: Optional[str] = None):
        """
        Initialize a tabular data input.
        
        Args:
            source: File path, URL, or raw content
            delimiter: Field delimiter (default: ',' for CSV)
            encoding: Text encoding (default: utf-8)
            mime_type: Optional explicit MIME type
        """
        self.source = source
        self.delimiter = delimiter
        self.encoding = encoding
        self._mime_type = mime_type
        self._cached_content = None
        self._cached_data = None
        
        # Validate inputs
        if not isinstance(source, (str, Path)):
            raise ValueError(f"Tabular source must be a string or Path, got {type(source)}")
    
    @property
    def media_type(self) -> str:
        """Return the type of media."""
        return "tabular"
    
    @property
    def mime_type(self) -> str:
        """
        Get the MIME type of the tabular data.
        
        Returns:
            MIME type string (e.g., 'text/csv', 'text/tab-separated-values')
        """
        if self._mime_type:
            return self._mime_type
            
        # Determine based on delimiter
        if self.delimiter == '\t':
            return 'text/tab-separated-values'
        return 'text/csv'
    
    def get_content(self) -> str:
        """
        Get the raw content of the tabular data.
        
        Returns:
            Raw content as string
            
        Raises:
            Exception: If the content cannot be loaded
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
                    raise Exception(f"Failed to download tabular data from URL: {e}")
                    
            # Handle file paths
            elif os.path.exists(source_str):
                try:
                    with open(source_str, 'r', encoding=self.encoding) as f:
                        self._cached_content = f.read()
                    return self._cached_content
                except Exception as e:
                    raise Exception(f"Failed to read tabular file: {e}")
                    
            # Handle raw content
            else:
                self._cached_content = source_str
                return self._cached_content
                
        except Exception as e:
            raise Exception(f"Failed to get tabular content: {e}")
    
    def get_data(self) -> List[List[str]]:
        """
        Get the parsed tabular data.
        
        Returns:
            List of rows, where each row is a list of strings
            
        Raises:
            Exception: If the data cannot be parsed
        """
        # Return cached data if available
        if self._cached_data is not None:
            return self._cached_data
            
        # Get raw content
        content = self.get_content()
        
        try:
            # Parse the content using csv module
            rows = []
            reader = csv.reader(io.StringIO(content), delimiter=self.delimiter)
            for row in reader:
                rows.append(row)
            
            self._cached_data = rows
            return rows
            
        except Exception as e:
            raise Exception(f"Failed to parse tabular data: {e}")
    
    def to_provider_format(self, provider: str) -> Any:
        """
        Convert the tabular data to a format suitable for the specified provider.
        
        Args:
            provider: The provider name
            
        Returns:
            Provider-specific format for the tabular data
            
        Raises:
            ValueError: If the provider is not supported
        """
        # Get the parsed data
        data = self.get_data()
        
        # Format for each provider
        if provider == "openai":
            # Format as a markdown table for OpenAI
            if not data:
                return {"type": "text", "text": ""}
                
            header = data[0]
            rows = data[1:]
            
            # Create markdown table
            table = "| " + " | ".join(header) + " |\n"
            table += "| " + " | ".join(["---"] * len(header)) + " |\n"
            for row in rows:
                table += "| " + " | ".join(row) + " |\n"
                
            return {
                "type": "text",
                "text": table
            }
            
        elif provider == "anthropic":
            # Format as a markdown table for Anthropic
            if not data:
                return {"type": "text", "text": ""}
                
            header = data[0]
            rows = data[1:]
            
            # Create markdown table
            table = "| " + " | ".join(header) + " |\n"
            table += "| " + " | ".join(["---"] * len(header)) + " |\n"
            for row in rows:
                table += "| " + " | ".join(row) + " |\n"
                
            return {
                "type": "text",
                "text": table
            }
            
        elif provider == "ollama":
            # Format as a markdown table for Ollama
            if not data:
                return ""
                
            header = data[0]
            rows = data[1:]
            
            # Create markdown table
            table = "| " + " | ".join(header) + " |\n"
            table += "| " + " | ".join(["---"] * len(header)) + " |\n"
            for row in rows:
                table += "| " + " | ".join(row) + " |\n"
            
            # For Ollama, return a format that can be appended to the prompt
            source_name = str(self.source)
            if isinstance(self.source, Path):
                source_name = self.source.name
            
            return f"\n===== JOINT FILES ======\n\n===== {source_name} =========\n{table}\n"
            
        elif provider == "huggingface":
            # Format as a markdown table for HuggingFace
            if not data:
                return {
                    "type": "tabular",
                    "content": "",
                    "mime_type": self.mime_type,
                    "delimiter": self.delimiter
                }
                
            header = data[0]
            rows = data[1:]
            
            # Create markdown table
            table = "| " + " | ".join(header) + " |\n"
            table += "| " + " | ".join(["---"] * len(header)) + " |\n"
            for row in rows:
                table += "| " + " | ".join(row) + " |\n"
                
            return {
                "type": "tabular",
                "content": table,
                "mime_type": self.mime_type,
                "delimiter": self.delimiter
            }
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the tabular data.
        
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "media_type": "tabular",
            "mime_type": self.mime_type,
            "encoding": self.encoding,
            "delimiter": self.delimiter
        }
        
        # Add file-specific metadata if it's a file
        source_str = str(self.source)
        if os.path.exists(source_str):
            metadata.update({
                "file_size": os.path.getsize(source_str),
                "last_modified": os.path.getmtime(source_str)
            })
            
        # Add data statistics if we have parsed data
        if self._cached_data is not None:
            metadata.update({
                "row_count": len(self._cached_data),
                "column_count": len(self._cached_data[0]) if self._cached_data else 0
            })
            
        return metadata 