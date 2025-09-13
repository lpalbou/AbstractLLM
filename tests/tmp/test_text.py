"""
Tests for text file handling in AbstractLLM.
"""

import os
import pytest
import requests
from pathlib import Path
from typing import Dict, Any, List, Union

from abstractllm.media.text import TextInput
from abstractllm.media.factory import MediaFactory

# Get the examples directory path
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

def test_text_input_init():
    """Test TextInput initialization."""
    # Test with file path
    text_path = os.path.join(EXAMPLES_DIR, "test_document.txt")
    text_input = TextInput(text_path)
    assert text_input.media_type == "text"
    assert text_input.mime_type == "text/plain"
    
    # Test with raw text
    raw_text = "This is a test string"
    text_input = TextInput(raw_text)
    assert text_input.media_type == "text"
    assert text_input.mime_type == "text/plain"
    
    # Test with custom encoding
    text_input = TextInput(text_path, encoding="utf-16")
    assert text_input.encoding == "utf-16"
    
    # Test with explicit MIME type
    text_input = TextInput(text_path, mime_type="text/markdown")
    assert text_input.mime_type == "text/markdown"

def test_text_content_loading():
    """Test loading text content."""
    text_path = os.path.join(EXAMPLES_DIR, "test_document.txt")
    text_input = TextInput(text_path)
    
    # Get content
    content = text_input.get_content()
    
    # Verify content
    assert content is not None
    assert isinstance(content, str)
    assert len(content) > 0
    assert "This is a test document" in content
    
    # Test caching
    assert text_input._cached_content is not None
    assert text_input._cached_content == content

def test_text_provider_formatting():
    """Test text formatting for different providers."""
    text_path = os.path.join(EXAMPLES_DIR, "test_document.txt")
    text_input = TextInput(text_path)
    
    # Test OpenAI format
    openai_format = text_input.to_provider_format("openai")
    assert isinstance(openai_format, dict)
    assert openai_format["type"] == "text"
    assert isinstance(openai_format["text"], str)
    
    # Test Anthropic format
    anthropic_format = text_input.to_provider_format("anthropic")
    assert isinstance(anthropic_format, dict)
    assert anthropic_format["type"] == "text"
    assert isinstance(anthropic_format["content"], str)
    
    # Test Ollama format
    ollama_format = text_input.to_provider_format("ollama")
    assert isinstance(ollama_format, str)
    
    # Test HuggingFace format
    hf_format = text_input.to_provider_format("huggingface")
    assert isinstance(hf_format, str)
    
    # Test invalid provider
    with pytest.raises(ValueError):
        text_input.to_provider_format("invalid_provider")

def test_text_metadata():
    """Test text metadata."""
    text_path = os.path.join(EXAMPLES_DIR, "test_document.txt")
    text_input = TextInput(text_path)
    
    # Get metadata
    metadata = text_input.metadata
    
    # Verify metadata
    assert metadata["media_type"] == "text"
    assert metadata["mime_type"] == "text/plain"
    assert metadata["encoding"] == "utf-8"
    assert "file_size" in metadata
    assert "last_modified" in metadata
    
    # Test metadata with raw text
    raw_text = "This is a test string"
    text_input = TextInput(raw_text)
    metadata = text_input.metadata
    
    assert metadata["media_type"] == "text"
    assert metadata["mime_type"] == "text/plain"
    assert metadata["encoding"] == "utf-8"
    assert "file_size" not in metadata  # No file size for raw text
    assert "last_modified" not in metadata  # No modification time for raw text

def test_factory_text_creation():
    """Test creating text inputs through MediaFactory."""
    text_path = os.path.join(EXAMPLES_DIR, "test_document.txt")
    
    # Test with file path
    text_input = MediaFactory.from_source(text_path)
    assert isinstance(text_input, TextInput)
    assert text_input.media_type == "text"
    
    # Test with explicit media type
    text_input = MediaFactory.from_source(text_path, media_type="text")
    assert isinstance(text_input, TextInput)
    
    # Test with dictionary source
    text_dict = {
        "type": "text",
        "source": "This is a test string",
        "encoding": "utf-8"
    }
    text_input = MediaFactory.from_source(text_dict)
    assert isinstance(text_input, TextInput)
    
    # Test with raw text and explicit type
    raw_text = "This is a test string"
    text_input = MediaFactory.from_source(raw_text, media_type="text")
    assert isinstance(text_input, TextInput)

def test_text_error_handling():
    """Test error handling in text processing."""
    # Test with non-existent file
    with pytest.raises(Exception):
        text_input = TextInput("nonexistent.txt")
        text_input.get_content()
    
    # Test with invalid encoding
    with pytest.raises(Exception):
        text_input = TextInput(os.path.join(EXAMPLES_DIR, "test_document.txt"), encoding="invalid")
        text_input.get_content()
    
    # Test with invalid source type
    with pytest.raises(ValueError):
        TextInput(123)  # type: ignore
    
    # Test with empty string
    text_input = TextInput("")
    assert text_input.get_content() == "" 