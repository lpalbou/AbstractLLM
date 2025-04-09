"""
Tests for tabular data handling in AbstractLLM.
"""

import os
import pytest
import requests
from pathlib import Path
from typing import Dict, Any, List, Union

from abstractllm.media.tabular import TabularInput
from abstractllm.media.factory import MediaFactory

# Get the examples directory path
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

def test_tabular_input_init():
    """Test TabularInput initialization."""
    # Test with CSV file
    csv_path = os.path.join(EXAMPLES_DIR, "test_data.csv")
    csv_input = TabularInput(csv_path)
    assert csv_input.media_type == "tabular"
    assert csv_input.mime_type == "text/csv"
    assert csv_input.delimiter == ","
    
    # Test with TSV file
    tsv_path = os.path.join(EXAMPLES_DIR, "test_data.tsv")
    tsv_input = TabularInput(tsv_path, delimiter="\t")
    assert tsv_input.media_type == "tabular"
    assert tsv_input.mime_type == "text/tab-separated-values"
    assert tsv_input.delimiter == "\t"
    
    # Test with custom encoding
    csv_input = TabularInput(csv_path, encoding="utf-16")
    assert csv_input.encoding == "utf-16"
    
    # Test with explicit MIME type
    csv_input = TabularInput(csv_path, mime_type="text/csv")
    assert csv_input.mime_type == "text/csv"

def test_tabular_content_loading():
    """Test loading tabular content."""
    # Test CSV loading
    csv_path = os.path.join(EXAMPLES_DIR, "test_data.csv")
    csv_input = TabularInput(csv_path)
    
    # Get raw content
    content = csv_input.get_content()
    assert content is not None
    assert isinstance(content, str)
    assert len(content) > 0
    assert "Name,Age,City,Occupation" in content
    
    # Get parsed data
    data = csv_input.get_data()
    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0] == ["Name", "Age", "City", "Occupation"]
    
    # Test TSV loading
    tsv_path = os.path.join(EXAMPLES_DIR, "test_data.tsv")
    tsv_input = TabularInput(tsv_path, delimiter="\t")
    
    # Get raw content
    content = tsv_input.get_content()
    assert content is not None
    assert isinstance(content, str)
    assert len(content) > 0
    assert "Product\tCategory\tPrice\tStock" in content
    
    # Get parsed data
    data = tsv_input.get_data()
    assert data is not None
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0] == ["Product", "Category", "Price", "Stock"]
    
    # Test caching
    assert tsv_input._cached_content is not None
    assert tsv_input._cached_data is not None

def test_tabular_provider_formatting():
    """Test tabular data formatting for different providers."""
    csv_path = os.path.join(EXAMPLES_DIR, "test_data.csv")
    csv_input = TabularInput(csv_path)
    
    # Test OpenAI format (markdown table)
    openai_format = csv_input.to_provider_format("openai")
    assert isinstance(openai_format, dict)
    assert openai_format["type"] == "text"
    assert isinstance(openai_format["text"], str)
    assert "| Name | Age | City | Occupation |" in openai_format["text"]
    assert "| --- | --- | --- | --- |" in openai_format["text"]
    
    # Test Anthropic format (markdown table)
    anthropic_format = csv_input.to_provider_format("anthropic")
    assert isinstance(anthropic_format, dict)
    assert anthropic_format["type"] == "text"
    assert isinstance(anthropic_format["content"], str)
    assert "| Name | Age | City | Occupation |" in anthropic_format["content"]
    
    # Test Ollama format (markdown table)
    ollama_format = csv_input.to_provider_format("ollama")
    assert isinstance(ollama_format, str)
    assert "| Name | Age | City | Occupation |" in ollama_format
    
    # Test HuggingFace format (raw content)
    hf_format = csv_input.to_provider_format("huggingface")
    assert isinstance(hf_format, str)
    assert "Name,Age,City,Occupation" in hf_format
    
    # Test invalid provider
    with pytest.raises(ValueError):
        csv_input.to_provider_format("invalid_provider")

def test_tabular_metadata():
    """Test tabular data metadata."""
    csv_path = os.path.join(EXAMPLES_DIR, "test_data.csv")
    csv_input = TabularInput(csv_path)
    
    # Get metadata
    metadata = csv_input.metadata
    
    # Verify metadata
    assert metadata["media_type"] == "tabular"
    assert metadata["mime_type"] == "text/csv"
    assert metadata["encoding"] == "utf-8"
    assert metadata["delimiter"] == ","
    assert "file_size" in metadata
    assert "last_modified" in metadata
    
    # Load data to get row/column counts
    csv_input.get_data()
    metadata = csv_input.metadata
    assert metadata["row_count"] > 0
    assert metadata["column_count"] > 0
    
    # Test metadata with raw content
    raw_csv = "a,b,c\n1,2,3"
    csv_input = TabularInput(raw_csv)
    metadata = csv_input.metadata
    
    assert metadata["media_type"] == "tabular"
    assert metadata["mime_type"] == "text/csv"
    assert metadata["encoding"] == "utf-8"
    assert metadata["delimiter"] == ","
    assert "file_size" not in metadata  # No file size for raw content
    assert "last_modified" not in metadata  # No modification time for raw content

def test_factory_tabular_creation():
    """Test creating tabular inputs through MediaFactory."""
    csv_path = os.path.join(EXAMPLES_DIR, "test_data.csv")
    tsv_path = os.path.join(EXAMPLES_DIR, "test_data.tsv")
    
    # Test with CSV file path
    csv_input = MediaFactory.from_source(csv_path)
    assert isinstance(csv_input, TabularInput)
    assert csv_input.media_type == "tabular"
    assert csv_input.delimiter == ","
    
    # Test with TSV file path
    tsv_input = MediaFactory.from_source(tsv_path)
    assert isinstance(tsv_input, TabularInput)
    assert tsv_input.media_type == "tabular"
    assert tsv_input.delimiter == "\t"
    
    # Test with explicit media type
    csv_input = MediaFactory.from_source(csv_path, media_type="tabular")
    assert isinstance(csv_input, TabularInput)
    
    # Test with dictionary source
    csv_dict = {
        "type": "tabular",
        "source": "a,b,c\n1,2,3",
        "delimiter": ",",
        "encoding": "utf-8"
    }
    csv_input = MediaFactory.from_source(csv_dict)
    assert isinstance(csv_input, TabularInput)
    
    # Test with raw content and explicit type
    raw_csv = "a,b,c\n1,2,3"
    csv_input = MediaFactory.from_source(raw_csv, media_type="tabular")
    assert isinstance(csv_input, TabularInput)

def test_tabular_error_handling():
    """Test error handling in tabular data processing."""
    # Test with non-existent file
    with pytest.raises(Exception):
        tabular_input = TabularInput("nonexistent.csv")
        tabular_input.get_content()
    
    # Test with invalid encoding
    with pytest.raises(Exception):
        tabular_input = TabularInput(os.path.join(EXAMPLES_DIR, "test_data.csv"), encoding="invalid")
        tabular_input.get_content()
    
    # Test with invalid source type
    with pytest.raises(ValueError):
        TabularInput(123)  # type: ignore
    
    # Test with invalid CSV format
    with pytest.raises(Exception):
        tabular_input = TabularInput("invalid,csv,data\nwith,missing,columns,extra")
        tabular_input.get_data()
    
    # Test with empty string
    tabular_input = TabularInput("")
    assert tabular_input.get_content() == ""
    assert tabular_input.get_data() == [] 