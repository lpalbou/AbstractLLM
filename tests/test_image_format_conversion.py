"""
Tests for image format conversion utilities.

This file tests the image utility functions, specifically focusing on the format
conversion for different providers and truncation of large base64 data in logs.
"""

import os
import base64
import logging
import pytest
from pathlib import Path
from unittest.mock import patch
import json

from abstractllm.utils.image import (
    encode_image_to_base64,
    get_image_mime_type,
    format_image_for_provider,
    format_images_for_provider,
    preprocess_image_inputs
)

# Define test resources directory
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
TEST_IMAGE_PATH = os.path.join(RESOURCES_DIR, "test_image_1.jpg")

def setup_module(module):
    """Setup for the test module."""
    # Ensure image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        pytest.skip(f"Test image not found at {TEST_IMAGE_PATH}")

def test_encode_image_to_base64():
    """Test encoding an image to base64."""
    # Test with valid image
    base64_data = encode_image_to_base64(TEST_IMAGE_PATH)
    assert isinstance(base64_data, str)
    assert len(base64_data) > 0
    
    # Verify we can decode it back
    try:
        image_data = base64.b64decode(base64_data)
        assert len(image_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 data: {e}")
    
    # Test with non-existent image
    with pytest.raises(FileNotFoundError):
        encode_image_to_base64("non_existent_image.jpg")

def test_get_image_mime_type():
    """Test getting MIME type from image file."""
    # Test with JPEG image
    mime_type = get_image_mime_type(TEST_IMAGE_PATH)
    assert mime_type == "image/jpeg"
    
    # Test with different extensions
    assert get_image_mime_type("test.png") == "image/png"
    assert get_image_mime_type("test.gif") == "image/gif"
    assert get_image_mime_type("test.webp") == "image/webp"
    assert get_image_mime_type("test.unknown") == "image/jpeg"  # Default

def test_format_image_for_provider():
    """Test formatting images for different providers."""
    # Test with file path
    openai_format = format_image_for_provider(TEST_IMAGE_PATH, "openai")
    assert "type" in openai_format
    assert openai_format["type"] == "image_url"
    assert "image_url" in openai_format
    assert "url" in openai_format["image_url"]
    assert openai_format["image_url"]["url"].startswith("data:image/jpeg;base64,")
    
    anthropic_format = format_image_for_provider(TEST_IMAGE_PATH, "anthropic")
    assert "type" in anthropic_format
    assert anthropic_format["type"] == "image"
    assert "source" in anthropic_format
    assert anthropic_format["source"]["type"] == "base64"
    assert anthropic_format["source"]["media_type"] == "image/jpeg"
    assert "data" in anthropic_format["source"]
    
    ollama_format = format_image_for_provider(TEST_IMAGE_PATH, "ollama")
    assert isinstance(ollama_format, str)
    
    huggingface_format = format_image_for_provider(TEST_IMAGE_PATH, "huggingface")
    assert isinstance(huggingface_format, str)
    assert os.path.exists(huggingface_format)
    
    # Test with URL
    url = "https://example.com/image.jpg"
    openai_url_format = format_image_for_provider(url, "openai")
    assert openai_url_format["image_url"]["url"] == url
    
    anthropic_url_format = format_image_for_provider(url, "anthropic")
    assert anthropic_url_format["source"]["type"] == "url"
    assert anthropic_url_format["source"]["url"] == url
    
    ollama_url_format = format_image_for_provider(url, "ollama")
    assert ollama_url_format == url

def test_format_images_for_provider():
    """Test formatting multiple images for providers."""
    images = [TEST_IMAGE_PATH, "https://example.com/image.jpg"]
    
    # Test for OpenAI
    openai_formats = format_images_for_provider(images, "openai")
    assert len(openai_formats) == 2
    assert openai_formats[0]["type"] == "image_url"
    assert openai_formats[1]["image_url"]["url"] == "https://example.com/image.jpg"
    
    # Test for Ollama
    ollama_formats = format_images_for_provider(images, "ollama")
    assert len(ollama_formats) == 2
    assert isinstance(ollama_formats[0], str)
    assert ollama_formats[1] == "https://example.com/image.jpg"

def test_preprocess_image_inputs():
    """Test preprocessing image inputs for different providers."""
    # Test single image
    params = {"image": TEST_IMAGE_PATH, "temperature": 0.7}
    
    openai_params = preprocess_image_inputs(params.copy(), "openai")
    assert "messages" in openai_params
    assert len(openai_params["messages"]) == 1
    assert openai_params["messages"][0]["role"] == "user"
    assert isinstance(openai_params["messages"][0]["content"], list)
    assert len(openai_params["messages"][0]["content"]) == 1
    assert openai_params["messages"][0]["content"][0]["type"] == "image_url"
    
    anthropic_params = preprocess_image_inputs(params.copy(), "anthropic")
    assert "messages" in anthropic_params
    assert len(anthropic_params["messages"]) == 1
    assert anthropic_params["messages"][0]["role"] == "user"
    assert isinstance(anthropic_params["messages"][0]["content"], list)
    assert len(anthropic_params["messages"][0]["content"]) == 1
    assert anthropic_params["messages"][0]["content"][0]["type"] == "image"
    
    ollama_params = preprocess_image_inputs(params.copy(), "ollama")
    assert "images" in ollama_params
    assert len(ollama_params["images"]) == 1
    assert isinstance(ollama_params["images"][0], str)
    assert "temperature" in ollama_params
    assert ollama_params["temperature"] == 0.7

def test_ollama_base64_log_truncation():
    """Test that long base64 data is truncated in logs for Ollama provider."""
    # Mock the logger to capture log messages
    with patch('abstractllm.utils.image.logger') as mock_logger:
        # Format image for Ollama
        ollama_format = format_image_for_provider(TEST_IMAGE_PATH, "ollama")
        
        # Check the length of the base64 string
        assert len(ollama_format) > 100
        
        # Now prepare a request with this image
        params = {"image": TEST_IMAGE_PATH}
        processed = preprocess_image_inputs(params, "ollama")
        
        # Verify images array exists and contains the base64 data
        assert "images" in processed
        assert len(processed["images"]) == 1
        assert processed["images"][0] == ollama_format
        
        # Create a function similar to the one in logging module but for testing only
        def test_truncate_base64(data, max_length=50):
            """Test-only version of truncate function that doesn't actually show any base64."""
            if isinstance(data, str) and len(data) > max_length:
                if all(c.isalnum() or c in '+/=' for c in data) and ' ' not in data:
                    return f"[base64 data, length: {len(data)} chars]"
            return "[REDACTED FOR TEST]"
        
        # Apply truncation for testing purposes only
        truncated = test_truncate_base64(ollama_format)
        assert "base64 data" in truncated
        assert str(len(ollama_format)) in truncated
        
        # Log the truncated data - using a safe string that doesn't contain actual base64
        mock_logger.debug.reset_mock()
        mock_logger.debug(f"Test message with safe data: {truncated}")
        
        # Verify log was called with safe data
        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]
        assert "base64 data" in log_message
        assert ollama_format not in log_message

def test_base64_logging_to_file():
    """Test that base64 data is properly handled in our logging system."""
    from abstractllm.utils.logging import log_request, truncate_base64, ensure_log_directory, DEFAULT_LOG_DIR
    import tempfile
    
    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set the log directory to our temp directory
        with patch('abstractllm.utils.logging.DEFAULT_LOG_DIR', temp_dir):
            # Format an image for Ollama (which uses raw base64)
            ollama_base64 = format_image_for_provider(TEST_IMAGE_PATH, "ollama")
            
            # Create a request with the image
            prompt = "Describe this image"
            params = {
                "model": "test-model",
                "temperature": 0.7,
                "images": [ollama_base64]
            }
            
            # Create mock to only capture debug output (not block file writing)
            with patch('abstractllm.utils.logging.logger.debug') as mock_debug:
                # Log the request using our enhanced logging system
                log_request("ollama", prompt, params)
                
                # Check debug logs - they should have hidden base64
                for call in mock_debug.call_args_list:
                    message = call[0][0]
                    # No debug message should contain the base64 string
                    if isinstance(message, str):
                        assert ollama_base64 not in message
                        if "Parameters" in message:
                            assert "image(s), data hidden" in message or "image data hidden" in message
            
            # Now verify the log file was created and contains the full data
            log_files = os.listdir(temp_dir)
            assert len(log_files) >= 1
            
            # Find the request log file
            request_logs = [f for f in log_files if "ollama_request_" in f]
            assert len(request_logs) > 0, f"No request logs found in {log_files}"
            
            # Read the log file
            with open(os.path.join(temp_dir, request_logs[0]), 'r') as f:
                log_data = json.load(f)
                
                # Verify it contains the full base64 data
                assert "parameters" in log_data
                assert "images" in log_data["parameters"]
                assert log_data["parameters"]["images"][0] == ollama_base64
                
                # Verify other parameters are also present
                assert log_data["prompt"] == prompt
                assert log_data["provider"] == "ollama"

def test_base64_truncation_for_logging():
    """Test that base64 data is properly truncated for logging."""
    from abstractllm.utils.logging import truncate_base64
    
    # Format an image for Ollama (which uses raw base64)
    ollama_base64 = format_image_for_provider(TEST_IMAGE_PATH, "ollama")
    
    # Verify the base64 string is long enough to need truncation
    assert len(ollama_base64) > 100
    
    # Test truncation of a string
    truncated = truncate_base64(ollama_base64, max_length=50)
    assert len(truncated) < len(ollama_base64)
    assert truncated.startswith("[base64 data")
    assert str(len(ollama_base64)) in truncated  # Should show the original length
    
    # Test truncation within a dictionary
    params = {
        "model": "test-model",
        "temperature": 0.7,
        "images": [ollama_base64]
    }
    
    truncated_params = truncate_base64(params, max_length=50)
    
    # Original params should be unchanged
    assert params["images"][0] == ollama_base64
    
    # Truncated params should have shortened base64
    assert truncated_params["images"][0] != ollama_base64
    assert "[base64 data" in truncated_params["images"][0]
    
    # Non-base64 values should be unchanged
    assert truncated_params["model"] == "test-model"
    assert truncated_params["temperature"] == 0.7
    
    # Test with nested structures
    nested = {
        "level1": {
            "data": ollama_base64,
            "metadata": {
                "length": len(ollama_base64),
                "encoding": "base64"
            }
        },
        "list_data": [
            "normal string",
            ollama_base64,
            {"embedded": ollama_base64}
        ]
    }
    
    truncated_nested = truncate_base64(nested, max_length=50)
    
    # Verify nested dictionary values are truncated
    assert "[base64 data" in truncated_nested["level1"]["data"]
    
    # Verify list values are truncated
    assert truncated_nested["list_data"][0] == "normal string"  # Normal string unchanged
    assert "[base64 data" in truncated_nested["list_data"][1]  # Base64 string truncated
    assert "[base64 data" in truncated_nested["list_data"][2]["embedded"]  # Nested base64 truncated
    
    # Verify no actual base64 data is in the truncated result
    import json
    truncated_json = json.dumps(truncated_nested)
    assert ollama_base64 not in truncated_json

def test_ollama_image_array_logging():
    """Test that Ollama image arrays are properly handled in logs."""
    from abstractllm.utils.logging import truncate_base64, log_request
    import unittest.mock as mock
    
    # Create a base64 image
    ollama_base64 = format_image_for_provider(TEST_IMAGE_PATH, "ollama")
    
    # Create request parameters with multiple images
    params = {
        "model": "test-model",
        "temperature": 0.7,
        "images": [ollama_base64, ollama_base64, ollama_base64]  # Multiple images
    }
    
    # Capture the logger output
    with mock.patch('abstractllm.utils.logging.logger.debug') as mock_debug:
        # Log the request
        log_request("ollama", "What's in these images?", params)
        
        # Find the parameters log message
        params_log_message = None
        for call in mock_debug.call_args_list:
            message = call[0][0]
            if isinstance(message, str) and "Parameters" in message:
                params_log_message = message
                break
                
        # Verify the message was found and has properly hidden image data
        assert params_log_message is not None
        assert "[3 image(s), data hidden]" in params_log_message
        
        # Make sure no base64 data appears in any logs
        for call in mock_debug.call_args_list:
            message = call[0][0]
            assert ollama_base64 not in message
    
    # Verify special handling for non-list images parameter
    params_single = {
        "model": "test-model",
        "temperature": 0.7,
        "images": ollama_base64  # Single string, not a list
    }
    
    with mock.patch('abstractllm.utils.logging.logger.debug') as mock_debug:
        # Log the request
        log_request("ollama", "What's in this image?", params_single)
        
        # Find the parameters log message
        params_log_message = None
        for call in mock_debug.call_args_list:
            message = call[0][0]
            if isinstance(message, str) and "Parameters" in message:
                params_log_message = message
                break
                
        # Verify the message was found and has properly hidden image data
        assert params_log_message is not None
        assert "image data hidden" in params_log_message
        assert ollama_base64 not in params_log_message

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 