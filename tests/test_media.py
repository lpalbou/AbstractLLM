"""
Tests for media handling in AbstractLLM.

These tests verify that images and other media can be properly handled and formatted
for various providers.
"""

import os
import tempfile
import pytest
import json
import requests
import base64
import shutil
from pathlib import Path
from typing import Dict, Any, Union

from abstractllm import ImageInput, MediaFactory, MediaProcessor
from abstractllm.media.image import ImageInput
from abstractllm.exceptions import ImageProcessingError

# Function to get local test image paths instead of remote URLs
def get_local_test_image_path(filename: str) -> str:
    """Get the path to a local test image file.
    
    Args:
        filename: Name of the test image file
        
    Returns:
        Absolute path to the test image file
    """
    # Determine the project root directory
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_file_dir, "examples")
    
    # Return the path to the specified file
    return os.path.join(examples_dir, filename)

# Local image paths for testing
GITHUB_IMAGE_URL = get_local_test_image_path("test_image_1.jpg")
GITHUB_PNG_URL = get_local_test_image_path("test_image_2.jpg")  # Using jpg instead of png

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_image_path(temp_dir):
    """Create a test image and return its path."""
    test_file = os.path.join(temp_dir, "test_image.jpg")
    
    # Try to create a real test image using PIL if available
    try:
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_file, format='JPEG')
    except ImportError:
        # Fallback if PIL is not available - create a minimal JPEG file
        with open(test_file, 'wb') as f:
            f.write(base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVQIHWP4z8DwHwyBDAEAFigCQX+pjGUAAAAASUVORK5CYII="
            ))
    
    yield test_file
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)

def copy_test_image(source_path: str, output_path: str) -> bool:
    """Copy a test image for testing.
    
    Args:
        source_path: Path of the source image
        output_path: Path where the image should be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        shutil.copy2(source_path, output_path)
        return True
    except Exception as e:
        print(f"Error copying image: {e}")
        return False

@pytest.fixture
def github_image_url():
    """Fixture to provide a local image path for testing."""
    # Verify the file exists
    if not os.path.exists(GITHUB_IMAGE_URL):
        pytest.skip(f"Local test image not found: {GITHUB_IMAGE_URL}")
    
    return GITHUB_IMAGE_URL

@pytest.fixture
def github_png_url():
    """Fixture to provide a second local image path for testing."""
    # Verify the file exists
    if not os.path.exists(GITHUB_PNG_URL):
        pytest.skip(f"Local test image not found: {GITHUB_PNG_URL}")
    
    return GITHUB_PNG_URL

@pytest.fixture
def downloaded_github_image(temp_dir):
    """Copy and provide a local test image."""
    image_path = os.path.join(temp_dir, "github_image.jpg")
    
    if not copy_test_image(GITHUB_IMAGE_URL, image_path):
        pytest.skip("Failed to copy local test image")
    
    return image_path

@pytest.fixture
def downloaded_github_png(temp_dir):
    """Copy and provide a second local test image."""
    image_path = os.path.join(temp_dir, "github_image.jpg")  # Using jpg instead of png
    
    if not copy_test_image(GITHUB_PNG_URL, image_path):
        pytest.skip("Failed to copy local test image")
    
    return image_path

@pytest.fixture
def real_remote_image_url():
    """Fixture to provide a real remote image URL.
    
    This uses a publicly accessible image URL that should be stable.
    Skips the test if the URL is not accessible.
    """
    # URL to a publicly accessible image that should be relatively stable
    url = "https://raw.githubusercontent.com/python/pythondotorg/main/static/img/python-logo.png"
    
    # Check if the URL is accessible
    try:
        response = _make_request(url, method='head')
        if response.status_code != 200:
            pytest.skip(f"Remote image URL not accessible: {url}, status code: {response.status_code}")
        return url
    except requests.RequestException as e:
        pytest.skip(f"Error accessing remote image URL: {e}")

def _make_request(url: str, method: str = 'head') -> requests.Response:
    """Make HTTP request with proper User-Agent headers."""
    headers = {
        'User-Agent': 'AbstractLLM Test/0.1.0 (https://github.com/lpalbou/abstractllm)'
    }
    
    if method.lower() == 'head':
        return requests.head(url, headers=headers)
    elif method.lower() == 'get':
        return requests.get(url, headers=headers)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

class TestImageInput:
    """Tests for the ImageInput class."""
    
    def test_init_with_path(self, test_image_path):
        """Test initialization with a local file path."""
        image = ImageInput(test_image_path)
        assert image.source == test_image_path
        assert image.detail_level == "auto"
        assert image.mime_type.startswith("image/")
    
    def test_init_with_url(self, github_image_url, github_png_url):
        """Test initialization with different image paths."""
        # Test local image paths
        jpeg_image = ImageInput(github_image_url)
        assert jpeg_image.source == github_image_url
        assert jpeg_image.mime_type == "image/jpeg"
        
        # Test local image 2
        jpg_image = ImageInput(github_png_url)
        assert jpg_image.source == github_png_url
        assert jpg_image.mime_type == "image/jpeg"
    
    def test_init_with_remote_url(self, real_remote_image_url):
        """Test initialization with an actual remote URL."""
        try:
            # Create image input with remote URL
            image = ImageInput(real_remote_image_url)
            
            # Verify properties
            assert image.source == real_remote_image_url
            assert image.detail_level == "auto"
            assert image.mime_type == "image/png"
            
            # Test getting content from the URL
            content = image.get_content()
            assert len(content) > 0
            
            # Test base64 encoding - this returns raw base64 without the MIME prefix
            base64_data = image.get_base64()
            # Verify it's a non-empty string
            assert isinstance(base64_data, str)
            assert len(base64_data) > 0
            # Verify it's valid base64
            try:
                decoded = base64.b64decode(base64_data)
                assert len(decoded) > 0
            except Exception as e:
                pytest.fail(f"Failed to decode base64 data: {e}")
            
            # Test formatting for different providers
            openai_format = image.to_provider_format("openai")
            assert openai_format["type"] == "image_url"
            # OpenAI can use either a remote URL directly or a base64 data URL
            url = openai_format["image_url"]["url"]
            assert url == real_remote_image_url or url.startswith(f"data:{image.mime_type};base64,")
            
            anthropic_format = image.to_provider_format("anthropic")
            assert anthropic_format["type"] == "image"
            # Anthropic supports either base64 or URL source
            if anthropic_format["source"]["type"] == "base64":
                assert "data" in anthropic_format["source"]
                assert "media_type" in anthropic_format["source"]
            else:
                assert anthropic_format["source"]["type"] == "url"
                assert anthropic_format["source"]["url"] == real_remote_image_url
            
            ollama_format = image.to_provider_format("ollama")
            assert isinstance(ollama_format, str)
            assert len(ollama_format) > 0
            
            huggingface_format = image.to_provider_format("huggingface")
            # HuggingFace can use either the URL or a downloaded copy
            assert huggingface_format == real_remote_image_url or os.path.exists(huggingface_format)
        
        except (ImageProcessingError, requests.RequestException) as e:
            pytest.skip(f"Error testing with remote URL: {e}")
    
    def test_format_for_openai(self, test_image_path, github_image_url, github_png_url):
        """Test formatting for OpenAI with different image types."""
        # Test with local file
        local_image = ImageInput(test_image_path)
        local_formatted = local_image.to_provider_format("openai")
        assert isinstance(local_formatted, dict)
        assert local_formatted["type"] == "image_url"
        assert local_formatted["image_url"]["url"].startswith("data:")
        
        # Test with local test image 1
        jpeg_image = ImageInput(github_image_url)
        jpeg_formatted = jpeg_image.to_provider_format("openai")
        assert isinstance(jpeg_formatted, dict)
        assert jpeg_formatted["type"] == "image_url"
        assert jpeg_formatted["image_url"]["url"].startswith("data:")
        
        # Test with local test image 2
        jpg_image = ImageInput(github_png_url)
        jpg_formatted = jpg_image.to_provider_format("openai")
        assert isinstance(jpg_formatted, dict)
        assert jpg_formatted["type"] == "image_url"
        assert jpg_formatted["image_url"]["url"].startswith("data:")
    
    def test_format_for_anthropic(self, test_image_path, github_image_url, github_png_url):
        """Test formatting for Anthropic with different image types."""
        # Test with local file
        local_image = ImageInput(test_image_path)
        local_formatted = local_image.to_provider_format("anthropic")
        assert isinstance(local_formatted, dict)
        assert local_formatted["type"] == "image"
        assert local_formatted["source"]["type"] == "base64"
        
        # Test with local test image 1
        jpeg_image = ImageInput(github_image_url)
        jpeg_formatted = jpeg_image.to_provider_format("anthropic")
        assert isinstance(jpeg_formatted, dict)
        assert jpeg_formatted["type"] == "image"
        assert jpeg_formatted["source"]["type"] == "base64"
        
        # Test with local test image 2
        jpg_image = ImageInput(github_png_url)
        jpg_formatted = jpg_image.to_provider_format("anthropic")
        assert isinstance(jpg_formatted, dict)
        assert jpg_formatted["type"] == "image"
        assert jpg_formatted["source"]["type"] == "base64"
    
    def test_format_for_ollama(self, test_image_path, github_image_url, github_png_url):
        """Test formatting for Ollama with different image types."""
        # Test with local file
        local_image = ImageInput(test_image_path)
        local_formatted = local_image.to_provider_format("ollama")
        assert isinstance(local_formatted, str)
        assert len(local_formatted) > 0  # Should be base64 data
        
        # Test with local test image 1
        jpeg_image = ImageInput(github_image_url)
        jpeg_formatted = jpeg_image.to_provider_format("ollama")
        assert isinstance(jpeg_formatted, str)
        assert len(jpeg_formatted) > 0  # Should be base64 data
        
        # Test with local test image 2
        jpg_image = ImageInput(github_png_url)
        jpg_formatted = jpg_image.to_provider_format("ollama")
        assert isinstance(jpg_formatted, str)
        assert len(jpg_formatted) > 0  # Should be base64 data
    
    def test_format_for_huggingface(self, test_image_path, github_image_url, github_png_url):
        """Test formatting for HuggingFace with different image types."""
        # Test with local file
        local_image = ImageInput(test_image_path)
        local_formatted = local_image.to_provider_format("huggingface")
        assert local_formatted == test_image_path  # Should be the file path
        
        # Test with local test image 1
        jpeg_image = ImageInput(github_image_url)
        jpeg_formatted = jpeg_image.to_provider_format("huggingface")
        assert jpeg_formatted == github_image_url  # Should be the file path
        
        # Test with local test image 2
        jpg_image = ImageInput(github_png_url)
        jpg_formatted = jpg_image.to_provider_format("huggingface")
        assert jpg_formatted == github_png_url  # Should be the file path

class TestMediaFactory:
    """Tests for the MediaFactory class."""
    
    def test_from_source_with_path(self, test_image_path):
        """Test creating media input from a local file path."""
        media = MediaFactory.from_source(test_image_path)
        assert isinstance(media, ImageInput)
        assert media.source == test_image_path
    
    def test_from_source_with_url(self, github_image_url, github_png_url):
        """Test creating media input from URL sources."""
        # Create from local file path (which the fixtures now provide)
        media1 = MediaFactory.from_source(github_image_url)
        assert isinstance(media1, ImageInput)
        assert media1.source == github_image_url
        
        # Create from second local file path
        media2 = MediaFactory.from_source(github_png_url)
        assert isinstance(media2, ImageInput)
        assert media2.source == github_png_url
    
    def test_from_source_with_explicit_type(self, github_image_url):
        """Test creating media input with explicit media type."""
        try:
            # Use "image" media type explicitly
            media = MediaFactory.from_source(github_image_url, media_type="image")
            assert isinstance(media, ImageInput)
            assert media.source == github_image_url
            
            # Test with invalid media type
            with pytest.raises(ValueError):
                MediaFactory.from_source(github_image_url, media_type="invalid_type")
        except (ImageProcessingError, requests.RequestException) as e:
            pytest.skip(f"Error creating media with explicit type: {e}")
    
    def test_from_source_with_media_input(self):
        """Test that from_source returns the input if it's already a MediaInput."""
        # Create an ImageInput
        image = ImageInput("test/path.jpg")
        
        # Pass it to from_source
        result = MediaFactory.from_source(image)
        
        # Should return the same instance
        assert result is image
    
    def test_from_sources_list(self, github_image_url, github_png_url):
        """Test creating media inputs from a list of sources."""
        # Create from a list of local file paths
        sources = [github_image_url, github_png_url]
        media_list = MediaFactory.from_sources(sources)
        
        assert len(media_list) == 2
        assert all(isinstance(m, ImageInput) for m in media_list)
        assert media_list[0].source == github_image_url
        assert media_list[1].source == github_png_url
        
        # Test with mixed source types
        mixed_sources = [
            github_image_url,  # Local file path
            {"source": github_png_url, "type": "image"}  # Dictionary format
        ]
        mixed_list = MediaFactory.from_sources(mixed_sources)
        
        assert len(mixed_list) == 2
        assert all(isinstance(m, ImageInput) for m in mixed_list)
        assert mixed_list[0].source == github_image_url
        assert mixed_list[1].source == github_png_url
    
    def test_from_source_with_dict(self, github_image_url):
        """Test creating media input from dictionary specification."""
        try:
            # Create with basic dict
            media1 = MediaFactory.from_source({
                "source": github_image_url,
                "type": "image"
            })
            assert isinstance(media1, ImageInput)
            assert media1.source == github_image_url
            
            # Create with more options
            media2 = MediaFactory.from_source({
                "source": github_image_url,
                "type": "image",
                "detail_level": "high"
            })
            assert isinstance(media2, ImageInput)
            assert media2.source == github_image_url
            assert media2.detail_level == "high"
            
            # Test invalid dict (missing source)
            with pytest.raises(ValueError):
                MediaFactory.from_source({"type": "image"})
            
            # Test invalid dict (missing type)
            with pytest.raises(ValueError):
                MediaFactory.from_source({"source": github_image_url})
        except (ImageProcessingError, requests.RequestException) as e:
            pytest.skip(f"Error creating media from dict: {e}")
    
    def test_from_source_with_remote_url(self, real_remote_image_url):
        """Test creating media input from an actual remote URL."""
        try:
            # Create media input from remote URL
            media = MediaFactory.from_source(real_remote_image_url)
            
            # Verify the media input
            assert isinstance(media, ImageInput)
            assert media.source == real_remote_image_url
            assert media.mime_type == "image/png"
            
            # Test with explicit media type
            media_explicit = MediaFactory.from_source(real_remote_image_url, media_type="image")
            assert isinstance(media_explicit, ImageInput)
            assert media_explicit.source == real_remote_image_url
        
        except (ImageProcessingError, requests.RequestException) as e:
            pytest.skip(f"Error creating media from remote URL: {e}")

class TestMediaProcessor:
    """Tests for the MediaProcessor class."""
    
    def test_process_local_image_ollama(self, test_image_path):
        """Test processing a local image for Ollama."""
        # Create test params
        params = {
            "prompt": "Describe this image",
            "image": test_image_path
        }
        
        # Process for Ollama
        processed = MediaProcessor.process_inputs(params, "ollama")
        
        # Verify the image was processed correctly
        assert "image" in processed
        assert isinstance(processed["image"], str)
        assert len(processed["image"]) > 0  # Should be base64 data
    
    def test_process_remote_image_ollama(self, github_image_url):
        """Test processing a local image file for Ollama."""
        # Create test params with a local file path
        params = {
            "prompt": "Describe this image",
            "image": github_image_url
        }
        
        # Process for Ollama
        processed = MediaProcessor.process_inputs(params, "ollama")
        
        # Verify the image was processed correctly
        assert "image" in processed
        # For a local file, Ollama will use base64 encoding
        assert isinstance(processed["image"], str)
        assert len(processed["image"]) > 0  # Should be base64 data
    
    def test_process_multiple_images(self, test_image_path, github_image_url):
        """Test processing multiple images."""
        try:
            # Create test params with multiple images
            params = {
                "prompt": "Describe these images",
                "images": [test_image_path, github_image_url]
            }
            
            # Test with OpenAI
            processed_openai = MediaProcessor.process_inputs(params, "openai")
            assert "messages" in processed_openai
            content = processed_openai["messages"][0]["content"]
            assert isinstance(content, list)
            assert len(content) == 3  # Text + 2 images
            
            # Check first item is text
            assert content[0]["type"] == "text"
            
            # Check image items
            assert content[1]["type"] == "image_url"
            assert content[2]["type"] == "image_url"
            
            # Test with Anthropic
            processed_anthropic = MediaProcessor.process_inputs(params, "anthropic")
            assert "messages" in processed_anthropic
            content = processed_anthropic["messages"][0]["content"]
            assert isinstance(content, list)
            assert len(content) == 3  # Text + 2 images
            
            # Check image items
            assert content[1]["type"] == "image"
            assert content[2]["type"] == "image"
        except (ImageProcessingError, requests.RequestException) as e:
            pytest.skip(f"Error processing multiple images: {e}")
    
    def test_process_single_image_openai(self, test_image_path, github_image_url):
        """Test processing a single image for OpenAI."""
        # Test with local file
        params_local = {
            "prompt": "Describe this image",
            "image": test_image_path
        }
        processed_local = MediaProcessor.process_inputs(params_local, "openai")
        assert "messages" in processed_local
        content = processed_local["messages"][0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # Text + image
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:")
        
        # Test with local test image
        params_remote = {
            "prompt": "Describe this image",
            "image": github_image_url
        }
        processed_remote = MediaProcessor.process_inputs(params_remote, "openai")
        content = processed_remote["messages"][0]["content"]
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:")
    
    def test_process_single_image_anthropic(self, test_image_path, github_image_url):
        """Test processing a single image for Anthropic."""
        # Test with local file
        params_local = {
            "prompt": "Describe this image",
            "image": test_image_path
        }
        processed_local = MediaProcessor.process_inputs(params_local, "anthropic")
        assert "messages" in processed_local
        content = processed_local["messages"][0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # Text + image
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        
        # Test with local test image
        params_remote = {
            "prompt": "Describe this image",
            "image": github_image_url
        }
        processed_remote = MediaProcessor.process_inputs(params_remote, "anthropic")
        content = processed_remote["messages"][0]["content"]
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
    
    def test_process_single_image_huggingface(self, test_image_path):
        """Test processing a single image for HuggingFace."""
        params = {
            "prompt": "Describe this image",
            "image": test_image_path
        }
        processed = MediaProcessor.process_inputs(params, "huggingface")
        
        # HuggingFace uses a simplified approach
        assert "image" in processed
        assert "prompt" in processed
    
    def test_process_with_existing_messages(self, test_image_path):
        """Test processing images with existing messages format."""
        # Create params with existing messages
        params = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Describe this image"
                }
            ],
            "image": test_image_path
        }
        
        # Process for OpenAI
        processed = MediaProcessor.process_inputs(params, "openai")
        
        # Verify structure was preserved and image was added
        assert "messages" in processed
        assert len(processed["messages"]) == 2
        assert processed["messages"][0]["role"] == "system"
        assert processed["messages"][1]["role"] == "user"
        
        # Check that image was added to the user message
        user_content = processed["messages"][1]["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 2
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image_url"
    
    def test_process_empty_image_list(self):
        """Test processing with an empty image list."""
        # Create params with empty images list
        params = {
            "prompt": "There should be no images",
            "images": []
        }
        
        # Process for different providers
        for provider in ["openai", "anthropic", "ollama", "huggingface"]:
            processed = MediaProcessor.process_inputs(params, provider)
            
            # Parameters should be unchanged
            if provider in ["openai", "anthropic"]:
                assert "messages" in processed
                content = processed["messages"][0]["content"]
                
                # Either a string or a list with one text item
                if isinstance(content, list):
                    assert len(content) == 1
                    assert content[0]["type"] == "text"
                else:
                    assert processed["messages"][0]["content"] == params["prompt"]
            else:
                assert "prompt" in processed
                assert processed["prompt"] == params["prompt"]
                assert "images" not in processed
    
    def test_process_invalid_image_source(self):
        """Test handling of invalid image sources."""
        # Create params with invalid image
        params = {
            "prompt": "This should fail",
            "image": "/nonexistent/path/image.jpg"
        }
        
        # Should raise an exception
        with pytest.raises(Exception) as exc_info:
            MediaProcessor.process_inputs(params, "openai")
        
        assert "Invalid image source" in str(exc_info.value) or "No such file" in str(exc_info.value)
    
    def test_process_null_image(self):
        """Test processing with null image value."""
        # Create params with None image
        params = {
            "prompt": "There should be no image",
            "image": None
        }
        
        # Process for OpenAI
        processed = MediaProcessor.process_inputs(params, "openai")
        
        # Image parameter should be removed
        assert "image" not in processed
        
        # Should have normal text prompt
        assert "messages" in processed
        if isinstance(processed["messages"][0]["content"], list):
            assert len(processed["messages"][0]["content"]) == 1
            assert processed["messages"][0]["content"][0]["type"] == "text"
        else:
            assert processed["messages"][0]["content"] == params["prompt"]
    
    def test_process_actual_remote_image(self, real_remote_image_url):
        """Test processing an actual remote image URL."""
        try:
            # Create test params with a remote URL
            params = {
                "prompt": "Describe this remote image",
                "image": real_remote_image_url
            }
            
            # Test with OpenAI
            processed_openai = MediaProcessor.process_inputs(params, "openai")
            assert "messages" in processed_openai
            content = processed_openai["messages"][0]["content"]
            assert isinstance(content, list)
            assert len(content) == 2  # Text + image
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            # For remote URLs, OpenAI might use the URL directly or convert to base64
            url = content[1]["image_url"]["url"]
            assert url == real_remote_image_url or url.startswith("data:image/")
            
            # Test with Anthropic
            processed_anthropic = MediaProcessor.process_inputs(params, "anthropic")
            assert "messages" in processed_anthropic
            content = processed_anthropic["messages"][0]["content"]
            assert isinstance(content, list)
            assert len(content) == 2  # Text + image
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image"
            # Anthropic supports both URL and base64 sources
            source = content[1]["source"]
            assert (source["type"] == "url" and source["url"] == real_remote_image_url) or source["type"] == "base64"
            
            # Test with Ollama
            processed_ollama = MediaProcessor.process_inputs(params, "ollama")
            assert "image" in processed_ollama
            assert isinstance(processed_ollama["image"], str)
            assert len(processed_ollama["image"]) > 0  # Should be base64 data
            
            # Test with HuggingFace
            processed_huggingface = MediaProcessor.process_inputs(params, "huggingface")
            assert "image" in processed_huggingface
            # HuggingFace can use either the URL or a downloaded copy
            assert processed_huggingface["image"] == real_remote_image_url or os.path.exists(processed_huggingface["image"])
        
        except (ImageProcessingError, requests.RequestException) as e:
            pytest.skip(f"Error processing remote image: {e}") 