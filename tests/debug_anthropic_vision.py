#!/usr/bin/env python
"""
Debug script for Anthropic vision capabilities.

This script tests image processing and vision capabilities with Anthropic
directly using real API calls without any mocking.
"""

import os
import sys
import base64
import logging
import argparse
from pathlib import Path
import requests
from typing import Dict, Any, List, Optional

# Add parent directory to path for importing
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from abstractllm import create_llm, ModelParameter, ModelCapability, ImageInput
from abstractllm.utils.logging import setup_logging
from abstractllm.exceptions import ImageProcessingError, UnsupportedFeatureError

# Set up logging to debug level
setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define test resources directory
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Define test image paths
TEST_IMAGES = {
    "jpeg": os.path.join(RESOURCES_DIR, "test_image_1.jpg"),
    "png": os.path.join(RESOURCES_DIR, "test_image_2.png"),
    "remote_jpeg": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/test_image_1.jpg",
    "remote_png": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/test_image_2.png"
}

def get_image_as_base64(image_path: str) -> str:
    """Get image as base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded image data
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def download_test_image(url: str, output_path: str) -> None:
    """Download a test image if it doesn't exist.
    
    Args:
        url: URL of the image to download
        output_path: Path where the image should be saved
    """
    if not os.path.exists(output_path):
        logger.info(f"Downloading test image from {url} to {output_path}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info(f"Image downloaded successfully")
        else:
            logger.error(f"Failed to download image: {response.status_code}")
            raise RuntimeError(f"Failed to download image: {response.status_code}")

def ensure_test_images_exist() -> bool:
    """Ensure test images exist, downloading them if necessary.
    
    Returns:
        True if all images are available, False otherwise
    """
    try:
        # Check/download JPEG image
        if not os.path.exists(TEST_IMAGES["jpeg"]):
            download_test_image(TEST_IMAGES["remote_jpeg"], TEST_IMAGES["jpeg"])
            
        # Check/download PNG image
        if not os.path.exists(TEST_IMAGES["png"]):
            download_test_image(TEST_IMAGES["remote_png"], TEST_IMAGES["png"])
            
        # Verify files exist and have content
        for img_path in [TEST_IMAGES["jpeg"], TEST_IMAGES["png"]]:
            if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                logger.error(f"Image file missing or empty: {img_path}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error ensuring test images: {e}")
        return False

def test_image_with_provider(image_path: str, provider: Any, prompt: str = "Describe this image in detail") -> str:
    """Test image processing with a provider.
    
    Args:
        image_path: Path or URL to the image
        provider: LLM provider instance
        prompt: Prompt to use with the image
        
    Returns:
        The provider's response
    """
    try:
        # Check file exists if it's a local path
        if os.path.isfile(image_path):
            logger.info(f"Using local image: {image_path} ({os.path.getsize(image_path)} bytes)")
        else:
            logger.info(f"Using remote image: {image_path}")
            
        # Generate response with the image
        logger.info(f"Generating response with prompt: {prompt}")
        response = provider.generate(prompt, image=image_path)
        
        return response
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def test_vision_with_direct_format(provider: Any, image_path: str) -> str:
    """Test vision capabilities using directly formatted messages.
    
    Args:
        provider: LLM provider instance
        image_path: Path to the image
        
    Returns:
        The provider's response
    """
    # Create an ImageInput object
    image_input = ImageInput(image_path)
    
    # Format for Anthropic
    formatted_image = image_input.to_provider_format("anthropic")
    
    # Create message content
    message_content = [
        {"type": "text", "text": "Create a list of descriptive keywords for this image"},
        formatted_image
    ]
    
    # Override parameters with explicit message format
    response = provider.generate(
        "", 
        **{
            "messages": [
                {"role": "user", "content": message_content}
            ]
        }
    )
    
    return response

def test_with_anthropic():
    """Test Anthropic vision capabilities with real API calls.
    
    Returns:
        True if all tests pass, False otherwise
    """
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        return False
        
    # Ensure test images are available
    if not ensure_test_images_exist():
        logger.error("Failed to ensure test images exist")
        return False
        
    # Set up model
    model = "claude-3-5-sonnet-20240620"
    
    # Create provider
    logger.info(f"Creating Anthropic provider with model: {model}")
    provider = create_llm("anthropic", **{
        ModelParameter.API_KEY: api_key,
        ModelParameter.MODEL: model,
        ModelParameter.MAX_TOKENS: 300,
        ModelParameter.TEMPERATURE: 0.1
    })
    
    # Check if the model supports vision
    capabilities = provider.get_capabilities()
    if not capabilities.get(ModelCapability.VISION):
        logger.error(f"Model {model} does not support vision")
        return False
        
    # Test 1: Using generate with local JPEG image path
    try:
        logger.info("\n--- Test 1: Using generate with local JPEG image path ---")
        response1 = test_image_with_provider(TEST_IMAGES["jpeg"], provider)
        logger.info(f"Response 1: {response1[:200]}...")
        print(f"\nResponse 1: {response1[:200]}...")
    except Exception as e:
        logger.error(f"Error in Test 1: {e}")
        
    # Test 2: Using generate with local PNG image path
    try:
        logger.info("\n--- Test 2: Using generate with local PNG image path ---")
        response2 = test_image_with_provider(TEST_IMAGES["png"], provider)
        logger.info(f"Response 2: {response2[:200]}...")
        print(f"\nResponse 2: {response2[:200]}...")
    except Exception as e:
        logger.error(f"Error in Test 2: {e}")
        
    # Test 3: Using generate with remote image URL (JPEG)
    try:
        logger.info("\n--- Test 3: Using generate with remote JPEG image URL ---")
        response3 = test_image_with_provider(TEST_IMAGES["remote_jpeg"], provider)
        logger.info(f"Response 3: {response3[:200]}...")
        print(f"\nResponse 3: {response3[:200]}...")
    except Exception as e:
        logger.error(f"Error in Test 3: {e}")
        
    # Test 4: Using generate with remote image URL (PNG)
    try:
        logger.info("\n--- Test 4: Using generate with remote PNG image URL ---")
        response4 = test_image_with_provider(TEST_IMAGES["remote_png"], provider)
        logger.info(f"Response 4: {response4[:200]}...")
        print(f"\nResponse 4: {response4[:200]}...")
    except Exception as e:
        logger.error(f"Error in Test 4: {e}")
        
    # Test 5: Using direct formatting
    try:
        logger.info("\n--- Test 5: Using direct formatting with ImageInput ---")
        response5 = test_vision_with_direct_format(provider, TEST_IMAGES["jpeg"])
        logger.info(f"Response 5: {response5[:200]}...")
        print(f"\nResponse 5: {response5[:200]}...")
    except Exception as e:
        logger.error(f"Error in Test 5: {e}")
        
    # Test 6: Error handling - invalid image path
    try:
        logger.info("\n--- Test 6: Error handling - invalid image path ---")
        invalid_path = "nonexistent_image.jpg"
        test_image_with_provider(invalid_path, provider)
        logger.error("Test 6 should have failed but didn't")
        print("\nTest 6 should have failed but didn't")
    except Exception as e:
        logger.info(f"Test 6 correctly failed with error: {e}")
        print(f"\nTest 6 correctly failed with error: {e}")
        
    # Test 7: Error handling - invalid image URL
    try:
        logger.info("\n--- Test 7: Error handling - invalid image URL ---")
        invalid_url = "https://example.com/nonexistent_image.jpg"
        test_image_with_provider(invalid_url, provider)
        logger.error("Test 7 should have failed but didn't")
        print("\nTest 7 should have failed but didn't")
    except Exception as e:
        logger.info(f"Test 7 correctly failed with error: {e}")
        print(f"\nTest 7 correctly failed with error: {e}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Anthropic vision capabilities")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    success = test_with_anthropic()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 