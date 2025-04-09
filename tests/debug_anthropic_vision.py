#!/usr/bin/env python
"""
Debug script for Anthropic vision capabilities.

This script tests image processing and vision capabilities with Anthropic
directly to diagnose issues with the tests.
"""

import os
import sys
import base64
import logging
from pathlib import Path

# Add parent directory to path for importing
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from abstractllm import create_llm, ModelParameter
from abstractllm.utils.logging import setup_logging
from abstractllm.utils.image import format_image_for_provider, preprocess_image_inputs

# Set up logging to debug level
setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_image_as_base64(image_path):
    """Get image as base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def test_with_anthropic():
    """Test Anthropic vision capabilities."""
    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        return False

    # Set up model
    model = "claude-3-5-sonnet-20240620"
    image_path = os.path.join(os.path.dirname(__file__), "resources", "test_image_1.jpg")
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return False
    
    print(f"Testing image: {image_path}")
    print(f"File size: {os.path.getsize(image_path)} bytes")
    
    # Create provider
    provider = create_llm("anthropic", **{
        ModelParameter.API_KEY: api_key,
        ModelParameter.MODEL: model,
        ModelParameter.MAX_TOKENS: 300,
        ModelParameter.TEMPERATURE: 0.1
    })
    
    # First, try using the generate method directly with image path
    print("\n--- Test 1: Using generate with image path ---")
    try:
        response1 = provider.generate(
            "Create a list of descriptive keywords for the attached image", 
            image=image_path
        )
        print(f"Response 1: {response1[:200]}...")
    except Exception as e:
        print(f"Error in Test 1: {e}")
    
    # Second, try with explicit base64 encoding
    print("\n--- Test 2: Using generate with base64 image ---")
    try:
        # Get base64 encoded image
        image_base64 = get_image_as_base64(image_path)
        
        # Format for Anthropic
        formatted_image = {
            "type": "image", 
            "source": {
                "type": "base64", 
                "media_type": "image/jpeg", 
                "data": image_base64
            }
        }
        
        # Create message content
        message_content = [
            {"type": "text", "text": "Create a list of descriptive keywords for the attached image"},
            formatted_image
        ]
        
        # Override parameters with explicit message format
        response2 = provider.generate(
            "", 
            **{
                "messages": [
                    {"role": "user", "content": message_content}
                ]
            }
        )
        print(f"Response 2: {response2[:200]}...")
    except Exception as e:
        print(f"Error in Test 2: {e}")
    
    # Third, try direct API call
    print("\n--- Test 3: Using Anthropic API directly ---")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        # Get base64 encoded image
        image_base64 = get_image_as_base64(image_path)
        
        # Create the message
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Create a list of descriptive keywords for the attached image"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                }
            ]
        }
        
        # Make the API call
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[message]
        )
        
        response3 = response.content[0].text
        print(f"Response 3: {response3[:200]}...")
    except Exception as e:
        print(f"Error in Test 3: {e}")
    
    return True

if __name__ == "__main__":
    success = test_with_anthropic()
    if not success:
        sys.exit(1) 