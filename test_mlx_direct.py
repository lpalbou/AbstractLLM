#!/usr/bin/env python3
"""
Direct test script for MLX provider in AbstractLLM.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import AbstractLLM components
from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.enums import ModelParameter
from abstractllm.media.image import ImageInput

def test_mlx_provider(model_name, image_path, prompt):
    """
    Test the MLX provider directly.
    
    Args:
        model_name: The name of the MLX model to use
        image_path: Path to the image file to use
        prompt: The prompt to send to the model
    """
    print(f"Testing MLX provider with model: {model_name}")
    print(f"Image path: {image_path}")
    print(f"Prompt: {prompt}")
    
    try:
        # Create provider
        provider = MLXProvider({
            ModelParameter.MODEL: model_name,
            ModelParameter.MAX_TOKENS: 512,
            ModelParameter.TEMPERATURE: 0.2
        })
        
        print("Provider created successfully")
        print(f"Is vision model: {provider._is_vision_model}")
        print(f"Model type: {provider._model_type}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
            
        # Process image
        image_input = ImageInput(image_path)
        
        # Generate response
        print(f"Generating response...")
        response = provider.generate(prompt, files=[image_path])
        
        print("\nResponse:")
        print(response.content)
        print(f"\nUsage: {response.usage}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python test_mlx_direct.py MODEL_NAME IMAGE_PATH [PROMPT]")
        print("\nExample:")
        print("python test_mlx_direct.py mlx-community/llava-1.5-7b-4bit tests/examples/mountain_path_resized_standardized.jpg 'What is in this image?'")
        sys.exit(1)
        
    model_name = sys.argv[1]
    image_path = sys.argv[2]
    
    # Use default prompt if not provided
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Describe what you see in this image."
    
    test_mlx_provider(model_name, image_path, prompt)

if __name__ == "__main__":
    main() 