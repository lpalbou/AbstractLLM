#!/usr/bin/env python3
"""
Test script for the MLX provider in AbstractLLM.

This script tests both text generation and vision capabilities.
"""

import os
import argparse
import sys
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for key dependencies
def check_dependencies():
    """Check for necessary dependencies."""
    try:
        import mlx
        logger.info(f"Found MLX: {mlx.__version__ if hasattr(mlx, '__version__') else 'unknown version'}")
    except ImportError:
        logger.error("MLX not found. Please install with: pip install mlx")
        return False
    
    try:
        import mlx_lm
        logger.info(f"Found MLX-LM: {mlx_lm.__version__ if hasattr(mlx_lm, '__version__') else 'unknown version'}")
    except ImportError:
        logger.error("MLX-LM not found. Please install with: pip install mlx-lm")
        return False
    
    try:
        import mlx_vlm
        logger.info(f"Found MLX-VLM: {mlx_vlm.__version__ if hasattr(mlx_vlm, '__version__') else 'unknown version'}")
    except ImportError:
        logger.warning("MLX-VLM not found. Vision capabilities will be disabled.")
        return True
    
    # Check for PyTorch (needed for some vision processing)
    try:
        import torch
        logger.info(f"Found PyTorch: {torch.__version__}")
    except ImportError:
        logger.warning("PyTorch not found but may be needed for vision models.")
        logger.warning("Install with: pip install torch")
    
    return True

def find_compatible_vision_model():
    """Find a compatible vision model from a list of known working models."""
    # List of models known to work well with MLX-VLM
    COMPATIBLE_MODELS = [
        "mlx-community/llava-v1.5-7b-mlx",
        "mlx-community/llava-v1.5-13b-mlx",
        "mlx-community/llava-1.5-7b-4bit",
        "mlx-community/bakllava-1-4bit",
        "mlx-community/phi-3-vision-128k-instruct-4bit"
    ]
    
    # Try to import huggingface_hub to check for locally cached models
    try:
        from huggingface_hub import scan_cache_dir
        
        # Get locally cached models
        cache_info = scan_cache_dir()
        cached_repos = [repo.repo_id for repo in cache_info.repos]
        
        # Find the first compatible model that's already cached
        for model in COMPATIBLE_MODELS:
            if any(model in repo for repo in cached_repos):
                logger.info(f"Found cached compatible model: {model}")
                return model
                
        # If no cached compatible model, return the first in the list
        logger.info(f"No cached compatible model found, will use: {COMPATIBLE_MODELS[0]}")
        return COMPATIBLE_MODELS[0]
    except Exception:
        # If huggingface_hub is not installed or any error occurs, use the first model
        return COMPATIBLE_MODELS[0]

from abstractllm import create_llm
from abstractllm.enums import ModelParameter

def test_text_generation(model_name=None, prompt=None):
    """Test text generation with MLX provider."""
    print(f"Testing text generation with model: {model_name}")
    print(f"Prompt: '{prompt}'")
    
    # Create LLM instance with MLX provider
    llm = create_llm("mlx", **{
        ModelParameter.MODEL: model_name,
        ModelParameter.MAX_TOKENS: 100,
        ModelParameter.TEMPERATURE: 0.1
    })
    
    # Generate text
    print("\nGenerating response...")
    response = llm.generate(prompt)
    
    # Print response
    print(f"\nResponse: {response.content}")
    print(f"Usage: {response.usage}")
    print("\nText generation test completed successfully!")
    
    return response

def test_vision_generation(model_name=None, image_path=None, prompt=None):
    """Test vision capabilities with MLX provider."""
    print(f"Testing vision generation with model: {model_name}")
    print(f"Image: {image_path}")
    print(f"Prompt: '{prompt}'")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    try:
        # Check for PyTorch if we're using a vision model
        try:
            import torch
            print(f"PyTorch is available (version {torch.__version__})")
        except ImportError:
            print("Warning: PyTorch is not installed but may be needed for some vision models.")
            print("Install with: pip install torch")
    
        # Create LLM instance with MLX provider
        print("Creating LLM instance...")
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.MAX_TOKENS: 100,
            ModelParameter.TEMPERATURE: 0.1
        })
        
        # Generate text from image
        print("\nGenerating response...")
        response = llm.generate(prompt, files=[image_path])
        
        # Print response
        print(f"\nResponse: {response.content}")
        print(f"Usage: {response.usage}")
        print("\nVision generation test completed successfully!")
        
        return response
        
    except Exception as e:
        print(f"\nError during vision generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide more helpful error messages
        error_msg = str(e).lower()
        if "patch_size" in error_msg or "nonetype" in error_msg:
            print("\nThis appears to be an issue with the patch_size setting in the vision model.")
            print("Possible solutions:")
            print("1. Install PyTorch: pip install torch")
            print("2. Try a different vision model that's known to work with MLX:")
            print("   - mlx-community/llava-v1.5-7b-mlx")
            print("   - mlx-community/bakllava-1-4bit")
            print("   - mlx-community/phi-3-vision-128k-instruct-4bit")
        
        return None

def main():
    """Main function to run the tests."""
    # Check dependencies first
    if not check_dependencies():
        logger.error("Missing critical dependencies. Please install them and try again.")
        return 1
    
    # Find compatible vision model
    compatible_vision_model = find_compatible_vision_model()
        
    parser = argparse.ArgumentParser(description="Test MLX provider in AbstractLLM")
    parser.add_argument("--mode", choices=["text", "vision", "both"], default="both",
                      help="Test mode: text, vision, or both")
    parser.add_argument("--text-model", default="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
                      help="Text model to use")
    parser.add_argument("--vision-model", default=compatible_vision_model,
                      help="Vision model to use")
    parser.add_argument("--image", default="tests/examples/test_image_336x336.jpg",
                      help="Path to test image")
    parser.add_argument("--text-prompt", default="Explain quantum computing in simple terms.",
                      help="Prompt for text generation")
    parser.add_argument("--vision-prompt", default="What do you see in this image?",
                      help="Prompt for vision generation")
    
    args = parser.parse_args()
    
    # Create directory for test results if it doesn't exist
    os.makedirs("tests/examples", exist_ok=True)
    
    # If test image doesn't exist, create a simple test image
    if not os.path.exists(args.image):
        print(f"Test image not found: {args.image}")
        print("Creating a simple test image...")
        
        try:
            from PIL import Image, ImageDraw
            
            img_size = (336, 336)
            img = Image.new("RGB", img_size, (240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Draw some shapes
            draw.rectangle([(20, 20), (img_size[0]-20, img_size[1]-20)], outline=(200, 0, 0), width=5)
            draw.ellipse([(50, 50), (img_size[0]-50, img_size[1]-50)], outline=(0, 0, 200), width=5)
            draw.line([(20, 20), (img_size[0]-20, img_size[1]-20)], fill=(0, 150, 0), width=5)
            draw.line([(20, img_size[1]-20), (img_size[0]-20, 20)], fill=(0, 150, 0), width=5)
            
            # Add text
            text = f"Test Image {img_size[0]}x{img_size[1]}"
            draw.text((img_size[0]//2 - 80, img_size[1]//2 - 10), text, fill=(0, 0, 0))
            
            # Save the image
            os.makedirs(os.path.dirname(args.image), exist_ok=True)
            img.save(args.image)
            print(f"Created test image: {args.image}")
        except Exception as e:
            print(f"Error creating test image: {e}")
            return
    
    # Run selected tests
    if args.mode in ["text", "both"]:
        print("\n" + "="*50)
        print("TESTING TEXT GENERATION")
        print("="*50)
        test_text_generation(args.text_model, args.text_prompt)
    
    if args.mode in ["vision", "both"]:
        print("\n" + "="*50)
        print("TESTING VISION GENERATION")
        print("="*50)
        test_vision_generation(args.vision_model, args.image, args.vision_prompt)

if __name__ == "__main__":
    main() 