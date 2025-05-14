#!/usr/bin/env python3
"""
Test MLX vision capabilities directly.

This script tests the MLX vision capabilities by loading a small vision model
and generating a response for a test image.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required packages
try:
    from abstractllm import create_llm, ModelParameter
    from abstractllm.providers.mlx_provider import MLXProvider
    import mlx.core as mx
    from PIL import Image
except ImportError as e:
    logger.error(f"Required package not found: {e}")
    logger.error("Please install with: pip install abstractllm mlx pillow")
    sys.exit(1)

def process_image(image_path: Union[str, Path], model_name: str, debug: bool = False) -> None:
    """
    Process an image using an MLX vision model.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the MLX vision model to use
        debug: Whether to enable debug logging
    """
    if debug:
        logging.getLogger("abstractllm").setLevel(logging.DEBUG)
    
    # Check if image exists
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    logger.info(f"Processing image: {image_path}")
    logger.info(f"Using model: {model_name}")
    
    try:
        # Create LLM
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.TEMPERATURE: 0.2,  # Low temperature for more deterministic output
            ModelParameter.MAX_TOKENS: 1000,  # Reasonable output length
        })
        
        # Generate response
        response = llm.generate(
            prompt="What do you see in this image? Describe it in detail.",
            files=[str(image_path)]
        )
        
        # Print response
        print("\n" + "="*50)
        print(f"Image: {image_path.name}")
        print(f"Model: {model_name}")
        print("="*50)
        print(response.content)
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MLX vision capabilities")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="mlx-community/smolvlm-1.7b-4bit", 
                        help="MLX vision model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    process_image(args.image, args.model, args.debug)

if __name__ == "__main__":
    main() 