#!/usr/bin/env python3
"""
Direct MLX Vision Image Processing Script

This script bypasses the AbstractLLM interface and directly uses the MLXProvider
class to process images with vision-capable MLX models.

Usage:
    python mlx_vision_direct.py [--model MODEL_NAME] [--image IMAGE_PATH]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image
import mlx.core as mx
import numpy as np

from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.media.image import ImageInput
from abstractllm.enums import ModelParameter

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default MLX vision model
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

def process_image(image_path: Union[str, Path], model_name: str, debug: bool = False) -> None:
    """
    Process an image using a vision-capable MLX model.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the MLX vision model to use
        debug: Enable debug mode for more verbose output
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    # Enable debug logging if requested
    if debug:
        logging.getLogger("abstractllm").setLevel(logging.DEBUG)
    
    try:
        # Create MLXProvider instance directly
        provider = MLXProvider({
            ModelParameter.MODEL: model_name,
            ModelParameter.TEMPERATURE: 0.2,
            ModelParameter.MAX_TOKENS: 1024
        })
        
        # Force vision capabilities for known vision models
        if "vl" in model_name.lower() or "vision" in model_name.lower() or "llava" in model_name.lower():
            print("Explicitly enabling vision capabilities")
            provider._is_vision_model = True
            
            # Determine model type based on name
            if "qwen" in model_name.lower():
                provider._model_type = "qwen-vl"
            elif "llava" in model_name.lower():
                provider._model_type = "llava"
            elif "gemma" in model_name.lower():
                provider._model_type = "gemma"
            else:
                provider._model_type = "default"
            
            print(f"Set model type to: {provider._model_type}")
        
        # Create ImageInput
        print(f"Loading image: {image_path}")
        image_input = ImageInput(image_path)
        
        # Process the image
        print("Processing image...")
        processed_image = provider._process_image(image_input)
        
        # Print information about the processed image
        print("\nProcessed Image Information:")
        print(f"Type: {type(processed_image)}")
        print(f"Shape: {processed_image.shape}")
        print(f"Dtype: {processed_image.dtype}")
        
        # Get model configuration
        config = provider._get_model_config()
        print("\nModel Configuration:")
        print(f"Image Size: {config['image_size']}")
        print(f"Prompt Format: {config['prompt_format']}")
        
        print("\nImage processing successful!")
        
        # Attempt to format a prompt with the image
        prompt = "Describe what you see in this image."
        formatted_prompt = provider._format_prompt(prompt, 1)
        print(f"\nFormatted Prompt: {formatted_prompt}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process images using MLX vision models directly")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"MLX vision model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--image", type=str,
                        help="Path to the image file to process")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for more verbose output")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MLX Vision Direct Image Processing")
    print("=" * 80)
    print(f"Using model: {args.model}")
    print("=" * 80)
    
    # If image path is provided via command line, process it directly
    if args.image:
        process_image(args.image, args.model, args.debug)
        return
    
    # Otherwise, enter REPL mode
    while True:
        # Get image path from user
        image_path = input("\nEnter image path (or 'exit' to quit): ").strip()
        
        # Check for exit command
        if image_path.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        
        # Skip empty input
        if not image_path:
            print("Please enter a valid image path")
            continue
        
        # Process the image
        process_image(image_path, args.model, args.debug)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
        sys.exit(0) 