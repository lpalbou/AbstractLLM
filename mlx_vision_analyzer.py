#!/usr/bin/env python3
"""
MLX Vision Image Analyzer

This script uses the MLXProvider directly to analyze images with vision-capable MLX models.
It loads the model, processes the image, and generates a text analysis.

Usage:
    python mlx_vision_analyzer.py [--model MODEL_NAME] [--image IMAGE_PATH] [--prompt PROMPT]
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Union, List

from PIL import Image
import mlx.core as mx
import numpy as np

from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.media.image import ImageInput
from abstractllm.enums import ModelParameter
from abstractllm.types import GenerateResponse

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default MLX vision model
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
DEFAULT_PROMPT = "Analyze this image in detail. Describe what you see, including objects, people, scenery, colors, and any notable elements."

def analyze_image(image_path: Union[str, Path], model_name: str, prompt: str = DEFAULT_PROMPT, debug: bool = False) -> str:
    """
    Analyze an image using a vision-capable MLX model.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the MLX vision model to use
        prompt: The prompt to use for image analysis
        debug: Enable debug mode for more verbose output
        
    Returns:
        Analysis result as a string
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Enable debug logging if requested
    if debug:
        logging.getLogger("abstractllm").setLevel(logging.DEBUG)
    
    print(f"Loading model: {model_name}")
    print("This may take a while for the first run...")
    
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
    
    print(f"Image processed successfully: shape={processed_image.shape}, dtype={processed_image.dtype}")
    
    # Format prompt with image token
    formatted_prompt = provider._format_prompt(prompt, 1)
    print(f"Formatted prompt: {formatted_prompt}")
    
    # Load the model if not already loaded
    if not provider._is_loaded:
        print("Loading the model...")
        provider.load_model()
    
    # Generate analysis
    print("Generating analysis...")
    try:
        # Use the provider's _generate_vision method directly
        if hasattr(provider, '_generate_vision'):
            response = provider._generate_vision(formatted_prompt, [processed_image])
            return response
        else:
            # Fall back to the standard generate method
            response = provider.generate(
                prompt=prompt,
                files=[image_path]
            )
            return response.content
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        return f"Error: {str(e)}"

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze images using MLX vision models")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"MLX vision model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--image", type=str,
                        help="Path to the image file to analyze")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Prompt to use for image analysis")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for more verbose output")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MLX Vision Image Analyzer")
    print("=" * 80)
    print(f"Using model: {args.model}")
    print("=" * 80)
    
    # If image path is provided via command line, analyze it directly
    if args.image:
        try:
            result = analyze_image(args.image, args.model, args.prompt, args.debug)
            print("\n" + "=" * 40 + " ANALYSIS " + "=" * 40)
            print(result)
            print("=" * 89)
        except Exception as e:
            print(f"Error: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
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
        
        # Get custom prompt (optional)
        custom_prompt = input("Enter custom prompt (or press Enter for default): ").strip()
        if not custom_prompt:
            custom_prompt = DEFAULT_PROMPT
        
        try:
            # Analyze the image
            start_time = time.time()
            result = analyze_image(image_path, args.model, custom_prompt, args.debug)
            elapsed_time = time.time() - start_time
            
            # Print the analysis
            print("\n" + "=" * 40 + " ANALYSIS " + "=" * 40)
            print(result)
            print("=" * 89)
            print(f"Analysis completed in {elapsed_time:.2f} seconds")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
        sys.exit(0) 