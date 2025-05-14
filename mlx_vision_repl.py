#!/usr/bin/env python3
"""
Simple REPL for image analysis using AbstractLLM with MLX vision models.

Usage:
    python mlx_vision_repl.py [--model MODEL_NAME]

Example:
    python mlx_vision_repl.py --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Union

from abstractllm import create_llm
from abstractllm.enums import ModelParameter
from abstractllm.exceptions import (
    UnsupportedFeatureError,
    ImageProcessingError,
    FileProcessingError,
    MemoryExceededError
)

# Default MLX vision model
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_image(image_path: Union[str, Path], model_name: str, debug: bool = False) -> str:
    """
    Analyze an image using a vision-capable MLX model.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the MLX vision model to use
        debug: Enable debug mode for more verbose output
        
    Returns:
        Analysis result as a string
    """
    # Ensure the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"Loading model: {model_name}")
    print("This may take a while for the first run...")
    
    # Enable debug logging if requested
    if debug:
        logging.getLogger("abstractllm").setLevel(logging.DEBUG)
    
    # Create the LLM with the specified model
    llm = create_llm("mlx", **{
        ModelParameter.MODEL: model_name,
        ModelParameter.TEMPERATURE: 0.2,  # Lower temperature for more factual analysis
        ModelParameter.MAX_TOKENS: 1024   # Reasonable response length
    })
    
    # Access the provider directly to check and set vision capabilities
    if hasattr(llm, "_provider"):
        provider = llm._provider
        if hasattr(provider, "_is_vision_model"):
            # Force-enable vision capabilities for known vision models
            if "vl" in model_name.lower() or "vision" in model_name.lower() or "llava" in model_name.lower():
                print("Explicitly enabling vision capabilities for this model")
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
    
    # Check if the model supports vision
    capabilities = llm.get_capabilities()
    print(f"Model capabilities: {capabilities}")
    
    if not capabilities.get("vision", False):
        raise UnsupportedFeatureError(
            "vision",
            f"The model {model_name} does not support vision capabilities",
            provider="mlx"
        )
    
    print(f"Analyzing image: {image_path}")
    
    # Generate analysis
    response = llm.generate(
        prompt="Analyze this image in detail. Describe what you see, including objects, people, scenery, colors, and any notable elements.",
        files=[image_path]
    )
    
    return response.content

def main():
    """Main REPL function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze images using MLX vision models")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"MLX vision model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for more verbose output")
    args = parser.parse_args()
    
    print("=" * 80)
    print("MLX Vision Image Analysis REPL")
    print("=" * 80)
    print(f"Using model: {args.model}")
    print("Enter 'exit', 'quit', or 'q' to exit")
    print("=" * 80)
    
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
        
        try:
            # Analyze the image
            result = analyze_image(image_path, args.model, args.debug)
            
            # Print the analysis
            print("\n" + "=" * 40 + " ANALYSIS " + "=" * 40)
            print(result)
            print("=" * 89)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except UnsupportedFeatureError as e:
            print(f"Error: {e}")
        except ImageProcessingError as e:
            print(f"Error processing image: {e}")
        except MemoryExceededError as e:
            print(f"Error: Memory limit exceeded. Try a smaller image or a different model.")
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