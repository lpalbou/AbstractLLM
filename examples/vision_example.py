#!/usr/bin/env python3
"""
AbstractLLM Vision Example

This example demonstrates how to use vision capabilities in AbstractLLM
with MLX vision models on Apple Silicon.
"""

import os
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AbstractLLM Vision Example")
    parser.add_argument("--image", type=str, default="test_images/sample.jpg",
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Prompt to use for image description")
    parser.add_argument("--model", type=str, default="mlx-community/paligemma-3b-mix-448-8bit",
                        help="Vision model to use")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation")
    parser.add_argument("--stream", action="store_true",
                        help="Stream the response")
    return parser.parse_args()

def main():
    """Run the vision example."""
    args = parse_args()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    try:
        # Import required packages
        try:
            from abstractllm import create_llm
            import PIL.Image
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.info("Installing required packages...")
            os.system("pip install abstractllm pillow")
            from abstractllm import create_llm
            import PIL.Image
        
        # Load the image
        image = PIL.Image.open(image_path)
        
        # Create configuration for the model
        config = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        }
        
        # Create the LLM
        logger.info(f"Creating LLM with model: {args.model}")
        llm = create_llm("mlx", **config)
        
        # Generate response
        logger.info("Generating vision response...")
        
        if args.stream:
            # Stream the response
            print("\nStreaming response:")
            print("-" * 40)
            full_text = ""
            for response in llm.generate(args.prompt, images=[image], stream=True):
                # Get the new content
                new_content = response.content[len(full_text):]
                print(new_content, end="", flush=True)
                full_text = response.content
            print("\n" + "-" * 40)
        else:
            # Generate complete response
            response = llm.generate(args.prompt, images=[image])
            
            # Print result
            print("\nResponse:")
            print("-" * 40)
            print(response.content)
            print("-" * 40)
            
            # Print usage statistics if available
            if response.usage:
                print(f"Usage stats:")
                for key, value in response.usage.items():
                    print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 