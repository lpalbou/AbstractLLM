#!/usr/bin/env python3
"""
Phi Vision Test Script

This script tests the vision capabilities of the Phi-3.5 vision model
specifically handling the image tags correctly.
"""

import os
import argparse
import time
import logging
from pathlib import Path
import torch  # We need PyTorch for Phi-3.5 vision

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Phi-3.5 vision capabilities")
    parser.add_argument("--image", type=str, default="test_images/sample.jpg",
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Prompt to use for image description")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation")
    return parser.parse_args()

def test_phi_vision(image_path, prompt, max_tokens, temperature):
    """Test Phi-3.5 vision capabilities."""
    try:
        # Make sure image exists
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False
        
        # Check for required packages and install if missing
        try:
            import mlx.core as mx
            import mlx_vlm
            from PIL import Image
            from transformers import AutoProcessor
            logger.info(f"MLX VLM version: {mlx_vlm.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.info("Installing required packages...")
            os.system("pip install mlx mlx-vlm==0.1.26 pillow torch transformers")
            import mlx.core as mx
            import mlx_vlm
            from PIL import Image
            from transformers import AutoProcessor
            logger.info("Required packages installed")
        
        # Load the Phi-3.5 vision model
        logger.info("Loading Phi-3.5 vision model...")
        model_name = "mlx-community/Phi-3.5-vision-instruct-4bit"
        
        start_time = time.time()
        
        # Open the image
        image = Image.open(image_path)
        
        # Try a different approach using transformers first to format correctly
        try:
            logger.info("Using transformers for preprocessing...")
            # Load the processor directly
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # Format the prompt for the Phi model - important to include <image_0> tag
            formatted_prompt = f"<image_0> {prompt}"
            logger.info(f"Formatted prompt: {formatted_prompt}")
            
            # Process inputs directly with transformers
            inputs = processor(
                text=formatted_prompt,
                images=image, 
                return_tensors="pt"
            )
            
            logger.info("Successfully processed inputs with transformers")
            
            # Now load the model with MLX-VLM
            from mlx_vlm import load, generate
            model, mlx_processor = load(model_name, trust_remote_code=True)
            
            # Generate the response using the processed image and prompt
            logger.info("Generating response...")
            output = generate(
                model, 
                mlx_processor, 
                formatted_prompt, 
                [str(image_path)],
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print the result
            logger.info("="*50)
            logger.info(f"Model: Phi-3.5 vision")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Processing time: {total_time:.2f} seconds")
            logger.info("="*50)
            logger.info("RESULT:")
            logger.info(output)
            logger.info("="*50)
            
            return True
            
        except Exception as e:
            logger.error(f"Error with transformers approach: {e}")
            
            # Try alternative approach using MLX-VLM directly
            logger.info("Trying alternative approach with MLX-VLM...")
            
            # Load the model with MLX-VLM
            from mlx_vlm import load, generate
            model, processor = load(model_name, trust_remote_code=True)
            
            # Format the prompt for Phi model
            system_prompt = "You are a helpful assistant that can see images and answer questions about them."
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image_0>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate response
            logger.info("Generating response...")
            output = generate(
                model, 
                processor, 
                formatted_prompt, 
                [str(image_path)],
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print the result
            logger.info("="*50)
            logger.info(f"Model: Phi-3.5 vision")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Processing time: {total_time:.2f} seconds")
            logger.info("="*50)
            logger.info("RESULT:")
            logger.info(output)
            logger.info("="*50)
            
            return True
    
    except Exception as e:
        logger.error(f"Error during vision test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    logger.info(f"Testing Phi-3.5 vision capabilities with image: {args.image}")
    
    # Run the test
    success = test_phi_vision(
        args.image,
        args.prompt,
        args.max_tokens,
        args.temperature
    )
    
    if success:
        logger.info("Phi-3.5 vision test completed successfully")
    else:
        logger.error("Phi-3.5 vision test failed")
        exit(1)

if __name__ == "__main__":
    main() 