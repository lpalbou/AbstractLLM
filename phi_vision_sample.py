#!/usr/bin/env python3
"""
Minimal example to test the Phi-3.5 vision model with MLX.
Based on the Josef Albers repository examples.
"""

import time
import argparse
import logging
from pathlib import Path
from PIL import Image
import mlx.core as mx
import mlx_vlm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test vision model capabilities")
    parser.add_argument("--model", type=str, default="paligemma",
                        help="Model to use: 'phi', 'qwen', or 'paligemma'")
    parser.add_argument("--image", type=str, default="test_images/sample.jpg",
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Prompt to use for image description")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    return parser.parse_args()

def test_vision_model(model_name, image_path, prompt, max_tokens):
    """Test vision model capabilities."""
    
    # Model paths dictionary
    MODEL_PATHS = {
        "phi": "mlx-community/Phi-3.5-vision-instruct-4bit",
        "qwen": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        "paligemma": "mlx-community/paligemma-3b-mix-448-8bit"
    }
    
    # Make sure we have a valid model
    if model_name not in MODEL_PATHS:
        logger.error(f"Unknown model: {model_name}. Choose from: {', '.join(MODEL_PATHS.keys())}")
        return False
        
    # Get the actual model path
    model_path = MODEL_PATHS[model_name]
    logger.info(f"Using model: {model_path}")
    
    # Make sure image exists
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return False
    
    # Set trust_remote_code based on model
    trust_remote_code = "phi" in model_name.lower()
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load the model
        logger.info(f"Loading model: {model_path}")
        model, processor = mlx_vlm.load(model_path, trust_remote_code=trust_remote_code)
        
        # Format prompt based on model type
        if model_name == "paligemma":
            formatted_prompt = f"<image>{prompt}"
        elif model_name == "phi":
            formatted_prompt = f"<image_0> {prompt}"
        elif model_name == "qwen":
            # For Qwen, we need to use the chat template from config
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
            config = load_config(model_path)
            formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
        else:
            formatted_prompt = prompt
            
        logger.info(f"Formatted prompt: {formatted_prompt}")
        
        # Generate output
        logger.info("Generating response...")
        output = mlx_vlm.generate(
            model, 
            processor, 
            formatted_prompt, 
            [str(image_path)], 
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        # Calculate time
        elapsed_time = time.time() - start_time
        
        # Print results
        logger.info("="*50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Time: {elapsed_time:.2f} seconds")
        logger.info("="*50)
        logger.info("RESULT:")
        logger.info(output)
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    args = parse_args()
    success = test_vision_model(args.model, args.image, args.prompt, args.max_tokens)
    if success:
        logger.info(f"Test with {args.model} model completed successfully")
    else:
        logger.error(f"Test with {args.model} model failed")
        exit(1)

if __name__ == "__main__":
    main() 