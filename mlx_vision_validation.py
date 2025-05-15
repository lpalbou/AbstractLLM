#!/usr/bin/env python3
"""
MLX Vision Models Validation Script

This script tests and validates the vision capabilities of various
MLX vision models (Paligemma and Qwen2-VL).

Note: Currently Phi-3.5-vision-instruct-4bit is not compatible with MLX.
"""

import os
import argparse
import time
import logging
import json
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of MLX-compatible vision models to test
VISION_MODELS = {
    "qwen": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "paligemma": "mlx-community/paligemma-3b-mix-448-8bit"
    # Phi model currently not working with MLX due to image tag compatibility issues
    # "phi": "mlx-community/Phi-3.5-vision-instruct-4bit", 
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate MLX vision models")
    parser.add_argument("--model", type=str, choices=list(VISION_MODELS.keys()), default="paligemma",
                        help=f"Model type to test. Available: {', '.join(VISION_MODELS.keys())}")
    parser.add_argument("--image", type=str, default="test_images/sample.jpg",
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Prompt to use for image description")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation")
    parser.add_argument("--test-all", action="store_true",
                        help="Test all available models")
    parser.add_argument("--output-json", type=str, default="vision_models_results.json",
                        help="Path to save test results as JSON")
    return parser.parse_args()

def test_vision_model(model_name, image_path, prompt, max_tokens, temperature):
    """Test the vision capabilities of a specified MLX model."""
    try:
        # Make sure image exists
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False, None
        
        # Get the model path
        model_path = VISION_MODELS.get(model_name)
        if not model_path:
            logger.error(f"Unknown model: {model_name}")
            return False, None
            
        logger.info(f"Testing model: {model_path}")
        
        # Check for required packages and install if missing
        try:
            import mlx.core as mx
            import mlx_vlm
            from PIL import Image
            logger.info(f"MLX VLM version: {mlx_vlm.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.info("Installing required packages...")
            os.system("pip install mlx mlx-vlm==0.1.26 pillow")
            import mlx.core as mx
            import mlx_vlm
            from PIL import Image
            logger.info("Required packages installed")
        
        # Load the model
        start_time = time.time()
        
        # Import needed modules
        from mlx_vlm import load, generate
        
        # Load the model with appropriate settings
        logger.info(f"Loading model: {model_name}")
        model, processor = load(model_path)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Format the prompt according to model type
        if model_name == "paligemma":
            formatted_prompt = f"<image>{prompt}"
        elif model_name == "qwen":
            # For Qwen, we need to use the chat template from config
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
            config = load_config(model_path)
            formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
        else:
            formatted_prompt = prompt
            
        logger.info(f"Formatted prompt: {formatted_prompt}")
        
        # Open and process the image
        logger.info(f"Processing image: {image_path}")
        
        # Generate the response
        logger.info("Generating description...")
        gen_start_time = time.time()
        
        output = generate(
            model, 
            processor, 
            formatted_prompt, 
            [str(image_path)], 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        gen_time = time.time() - gen_start_time
        total_time = time.time() - start_time
        
        logger.info(f"Description generated in {gen_time:.2f} seconds")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Extract the result
        if isinstance(output, tuple) and len(output) == 2:
            text_output, stats = output
        else:
            text_output = output
            stats = {}
            
        # Calculate tokens per second
        if 'generation_tps' in stats:
            tokens_per_second = stats['generation_tps']
        elif 'output_tokens' in stats and gen_time > 0:
            tokens_per_second = stats.get('output_tokens', 0) / gen_time
        else:
            tokens_per_second = 0
            
        # Format results
        result = {
            "model": model_name,
            "model_path": model_path,
            "prompt": prompt,
            "output": text_output,
            "load_time": load_time,
            "generation_time": gen_time,
            "total_time": total_time,
            "tokens_per_second": tokens_per_second,
            "stats": stats
        }
        
        # Print the result
        logger.info("="*50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Prompt: {prompt}")
        logger.info("="*50)
        logger.info("RESULT:")
        logger.info(text_output)
        logger.info("="*50)
        logger.info(f"Performance: {tokens_per_second:.2f} tokens/sec")
        logger.info("="*50)
        
        return True, result
    
    except Exception as e:
        logger.error(f"Error during vision test: {e}")
        import traceback
        traceback.print_exc()
        return False, {"model": model_name, "error": str(e)}

def main():
    """Main function."""
    args = parse_args()
    
    # If test-all flag is set, test all available models
    models_to_test = list(VISION_MODELS.keys()) if args.test_all else [args.model]
    
    results = []
    success_count = 0
    
    for model_name in models_to_test:
        logger.info(f"Testing MLX vision capabilities with model: {model_name}")
        
        # Run the test
        success, result = test_vision_model(
            model_name,
            args.image,
            args.prompt,
            args.max_tokens,
            args.temperature
        )
        
        if result:
            results.append(result)
            
        if success:
            logger.info(f"Vision test for {model_name} completed successfully")
            success_count += 1
        else:
            logger.error(f"Vision test for {model_name} failed")
    
    # Save results to JSON if specified
    if args.output_json and results:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")
    
    # Final summary
    logger.info(f"Vision testing completed: {success_count}/{len(models_to_test)} models successful")
    
    # Exit with error if any test failed
    if success_count < len(models_to_test):
        exit(1)

if __name__ == "__main__":
    main() 