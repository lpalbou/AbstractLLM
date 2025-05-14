#!/usr/bin/env python3
"""
Test script for MLX vision capabilities in AbstractLLM.
"""

import os
import sys
import logging

# Configure logging with DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("test_mlx_vision")
# Enable debug logging for model factory and provider
logging.getLogger("abstractllm.providers.mlx_model_factory").setLevel(logging.DEBUG)
logging.getLogger("abstractllm.providers.mlx_provider").setLevel(logging.DEBUG)

# Import AbstractLLM
try:
    from abstractllm import create_llm
    from abstractllm.enums import ModelParameter, ModelCapability
except ImportError as e:
    logger.error(f"Failed to import AbstractLLM: {e}")
    logger.error("Make sure abstractllm is installed or PYTHONPATH is set correctly")
    sys.exit(1)

# Define constants - updated with locally available model from cache
MLX_VISION_MODEL = "mlx-community/llava-1.5-7b-4bit"  # Changed to use locally cached model
TEST_IMAGE = "./gh/mlx-vlm/examples/images/cats.jpg"  # Use a test image

def main():
    """Run the MLX vision test."""
    # Check if the test image exists
    if not os.path.exists(TEST_IMAGE):
        logger.error(f"Test image not found: {TEST_IMAGE}")
        logger.error("Please update the TEST_IMAGE path to point to a valid image")
        return 1
    
    logger.info(f"Using test image: {TEST_IMAGE}")
    logger.info(f"Using MLX vision model: {MLX_VISION_MODEL}")
    
    try:
        # Create the MLX provider with a vision model
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: MLX_VISION_MODEL,
            ModelParameter.TEMPERATURE: 0.2,  # Lower temperature for more deterministic results
            ModelParameter.MAX_TOKENS: 200     # Limit response length
        })
        
        # Check if vision is supported
        capabilities = llm.get_capabilities()
        logger.info(f"Model capabilities: {capabilities}")
        
        if not capabilities.get(ModelCapability.VISION):
            logger.error(f"Vision is not supported by the model: {MLX_VISION_MODEL}")
            return 1
        
        # Generate a response with the image
        logger.info("Generating response with image...")
        response = llm.generate(
            prompt="What can you see in this image? Describe it in detail.",
            files=[TEST_IMAGE]
        )
        
        # Print the response
        logger.info("Response:")
        print("\n" + "-" * 80)
        print(response.content)
        print("-" * 80 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 