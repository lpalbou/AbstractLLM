#!/usr/bin/env python3
"""
Test script for MLX vision capabilities in AbstractLLM with standardized image size.
"""

import os
import sys
import logging
from PIL import Image

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

# Define constants
MLX_VISION_MODEL = "mlx-community/llava-1.5-7b-4bit"  # Use a known-to-work model
TEST_IMAGE = "test_336.jpg"  # Use our precisely sized test image
IMAGE_SIZE = (336, 336)  # Standard size for LLaVA models

def preprocess_image(input_path, output_path=None, target_size=IMAGE_SIZE):
    """Standardize image to target size with aspect ratio preservation."""
    if output_path is None:
        output_path = input_path
        
    # Open the image
    img = Image.open(input_path)
    width, height = img.size
    
    # Check if resizing is needed
    if width == target_size[0] and height == target_size[1]:
        logger.info(f"Image already at target size {target_size}, skipping preprocessing")
        return input_path
        
    # Resize while maintaining aspect ratio
    if width > height:
        new_width = target_size[0]
        new_height = int((height / width) * target_size[0])
    else:
        new_height = target_size[1]
        new_width = int((width / height) * target_size[1])
        
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create a black background image in the target size
    final_img = Image.new("RGB", target_size, (0, 0, 0))
    
    # Paste the resized image onto the background, centered
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    final_img.paste(resized_img, (paste_x, paste_y))
    
    # Save the output
    final_img.save(output_path)
    logger.info(f"Standardized image saved to: {output_path}")
    return output_path

def main():
    """Run the MLX vision test."""
    # Ensure the test image exists or create it
    if not os.path.exists(TEST_IMAGE):
        logger.warning(f"Test image not found, creating a blank {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} image")
        img = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))
        img.save(TEST_IMAGE)
    else:
        # Make sure the test image is standardized
        preprocess_image(TEST_IMAGE, TEST_IMAGE, IMAGE_SIZE)
    
    logger.info(f"Using test image: {TEST_IMAGE}")
    logger.info(f"Using MLX vision model: {MLX_VISION_MODEL}")
    
    try:
        # Create the MLX provider with a vision model
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: MLX_VISION_MODEL,
            ModelParameter.TEMPERATURE: 0.2,
            ModelParameter.MAX_TOKENS: 200
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