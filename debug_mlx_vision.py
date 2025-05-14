#!/usr/bin/env python3
"""
Debug script for MLX-VLM vision capabilities.
This script tries to use MLX-VLM directly without AbstractLLM to diagnose issues.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to test MLX-VLM directly."""
    print("\n=== MLX-VLM Direct Debug Test ===")
    
    # Test image - check if exists
    test_image = "tests/examples/mountain_path.jpg"
    if not os.path.exists(test_image):
        test_image = input("Enter path to a test image: ")
        if not os.path.exists(test_image):
            print(f"Image not found: {test_image}")
            return
    
    # Test model
    test_model = "mlx-community/llava-1.5-7b-4bit"
    test_prompt = "Describe what you see in this image."
    
    print(f"Testing with model: {test_model}")
    print(f"Testing with image: {test_image}")
    print(f"Testing with prompt: {test_prompt}")
    
    # Step 1: Check imports
    print("\nStep 1: Checking imports...")
    
    try:
        import mlx.core as mx
        print("✓ MLX imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MLX: {e}")
        return
    
    try:
        import mlx_vlm
        print(f"✓ MLX-VLM imported successfully (version: {mlx_vlm.__version__})")
        print("Available functions:", [f for f in dir(mlx_vlm) if not f.startswith('_')])
    except (ImportError, AttributeError) as e:
        print(f"✗ Failed to import MLX-VLM: {e}")
        return
    
    try:
        from PIL import Image
        print("✓ PIL imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PIL: {e}")
        return
    
    try:
        import torch
        print(f"✓ PyTorch imported successfully (version: {torch.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return
    
    # Step 2: Load test image
    print("\nStep 2: Loading test image...")
    try:
        image = Image.open(test_image)
        print(f"✓ Image loaded successfully: {image.width}x{image.height}, {image.format}, {image.mode}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return
    
    # Step 3: Try loading the model directly
    print("\nStep 3: Loading model directly...")
    try:
        # Use mlx_vlm.load instead of mlx_vlm.utils.load_vlm
        model, processor = mlx_vlm.load(test_model)
        print("✓ Model and processor loaded successfully")
        
        # Fix patch_size in processor if it's None
        if hasattr(processor, 'patch_size') and processor.patch_size is None:
            # Common patch_size values: 14 or 16
            processor.patch_size = 14
            print(f"✓ Fixed missing patch_size in processor (set to {processor.patch_size})")
        elif hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'patch_size') and processor.image_processor.patch_size is None:
            processor.image_processor.patch_size = 14
            print(f"✓ Fixed missing patch_size in image_processor (set to {processor.image_processor.patch_size})")
        
        # Print processor attributes
        print("\nProcessor Information:")
        if hasattr(processor, 'image_processor'):
            attrs = dir(processor.image_processor)
            print(f"Image processor attributes: {[a for a in attrs if not a.startswith('_')]}")
        else:
            attrs = dir(processor)
            print(f"Processor attributes: {[a for a in attrs if not a.startswith('_')]}")
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Format prompt with image tag
    print("\nStep 4: Formatting prompt...")
    
    try:
        # Get model type from model name
        model_type = "llava"  # Default
        if "phi-3" in test_model.lower() and "vision" in test_model.lower():
            model_type = "phi3_v"
            formatted_prompt = f"Image: <image>\nUser: {test_prompt}\nAssistant:"
        elif "llava" in test_model.lower():
            model_type = "llava"
            formatted_prompt = f"<image>\n{test_prompt}"
        elif "qwen" in test_model.lower() and ("vl" in test_model.lower() or "vision" in test_model.lower()):
            model_type = "qwen-vl"
            formatted_prompt = f"<img>{test_prompt}"
        else:
            # Generic fallback
            formatted_prompt = f"<image>\n{test_prompt}"
        
        print(f"✓ Prompt formatted for model type '{model_type}': {formatted_prompt}")
    except Exception as e:
        print(f"✗ Failed to format prompt: {e}")
        return
    
    # Check if image needs to be processed first
    print("\nStep 4.5: Pre-processing image...")
    try:
        processed_image = None
        if hasattr(mlx_vlm, 'process_image'):
            processed_image = mlx_vlm.process_image(image)
            print("✓ Pre-processed image using mlx_vlm.process_image")
        elif hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'preprocess'):
            # For transformers-style processors
            processed_image = processor.image_processor.preprocess(image, return_tensors="pt")
            print("✓ Pre-processed image using processor.image_processor.preprocess")
    except Exception as e:
        print(f"✗ Failed to pre-process image: {e}")
        processed_image = None
        print("Using original image without pre-processing")
    
    # Step 5: Generate from image
    print("\nStep 5: Generating text from image...")
    try:
        image_input = processed_image if processed_image is not None else image
        
        # First try with process_inputs
        from mlx_vlm.utils import process_inputs
        print("Trying to use process_inputs directly...")
        try:
            inputs = mlx_vlm.utils.process_inputs(processor, [image_input], [formatted_prompt])
            print("✓ Successfully processed inputs")
        except Exception as e:
            print(f"✗ Failed to process inputs directly: {e}")
        
        # Try generation 
        response = mlx_vlm.generate(
            model,
            processor,
            prompt=formatted_prompt,
            image=image_input,
            max_tokens=100,
            temperature=0.1
        )
        print("\n=== Generation Result ===")
        print(response)
        print("=== End of Result ===")
    except Exception as e:
        print(f"✗ Failed to generate text from image: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with a different model
        print("\nTrying with a different model...")
        try:
            print("Loading BakLLaVA model instead...")
            alt_model = "mlx-community/bakllava-1-4bit"
            model2, processor2 = mlx_vlm.load(alt_model)
            
            # Fix patch_size if needed
            if hasattr(processor2, 'patch_size') and processor2.patch_size is None:
                processor2.patch_size = 14
            elif hasattr(processor2, 'image_processor') and hasattr(processor2.image_processor, 'patch_size') and processor2.image_processor.patch_size is None:
                processor2.image_processor.patch_size = 14
                
            alt_formatted_prompt = f"<image>\n{test_prompt}"
            response = mlx_vlm.generate(
                model2,
                processor2,
                prompt=alt_formatted_prompt,
                image=image,
                max_tokens=100
            )
            print("\n=== Generation Result (Alternative Model) ===")
            print(response)
            print("=== End of Result ===")
        except Exception as e2:
            print(f"✗ Alternative model also failed: {e2}")
            traceback.print_exc()

if __name__ == "__main__":
    main() 