#!/usr/bin/env python3
"""
Test script for direct vision generation in MLX provider.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AbstractLLM components
from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.enums import ModelParameter

def main():
    """Main function to test direct vision generation."""
    if len(sys.argv) < 3:
        print("Usage: python test_direct_vision.py MODEL_NAME IMAGE_PATH [PROMPT]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    image_path = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Describe this image"
    
    print(f"Testing direct vision generation with model: {model_name}")
    print(f"Image path: {image_path}")
    print(f"Prompt: {prompt}")
    
    try:
        # Create provider
        provider = MLXProvider({
            ModelParameter.MODEL: model_name,
            ModelParameter.MAX_TOKENS: 512,
            ModelParameter.TEMPERATURE: 0.2
        })
        
        print("Provider created successfully")
        print(f"Is vision model: {provider._is_vision_model}")
        print(f"Model type: {provider._model_type}")
        
        # Load model
        provider.load_model()
        print("Model loaded successfully")
        
        # Inspect model structure
        print("\nModel structure:")
        print(f"Model type: {type(provider._model)}")
        print(f"Model attributes: {dir(provider._model)}")
        
        if hasattr(provider._model, 'model'):
            print(f"\nModel.model type: {type(provider._model.model)}")
            print(f"Model.model attributes: {dir(provider._model.model)}")
            
            # Try to find the forward method
            if hasattr(provider._model.model, 'forward'):
                import inspect
                print("\nForward method signature:")
                try:
                    print(inspect.signature(provider._model.model.forward))
                except Exception as e:
                    print(f"Could not get signature: {e}")
        
        # Get tokenizer
        if hasattr(provider._processor, 'tokenizer'):
            tokenizer = provider._processor.tokenizer
        else:
            tokenizer = provider._processor
        
        # Use the direct generation method
        print("Starting direct generation...")
        text = provider._generate_vision_directly(
            provider._model,
            tokenizer,
            image_path,
            prompt,
            max_tokens=100
        )
        
        print("\nGenerated text:")
        print(text)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 