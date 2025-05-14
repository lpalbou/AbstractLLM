#!/usr/bin/env python3
"""
Test script for using mlx_lm directly.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to test mlx_lm generation."""
    if len(sys.argv) < 2:
        print("Usage: python test_mlx_lm.py MODEL_NAME [PROMPT]")
        sys.exit(1)
    
    model_name = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello, world!"
    
    print(f"Testing mlx_lm with model: {model_name}")
    print(f"Prompt: {prompt}")
    
    try:
        # Import mlx_lm
        import mlx_lm
        
        # Load the model
        print("Loading model...")
        model, tokenizer = mlx_lm.load(model_name)
        print("Model loaded successfully")
        
        # Generate text
        print("Generating text...")
        text = mlx_lm.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=100,
            verbose=True
        )
        
        print("\nGenerated text:")
        print(text)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 