#!/usr/bin/env python3
"""
Basic test script for MLX provider in AbstractLLM.
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import AbstractLLM
try:
    from abstractllm import create_llm, ModelParameter
    print("Successfully imported AbstractLLM")
except ImportError as e:
    print(f"Failed to import AbstractLLM: {e}")
    sys.exit(1)

def test_mlx_provider():
    """Test basic functionality of MLX provider."""
    try:
        # Create MLX provider with an available model
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit",
            ModelParameter.MAX_TOKENS: 100  # Limit tokens for faster test
        })
        print(f"Successfully created MLX provider with model: {llm.config_manager.get_param(ModelParameter.MODEL)}")
        
        # Test basic generation
        prompt = "Hello, what is your name?"
        print(f"Testing generation with prompt: '{prompt}'")
        response = llm.generate(prompt)
        
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        
        return True
    except Exception as e:
        print(f"Error testing MLX provider: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting MLX provider test...")
    success = test_mlx_provider()
    print(f"Test {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1) 