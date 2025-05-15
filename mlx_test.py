#!/usr/bin/env python3
"""
Simple test script for MLX provider.

This script tests text generation with different model architectures.
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from abstractllm import create_llm
    from abstractllm.enums import ModelParameter
    from abstractllm.types import GenerateResponse
except ImportError:
    logger.error("AbstractLLM not found. Please install it or ensure it's in your PYTHONPATH.")
    sys.exit(1)

# Test models by architecture
TEST_MODELS = {
    "qwen": "mlx-community/Qwen3-4B-4bit",  # Qwen models work well with MLX
    "llama": "mlx-community/Qwen3-4B-4bit",  # Fallback to Qwen when llama is selected
    "phi": "mlx-community/defog-sqlcoder-7b-2",  # SQL code generation model
    "code": "mlx-community/defog-sqlcoder-7b-2",  # Another entry for code models
    "mistral": "mlx-community/mixtral-8x22b-4bit"
}

# Test prompts with and without system prompt
TEST_PROMPTS = [
    {
        "name": "Simple question",
        "prompt": "What is the capital of France?",
        "system_prompt": None
    },
    {
        "name": "Question with system prompt",
        "prompt": "What is your favorite food?", 
        "system_prompt": "You are a helpful assistant who loves Italian cuisine."
    }
]

def check_dependencies() -> bool:
    """Check if MLX dependencies are installed."""
    try:
        import mlx
        logger.info(f"Found MLX version: {getattr(mlx, '__version__', 'unknown')}")
    except ImportError:
        logger.error("MLX not found. Install with: pip install mlx")
        return False
    
    try:
        import mlx_lm
        logger.info(f"Found MLX-LM version: {getattr(mlx_lm, '__version__', 'unknown')}")
    except ImportError:
        logger.error("MLX-LM not found. Install with: pip install mlx-lm")
        return False
    
    return True

def test_model(model_name: str, temperature: float, max_tokens: int) -> None:
    """Test a specific model with all test prompts."""
    logger.info(f"Testing model: {model_name}")
    
    try:
        # Create LLM
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.TEMPERATURE: temperature,
            ModelParameter.MAX_TOKENS: max_tokens
        })
        
        logger.info(f"Model created successfully")
        
        # Run tests
        for test in TEST_PROMPTS:
            logger.info(f"Running test: {test['name']}")
            
            start_time = time.time()
            response = llm.generate(
                prompt=test["prompt"],
                system_prompt=test["system_prompt"]
            )
            duration = time.time() - start_time
            
            logger.info(f"Response (took {duration:.2f}s):")
            logger.info("-" * 50)
            print(response.content)
            logger.info("-" * 50)
            
            if hasattr(response, "usage") and response.usage:
                logger.info(f"Token usage: {response.usage}")
            
            # A little breathing room between tests
            time.sleep(1)
        
    except Exception as e:
        logger.error(f"Error testing {model_name}: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MLX models")
    parser.add_argument("--model", type=str, choices=TEST_MODELS.keys(),
                        help="Model architecture to test (qwen, llama, phi, mistral)")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="Temperature (0-1)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    
    print("NOTE: This test script uses specific MLX-compatible models from Hugging Face.")
    print("For best results, ensure you're using models that have been properly converted to MLX format.")
    print("Visit https://huggingface.co/mlx-community for available models.")
    
    args = parser.parse_args()
    
    if not check_dependencies():
        sys.exit(1)
    
    # If model specified, test only that model
    if args.model:
        model_name = TEST_MODELS[args.model]
        logger.info(f"Testing specific model architecture: {args.model}")
        test_model(model_name, args.temp, args.max_tokens)
    else:
        # Otherwise, test all models
        logger.info(f"Testing all model architectures")
        for arch_name, model_name in TEST_MODELS.items():
            logger.info(f"Testing architecture: {arch_name}")
            test_model(model_name, args.temp, args.max_tokens)
            # Give time between models to cool down
            time.sleep(2)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 