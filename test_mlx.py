#!/usr/bin/env python3
"""
Test script for the MLX provider in AbstractLLM.
"""

import logging
import argparse
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import AbstractLLM
try:
    from abstractllm import create_llm
    from abstractllm.enums import ModelParameter
except ImportError:
    print("Failed to import AbstractLLM. Make sure it's installed.")
    sys.exit(1)

def test_text_generation(model_name: str, prompt: str, stream: bool = False):
    """Test basic text generation with the MLX provider."""
    print(f"\n=== Testing Text Generation ===")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Stream: {stream}")
    
    try:
        # Create the LLM
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.MAX_TOKENS: 100,  # Limit for quick testing
            ModelParameter.TEMPERATURE: 0.7
        })
        
        # Generate text
        if stream:
            print("\nResponse (streaming):")
            for chunk in llm.generate(prompt, stream=True):
                print(chunk.content, end="", flush=True)
            print("\n")
        else:
            print("\nGenerating response...")
            response = llm.generate(prompt)
            print(f"\nResponse: {response.content}")
            print(f"Usage: {response.usage}")
        
        print("\n=== Text Generation Test Completed Successfully ===")
        return True
    except Exception as e:
        print(f"\nError during text generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vision_generation(model_name: str, prompt: str, image_path: str, stream: bool = False):
    """Test vision generation with the MLX provider."""
    print(f"\n=== Testing Vision Generation ===")
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt}")
    print(f"Image: {image_path}")
    print(f"Stream: {stream}")
    
    try:
        # Check if image exists
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"Image file not found: {image_path}")
            return False
        
        # Create the LLM
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.MAX_TOKENS: 100,  # Limit for quick testing
            ModelParameter.TEMPERATURE: 0.7
        })
        
        # Generate text with image
        if stream:
            print("\nResponse (streaming):")
            for chunk in llm.generate(prompt, files=[image_path], stream=True):
                print(chunk.content, end="", flush=True)
            print("\n")
        else:
            print("\nGenerating response...")
            response = llm.generate(prompt, files=[image_path])
            print(f"\nResponse: {response.content}")
            print(f"Usage: {response.usage}")
        
        print("\n=== Vision Generation Test Completed Successfully ===")
        return True
    except Exception as e:
        print(f"\nError during vision generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test MLX provider in AbstractLLM')
    parser.add_argument('--text-model', default="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
                        help='Model name for text generation test')
    parser.add_argument('--vision-model', default="mlx-community/llava-1.5-7b-4bit",
                        help='Model name for vision generation test')
    parser.add_argument('--text-prompt', default="What is the capital of France?",
                        help='Prompt for text generation test')
    parser.add_argument('--vision-prompt', default="What's in this image?",
                        help='Prompt for vision generation test')
    parser.add_argument('--image', default="tests/examples/mountain_path.jpg",
                        help='Image path for vision generation test')
    parser.add_argument('--stream', action='store_true',
                        help='Use streaming generation')
    parser.add_argument('--test-type', choices=['text', 'vision', 'both'], default='both',
                        help='Type of test to run')
    
    args = parser.parse_args()
    
    success = True
    
    # Run tests based on test type
    if args.test_type in ['text', 'both']:
        text_success = test_text_generation(args.text_model, args.text_prompt, args.stream)
        success = success and text_success
    
    if args.test_type in ['vision', 'both']:
        vision_success = test_vision_generation(args.vision_model, args.vision_prompt, args.image, args.stream)
        success = success and vision_success
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 