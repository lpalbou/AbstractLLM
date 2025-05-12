#!/usr/bin/env python3
"""
Simple test script for the MLX provider in AbstractLLM.
This script demonstrates basic usage of the MLX provider with a real model.
"""

import time
import argparse
from pathlib import Path
from abstractllm import create_llm

def main():
    """Run a simple test of the MLX provider."""
    parser = argparse.ArgumentParser(description="Test the MLX provider in AbstractLLM")
    parser.add_argument("--model", default="mlx-community/Mistral-7B-Instruct-v0.2-4bit-MLX",
                        help="Model to use for testing")
    parser.add_argument("--prompt", default="Explain what MLX is in 3 sentences.",
                        help="Prompt to use for generation")
    parser.add_argument("--stream", action="store_true", help="Use streaming generation")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--files", nargs="+", type=str, default=None, 
                        help="Paths to files to include in the prompt")
    
    args = parser.parse_args()
    
    print(f"Testing MLX provider with model: {args.model}")
    print(f"Prompt: {args.prompt}")
    if args.system_prompt:
        print(f"System prompt: {args.system_prompt}")
    if args.files:
        print(f"Files: {', '.join(args.files)}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Streaming: {args.stream}")
    print("-" * 50)
    
    # Create the LLM
    start_time = time.time()
    llm = create_llm(
        "mlx",
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Convert file paths to Path objects if provided
    files = None
    if args.files:
        files = [Path(f) for f in args.files]
    
    # Generate text
    if args.stream:
        print("Streaming response:")
        for chunk in llm.generate(
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            files=files,
            stream=True
        ):
            print(chunk.content, end="", flush=True)
        print("\n")
    else:
        print("Generating response...")
        response = llm.generate(
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            files=files
        )
        print("\nResponse:")
        print(response.content)
        
        # Print usage statistics
        if hasattr(response, 'usage') and response.usage:
            print("\nUsage statistics:")
            for key, value in response.usage.items():
                print(f"  {key}: {value}")
    
    # Print timing information
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    
    # Print capabilities
    print("\nProvider capabilities:")
    for capability, value in llm.get_capabilities().items():
        print(f"  {capability}: {value}")

if __name__ == "__main__":
    main() 