#!/usr/bin/env python3
"""
Example script demonstrating the MLX provider in AbstractLLM.

This script shows how to use the MLX provider to generate text using a local MLX model.
"""

import argparse
import time
from pathlib import Path
from abstractllm import create_llm

def main():
    """Run the MLX provider example."""
    parser = argparse.ArgumentParser(description="Example using the MLX provider in AbstractLLM")
    parser.add_argument("--model", default="mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit",
                        help="Model to use for generation")
    parser.add_argument("--prompt", default="Explain what MLX is in 3 sentences.",
                        help="Prompt to use for generation")
    parser.add_argument("--system-prompt", 
                        help="Optional system prompt to use")
    parser.add_argument("--stream", action="store_true", 
                        help="Use streaming generation")
    parser.add_argument("--max-tokens", type=int, default=500,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--file", type=str, 
                        help="Optional file to include in the prompt")
    args = parser.parse_args()

    # Create the MLX provider
    print(f"Creating MLX provider with model: {args.model}")
    llm = create_llm("mlx", 
                     model=args.model,
                     temperature=args.temperature,
                     max_tokens=args.max_tokens)
    
    # Prepare files if specified
    files = None
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            files = [file_path]
            print(f"Including file: {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")

    # Generate text
    start_time = time.time()
    
    if args.stream:
        print("\nGenerating response (streaming):")
        print("-" * 50)
        for chunk in llm.generate(args.prompt, 
                                 system_prompt=args.system_prompt,
                                 files=files,
                                 stream=True):
            print(chunk.content, end="", flush=True)
        print("\n" + "-" * 50)
    else:
        print("\nGenerating response:")
        print("-" * 50)
        response = llm.generate(args.prompt, 
                               system_prompt=args.system_prompt,
                               files=files)
        print(response.content)
        print("-" * 50)
    
    elapsed = time.time() - start_time
    print(f"\nGeneration completed in {elapsed:.2f} seconds")
    
    # Show model capabilities
    capabilities = llm.get_capabilities()
    print("\nModel capabilities:")
    for cap, value in capabilities.items():
        print(f"- {cap}: {value}")

if __name__ == "__main__":
    main() 