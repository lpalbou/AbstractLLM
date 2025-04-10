#!/usr/bin/env python3
"""
Simple test script for AbstractLLM providers.
"""

import os
import sys
from pathlib import Path
import argparse

from abstractllm import create_llm
from abstractllm.enums import ModelParameter


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test AbstractLLM providers')
    parser.add_argument('prompt', help='The prompt to send to the model')
    parser.add_argument('--provider', '-p', default='openai', choices=['openai', 'anthropic', 'ollama', 'huggingface'],
                      help='Provider to use (default: openai)')
    parser.add_argument('--model', '-m', help='Specific model to use (if not specified, uses provider default)')
    parser.add_argument('--file', '-f', help='Optional file to process (image, text, csv, etc.)')
    parser.add_argument('--api-key', help='API key (can also use environment variable)')
    args = parser.parse_args()

    # Determine which environment variable to check based on provider
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY"
    }

    # Get API key from args or environment
    env_var = env_var_map.get(args.provider)
    api_key = None
    if env_var:
        api_key = args.api_key or os.environ.get(env_var)
        if not api_key:
            print(f"Error: {args.provider} API key not provided. Use --api-key or set {env_var} environment variable.")
            sys.exit(1)

    try:
        # Create provider configuration (empty by default)
        config = {}
        
        # Add API key if provided
        if api_key:
            config[ModelParameter.API_KEY] = api_key
            
        # Add model only if explicitly specified
        if args.model:
            config[ModelParameter.MODEL] = args.model
            print(f"\nInitializing {args.provider} provider with specified model: {args.model}")
        else:
            print(f"\nInitializing {args.provider} provider with default model")

        # Create provider instance
        llm = create_llm(args.provider, **config)

        # Prepare files list if file is provided
        files = [args.file] if args.file else None

        # Generate response
        print(f"\nSending request to {args.provider}...")
        response = llm.generate(
            prompt=args.prompt,
            files=files
        )

        print(f"\nResponse from {args.provider}:")
        print("=" * 40)
        print(response)
        print("=" * 40)

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 