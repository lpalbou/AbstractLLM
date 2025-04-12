#!/usr/bin/env python3
"""
Simple test script for AbstractLLM providers.
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse

from abstractllm import create_llm, configure_logging
from abstractllm.enums import ModelParameter

def ensure_logs_dir():
    """Ensure the logs directory exists."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir

def write_debug_info(provider: str, payload: dict):
    """Write debug information to a JSON file."""
    logs_dir = ensure_logs_dir()
    debug_file = logs_dir / "request.json"
    
    # Create a copy for logging to avoid modifying the original
    debug_payload = payload.copy()
    
    # If there's image data, add some debug info about it
    if "messages" in debug_payload:
        for message in debug_payload["messages"]:
            if "content" in message:
                for content in message["content"]:
                    if content.get("type") == "image" and content.get("source", {}).get("type") == "base64":
                        # Get the first and last 50 chars of base64 data
                        data = content["source"]["data"]
                        content["source"]["data_debug"] = {
                            "length": len(data),
                            "start": data[:50],
                            "end": data[-50:],
                            "mime_type": content["source"]["media_type"]
                        }
    
    debug_info = {
        "provider": provider,
        "payload": debug_payload
    }
    
    with open(debug_file, 'w') as f:
        json.dump(debug_info, f, indent=2)
    print(f"\nDebug info written to {debug_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test AbstractLLM providers')
    parser.add_argument('prompt', help='The prompt to send to the model')
    parser.add_argument('--provider', '-p', default='openai', choices=['openai', 'anthropic', 'ollama', 'huggingface'],
                      help='Provider to use (default: openai)')
    parser.add_argument('--model', '-m', help='Specific model to use (if not specified, uses provider default)')
    parser.add_argument('--file', '-f', help='Optional file to process (image, text, csv, etc.)')
    parser.add_argument('--api-key', help='API key (can also use environment variable)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to log the exact payload sent to provider')
    parser.add_argument('--log-dir', help='Directory to store logs (default: logs/ in current directory)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='Logging level (default: INFO)')
    parser.add_argument('--console-output', action='store_true', help='Force console output even when logging to files')
    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    log_dir = args.log_dir or os.path.join(os.getcwd(), "logs")
    
    # Set up logging configuration
    configure_logging(
        log_dir=log_dir,
        log_level=log_level,
        provider_level=log_level,
        console_output=args.console_output
    )
    
    # Log the configuration
    logger = logging.getLogger("abstractllm")
    logger.info(f"Logging to directory: {log_dir}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Console output: {'enabled' if args.console_output else 'auto'}")

    # Providers that always require API keys
    required_api_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }

    try:
        # Create provider configuration
        config = {}
        
        # Handle API key for providers that require it
        if args.provider in required_api_keys:
            env_var = required_api_keys[args.provider]
            api_key = args.api_key or os.environ.get(env_var)
            if not api_key:
                print(f"Error: {args.provider} API key not provided. Use --api-key or set {env_var} environment variable.")
                sys.exit(1)
            config[ModelParameter.API_KEY] = api_key
        else:
            # For other providers, add API key only if explicitly provided
            if args.api_key:
                config[ModelParameter.API_KEY] = args.api_key
            elif os.environ.get(f"{args.provider.upper()}_API_KEY"):
                config[ModelParameter.API_KEY] = os.environ.get(f"{args.provider.upper()}_API_KEY")
            
        # Add model only if explicitly specified
        if args.model:
            config[ModelParameter.MODEL] = args.model
            logger.info(f"Using specified model: {args.model}")
        else:
            logger.info("Using default model for provider")

        # Create provider instance
        llm = create_llm(args.provider, **config)

        # Prepare files list if file is provided
        files = [args.file] if args.file else None

        # If debug mode is enabled, we need to capture the payload
        if args.debug:
            # Get the provider instance to access its internals
            provider_instance = llm
            
            # Get necessary parameters
            model = provider_instance.config_manager.get_param(ModelParameter.MODEL)
            temperature = provider_instance.config_manager.get_param(ModelParameter.TEMPERATURE)
            max_tokens = provider_instance.config_manager.get_param(ModelParameter.MAX_TOKENS)
            
            # Process files if any
            processed_files = []
            if files:
                from abstractllm.media.factory import MediaFactory
                for file_path in files:
                    media_input = MediaFactory.from_source(file_path)
                    processed_files.append(media_input)
            
            # Prepare messages array
            content = []
            if processed_files:
                for media_input in processed_files:
                    content.append(media_input.to_provider_format(args.provider))
            content.append({
                "type": "text",
                "text": args.prompt
            })
            
            # Create the full payload that will be sent
            payload = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            # Write debug info
            write_debug_info(args.provider, payload)

        # Generate response
        logger.info(f"Sending request to {args.provider}...")
        response = llm.generate(
            prompt=args.prompt,
            files=files
        )

        print(f"\nResponse from {args.provider}:")
        print("=" * 40)
        print(response)
        print("=" * 40)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 