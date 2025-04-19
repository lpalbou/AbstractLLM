#!/usr/bin/env python3
"""
Test script for the refactored HuggingFace provider implementation with direct pipeline usage.
This version bypasses most of the provider infrastructure to directly test the text pipeline.
"""

import os
import sys
import json
import logging
from pathlib import Path
import argparse

from abstractllm import configure_logging
from abstractllm.enums import ModelParameter
from abstractllm.media.factory import MediaFactory
from abstractllm.media.text import TextInput
from abstractllm.providers.huggingface.text_pipeline import TextGenerationPipeline
from abstractllm.providers.huggingface.model_types import ModelConfig, ModelArchitecture

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
    parser = argparse.ArgumentParser(description='Test refactored HuggingFace TextGenerationPipeline')
    parser.add_argument('prompt', help='The prompt to send to the model')
    parser.add_argument('--model', '-m', 
                       default='https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_L.gguf',
                       help='Model to use (default: Phi-4-mini GGUF)')
    parser.add_argument('--file', '-f', help='Optional file to process (image, text, csv, etc.)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to log the exact payload sent to provider')
    parser.add_argument('--log-dir', help='Directory to store logs (default: logs/ in current directory)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='DEBUG',
                       help='Logging level (default: DEBUG)')
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
        console_output=True
    )
    
    # Create a file handler to capture detailed logs
    log_file = os.path.join(log_dir, "huggingface_test.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the root logger and the abstractllm loggers
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Log the configuration
    logger = logging.getLogger("abstractllm")
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Using model: {args.model}")
    
    # Add very verbose logging for troubleshooting
    root_logger.setLevel(log_level)

    try:
        # Create model configuration 
        model_config = ModelConfig(
            architecture=ModelArchitecture.DECODER_ONLY,
            trust_remote_code=True,
            use_flash_attention=False,
            device_map="auto"
        )
        
        # Create the pipeline directly
        logger.info("Creating text generation pipeline...")
        pipeline = TextGenerationPipeline()
        
        # Load the model
        logger.info(f"Loading model: {args.model}")
        pipeline.load(args.model, model_config)
        
        # Prepare the input
        text_input = TextInput(args.prompt)
        
        # Process the input
        logger.info("Generating response...")
        result = pipeline.process([text_input], max_new_tokens=1024, temperature=0.7)
        
        print("\nResponse from TextGenerationPipeline:")
        print("=" * 50)
        print(result)
        print("=" * 50)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main() 