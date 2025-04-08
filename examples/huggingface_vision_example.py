#!/usr/bin/env python
"""
HuggingFace vision capability example for AbstractLLM.

This script demonstrates how to use AbstractLLM with HuggingFace vision-capable models.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.utils.logging import setup_logging
from abstractllm.providers.huggingface import VISION_CAPABLE_MODELS

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test HuggingFace vision capabilities")
    parser.add_argument("--model", choices=VISION_CAPABLE_MODELS, default="microsoft/Phi-4-multimodal-instruct",
                      help="Vision-capable model to use")
    parser.add_argument("--image", default="https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg",
                      help="URL or path to an image to analyze")
    parser.add_argument("--prompt", default="What can you see in this image? Please describe it in detail.",
                      help="Prompt to use for image analysis")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                      help="Device to run the model on")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    args = parser.parse_args()

    # Set up logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    
    print(f"\n=== HuggingFace Vision Example: {args.model} ===")
    
    # Create the LLM instance with a vision-capable model
    llm = create_llm("huggingface", **{
        ModelParameter.MODEL: args.model,
        ModelParameter.DEVICE: args.device,
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 500,
        "trust_remote_code": True,
        "load_timeout": 600  # Longer timeout for larger models
    })
    
    # Check if vision is supported
    capabilities = llm.get_capabilities()
    if not capabilities.get(ModelCapability.VISION):
        print(f"Vision capability not supported with the selected model: {args.model}")
        print(f"Available vision-capable models: {', '.join(VISION_CAPABLE_MODELS)}")
        sys.exit(1)
    else:
        print("Vision capability supported!")
        
    # Use the vision capability with an image URL or file path
    print(f"\nPrompt: {args.prompt}")
    print(f"Image: {args.image}")
    
    # Generate response
    try:
        response = llm.generate(args.prompt, image=args.image)
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"Error generating response: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 