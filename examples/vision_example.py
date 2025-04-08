#!/usr/bin/env python
"""
Vision capability example for AbstractLLM.

This script demonstrates how to use AbstractLLM with vision-capable models.
"""

import os
import sys
import logging
from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.utils.logging import setup_logging

def main():
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Example image URL (Eiffel Tower)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg"
    
    # Example 1: Using OpenAI Vision model
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("USE_VISION", "false").lower() == "true":
        print("\n=== OpenAI Vision Example ===")
        
        # Create the LLM instance with a vision-capable model
        llm = create_llm("openai", **{
            ModelParameter.MODEL: "gpt-4o",  # Vision-capable model
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 500
        })
        
        # Check if vision is supported
        capabilities = llm.get_capabilities()
        if not capabilities.get(ModelCapability.VISION):
            print("Vision capability not supported with the selected model")
            print("Try using gpt-4-vision-preview or gpt-4o instead")
        else:
            print("Vision capability supported!")
            
            # Use the vision capability with an image URL
            prompt = "What can you see in this image? Please describe it in detail."
            print(f"\nPrompt: {prompt}")
            print(f"Image: {image_url}")
            
            response = llm.generate(prompt, image=image_url)
            print(f"\nResponse: {response}")
    else:
        print("Skipping OpenAI Vision example (OPENAI_API_KEY not set or USE_VISION=false)")
    
    # Example 2: Using Anthropic Claude Vision model
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("USE_VISION", "false").lower() == "true":
        print("\n=== Anthropic Vision Example ===")
        
        # Create the LLM instance with a vision-capable model
        llm = create_llm("anthropic", **{
            ModelParameter.MODEL: "claude-3-5-sonnet-20240620",  # Vision-capable model
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 500
        })
        
        # Check if vision is supported
        capabilities = llm.get_capabilities()
        if not capabilities.get(ModelCapability.VISION):
            print("Vision capability not supported with the selected model")
            print("Try using claude-3-opus, claude-3-sonnet, or claude-3-haiku instead")
        else:
            print("Vision capability supported!")
            
            # Use the vision capability with an image URL
            prompt = "What can you see in this image? Please describe it in detail."
            print(f"\nPrompt: {prompt}")
            print(f"Image: {image_url}")
            
            response = llm.generate(prompt, image=image_url)
            print(f"\nResponse: {response}")
    else:
        print("Skipping Anthropic Vision example (ANTHROPIC_API_KEY not set or USE_VISION=false)")
    
    # Example 3: Using Ollama with a vision-capable model
    if os.environ.get("USE_VISION", "false").lower() == "true":
        print("\n=== Ollama Vision Example ===")
        
        # Try to check if Ollama is running and if a vision model is available
        vision_model = None
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Look for vision-capable models
                for model in models:
                    model_name = model.get("name", "")
                    if "vision" in model_name.lower() or "janus" in model_name.lower():
                        vision_model = model_name
                        break
        except Exception:
            pass
        
        if not vision_model:
            print("No vision-capable Ollama model found. Try pulling one:")
            print("  ollama pull llama3.2-vision:latest")
            print("  ollama pull erwan2/DeepSeek-Janus-Pro-7B:latest")
        else:
            # Create the LLM instance with the found vision-capable model
            llm = create_llm("ollama", **{
                ModelParameter.MODEL: vision_model,
                ModelParameter.TEMPERATURE: 0.7,
                ModelParameter.MAX_TOKENS: 500
            })
            
            # Check if vision is supported
            capabilities = llm.get_capabilities()
            if not capabilities.get(ModelCapability.VISION):
                print(f"Vision capability not supported with the model {vision_model}")
            else:
                print(f"Vision capability supported with model {vision_model}!")
                
                # Use the vision capability with an image URL
                prompt = "What can you see in this image? Please describe it in detail."
                print(f"\nPrompt: {prompt}")
                print(f"Image: {image_url}")
                
                response = llm.generate(prompt, image=image_url)
                print(f"\nResponse: {response}")
    else:
        print("Skipping Ollama Vision example (USE_VISION=false)")
    
    # Multiple images example with OpenAI
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("USE_VISION", "false").lower() == "true":
        print("\n=== Multiple Images Example (OpenAI) ===")
        
        # Create the LLM instance with a vision-capable model
        llm = create_llm("openai", **{
            ModelParameter.MODEL: "gpt-4o",  # Vision-capable model
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 500
        })
        
        # Check if vision is supported
        capabilities = llm.get_capabilities()
        if capabilities.get(ModelCapability.VISION):
            # Example with multiple images
            image_url1 = "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg"
            image_url2 = "https://upload.wikimedia.org/wikipedia/commons/4/4b/La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg"
            
            prompt = "Compare these two images of the Eiffel Tower. What are the differences in perspective, lighting, and composition?"
            print(f"\nPrompt: {prompt}")
            print(f"Images: {image_url1}, {image_url2}")
            
            response = llm.generate(prompt, images=[image_url1, image_url2])
            print(f"\nResponse: {response}")

if __name__ == "__main__":
    main() 