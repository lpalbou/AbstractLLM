#!/usr/bin/env python
"""
Media handling example for AbstractLLM.

This script demonstrates how to use AbstractLLM with the new media handling system.
"""

import os
import sys
import logging
from pathlib import Path
from abstractllm import create_llm, ImageInput, ModelParameter, ModelCapability
from abstractllm.utils.logging import setup_logging

def main():
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Example image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg"
    
    print("\n=== Media Handling Examples ===")
    
    # Example 1: Basic Image Handling with URLs
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("USE_VISION", "false").lower() == "true":
        print("\n=== Basic Image Handling (OpenAI) ===")
        
        # Create the LLM instance with a vision-capable model
        llm = create_llm("openai", **{
            ModelParameter.MODEL: "gpt-4o",
            ModelParameter.TEMPERATURE: 0.7
        })
        
        # Check if vision is supported
        capabilities = llm.get_capabilities()
        if not capabilities.get(ModelCapability.VISION):
            print("Vision capability not supported with the selected model")
        else:
            # Simple URL-based image example
            prompt = "What landmark is shown in this image?"
            print(f"\nPrompt: {prompt}")
            print(f"Image URL: {image_url}")
            
            # Use the new image parameter
            response = llm.generate(prompt, image=image_url)
            print(f"\nResponse: {response}")
    else:
        print("Skipping OpenAI example (OPENAI_API_KEY not set or USE_VISION=false)")
    
    # Example 2: Advanced Image Handling with ImageInput
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("USE_VISION", "false").lower() == "true":
        print("\n=== Advanced Image Handling (Anthropic) ===")
        
        # Create the LLM instance with a vision-capable model
        llm = create_llm("anthropic", **{
            ModelParameter.MODEL: "claude-3-opus-20240229",
            ModelParameter.TEMPERATURE: 0.7
        })
        
        # Check if vision is supported
        capabilities = llm.get_capabilities()
        if not capabilities.get(ModelCapability.VISION):
            print("Vision capability not supported with the selected model")
        else:
            # Create an ImageInput with detail level
            image = ImageInput(image_url, detail_level="high")
            
            prompt = "Describe this landmark in detail, including its architectural features."
            print(f"\nPrompt: {prompt}")
            print(f"Image: {image_url} (with high detail level)")
            
            # Use the ImageInput object
            response = llm.generate(prompt, image=image)
            print(f"\nResponse: {response}")
    else:
        print("Skipping Anthropic example (ANTHROPIC_API_KEY not set or USE_VISION=false)")
    
    # Example 3: Multiple Images
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("USE_VISION", "false").lower() == "true":
        print("\n=== Multiple Images Example (OpenAI) ===")
        
        # Create the LLM instance with a vision-capable model
        llm = create_llm("openai", **{
            ModelParameter.MODEL: "gpt-4o",
            ModelParameter.TEMPERATURE: 0.7
        })
        
        # Check if vision is supported
        capabilities = llm.get_capabilities()
        if capabilities.get(ModelCapability.VISION):
            # Example with multiple images
            image_url1 = "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg"
            image_url2 = "https://upload.wikimedia.org/wikipedia/commons/4/4b/La_Tour_Eiffel_vue_de_la_Tour_Saint-Jacques%2C_Paris_ao%C3%BBt_2014_%282%29.jpg"
            
            prompt = "Compare these two images of the Eiffel Tower. What are the differences in perspective and lighting?"
            print(f"\nPrompt: {prompt}")
            print(f"Images: Two different views of the Eiffel Tower")
            
            # Use the images parameter with a list of URLs
            response = llm.generate(prompt, images=[image_url1, image_url2])
            print(f"\nResponse: {response}")
    else:
        print("Skipping multiple images example (OPENAI_API_KEY not set or USE_VISION=false)")
    
    # Example 4: Ollama with Vision
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200 and os.environ.get("USE_VISION", "false").lower() == "true":
            models = response.json().get("models", [])
            vision_model = None
            
            # Find a vision-capable model
            for model in models:
                model_name = model.get("name", "").lower()
                if any(vision_term in model_name for vision_term in ["llava", "vision", "multimodal", "bakllava"]):
                    vision_model = model.get("name")
                    break
            
            if vision_model:
                print(f"\n=== Ollama Vision Example ({vision_model}) ===")
                
                # Create LLM with vision-capable model
                llm = create_llm("ollama", **{
                    ModelParameter.MODEL: vision_model,
                    ModelParameter.BASE_URL: "http://localhost:11434"
                })
                
                capabilities = llm.get_capabilities()
                if capabilities.get(ModelCapability.VISION):
                    prompt = "What famous landmark is shown in this image?"
                    print(f"\nPrompt: {prompt}")
                    print(f"Image URL: {image_url}")
                    
                    # Use with Ollama
                    response = llm.generate(prompt, image=image_url)
                    print(f"\nResponse: {response}")
            else:
                print("\nNo vision-capable Ollama models found. Try installing one with 'ollama pull llava'")
    except Exception as e:
        print(f"\nSkipping Ollama example: {e}")

if __name__ == "__main__":
    main() 