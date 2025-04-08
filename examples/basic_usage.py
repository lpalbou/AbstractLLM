#!/usr/bin/env python
"""
Basic usage examples for AbstractLLM.

This script demonstrates how to use AbstractLLM with different providers.
"""

import os
import sys
import logging
from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.utils.logging import setup_logging

def main():
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Example 1: Using OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        print("\n=== OpenAI Example ===")
        
        # Create the LLM instance
        llm = create_llm("openai", **{
            ModelParameter.MODEL: "gpt-3.5-turbo",
            ModelParameter.TEMPERATURE: 0.7
        })
        
        # Check capabilities
        capabilities = llm.get_capabilities()
        print(f"Capabilities: {capabilities}")
        
        # Generate a response
        prompt = "Explain quantum computing in simple terms."
        print(f"\nPrompt: {prompt}")
        
        response = llm.generate(prompt)
        print(f"\nResponse: {response}")
        
        # Using a system prompt
        system_prompt = "You are a university professor explaining concepts to undergraduates."
        print(f"\nSystem prompt: {system_prompt}")
        
        response = llm.generate(prompt, system_prompt=system_prompt)
        print(f"\nResponse with system prompt: {response}")
        
        # Streaming example
        if capabilities.get(ModelCapability.STREAMING):
            print("\nStreaming response:")
            stream = llm.generate("Count from 1 to 5 slowly.", stream=True)
            for chunk in stream:
                print(chunk, end="", flush=True)
            print()
    else:
        print("Skipping OpenAI example (OPENAI_API_KEY not set)")
    
    # Example 2: Using Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n=== Anthropic Example ===")
        
        # Create the LLM instance
        llm = create_llm("anthropic", **{
            ModelParameter.MODEL: "claude-instant-1.2",
            ModelParameter.TEMPERATURE: 0.7
        })
        
        # Check capabilities
        capabilities = llm.get_capabilities()
        print(f"Capabilities: {capabilities}")
        
        # Generate a response
        prompt = "Explain machine learning in simple terms."
        print(f"\nPrompt: {prompt}")
        
        response = llm.generate(prompt)
        print(f"\nResponse: {response}")
        
        # Using a system prompt
        if capabilities.get(ModelCapability.SYSTEM_PROMPT):
            system_prompt = "You are a high school teacher explaining concepts to students."
            print(f"\nSystem prompt: {system_prompt}")
            
            response = llm.generate(prompt, system_prompt=system_prompt)
            print(f"\nResponse with system prompt: {response}")
    else:
        print("Skipping Anthropic example (ANTHROPIC_API_KEY not set)")
    
    # Example 3: Using Ollama (if available)
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print("\n=== Ollama Example ===")
                
                # Create the LLM instance with the first available model
                model_name = models[0]["name"]
                print(f"Using model: {model_name}")
                
                llm = create_llm("ollama", **{
                    ModelParameter.BASE_URL: "http://localhost:11434",
                    ModelParameter.MODEL: model_name
                })
                
                # Generate a response
                prompt = "Write a haiku about programming."
                print(f"\nPrompt: {prompt}")
                
                response = llm.generate(prompt)
                print(f"\nResponse: {response}")
        else:
            print("Skipping Ollama example (Ollama not running)")
    except Exception:
        print("Skipping Ollama example (Ollama not available)")

if __name__ == "__main__":
    main() 