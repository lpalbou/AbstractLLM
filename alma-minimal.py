#!/usr/bin/env python3
"""
Minimal ALMA (AbstractLLM Agent) implementation with file reading and image description capability.
Uses the simplest approach to tool calling with an interactive REPL.

# Requirements
- AbstractLLM: pip install abstractllm
- MLX: pip install mlx mlx-lm mlx-vlm (for Apple Silicon only)
- Tool support: pip install abstractllm[tools]
- Vision support: pip install PIL
"""

import os
import sys
import logging
from pathlib import Path
import importlib

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Import abstractllm
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.factory import get_llm_providers
from abstractllm.enums import ModelParameter

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def describe_image(image_path: str) -> str:
    """
    Load an image and prepare it for model description.
    The image will be processed by the vision model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        A confirmation message (the actual description will be provided by the model)
    """
    try:
        from PIL import Image
        
        # Check if the file exists
        if not os.path.exists(image_path):
            return f"Error: Image file not found at path {image_path}"
            
        # Try to open the image to validate it
        img = Image.open(image_path)
        img_format = img.format
        img_size = img.size
        
        # Return a confirmation that will be sent back to the LLM
        return f"Image loaded successfully: {image_path} (Format: {img_format}, Size: {img_size[0]}x{img_size[1]})"
    except Exception as e:
        return f"Error loading image: {str(e)}"

def create_provider():
    """Create an LLM provider based on available options."""
    # List available providers
    providers_list = get_llm_providers()
    print(f"Available providers: {', '.join(providers_list)}")
    
    providers_to_try = []
    
    # Add MLX to the list if available in providers (prioritize vision models)
    if "mlx" in providers_list:
        # Try MLX provider with vision-capable models first
        providers_to_try.append(("mlx", {
            ModelParameter.MODEL: "mlx-community/paligemma-3b-mix-448-8bit"  # Vision model
        }))
        # Fallback to text-only model if vision model fails
        providers_to_try.append(("mlx", {
            ModelParameter.MODEL: "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-6bit"
        }))
    
    # Add fallback providers
    providers_to_try.extend([
        ("ollama", {"model": "llava"}),  # Vision-capable Ollama model
        ("ollama", {"model": "qwen2.5"}),  # Text-only fallback
        # Add more fallbacks if needed
    ])
    
    # Try each provider in order
    for provider_name, config in providers_to_try:
        try:
            print(f"Attempting to create provider: {provider_name} with model {config.get(ModelParameter.MODEL, config.get('model', 'unknown'))}")
            provider = create_llm(provider_name, **config)
            print(f"Successfully created {provider_name} provider")
            
            # Check if provider has vision capability
            if hasattr(provider, "get_capabilities"):
                capabilities = provider.get_capabilities()
                has_vision = capabilities.get("vision", False)
                print(f"Provider vision capability: {'YES' if has_vision else 'NO'}")
            
            return provider
        except Exception as e:
            print(f"Failed to create {provider_name} provider: {e}")
    
    # If all providers failed, exit
    print("ERROR: No working providers found")
    sys.exit(1)

def main():
    # Create a provider
    provider = create_provider()
    
    # Create session with the provider and tool functions
    try:
        session = Session(
            system_prompt="You are a helpful assistant that can read files and describe images when needed. "
                        "If you need to see a file's contents, use the read_file tool. "
                        "If you need to describe an image, use the describe_image tool.",
            provider=provider,
            tools=[read_file, describe_image]  # Both tools are automatically registered
        )
        
        print(f"\nMinimal ALMA with {provider.__class__.__name__} - Type 'exit' to quit")
        print("Examples:")
        print("- 'Read the file README.md and summarize it'")
        print("- 'Describe the image in test_images/sample.jpg'")
        
        # Check if test_images directory exists, create if not
        test_dir = Path("test_images")
        if not test_dir.exists():
            test_dir.mkdir(exist_ok=True)
            print("\nℹ️ Created test_images directory for storing test images")
            print("  Place images there and refer to them as 'test_images/your_image.jpg'")
    except Exception as e:
        print(f"Failed to create session: {e}")
        print("Please install required dependencies: pip install 'abstractllm[tools]'")
        return
    
    # Simple REPL loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
                
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Generate response with tool support
            print("\nAssistant: ", end="")
            
            # Add debug output
            print("\n[DEBUG] Generating response...")
            
            try:
                # Use the unified generate method
                response = session.generate(
                    prompt=user_input,
                    max_tool_calls=3  # Limit tool calls to avoid infinite loops
                )
                
                print(f"[DEBUG] Response type: {type(response)}")
                
                # Handle different response types:
                # - If response has .content attribute (tool was used), use that
                # - If response is a string (direct answer, no tool used), use as is
                if hasattr(response, 'content'):
                    print(f"[DEBUG] Response has content attribute: {hasattr(response, 'content')}")
                    if hasattr(response, 'has_tool_calls'):
                        print(f"[DEBUG] Response has_tool_calls method: {response.has_tool_calls()}")
                    
                    # Check if we're dealing with a tool call request that hasn't been resolved
                    if hasattr(response, 'has_tool_calls') and response.has_tool_calls():
                        print("[DEBUG] Still getting tool calls after max_tool_calls reached - forcing direct question")
                        
                        # If we're still getting tool calls after max_tool_calls, 
                        # the model is stuck in a loop. Force a direct question instead.
                        # First, get the content from the last tool execution
                        tool_content = None
                        for message in session.messages:
                            if hasattr(message, 'tool_results') and message.tool_results:
                                # Get the last tool result content
                                tool_content = message.tool_results[-1].get('output', '')
                        
                        if tool_content:
                            # For a summarization task, we ask the model directly with the content
                            direct_prompt = f"Here is the content of the file that was read. Please provide a concise summary:\n\n{tool_content}"
                            
                            # Generate response without tool support (direct query)
                            print("[DEBUG] Sending direct question with file content...")
                            direct_response = provider.generate(
                                prompt=direct_prompt,
                                system_prompt="You are a helpful assistant summarizing file contents."
                            )
                            
                            print(direct_response)
                        else:
                            print("Unable to get content from tool execution. Please try again.")
                    else:
                        # Normal content response
                        print(response.content)
                else:
                    # Direct string response
                    print(f"[DEBUG] Direct string response")
                    print(response)
            except Exception as e:
                print(f"\nError generating response: {str(e)}")
                print("Please try again with a different query.")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 