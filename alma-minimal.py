#!/usr/bin/env python3
"""
Minimal ALMA (AbstractLLM Agent) implementation with file reading capability.
Uses the simplest approach to tool calling with an interactive REPL.

# Requirements
- AbstractLLM: pip install abstractllm
- MLX: pip install mlx mlx-lm
- Tool support: pip install abstractllm[tools]
- Apple Silicon Mac (M1/M2/M3 series) for MLX
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
from abstractllm.providers.registry import register_provider, get_available_providers
from abstractllm.factory import get_llm_providers
from abstractllm.enums import ModelParameter

# Register MLX provider
def register_mlx_provider():
    """Register the MLX provider if available."""
    try:
        # Check if MLX dependencies are available
        import mlx.core
        import mlx_lm
        
        # Check if running on Apple Silicon
        import platform
        is_macos = platform.system().lower() == "darwin" 
        is_arm = platform.processor() == "arm"
        if not (is_macos and is_arm):
            print("MLX requires Apple Silicon. Current platform doesn't support MLX.")
            return False
        
        # Direct import approach
        try:
            from abstractllm.providers.mlx_provider import MLXProvider
            
            # Register the provider directly
            register_provider("mlx", "abstractllm.providers.mlx_provider", "MLXProvider")
            print("MLX provider registered successfully.")

            # Add MLX to the factory's _PROVIDERS dictionary directly
            import abstractllm.factory
            if "mlx" not in abstractllm.factory._PROVIDERS:
                abstractllm.factory._PROVIDERS["mlx"] = "abstractllm.providers.mlx_provider.MLXProvider"
                print("Added MLX to factory providers list.")

            return True
        except ImportError as e:
            print(f"Could not import MLXProvider: {e}")
            return False
    except ImportError as e:
        print(f"Failed to register MLX provider: {e}")
        return False
    except Exception as e:
        print(f"Error registering MLX provider: {e}")
        return False

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def create_provider():
    """Create an LLM provider based on available options."""
    # Register MLX provider if not already registered
    mlx_available = register_mlx_provider()
    
    # List available providers
    providers_list = get_llm_providers()
    registry = get_available_providers()
    print(f"Factory providers: {', '.join(providers_list)}")
    print(f"Registry providers: {', '.join(registry.keys())}")
    
    providers_to_try = []
    
    # Add MLX to the list if available
    if mlx_available and "mlx" in providers_list:
        # Use the Josiefied-Qwen3-8B model from MLX community
        providers_to_try.append(("mlx", {
            ModelParameter.MODEL: "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-6bit"
        }))
    
    # Add fallback providers
    providers_to_try.extend([
        ("ollama", {"model": "qwen2.5"}),
        # Add more fallbacks if needed
    ])
    
    # Try each provider in order
    for provider_name, config in providers_to_try:
        try:
            print(f"Attempting to create provider: {provider_name}")
            provider = create_llm(provider_name, **config)
            print(f"Successfully created {provider_name} provider")
            return provider
        except Exception as e:
            print(f"Failed to create {provider_name} provider: {e}")
    
    # If all providers failed, exit
    print("ERROR: No working providers found")
    sys.exit(1)

def main():
    # Create a provider
    provider = create_provider()
    
    # Create session with the provider and tool function
    try:
        session = Session(
            system_prompt="You are a helpful assistant that can read files when needed. "
                        "If you need to see a file's contents, use the read_file tool.",
            provider=provider,
            tools=[read_file]  # Function is automatically registered
        )
        
        print(f"\nMinimal ALMA with {provider.__class__.__name__} - Type 'exit' to quit")
        print("Example: 'Read the file README.md and summarize it'")
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