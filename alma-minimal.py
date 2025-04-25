#!/usr/bin/env python3
"""
Minimal ALMA (AbstractLLM Agent) implementation with file reading capability.
Uses the simplest approach to tool calling with an interactive REPL.

# Requirements
- AbstractLLM: pip install abstractllm[anthropic]
- Anthropic API key: export ANTHROPIC_API_KEY=your_api_key_here
"""

from abstractllm import create_llm
from abstractllm.session import Session
import os

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def main():    
    # Initialize the provider with the model - this is the key step
    # The Session will use this provider's model by default
    provider = create_llm("anthropic", 
                        model="claude-3-5-haiku-20241022")
    
    # Create session with the provider and tool function
    # The tool is automatically registered for both definition and execution
    session = Session(
        system_prompt="You are a helpful assistant that can read files when needed. "
                     "If you need to see a file's contents, use the read_file tool.",
        provider=provider,
        tools=[read_file]  # Function is automatically registered
    )
    
    print("\nMinimal ALMA - Type 'exit' to quit")
    print("Example: 'Read the file README.md and summarize it'")
    
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
            # The session will use the provider's model and registered tools
            print("\nAssistant: ", end="")
            response = session.generate_with_tools(
                prompt=user_input
            )
            print(response.content)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 