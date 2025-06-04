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
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

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
    
    model_name = "cogito"
    model_name = "qwen2.5"
    provider = create_llm("ollama", model=model_name)

    # TEST WITH ANTHROPIC
    # provider = create_llm("anthropic", 
    #                    model="claude-3-5-haiku-20241022")

    # TEST WITH OPENAI
    # provider = create_llm("openai", 
    #                     model="gpt-4o")

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
            
            # Add debug output
            print("\n[DEBUG] Generating response...")
            
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
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 