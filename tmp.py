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
import re

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# ANSI color codes
RED_BOLD = '\033[1m\033[31m'    # Red bold
GREY_ITALIC = '\033[3m\033[90m'  # Grey italic
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
RESET = '\033[0m'               # Reset formatting

class VerboseSession(Session):
    """Session class with enhanced tool execution logging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_call_count = 0
    
    def execute_tool_call(self, tool_call, tool_functions):
        """Override to add verbose tool execution logging."""
        self.tool_call_count += 1
        
        # Display tool execution in red bold
        print(f"\n{RED_BOLD}🔧 TOOL EXECUTION #{self.tool_call_count}{RESET}")
        print(f"{RED_BOLD}   Tool: {tool_call.name}{RESET}")
        print(f"{RED_BOLD}   Args: {tool_call.arguments}{RESET}")
        
        # Execute the tool using parent method
        result = super().execute_tool_call(tool_call, tool_functions)
        
        # Show result summary
        if result.get('error'):
            print(f"{RED_BOLD}   Result: ERROR - {result['error']}{RESET}")
        else:
            output_preview = str(result.get('output', ''))[:100]
            if len(output_preview) >= 100:
                output_preview += "..."
            print(f"{RED_BOLD}   Result: SUCCESS ({len(str(result.get('output', '')))} chars) - {output_preview}{RESET}")
        
        return result

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def parse_response_content(content: str):
    """
    Parse response content to extract think tags and clean content.
    
    Returns:
        tuple: (think_content, clean_content)
    """
    think_content = ""
    clean_content = content
    
    # Extract <think> content using regex
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, content, re.DOTALL)
    
    if think_matches:
        think_content = think_matches[0].strip()
        # Remove all <think>...</think> blocks from the main content
        clean_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
    
    return think_content, clean_content

def format_response_display(response):
    """
    Format and display a response with proper styling.
    
    Args:
        response: GenerateResponse object or string
    """
    if hasattr(response, 'content'):
        content = response.content
        
        # Parse the content
        think_content, clean_content = parse_response_content(content)
        
        # Display think content in grey italic if present
        if think_content:
            print(f"\n{GREY_ITALIC}<think>")
            print(think_content)
            print(f"</think>{RESET}\n")
        
        # Display clean content
        if clean_content:
            print(clean_content)
        
        # Display metadata in blue italic if available
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            model_name = getattr(response, 'model', 'unknown')
            
            metadata_parts = []
            if 'completion_tokens' in usage:
                metadata_parts.append(f"{usage['completion_tokens']} tokens")
            if 'time' in usage:
                metadata_parts.append(f"{usage['time']:.2f}s")
            metadata_parts.append(f"model: {model_name}")
            
            if metadata_parts:
                print(f"\n{BLUE_ITALIC}[{' | '.join(metadata_parts)}]{RESET}")
    else:
        # Handle string responses
        think_content, clean_content = parse_response_content(str(response))
        
        if think_content:
            print(f"\n{GREY_ITALIC}<think>")
            print(think_content)
            print(f"</think>{RESET}\n")
        
        if clean_content:
            print(clean_content)

def main():    
    # Initialize the provider with the model - this is the key step
    # The Session will use this provider's model by default
    
    model_name = "cogito"
    model_name = "qwen2.5"
    #provider = create_llm("ollama", model=model_name)
    provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")


    # TEST WITH ANTHROPIC
    # provider = create_llm("anthropic", 
    #                    model="claude-3-5-haiku-20241022")

    # TEST WITH OPENAI
    # provider = create_llm("openai", 
    #                     model="gpt-4o")

    # TEST WITH MLX (Apple Silicon only)
    # provider = create_llm("mlx", 
    #                     model="mlx-community/Qwen3-30B-A3B-4bit")

    # Create session with the provider and tool function
    # Use our enhanced VerboseSession for better tool logging
    session = VerboseSession(
        system_prompt="You are a helpful assistant that can read files when needed. "
                     "If you need to see a file's contents, use the read_file tool.",
        provider=provider,
        tools=[read_file]  # Function is automatically registered
    )
    
    # Check if running in interactive mode
    import sys
    is_interactive = sys.stdin.isatty()
    
    if not is_interactive:
        # Non-interactive mode - run a simple test
        print("Running in non-interactive mode - testing basic functionality...")
        test_prompt = "What is 2+2? Explain your reasoning."
        print(f"\nTest prompt: {test_prompt}")
        print("\nAssistant: ", end="")
        
        try:
            response = session.generate(
                prompt=test_prompt,
                max_tokens=1024
            )
            
            format_response_display(response)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    print("\nMinimal ALMA - Type '/exit', '/quit', or '/q' to quit")
    print("Example: 'Read the file README.md and summarize it'")
    
    # Simple REPL loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ["/exit", "/quit", "/q", "exit", "quit", "bye"]:
                print("Goodbye!")
                break
                
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Reset tool call counter for each new request
            session.tool_call_count = 0
            
            # Generate response with tool support
            print(f"\nAssistant:")
            
            # Use the unified generate method
            response = session.generate(
                prompt=user_input,
                max_tool_calls=3,  # Limit tool calls to avoid infinite loops
                max_tokens=2048    # Ensure enough tokens for complete response
            )
            
            # Handle different response types:
            # - If response has .content attribute (tool was used), use that
            # - If response is a string (direct answer, no tool used), use as is
            if hasattr(response, 'content'):
                # Check if we're dealing with a tool call request that hasn't been resolved
                if hasattr(response, 'has_tool_calls') and response.has_tool_calls():
                    print(f"\n{RED_BOLD}⚠️  TOOL LOOP DETECTED - Model still requesting tools after max_tool_calls reached{RESET}")
                    
                    # If we're still getting tool calls after max_tool_calls, 
                    # the model is stuck in a loop. Force a direct question instead.
                    # First, get the content from the last tool execution
                    tool_content = None
                    print(f"Looking through {len(session.messages)} messages for tool results...")
                    
                    for i, message in enumerate(session.messages):
                        if hasattr(message, 'tool_results') and message.tool_results:
                            for j, result in enumerate(message.tool_results):
                                if 'output' in result:
                                    tool_content = result['output']
                                    print(f"Found tool content from message {i} (length: {len(tool_content)})")
                                    break
                        if tool_content:
                            break
                    
                    if tool_content:
                        # For a summarization task, we ask the model directly with the content
                        direct_prompt = f"Here is the content of the file that was read. Please provide a concise summary:\n\n{tool_content}"
                        
                        # Generate response without tool support (direct query)
                        print(f"\n{RED_BOLD}🔄 Forcing direct response with file content...{RESET}")
                        direct_response = provider.generate(
                            prompt=direct_prompt,
                            system_prompt="You are a helpful assistant summarizing file contents.",
                            max_tokens=2048
                        )
                        
                        print()  # Add spacing
                        format_response_display(direct_response)
                    else:
                        print("Unable to get content from tool execution. Please try again.")
                else:
                    # Normal content response
                    print()  # Add spacing
                    format_response_display(response)
            else:
                # Direct string response
                print()  # Add spacing  
                format_response_display(response)
            
        except EOFError:
            # Handle EOF gracefully (Ctrl+D or redirected input)
            print("\nReceived EOF (Ctrl+D). Goodbye!")
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nReceived interrupt (Ctrl+C). Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 