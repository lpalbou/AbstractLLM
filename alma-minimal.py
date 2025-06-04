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
            print(f"{RED_BOLD}   Result: SUCCESS ({len(str(result.get('output', '')))} chars)")
        
        return result

def read_file(file_path: str, should_read_entire_file: bool = True, start_line_one_indexed: int = 1, end_line_one_indexed_inclusive: int = None) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        should_read_entire_file: Whether to read the entire file (default: True)
        start_line_one_indexed: Starting line number (1-indexed, default: 1)
        end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive, default: None for end of file)
    
    Returns:
        File contents or error message
    """
    try:
        with open(file_path, 'r') as f:
            if should_read_entire_file:
                return f.read()
            else:
                # Read specific line range
                lines = f.readlines()
                
                # Convert to 0-indexed for Python
                start_idx = max(0, start_line_one_indexed - 1)
                
                if end_line_one_indexed_inclusive is None:
                    end_idx = len(lines)
                else:
                    end_idx = min(len(lines), end_line_one_indexed_inclusive)
                
                # Extract the requested lines
                selected_lines = lines[start_idx:end_idx]
                
                # Add line numbers for clarity
                result_lines = []
                for i, line in enumerate(selected_lines, start=start_line_one_indexed):
                    result_lines.append(f"{i:4d}: {line.rstrip()}")
                
                return "\n".join(result_lines)
                
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
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
    provider = create_llm("mlx", 
                         model="mlx-community/Qwen3-30B-A3B-4bit",
                         #model="mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
                         max_tokens=4096)  # Set default max_tokens


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
        system_prompt="You are a helpful assistant that can execute tools (such as read_file) to gain further information and help resolve the user prompt. When you execute a tool, you gain additional insights that help you better understand the context and decide what to do next before answering the user prompt.",
        provider=provider,
        tools=[read_file]  # Function is automatically registered
    )
    
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
            
            # Generate response with tool support
            print(f"\nAssistant:")
            
            # Use the unified generate method - trust AbstractLLM to handle everything
            response = session.generate(
                prompt=user_input,
                max_tool_calls=25  # Limit tool calls to avoid infinite loops
            )
            
            # Simply display the response - trust AbstractLLM formatting
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