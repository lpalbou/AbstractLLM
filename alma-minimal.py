#!/usr/bin/env python3
"""
Minimal ALMA (AbstractLLM Agent) implementation with file reading capability.
Uses the simplest approach to tool calling with an interactive REPL.

# Requirements
- AbstractLLM: pip install abstractllm[anthropic]
- Anthropic API key: export ANTHROPIC_API_KEY=your_api_key_here

# Adding More Tools
To add more tools from the common_tools module, simply import and add them:

from abstractllm.tools.common_tools import (
    list_files, search_files, write_file, update_file,
    execute_command, search_internet, fetch_url, 
    fetch_and_parse_html, ask_user_multiple_choice
)

Then add them to the tools list:
    session = Session(
        system_prompt="...",
        provider=provider,
        tools=[read_file, list_files, search_files, write_file, ...]  # Add more tools here
    )
"""

from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.utils.logging import configure_logging
from abstractllm.utils.formatting import format_response_display, format_stats_display
from abstractllm.tools.common_tools import read_file, list_files
import os
import logging
import re

# Configure AbstractLLM logging for console output with tool execution details
configure_logging(
    log_level=logging.INFO,
    console_output=True,  # Force console output
)

# ANSI color codes for error messages
RED_BOLD = '\033[1m\033[31m'    # Red bold
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
RESET = '\033[0m'               # Reset formatting

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
    # Use the standard Session class - no need to override it
    session = Session(
        system_prompt="""You are a capable agent that EXECUTES tools and follows instructions directly. When a user asks you to follow instructions from a document, you should READ the document and then EXECUTE the steps as written, not just summarize them.

CRITICAL: When you need to read a file, you MUST actually call the read_file tool. Do NOT show code examples or pseudo-code. EXECUTE the tool call immediately.

For example, if you need to read a file, you should:
- Actually call read_file(file_path="path/to/file", should_read_entire_file=True)
- NOT show: ```python read_file(...)``` 
- NOT explain what you would do
- JUST DO IT

You have access to these tools:
- read_file(file_path, should_read_entire_file=True, start_line_one_indexed=1, end_line_one_indexed_inclusive=None)
- list_files(directory_path=".", pattern="*", recursive=False)

When following multi-step procedures:
1. Read the instructions first by CALLING read_file
2. Execute each step that requires a tool call by CALLING the tools
3. Continue to the next step based on the results
4. Complete the entire procedure unless instructed otherwise

You are an ACTION-TAKING agent, not just an advisor. Take action immediately when requested.""",
        provider=provider,
        tools=[read_file, list_files]  # Functions are automatically registered
    )
    
    print("\nMinimal ALMA - Type '/exit', '/quit', or '/q' to quit")
    print("Example: 'Read the file README.md and summarize it'")
    print("Commands: /stats, /save <filename>, /load <filename>")
    print("Type '/help' for more information")
    print(f"\n💡 Tip: See the common_tools module for more shareable tools:")
    print(f"   - File operations: list_files, search_files, write_file, update_file")
    print(f"   - Web operations: search_internet, fetch_url, fetch_and_parse_html")
    print(f"   - System: execute_command")
    print(f"   - User interaction: ask_user_multiple_choice")
    
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
            
            # Handle special commands
            if user_input.startswith("/"):
                command_parts = user_input.split()
                command = command_parts[0].lower()
                
                if command == "/stats":
                    # Show session statistics
                    try:
                        stats = session.get_stats()
                        print(f"\n{format_stats_display(stats)}")
                    except Exception as e:
                        print(f"{RED_BOLD}Error getting stats: {str(e)}{RESET}")
                    continue
                
                elif command == "/save":
                    # Save session to file
                    if len(command_parts) < 2:
                        print(f"{RED_BOLD}Usage: /save <filename>{RESET}")
                        continue
                    
                    filename = command_parts[1]
                    if not filename.endswith('.json'):
                        filename += '.json'
                    
                    try:
                        session.save(filename)
                        print(f"Session saved to {filename}")
                    except Exception as e:
                        print(f"{RED_BOLD}Error saving session: {str(e)}{RESET}")
                    continue
                
                elif command == "/load":
                    # Load session from file
                    if len(command_parts) < 2:
                        print(f"{RED_BOLD}Usage: /load <filename>{RESET}")
                        continue
                    
                    filename = command_parts[1]
                    if not filename.endswith('.json'):
                        filename += '.json'
                    
                    try:
                        # Load session and replace current session
                        new_session = Session.load(filename, provider=provider)
                        
                        # Transfer tools from current session to loaded session
                        new_session.tools = session.tools
                        new_session._tool_implementations = session._tool_implementations
                        
                        session = new_session
                        print(f"Session loaded from {filename}")
                        
                        # Show basic info about loaded session
                        stats = session.get_stats()
                        msg_count = stats["message_stats"]["total_messages"]
                        duration = stats["session_info"]["duration_hours"]
                        print(f"{BLUE_ITALIC}Loaded session: {msg_count} messages, {duration:.2f} hours{RESET}")
                        
                    except FileNotFoundError:
                        print(f"{RED_BOLD}Error: File '{filename}' not found{RESET}")
                    except Exception as e:
                        print(f"{RED_BOLD}Error loading session: {str(e)}{RESET}")
                    continue
                
                elif command == "/help":
                    # Show help information
                    print(f"\n{BLUE_ITALIC}💡 ALMA Session Management Help{RESET}")
                    print(f"\nAvailable Commands:")
                    print(f"  /stats                  - Show session statistics")
                    print(f"  /save <filename>        - Save current session to file")
                    print(f"  /load <filename>        - Load session from file")
                    print(f"  /help                   - Show this help message")
                    print(f"  /exit, /quit, /q        - Exit ALMA")
                    
                    print(f"\nSession Features:")
                    print(f"  • Conversation history is automatically saved")
                    print(f"  • Tool calls and results are tracked")
                    print(f"  • Sessions persist system prompts and metadata")
                    print(f"  • JSON format allows easy inspection and sharing")
                    
                    print(f"\nExample Usage:")
                    print(f"  /save my_conversation     - Saves to 'my_conversation.json'")
                    print(f"  /load my_conversation     - Loads from 'my_conversation.json'")
                    print(f"  /stats                    - Shows message counts, tool usage, etc.")
                    
                    print(f"\nTool Support:")
                    print(f"  Available tools: read_file, list_files")
                    print(f"  Example: 'Read the README.md file and summarize it'")
                    
                    print(f"\n📦 Available Common Tools:")
                    print(f"  See abstractllm.tools.common_tools for more shareable tools")
                    print(f"  Run: python examples/common_tools_demo.py")
                    continue
                
                else:
                    print(f"{RED_BOLD}Unknown command: {command}{RESET}")
                    print("Available commands: /stats, /save <filename>, /load <filename>, /help, /exit")
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