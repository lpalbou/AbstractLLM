#!/usr/bin/env python3
"""
Minimal ALMA (AbstractLLM Agent) implementation with file reading capability.
Uses the simplest approach to tool calling with an interactive REPL.

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
from abstractllm.utils.logging import configure_logging, log_step
from abstractllm.utils.formatting import (
    format_response_display, format_stats_display, format_last_interactions,
    format_system_prompt_info, format_update_result, format_tools_list,
    format_provider_switch_result, format_provider_info
)
from abstractllm.tools.common_tools import (
    read_file, list_files,
    get_system_info, get_performance_stats, get_running_processes,
    get_network_connections, get_disk_partitions, monitor_resource_usage
)
import os
import logging
import re
import sys


# ANSI color codes for error messages
RED_BOLD = '\033[1m\033[31m'     # Red bold
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
GREY_ITALIC = '\033[3m\033[90m'  # Grey italic
RESET = '\033[0m'                # Reset formatting


def start_session():
    #Store current provider info for /model command
    provider_name = "mlx" # or anthropic or openai or ollama
    model_name="mlx-community/Qwen3-30B-A3B-4bit"
    #model_name="mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-4Bit"
    #model_name="mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit"
    #model_name="a-m-team/AM-Thinking-v1"
    #model_name="claude-3-5-haiku-20241022"
    #model_name="gpt-4o"
    #model_name = "cogito"
    #model_name = "qwen2.5"

    provider = create_llm(provider_name, 
                         model=model_name,
                         max_tokens=4096)

    # Load the model immediately at startup
    print("🔄 Loading model...")
    provider.load_model()
    print("✅ Model loaded and ready!")

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
    return session
    
def start_repl(session):
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
                
                if command == "/model":
                    # Handle model switching
                    if len(command_parts) == 1:
                        # Show current model info from session
                        print(f"\n{BLUE_ITALIC}🤖 Model Information{RESET}")
                        provider_info = session.get_provider_info()
                        print(format_provider_info(provider_info))
                        continue
                    else:
                        # Switch to new model - parse provider:model format
                        new_model_spec = " ".join(command_parts[1:])
                        
                        if ":" not in new_model_spec:
                            print(f"{RED_BOLD}Error: Format should be 'provider:model'{RESET}")
                            continue
                        
                        provider_name, model_name = new_model_spec.split(":", 1)
                        
                        # Call session method and format result
                        result = session.switch_provider(provider_name.strip(), model_name.strip())
                        print(format_provider_switch_result(result))
                        continue
                
                elif command == "/stats":
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
                        # Load session - provider will be restored automatically from saved state
                        new_session = Session.load(filename)
                        
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
                
                elif command == "/last":
                    # Handle last command - show last X interactions
                    # Default to 1 if no parameter provided
                    count = abs(int(command_parts[1])) if len(command_parts) > 1 else 1
                    # Get structured data from session
                    interactions = session.get_last_interactions(count)
                    # Format and display
                    print(format_last_interactions(interactions))
                    continue
                
                elif command == "/help":
                    # Show help information
                    show_help()                    
                    continue
                
                elif command == "/tools":
                    # List all available tools with descriptions
                    tools = session.get_tools_list()
                    print(format_tools_list(tools))
                    continue
                
                elif command == "/system":
                    # Handle system command
                    if len(command_parts) == 1:
                        # Show current system prompt info
                        print(f"\n{BLUE_ITALIC}🤖 System Prompt Information{RESET}")
                        prompt_info = session.get_system_prompt_info()
                        print(format_system_prompt_info(prompt_info))
                        continue
                    else:
                        # Update system prompt
                        new_prompt = " ".join(command_parts[1:])
                        result = session.update_system_prompt(new_prompt)
                        print(format_update_result(result))
                        continue
                
                else:
                    print(f"{RED_BOLD}Unknown command: {command}{RESET}")
                    print("Available commands: /stats, /save <filename>, /load <filename>, /model [provider:model], /system [prompt], /last [count], /tools, /help, /exit")
                    continue
            
            # Generate response with tool support
            print(f"\nAssistant:")
            
            # Log the interaction steps for debugging
            log_step(1, "USER→AGENT", f"Received query: {user_input}")
            
            # Use the unified generate method - trust AbstractLLM to handle everything
            log_step(2, "AGENT→LLM", "Sending query to LLM with tool support enabled")
            response = session.generate(
                prompt=user_input,
                max_tool_calls=25  # Limit tool calls to avoid infinite loops
            )
            
            log_step(3, "LLM→AGENT", "Received response, displaying to user")
            
            # Simply display the response - trust AbstractLLM formatting
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



def set_logging():
    configure_logging(
        log_dir="logs", 
        console_level=logging.WARNING,  # Only warnings/errors to console
        file_level=logging.DEBUG        # Everything to file
    )
    print("📝 Logging enabled - warnings to console, detailed logs to logs/ directory")

def show_help():
    print(f"\n{BLUE_ITALIC}💡 ALMA Session Management Help{RESET}")
    print(f"\nAvailable Commands:")
    print(f"  /stats                  - Show session statistics")
    print(f"  /save <filename>        - Save current session to file")
    print(f"  /load <filename>        - Load session from file")
    print(f"  /model [provider:model] - Show current model or switch to new model")
    print(f"  /system [prompt]        - Show current system prompt or set new one")
    print(f"  /last [count]           - Show last X interactions (default: 1)")
    print(f"  /tools                  - List all available tools")
    print(f"  /help                   - Show this help message")
    print(f"  /exit, /quit, /q        - Exit ALMA")

    print(f"\nSession Features:")
    print(f"  • Conversation history is automatically saved")
    print(f"  • Tool calls and results are tracked")
    print(f"  • Sessions persist system prompts and metadata")
    print(f"  • JSON format allows easy inspection and sharing")

    print(f"\nModel Switching:")
    print(f"  • Format: provider:model_name")
    print(f"  • Supported providers: ollama, mlx, anthropic, openai")
    print(f"  • Previous model is unloaded from memory")
    print(f"  • Conversation history and tools are preserved")
    print(f"  • Examples:")
    print(f"    - ollama:qwen3:30b-a3b-q4_K_M")
    print(f"    - mlx:mlx-community/Qwen3-30B-A3B-4bit")
    print(f"    - anthropic:claude-3-5-haiku-20241022")
    print(f"    - openai:gpt-4o")
    
    print(f"\nTool Support:")
    print(f"  Available tools: read_file, list_files, system monitoring tools")
    print(f"  Example: 'Read the README.md file and summarize it'")    


def main():    
    # Set logging
    set_logging()

    # Show help
    show_help()

    # Create session
    session = start_session()

    # Start REPL loop
    start_repl(session)


if __name__ == "__main__":
    main() 