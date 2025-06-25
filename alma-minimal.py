# -*- coding: utf-8 -*-
"""
ALMA Minimal - A simple command-line interface for AbstractLLM

This script provides a minimal implementation of the ALMA agent using AbstractLLM.
It supports text generation and tool usage in a simple REPL interface.

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

# Command Line Usage
Examples:
    python alma-minimal.py --provider mlx --model "mlx-community/Qwen3-30B-A3B-4bit"
    python alma-minimal.py --provider anthropic --model "claude-3-5-haiku-20241022"
    python alma-minimal.py --prompt "List all Python files in the current directory"
    python alma-minimal.py --provider mlx --model "mlx-community/Qwen3-30B-A3B-4bit" --prompt "Read the README.md file"
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
    read_file, list_files, search_files,
    get_system_info, get_performance_stats, get_running_processes,
    get_network_connections, get_disk_partitions, monitor_resource_usage
)
import os
import logging
import re
import sys
import argparse


# ANSI color codes for error messages
RED_BOLD = '\033[1m\033[31m'     # Red bold
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
GREY_ITALIC = '\033[3m\033[90m'  # Grey italic
RESET = '\033[0m'                # Reset formatting


# Define available tools globally for access in all functions
AVAILABLE_TOOLS = [
    read_file,
    list_files,
    search_files,
    get_system_info,
    get_performance_stats
]

def start_session(provider_name, model_name, max_tokens = 4096):
    provider = create_llm(provider_name, 
                         model=model_name,
                         max_tokens=max_tokens)

    print(f"Connected to {provider_name} provider with model {model_name}")

    # Create session with the provider and tool functions
    # Use the standard Session class - no need to override it
    
    session = Session(
        system_prompt="""You are a capable agent that EXECUTES tools and follows instructions directly. When a user asks you to follow instructions from a document, you should READ the document and then EXECUTE the steps as written, not just summarize them.

CRITICAL: When you need to perform an action, you MUST actually call the appropriate tool. Do NOT show code examples or pseudo-code. EXECUTE the tool call immediately.

When following multi-step procedures:
1. Read the instructions first if needed
2. Execute each step that requires a tool call by CALLING the tools
3. Continue to the next step based on the results
4. Complete the entire procedure unless instructed otherwise

You are an ACTION-TAKING agent, not just an advisor. Take action immediately when requested.""",
        provider=provider,
        tools=AVAILABLE_TOOLS  # Register all available tools
    )
    return session


def execute_single_prompt(session, prompt: str):
    """Execute a single prompt and display the result without starting REPL."""
    try:
        print(f"\n{BLUE_ITALIC}Executing prompt:{RESET} {prompt}")
        print(f"\n{BLUE_ITALIC}Assistant:{RESET}")
        
        # Log the interaction steps for debugging
        log_step(1, "USER→AGENT", f"Received query: {prompt}")
        
        # Use the unified generate method with tools parameter
        log_step(2, "AGENT→LLM", "Sending query to LLM with tool support enabled")
        response = session.generate(
            prompt=prompt,
            tools=AVAILABLE_TOOLS,  # Directly pass the tool functions
            max_tool_calls=25  # Limit tool calls to avoid infinite loops
        )
        
        log_step(3, "LLM→AGENT", "Received response, displaying to user")
        
        # Simply display the response - trust AbstractLLM formatting
        format_response_display(response)
        
    except Exception as e:
        print(f"\n{RED_BOLD}Error executing prompt: {str(e)}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
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
            # Log the interaction steps for debugging
            log_step(1, "USER→AGENT", f"Received query: {user_input}")
            
            # Use the unified generate method with tools parameter
            log_step(2, "AGENT→LLM", "Sending query to LLM with tool support enabled")
            response = session.generate(
                prompt=user_input,
                tools=AVAILABLE_TOOLS,  # Directly pass the tool functions
                max_tool_calls=25  # Limit tool calls to avoid infinite loops
            )
            
            log_step(3, "LLM→AGENT", "Received response, displaying to user")
            
            # Show "Assistant:" only when we have the response
            print(f"\nAssistant:")
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ALMA (AbstractLLM Agent) - Minimal implementation with tool support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
    Launch with default settings (mlx:mlx-community/Qwen3-30B-A3B-4bit)
  
  %(prog)s --provider anthropic --model claude-3-5-haiku-20241022
    Launch with Anthropic Claude model
  
  %(prog)s --provider openai --model gpt-4o
    Launch with OpenAI GPT-4o model
  
  %(prog)s --prompt "List all Python files in the current directory"
    Execute a single prompt and exit
  
  %(prog)s --provider mlx --model "mlx-community/Qwen3-30B-A3B-4bit" --prompt "Read the README.md file"
    Execute a prompt with specific provider and model

Supported providers: mlx, anthropic, openai, ollama
        """
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="mlx",
        help="LLM provider to use (default: mlx). Options: mlx, anthropic, openai, ollama"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-30B-A3B-4bit",        
        help="Model name to use (default: mlx-community/Qwen3-30B-A3B-4bit)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Execute a single prompt and exit (non-interactive mode)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for generation (default: 4096)"
    )
    
    return parser.parse_args()


def main():    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging
    set_logging()

    # Use command line arguments or defaults
    provider_name = args.provider
    model_name = args.model
    max_tokens = args.max_tokens

    print(f"{BLUE_ITALIC}🚀 Starting ALMA with {provider_name}:{model_name}{RESET}")

    # Create session
    session = start_session(provider_name, model_name, max_tokens)

    # If prompt is provided, execute it and exit
    if args.prompt:
        execute_single_prompt(session, args.prompt)
        return

    # Show help for interactive mode
    show_help()

    # Start REPL loop
    start_repl(session)


if __name__ == "__main__":
    main() 