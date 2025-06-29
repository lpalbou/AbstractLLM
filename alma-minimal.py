#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Minimal - A simple command-line interface for AbstractLLM
This script provides a minimal implementation of the ALMA agent using AbstractLLM.
It supports text generation and tool usage in a simple REPL interface.
"""

from abstractllm import create_llm, create_session
from abstractllm.utils.logging import configure_logging, log_step
from abstractllm.utils.formatting import (
    format_response_display, format_stats_display, format_last_interactions,
    format_system_prompt_info, format_update_result, format_tools_list,
    format_provider_switch_result, format_provider_info
)
from abstractllm.tools.common_tools import (
    read_file, list_files, search_files
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
            tools=[read_file, list_files, search_files],  # Directly pass the tool functions
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
                        from abstractllm.session import Session
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
                    show_help()
                    continue
            
            # Generate response with tool support
            # Log the interaction steps for debugging
            log_step(1, "USER→AGENT", f"Received query: {user_input}")
            
            # Use the unified generate method with tools parameter
            log_step(2, "AGENT→LLM", "Sending query to LLM with tool support enabled")
            response = session.generate(
                prompt=user_input,
                tools=[read_file, list_files, search_files],  # Directly pass the tool functions
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
    print(f"  /stats                            - Show session statistics")
    print(f"  /save <filename>                  - Save current session to file")
    print(f"  /load <filename>                  - Load session from file")
    print(f"  /model <model_name:optional>      - Show current model or switch to new model")
    print(f"  /system <system_prompt:optional>  - Show current system prompt or set new one (optional)")
    print(f"  /last <count:optional>            - Show last X interactions (default: 1)")
    print(f"  /tools                            - List all available tools")
    print(f"  /help                             - Show this help message")
    print(f"  /exit, /quit, /q                  - Exit ALMA")
    print(f"\n{BLUE_ITALIC}📎 File Attachment Syntax (AbstractLLM Core Feature):{RESET}")
    print(f"  @file.txt                         - Attach single file temporarily")
    print(f"  @folder/                          - Attach all files in folder")
    print(f"  @*.py                             - Attach files matching pattern")
    print(f"  Example: 'Analyze @data.csv and @config.json for errors'")
    print(f"  Note: Files are attached temporarily and NOT saved to conversation history")


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
        default="ollama",
        help="LLM provider to use (default: mlx). Options: mlx, anthropic, openai, ollama"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:4b",
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

    print(f"{BLUE_ITALIC}🚀 Starting ALMA with {args.provider}:{args.model}{RESET}")

    session = create_session(args.provider, 
                         model=args.model,
                         max_tokens=args.max_tokens,
                         tools=[read_file, list_files, search_files])

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