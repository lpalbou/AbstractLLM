#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Minimal - A simple command-line interface for AbstractLLM
This script provides a minimal implementation of the ALMA agent using AbstractLLM.
It supports text generation and tool usage in a simple REPL interface.
"""

# Always use enhanced features - no backward compatibility needed
from abstractllm.factory import create_session
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
from abstractllm.tools.common_tools import read_file, list_files, search_files
from abstractllm.utils.logging import configure_logging, log_step
from abstractllm.utils.formatting import (
    format_response_display, format_stats_display, format_last_interactions,
    format_system_prompt_info, format_update_result, format_tools_list,
    format_provider_switch_result, format_provider_info
)
import os
import logging
import re
import sys
import argparse
import time


# ANSI color codes for error messages
RED_BOLD = '\033[1m\033[31m'     # Red bold
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
GREY_ITALIC = '\033[3m\033[90m'  # Grey italic
RESET = '\033[0m'                # Reset formatting
GREEN_BOLD = '\033[1m\033[32m'   # Green bold



def execute_single_prompt(session, prompt: str, stream: bool = False, args=None):
    """Execute a single prompt with full enhanced features."""
    try:
        print(f"\n{BLUE_ITALIC}Executing prompt:{RESET} {prompt}")
        print(f"\n{BLUE_ITALIC}Assistant:{RESET}")
        
        # Log the interaction steps for debugging
        log_step(1, "USER→AGENT", f"Received query: {prompt}")
        
        # Use the unified generate method with tools parameter
        log_step(2, "AGENT→LLM", "Sending query to LLM with tool support enabled")
        # Prepare structured config if requested
        structured_config = None
        if args and args.structured_output:
            structured_config = StructuredResponseConfig(
                format=ResponseFormat.JSON if args.structured_output == 'json' else ResponseFormat.YAML,
                force_valid_json=True,
                max_retries=3,
                temperature_override=0.0
            )
        
        # Always use enhanced features
        response = session.generate(
            prompt=prompt,
            tools=[read_file, list_files, search_files],
            max_tool_calls=25,
            stream=stream,
            use_memory_context=True,  # Always use memory context
            create_react_cycle=True,   # Always create ReAct cycle
            structured_config=structured_config
        )
        
        log_step(3, "LLM→AGENT", "Received response, displaying to user")
        
        # Handle streaming vs non-streaming responses
        if stream:
            # For streaming, iterate through the generator and display progressively
            accumulated_text = ""
            start_time = time.time()
            
            try:
                for chunk in response:
                    # Session streaming yields strings directly, not GenerateResponse objects
                    if isinstance(chunk, str):
                        accumulated_text += chunk
                        # Print just the new token
                        print(chunk, end='', flush=True)
                    elif hasattr(chunk, 'content'):
                        # Fallback for GenerateResponse objects (direct provider streaming)
                        token_content = chunk.content
                        if token_content:  # Only add non-empty content
                            accumulated_text += token_content
                            # Print just the new token
                            print(token_content, end='', flush=True)
                    
                # Print a newline at the end
                print()
                
                # Calculate detailed stats for streaming
                if accumulated_text:
                    end_time = time.time()
                    time_taken = end_time - start_time
                    
                    # Get session info for context
                    try:
                        # Get the last interaction to extract context info
                        interactions = session.get_last_interactions(1)
                        
                        # Calculate tokens
                        completion_tokens = len(accumulated_text.split())
                        
                        # Estimate context tokens from the prompt length
                        # This is an approximation since we don't have exact tokenization
                        prompt_tokens = len(prompt.split()) + 10  # Add some for system prompt and formatting
                        
                        # Get model info - try multiple approaches
                        model_name = args.model if args else "unknown"  # Use the original model argument as fallback
                        try:
                            provider_info = session.get_provider_info()
                            if provider_info.get('model'):
                                model_name = provider_info['model']
                        except:
                            pass
                        
                        # Build metadata parts
                        metadata_parts = []
                        
                        # Format context tokens with space separator for thousands
                        formatted_context = f"{prompt_tokens:,}".replace(',', ' ')
                        metadata_parts.append(f"context : {formatted_context} tk")
                        
                        # Generated tokens
                        metadata_parts.append(f"generated : {completion_tokens} tk")
                        
                        # Time
                        metadata_parts.append(f"time : {time_taken:.1f}s")
                        
                        # Speed (tokens per second)
                        if completion_tokens > 0 and time_taken > 0:
                            speed = completion_tokens / time_taken
                            metadata_parts.append(f"speed : {speed:.1f} tk/s")
                        
                        # Model name
                        metadata_parts.append(f"model: {model_name}")
                        
                        if metadata_parts:
                            print(f"\n{BLUE_ITALIC}[{' | '.join(metadata_parts)}]{RESET}")
                        
                    except Exception as e:
                        # Fallback stats if metadata extraction fails
                        final_tokens = len(accumulated_text.split())
                        speed = final_tokens / time_taken if time_taken > 0 else 0
                        model_name = args.model if args else "unknown"
                        print(f"\n{BLUE_ITALIC}[generated : {final_tokens} tk | time : {time_taken:.1f}s | speed : {speed:.1f} tk/s | model: {model_name}]{RESET}")
                        
            except Exception as e:
                print(f"\n{RED_BOLD}Error during streaming: {str(e)}{RESET}")
                raise
        else:
            # For non-streaming, use the existing formatting
            format_response_display(response)
        
    except Exception as e:
        print(f"\n{RED_BOLD}Error executing prompt: {str(e)}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
def start_repl(session, stream: bool = False, args=None):
    """Start the REPL (Read-Eval-Print Loop) for interactive conversation."""
    print(f"\n{BLUE_ITALIC}💬 Starting interactive session. Type '/help' for commands or '/exit' to quit.{RESET}")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                
                if command in ['exit', 'quit', 'q']:
                    print(f"{BLUE_ITALIC}Goodbye!{RESET}")
                    break
                elif command == 'help':
                    show_help()
                    continue
                elif command == 'stats':
                    stats = session.get_stats()
                    print(format_stats_display(stats))
                    continue
                elif command.startswith('save '):
                    filename = command[5:].strip()
                    if filename:
                        try:
                            session.save(filename)
                            print(f"{GREEN_BOLD}Session saved to {filename}{RESET}")
                        except Exception as e:
                            print(f"{RED_BOLD}Error saving session: {str(e)}{RESET}")
                    else:
                        print(f"{RED_BOLD}Please provide a filename: /save <filename>{RESET}")
                    continue
                elif command.startswith('load '):
                    filename = command[5:].strip()
                    if filename:
                        try:
                            session = Session.load(filename)
                            print(f"{GREEN_BOLD}Session loaded from {filename}{RESET}")
                        except Exception as e:
                            print(f"{RED_BOLD}Error loading session: {str(e)}{RESET}")
                    else:
                        print(f"{RED_BOLD}Please provide a filename: /load <filename>{RESET}")
                    continue
                elif command == 'model' or command.startswith('model '):
                    if command == 'model':
                        # Show current model
                        provider_info = session.get_provider_info()
                        print(format_provider_info(provider_info))
                    else:
                        # Switch model
                        new_model = command[6:].strip()
                        if new_model:
                            try:
                                result = session.switch_provider(session.provider.__class__.__name__.lower(), new_model)
                                print(format_provider_switch_result(result))
                            except Exception as e:
                                print(f"{RED_BOLD}Error switching model: {str(e)}{RESET}")
                        else:
                            print(f"{RED_BOLD}Please provide a model name: /model <model_name>{RESET}")
                    continue
                elif command == 'system' or command.startswith('system '):
                    if command == 'system':
                        # Show current system prompt
                        system_info = session.get_system_prompt_info()
                        print(format_system_prompt_info(system_info))
                    else:
                        # Update system prompt
                        new_prompt = command[7:].strip()
                        if new_prompt:
                            result = session.update_system_prompt(new_prompt)
                            print(format_update_result(result))
                        else:
                            print(f"{RED_BOLD}Please provide a system prompt: /system <prompt>{RESET}")
                    continue
                elif command == 'last' or command.startswith('last '):
                    try:
                        if command == 'last':
                            count = 1
                        else:
                            count = int(command[5:].strip())
                        interactions = session.get_last_interactions(count)
                        print(format_last_interactions(interactions))
                    except ValueError:
                        print(f"{RED_BOLD}Please provide a valid number: /last <count>{RESET}")
                    except Exception as e:
                        print(f"{RED_BOLD}Error getting interactions: {str(e)}{RESET}")
                    continue
                elif command == 'tools':
                    tools_list = session.get_tools_list()
                    print(format_tools_list(tools_list))
                    continue
                elif command.startswith('system '):
                    # Handle system prompt update
                    new_prompt = command[7:].strip()
                    if new_prompt:
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
            
            # Always use enhanced features
            log_step(2, "AGENT→LLM", "Processing with memory and reasoning")
            
            response = session.generate(
                prompt=user_input,
                tools=[read_file, list_files, search_files],
                max_tool_calls=25,
                stream=stream,
                use_memory_context=True,  # Always use memory
                create_react_cycle=True   # Always use ReAct cycles
            )
            
            log_step(3, "LLM→AGENT", "Received response, displaying to user")
            
            # Show "Assistant:" only when we have the response
            print(f"\nAssistant:")
            
            # Handle streaming vs non-streaming responses in REPL
            if stream:
                # For streaming in REPL, iterate through the generator and display progressively
                accumulated_text = ""
                start_time = time.time()
                
                try:
                    for chunk in response:
                        # Session streaming yields strings directly
                        if isinstance(chunk, str):
                            accumulated_text += chunk
                            print(chunk, end='', flush=True)
                        elif hasattr(chunk, 'content'):
                            # Fallback for GenerateResponse objects
                            token_content = chunk.content
                            if token_content:
                                accumulated_text += token_content
                                print(token_content, end='', flush=True)
                    
                    # Print a newline at the end
                    print()
                    
                    # Calculate detailed stats for streaming
                    if accumulated_text:
                        end_time = time.time()
                        time_taken = end_time - start_time
                        
                        # Get session info for context
                        try:
                            # Get the last interaction to extract context info
                            interactions = session.get_last_interactions(1)
                            
                            # Calculate tokens
                            completion_tokens = len(accumulated_text.split())
                            
                            # Estimate context tokens from the prompt length
                            # This is an approximation since we don't have exact tokenization
                            prompt_tokens = len(user_input.split()) + 10  # Add some for system prompt and formatting
                            
                            # Get model info - try multiple approaches
                            model_name = args.model if args else "unknown"  # Use the original model argument as fallback
                            try:
                                provider_info = session.get_provider_info()
                                if provider_info.get('model'):
                                    model_name = provider_info['model']
                            except:
                                pass
                            
                            # Build metadata parts
                            metadata_parts = []
                            
                            # Format context tokens with space separator for thousands
                            formatted_context = f"{prompt_tokens:,}".replace(',', ' ')
                            metadata_parts.append(f"context : {formatted_context} tk")
                            
                            # Generated tokens
                            metadata_parts.append(f"generated : {completion_tokens} tk")
                            
                            # Time
                            metadata_parts.append(f"time : {time_taken:.1f}s")
                            
                            # Speed (tokens per second)
                            if completion_tokens > 0 and time_taken > 0:
                                speed = completion_tokens / time_taken
                                metadata_parts.append(f"speed : {speed:.1f} tk/s")
                            
                            # Model name
                            metadata_parts.append(f"model: {model_name}")
                            
                            if metadata_parts:
                                print(f"\n{BLUE_ITALIC}[{' | '.join(metadata_parts)}]{RESET}")
                            
                        except Exception as e:
                            # Fallback stats if metadata extraction fails
                            final_tokens = len(accumulated_text.split())
                            speed = final_tokens / time_taken if time_taken > 0 else 0
                            model_name = args.model if args else "unknown"
                            print(f"\n{BLUE_ITALIC}[generated : {final_tokens} tk | time : {time_taken:.1f}s | speed : {speed:.1f} tk/s | model: {model_name}]{RESET}")
                        
                except Exception as e:
                    print(f"\n{RED_BOLD}Error during streaming: {str(e)}{RESET}")
                    import traceback
                    traceback.print_exc()
            else:
                # For non-streaming, use the existing formatting
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
    Launch with default settings (ollama:qwen3:4b)
  
  %(prog)s --provider anthropic --model claude-3-5-haiku-20241022
    Launch with Anthropic Claude model
  
  %(prog)s --provider openai --model gpt-4o
    Launch with OpenAI GPT-4o model
  
  %(prog)s --prompt "List all Python files in the current directory"
    Execute a single prompt and exit
  
  %(prog)s --provider mlx --model "mlx-community/Qwen3-30B-A3B-4bit" --prompt "Read the README.md file"
    Execute a prompt with specific provider and model
  
  %(prog)s --provider mlx --model "~/.cache/huggingface/hub/models--published--Qwen3_1.7B_bf16" --stream --prompt "Explain Python"
    Execute a streaming prompt with local MLX model
  
  %(prog)s --provider mlx --model "mlx-community/Qwen3-30B-A3B-4bit" --repetition-penalty 1.1 --temperature 0.8
    Launch with custom repetition penalty and temperature

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
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (only for single prompt mode)"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty for generation (default: model-specific, typically 1.0-1.2)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for generation (default: model-specific, typically 0.7)"
    )
    
    parser.add_argument(
        "--memory",
        type=str,
        default=None,
        help="Path to persist memory across sessions (e.g., agent.pkl)"
    )
    
    parser.add_argument(
        "--structured-output",
        type=str,
        choices=['json', 'yaml'],
        default=None,
        help="Force structured output format"
    )
    
    return parser.parse_args()


def main():    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging
    set_logging()

    print(f"{BLUE_ITALIC}🚀 Starting ALMA with {args.provider}:{args.model}{RESET}")

    # Prepare configuration parameters
    config_params = {
        "max_tokens": args.max_tokens
    }
    
    # Add temperature if provided
    if args.temperature is not None:
        config_params["temperature"] = args.temperature
        print(f"{BLUE_ITALIC}Using temperature: {args.temperature}{RESET}")
    
    # Add repetition penalty if provided (for MLX provider)
    if args.repetition_penalty is not None:
        config_params["repetition_penalty"] = args.repetition_penalty
        print(f"{BLUE_ITALIC}Using repetition penalty: {args.repetition_penalty}{RESET}")
    
    # Show streaming status if enabled
    if args.stream:
        print(f"{BLUE_ITALIC}Streaming enabled for single prompt mode{RESET}")

    # Always create enhanced session with all SOTA features
    print(f"{BLUE_ITALIC}🧠 Creating intelligent agent with memory and reasoning{RESET}")
    
    session = create_session(
        args.provider,
        model=args.model,
        enable_memory=True,
        enable_retry=True,
        persist_memory=args.memory,
        memory_config={
            'working_memory_size': 10,
            'consolidation_threshold': 5
        },
        tools=[read_file, list_files, search_files],
        system_prompt="You are an intelligent AI assistant with memory and reasoning capabilities.",
        **config_params
    )
    
    if args.memory:
        print(f"{BLUE_ITALIC}💾 Memory persisted to: {args.memory}{RESET}")

    # If prompt is provided, execute it and exit
    if args.prompt:
        execute_single_prompt(session, args.prompt, args.stream, args)
        
        # Always show memory insights
        if session.memory:
            print(f"\n{BLUE_ITALIC}📊 Memory Insights:{RESET}")
            stats = session.memory.get_statistics()
            print(f"  Knowledge Graph: {stats['knowledge_graph']['total_facts']} facts")
            print(f"  ReAct Cycles: {stats['total_react_cycles']} completed")
            print(f"  Memory Links: {stats['link_statistics']['total_links']} connections")
            
            # Show sample facts if any
            if session.memory.knowledge_graph.facts:
                print(f"\n  {GREEN_BOLD}Sample Knowledge Triples:{RESET}")
                for i, (_, fact) in enumerate(list(session.memory.knowledge_graph.facts.items())[:3]):
                    print(f"    • {fact.subject} --[{fact.predicate}]--> {fact.object}")
        return

    # Show help for interactive mode
    show_help()

    # Start REPL loop
    start_repl(session, args.stream, args)


if __name__ == "__main__":
    main()