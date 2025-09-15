#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA CLI - Global command line interface for AbstractLLM agent.

This module provides the global 'alma' command that launches the intelligent agent
with all SOTA features including hierarchical memory, ReAct reasoning, and tools.
"""

from abstractllm.factory import create_session
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
from abstractllm.tools.common_tools import read_file, list_files, search_files
from abstractllm.tools.enhanced import tool
from abstractllm.utils.logging import configure_logging
from abstractllm.interface import ModelParameter
from abstractllm.utils.display import display_response, display_error, display_thinking, display_success, Colors, Symbols
from abstractllm.types import GenerateResponse
from abstractllm.utils.response_helpers import enhance_string_response, save_interaction_context
from abstractllm.utils.commands import create_command_processor
import argparse
import sys
import logging
import threading
import time
from pathlib import Path
from pydantic import Field

# Colors for output
BLUE = '\033[34m'
GREEN = '\033[32m'
RESET = '\033[0m'


class Spinner:
    """Simple, elegant thinking indicator with animated dots."""
    
    def __init__(self):
        self.dot_states = ['', '.', '..', '...']
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the thinking animation."""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the animation and clear the line."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the thinking line
        print('\r' + ' ' * 15, end='\r')
    
    def _animate(self):
        """Run the thinking animation with animated dots."""
        idx = 0
        while self.running:
            dots = self.dot_states[idx % len(self.dot_states)]
            # Grey italic: \033[3m\033[90m for italic grey, \033[0m to reset
            print(f'\r\033[3m\033[90mThinking{dots}\033[0m', end='', flush=True)
            idx += 1
            time.sleep(0.5)  # 500ms between dot states for a calm feeling


@tool(
    description="Write content to a file, creating directories if needed",
    tags=["file", "write", "output"],
    when_to_use="When you need to create a new file or overwrite existing content",
    examples=[
        {
            "description": "Write a simple text file",
            "arguments": {
                "file_path": "output.txt",
                "content": "Hello, world!"
            }
        },
        {
            "description": "Create a Python script",
            "arguments": {
                "file_path": "script.py", 
                "content": "#!/usr/bin/env python\nprint('Hello from Python!')"
            }
        }
    ]
)
def write_file(
    file_path: str = Field(description="Path to the file to write", min_length=1),
    content: str = Field(description="Content to write to the file", default=""),
    create_dirs: bool = Field(description="Create parent directories if they don't exist", default=True)
) -> str:
    """
    Write content to a file with robust error handling.
    
    This tool creates or overwrites a file with the specified content.
    It can optionally create parent directories if they don't exist.
    
    Args:
        file_path: Path to the file to write (relative or absolute)
        content: The content to write to the file
        create_dirs: Whether to create parent directories if they don't exist
        
    Returns:
        Success message with file information
        
    Raises:
        PermissionError: If lacking write permissions
        OSError: If there are filesystem issues
    """
    try:
        # Convert to Path object for better handling
        path = Path(file_path)
        
        # Create parent directories if requested and they don't exist
        if create_dirs and path.parent != path:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the content to the file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Get file size for confirmation
        file_size = path.stat().st_size
        
        return f"✅ Successfully wrote {file_size} bytes to '{file_path}'"
        
    except PermissionError:
        return f"❌ Permission denied: Cannot write to '{file_path}'"
    except FileNotFoundError:
        return f"❌ Directory not found: Parent directory of '{file_path}' does not exist"
    except OSError as e:
        return f"❌ File system error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error writing file: {str(e)}"


def create_agent(provider="ollama", model="qwen3:4b", memory_path=None, max_tool_calls=25, 
                 seed=None, top_p=None, max_input_tokens=None, frequency_penalty=None, presence_penalty=None):
    """Create an enhanced agent with all SOTA features."""
    
    print(f"{BLUE}🧠 Creating intelligent agent with:{RESET}")
    print(f"  • Hierarchical memory system")
    print(f"  • ReAct reasoning cycles")
    print(f"  • Knowledge graph extraction")
    print(f"  • Tool capabilities")
    print(f"  • Retry strategies\n")
    
    # Build configuration with SOTA parameters
    config = {
        'model': model,
        'enable_memory': True,
        'enable_retry': True,
        'persist_memory': memory_path,
        'memory_config': {
            'working_memory_size': 10,
            'consolidation_threshold': 5
        },
        'tools': [read_file, list_files, search_files, write_file],
        'system_prompt': "You are an intelligent AI assistant with memory and reasoning capabilities.",
        'max_tokens': 2048,
        'temperature': 0.7,
        'max_tool_calls': max_tool_calls
    }
    
    # Add SOTA parameters if specified
    if seed is not None:
        config[ModelParameter.SEED] = seed
    if top_p is not None:
        config[ModelParameter.TOP_P] = top_p
    if max_input_tokens is not None:
        config[ModelParameter.MAX_INPUT_TOKENS] = max_input_tokens
    if frequency_penalty is not None:
        config[ModelParameter.FREQUENCY_PENALTY] = frequency_penalty
    if presence_penalty is not None:
        config[ModelParameter.PRESENCE_PENALTY] = presence_penalty
    
    session = create_session(provider, **config)
    
    if memory_path:
        print(f"{GREEN}💾 Memory persisted to: {memory_path}{RESET}\n")
    
    return session


def run_query(session, prompt, structured_output=None):
    """Execute a query with the agent and display beautiful results."""
    
    # Start thinking animation
    spinner = Spinner()
    spinner.start()
    
    # Configure structured output if requested
    config = None
    if structured_output:
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON if structured_output == "json" else ResponseFormat.YAML,
            force_valid_json=True,
            max_retries=3,
            temperature_override=0.0
        )
    
    try:
        # Try SOTA features first, fallback to simple generation
        try:
            response = session.generate(
                prompt=prompt,
                use_memory_context=True,    # Inject relevant memories
                create_react_cycle=True,     # Create ReAct cycle with scratchpad
                structured_config=config     # Structured output if configured
            )
        except Exception as sota_error:
            # Fallback to simple generation without SOTA features
            print(f"\r{Colors.DIM}Note: Using simplified mode due to session compatibility{Colors.RESET}")
            response = session.generate_with_tools(
                prompt=prompt,
                max_tool_calls=session.max_tool_calls if hasattr(session, 'max_tool_calls') else 25
            )
        
        # Convert string responses to enhanced GenerateResponse objects
        if isinstance(response, str):
            response = enhance_string_response(
                content=response,
                model=getattr(session._provider, 'config_manager', {}).get_param('model') if hasattr(session, '_provider') else 'unknown'
            )
        
        # Stop spinner before displaying response
        spinner.stop()
        
        # Save interaction context for facts/scratchpad commands
        if isinstance(response, GenerateResponse):
            save_interaction_context(response, prompt)
            display_response(response)
        else:
            # Ultimate fallback
            print(f"\n{Colors.BRIGHT_GREEN}Response:{Colors.RESET} {response}")
        
        return response
        
    except Exception as e:
        # Stop spinner before displaying error
        spinner.stop()
        display_error(str(e))
        return None


def show_memory_insights(session):
    """Display memory system insights."""
    
    if not hasattr(session, 'memory'):
        return
    
    memory = session.memory
    stats = memory.get_statistics()
    
    print(f"\n{BLUE}📊 Memory Insights:{RESET}")
    print(f"  • Working Memory: {stats['memory_distribution']['working_memory']} items")
    print(f"  • Episodic Memory: {stats['memory_distribution']['episodic_memory']} experiences")
    print(f"  • Knowledge Graph: {stats['knowledge_graph']['total_facts']} facts")
    print(f"  • ReAct Cycles: {stats['total_react_cycles']} ({stats['successful_cycles']} successful)")
    print(f"  • Bidirectional Links: {stats['link_statistics']['total_links']}")
    
    # Show sample facts from knowledge graph
    if memory.knowledge_graph.facts:
        print(f"\n  {GREEN}Sample Knowledge Graph Triples:{RESET}")
        for i, (fact_id, fact) in enumerate(list(memory.knowledge_graph.facts.items())[:5]):
            print(f"    {i+1}. {fact.subject} --[{fact.predicate}]--> {fact.object}")
    
    # Show current ReAct cycle if active
    if session.current_cycle:
        cycle = session.current_cycle
        print(f"\n  {GREEN}Current ReAct Cycle:{RESET}")
        print(f"    ID: {cycle.cycle_id}")
        print(f"    Query: {cycle.query[:100]}...")
        print(f"    Thoughts: {len(cycle.thoughts)}")
        print(f"    Actions: {len(cycle.actions)}")
        print(f"    Observations: {len(cycle.observations)}")


def interactive_mode(session):
    """Run enhanced interactive chat with slash command support."""
    from abstractllm.utils.enhanced_input import get_enhanced_input, format_input_info

    # Create command processor
    cmd_processor = create_command_processor(session)

    print(f"\n{Colors.BRIGHT_BLUE}{Symbols.SPARKLES} Enhanced Interactive Mode{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.DIM}Type {Colors.BRIGHT_BLUE}/help{Colors.DIM} for commands or ask questions directly.{Colors.RESET}")
    print(f"{Colors.DIM}Use {Colors.BRIGHT_BLUE}/exit{Colors.DIM} to quit.{Colors.RESET}")
    print(f"{Colors.DIM}Enter your query and press {Colors.BRIGHT_BLUE}Enter{Colors.DIM} to submit (supports up to 8k tokens).{Colors.RESET}\n")

    while True:
        try:
            # Use simple long input with 8k token support
            user_input = get_enhanced_input(
                prompt=f"{Colors.BRIGHT_GREEN}alma>{Colors.RESET} ",
                max_chars=32768  # ~8k tokens
            )

            if not user_input:
                continue

            # Show input info for multi-line inputs
            if '\n' in user_input or len(user_input) > 500:
                info = format_input_info(user_input)
                print(f"{Colors.DIM}{info}{Colors.RESET}")

            # Process slash commands
            if cmd_processor.process_command(user_input):
                continue

            # Regular query - generate response
            response = run_query(session, user_input)

        except KeyboardInterrupt:
            print(f"\n\n{Colors.BRIGHT_GREEN}{Symbols.CHECKMARK} Goodbye!{Colors.RESET}")
            break
        except SystemExit:
            # Exit command was used - no additional message needed
            break
        except Exception as e:
            display_error(f"Unexpected error: {str(e)}")
            print(f"{Colors.DIM}You can continue or type {Colors.BRIGHT_BLUE}/exit{Colors.DIM} to quit.{Colors.RESET}")


def main():
    """Main entry point for the global 'alma' command."""
    
    parser = argparse.ArgumentParser(
        description="ALMA - Intelligent agent with AbstractLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  alma
    Interactive chat with memory and tools
  
  alma --prompt "What files are here?"
    Single query execution
  
  alma --memory agent.pkl --prompt "Remember my name is Alice"
    Use persistent memory
  
  alma --structured json --prompt "List 3 colors with hex codes"
    Get structured JSON output
  
  alma --provider openai --seed 12345 --top-p 0.8 --prompt "Generate text"
    Use SOTA parameters for reproducible, controlled generation
  
  alma --provider openai --frequency-penalty 1.0 --presence-penalty 0.5
    Use OpenAI-specific parameters for content control
"""
    )
    
    parser.add_argument(
        "--provider",
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    
    parser.add_argument(
        "--model",
        default="qwen3:4b",
        help="Model to use (default: qwen3:4b)"
    )
    
    parser.add_argument(
        "--prompt",
        help="Single prompt to execute (exits after)"
    )
    
    parser.add_argument(
        "--memory",
        help="Path to persist memory (e.g., agent.pkl)"
    )
    
    parser.add_argument(
        "--structured",
        choices=["json", "yaml"],
        help="Force structured output format"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )
    
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=25,
        help="Maximum number of tool call iterations (default: 25)"
    )
    
    # SOTA parameters
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible generation"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling parameter (0.0-1.0)"
    )
    
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="Maximum input context length"
    )
    
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        help="Frequency penalty (-2.0 to 2.0, OpenAI only)"
    )
    
    parser.add_argument(
        "--presence-penalty", 
        type=float,
        help="Presence penalty (-2.0 to 2.0, OpenAI only)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        configure_logging(console_level=logging.DEBUG)
    else:
        configure_logging(console_level=logging.WARNING)
    
    # Create agent
    session = create_agent(
        provider=args.provider,
        model=args.model,
        memory_path=args.memory,
        max_tool_calls=args.max_tool_calls,
        seed=args.seed,
        top_p=getattr(args, 'top_p', None),
        max_input_tokens=getattr(args, 'max_input_tokens', None),
        frequency_penalty=getattr(args, 'frequency_penalty', None),
        presence_penalty=getattr(args, 'presence_penalty', None)
    )
    
    # Execute single prompt or start interactive mode
    if args.prompt:
        print(f"\n{Colors.BRIGHT_CYAN}{Symbols.TARGET} Query:{Colors.RESET} {Colors.WHITE}{args.prompt}{Colors.RESET}\n")
        response = run_query(session, args.prompt, args.structured)
        
        # Only show memory insights if response was successful
        if response is not None:
            show_memory_insights(session)
    else:
        interactive_mode(session)
    
    # Save memory if persisting
    if args.memory and hasattr(session, 'memory') and session.memory:
        session.memory.save_to_disk()
        print(f"\n{Colors.BRIGHT_GREEN}{Symbols.CHECKMARK} Memory saved to {args.memory}{Colors.RESET}")


if __name__ == "__main__":
    main()
