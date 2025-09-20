#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA CLI - Global command line interface for AbstractLLM agent.

This module provides the global 'alma' command that launches the intelligent agent
with all SOTA features including hierarchical memory, ReAct reasoning, and tools.
"""

from abstractllm.factory import create_session
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
from abstractllm.tools.common_tools import read_file, list_files, search_files, write_file
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



def create_agent(provider="ollama", model="qwen3:4b", memory_path=None, max_tool_calls=25,
                 seed=None, top_p=None, max_input_tokens=None, frequency_penalty=None, presence_penalty=None,
                 enable_facts=False, stream=False, quiet=False):
    """Create an enhanced agent with all SOTA features including cognitive abstractions."""

    if not quiet:
        print(f"{BLUE}🧠 Creating intelligent agent with:{RESET}")
        print(f"  • Hierarchical memory system")
        print(f"  • ReAct reasoning cycles")
        print(f"  • Enhanced semantic fact extraction")
        print(f"  • Value resonance evaluation")
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
        # Remove hardcoded max_tokens to let providers use their intelligent defaults
        # Provider-specific defaults: OpenAI/Anthropic ~4k, Ollama ~2k, LM Studio ~8k, MLX ~4k
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

    # Try to create cognitive-enhanced session only if facts are enabled
    if enable_facts:
        try:
            from abstractllm.cognitive.integrations import create_cognitive_session

            # Remove model from config to avoid duplicate parameter
            cognitive_config = config.copy()
            cognitive_config.pop('model', None)

            session = create_cognitive_session(
                provider=provider,
                model=model,
                cognitive_features=['facts'],  # Only facts extraction for now
                cognitive_model="granite3.3:2b",
                **cognitive_config
            )

            if not quiet:
                print(f"{GREEN}✨ Cognitive enhancements loaded successfully{RESET}")
                print(f"  • Semantic fact extraction with granite3.3:2b")
                print(f"  • Enhanced ontological knowledge extraction")
                print(f"  • Dublin Core, Schema.org, SKOS, CiTO frameworks")
                print(f"  • Use /facts to view extracted knowledge\n")

        except ImportError as e:
            if not quiet:
                print(f"{BLUE}ℹ️ Cognitive features not available: {e}{RESET}")
                print(f"  • Using standard session with basic features\n")
            session = create_session(provider, **config)
        except Exception as e:
            if not quiet:
                print(f"{BLUE}ℹ️ Falling back to standard session: {e}{RESET}\n")
            session = create_session(provider, **config)
    else:
        # Create standard session without cognitive features
        if not quiet:
            print(f"{BLUE}ℹ️ Using standard session (facts extraction disabled){RESET}")
            print(f"  • Use --enable-facts to enable cognitive features\n")
        session = create_session(provider, **config)

    if memory_path and not quiet:
        print(f"{GREEN}💾 Memory persisted to: {memory_path}{RESET}\n")

    # Configure streaming mode if requested
    if stream:
        session.default_streaming = True
        if not quiet:
            print(f"{GREEN}🔄 Streaming mode enabled{RESET}")
            print(f"  • Responses will stream progressively\n")

    return session


def run_query(session, prompt, structured_output=None, quiet=False):
    """Execute a query with the agent and display beautiful results."""
    
    # Start thinking animation (unless in quiet mode)
    if not quiet:
        spinner = Spinner()
        spinner.start()
    else:
        spinner = None
    
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
        # Use unified generate API with SOTA features
        # Note: streaming respects session.default_streaming when no explicit stream param provided
        response = session.generate(
            prompt=prompt,
            use_memory_context=True,    # Inject relevant memories
            create_react_cycle=True,     # Create ReAct cycle with scratchpad
            structured_config=config     # Structured output if configured
        )
        
        # Convert string responses to enhanced GenerateResponse objects
        if isinstance(response, str):
            response = enhance_string_response(
                content=response,
                model=getattr(session._provider, 'config_manager', {}).get_param('model') if hasattr(session, '_provider') else 'unknown'
            )

        # Handle streaming generator responses
        elif hasattr(response, '__iter__') and hasattr(response, '__next__'):
            # Stop spinner before streaming starts
            if spinner:
                spinner.stop()

            # Start timing for accurate duration calculation
            import time
            start_time = time.time()

            # Start with clean newline (no header) unless in quiet mode
            if not quiet:
                print()
            accumulated_content = ""
            tool_results = []
            provider_usage = None  # Capture actual provider usage data
            last_react_cycle_id = None  # Capture the unified interaction ID

            try:
                thinking_mode = False  # Track if we're in a thinking section
                first_content_line = True  # Track if we need to add alma> prefix

                for chunk in response:
                    # Handle GenerateResponse objects with content
                    if hasattr(chunk, 'content') and chunk.content is not None:
                        chunk_text = chunk.content
                        
                        # Capture the unified react_cycle_id from chunks
                        if hasattr(chunk, 'react_cycle_id') and chunk.react_cycle_id:
                            last_react_cycle_id = chunk.react_cycle_id

                        # Check for special tool execution chunk types
                        chunk_type = getattr(chunk, 'raw_response', {}).get('type', 'content')

                        if chunk_type == 'tool_execution_start':
                            # Tool execution start - display in yellow, don't add to accumulated content
                            print(f"{Colors.YELLOW}{chunk_text}{Colors.RESET}", end="", flush=True)
                        elif chunk_type in ['tool_completed', 'tool_error']:
                            # Tool completion indicator - display in yellow (✓ or ❌), don't add to accumulated content
                            print(f"{Colors.YELLOW}{chunk_text}{Colors.RESET}", end="", flush=True)
                        else:
                            # Regular content - add to accumulated content
                            accumulated_content += chunk_text

                            if quiet:
                                # In quiet mode, just print the content without formatting
                                print(chunk_text, end="", flush=True)
                            else:
                                # Full formatting for interactive mode
                                # Check for thinking tags to style appropriately
                                if '<think>' in chunk_text and not thinking_mode:
                                    thinking_mode = True
                                    # Apply dim italic styling for thinking content
                                    chunk_to_display = chunk_text.replace('<think>', f'{Colors.DIM}<think>{Colors.RESET}{Colors.DIM}')
                                elif '</think>' in chunk_text and thinking_mode:
                                    thinking_mode = False
                                    chunk_to_display = chunk_text.replace('</think>', f'</think>{Colors.RESET}')
                                elif thinking_mode:
                                    # We're inside thinking tags - apply dim italic styling
                                    chunk_to_display = f'{Colors.DIM}{chunk_text}{Colors.RESET}' if chunk_text.strip() else chunk_text
                                else:
                                    chunk_to_display = chunk_text

                                # Text content - add alma> prefix to first content line
                                if first_content_line and not thinking_mode and chunk_to_display.strip():
                                    # First non-thinking content gets alma> prefix
                                    lines = chunk_to_display.split('\n')
                                    if lines:
                                        lines[0] = f"{Colors.BLUE}alma>{Colors.RESET} {lines[0]}"
                                        chunk_to_display = '\n'.join(lines)
                                    first_content_line = False

                                print(chunk_to_display, end="", flush=True)

                        # Capture usage data if available
                        if hasattr(chunk, 'usage') and chunk.usage:
                            provider_usage = chunk.usage
                    elif isinstance(chunk, str):
                        # Fallback for string chunks (legacy support)
                        chunk_text = chunk

                        # Check for thinking tags to style appropriately
                        if '<think>' in chunk_text and not thinking_mode:
                            thinking_mode = True
                            # Apply dim italic styling for thinking content
                            chunk_to_display = chunk_text.replace('<think>', f'{Colors.DIM}<think>{Colors.RESET}{Colors.DIM}')
                        elif '</think>' in chunk_text and thinking_mode:
                            thinking_mode = False
                            chunk_to_display = chunk_text.replace('</think>', f'</think>{Colors.RESET}')
                        elif thinking_mode:
                            # We're inside thinking tags - apply dim italic styling
                            chunk_to_display = f'{Colors.DIM}{chunk_text}{Colors.RESET}' if chunk_text.strip() else chunk_text
                        else:
                            chunk_to_display = chunk_text

                        # Text content - add alma> prefix to first content line
                        if first_content_line and not thinking_mode and chunk_to_display.strip():
                            # First non-thinking content gets alma> prefix
                            lines = chunk_to_display.split('\n')
                            if lines:
                                lines[0] = f"{Colors.BLUE}alma>{Colors.RESET} {lines[0]}"
                                chunk_to_display = '\n'.join(lines)
                            first_content_line = False

                        print(chunk_to_display, end="", flush=True)
                        accumulated_content += chunk_text
                    elif hasattr(chunk, 'usage') and chunk.usage:
                        # Capture actual provider usage data when available
                        provider_usage = chunk.usage
                    elif isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                        # Tool result - store for later processing
                        tool_results.append(chunk)

                        # Extract tool call details for rich display (matching non-streaming mode)
                        tool_call = chunk.get('tool_call', {})
                        tool_name = tool_call.get('name', 'unknown')
                        tool_args = tool_call.get('arguments', {})

                        # Format arguments string (matching BaseProvider format)
                        if isinstance(tool_args, dict) and tool_args:
                            # Convert dict to argument string like BaseProvider does
                            args_parts = []
                            for key, value in tool_args.items():
                                if isinstance(value, str):
                                    args_parts.append(f"{key}={repr(value)}")
                                else:
                                    args_parts.append(f"{key}={value}")
                            args_str = ", ".join(args_parts)
                        else:
                            args_str = str(tool_args) if tool_args else ""

                        # Display detailed tool execution info (matching non-streaming format)
                        if args_str:
                            tool_message = f"🔧 LLM called {tool_name}({args_str})"
                        else:
                            tool_message = f"🔧 LLM called {tool_name}()"

                        print(f"\n{Colors.YELLOW}{tool_message}{Colors.RESET}", flush=True)

                print()  # Single newline for spacing like non-streaming mode

                # Calculate timing (same as non-streaming mode)
                end_time = time.time()
                reasoning_time = end_time - start_time

                # Use provider usage data if available, otherwise estimate
                if provider_usage:
                    # Use actual provider usage data (same as non-streaming mode)
                    usage_data = provider_usage
                    # Add timing data if missing
                    if isinstance(usage_data, dict) and 'total_time' not in usage_data:
                        usage_data = usage_data.copy()
                        usage_data['total_time'] = reasoning_time
                else:
                    # Fallback: Create usage data with proper context estimation
                    # Get system prompt and conversation history to match non-streaming context calculation
                    system_prompt_tokens = 0
                    conversation_tokens = 0

                    if hasattr(session, 'system_prompt') and session.system_prompt:
                        system_prompt_tokens = len(session.system_prompt.split()) * 1.3

                    # Get conversation history tokens (excluding current response)
                    if hasattr(session, 'messages') and session.messages:
                        for msg in session.messages[:-1]:  # Exclude the response we just added
                            if isinstance(msg, dict) and 'content' in msg:
                                conversation_tokens += len(str(msg['content']).split()) * 1.3

                    # User prompt tokens
                    user_prompt_tokens = len(prompt.split()) * 1.3

                    # Total context = system prompt + conversation + user prompt
                    context_estimate = int(system_prompt_tokens + conversation_tokens + user_prompt_tokens)
                    completion_tokens = int(len(accumulated_content.split()) * 1.3)
                    total_tokens = context_estimate + completion_tokens

                    usage_data = {
                        "prompt_tokens": context_estimate,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "total_time": reasoning_time
                    }

                # Create a GenerateResponse-like object with complete metrics (same as non-streaming)
                response = enhance_string_response(
                    content=accumulated_content,
                    model=getattr(session._provider, 'config_manager', {}).get_param('model') if hasattr(session, '_provider') else 'unknown',
                    usage=usage_data,
                    tools_executed=tool_results,
                    reasoning_time=reasoning_time,  # Include timing for speed calculation
                    react_cycle_id=last_react_cycle_id  # Use the unified interaction ID
                )

                # Save interaction context with proper session_id
                session_id = getattr(session.memory, 'session_id', None) if hasattr(session, 'memory') else f"session_{session.id[:8]}"
                save_interaction_context(response, prompt, session_id)

                # Display metrics line (same as non-streaming mode) unless in quiet mode
                if not quiet:
                    from abstractllm.utils.display import format_metrics_line
                    metrics_line = format_metrics_line(response)
                    if metrics_line:
                        print(f"{metrics_line}")

                    # Add final newline (matching non-streaming mode)
                    print()
                else:
                    # In quiet mode, just add a single newline after content
                    print()

                return response

            except Exception as stream_error:
                print(f"\n{Colors.BRIGHT_RED}Streaming error: {stream_error}{Colors.RESET}")
                return None

        # Stop spinner before displaying response (non-streaming path)
        if spinner:
            spinner.stop()

        # Save interaction context for facts/scratchpad commands with proper session_id
        if isinstance(response, GenerateResponse):
            session_id = getattr(session.memory, 'session_id', None) if hasattr(session, 'memory') else f"session_{session.id[:8]}"
            save_interaction_context(response, prompt, session_id)
            if quiet:
                # In quiet mode, only show the content
                print(response.content or "")
            else:
                display_response(response)
        else:
            # Ultimate fallback
            if quiet:
                print(response)
            else:
                print(f"\n{Colors.BRIGHT_GREEN}Response:{Colors.RESET} {response}")
        
        return response
        
    except Exception as e:
        # Stop spinner before displaying error
        if spinner:
            spinner.stop()
        if not quiet:
            display_error(str(e))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return None


def show_memory_insights(session):
    """Display comprehensive memory and observability insights."""

    # Show traditional memory system insights
    if hasattr(session, 'memory') and session.memory:
        memory = session.memory
        stats = memory.get_statistics()

        print(f"\n{BLUE}📊 Memory System Insights:{RESET}")
        print(f"  • Working Memory: {stats['memory_distribution']['working_memory']} items")
        print(f"  • Episodic Memory: {stats['memory_distribution']['episodic_memory']} experiences")
        print(f"  • Knowledge Graph: {stats['knowledge_graph']['total_facts']} facts")
        print(f"  • ReAct Cycles: {stats['total_react_cycles']} ({stats['successful_cycles']} successful)")
        print(f"  • Bidirectional Links: {stats['link_statistics']['total_links']}")

        # Show sample facts from knowledge graph
        if memory.knowledge_graph.facts:
            print(f"\n  {GREEN}Sample Knowledge Graph Triples:{RESET}")
            for i, (fact_id, fact) in enumerate(list(memory.knowledge_graph.facts.items())[:3]):
                print(f"    {i+1}. {fact.subject} --[{fact.predicate}]--> {fact.object}")

        # Show current ReAct cycle if active
        if hasattr(session, 'current_cycle') and session.current_cycle:
            cycle = session.current_cycle
            print(f"\n  {GREEN}Current ReAct Cycle:{RESET}")
            print(f"    ID: {cycle.cycle_id}")
            print(f"    Query: {cycle.query[:100]}...")
            print(f"    Thoughts: {len(cycle.thoughts)}")
            print(f"    Actions: {len(cycle.actions)}")
            print(f"    Observations: {len(cycle.observations)}")

    # Show LanceDB observability insights
    if hasattr(session, 'lance_store') and session.lance_store:
        try:
            stats = session.lance_store.get_stats()

            print(f"\n{BLUE}🚀 LanceDB Observability:{RESET}")
            print(f"  • Session ID: {session.id[:16]}...")
            print(f"  • Users Tracked: {stats.get('users', {}).get('count', 0)}")
            print(f"  • Sessions Stored: {stats.get('sessions', {}).get('count', 0)}")
            print(f"  • Interactions: {stats.get('interactions', {}).get('count', 0)} (with embeddings)")

            react_count = stats.get('react_cycles', {}).get('count', 0) if 'react_cycles' in stats else 0
            if react_count > 0:
                print(f"  • ReAct Cycles: {react_count} (searchable reasoning)")

            total_size = sum(table.get('size_mb', 0) for table in stats.values() if isinstance(table, dict))
            print(f"  • Storage: {total_size:.2f} MB")
            print(f"  • Embeddings: {'✅ Active' if session.embedder else '❌ Disabled'}")

            print(f"\n  {GREEN}Enhanced Search Capabilities:{RESET}")
            print(f"    • Semantic search: /search <query>")
            print(f"    • Time-based queries: /timeframe <start> <end>")
            print(f"    • Similarity discovery: /similar <text>")
            print(f"    • Cross-session knowledge retrieval")

        except Exception as e:
            print(f"\n{BLUE}🚀 LanceDB Observability: Error retrieving stats{RESET}")
    else:
        print(f"\n{BLUE}📦 Enhanced Observability Available:{RESET}")
        print(f"  Install: pip install lancedb sentence-transformers")
        print(f"  Features: Semantic search, time-based queries, RAG retrieval")


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
                prompt=f"{Colors.BRIGHT_GREEN}user>{Colors.RESET} ",
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

    parser.add_argument(
        "--enable-facts",
        action="store_true",
        help="Enable cognitive fact extraction (disabled by default)"
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode for progressive response display"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        configure_logging(console_level=logging.DEBUG)
    else:
        configure_logging(console_level=logging.WARNING)
    
    # Create agent (quiet mode for single prompt execution)
    session = create_agent(
        provider=args.provider,
        model=args.model,
        memory_path=args.memory,
        max_tool_calls=args.max_tool_calls,
        seed=args.seed,
        top_p=getattr(args, 'top_p', None),
        max_input_tokens=getattr(args, 'max_input_tokens', None),
        frequency_penalty=getattr(args, 'frequency_penalty', None),
        presence_penalty=getattr(args, 'presence_penalty', None),
        enable_facts=getattr(args, 'enable_facts', False),
        stream=getattr(args, 'stream', False),
        quiet=bool(args.prompt)  # Quiet mode when using --prompt
    )
    
    # Execute single prompt or start interactive mode
    if args.prompt:
        # Use quiet mode for --prompt (clean output for scripting)
        response = run_query(session, args.prompt, args.structured, quiet=True)
    else:
        interactive_mode(session)
    
    # Save memory if persisting
    if args.memory and hasattr(session, 'memory') and session.memory:
        session.memory.save_to_disk()
        print(f"\n{Colors.BRIGHT_GREEN}{Symbols.CHECKMARK} Memory saved to {args.memory}{Colors.RESET}")


if __name__ == "__main__":
    main()
