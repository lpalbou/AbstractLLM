"""
Slash command system for alma_simple.py interactive mode.

Provides a comprehensive command interface for memory management,
session control, and agent interaction.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from abstractllm.utils.display import (
    Colors, Symbols, display_error, display_info, display_success,
    colorize, create_divider
)


class CommandProcessor:
    """Processes slash commands in interactive mode."""
    
    def __init__(self, session, display_func=None):
        """Initialize command processor."""
        self.session = session
        self.display_func = display_func or print
        self.command_history = []
        
        # Register available commands
        self.commands = {
            'help': self._cmd_help,
            'h': self._cmd_help,
            'memory': self._cmd_memory,
            'mem': self._cmd_memory,
            'save': self._cmd_save,
            'load': self._cmd_load,
            'export': self._cmd_export,
            'import': self._cmd_import,
            'facts': self._cmd_facts,
            'working': self._cmd_working,
            'links': self._cmd_links,
            'scratchpad': self._cmd_scratchpad,
            'scratch': self._cmd_scratchpad,
            'history': self._cmd_history,
            'last': self._cmd_last,
            'clear': self._cmd_clear,
            'reset': self._cmd_reset,
            'status': self._cmd_status,
            'stats': self._cmd_stats,
            'config': self._cmd_config,
            'context': self._cmd_context,
            'seed': self._cmd_seed,
            'temperature': self._cmd_temperature,
            'temp': self._cmd_temperature,
            'memory-facts': self._cmd_memory_facts,
            'system': self._cmd_system,
            'stream': self._cmd_stream,
            'tools': self._cmd_tools,
            'search': self._cmd_search,
            's': self._cmd_search,
            'timeframe': self._cmd_timeframe,
            'tf': self._cmd_timeframe,
            'similar': self._cmd_similar,
            'values': self._cmd_values,
            'exit': self._cmd_exit,
            'quit': self._cmd_exit,
            'q': self._cmd_exit,
        }
    
    def process_command(self, command_line: str) -> bool:
        """
        Process a slash command.
        
        Returns:
            True if command was processed, False if it's a regular query
        """
        if not command_line.startswith('/'):
            return False
        
        # Parse command and arguments
        parts = command_line[1:].strip().split()
        if not parts:
            self._cmd_help()
            return True
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Track command history
        self.command_history.append({
            'timestamp': datetime.now().isoformat(),
            'command': command_line,
            'parsed_cmd': cmd,
            'args': args
        })
        
        # Execute command
        if cmd in self.commands:
            try:
                self.commands[cmd](args)
            except Exception as e:
                display_error(f"Command failed: {str(e)}")
        else:
            display_error(f"Unknown command: {cmd}")
            print(f"{Colors.DIM}Type {colorize('/help', Colors.BRIGHT_BLUE)} for available commands{Colors.RESET}")
        
        # Add empty line after command for better spacing
        print()
        return True
    
    def _cmd_help(self, args: List[str]) -> None:
        """Display help information."""
        print(f"\n{colorize(f'{Symbols.INFO} Available Commands', Colors.BRIGHT_CYAN, bold=True)}")
        print(create_divider(60, "─", Colors.CYAN))
        
        commands_info = [
            ("Memory Management", [
                ("/memory, /mem", "Show memory insights & context size"),
                ("/memory <number>", "Set max input tokens"),
                ("/save <file>", "Save complete session state"),
                ("/load <file>", "Load complete session state"),
                ("/export <file>", "Export memory to JSON"),
                ("/import <file>", "Import memory from JSON"),
                ("/facts [query]", "Show extracted facts"),
                ("/working", "Show working memory contents (recent active items)"),
                ("/links", "Visualize memory links between components"),
                ("/scratchpad, /scratch", "Show reasoning traces")
            ]),
            ("Session Control", [
                ("/history", "Show command history"),
                ("/last [count]", "Replay conversation messages"),
                ("/context [ID]", "Show full context sent to LLM (or specific interaction)"),
                ("/seed [number|random]", "Set/show random seed for deterministic generation"),
                ("/temperature, /temp", "Set/show temperature for generation randomness"),
                ("/memory-facts [max conf occur]", "Configure facts inclusion in memory context"),
                ("/system [prompt]", "Set/show system prompt for the session"),
                ("/stream [on|off]", "Toggle streaming mode for responses and ReAct loops"),
                ("/tools [tool_name]", "Show registered tools or toggle a specific tool"),
                ("/clear", "Clear conversation history"),
                ("/reset", "Reset current session"),
                ("/reset full", "⚠️  PURGE ALL storage (all sessions, embeddings)"),
                ("/status", "Show session status"),
                ("/stats", "Show detailed statistics"),
                ("/config", "Show current configuration")
            ]),
            ("LanceDB Search & Observability", [
                ("/search, /s <query>", "Semantic search across all interactions"),
                ("/search <query> --user <id>", "Search by user with semantic matching"),
                ("/search <query> --from <date>", "Search with date filtering"),
                ("/search <query> --to <date>", "Search up to specific date"),
                ("/timeframe, /tf <start> <end>", "Search exact timeframe (YYYY-MM-DD format)"),
                ("/timeframe <start> <end> <user>", "Timeframe search for specific user"),
                ("/similar <text>", "Find interactions similar to given text"),
                ("/similar <text> --limit <n>", "Limit similarity search results"),
                ("/values", "Show value resonance for entire conversation"),
                ("/values <interaction_id>", "Show value resonance for specific interaction")
            ]),
            ("Navigation", [
                ("/help, /h", "Show this help message"),
                ("/exit, /quit, /q", "Exit interactive mode")
            ])
        ]
        
        for category, commands in commands_info:
            print(f"\n{colorize(f'  {category}:', Colors.BRIGHT_YELLOW, bold=True)}")
            for cmd, description in commands:
                print(f"    {colorize(cmd, Colors.BRIGHT_GREEN):<20} {colorize(description, Colors.WHITE)}")
        
        print(f"\n{colorize('Usage Examples:', Colors.BRIGHT_YELLOW, bold=True)}")
        examples = [
            # Core session management
            "/save my_session.pkl",
            "/load my_session.pkl",
            "/memory 16384",
            "/temperature 0.3",
            "/system You are a helpful coding assistant",

            # LanceDB semantic search examples
            "/search debugging cache problems",
            "/search machine learning --from 2025-09-01",
            "/timeframe 2025-09-15T10:00 2025-09-15T12:15",
            "/similar how to optimize database queries",
            "/search error handling --user alice --limit 5",

            # Memory and observability
            "/working",
            "/facts machine learning",
            "/links",
            "/context",
            "/stats",

            # Session and storage management
            "/reset",
            "/reset full"
        ]
        for example in examples:
            print(f"  {colorize(example, Colors.BRIGHT_BLUE)}")

        # Show LanceDB availability status
        if hasattr(self.session, 'lance_store') and self.session.lance_store:
            print(f"\n{colorize('🚀 LanceDB Enhanced Search: ACTIVE', Colors.BRIGHT_GREEN, bold=True)}")
            print(f"  • Semantic search with embeddings enabled")
            print(f"  • Time-based queries with microsecond precision")
            print(f"  • Cross-session knowledge persistence")
            print(f"  • RAG-powered context retrieval")
        else:
            print(f"\n{colorize('📦 LanceDB Enhanced Search: AVAILABLE', Colors.BRIGHT_YELLOW, bold=True)}")
            print(f"  Install with: {colorize('pip install lancedb sentence-transformers', Colors.BRIGHT_BLUE)}")
            print(f"  • Semantic search across all conversations")
            print(f"  • Precise timeframe filtering")
            print(f"  • Similarity-based interaction discovery")

        # Add spacing after help for better readability
    
    def _cmd_memory(self, args: List[str]) -> None:
        """Show memory system insights or set token limits.

        Usage:
        /mem                     - Show memory overview
        /mem <number>           - Set max input tokens (legacy)
        /mem input <number>     - Set max input tokens
        /mem output <number>    - Set max output tokens
        /mem input <in> output <out> - Set both limits
        /mem reset              - Reset to model defaults
        """
        from abstractllm.enums import ModelParameter

        # Check if we have a provider
        if not hasattr(self.session, '_provider') or not self.session._provider:
            display_error("No provider available")
            return

        provider = self.session._provider

        # Parse arguments for token limit setting
        if args:
            if args[0] == "reset":
                # Reset to model defaults
                try:
                    provider.apply_model_defaults()
                    limits = provider.get_memory_limits()
                    display_success("Memory limits reset to model defaults")
                    input_tokens = f"{limits['input']:,}"
                    output_tokens = f"{limits['output']:,}"
                    print(f"  • Input: {colorize(input_tokens, Colors.WHITE)} tokens")
                    print(f"  • Output: {colorize(output_tokens, Colors.WHITE)} tokens")
                except Exception as e:
                    display_error(f"Failed to reset limits: {e}")
                return

            elif args[0].isdigit():
                # Legacy: single number sets input tokens
                new_input_tokens = int(args[0])
                try:
                    result = provider.set_memory_limits(max_input_tokens=new_input_tokens)
                    display_success(f"Max input tokens set to {new_input_tokens:,}")
                    if new_input_tokens > result['model_input_limit']:
                        print(f"  {colorize('⚠️  Warning:', Colors.BRIGHT_YELLOW)} Exceeds model limit of {result['model_input_limit']:,}")
                except Exception as e:
                    display_error(f"Failed to set input tokens: {e}")
                return

            elif args[0] in ["input", "output"]:
                # Enhanced syntax: /mem input 8000 output 2048
                input_tokens = None
                output_tokens = None

                try:
                    i = 0
                    while i < len(args):
                        if args[i] == "input" and i + 1 < len(args) and args[i + 1].isdigit():
                            input_tokens = int(args[i + 1])
                            i += 2
                        elif args[i] == "output" and i + 1 < len(args) and args[i + 1].isdigit():
                            output_tokens = int(args[i + 1])
                            i += 2
                        else:
                            i += 1

                    if input_tokens is None and output_tokens is None:
                        display_error("Usage: /mem input <number> | /mem output <number> | /mem input <in> output <out>")
                        return

                    result = provider.set_memory_limits(max_input_tokens=input_tokens, max_output_tokens=output_tokens)

                    if input_tokens is not None:
                        display_success(f"Max input tokens set to {input_tokens:,}")
                        if input_tokens > result['model_input_limit']:
                            print(f"  {colorize('⚠️  Warning:', Colors.BRIGHT_YELLOW)} Exceeds model limit of {result['model_input_limit']:,}")

                    if output_tokens is not None:
                        display_success(f"Max output tokens set to {output_tokens:,}")
                        if output_tokens > result['model_output_limit']:
                            print(f"  {colorize('⚠️  Warning:', Colors.BRIGHT_YELLOW)} Exceeds model limit of {result['model_output_limit']:,}")

                except Exception as e:
                    display_error(f"Failed to set memory limits: {e}")
                return

            else:
                display_error("Invalid argument. Use: /mem [input <number>] [output <number>] | /mem reset")
                return

        # Show memory overview (no arguments provided)
        print(f"\n{colorize(f'{Symbols.BRAIN} Memory System Overview', Colors.BRIGHT_BLUE, bold=True)}")
        print(create_divider(60, "─", Colors.BLUE))

        # Model information at the top
        try:
            limits = provider.get_memory_limits()
            model_input_max = f"{limits['model_input_limit']:,}"
            model_output_max = f"{limits['model_output_limit']:,}"
            model_name = limits.get('model', 'Unknown')
            print(f"  {colorize('Model:', Colors.BRIGHT_CYAN)} {colorize(model_name, Colors.WHITE)}")
            print(f"  {colorize('Model Max:', Colors.BRIGHT_CYAN)} {colorize(model_input_max, Colors.WHITE)} input / {colorize(model_output_max, Colors.WHITE)} output")
        except Exception:
            print(f"  {colorize('Model:', Colors.BRIGHT_CYAN)} {colorize('Unknown', Colors.DIM)}")

        # Simple context calculation from session messages
        used_tokens = 0
        context_source = "no context"

        try:
            # Get the current conversation context from session
            if hasattr(self.session, 'messages') and self.session.messages:
                # Build context from system prompt + messages
                context_parts = []

                # Add system prompt if available
                if hasattr(self.session, 'system_prompt') and self.session.system_prompt:
                    context_parts.append(f"System: {self.session.system_prompt}")

                # Add conversation messages
                for msg in self.session.messages:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        context_parts.append(f"{msg.role}: {msg.content}")

                if context_parts:
                    full_context = "\n".join(context_parts)
                    # Simple token estimation: ~4 characters per token
                    used_tokens = len(full_context) // 4
                    context_source = f"{len(self.session.messages)} messages"

        except Exception:
            pass  # Skip if session doesn't have expected structure

        # Token limits using unified system
        try:
            limits = provider.get_memory_limits()

            print(f"  {colorize('Token Usage & Limits:', Colors.BRIGHT_CYAN)}")

            # Input context usage vs limit
            input_limit = limits['input']
            if used_tokens > 0:
                usage_ratio = (used_tokens / input_limit) * 100 if input_limit > 0 else 0
                usage_color = Colors.GREEN if usage_ratio < 50 else Colors.YELLOW if usage_ratio < 80 else Colors.RED
                usage_display = f"{used_tokens:,}"
                limit_display = f"{input_limit:,}"
                percent_display = f"{usage_ratio:.1f}%"
                print(f"    • Input Context: {colorize(usage_display, Colors.WHITE)} / {colorize(limit_display, Colors.DIM)} tokens ({colorize(percent_display, usage_color)})")
                print(f"      {colorize(f'Source: {context_source}', Colors.DIM)}")
            else:
                limit_display = f"{input_limit:,}"
                print(f"    • Input Context: {colorize('0', Colors.WHITE)} / {colorize(limit_display, Colors.DIM)} tokens ({colorize(context_source, Colors.DIM)})")

            # Output generation limit
            output_limit = limits['output']
            output_display = f"{output_limit:,}"
            print(f"    • Output Limit: {colorize(output_display, Colors.WHITE)} tokens max")

            # Configuration info
            print(f"    • {colorize('Commands:', Colors.DIM)} /mem input <n> | /mem output <n> | /mem reset")

        except Exception as e:
            print(f"  {colorize('Token Limits:', Colors.BRIGHT_CYAN)} {colorize('Error retrieving limits', Colors.RED)}")
            print(f"    {colorize(f'Error: {e}', Colors.RED)}")

        # Generation parameters using unified system
        print(f"\n  {colorize('Generation Parameters:', Colors.BRIGHT_CYAN)}")
        param_names = [
            (ModelParameter.TEMPERATURE, "Temperature"),
            (ModelParameter.TOP_P, "Top-P"),
            (ModelParameter.SEED, "Seed"),
            (ModelParameter.FREQUENCY_PENALTY, "Frequency Penalty"),
            (ModelParameter.PRESENCE_PENALTY, "Presence Penalty"),
            (ModelParameter.TOP_K, "Top-K"),
            (ModelParameter.REPETITION_PENALTY, "Repetition Penalty")
        ]

        shown_params = 0
        for param_enum, display_name in param_names:
            value = provider.get_parameter(param_enum)
            if value is not None:
                print(f"    • {display_name}: {colorize(f'{value}', Colors.WHITE)}")
                shown_params += 1

        if shown_params == 0:
            print(f"    • {colorize('Using model defaults', Colors.DIM)}")

        # Provider info
        provider_name = type(provider).__name__.replace('Provider', '').lower()
        base_url = provider.get_parameter(ModelParameter.BASE_URL)
        print(f"    • Provider: {colorize(provider_name.title(), Colors.DIM)}")
        if base_url and base_url != "N/A":
            print(f"    • Base URL: {colorize(base_url, Colors.DIM)}")

        # Memory system stats (if available)
        if hasattr(self.session, 'memory') and self.session.memory:
            try:
                memory = self.session.memory
                stats = memory.get_statistics()

                if isinstance(stats, dict):
                    print(f"\n  {colorize('Memory System:', Colors.BRIGHT_GREEN)}")

                    # Working memory
                    if 'memory_distribution' in stats and isinstance(stats['memory_distribution'], dict):
                        dist = stats['memory_distribution']
                        working_memory_count = str(dist.get('working_memory', 0))
                        episodic_memory_count = str(dist.get('episodic_memory', 0))
                        print(f"    • Working Memory: {colorize(working_memory_count, Colors.WHITE)} items")
                        print(f"    • Episodic Memory: {colorize(episodic_memory_count, Colors.WHITE)} experiences")

                    # Knowledge graph
                    if 'knowledge_graph' in stats and isinstance(stats['knowledge_graph'], dict):
                        kg_stats = stats['knowledge_graph']
                        total_facts_count = str(kg_stats.get('total_facts', 0))
                        print(f"    • Knowledge Graph: {colorize(total_facts_count, Colors.WHITE)} facts")

                    # ReAct cycles
                    total_cycles = stats.get('total_react_cycles', 0)
                    successful_cycles = stats.get('successful_cycles', 0)
                    if total_cycles > 0:
                        success_rate = (successful_cycles / total_cycles) * 100
                        rate_color = Colors.GREEN if success_rate > 80 else Colors.YELLOW if success_rate > 50 else Colors.RED
                        cycles_count = str(total_cycles)
                        success_percent = f'{success_rate:.1f}%'
                        print(f"    • ReAct Cycles: {colorize(cycles_count, Colors.WHITE)} total ({colorize(success_percent, rate_color)} success)")

                    # Memory health
                    if hasattr(memory, 'get_memory_health_report'):
                        try:
                            health = memory.get_memory_health_report()
                            if isinstance(health, dict) and 'overall_health' in health:
                                health_score = health['overall_health']
                                health_color = Colors.BRIGHT_GREEN if health_score > 0.8 else Colors.BRIGHT_YELLOW if health_score > 0.5 else Colors.BRIGHT_RED
                                health_percent = f'{health_score:.1%}'
                                print(f"    • Health Score: {colorize(health_percent, health_color)}")
                        except Exception:
                            pass

            except Exception as e:
                print(f"  {colorize('Memory System:', Colors.BRIGHT_GREEN)} {colorize('Stats unavailable', Colors.DIM)}")

        print()  # Add spacing
    
    def _cmd_save(self, args: List[str]) -> None:
        """Save complete session state."""
        if not args:
            display_error("Usage: /save <filename>")
            return
        
        filename = args[0]
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        try:
            # Create comprehensive session state
            session_state = {
                'timestamp': datetime.now().isoformat(),
                'messages': [msg.to_dict() for msg in self.session.messages],
                'system_prompt': self.session.system_prompt,
                'metadata': self.session.metadata,
                'command_history': self.command_history,
                'provider_config': getattr(self.session, 'provider_config', {}),
                'tools': [tool.__name__ if callable(tool) else str(tool) for tool in self.session.tools] if self.session.tools else [],
                'default_streaming': getattr(self.session, 'default_streaming', False)
            }
            
            # Add memory state if available
            if hasattr(self.session, 'memory') and self.session.memory:
                try:
                    memory = self.session.memory
                    
                    # Create a comprehensive memory snapshot directly
                    memory_snapshot = {
                        "version": "2.0",
                        "session_id": memory.session_id if hasattr(memory, 'session_id') else "unknown",
                        "session_start": memory.session_start.isoformat() if hasattr(memory, 'session_start') else datetime.now().isoformat(),
                        "working_memory": memory.working_memory if hasattr(memory, 'working_memory') else [],
                        "episodic_memory": memory.episodic_memory if hasattr(memory, 'episodic_memory') else [],
                        "chat_history": memory.chat_history if hasattr(memory, 'chat_history') else [],
                        "configuration": {
                            "working_memory_size": getattr(memory, 'working_memory_size', 10),
                            "episodic_consolidation_threshold": getattr(memory, 'episodic_consolidation_threshold', 5)
                        }
                    }
                    
                    # Add knowledge graph facts
                    if hasattr(memory, 'knowledge_graph') and memory.knowledge_graph:
                        facts_dict = {}
                        if hasattr(memory.knowledge_graph, 'facts') and memory.knowledge_graph.facts:
                            for fact_id, fact in memory.knowledge_graph.facts.items():
                                if hasattr(fact, 'to_dict'):
                                    facts_dict[fact_id] = fact.to_dict()
                                else:
                                    # Fallback for simple fact objects
                                    facts_dict[fact_id] = {
                                        "subject": getattr(fact, 'subject', ''),
                                        "predicate": getattr(fact, 'predicate', ''),
                                        "object": getattr(fact, 'object', ''),
                                        "confidence": getattr(fact, 'confidence', 0.5),
                                        "importance": getattr(fact, 'importance', 1.0),
                                        "access_count": getattr(fact, 'access_count', 0)
                                    }
                        memory_snapshot["semantic_memory"] = facts_dict
                    
                    # Add ReAct cycles
                    if hasattr(memory, 'react_cycles') and memory.react_cycles:
                        cycles_dict = {}
                        for cycle_id, cycle in memory.react_cycles.items():
                            if hasattr(cycle, 'to_dict'):
                                cycles_dict[cycle_id] = cycle.to_dict()
                            else:
                                # Fallback
                                cycles_dict[cycle_id] = {
                                    "cycle_id": getattr(cycle, 'cycle_id', cycle_id),
                                    "query": getattr(cycle, 'query', ''),
                                    "success": getattr(cycle, 'success', False)
                                }
                        memory_snapshot["react_cycles"] = cycles_dict
                    
                    # Add memory links
                    if hasattr(memory, 'links') and memory.links:
                        links_list = []
                        for link in memory.links:
                            try:
                                if hasattr(link, 'source_type') and hasattr(link.source_type, 'value'):
                                    source_type_val = link.source_type.value
                                else:
                                    source_type_val = str(getattr(link, 'source_type', 'unknown'))
                                    
                                if hasattr(link, 'target_type') and hasattr(link.target_type, 'value'):
                                    target_type_val = link.target_type.value
                                else:
                                    target_type_val = str(getattr(link, 'target_type', 'unknown'))
                                
                                link_dict = {
                                    "source_type": source_type_val,
                                    "source_id": getattr(link, 'source_id', ''),
                                    "target_type": target_type_val,
                                    "target_id": getattr(link, 'target_id', ''),
                                    "relationship": getattr(link, 'relationship', ''),
                                    "strength": getattr(link, 'strength', 1.0),
                                    "metadata": getattr(link, 'metadata', {}),
                                    "created_at": getattr(link, 'created_at', datetime.now()).isoformat() if hasattr(getattr(link, 'created_at', None), 'isoformat') else str(getattr(link, 'created_at', datetime.now())),
                                    "accessed_count": getattr(link, 'accessed_count', 0)
                                }
                                links_list.append(link_dict)
                            except Exception:
                                # Skip problematic links
                                continue
                        memory_snapshot["links"] = links_list
                    
                    session_state['memory_snapshot'] = memory_snapshot
                    
                except Exception as mem_error:
                    print(f"  {colorize('Memory save warning:', Colors.BRIGHT_YELLOW)} {str(mem_error)}")
                    # Continue without memory data
            
            # Save complete state
            with open(filename, 'wb') as f:
                pickle.dump(session_state, f)
            
            display_success(f"Session saved to {filename}")
            
            # Show what was saved
            size_bytes = os.path.getsize(filename)
            if size_bytes < 1024:
                size_display = f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                size_display = f"{size_bytes / 1024:.1f} KB"
            else:
                size_display = f"{size_bytes / (1024 * 1024):.2f} MB"
            
            print(f"  {colorize('File size:', Colors.DIM)} {size_display}")
            print(f"  {colorize('Messages:', Colors.DIM)} {len(session_state['messages'])}")
            print(f"  {colorize('Commands:', Colors.DIM)} {len(self.command_history)}")
            
            # Show memory components saved
            if 'memory_snapshot' in session_state:
                memory_info = []
                memory_snapshot = session_state['memory_snapshot']
                if 'semantic_memory' in memory_snapshot and memory_snapshot['semantic_memory']:
                    facts_count = len(memory_snapshot['semantic_memory'])
                    memory_info.append(f"{facts_count} facts")
                if 'working_memory' in memory_snapshot and memory_snapshot['working_memory']:
                    working_count = len(memory_snapshot['working_memory'])
                    memory_info.append(f"{working_count} working memory")
                if 'react_cycles' in memory_snapshot and memory_snapshot['react_cycles']:
                    cycles_count = len(memory_snapshot['react_cycles'])
                    memory_info.append(f"{cycles_count} ReAct cycles")
                if 'links' in memory_snapshot and memory_snapshot['links']:
                    links_count = len(memory_snapshot['links'])
                    memory_info.append(f"{links_count} links")
                
                memory_desc = ", ".join(memory_info) if memory_info else "empty"
                print(f"  {colorize('Memory:', Colors.DIM)} {memory_desc}")
            
        except Exception as e:
            display_error(f"Failed to save session: {str(e)}")
    
    def _cmd_load(self, args: List[str]) -> None:
        """Load complete session state."""
        if not args:
            display_error("Usage: /load <filename>")
            return
        
        filename = args[0]
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        if not os.path.exists(filename):
            display_error(f"File not found: {filename}")
            return
        
        try:
            with open(filename, 'rb') as f:
                session_state = pickle.load(f)
            
            # Restore basic session state
            # Note: We can't completely replace the session object, but we can restore its state
            if 'messages' in session_state:
                from abstractllm.types import Message
                self.session.messages = [
                    Message(role=msg['role'], content=msg['content'], name=msg.get('name'))
                    for msg in session_state['messages']
                ]
            
            if 'system_prompt' in session_state:
                self.session.system_prompt = session_state['system_prompt']
            
            if 'metadata' in session_state:
                self.session.metadata.update(session_state['metadata'])
            
            if 'command_history' in session_state:
                self.command_history = session_state['command_history']

            if 'default_streaming' in session_state:
                self.session.default_streaming = session_state['default_streaming']
            
            # Restore memory if available
            if 'memory_snapshot' in session_state and hasattr(self.session, 'memory') and self.session.memory:
                try:
                    memory = self.session.memory
                    memory_snapshot = session_state['memory_snapshot']
                    
                    # Restore basic memory attributes
                    if 'session_id' in memory_snapshot:
                        memory.session_id = memory_snapshot['session_id']
                    
                    if 'session_start' in memory_snapshot:
                        try:
                            memory.session_start = datetime.fromisoformat(memory_snapshot['session_start'])
                        except:
                            memory.session_start = datetime.now()
                    
                    # Restore working memory
                    if 'working_memory' in memory_snapshot:
                        memory.working_memory = memory_snapshot['working_memory']
                    
                    # Restore episodic memory
                    if 'episodic_memory' in memory_snapshot:
                        memory.episodic_memory = memory_snapshot['episodic_memory']
                    
                    # Restore chat history
                    if 'chat_history' in memory_snapshot:
                        memory.chat_history = memory_snapshot['chat_history']
                    
                    # Restore knowledge graph facts
                    if 'semantic_memory' in memory_snapshot and hasattr(memory, 'knowledge_graph'):
                        facts_dict = memory_snapshot['semantic_memory']
                        from abstractllm.memory import Fact  # Import the Fact class
                        
                        # Clear existing facts
                        memory.knowledge_graph.facts = {}
                        from collections import defaultdict
                        memory.knowledge_graph.subject_index = defaultdict(list)
                        memory.knowledge_graph.predicate_index = defaultdict(list)
                        memory.knowledge_graph.object_index = defaultdict(list)
                        
                        # Restore facts
                        for fact_id, fact_data in facts_dict.items():
                            try:
                                fact = Fact(
                                    fact_id=fact_id,
                                    subject=fact_data.get('subject', ''),
                                    predicate=fact_data.get('predicate', ''),
                                    object=fact_data.get('object', ''),
                                    confidence=fact_data.get('confidence', 0.5),
                                    importance=fact_data.get('importance', 1.0)
                                )
                                fact.access_count = fact_data.get('access_count', 0)
                                memory.knowledge_graph.facts[fact_id] = fact
                                
                                # Rebuild indexes
                                memory.knowledge_graph.subject_index[fact.subject].append(fact_id)
                                memory.knowledge_graph.predicate_index[fact.predicate].append(fact_id)
                                memory.knowledge_graph.object_index[fact.object].append(fact_id)
                                
                            except Exception as fact_error:
                                print(f"  {colorize('Fact restore warning:', Colors.BRIGHT_YELLOW)} {str(fact_error)}")
                                continue
                    
                    # Restore ReAct cycles
                    if 'react_cycles' in memory_snapshot and hasattr(memory, 'react_cycles'):
                        cycles_dict = memory_snapshot['react_cycles']
                        from abstractllm.memory import ReActCycle  # Import the ReActCycle class
                        
                        memory.react_cycles = {}
                        for cycle_id, cycle_data in cycles_dict.items():
                            try:
                                # Use the from_dict class method if available
                                if hasattr(ReActCycle, 'from_dict') and isinstance(cycle_data, dict):
                                    # Ensure required fields are present
                                    if 'cycle_id' not in cycle_data:
                                        cycle_data['cycle_id'] = cycle_id
                                    if 'query' not in cycle_data:
                                        cycle_data['query'] = ''
                                    if 'start_time' not in cycle_data:
                                        cycle_data['start_time'] = datetime.now().isoformat()
                                    
                                    cycle = ReActCycle.from_dict(cycle_data)
                                else:
                                    # Fallback to manual construction with correct parameters
                                    cycle = ReActCycle(
                                        cycle_id=cycle_data.get('cycle_id', cycle_id),
                                        query=cycle_data.get('query', '')
                                    )
                                    
                                    # Set additional fields
                                    if 'success' in cycle_data:
                                        cycle.success = cycle_data['success']
                                    if 'start_time' in cycle_data:
                                        try:
                                            cycle.start_time = datetime.fromisoformat(cycle_data['start_time'])
                                        except:
                                            cycle.start_time = datetime.now()
                                    if 'end_time' in cycle_data and cycle_data['end_time']:
                                        try:
                                            cycle.end_time = datetime.fromisoformat(cycle_data['end_time'])
                                        except:
                                            pass
                                            
                                memory.react_cycles[cycle_id] = cycle
                                
                            except Exception as cycle_error:
                                print(f"  {colorize('Cycle restore warning:', Colors.BRIGHT_YELLOW)} {str(cycle_error)}")
                                continue
                    
                    # Restore memory links
                    if 'links' in memory_snapshot and hasattr(memory, 'links'):
                        from abstractllm.memory import MemoryLink, MemoryComponent  # Import correct classes
                        
                        memory.links = []
                        from collections import defaultdict
                        memory.link_index = defaultdict(list)
                        
                        for link_data in memory_snapshot['links']:
                            try:
                                # Convert string back to MemoryComponent enum
                                source_type_str = link_data['source_type']
                                target_type_str = link_data['target_type']
                                
                                # Handle potential enum value mismatches
                                source_type = None
                                target_type = None
                                
                                try:
                                    source_type = MemoryComponent(source_type_str)
                                except ValueError:
                                    # Skip invalid enum values with a warning
                                    print(f"  {colorize('Link restore warning:', Colors.BRIGHT_YELLOW)} Invalid source type '{source_type_str}'")
                                    continue
                                    
                                try:
                                    target_type = MemoryComponent(target_type_str)
                                except ValueError:
                                    # Skip invalid enum values with a warning
                                    print(f"  {colorize('Link restore warning:', Colors.BRIGHT_YELLOW)} Invalid target type '{target_type_str}'")
                                    continue
                                
                                link = MemoryLink(
                                    source_type=source_type,
                                    source_id=link_data['source_id'],
                                    target_type=target_type,
                                    target_id=link_data['target_id'],
                                    relationship=link_data['relationship'],
                                    strength=link_data.get('strength', 1.0),
                                    metadata=link_data.get('metadata', {})
                                )
                                
                                if 'created_at' in link_data:
                                    try:
                                        link.created_at = datetime.fromisoformat(link_data['created_at'])
                                    except:
                                        link.created_at = datetime.now()
                                
                                link.accessed_count = link_data.get('accessed_count', 0)
                                memory.links.append(link)
                                
                                # Rebuild link index
                                link_key = f"{link.source_type.value}:{link.source_id}"
                                memory.link_index[link_key].append(link)
                                
                            except Exception as link_error:
                                print(f"  {colorize('Link restore warning:', Colors.BRIGHT_YELLOW)} {str(link_error)}")
                                continue
                    
                    # Restore configuration
                    if 'configuration' in memory_snapshot:
                        config = memory_snapshot['configuration']
                        memory.working_memory_size = config.get('working_memory_size', 10)
                        memory.episodic_consolidation_threshold = config.get('episodic_consolidation_threshold', 5)
                    
                except Exception as mem_error:
                    print(f"  {colorize('Memory restore warning:', Colors.BRIGHT_YELLOW)} {str(mem_error)}")
                    # Continue without memory restoration
            
            display_success(f"Session loaded from {filename}")
            
            # Show what was loaded
            print(f"  {colorize('Messages restored:', Colors.DIM)} {len(self.session.messages)}")
            print(f"  {colorize('Commands restored:', Colors.DIM)} {len(self.command_history)}")
            if 'memory_data' in session_state:
                print(f"  {colorize('Memory restored:', Colors.DIM)} Yes")
            
            # Show session info
            if session_state.get('timestamp'):
                print(f"  {colorize('Saved on:', Colors.DIM)} {session_state['timestamp']}")
            
        except Exception as e:
            display_error(f"Failed to load session: {str(e)}")
    
    def _cmd_export(self, args: List[str]) -> None:
        """Export memory to JSON format."""
        if not args:
            display_error("Usage: /export <filename>")
            return
        
        filename = args[0]
        if not filename.endswith('.json'):
            filename += '.json'
        
        if not hasattr(self.session, 'memory') or not self.session.memory:
            display_error("Memory system not available")
            return
        
        try:
            memory = self.session.memory
            
            # Create exportable memory data
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'statistics': memory.get_statistics(),
                'facts': [
                    {
                        'id': fact_id,
                        'subject': fact.subject,
                        'predicate': fact.predicate,
                        'object': fact.object,
                        'confidence': fact.confidence,
                        'importance': fact.importance,
                        'access_count': fact.access_count,
                        'timestamp': fact.timestamp.isoformat() if fact.timestamp else None
                    }
                    for fact_id, fact in memory.knowledge_graph.facts.items()
                ],
                'working_memory': [
                    {
                        'content': item.content,
                        'importance': item.importance,
                        'timestamp': item.timestamp.isoformat()
                    }
                    for item in memory.working_memory
                ],
                'episodic_memory': [
                    {
                        'content': item.content,
                        'timestamp': item.timestamp.isoformat(),
                        'importance': item.importance
                    }
                    for item in memory.episodic_memory
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            display_success(f"Memory exported to {filename}")
            
            # Show export stats
            size_kb = os.path.getsize(filename) / 1024
            print(f"  {colorize('File size:', Colors.DIM)} {size_kb:.1f} KB")
            print(f"  {colorize('Facts exported:', Colors.DIM)} {len(export_data['facts'])}")
            print(f"  {colorize('Working memory:', Colors.DIM)} {len(export_data['working_memory'])}")
            print(f"  {colorize('Episodic memory:', Colors.DIM)} {len(export_data['episodic_memory'])}")
            
        except Exception as e:
            display_error(f"Failed to export memory: {str(e)}")
    
    def _cmd_import(self, args: List[str]) -> None:
        """Import memory from JSON format."""
        display_info("Import functionality requires memory system reconstruction - use /load for complete session restore")
    
    def _cmd_facts(self, args: List[str]) -> None:
        """Show extracted facts, toggle fact extraction, or show facts for specific interaction.

        Usage:
          /facts                 - Show all facts from memory
          /facts on              - Enable fact extraction during conversations
          /facts off             - Disable fact extraction during conversations
          /facts <interaction_id> - Show facts for specific interaction (8+ hex chars)
          /facts <query>         - Filter facts by text search
        """
        # Handle toggle commands first
        if args and len(args) == 1:
            toggle_arg = args[0].lower()
            if toggle_arg in ['on', 'off']:
                self._handle_facts_toggle(toggle_arg)
                return

            # Check if first argument looks like an interaction ID (8+ hex chars or cycle_...)
            if (len(args[0]) >= 8 and all(c in '0123456789abcdef' for c in args[0].lower())) or args[0].startswith('cycle_'):
                self._cmd_facts_unified(args)
                return

        # Show current fact extraction status if no arguments
        if not args:
            self._show_facts_status()

        # Show memory facts with optional query filter
        if not hasattr(self.session, 'memory') or not self.session.memory:
            display_error("Memory system not available")
            return

        facts = self.session.memory.knowledge_graph.facts

        if not facts:
            display_info("No facts extracted yet")
            return

        query = ' '.join(args) if args else None

        print(f"\n{colorize(f'{Symbols.KEY} Knowledge Facts', Colors.BRIGHT_YELLOW, bold=True)}")
        if query:
            print(f"{colorize(f'Filtered by: {query}', Colors.DIM, italic=True)}")
        print(create_divider(60, "─", Colors.YELLOW))

        displayed = 0
        for fact_id, fact in facts.items():
            # Simple text matching if query provided
            if query:
                fact_text = f"{fact.subject} {fact.predicate} {fact.object}".lower()
                if query.lower() not in fact_text:
                    continue

            confidence_color = Colors.BRIGHT_GREEN if fact.confidence > 0.8 else Colors.BRIGHT_YELLOW if fact.confidence > 0.5 else Colors.BRIGHT_RED

            print(f"  {displayed + 1}. {colorize(fact.subject, Colors.BRIGHT_BLUE)} "
                  f"--[{colorize(fact.predicate, Colors.BRIGHT_CYAN)}]--> "
                  f"{colorize(fact.object, Colors.BRIGHT_GREEN)}")
            # Get usage count safely
            usage_count = getattr(fact, 'usage_count', getattr(fact, 'access_count', 0))
            print(f"     {colorize(f'Confidence: {fact.confidence:.1%}', confidence_color)} "
                  f"{colorize(f'| Importance: {fact.importance:.1f}', Colors.DIM)} "
                  f"{colorize(f'| Used: {usage_count}x', Colors.DIM)}")

            displayed += 1

        # Show total count (removed artificial limit)
        if displayed > 0:
            print(f"\n{colorize(f'Total: {displayed} facts displayed', Colors.DIM, italic=True)}")

    def _handle_facts_toggle(self, toggle_state: str) -> None:
        """Toggle fact extraction on or off during conversations."""
        # Check if session has cognitive enhancer
        if not hasattr(self.session, '_cognitive_enhancer'):
            display_error("Cognitive fact extraction not available in this session")
            display_info("Start alma with cognitive features enabled to use fact extraction")
            return

        enhancer = self.session._cognitive_enhancer

        if toggle_state == 'on':
            # Enable facts in cognitive features
            enhancer.enabled_features.add('facts')
            display_success("🧠 Fact extraction enabled")
            print(f"  {colorize('Facts will be extracted during conversations using', Colors.DIM)}")
            print(f"  {colorize('semantic ontological framework (Dublin Core, Schema.org, SKOS, CiTO)', Colors.DIM)}")

        elif toggle_state == 'off':
            # Disable facts in cognitive features
            enhancer.enabled_features.discard('facts')
            display_info("🚫 Fact extraction disabled")
            print(f"  {colorize('Facts will no longer be extracted during conversations', Colors.DIM)}")
            print(f"  {colorize('Existing facts in memory are preserved', Colors.DIM)}")

    def _show_facts_status(self) -> None:
        """Show current fact extraction status and available facts."""
        print(f"\n{colorize(f'{Symbols.KEY} Fact Extraction Status', Colors.BRIGHT_YELLOW, bold=True)}")
        print(create_divider(60, "─", Colors.YELLOW))

        # Check cognitive enhancer availability
        if not hasattr(self.session, '_cognitive_enhancer'):
            print(f"  {colorize('Status:', Colors.BRIGHT_BLUE)} {colorize('Not Available', Colors.BRIGHT_RED)}")
            print(f"  {colorize('Reason:', Colors.DIM)} Cognitive features not enabled in this session")
            print(f"\n{colorize('To enable cognitive fact extraction:', Colors.BRIGHT_BLUE)}")
            print(f"  • Restart alma with cognitive features enabled")
            print(f"  • Use the enhanced session factory")
            return

        enhancer = self.session._cognitive_enhancer
        facts_enabled = 'facts' in enhancer.enabled_features

        # Show extraction status
        status_color = Colors.BRIGHT_GREEN if facts_enabled else Colors.BRIGHT_RED
        status_text = "Enabled" if facts_enabled else "Disabled"
        print(f"  {colorize('Status:', Colors.BRIGHT_BLUE)} {colorize(status_text, status_color)}")

        if facts_enabled:
            print(f"  {colorize('Model:', Colors.DIM)} {enhancer.model}")
            print(f"  {colorize('Framework:', Colors.DIM)} Semantic ontological extraction")
            print(f"  {colorize('Ontologies:', Colors.DIM)} Dublin Core, Schema.org, SKOS, CiTO")

        # Show memory facts count
        if hasattr(self.session, 'memory') and self.session.memory:
            facts_count = len(self.session.memory.knowledge_graph.facts)
            print(f"  {colorize('Facts in Memory:', Colors.BRIGHT_BLUE)} {colorize(str(facts_count), Colors.BRIGHT_YELLOW)}")

        # Show available commands
        print(f"\n{colorize('Available Commands:', Colors.BRIGHT_BLUE)}")
        if facts_enabled:
            print(f"  • {colorize('/facts off', Colors.BRIGHT_CYAN)} - Disable fact extraction")
        else:
            print(f"  • {colorize('/facts on', Colors.BRIGHT_CYAN)} - Enable fact extraction")
        print(f"  • {colorize('/facts <query>', Colors.BRIGHT_CYAN)} - Search facts by text")
        print(f"  • {colorize('/facts <id>', Colors.BRIGHT_CYAN)} - Show facts for interaction")

    def _cmd_working(self, args: List[str]) -> None:
        """Show working memory contents (most recent, active items)."""
        if not hasattr(self.session, 'memory') or not self.session.memory:
            display_error("Memory system not available")
            return

        memory = self.session.memory
        working_items = memory.working_memory

        print(f"\n{colorize(f'{Symbols.BRAIN} Working Memory Contents', Colors.BRIGHT_CYAN, bold=True)}")
        print(create_divider(60, "─", Colors.CYAN))

        if not working_items:
            display_info("Working memory is empty")
            return

        print(f"{colorize('Most recent active items:', Colors.BRIGHT_YELLOW)}")
        print(f"{colorize(f'Capacity: {len(working_items)}/{memory.working_memory_size} items', Colors.DIM)}")
        print()

        # Sort by timestamp (most recent first)
        sorted_items = sorted(working_items, key=lambda x: x.get('timestamp', ''), reverse=True)

        for i, item in enumerate(sorted_items):
            # Format timestamp
            timestamp = item.get('timestamp', 'Unknown')
            if timestamp != 'Unknown':
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime('%H:%M:%S')
                except:
                    timestamp = timestamp[:19] if len(timestamp) > 19 else timestamp

            # Get item type and content
            item_type = item.get('type', 'item')
            content = item.get('content', str(item))

            # Truncate long content
            if len(content) > 100:
                content = content[:97] + "..."

            # Color code by type
            type_colors = {
                'message': Colors.BRIGHT_GREEN,
                'thought': Colors.BRIGHT_BLUE,
                'action': Colors.BRIGHT_YELLOW,
                'observation': Colors.BRIGHT_CYAN,
                'consolidation': Colors.BRIGHT_MAGENTA
            }
            type_color = type_colors.get(item_type, Colors.WHITE)

            # Display item
            print(f"  {i+1}. {colorize(f'[{item_type.upper()}]', type_color)} "
                  f"{colorize(timestamp, Colors.DIM)} - {content}")

            # Show importance if available
            importance = item.get('importance')
            if importance is not None:
                importance_color = Colors.BRIGHT_GREEN if importance > 0.7 else Colors.BRIGHT_YELLOW if importance > 0.4 else Colors.DIM
                print(f"     {colorize(f'Importance: {importance:.1f}', importance_color)}")

        print(f"\n{colorize('💡 Tip:', Colors.BRIGHT_YELLOW)} Working memory stores the most recent active items")
        print(f"{colorize('Items are automatically moved to episodic memory when capacity is exceeded', Colors.DIM)}")

    def _cmd_links(self, args: List[str]) -> None:
        """Visualize memory links between different memory components."""
        if not hasattr(self.session, 'memory') or not self.session.memory:
            display_error("Memory system not available")
            return

        memory = self.session.memory

        print(f"\n{colorize(f'{Symbols.LINK} Memory Links System', Colors.BRIGHT_MAGENTA, bold=True)}")
        print(create_divider(60, "─", Colors.MAGENTA))

        # Explain what links are
        print(f"{colorize('What are Memory Links?', Colors.BRIGHT_YELLOW)}")
        print(f"Memory links connect related items across different memory stores:")
        print(f"• {colorize('Facts ↔ Working Memory', Colors.BRIGHT_CYAN)} - Facts referenced in recent conversations")
        print(f"• {colorize('ReAct Cycles ↔ Facts', Colors.BRIGHT_BLUE)} - Knowledge used during reasoning")
        print(f"• {colorize('Chat Messages ↔ Facts', Colors.BRIGHT_GREEN)} - Facts extracted from messages")
        print(f"• {colorize('Cross-references', Colors.BRIGHT_WHITE)} - Related concepts and themes")

        # Get link statistics
        total_links = len(memory.links)
        if total_links == 0:
            print(f"\n{colorize('Status:', Colors.BRIGHT_YELLOW)} No memory links created yet")
            print(f"{colorize('Links are created automatically as you have conversations and the system learns connections', Colors.DIM)}")
            return

        print(f"\n{colorize(f'Current Links: {total_links} active connections', Colors.BRIGHT_CYAN)}")

        # Show link breakdown by type
        link_types = {}
        for link in memory.links:
            link_type = f"{link.source_type.value} → {link.target_type.value}"
            link_types[link_type] = link_types.get(link_type, 0) + 1

        if link_types:
            print(f"\n{colorize('Link Types:', Colors.BRIGHT_YELLOW)}")
            for link_type, count in sorted(link_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {colorize(link_type, Colors.BRIGHT_WHITE)}: {colorize(str(count), Colors.BRIGHT_CYAN)} connections")

        # Show visualization
        visualization = self.session.visualize_memory_links()
        if visualization:
            print(f"\n{colorize('Link Visualization:', Colors.BRIGHT_YELLOW)}")
            print(visualization)

        # Usage tips
        print(f"\n{colorize('💡 Usage Tips:', Colors.BRIGHT_YELLOW)}")
        print(f"• Links help the AI remember context and make connections")
        print(f"• Stronger links (more ●) indicate more important relationships")
        print(f"• Links are created automatically based on conversation patterns")
        print(f"• Use {colorize('/facts', Colors.BRIGHT_BLUE)} to see the knowledge these links connect")

    def _show_interaction_scratchpad(self, interaction_id: str) -> None:
        """Show Think → Act → Observe cycles - MINIMALIST IMPLEMENTATION."""
        from pathlib import Path
        import json
        from abstractllm.utils.display import display_error, Colors

        # Convert ID to standard format
        if interaction_id.startswith('cycle_'):
            cycle_id = interaction_id
        else:
            cycle_id = f"cycle_{interaction_id}"

        short_id = cycle_id.replace('cycle_', '')

        # Find scratchpad entries
        base_dir = Path.home() / ".abstractllm" / "sessions"
        scratchpad_entries = []
        cycle_data = None

        if base_dir.exists():
            for session_dir in base_dir.iterdir():
                if session_dir.is_dir():
                    # Get scratchpad entries
                    scratchpad_dir = session_dir / "scratchpads"
                    if scratchpad_dir.exists():
                        for scratchpad_file in scratchpad_dir.glob("scratchpad_*.json"):
                            try:
                                with open(scratchpad_file, 'r') as f:
                                    data = json.load(f)
                                entries = [e for e in data.get('entries', []) if e.get('cycle_id') == cycle_id]
                                if entries:
                                    scratchpad_entries = entries
                                    break
                            except Exception:
                                continue

                    # Get cycle data
                    interactions_dir = session_dir / "interactions"
                    if interactions_dir.exists():
                        for interaction_dir in interactions_dir.iterdir():
                            if interaction_dir.is_dir() and cycle_id in interaction_dir.name:
                                cycle_files = list(interaction_dir.glob("cycle_*.json"))
                                if cycle_files:
                                    try:
                                        with open(cycle_files[0], 'r') as f:
                                            cycle_data = json.load(f)
                                            break
                                    except Exception:
                                        continue

                    if scratchpad_entries:
                        break

        if not scratchpad_entries:
            display_error(f"No scratchpad found for: {short_id}")
            return

        # Get query
        query = scratchpad_entries[0].get('metadata', {}).get('query', 'Unknown query') if scratchpad_entries else 'Unknown'

        # Display header
        print(f"\n{Colors.BRIGHT_CYAN}🧠 ReAct Scratchpad - {short_id}{Colors.RESET}")
        print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"\n{Colors.BRIGHT_BLUE}📋 Query: {Colors.WHITE}{query}{Colors.RESET}")

        # Show warning about verbatim content
        has_tools = cycle_data and 'tools_executed' in cycle_data
        print(f"{Colors.DIM}Note: This shows available reasoning data. True LLM verbatim output is not stored.{Colors.RESET}")

        # Display phases
        for entry in scratchpad_entries:
            phase = entry.get('phase', 'unknown')
            content = entry.get('content', '')

            if phase == 'cycle_start':
                print(f"\n{Colors.BRIGHT_GREEN}🚀 START{Colors.RESET}")
                print(f"{Colors.GREEN}{content}{Colors.RESET}")

            elif phase == 'thinking':
                print(f"\n{Colors.BRIGHT_YELLOW}🤔 THINK{Colors.RESET}")
                print(f"{Colors.YELLOW}{content}{Colors.RESET}")

            elif phase == 'acting':
                print(f"\n{Colors.BRIGHT_MAGENTA}⚡ ACT{Colors.RESET}")
                if has_tools:
                    # Show reconstructed tool calls (not true verbatim)
                    for tool in cycle_data['tools_executed']:
                        tool_name = tool.get('name', 'unknown')
                        tool_args = tool.get('arguments', {})
                        print(f"{Colors.MAGENTA}Tool: {tool_name}{Colors.RESET}")
                        if tool_args:
                            args_str = json.dumps(tool_args, indent=2)
                            print(f"{Colors.DIM}{args_str}{Colors.RESET}")
                else:
                    print(f"{Colors.MAGENTA}{content}{Colors.RESET}")

            elif phase == 'observing':
                print(f"\n{Colors.BRIGHT_BLUE}👁️ OBSERVE{Colors.RESET}")
                if has_tools:
                    # Show actual tool results
                    for tool in cycle_data['tools_executed']:
                        result = tool.get('result', 'No result')
                        print(f"{Colors.BLUE}{result}{Colors.RESET}")
                        exec_time = tool.get('execution_time')
                        if exec_time:
                            print(f"{Colors.DIM}⏱️ {exec_time:.3f}s{Colors.RESET}")
                else:
                    print(f"{Colors.BLUE}{content}{Colors.RESET}")

            elif phase == 'final_answer':
                print(f"\n{Colors.BRIGHT_GREEN}📝 FINAL ANSWER{Colors.RESET}")
                final_content = cycle_data.get('response_content', content) if cycle_data else content
                print(f"{Colors.GREEN}{final_content}{Colors.RESET}")

            elif phase == 'cycle_complete':
                print(f"\n{Colors.BRIGHT_GREEN}✅ COMPLETE{Colors.RESET}")
                print(f"{Colors.GREEN}{content}{Colors.RESET}")

        print(f"\n{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print()

    def _cmd_scratchpad(self, args: List[str]) -> None:
        """Show Think → Act → Observe reasoning cycles - MINIMALIST IMPLEMENTATION."""
        if args:
            interaction_id = args[0]
            self._show_interaction_scratchpad(interaction_id)
        else:
            self._list_recent_interactions()

    def _list_recent_interactions(self) -> None:
        """List recent interactions - MINIMALIST IMPLEMENTATION."""
        from pathlib import Path
        import json
        from abstractllm.utils.display import Colors, colorize

        print(f"\n{colorize('🧠 Recent Scratchpads', Colors.BRIGHT_CYAN)}")
        print(f"{colorize('─' * 40, Colors.CYAN)}")

        base_dir = Path.home() / ".abstractllm" / "sessions"
        interactions = []

        if base_dir.exists():
            for session_dir in base_dir.iterdir():
                if session_dir.is_dir():
                    interactions_dir = session_dir / "interactions"
                    if interactions_dir.exists():
                        for interaction_dir in interactions_dir.iterdir():
                            if interaction_dir.is_dir():
                                context_file = interaction_dir / "context.json"
                                if context_file.exists():
                                    try:
                                        with open(context_file, 'r') as f:
                                            context = json.load(f)
                                        interaction_id = context.get('interaction_id', interaction_dir.name)
                                        short_id = interaction_id.replace('interaction_', '').replace('cycle_', '')
                                        query = context.get('query', 'Unknown query')
                                        timestamp = context.get('timestamp', '')
                                        interactions.append({'id': short_id, 'query': query, 'timestamp': timestamp})
                                    except Exception:
                                        continue

        if not interactions:
            print(f"{colorize('No scratchpads found', Colors.DIM)}")
            return

        # Sort and show recent interactions
        interactions.sort(key=lambda x: x['timestamp'], reverse=True)
        for i, interaction in enumerate(interactions[:5], 1):
            query_preview = interaction['query'][:40] + "..." if len(interaction['query']) > 40 else interaction['query']
            print(f"  {i}. {colorize(interaction['id'], Colors.BRIGHT_BLUE)} - {query_preview}")

        print(f"\n{colorize('Usage:', Colors.DIM)} /scratch <ID> to view reasoning")
    
    def _cmd_history(self, args: List[str]) -> None:
        """Show command history."""
        if not self.command_history:
            display_info("No command history available")
            return
        
        print(f"\n{colorize(f'{Symbols.CLOCK} Command History', Colors.BRIGHT_WHITE, bold=True)}")
        print(create_divider(60, "─", Colors.WHITE))
        
        # Show last 10 commands
        recent_commands = self.command_history[-10:]
        for i, cmd_info in enumerate(recent_commands, 1):
            timestamp = cmd_info['timestamp'][:19]  # Remove microseconds
            command = cmd_info['command']
            print(f"  {i:2d}. {colorize(timestamp, Colors.DIM)} {colorize(command, Colors.BRIGHT_GREEN)}")
    
    def _cmd_last(self, args: List[str]) -> None:
        """Replay conversation messages."""
        if not hasattr(self.session, 'messages') or not self.session.messages:
            display_info("No conversation messages to replay")
            return
        
        # Parse count parameter
        count = None
        if args:
            try:
                count = int(args[0])
                if count <= 0:
                    display_error("Count must be a positive integer")
                    return
            except ValueError:
                display_error(f"Invalid count '{args[0]}' - must be an integer")
                return
        
        # Get messages to display
        messages = self.session.messages
        if count:
            messages = messages[-count*2:] if len(messages) >= count*2 else messages
            display_title = f"Last {min(count, len(messages)//2)} Interaction(s)"
        else:
            display_title = f"Complete Conversation ({len(messages)} messages)"
        
        print(f"\n{colorize(f'{Symbols.CHAT} {display_title}', Colors.BRIGHT_CYAN, bold=True)}")
        # Add spacing after status for better readability
        
        # Group messages into interactions
        interactions = self._group_messages_into_interactions(messages)
        
        for i, interaction in enumerate(interactions, 1):
            user_msg = interaction.get('user')
            assistant_msg = interaction.get('assistant')
            
            # Interaction header
            print(f"\n{colorize(f'{Symbols.ARROW_RIGHT} Interaction {i}', Colors.BRIGHT_YELLOW, bold=True)}")
            print(create_divider(70, "─", Colors.YELLOW))
            
            # User message
            if user_msg:
                print(f"\n{colorize('👤 User:', Colors.BRIGHT_BLUE, bold=True)}")
                print(self._format_message_content(user_msg['content']))
            
            # Assistant message
            if assistant_msg:
                print(f"\n{colorize('🤖 Assistant:', Colors.BRIGHT_GREEN, bold=True)}")
                assistant_content = assistant_msg['content']
                
                # Check if it contains thinking tags
                if '<think>' in assistant_content and '</think>' in assistant_content:
                    # Extract and format thinking vs response
                    import re
                    think_match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1).strip()
                        response = assistant_content.split('</think>')[-1].strip()
                        
                        # Show thinking process (collapsed)
                        think_preview = thinking.split('\n')[0][:100] + "..." if len(thinking) > 100 else thinking[:100]
                        print(f"  {colorize('[THINKING]', Colors.DIM)} {colorize(think_preview, Colors.DIM)}")
                        
                        # Show main response
                        if response:
                            print(self._format_message_content(response))
                    else:
                        print(self._format_message_content(assistant_content))
                else:
                    print(self._format_message_content(assistant_content))
        
        # Summary footer
        # Add spacing after history for better readability
        total_interactions = len(interactions)
        if count and total_interactions > count:
            print(f"{colorize(f'Showing last {count} of {total_interactions} total interactions', Colors.DIM)}")
        else:
            print(f"{colorize(f'Complete conversation: {total_interactions} interactions', Colors.DIM)}")
    
    def _group_messages_into_interactions(self, messages: list) -> list:
        """Group messages into user-assistant interaction pairs."""
        interactions = []
        current_interaction = {}
        
        for msg in messages:
            if hasattr(msg, 'role'):
                role = msg.role
                content = msg.content
            else:
                # Handle dict-like message objects
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
            
            if role == 'user':
                # Start new interaction
                if current_interaction:
                    interactions.append(current_interaction)
                current_interaction = {'user': {'role': role, 'content': content}}
            elif role == 'assistant':
                # Complete current interaction
                if 'user' in current_interaction:
                    current_interaction['assistant'] = {'role': role, 'content': content}
                else:
                    # Orphaned assistant message, create interaction
                    current_interaction = {'assistant': {'role': role, 'content': content}}
            
        # Add final interaction if exists
        if current_interaction:
            interactions.append(current_interaction)
        
        return interactions
    
    def _format_message_content(self, content: str, indent: str = "  ") -> str:
        """Format message content with proper indentation and wrapping."""
        if not content:
            return f"{indent}{colorize('(empty message)', Colors.DIM)}"
        
        # Split content into lines and add indentation
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip():
                # Wrap long lines
                if len(line) > 100:
                    # Simple word wrapping
                    words = line.split(' ')
                    current_line = indent
                    for word in words:
                        if len(current_line + word) > 100:
                            formatted_lines.append(current_line.rstrip())
                            current_line = indent + word + " "
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        formatted_lines.append(current_line.rstrip())
                else:
                    formatted_lines.append(f"{indent}{line}")
            else:
                formatted_lines.append("")  # Preserve empty lines
        
        return '\n'.join(formatted_lines)
    
    def _cmd_clear(self, args: List[str]) -> None:
        """Clear conversation history."""
        self.session.messages.clear()
        self.session._last_assistant_idx = -1
        display_success("Conversation history cleared")
    
    def _cmd_reset(self, args: List[str]) -> None:
        """Reset session or completely purge all storage."""

        # Check if this is a full reset
        if args and args[0].lower() == 'full':
            self._reset_full_storage()
        else:
            self._reset_current_session()

    def _reset_current_session(self) -> None:
        """Reset only the current session."""
        print(f"{colorize('Reset current session?', Colors.YELLOW)}")
        confirm = input(f"{colorize('[y/N]: ', Colors.BRIGHT_YELLOW)}")

        if confirm.lower() in ['y', 'yes']:
            self._clear_session_data()
            display_success("Session reset")
        else:
            display_info("Cancelled")

    def _clear_session_data(self) -> None:
        """Clear all session data."""
        self.session.messages.clear()
        self.session._last_assistant_idx = -1

        if hasattr(self.session, 'memory') and self.session.memory:
            from abstractllm.memory import HierarchicalMemory
            self.session.memory = HierarchicalMemory()

        self.command_history.clear()

    def _reset_full_storage(self) -> None:
        """Completely purge all LanceDB storage and sessions."""
        print(f"{colorize('🔥 This will DELETE ALL storage permanently', Colors.BRIGHT_RED, bold=True)}")

        confirm = input(f"{colorize('Type \"DELETE\" to confirm: ', Colors.BRIGHT_YELLOW)}")
        if confirm != "DELETE":
            display_info("Deletion cancelled - confirmation text did not match")
            return

        self._purge_storage()
        self._clear_session_data()

        display_success("🔥 STORAGE PURGED - Fresh start ready")

    def _purge_storage(self) -> None:
        """Purge all storage directories."""
        import shutil
        from pathlib import Path
        import gc
        import time

        # Close connections
        self.session.lance_store = None
        self.session.embedder = None
        gc.collect()

        # Delete storage
        paths = [
            Path.home() / ".abstractllm" / "lancedb",
            Path.home() / ".abstractllm" / "embeddings"
        ]

        for path in paths:
            if path.exists():
                shutil.rmtree(path)

        time.sleep(0.5)

        # Recreate fresh
        from abstractllm.storage import ObservabilityStore, EmbeddingManager
        self.session.lance_store = ObservabilityStore()
        self.session.embedder = EmbeddingManager()
    
    def _cmd_status(self, args: List[str]) -> None:
        """Show session status."""
        print(f"\n{colorize(f'{Symbols.INFO} Session Status', Colors.BRIGHT_BLUE, bold=True)}")
        print(create_divider(60, "─", Colors.BLUE))
        
        # Basic session info
        print(f"  {colorize('Session ID:', Colors.BRIGHT_GREEN)} {self.session.id}")
        print(f"  {colorize('Created:', Colors.BRIGHT_GREEN)} {self.session.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {colorize('Messages:', Colors.BRIGHT_GREEN)} {len(self.session.messages)}")
        print(f"  {colorize('Tools:', Colors.BRIGHT_GREEN)} {len(self.session.tools) if self.session.tools else 0}")
        
        # Provider info
        if hasattr(self.session, '_provider') and self.session._provider:
            provider = self.session._provider
            provider_name = provider.__class__.__name__.replace('Provider', '')
            print(f"  {colorize('Provider:', Colors.BRIGHT_CYAN)} {provider_name}")
            
            # Model info
            if hasattr(provider, 'config_manager'):
                model = provider.config_manager.get_param('model')
                if model:
                    print(f"  {colorize('Model:', Colors.BRIGHT_CYAN)} {model}")
        
        # Memory status
        memory_status = "Enabled" if hasattr(self.session, 'memory') and self.session.memory else "Disabled"
        memory_color = Colors.BRIGHT_GREEN if memory_status == "Enabled" else Colors.BRIGHT_RED
        print(f"  {colorize('Memory:', memory_color)} {memory_status}")
        
        # Command history
        print(f"  {colorize('Commands run:', Colors.BRIGHT_MAGENTA)} {len(self.command_history)}")
    
    def _cmd_stats(self, args: List[str]) -> None:
        """Show detailed statistics."""
        self._cmd_status(args)
        if hasattr(self.session, 'memory') and self.session.memory:
            print()
            self._cmd_memory(args)

        # Show LanceDB observability statistics
        if hasattr(self.session, 'lance_store') and self.session.lance_store:
            try:
                stats = self.session.lance_store.get_stats()

                print(f"\n{colorize(f'{Symbols.CHART} LanceDB Observability Storage', Colors.BRIGHT_CYAN, bold=True)}")
                print(create_divider(60, "─", Colors.CYAN))

                print(f"  {colorize('Session ID:', Colors.CYAN)} {self.session.id[:16]}...")
                print(f"  {colorize('Users:', Colors.CYAN)} {stats.get('users', {}).get('count', 0)}")
                print(f"  {colorize('Sessions:', Colors.CYAN)} {stats.get('sessions', {}).get('count', 0)}")
                print(f"  {colorize('Interactions stored:', Colors.CYAN)} {stats.get('interactions', {}).get('count', 0)}")
                print(f"  {colorize('ReAct cycles:', Colors.CYAN)} {stats.get('react_cycles', {}).get('count', 0) if 'react_cycles' in stats else 0}")

                total_size = sum(table.get('size_mb', 0) for table in stats.values() if isinstance(table, dict))
                print(f"  {colorize('Total storage:', Colors.CYAN)} {total_size:.2f} MB")
                print(f"  {colorize('Embeddings enabled:', Colors.CYAN)} {bool(self.session.embedder)}")

                print(f"\n{colorize('Available Commands:', Colors.BRIGHT_BLUE)}")
                print(f"  • {colorize('/search <query>', Colors.BRIGHT_BLUE)} - Semantic search with embeddings")
                print(f"  • {colorize('/timeframe <start> <end>', Colors.BRIGHT_BLUE)} - Search by exact time")
                print(f"  • {colorize('/similar <text>', Colors.BRIGHT_BLUE)} - Find similar interactions")

            except Exception as e:
                print(f"\n{colorize('LanceDB stats unavailable', Colors.DIM)}")
                logger.debug(f"LanceDB stats error: {e}")
        else:
            print(f"\n{colorize('LanceDB observability not available', Colors.YELLOW)}")
            print(f"  {colorize('Install lancedb and sentence-transformers for enhanced search', Colors.DIM)}")
    
    def _cmd_config(self, args: List[str]) -> None:
        """Show current configuration."""
        print(f"\n{colorize(f'{Symbols.GEAR} Configuration', Colors.BRIGHT_GREEN, bold=True)}")
        print(create_divider(60, "─", Colors.GREEN))
        
        # Provider config
        if hasattr(self.session, '_provider') and self.session._provider:
            provider = self.session._provider
            if hasattr(provider, 'config_manager'):
                try:
                    # Get config items safely
                    config_items = []
                    if hasattr(provider.config_manager, '_config'):
                        config = provider.config_manager._config
                        for key, value in config.items():
                            if 'key' in str(key).lower():  # Hide API keys
                                value = "***HIDDEN***"
                            config_items.append((key, value))
                    
                    if config_items:
                        for key, value in config_items:
                            print(f"  {colorize(f'{key}:', Colors.BRIGHT_BLUE)} {colorize(str(value), Colors.WHITE)}")
                    else:
                        print(f"  {colorize('No configuration items available', Colors.DIM)}")
                        
                except Exception as e:
                    print(f"  {colorize('Config access error:', Colors.BRIGHT_RED)} {str(e)}")
        
        # Session config
        print(f"\n{colorize('Session Config:', Colors.BRIGHT_YELLOW)}")
        if hasattr(self.session, 'max_tool_calls'):
            print(f"  {colorize('Max tool calls:', Colors.BRIGHT_BLUE)} {colorize(str(self.session.max_tool_calls), Colors.WHITE)}")

        streaming_status = getattr(self.session, 'default_streaming', False)
        streaming_text = colorize("ENABLED", Colors.BRIGHT_GREEN) if streaming_status else colorize("DISABLED", Colors.BRIGHT_RED)
        print(f"  {colorize('Default streaming:', Colors.BRIGHT_BLUE)} {streaming_text}")

        print(f"  {colorize('System prompt:', Colors.BRIGHT_BLUE)} {colorize('Set' if self.session.system_prompt else 'None', Colors.WHITE)}")

    def _cmd_context(self, args: List[str]) -> None:
        """Show the exact verbatim context sent to the LLM."""
        # Check if specific context ID is requested
        if args and not args[0] in ["compact", "debug", "full"]:
            interaction_id = args[0]
            # Handle both formats: "4258e5b8" and "cycle_4258e5b8"
            if interaction_id.startswith('cycle_'):
                interaction_id = interaction_id[6:]  # Remove 'cycle_' prefix

            self._show_specific_context_unified(interaction_id)
            return

        # Show current/last context (existing behavior)
        # First try to get verbatim context from the provider
        if hasattr(self.session, '_provider') and hasattr(self.session._provider, 'get_last_verbatim_context'):
            verbatim_data = self.session._provider.get_last_verbatim_context()

            if verbatim_data:
                print(f"\n╔══════════════ EXACT VERBATIM LLM INPUT ══════════════╗")
                print(f"║ Timestamp: {verbatim_data['timestamp']}")
                print(f"║ Model: {verbatim_data['model']}")
                print(f"║ Provider: {verbatim_data['provider']}")
                print(f"╚═══════════════════════════════════════════════════════╝")
                print()
                print(verbatim_data['context'])
                return

        # Fallback to old context logging system
        from abstractllm.utils.context_logging import get_context_logger

        logger = get_context_logger()

        # Determine format
        format = "full"
        if args:
            if args[0] in ["compact", "debug"]:
                format = args[0]

        context = logger.get_last_context(format)

        if context:
            print(context)
        else:
            display_info("No context has been sent to the LLM yet in this session")

    def _show_specific_context(self, context_id: str) -> None:
        """Show context for a specific interaction or step ID."""
        from abstractllm.utils.display import Colors, colorize
        from pathlib import Path
        import json
        import gzip

        # Use unified storage location
        storage_locations = [
            # Unified context storage (.abstractllm directory)
            Path.home() / ".abstractllm" / "sessions"
        ]

        context_data = None
        source_location = None

        # Find the interaction context file (handle bare ID or full interaction_id)
        search_id = context_id if context_id.startswith('interaction_') else f"interaction_{context_id}"

        for base_path in storage_locations:
            for session_dir in base_path.iterdir():
                if session_dir.is_dir():
                    interaction_file = session_dir / "interactions" / search_id / "context.json"
                    if interaction_file.exists():
                        try:
                            with open(interaction_file, 'r') as f:
                                context_data = json.load(f)
                            source_location = str(interaction_file)
                            break
                        except Exception:
                            continue
                if context_data:
                    break

        if not context_data:
            display_error(f"Interaction not found: {context_id}")
            print(f"\n{colorize('Usage: /context <interaction_id>', Colors.DIM)}")
            return

        # Display the context
        short_id = context_id.replace('interaction_', '')

        # Display context data (all contexts are created equal - no legacy distinction)
        if 'verbatim_context' in context_data:
            self._display_verbatim_context(context_data, short_id, source_location)
        elif 'system_prompt' in context_data or 'messages' in context_data:
            self._display_enhanced_context(context_data, short_id, source_location)
        else:
            # Display interaction data in standard format
            self._display_interaction_context(context_data, short_id, source_location)

    def _display_enhanced_context(self, context_data: dict, short_id: str, source: str) -> None:
        """Display enhanced context data with full LLM context."""
        from abstractllm.utils.display import Colors, colorize

        print(f"\n{colorize('🔍 LLM Context Details', Colors.BRIGHT_CYAN, bold=True)} - {colorize(short_id, Colors.WHITE)}")
        print(f"{colorize('─' * 60, Colors.CYAN)}")

        # Context metadata
        print(f"\n{colorize('📋 Context Metadata', Colors.BRIGHT_BLUE)}")
        print(f"  {colorize('Context ID:', Colors.CYAN)} {context_data.get('context_id', 'unknown')}")
        print(f"  {colorize('Type:', Colors.CYAN)} {context_data.get('context_type', 'unknown')}")
        if context_data.get('step_number'):
            print(f"  {colorize('Step:', Colors.CYAN)} #{context_data['step_number']} ({context_data.get('reasoning_phase', 'unknown')})")
        print(f"  {colorize('Provider:', Colors.CYAN)} {context_data.get('provider', 'unknown')}")
        print(f"  {colorize('Model:', Colors.CYAN)} {context_data.get('model', 'unknown')}")
        print(f"  {colorize('Timestamp:', Colors.CYAN)} {context_data.get('timestamp', 'unknown')}")
        if context_data.get('total_tokens'):
            print(f"  {colorize('Est. Tokens:', Colors.CYAN)} {context_data['total_tokens']:,}")

        # System prompt
        if context_data.get('system_prompt'):
            print(f"\n{colorize('🎯 System Prompt', Colors.BRIGHT_BLUE)}")
            print(f"{colorize('─' * 40, Colors.BLUE)}")
            system_prompt = context_data['system_prompt']
            # Truncate if very long
            if len(system_prompt) > 2000:
                print(f"{system_prompt[:2000]}...")
                print(f"{colorize(f'[Truncated - {len(system_prompt):,} total characters]', Colors.DIM)}")
            else:
                print(system_prompt)

        # Messages/Conversation History
        if context_data.get('messages'):
            print(f"\n{colorize('💬 Conversation History', Colors.BRIGHT_BLUE)} ({len(context_data['messages'])} messages)")
            print(f"{colorize('─' * 40, Colors.BLUE)}")

            for i, message in enumerate(context_data['messages']):
                role = message.get('role', 'unknown')
                content = str(message.get('content', ''))

                role_color = Colors.BRIGHT_GREEN if role == 'user' else Colors.BRIGHT_YELLOW if role == 'assistant' else Colors.CYAN
                print(f"\n{colorize(f'{i+1}. {role.title()}:', role_color)}")

                # Truncate long messages
                if len(content) > 1000:
                    print(f"  {content[:1000]}...")
                    print(f"  {colorize(f'[Truncated - {len(content):,} total characters]', Colors.DIM)}")
                else:
                    print(f"  {content}")

        # Tools Available
        if context_data.get('tools'):
            print(f"\n{colorize('🔧 Tools Available', Colors.BRIGHT_BLUE)} ({len(context_data['tools'])} tools)")
            print(f"{colorize('─' * 40, Colors.BLUE)}")

            for tool in context_data['tools']:
                if isinstance(tool, dict):
                    tool_name = tool.get('name') or tool.get('function', {}).get('name', 'unknown')
                    print(f"  • {colorize(tool_name, Colors.YELLOW)}")

        # Model Parameters
        if context_data.get('model_params'):
            print(f"\n{colorize('⚙️ Model Parameters', Colors.BRIGHT_BLUE)}")
            print(f"{colorize('─' * 40, Colors.BLUE)}")
            for key, value in context_data['model_params'].items():
                print(f"  {colorize(key + ':', Colors.CYAN)} {value}")

        print(f"\n{colorize(f'📁 Source: {source}', Colors.DIM)}")

    def _display_interaction_context(self, context_data: dict, short_id: str, source: str) -> None:
        """Display interaction context data."""
        from abstractllm.utils.display import Colors, colorize

        print(f"\n{colorize('🔍 Interaction Context', Colors.BRIGHT_CYAN, bold=True)} - {colorize(short_id, Colors.WHITE)}")
        print(f"{colorize('─' * 60, Colors.CYAN)}")

        # Show available data
        print(f"\n{colorize('📋 Interaction Information', Colors.BRIGHT_BLUE)}")
        print(f"  {colorize('Query:', Colors.CYAN)} {context_data.get('query', 'N/A')}")
        print(f"  {colorize('Model:', Colors.CYAN)} {context_data.get('model', 'N/A')}")
        print(f"  {colorize('Timestamp:', Colors.CYAN)} {context_data.get('timestamp', 'N/A')}")

        if context_data.get('tools_executed'):
            print(f"\n{colorize('🔧 Tools Executed', Colors.BRIGHT_BLUE)} ({len(context_data['tools_executed'])} tools)")
            print(f"{colorize('─' * 40, Colors.BLUE)}")
            for i, tool in enumerate(context_data['tools_executed']):
                tool_name = tool.get('name', 'unknown')
                print(f"  {i+1}. {colorize(tool_name, Colors.YELLOW)}")

        # Try to get and display the actual response content (verbatim)
        try:
            from pathlib import Path
            import json

            # Extract session and interaction from source path
            source_path = Path(source)
            interaction_dir = source_path.parent

            # Look for cycle files in the same directory
            cycle_files = list(interaction_dir.glob("cycle_*.json"))
            if cycle_files:
                with open(cycle_files[0], 'r') as f:
                    cycle_data = json.load(f)
                response_content = cycle_data.get('response_content', '')

                if response_content:
                    print(f"\n{colorize('🎯 VERBATIM RESPONSE CONTENT', Colors.BRIGHT_RED, bold=True)}")
                    print(f"{colorize('─' * 60, Colors.RED)}")
                    print(f"{colorize('This is the exact content generated by the LLM:', Colors.YELLOW)}")
                    print(f"{colorize('─' * 60, Colors.RED)}")
                    print()
                    print(response_content)
                    print()
                    print(f"{colorize('─' * 60, Colors.RED)}")
        except Exception as e:
            pass  # Don't break if verbatim content unavailable

        print(f"\n{colorize('💡 Tip:', Colors.BRIGHT_CYAN)} Use /scratch {short_id} to see the ReAct reasoning trace.")
        print(f"{colorize(f'📁 Source: {source}', Colors.DIM)}")

    def _display_verbatim_context(self, context_data: dict, short_id: str, source: str) -> None:
        """Display exact verbatim context data (the EXACT payload sent to LLM)."""
        from abstractllm.utils.display import Colors, colorize

        print(f"\n{colorize('🔍 EXACT VERBATIM LLM CONTEXT', Colors.BRIGHT_CYAN, bold=True)} - {colorize(short_id, Colors.WHITE)}")
        print(f"{colorize('─' * 60, Colors.CYAN)}")

        # Context metadata
        print(f"\n{colorize('📋 Context Metadata', Colors.BRIGHT_BLUE)}")
        print(f"  {colorize('Context ID:', Colors.CYAN)} {context_data.get('context_id', 'unknown')}")
        print(f"  {colorize('Type:', Colors.CYAN)} {context_data.get('context_type', 'unknown')}")
        if context_data.get('step_number'):
            print(f"  {colorize('Step:', Colors.CYAN)} #{context_data['step_number']} ({context_data.get('reasoning_phase', 'unknown')})")
        print(f"  {colorize('Provider:', Colors.CYAN)} {context_data.get('provider', 'unknown')}")
        print(f"  {colorize('Model:', Colors.CYAN)} {context_data.get('model', 'unknown')}")
        if context_data.get('endpoint'):
            print(f"  {colorize('Endpoint:', Colors.CYAN)} {context_data['endpoint']}")
        print(f"  {colorize('Timestamp:', Colors.CYAN)} {context_data.get('timestamp', 'unknown')}")
        if context_data.get('total_chars'):
            print(f"  {colorize('Size:', Colors.CYAN)} {context_data['total_chars']:,} characters")

        # EXACT VERBATIM CONTEXT
        verbatim_context = context_data.get('verbatim_context', '')
        if verbatim_context:
            print(f"\n{colorize('🎯 EXACT VERBATIM PAYLOAD SENT TO LLM', Colors.BRIGHT_RED, bold=True)}")
            print(f"{colorize('─' * 60, Colors.RED)}")
            print(f"{colorize('⚠️  This is the EXACT content sent to the LLM - no formatting applied', Colors.YELLOW)}")
            print(f"{colorize('─' * 60, Colors.RED)}")

            # Display the EXACT verbatim context
            print(verbatim_context)

            print(f"{colorize('─' * 60, Colors.RED)}")
            print(f"{colorize('END OF EXACT VERBATIM PAYLOAD', Colors.RED)}")
        else:
            print(f"\n{colorize('❌ No verbatim context available', Colors.RED)}")

        print(f"\n{colorize(f'📁 Source: {source}', Colors.DIM)}")

    def _show_specific_context_unified(self, interaction_id: str) -> None:
        """Show context using LanceDB store."""
        # Try LanceDB first
        if hasattr(self.session, 'lance_store') and self.session.lance_store:
            try:
                # Use exact ID lookup to find the interaction
                context_data = self.session.lance_store.get_interaction_by_id(interaction_id)
                if context_data:
                    context = context_data.get('context_verbatim', '')

                    if context:
                        print(f"\n{colorize(f'{Symbols.TARGET} LLM Context for Interaction', Colors.BRIGHT_YELLOW, bold=True)} - {colorize(interaction_id[:8], Colors.WHITE)}")
                        print(create_divider(80, "─", Colors.YELLOW))
                        print(f"\n{context}\n")
                        print(create_divider(80, "─", Colors.YELLOW))

                        timestamp = context_data.get('timestamp', 'Unknown time')
                        query = context_data.get('query', 'Unknown query')

                        # Extract real query from context_verbatim if it contains session info
                        if context and ('Session:' in query or 'Current Reasoning' in query):
                            # Extract user query from context (format: "User: <query>")
                            if 'User:' in context:
                                lines = context.split('\n')
                                # Find the last "User:" line (most recent query)
                                for line in reversed(lines):
                                    if line.strip().startswith('User:'):
                                        extracted_query = line.replace('User:', '').strip()
                                        if extracted_query:
                                            query = extracted_query
                                            break

                        print(f"\n{colorize('Timestamp:', Colors.DIM)} {timestamp}")
                        print(f"{colorize('Query:', Colors.DIM)} {query[:100]}{'...' if len(query) > 100 else ''}")
                        return
            except Exception as e:
                logger.debug(f"LanceDB context lookup failed: {e}")

        # Fallback to unified storage system
        try:
            self._show_specific_context(interaction_id)
            return
        except Exception:
            pass

        display_error(f"Context not found for interaction ID: {interaction_id}")
        print(f"\n{colorize('💡 Tip: Use /stats to see available interactions', Colors.DIM)}")

    def _cmd_facts_unified(self, args: List[str]) -> None:
        """Show extracted facts for a specific interaction."""
        if not args:
            # Show general memory facts (existing behavior)
            self._cmd_facts([])
            return

        interaction_id = args[0]
        if interaction_id.startswith('cycle_'):
            interaction_id = interaction_id[6:]  # Remove 'cycle_' prefix

        # Try LanceDB first
        if hasattr(self.session, 'lance_store') and self.session.lance_store:
            try:
                context_data = self.session.lance_store.get_interaction_by_id(interaction_id)
                if context_data:
                    facts = context_data.get('facts_extracted', [])

                    if not facts:
                        display_info(f"No facts extracted for interaction {interaction_id}")
                        return

                    print(f"\n{colorize(f'{Symbols.KEY} Extracted Facts for Interaction', Colors.BRIGHT_YELLOW, bold=True)} - {colorize(interaction_id[:8], Colors.WHITE)}")
                    timestamp = context_data.get('timestamp', 'Unknown time')
                    print(f"{colorize(f'From: {timestamp}', Colors.DIM, italic=True)}")
                    print(create_divider(60, "─", Colors.YELLOW))

                    for i, fact in enumerate(facts, 1):
                        fact_text = fact if isinstance(fact, str) else str(fact)
                        print(f"  {colorize(f'{i}.', Colors.BRIGHT_YELLOW)} {fact_text}")

                    print(f"\n{colorize(f'Total: {len(facts)} facts extracted', Colors.BRIGHT_YELLOW)}")
                    return

            except Exception as e:
                logger.debug(f"LanceDB facts lookup failed: {e}")

        display_info(f"No facts found for interaction {interaction_id}")

    def _cmd_seed(self, args: List[str]) -> None:
        """Show or set random seed for deterministic generation."""
        from abstractllm.interface import ModelParameter

        if not args:
            # Show current seed
            current_seed = self.session._provider.config_manager.get_param(ModelParameter.SEED)
            if current_seed is not None:
                print(f"{colorize('🎲 Current seed:', Colors.BRIGHT_CYAN)} {colorize(str(current_seed), Colors.WHITE)}")
                print(f"{colorize('Mode:', Colors.DIM)} Deterministic generation")
            else:
                print(f"{colorize('🎲 Current seed:', Colors.BRIGHT_CYAN)} {colorize('None (random)', Colors.WHITE)}")
                print(f"{colorize('Mode:', Colors.DIM)} Random generation")
            return

        seed_arg = args[0].lower()

        if seed_arg in ["random", "none", "null", "off"]:
            # Disable seed (random generation) and restore original temperature
            self.session._provider.config_manager.update_config({
                ModelParameter.SEED: None,
                ModelParameter.TEMPERATURE: 0.7  # Restore CLI default
            })
            display_success(f"🎲 Seed disabled - switched to random generation")
            print(f"{colorize('🔧 Restored:', Colors.BRIGHT_CYAN)} Temperature reset to 0.7 (CLI default)")
        else:
            # Set specific seed
            try:
                seed_value = int(seed_arg)

                # Get current temperature to check if it's too high for determinism
                current_temp = self.session._provider.config_manager.get_param(ModelParameter.TEMPERATURE)

                # Set seed
                self.session._provider.config_manager.update_config({ModelParameter.SEED: seed_value})

                # For true determinism, also set temperature to 0
                if current_temp is None or current_temp > 0.1:
                    self.session._provider.config_manager.update_config({ModelParameter.TEMPERATURE: 0.0})
                    display_success(f"🎲 Seed set to {seed_value} and temperature set to 0.0 for deterministic generation")
                    print(f"{colorize('🔧 Auto-adjustment:', Colors.BRIGHT_CYAN)} Temperature changed from {current_temp} to 0.0 for true determinism")
                else:
                    display_success(f"🎲 Seed set to {seed_value} - deterministic generation enabled")

                # Show tips about deterministic generation
                print(f"{colorize('💡 Tip:', Colors.BRIGHT_YELLOW)} With seed={seed_value} + temperature=0.0, identical prompts will produce identical outputs")
                print(f"{colorize('📝 Note:', Colors.DIM)} Use '/seed random' to restore random generation and original temperature")
            except ValueError:
                display_error(f"Invalid seed value: '{args[0]}'. Use a number or 'random'")
                print(f"{colorize('Usage:', Colors.DIM)} /seed 42, /seed random")

    def _cmd_temperature(self, args: List[str]) -> None:
        """Show or set temperature for generation randomness."""
        from abstractllm.interface import ModelParameter

        if not args:
            # Show current temperature
            current_temp = self.session._provider.config_manager.get_param(ModelParameter.TEMPERATURE)
            if current_temp is not None:
                print(f"{colorize('🌡️ Current temperature:', Colors.BRIGHT_CYAN)} {colorize(str(current_temp), Colors.WHITE)}")
                if current_temp == 0.0:
                    print(f"{colorize('Mode:', Colors.DIM)} Deterministic generation (no randomness)")
                elif current_temp < 0.3:
                    print(f"{colorize('Mode:', Colors.DIM)} Low randomness (focused)")
                elif current_temp < 0.7:
                    print(f"{colorize('Mode:', Colors.DIM)} Medium randomness (balanced)")
                else:
                    print(f"{colorize('Mode:', Colors.DIM)} High randomness (creative)")
            else:
                print(f"{colorize('🌡️ Current temperature:', Colors.BRIGHT_CYAN)} {colorize('Not set (using provider default)', Colors.WHITE)}")
            return

        # Set temperature
        try:
            temp_value = float(args[0])

            # Validate temperature range
            if temp_value < 0.0 or temp_value > 2.0:
                display_error(f"Temperature must be between 0.0 and 2.0, got {temp_value}")
                print(f"{colorize('Valid range:', Colors.DIM)} 0.0 (deterministic) to 2.0 (very creative)")
                return

            # Update temperature
            self.session._provider.config_manager.update_config({ModelParameter.TEMPERATURE: temp_value})

            # Provide feedback about the change
            if temp_value == 0.0:
                display_success(f"🌡️ Temperature set to {temp_value} - deterministic generation")
                print(f"{colorize('💡 Tip:', Colors.BRIGHT_YELLOW)} Use with /seed for fully reproducible outputs")
            elif temp_value < 0.3:
                display_success(f"🌡️ Temperature set to {temp_value} - low randomness (focused responses)")
            elif temp_value < 0.7:
                display_success(f"🌡️ Temperature set to {temp_value} - medium randomness (balanced)")
            else:
                display_success(f"🌡️ Temperature set to {temp_value} - high randomness (creative responses)")

            print(f"{colorize('📝 Note:', Colors.DIM)} Higher values = more creative but less predictable")

        except ValueError:
            display_error(f"Invalid temperature value: '{args[0]}'. Use a decimal number")
            print(f"{colorize('Usage:', Colors.DIM)} /temperature 0.7, /temperature 0.0 (deterministic)")
            print(f"{colorize('Examples:', Colors.DIM)} 0.0=deterministic, 0.3=focused, 0.7=balanced, 1.0=creative")

    def _cmd_memory_facts(self, args: List[str]) -> None:
        """Configure facts inclusion in memory context."""
        if not args:
            # Show current settings
            max_facts = getattr(self.session, 'memory_facts_max', 10)
            min_confidence = getattr(self.session, 'memory_facts_min_confidence', 0.3)
            min_occurrences = getattr(self.session, 'memory_facts_min_occurrences', 1)

            print(f"{colorize('📚 Memory Facts Configuration:', Colors.BRIGHT_CYAN)}")
            print(f"{colorize('─' * 50, Colors.DIM)}")
            print(f"{colorize('Max facts:', Colors.WHITE)} {max_facts}")
            print(f"{colorize('Min confidence:', Colors.WHITE)} {min_confidence}")
            print(f"{colorize('Min occurrences:', Colors.WHITE)} {min_occurrences}")
            print(f"{colorize('─' * 50, Colors.DIM)}")
            print(f"{colorize('💡 Higher confidence = more reliable facts', Colors.DIM)}")
            print(f"{colorize('💡 Higher occurrences = frequently mentioned facts', Colors.DIM)}")
            return

        if len(args) != 3:
            display_error("Usage: /memory-facts <max-facts> <min-confidence> <min-occurrences>")
            print(f"{colorize('Example:', Colors.DIM)} /memory-facts 15 0.4 2")
            print(f"{colorize('Ranges:', Colors.DIM)} max-facts: 1-50, confidence: 0.0-1.0, occurrences: 1+")
            return

        try:
            max_facts = int(args[0])
            min_confidence = float(args[1])
            min_occurrences = int(args[2])

            # Validate ranges
            if not (1 <= max_facts <= 50):
                display_error(f"Max facts must be between 1 and 50, got {max_facts}")
                return
            if not (0.0 <= min_confidence <= 1.0):
                display_error(f"Min confidence must be between 0.0 and 1.0, got {min_confidence}")
                return
            if min_occurrences < 1:
                display_error(f"Min occurrences must be at least 1, got {min_occurrences}")
                return

            # Update session configuration
            self.session.memory_facts_max = max_facts
            self.session.memory_facts_min_confidence = min_confidence
            self.session.memory_facts_min_occurrences = min_occurrences

            display_success(f"📚 Memory facts configuration updated:")
            print(f"{colorize('Max facts:', Colors.WHITE)} {max_facts}")
            print(f"{colorize('Min confidence:', Colors.WHITE)} {min_confidence}")
            print(f"{colorize('Min occurrences:', Colors.WHITE)} {min_occurrences}")

            if max_facts > 20:
                print(f"{colorize('📝 Note:', Colors.BRIGHT_YELLOW)} High fact count may use more context tokens")
            if min_confidence > 0.7:
                print(f"{colorize('📝 Note:', Colors.BRIGHT_YELLOW)} High confidence threshold may exclude useful facts")

        except ValueError:
            display_error("Invalid parameter values. Use integers for counts, decimal for confidence")
            print(f"{colorize('Usage:', Colors.DIM)} /memory-facts 10 0.3 1")

    def _cmd_system(self, args: List[str]) -> None:
        """Show or set system prompt."""
        if not args:
            # Show current system prompt
            if hasattr(self.session, 'system_prompt') and self.session.system_prompt:
                print(f"{colorize('🎯 Current system prompt:', Colors.BRIGHT_CYAN)}")
                print(f"{colorize('─' * 50, Colors.DIM)}")
                print(f"{colorize(self.session.system_prompt, Colors.WHITE)}")
                print(f"{colorize('─' * 50, Colors.DIM)}")
                print(f"{colorize('Length:', Colors.DIM)} {len(self.session.system_prompt)} characters")
            else:
                print(f"{colorize('🎯 System prompt:', Colors.BRIGHT_CYAN)} {colorize('Not set (using default)', Colors.WHITE)}")
            return

        # Set new system prompt (join all args to handle multi-word prompts)
        new_prompt = ' '.join(args)

        if not new_prompt.strip():
            display_error("System prompt cannot be empty")
            print(f"{colorize('Usage:', Colors.DIM)} /system Your custom system prompt here")
            return

        # Update system prompt
        self.session.system_prompt = new_prompt

        # Provide feedback
        display_success(f"🎯 System prompt updated")
        print(f"{colorize('New prompt (first 100 chars):', Colors.DIM)} {new_prompt[:100]}{'...' if len(new_prompt) > 100 else ''}")
        print(f"{colorize('Length:', Colors.DIM)} {len(new_prompt)} characters")
        print(f"{colorize('💡 Tip:', Colors.BRIGHT_YELLOW)} System prompt affects all future messages in this session")

    def _cmd_stream(self, args: List[str]) -> None:
        """Toggle or show streaming mode for the session."""
        if not args:
            # Show current streaming setting
            current_setting = getattr(self.session, 'default_streaming', False)
            mode_text = colorize("ENABLED", Colors.BRIGHT_GREEN) if current_setting else colorize("DISABLED", Colors.BRIGHT_RED)
            print(f"{colorize('🔄 Streaming mode:', Colors.BRIGHT_CYAN)} {mode_text}")

            if current_setting:
                print(f"{colorize('Behavior:', Colors.DIM)} Responses will stream progressively with real-time tool execution")
                print(f"{colorize('ReAct Loop:', Colors.DIM)} Tool calls and results stream as they execute")
            else:
                print(f"{colorize('Behavior:', Colors.DIM)} Responses will be delivered as complete messages")
                print(f"{colorize('ReAct Loop:', Colors.DIM)} Tool execution completes before showing final result")

            print(f"{colorize('💡 Toggle:', Colors.BRIGHT_YELLOW)} Use '/stream on' or '/stream off' to change")
            return

        # Parse argument
        setting = args[0].lower()

        if setting in ['on', 'true', '1', 'enable', 'enabled']:
            self.session.default_streaming = True
            display_success("🔄 Streaming mode enabled")
            print(f"{colorize('Behavior:', Colors.DIM)} Future responses will stream progressively")
            print(f"{colorize('ReAct Loops:', Colors.DIM)} Tool execution will be visible in real-time")
            print(f"{colorize('Override:', Colors.DIM)} You can still use explicit stream=True/False in code")
        elif setting in ['off', 'false', '0', 'disable', 'disabled']:
            self.session.default_streaming = False
            display_success("🔄 Streaming mode disabled")
            print(f"{colorize('Behavior:', Colors.DIM)} Future responses will be delivered as complete messages")
            print(f"{colorize('ReAct Loops:', Colors.DIM)} Tool execution will complete before showing results")
            print(f"{colorize('Override:', Colors.DIM)} You can still use explicit stream=True/False in code")
        else:
            display_error(f"Invalid streaming setting: '{setting}'")
            print(f"{colorize('Usage:', Colors.DIM)} /stream [on|off]")
            print(f"{colorize('Examples:', Colors.DIM)} /stream on, /stream off, /stream (to show current)")

    def _cmd_tools(self, args: List[str]) -> None:
        """Show registered tools or toggle a specific tool."""
        # Check if tools functionality is available
        try:
            from abstractllm.tools import ToolDefinition
            from abstractllm.session import TOOLS_AVAILABLE
            if not TOOLS_AVAILABLE:
                display_error("Tools functionality is not available. Install required dependencies.")
                return
        except ImportError:
            display_error("Tools functionality is not available. Install required dependencies.")
            return

        if not args:
            # Show all registered tools
            if not hasattr(self.session, 'tools') or not self.session.tools:
                print(f"{colorize('🔧 Registered tools:', Colors.BRIGHT_CYAN)} {colorize('None', Colors.WHITE)}")
                print(f"{colorize('💡 Tip:', Colors.BRIGHT_YELLOW)} Add tools using session.add_tool() or the tools parameter")
                return

            print(f"{colorize('🔧 Registered tools:', Colors.BRIGHT_CYAN)} {colorize(str(len(self.session.tools)), Colors.WHITE)}")
            print(f"{colorize('─' * 60, Colors.DIM)}")

            for i, tool in enumerate(self.session.tools, 1):
                # Check if tool is active (present in both tools list and implementations)
                is_active = hasattr(tool, 'name') and tool.name in getattr(self.session, '_tool_implementations', {})
                status_icon = "✅" if is_active else "❌"
                status_text = colorize("ACTIVE", Colors.BRIGHT_GREEN) if is_active else colorize("INACTIVE", Colors.BRIGHT_RED)

                tool_name = getattr(tool, 'name', 'Unknown')
                tool_desc = getattr(tool, 'description', 'No description')

                print(f"  {i}. {status_icon} {colorize(tool_name, Colors.BRIGHT_WHITE)} - {status_text}")
                print(f"     {colorize(tool_desc, Colors.DIM)}")

                # Show parameters if available
                if hasattr(tool, 'parameters') and tool.parameters:
                    param_names = list(tool.parameters.keys()) if isinstance(tool.parameters, dict) else []
                    if param_names:
                        params_str = ", ".join(param_names[:3])
                        if len(param_names) > 3:
                            params_str += f", ... (+{len(param_names) - 3} more)"
                        print(f"     {colorize('Parameters:', Colors.DIM)} {params_str}")
                print()

            print(f"{colorize('💡 Usage:', Colors.BRIGHT_YELLOW)} /tools <tool_name> to toggle a specific tool")
            return

        # Toggle specific tool
        tool_name = args[0]

        if not hasattr(self.session, 'tools') or not self.session.tools:
            display_error(f"No tools registered. Cannot toggle '{tool_name}'")
            return

        # Find the tool by name
        target_tool = None
        for tool in self.session.tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                target_tool = tool
                break

        if not target_tool:
            display_error(f"Tool '{tool_name}' not found")
            available_tools = [getattr(t, 'name', 'Unknown') for t in self.session.tools if hasattr(t, 'name')]
            if available_tools:
                print(f"{colorize('Available tools:', Colors.DIM)} {', '.join(available_tools)}")
            return

        # Check current status and toggle
        is_currently_active = tool_name in getattr(self.session, '_tool_implementations', {})

        if is_currently_active:
            # Deactivate tool: remove from implementations but keep in tools list
            if hasattr(self.session, '_tool_implementations') and tool_name in self.session._tool_implementations:
                del self.session._tool_implementations[tool_name]
            display_success(f"🔧 Tool '{tool_name}' deactivated")
            print(f"{colorize('Status:', Colors.DIM)} Tool is now inactive and won't be available for use")
        else:
            # Reactivate tool: add back to implementations if we have the definition
            if hasattr(target_tool, 'function') and callable(target_tool.function):
                # Re-register the function implementation
                if not hasattr(self.session, '_tool_implementations'):
                    self.session._tool_implementations = {}
                self.session._tool_implementations[tool_name] = target_tool.function
                display_success(f"🔧 Tool '{tool_name}' activated")
                print(f"{colorize('Status:', Colors.DIM)} Tool is now active and available for use")
            else:
                display_error(f"Cannot reactivate '{tool_name}': original function not available")
                print(f"{colorize('Note:', Colors.DIM)} Tool definition exists but function implementation is missing")

    def _cmd_search(self, args: List[str]) -> None:
        """Search interactions using semantic similarity with optional filters.

        Usage: /search <query> [--user <id>] [--from <date>] [--to <date>] [--limit <n>]

        Examples:
          /search debugging cache problems
          /search "machine learning" --from 2025-09-01 --limit 5
          /search optimization --user alice
        """
        if not args:
            display_error("Usage: /search <query> [--user <id>] [--from <date>] [--to <date>] [--limit <n>]")
            return

        # Check if LanceDB is available
        if not hasattr(self.session, 'lance_store') or not self.session.lance_store:
            display_error("LanceDB search is not available. Using legacy search...")
            # Fallback to existing context search
            self._cmd_context(args)
            return

        try:
            # Parse arguments
            query_parts = []
            filters = {}
            limit = 10
            i = 0

            while i < len(args):
                if args[i] == '--user' and i + 1 < len(args):
                    filters['user_id'] = args[i + 1]
                    i += 2
                elif args[i] == '--from' and i + 1 < len(args):
                    try:
                        filters['start_time'] = datetime.fromisoformat(args[i + 1])
                    except ValueError:
                        display_error(f"Invalid date format: {args[i + 1]}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
                        return
                    i += 2
                elif args[i] == '--to' and i + 1 < len(args):
                    try:
                        filters['end_time'] = datetime.fromisoformat(args[i + 1])
                    except ValueError:
                        display_error(f"Invalid date format: {args[i + 1]}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
                        return
                    i += 2
                elif args[i] == '--limit' and i + 1 < len(args):
                    try:
                        limit = int(args[i + 1])
                    except ValueError:
                        display_error(f"Invalid limit: {args[i + 1]}")
                        return
                    i += 2
                else:
                    query_parts.append(args[i])
                    i += 1

            if not query_parts:
                display_error("No search query provided")
                return

            query_text = ' '.join(query_parts)

            # Perform semantic search
            print(f"{colorize('🔍 Searching for:', Colors.BRIGHT_CYAN)} {colorize(query_text, Colors.WHITE)}")
            if filters:
                filter_desc = []
                if 'user_id' in filters:
                    filter_desc.append(f"user: {filters['user_id']}")
                if 'start_time' in filters:
                    filter_desc.append(f"from: {filters['start_time'].strftime('%Y-%m-%d')}")
                if 'end_time' in filters:
                    filter_desc.append(f"to: {filters['end_time'].strftime('%Y-%m-%d')}")
                print(f"{colorize('🔧 Filters:', Colors.DIM)} {', '.join(filter_desc)}")

            results = self.session.lance_store.semantic_search(query_text, limit=limit, filters=filters)

            if not results:
                print(f"{colorize('📭 No results found', Colors.YELLOW)}")
                return

            print(f"{colorize('📄 Found:', Colors.BRIGHT_GREEN)} {colorize(str(len(results)), Colors.WHITE)} results")
            print()

            for i, result in enumerate(results, 1):
                timestamp = result.get('timestamp', 'Unknown time')
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass

                similarity = result.get('_distance', 0)
                # Convert distance to similarity (lower distance = higher similarity)
                similarity_pct = max(0, (2.0 - similarity) / 2.0 * 100) if similarity else 0

                print(f"{colorize(f'{i}.', Colors.BRIGHT_BLUE)} {colorize(timestamp, Colors.DIM)} "
                      f"{colorize(f'({similarity_pct:.1f}% similar)', Colors.GREEN)}")

                # Extract real query from context_verbatim if it contains session info
                query = result.get('query', 'Unknown query')
                if 'context_verbatim' in result and ('Session:' in query or query == 'Unknown query'):
                    context = result['context_verbatim']
                    # Extract user query from context (format: "User: <query>")
                    if 'User:' in context:
                        lines = context.split('\n')
                        # Find the last "User:" line (most recent query)
                        for line in reversed(lines):
                            if line.strip().startswith('User:'):
                                extracted_query = line.replace('User:', '').strip()
                                if extracted_query:
                                    query = extracted_query
                                    break

                if len(query) > 100:
                    query = query[:97] + "..."
                print(f"   {colorize('Q:', Colors.CYAN)} {query}")

                # Show best matching chunks instead of truncated response
                response = result.get('response', 'No response')
                if response == 'Processing...':
                    response = "[Response was generated but not captured in storage]"
                    print(f"   {colorize('A:', Colors.MAGENTA)} {response}")
                elif hasattr(self.session, 'lance_store') and self.session.lance_store:
                    # Use chunking to show relevant parts
                    try:
                        chunks = self.session.lance_store._chunk_response_content(response, max_chunk_size=250)
                        best_chunks = self.session.lance_store._find_best_matching_chunks(query_text, chunks, max_chunks=2)

                        if best_chunks:
                            print(f"   {colorize('A:', Colors.MAGENTA)} {colorize('(relevant parts)', Colors.DIM)}")
                            for chunk in best_chunks:
                                chunk_preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                                print(f"   {colorize('   →', Colors.YELLOW)} {chunk_preview}")
                        else:
                            # Fallback to truncated response
                            if len(response) > 150:
                                response = response[:147] + "..."
                            print(f"   {colorize('A:', Colors.MAGENTA)} {response}")
                    except Exception as e:
                        # Fallback to original behavior
                        if len(response) > 150:
                            response = response[:147] + "..."
                        print(f"   {colorize('A:', Colors.MAGENTA)} {response}")
                else:
                    # No LanceDB available - use original behavior
                    if len(response) > 150:
                        response = response[:147] + "..."
                    print(f"   {colorize('A:', Colors.MAGENTA)} {response}")
                print()

        except Exception as e:
            display_error(f"Search failed: {e}")

    def _cmd_timeframe(self, args: List[str]) -> None:
        """Search interactions within a specific timeframe.

        Usage: /timeframe <start> <end> [user_id]

        Examples:
          /timeframe 2025-09-15T10:00 2025-09-15T12:15
          /timeframe 2025-09-15 2025-09-16 alice
        """
        if len(args) < 2:
            display_error("Usage: /timeframe <start> <end> [user_id]")
            display_info("Date formats: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
            return

        # Check if LanceDB is available
        if not hasattr(self.session, 'lance_store') or not self.session.lance_store:
            display_error("LanceDB timeframe search is not available")
            return

        try:
            # Parse start and end times
            start_time = datetime.fromisoformat(args[0])
            end_time = datetime.fromisoformat(args[1])
            user_id = args[2] if len(args) > 2 else None

            print(f"{colorize('📅 Searching timeframe:', Colors.BRIGHT_CYAN)}")
            print(f"   {colorize('From:', Colors.DIM)} {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   {colorize('To:', Colors.DIM)} {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if user_id:
                print(f"   {colorize('User:', Colors.DIM)} {user_id}")

            # Perform timeframe search
            results = self.session.lance_store.search_by_timeframe(start_time, end_time, user_id)

            if results.empty:
                print(f"{colorize('📭 No interactions found in this timeframe', Colors.YELLOW)}")
                return

            print(f"{colorize('📄 Found:', Colors.BRIGHT_GREEN)} {colorize(str(len(results)), Colors.WHITE)} interactions")
            print()

            for _, result in results.iterrows():
                timestamp = result.get('timestamp', 'Unknown time')
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass

                interaction_id = result.get('interaction_id', 'Unknown ID')[:8]

                print(f"{colorize('•', Colors.BRIGHT_BLUE)} {colorize(timestamp, Colors.DIM)} "
                      f"{colorize(f'[{interaction_id}]', Colors.GREEN)}")

                query = result.get('query', 'Unknown query')
                if len(query) > 100:
                    query = query[:97] + "..."
                print(f"   {query}")
                print()

        except ValueError as e:
            display_error(f"Invalid date format: {e}")
            display_info("Use format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")
        except Exception as e:
            display_error(f"Timeframe search failed: {e}")

    def _cmd_similar(self, args: List[str]) -> None:
        """Find interactions similar to the given text.

        Usage: /similar <text> [--limit <n>]

        Examples:
          /similar "how to optimize database queries"
          /similar debugging --limit 3
        """
        if not args:
            display_error("Usage: /similar <text> [--limit <n>]")
            return

        # Check if LanceDB is available
        if not hasattr(self.session, 'lance_store') or not self.session.lance_store:
            display_error("LanceDB similarity search is not available")
            return

        try:
            # Parse arguments
            text_parts = []
            limit = 5
            i = 0

            while i < len(args):
                if args[i] == '--limit' and i + 1 < len(args):
                    try:
                        limit = int(args[i + 1])
                    except ValueError:
                        display_error(f"Invalid limit: {args[i + 1]}")
                        return
                    i += 2
                else:
                    text_parts.append(args[i])
                    i += 1

            if not text_parts:
                display_error("No text provided for similarity search")
                return

            search_text = ' '.join(text_parts)

            print(f"{colorize('🔍 Finding similar to:', Colors.BRIGHT_CYAN)} {colorize(search_text, Colors.WHITE)}")

            # Use combined search to get both interactions and ReAct cycles
            results = self.session.lance_store.search_combined(search_text, limit=limit)

            interactions = results.get('interactions', [])
            react_cycles = results.get('react_cycles', [])

            if not interactions and not react_cycles:
                print(f"{colorize('📭 No similar interactions found', Colors.YELLOW)}")
                return

            total_results = len(interactions) + len(react_cycles)
            print(f"{colorize('📄 Found:', Colors.BRIGHT_GREEN)} {colorize(str(total_results), Colors.WHITE)} similar items")
            print()

            # Display interactions
            if interactions:
                print(f"{colorize('💬 Similar Interactions:', Colors.BRIGHT_MAGENTA)}")
                for i, result in enumerate(interactions, 1):
                    timestamp = result.get('timestamp', 'Unknown time')
                    if isinstance(timestamp, str):
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pass

                    similarity = result.get('_distance', 0)
                    similarity_pct = max(0, (2.0 - similarity) / 2.0 * 100) if similarity else 0

                    print(f"   {colorize(f'{i}.', Colors.BRIGHT_BLUE)} {colorize(timestamp, Colors.DIM)} "
                          f"{colorize(f'({similarity_pct:.1f}% similar)', Colors.GREEN)}")

                    query = result.get('query', 'Unknown query')
                    if len(query) > 80:
                        query = query[:77] + "..."
                    print(f"      {query}")
                print()

            # Display ReAct cycles
            if react_cycles:
                print(f"{colorize('🧠 Similar ReAct Reasoning:', Colors.BRIGHT_CYAN)}")
                for i, result in enumerate(react_cycles, 1):
                    timestamp = result.get('timestamp', 'Unknown time')
                    if isinstance(timestamp, str):
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pass

                    similarity = result.get('_distance', 0)
                    similarity_pct = max(0, (2.0 - similarity) / 2.0 * 100) if similarity else 0

                    print(f"   {colorize(f'{i}.', Colors.BRIGHT_BLUE)} {colorize(timestamp, Colors.DIM)} "
                          f"{colorize(f'({similarity_pct:.1f}% similar)', Colors.GREEN)}")

                    react_id = result.get('react_id', 'Unknown')[:8]
                    print(f"      ReAct cycle: {react_id}")
                print()

        except Exception as e:
            display_error(f"Similarity search failed: {e}")

    def _cmd_values(self, args: List[str]) -> None:
        """Show value resonance for interactions. Usage: /values [interaction_id]"""

        # Try to get cognitive enhancer, or create ValueResonance directly
        value_evaluator = None

        if hasattr(self.session, '_cognitive_enhancer') and self.session._cognitive_enhancer:
            # Use existing cognitive enhancer
            enhancer = self.session._cognitive_enhancer
            if enhancer.value_evaluator and enhancer.value_evaluator.is_available():
                value_evaluator = enhancer.value_evaluator

        if not value_evaluator:
            # Create ValueResonance directly with default values
            try:
                from abstractllm.cognitive import ValueResonance
                value_evaluator = ValueResonance(
                    llm_provider="ollama",
                    model="granite3.3:2b"
                )
                if not value_evaluator.is_available():
                    display_error("ValueResonance evaluator not available. Check that granite3.3:2b is accessible via ollama.")
                    return
                print(f"{colorize('ℹ️ Using default ValueResonance with standard values', Colors.BLUE)}")
            except Exception as e:
                display_error(f"Could not initialize ValueResonance: {e}")
                display_info("Make sure granite3.3:2b is available via ollama")
                return

        try:
            if args and len(args) > 0:
                # Show value resonance for specific interaction
                interaction_id = args[0]

                if hasattr(self.session, 'lance_store') and self.session.lance_store:
                    # Try to find the interaction in LanceDB
                    try:
                        interaction = self.session.lance_store.get_interaction(interaction_id)
                        if interaction:
                            print(f"{colorize('🎯 Value Resonance for Interaction:', Colors.BRIGHT_CYAN)} {colorize(interaction_id[:8], Colors.WHITE)}")

                            # Create interaction content for evaluation
                            interaction_content = f"User: {interaction.get('query', 'Unknown query')}\nAssistant: {interaction.get('response', 'Unknown response')}"

                            # Evaluate value resonance
                            assessment = value_evaluator.evaluate_interaction(interaction_content, "specific interaction")

                            self._display_value_assessment(assessment)
                            return
                        else:
                            display_error(f"Interaction {interaction_id} not found in session history.")
                            return
                    except Exception as e:
                        display_error(f"Error retrieving interaction: {e}")
                        return
                else:
                    # Fallback: search in session messages for the interaction
                    if hasattr(self.session, 'messages') and self.session.messages:
                        # Try to find interaction by searching recent messages
                        display_error(f"Interaction ID lookup not available. Use /values without ID for conversation summary.")
                        return
            else:
                # Show value resonance for entire conversation
                print(f"{colorize('📊 Value Resonance for Entire Conversation', Colors.BRIGHT_CYAN)}")

                if hasattr(self.session, 'messages') and self.session.messages:
                    # Create conversation content
                    conversation_messages = []
                    for msg in self.session.messages:
                        if hasattr(msg, 'role') and hasattr(msg, 'content'):
                            if msg.role != 'system':  # Skip system messages
                                conversation_messages.append(f"{msg.role.title()}: {msg.content}")

                    if not conversation_messages:
                        display_info("No conversation content to analyze.")
                        return

                    conversation_content = "\n\n".join(conversation_messages)

                    # Evaluate overall conversation resonance
                    assessment = value_evaluator.evaluate_interaction(conversation_content, "full conversation")

                    self._display_value_assessment(assessment)
                else:
                    display_info("No conversation history available for analysis.")

        except Exception as e:
            display_error(f"Value resonance analysis failed: {e}")

    def _display_value_assessment(self, assessment) -> None:
        """Display a value assessment in a formatted way"""
        try:
            # Show overall resonance level
            level = assessment.get_resonance_level()
            resonance_score = assessment.overall_resonance

            # Color code the resonance level
            if resonance_score >= 0.7:
                level_color = Colors.BRIGHT_GREEN
            elif resonance_score >= 0.3:
                level_color = Colors.GREEN
            elif resonance_score >= -0.3:
                level_color = Colors.YELLOW
            elif resonance_score >= -0.7:
                level_color = Colors.RED
            else:
                level_color = Colors.BRIGHT_RED

            print(f"\n{colorize('📈 Overall Resonance:', Colors.BRIGHT_BLUE)} {colorize(f'{resonance_score:+.2f}', level_color)} ({colorize(level, level_color)})")
            print()

            # Show individual value evaluations
            print(f"{colorize('🎯 Individual Value Scores:', Colors.BRIGHT_BLUE)}")
            for evaluation in assessment.evaluations:
                formatted_output = evaluation.format_output()

                # Extract score for color coding
                score = evaluation.score
                if score >= 0.5:
                    score_color = Colors.GREEN
                elif score >= 0.0:
                    score_color = Colors.YELLOW
                else:
                    score_color = Colors.RED

                # Display with color coding
                parts = formatted_output.split(' reason : ')
                if len(parts) == 2:
                    value_and_score = parts[0]
                    reasoning = parts[1]
                    print(f"  {colorize(value_and_score, score_color)} reason : {reasoning}")
                else:
                    print(f"  {formatted_output}")

            print()

        except Exception as e:
            display_error(f"Error displaying value assessment: {e}")

    def _cmd_exit(self, args: List[str]) -> None:
        """Exit interactive mode."""
        display_success("Goodbye!")
        # Use a custom exception to differentiate from Ctrl+C
        raise SystemExit(0)  # Will be caught by interactive mode


def create_command_processor(session, display_func=None) -> CommandProcessor:
    """Create a command processor for the session."""
    return CommandProcessor(session, display_func)