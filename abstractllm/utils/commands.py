"""
Slash command system for alma_simple.py interactive mode.

Provides a comprehensive command interface for memory management,
session control, and agent interaction.
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
                ("/context", "Show full context sent to LLM"),
                ("/seed [number|random]", "Set/show random seed for deterministic generation"),
                ("/temperature, /temp", "Set/show temperature for generation randomness"),
                ("/clear", "Clear conversation history"),
                ("/reset", "Reset entire session"),
                ("/status", "Show session status"),
                ("/stats", "Show detailed statistics"),
                ("/config", "Show current configuration")
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
            "/save my_session.pkl",
            "/load my_session.pkl",
            "/memory 16384",
            "/temperature 0.3",
            "/working",
            "/facts machine learning",
            "/links",
            "/seed 42",
            "/last 3",
            "/context",
            "/export memory_backup.json"
        ]
        for example in examples:
            print(f"  {colorize(example, Colors.BRIGHT_BLUE)}")
        
        # Add spacing after help for better readability
    
    def _cmd_memory(self, args: List[str]) -> None:
        """Show memory system insights or set max tokens."""
        # Check if setting max tokens
        if args and args[0].isdigit():
            new_max_tokens = int(args[0])
            # Set max tokens in session config
            if hasattr(self.session, '_provider') and self.session._provider:
                if hasattr(self.session._provider, 'config_manager'):
                    from abstractllm.interface import ModelParameter
                    self.session._provider.config_manager.update_config({
                        ModelParameter.MAX_INPUT_TOKENS: new_max_tokens
                    })
                    display_success(f"Max input tokens set to {new_max_tokens:,}")
                else:
                    display_error("Provider does not support configuration changes")
            else:
                display_error("No provider available to configure")
            return

        if not hasattr(self.session, 'memory') or not self.session.memory:
            display_error("Memory system not available")
            return

        memory = self.session.memory
        try:
            stats = memory.get_statistics()

            # Debug: Check if stats is actually a dictionary
            if not isinstance(stats, dict):
                display_error(f"Memory statistics returned {type(stats).__name__} instead of dict: {str(stats)[:200]}...")
                return

        except Exception as e:
            display_error(f"Failed to get memory statistics: {str(e)}")
            return
        
        print(f"\n{colorize(f'{Symbols.BRAIN} Memory System Overview', Colors.BRIGHT_BLUE, bold=True)}")
        print(create_divider(60, "─", Colors.BLUE))

        # Context size information
        from abstractllm.utils.context_logging import get_context_logger
        logger = get_context_logger()

        # Get context usage and limits
        used_tokens = 0
        max_tokens = "Unknown"

        # Calculate used tokens from last context
        if logger.last_context:
            context_str = json.dumps(logger.last_context, ensure_ascii=False)
            char_count = len(context_str)
            # Estimate tokens (roughly 4 chars per token)
            used_tokens = char_count // 4

        # Get configured or model max tokens
        if hasattr(self.session, '_provider') and self.session._provider:
            provider = self.session._provider
            model_name = None
            user_max_tokens = None

            # Get model name and user configuration
            if hasattr(provider, 'config_manager'):
                from abstractllm.interface import ModelParameter
                model_name = provider.config_manager.get_param(ModelParameter.MODEL)
                user_max_tokens = provider.config_manager.get_param(ModelParameter.MAX_INPUT_TOKENS)

            # Determine the actual max tokens being used
            if user_max_tokens:
                max_tokens = user_max_tokens
                source = "user-configured"
            elif model_name:
                try:
                    from abstractllm.architectures.detection import get_model_capabilities
                    capabilities = get_model_capabilities(model_name)
                    if capabilities:
                        context_length = capabilities.get('context_length')
                        if context_length and isinstance(context_length, int):
                            max_tokens = context_length
                            source = "model default"
                except Exception:
                    pass

            # Display context usage in the requested format
            print(f"  {colorize('Context Usage:', Colors.BRIGHT_CYAN)}")
            if isinstance(max_tokens, int):
                usage_ratio = (used_tokens / max_tokens) * 100 if max_tokens > 0 else 0
                usage_color = Colors.GREEN if usage_ratio < 50 else Colors.YELLOW if usage_ratio < 80 else Colors.RED
                print(f"    • Tokens: {colorize(f'{used_tokens:,}', Colors.WHITE)} / {colorize(f'{max_tokens:,}', Colors.WHITE)} ({colorize(f'{usage_ratio:.1f}%', usage_color)})")
                print(f"    • Source: {colorize(source, Colors.DIM)}")
            else:
                print(f"    • Tokens: {colorize(f'{used_tokens:,}', Colors.WHITE)} / {colorize(str(max_tokens), Colors.DIM)}")

            # Show max output tokens if available
            if model_name:
                try:
                    from abstractllm.architectures.detection import get_model_capabilities
                    capabilities = get_model_capabilities(model_name)
                    if capabilities:
                        max_output = capabilities.get('max_output_tokens', 'Unknown')
                        if max_output != 'Unknown':
                            print(f"    • Max Output: {colorize(f'{max_output:,}' if isinstance(max_output, int) else max_output, Colors.WHITE)}")
                except Exception:
                    pass

            print(f"    • {colorize('Change limit:', Colors.DIM)} /mem <number>")

        print()  # Add spacing

        # Memory distribution - check if keys exist
        if 'memory_distribution' in stats and isinstance(stats['memory_distribution'], dict):
            dist = stats['memory_distribution']
            print(f"  {colorize('Working Memory:', Colors.BRIGHT_GREEN)} {dist.get('working_memory', 0)} items")
            print(f"  {colorize('Episodic Memory:', Colors.BRIGHT_GREEN)} {dist.get('episodic_memory', 0)} experiences")
        else:
            print(f"  {colorize('Working Memory:', Colors.BRIGHT_GREEN)} 0 items")
            print(f"  {colorize('Episodic Memory:', Colors.BRIGHT_GREEN)} 0 experiences")
            
        # Knowledge graph stats
        if 'knowledge_graph' in stats and isinstance(stats['knowledge_graph'], dict):
            kg_stats = stats['knowledge_graph']
            print(f"  {colorize('Knowledge Graph:', Colors.BRIGHT_GREEN)} {kg_stats.get('total_facts', 0)} facts")
        else:
            print(f"  {colorize('Knowledge Graph:', Colors.BRIGHT_GREEN)} 0 facts")
        
        # ReAct cycles
        total_cycles = stats.get('total_react_cycles', 0)
        successful_cycles = stats.get('successful_cycles', 0)
        print(f"  {colorize('ReAct Cycles:', Colors.BRIGHT_CYAN)} {total_cycles} total ({successful_cycles} successful)")
        
        # Links - check for both possible key names
        total_links = 0
        if 'link_statistics' in stats and isinstance(stats['link_statistics'], dict):
            total_links = stats['link_statistics'].get('total_links', 0)
        elif 'memory_distribution' in stats and isinstance(stats['memory_distribution'], dict):
            total_links = stats['memory_distribution'].get('total_links', 0)
            
        print(f"  {colorize('Bidirectional Links:', Colors.BRIGHT_MAGENTA)} {total_links}")
        
        # Memory health - only if available
        try:
            if hasattr(memory, 'get_memory_health_report'):
                health = memory.get_memory_health_report()
                if isinstance(health, dict) and 'overall_health' in health:
                    health_score = health['overall_health']
                    health_color = Colors.BRIGHT_GREEN if health_score > 0.8 else Colors.BRIGHT_YELLOW if health_score > 0.5 else Colors.BRIGHT_RED
                    print(f"  {colorize('Health Score:', health_color)} {health_score:.1%}")
        except Exception:
            pass  # Skip health if not available
        
        # Recent facts - only if available
        try:
            if hasattr(memory, 'knowledge_graph') and hasattr(memory.knowledge_graph, 'facts') and memory.knowledge_graph.facts:
                print(f"\n{colorize('Recent Facts:', Colors.BRIGHT_YELLOW)}")
                for i, (fact_id, fact) in enumerate(list(memory.knowledge_graph.facts.items())[-3:]):
                    print(f"    {i+1}. {fact.subject} --[{fact.predicate}]--> {fact.object}")
        except Exception:
            pass  # Skip facts if not available
    
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
                'tools': [tool.__name__ if callable(tool) else str(tool) for tool in self.session.tools] if self.session.tools else []
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
        """Show extracted facts, optionally filtered by query."""
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
            print(f"     {colorize(f'Confidence: {fact.confidence:.1%}', confidence_color)} "
                  f"{colorize(f'| Importance: {fact.importance:.1f}', Colors.DIM)} "
                  f"{colorize(f'| Used: {fact.access_count}x', Colors.DIM)}")
            
            displayed += 1

        # Show total count (removed artificial limit)
        if displayed > 0:
            print(f"\n{colorize(f'Total: {displayed} facts displayed', Colors.DIM, italic=True)}")

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
    
    def _cmd_scratchpad(self, args: List[str]) -> None:
        """Show reasoning traces for a specific interaction or all interactions."""
        
        # If a response ID is provided, show specific interaction scratchpad
        if args:
            response_id = args[0]
            # Handle both formats: "4258e5b8" and "cycle_4258e5b8"
            if not response_id.startswith('cycle_'):
                response_id = f"cycle_{response_id}"
            from abstractllm.utils.response_helpers import scratchpad_command
            scratchpad_command(response_id)
            return
            
        # Otherwise, show general reasoning traces summary
        if not hasattr(self.session, 'memory') or not self.session.memory:
            display_error("Memory system not available")
            return
        
        try:
            stats = self.session.memory.get_statistics()
            
            # Debug: Check if stats is actually a dictionary
            if not isinstance(stats, dict):
                display_error("Memory statistics not available")
                return
                
        except Exception as e:
            display_error(f"Failed to get memory statistics: {str(e)}")
            return
        
        print(f"\n{colorize(f'{Symbols.BRAIN} Reasoning Traces', Colors.BRIGHT_CYAN, bold=True)}")
        print(create_divider(60, "─", Colors.CYAN))
        
        total_cycles = stats.get('total_react_cycles', 0)
        successful_cycles = stats.get('successful_cycles', 0)
        
        print(f"  {colorize('Total ReAct Cycles:', Colors.BRIGHT_GREEN)} {total_cycles}")
        print(f"  {colorize('Successful Cycles:', Colors.BRIGHT_GREEN)} {successful_cycles}")
        
        print(f"\n{colorize('Usage:', Colors.BRIGHT_YELLOW)}")
        print(f"  {colorize('/scratch RESPONSE_ID', Colors.BRIGHT_BLUE)} - Show specific interaction scratchpad")
        print(f"  {colorize('Response IDs are shown in the metrics line after each response', Colors.DIM)}")
        
        # Show current cycle if available  
        if hasattr(self.session, 'current_cycle') and self.session.current_cycle:
            cycle = self.session.current_cycle
            print(f"\n{colorize('Current Cycle:', Colors.BRIGHT_YELLOW)}")
            print(f"    {colorize('ID:', Colors.BRIGHT_BLUE)} {cycle.cycle_id}")
            print(f"    {colorize('Query:', Colors.WHITE)} {cycle.query[:80]}...")
            if hasattr(cycle, 'thoughts'):
                print(f"    {colorize('Thoughts:', Colors.GREEN)} {len(cycle.thoughts)}")
            if hasattr(cycle, 'actions'):
                print(f"    {colorize('Actions:', Colors.YELLOW)} {len(cycle.actions)}")
            if hasattr(cycle, 'observations'):
                print(f"    {colorize('Observations:', Colors.CYAN)} {len(cycle.observations)}")
    
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
        """Reset entire session."""
        print(f"{colorize('⚠️  This will reset ALL session data (messages, memory, history)', Colors.BRIGHT_RED)}")
        confirm = input(f"{colorize('Continue? [y/N]: ', Colors.BRIGHT_YELLOW)}")
        
        if confirm.lower() in ['y', 'yes']:
            # Clear messages
            self.session.messages.clear()
            self.session._last_assistant_idx = -1
            
            # Reset memory if available
            if hasattr(self.session, 'memory') and self.session.memory:
                # Create new memory instance
                from abstractllm.memory import HierarchicalMemory
                self.session.memory = HierarchicalMemory()
            
            # Clear command history
            self.command_history.clear()
            
            display_success("Session completely reset")
        else:
            display_info("Reset cancelled")
    
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
        
        print(f"  {colorize('System prompt:', Colors.BRIGHT_BLUE)} {colorize('Set' if self.session.system_prompt else 'None', Colors.WHITE)}")

    def _cmd_context(self, args: List[str]) -> None:
        """Show the exact verbatim context sent to the LLM."""
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

    def _cmd_exit(self, args: List[str]) -> None:
        """Exit interactive mode."""
        display_success("Goodbye!")
        # Use a custom exception to differentiate from Ctrl+C
        raise SystemExit(0)  # Will be caught by interactive mode


def create_command_processor(session, display_func=None) -> CommandProcessor:
    """Create a command processor for the session."""
    return CommandProcessor(session, display_func)