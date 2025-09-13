#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Simple - Clean demonstrator for AbstractLLM with enhanced features.

This example shows how to create an intelligent agent with:
- Hierarchical memory (working, episodic, semantic)
- ReAct reasoning cycles with scratchpads
- Knowledge graph extraction
- Tool support
- Structured responses
"""

from abstractllm.factory import create_session
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
from abstractllm.tools.common_tools import read_file, list_files, search_files
from abstractllm.utils.logging import configure_logging
import argparse
import sys
import logging

# Colors for output
BLUE = '\033[34m'
GREEN = '\033[32m'
RESET = '\033[0m'


def create_agent(provider="ollama", model="qwen3:4b", memory_path=None, max_tool_calls=25):
    """Create an enhanced agent with all SOTA features."""
    
    print(f"{BLUE}🧠 Creating intelligent agent with:{RESET}")
    print(f"  • Hierarchical memory system")
    print(f"  • ReAct reasoning cycles")
    print(f"  • Knowledge graph extraction")
    print(f"  • Tool capabilities")
    print(f"  • Retry strategies\n")
    
    session = create_session(
        provider,
        model=model,
        enable_memory=True,
        enable_retry=True,
        persist_memory=memory_path,
        memory_config={
            'working_memory_size': 10,
            'consolidation_threshold': 5
        },
        tools=[read_file, list_files, search_files],
        system_prompt="You are an intelligent AI assistant with memory and reasoning capabilities.",
        max_tokens=2048,
        temperature=0.7,
        max_tool_calls=max_tool_calls
    )
    
    if memory_path:
        print(f"{GREEN}💾 Memory persisted to: {memory_path}{RESET}\n")
    
    return session


def run_query(session, prompt, structured_output=None):
    """Execute a query with the agent."""
    
    # Configure structured output if requested
    config = None
    if structured_output:
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON if structured_output == "json" else ResponseFormat.YAML,
            force_valid_json=True,
            max_retries=3,
            temperature_override=0.0
        )
    
    # Execute with all features
    response = session.generate(
        prompt=prompt,
        use_memory_context=True,    # Inject relevant memories
        create_react_cycle=True,     # Create ReAct cycle with scratchpad
        structured_config=config     # Structured output if configured
    )
    
    return response


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
    """Run interactive chat with the agent."""
    
    print(f"\n{BLUE}💬 Interactive mode. Type 'exit' to quit, 'memory' for insights.{RESET}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'memory':
                show_memory_insights(session)
                continue
            elif not user_input:
                continue
            
            # Generate response
            response = run_query(session, user_input)
            
            # Display response
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="ALMA Simple - Intelligent agent with AbstractLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
    Interactive chat with memory and tools
  
  %(prog)s --prompt "What files are here?"
    Single query execution
  
  %(prog)s --memory agent.pkl --prompt "Remember my name is Alice"
    Use persistent memory
  
  %(prog)s --structured json --prompt "List 3 colors with hex codes"
    Get structured JSON output
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
        max_tool_calls=args.max_tool_calls
    )
    
    # Execute single prompt or start interactive mode
    if args.prompt:
        print(f"\n{BLUE}Query:{RESET} {args.prompt}\n")
        response = run_query(session, args.prompt, args.structured)
        print(f"{GREEN}Response:{RESET} {response}")
        show_memory_insights(session)
    else:
        interactive_mode(session)
    
    # Save memory if persisting
    if args.memory and session.memory:
        session.memory.save_to_disk()
        print(f"\n{GREEN}💾 Memory saved to {args.memory}{RESET}")


if __name__ == "__main__":
    main()