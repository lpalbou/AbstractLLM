#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Enhanced - Simple agent with SOTA memory, retry, and structured responses.

This is a minimal implementation showing how the enhanced features work seamlessly.
All complexity is hidden - the agent automatically:
- Maintains hierarchical memory with ReAct cycles
- Retries on failures with exponential backoff  
- Handles structured responses
- Extracts facts and builds knowledge graphs
"""

from abstractllm.factory_enhanced import create_enhanced_session
from abstractllm.tools.common_tools import read_file, list_files, search_files
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
from abstractllm.utils.logging import configure_logging
import os
import sys
import argparse
import json
from pathlib import Path

# ANSI colors for output
BLUE_ITALIC = '\033[3m\033[34m'
GREEN_BOLD = '\033[1m\033[32m'
GREY_ITALIC = '\033[3m\033[90m'
RESET = '\033[0m'


def main():
    """Main entry point for ALMA Enhanced."""
    parser = argparse.ArgumentParser(
        description="ALMA Enhanced - Minimal agent with SOTA features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple query (memory and retry enabled by default)
  %(prog)s --prompt "What is retry logic in Python?"
  
  # With persistent memory
  %(prog)s --prompt "Remember this: I like Python" --persist-memory ./memory
  
  # Query memory
  %(prog)s --prompt "What do you know about Python?" --persist-memory ./memory
  
  # Structured response
  %(prog)s --prompt "Generate a user profile" --structured-json
  
  # Show memory stats
  %(prog)s --prompt "Hello" --show-stats
"""
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["openai", "anthropic", "ollama", "huggingface", "mlx"],
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:4b",
        help="Model name (provider-specific)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Query for the agent"
    )
    
    parser.add_argument(
        "--persist-memory",
        type=str,
        help="Path to persist memory across sessions"
    )
    
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable hierarchical memory"
    )
    
    parser.add_argument(
        "--disable-retry",
        action="store_true",
        help="Disable retry strategies"
    )
    
    parser.add_argument(
        "--structured-json",
        action="store_true",
        help="Request structured JSON response"
    )
    
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show memory statistics after query"
    )
    
    parser.add_argument(
        "--show-links",
        action="store_true",
        help="Show memory link visualization"
    )
    
    parser.add_argument(
        "--query-memory",
        type=str,
        help="Query memory for specific information"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        configure_logging(console_level="DEBUG", file_level="DEBUG")
    else:
        configure_logging(console_level="WARNING", file_level="INFO")
    
    print(f"{GREEN_BOLD}🚀 ALMA Enhanced - Agent with SOTA Features{RESET}")
    print(f"{GREY_ITALIC}Provider: {args.provider}, Model: {args.model}{RESET}")
    
    # Create enhanced session (automatically includes memory & retry)
    print(f"\n{BLUE_ITALIC}Creating enhanced session...{RESET}")
    
    session = create_enhanced_session(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_memory=not args.disable_memory,
        enable_retry=not args.disable_retry,
        persist_memory=args.persist_memory,
        system_prompt="""You are a helpful AI assistant with advanced memory and reasoning capabilities.
You can remember information across conversations, retry on failures, and provide structured responses.
When using tools, think step by step and explain your reasoning."""
    )
    
    # Add tools
    session.add_tool(list_files)
    session.add_tool(read_file)
    session.add_tool(search_files)
    
    # Query memory if requested
    if args.query_memory:
        print(f"\n{BLUE_ITALIC}Querying memory for: {args.query_memory}{RESET}")
        results = session.query_memory(args.query_memory)
        if results:
            print(f"\nMemory query results:")
            if results.get("facts"):
                print(f"  Facts found: {len(results['facts'])}")
                for fact in results["facts"][:3]:
                    print(f"    - {fact}")
            if results.get("react_cycles"):
                print(f"  Related queries: {len(results['react_cycles'])}")
                for cycle in results["react_cycles"][:3]:
                    print(f"    - {cycle['query'][:50]}...")
        else:
            print("No memory available")
    
    # Main query
    print(f"\n{BLUE_ITALIC}Query: {args.prompt}{RESET}")
    
    try:
        # Prepare structured config if requested
        structured_config = None
        if args.structured_json:
            structured_config = StructuredResponseConfig(
                format=ResponseFormat.JSON,
                temperature_override=0.3,
                max_retries=3
            )
            print(f"{GREY_ITALIC}Using structured JSON response{RESET}")
        
        # Generate response
        # The session automatically:
        # - Creates a ReAct cycle for this query
        # - Adds memory context from previous interactions
        # - Retries on failures with exponential backoff
        # - Extracts facts and updates knowledge graph
        
        response = session.generate(
            prompt=args.prompt,
            tools=[list_files, read_file, search_files],
            max_tool_calls=10,
            structured_config=structured_config
        )
        
        print(f"\n{GREEN_BOLD}Response:{RESET}")
        
        # Format output based on type
        if isinstance(response, dict):
            # Structured response
            print(json.dumps(response, indent=2))
        else:
            # Regular response
            print(response)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Show statistics if requested
    if args.show_stats:
        stats = session.get_memory_stats()
        if stats:
            print(f"\n{BLUE_ITALIC}Memory Statistics:{RESET}")
            print(f"  Session ID: {stats['session_id']}")
            print(f"  Working memory: {stats['working_memory_size']} items")
            print(f"  Episodic memory: {stats['episodic_memory_size']} items")
            print(f"  Facts extracted: {stats['total_facts']}")
            print(f"  ReAct cycles: {stats['total_react_cycles']}")
            print(f"  Success rate: {stats['success_rate']:.0%}")
            print(f"  Total links: {stats['total_links']}")
    
    # Show link visualization if requested
    if args.show_links:
        visualization = session.visualize_memory_links()
        if visualization:
            print(f"\n{BLUE_ITALIC}Memory Links:{RESET}")
            print(visualization)
    
    # Save memory if persistent
    if args.persist_memory:
        session.save_memory()
        print(f"\n{GREEN_BOLD}✅ Memory saved to {args.persist_memory}{RESET}")
    
    print(f"\n{GREEN_BOLD}✨ Done!{RESET}")


if __name__ == "__main__":
    main()