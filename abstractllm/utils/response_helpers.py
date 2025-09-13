"""
Helper utilities for enhanced response handling and interaction tracking.
"""

from typing import Any, Optional, Dict, List
from datetime import datetime
import uuid
import json

from abstractllm.types import GenerateResponse


def enhance_string_response(
    content: str, 
    model: Optional[str] = None,
    usage: Optional[Dict[str, int]] = None,
    tools_executed: Optional[List[Dict[str, Any]]] = None,
    reasoning_time: Optional[float] = None
) -> GenerateResponse:
    """Convert a string response to an enhanced GenerateResponse object."""
    
    # Generate a cycle ID for tracking
    cycle_id = f"cycle_{str(uuid.uuid4())[:8]}"
    
    return GenerateResponse(
        content=content,
        model=model,
        usage=usage or {"total_tokens": len(content.split()), "completion_tokens": len(content.split()), "prompt_tokens": 0},
        react_cycle_id=cycle_id,
        tools_executed=tools_executed or [],
        total_reasoning_time=reasoning_time,
        facts_extracted=[],
        reasoning_trace=None
    )


def save_interaction_context(response: GenerateResponse, query: str) -> str:
    """Save interaction context for later reference commands like facts() and scratchpad()."""
    
    if not response.react_cycle_id:
        return ""
    
    # Create interaction context file
    context_file = f"/tmp/alma_interaction_{response.react_cycle_id}.json"
    
    context = {
        "cycle_id": response.react_cycle_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response_content": response.content,
        "model": response.model,
        "usage": response.usage,
        "tools_executed": response.tools_executed,
        "facts_extracted": response.facts_extracted,
        "reasoning_time": response.total_reasoning_time,
        "scratchpad_file": response.scratchpad_file
    }
    
    try:
        with open(context_file, 'w') as f:
            json.dump(context, f, indent=2)
        return context_file
    except Exception:
        return ""


def facts_command(cycle_id: str) -> None:
    """Display facts extracted from a specific interaction."""
    from abstractllm.utils.display import display_info, display_error, Colors
    
    context_file = f"/tmp/alma_interaction_{cycle_id}.json"
    
    try:
        with open(context_file, 'r') as f:
            context = json.load(f)
        
        facts = context.get('facts_extracted', [])
        
        if facts:
            print(f"\n{Colors.BRIGHT_YELLOW}📋 Facts Extracted from {cycle_id}:{Colors.RESET}")
            print(f"{Colors.YELLOW}{'─' * 50}{Colors.RESET}")
            for i, fact in enumerate(facts, 1):
                print(f"  {i}. {fact}")
        else:
            display_info(f"No facts extracted in interaction {cycle_id}")
    
    except FileNotFoundError:
        display_error(f"Interaction {cycle_id} not found")
    except Exception as e:
        display_error(f"Error reading interaction data: {str(e)}")


def scratchpad_command(cycle_id: str) -> None:
    """Display scratchpad/reasoning trace from a specific interaction."""
    from abstractllm.utils.display import display_info, display_error, Colors
    import re
    
    context_file = f"/tmp/alma_interaction_{cycle_id}.json"
    
    try:
        with open(context_file, 'r') as f:
            context = json.load(f)
        
        print(f"\n{Colors.BRIGHT_CYAN}🧠 Complete Scratchpad for {cycle_id}:{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}")
        
        # Extract scratchpad content from response_content
        response_content = context.get('response_content', '')
        if response_content:
            # Look for <think>...</think> tags in the response content
            think_match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
            if think_match:
                scratchpad_content = think_match.group(1).strip()
                print(f"{Colors.DIM}[THINKING PROCESS]{Colors.RESET}")
                print(scratchpad_content)
                print(f"\n{Colors.DIM}[END THINKING]{Colors.RESET}")
                
                # Also show the final response if it exists
                response_after_think = response_content.split('</think>')[-1].strip()
                if response_after_think:
                    print(f"\n{Colors.BRIGHT_GREEN}[FINAL RESPONSE]{Colors.RESET}")
                    print(response_after_think)
            else:
                # No <think> tags, show the full response content
                print(f"{Colors.DIM}[FULL RESPONSE CONTENT]{Colors.RESET}")
                print(response_content)
        
        # Check for separate scratchpad file (legacy support)
        scratchpad_file = context.get('scratchpad_file')
        if scratchpad_file:
            try:
                with open(scratchpad_file, 'r') as sf:
                    scratchpad_content = sf.read()
                print(f"\n{Colors.BRIGHT_YELLOW}[EXTERNAL SCRATCHPAD FILE]{Colors.RESET}")
                print(scratchpad_content)
            except FileNotFoundError:
                pass
        
        # Show additional context information
        query = context.get('query', '')
        if query:
            print(f"\n{Colors.BRIGHT_BLUE}[ORIGINAL QUERY]{Colors.RESET}")
            print(query)
        
        # Show tools executed (without truncation)
        tools = context.get('tools_executed', [])
        if tools:
            print(f"\n{Colors.BRIGHT_MAGENTA}[TOOLS EXECUTED]{Colors.RESET}")
            for i, tool in enumerate(tools, 1):
                tool_name = tool.get('name', 'unknown')
                tool_result = tool.get('result', 'No result')
                print(f"  {i}. {Colors.BRIGHT_CYAN}{tool_name}{Colors.RESET}")
                print(f"     Result: {tool_result}")  # NO TRUNCATION!
        
        # Show metadata
        timestamp = context.get('timestamp', 'Unknown')
        reasoning_time = context.get('reasoning_time')
        print(f"\n{Colors.DIM}[METADATA]{Colors.RESET}")
        print(f"  Timestamp: {timestamp}")
        if reasoning_time:
            print(f"  Reasoning Time: {reasoning_time:.2f}s")
        
        print(f"\n{Colors.CYAN}{'─' * 70}{Colors.RESET}")
        
    except FileNotFoundError:
        display_error(f"Interaction {cycle_id} not found")
    except Exception as e:
        display_error(f"Error reading interaction data: {str(e)}")


# Make these available as global functions for CLI use
def facts(cycle_id: str) -> None:
    """Helper function for facts command."""
    facts_command(cycle_id)


def scratchpad(cycle_id: str) -> None:
    """Helper function for scratchpad command.""" 
    scratchpad_command(cycle_id)


# Add to built-ins for easy CLI access
import builtins
builtins.facts = facts
builtins.scratchpad = scratchpad