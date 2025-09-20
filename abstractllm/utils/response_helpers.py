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
    reasoning_time: Optional[float] = None,
    react_cycle_id: Optional[str] = None
) -> GenerateResponse:
    """Convert a string response to an enhanced GenerateResponse object."""
    
    # Use provided cycle ID or generate a new one
    cycle_id = react_cycle_id or f"cycle_{str(uuid.uuid4())[:8]}"
    
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


def save_interaction_context(response: GenerateResponse, query: str, session_id: str = None) -> str:
    """Save enhanced interaction context with structured ReAct cycle data."""

    if not response.react_cycle_id:
        return ""

    # Create unified storage directory structure
    from pathlib import Path
    import uuid

    # Extract interaction_id from response (now properly formatted)
    interaction_id = response.react_cycle_id
    if not interaction_id:
        interaction_id = f"interaction_{str(uuid.uuid4())[:8]}"

    # Use provided session_id or generate fallback
    if not session_id:
        session_id = f"session_{str(uuid.uuid4())[:8]}"

    # Create interaction directory in unified location
    interaction_dir = Path.home() / ".abstractllm" / "sessions" / session_id / "interactions" / interaction_id
    interaction_dir.mkdir(parents=True, exist_ok=True)

    # Save context to unified location (interaction metadata)
    context_file = interaction_dir / "context.json"
    
    # Extract structured thinking phases from response content
    structured_thinking = _extract_structured_thinking(response.content) if response.content else {}
    
    # Structure ReAct cycle data
    react_cycle = {
        "id": response.react_cycle_id,
        "query": query,
        "thinking_phases": structured_thinking.get("phases", []),
        "reasoning_summary": structured_thinking.get("summary", ""),
        "actions": [],
        "final_response": structured_thinking.get("final_response", "")
    }
    
    # Process tools executed into structured actions
    if response.tools_executed:
        for i, tool in enumerate(response.tools_executed):
            action = {
                "step": i + 1,
                "action_type": "tool_call",
                "tool_name": tool.get('name', 'unknown'),
                "arguments": tool.get('arguments', {}),
                "result": tool.get('result', ''),
                "success": bool(tool.get('result'))
            }
            react_cycle["actions"].append(action)
    
    # Interaction metadata (high-level info)
    interaction_context = {
        "interaction_id": interaction_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "model": response.model,
        "usage": response.usage,
        "reasoning_time": response.total_reasoning_time,
        "facts_extracted": response.facts_extracted,
        "status": "completed",
        "cycles": [react_cycle["cycle_id"]] if react_cycle.get("cycle_id") else []
    }

    # ReAct cycle data (detailed reasoning trace)
    cycle_data = {
        "cycle_id": react_cycle.get("cycle_id", f"cycle_{str(uuid.uuid4())[:8]}"),
        "interaction_id": interaction_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response_content": response.content,
        "structured_thinking": structured_thinking,
        "actions": react_cycle.get("actions", []),
        "tools_executed": response.tools_executed,  # Keep original for backward compatibility
        
        # Metrics and analysis
        "analysis": {
            "complexity_score": _calculate_complexity_score(query, response),
            "reasoning_depth": len(structured_thinking.get("phases", [])),
            "tool_usage": len(response.tools_executed) if response.tools_executed else 0,
            "success_indicators": _extract_success_indicators(response.content, response.tools_executed)
        }
    }

    try:
        # Save interaction context (metadata)
        with open(context_file, 'w') as f:
            json.dump(interaction_context, f, indent=2)

        # Save ReAct cycle data separately
        cycle_id = cycle_data["cycle_id"]
        cycle_file = interaction_dir / f"{cycle_id}.json"
        with open(cycle_file, 'w') as f:
            json.dump(cycle_data, f, indent=2)

        return str(context_file)
    except Exception:
        return ""


def _extract_structured_thinking(content: str) -> dict:
    """Extract and structure thinking phases from response content."""
    import re
    
    if not content:
        return {}
    
    structured = {
        "phases": [],
        "summary": "",
        "final_response": ""
    }
    
    # Extract thinking content
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        
        # Parse thinking phases
        phases = _parse_reasoning_phases(think_content)
        structured["phases"] = phases
        
        # Generate summary
        if phases:
            key_points = []
            for phase in phases:
                if len(phase.get('content', '')) > 50:  # Only include substantial phases
                    # Extract first sentence as summary
                    first_sentence = phase['content'].split('.')[0] + '.'
                    if len(first_sentence) < 200:  # Avoid overly long sentences
                        key_points.append(first_sentence)
            
            structured["summary"] = " ".join(key_points[:3])  # Top 3 key points
        
        # Extract final response
        structured["final_response"] = content.split('</think>')[-1].strip()
    else:
        # No think tags, treat entire content as final response
        structured["final_response"] = content.strip()
    
    return structured


def _calculate_complexity_score(query: str, response: GenerateResponse) -> float:
    """Calculate a complexity score for the interaction (0.0-1.0)."""
    score = 0.0
    
    # Query complexity factors
    query_words = len(query.split())
    if query_words > 10:
        score += 0.2
    elif query_words > 5:
        score += 0.1
    
    # Response complexity factors
    if response.tools_executed:
        score += min(0.3, len(response.tools_executed) * 0.1)  # Up to 0.3 for tool usage
    
    if response.total_reasoning_time and response.total_reasoning_time > 5:
        score += min(0.2, response.total_reasoning_time / 25)  # Up to 0.2 for reasoning time
    
    if response.content:
        content_length = len(response.content)
        if content_length > 1000:
            score += 0.2
        elif content_length > 500:
            score += 0.1
    
    if response.facts_extracted and len(response.facts_extracted) > 0:
        score += 0.1
    
    return min(1.0, score)  # Cap at 1.0


def _extract_success_indicators(content: str, tools_executed: list) -> dict:
    """Extract indicators of interaction success."""
    indicators = {
        "has_definitive_answer": False,
        "used_tools_successfully": False,
        "showed_reasoning": False,
        "provided_examples": False
    }
    
    if content:
        content_lower = content.lower()
        
        # Check for definitive answers
        definitive_phrases = ['the answer is', 'result is', 'here are', 'found', 'located']
        indicators["has_definitive_answer"] = any(phrase in content_lower for phrase in definitive_phrases)
        
        # Check for reasoning
        reasoning_phrases = ['because', 'since', 'therefore', 'first', 'then', 'next']
        indicators["showed_reasoning"] = any(phrase in content_lower for phrase in reasoning_phrases)
        
        # Check for examples
        example_phrases = ['for example', 'such as', 'like', 'including']
        indicators["provided_examples"] = any(phrase in content_lower for phrase in example_phrases)
    
    # Check tool success
    if tools_executed:
        successful_tools = [t for t in tools_executed if t.get('result')]
        indicators["used_tools_successfully"] = len(successful_tools) > 0
    
    return indicators


def facts_command(cycle_id: str) -> None:
    """Display facts extracted from a specific interaction."""
    from abstractllm.utils.display import display_info, display_error, Colors
    from pathlib import Path

    # Look for context file in unified storage location
    base_dir = Path.home() / ".abstractllm" / "sessions"
    context_file = None

    # Search across all sessions for the interaction
    if base_dir.exists():
        for session_dir in base_dir.iterdir():
            if session_dir.is_dir():
                interaction_file = session_dir / "interactions" / cycle_id / "context.json"
                if interaction_file.exists():
                    context_file = interaction_file
                    break

    if not context_file:
        display_error(f"Interaction {cycle_id.replace('cycle_', '')} not found")
        return

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

    except Exception as e:
        display_error(f"Error reading interaction data: {str(e)}")


def scratchpad_command(interaction_id: str) -> None:
    """Display ReAct scratchpad for an interaction."""
    from abstractllm.utils.display import display_info, display_error, Colors
    from pathlib import Path
    import json

    # Handle bare ID or full interaction_id
    search_id = interaction_id if interaction_id.startswith('interaction_') else f"interaction_{interaction_id}"

    # Find the cycle file in the interaction directory
    base_dir = Path.home() / ".abstractllm" / "sessions"
    cycle_data = None
    cycle_file = None

    if base_dir.exists():
        found = False
        for session_dir in base_dir.iterdir():
            if session_dir.is_dir():
                interaction_dir = session_dir / "interactions" / search_id
                if interaction_dir.exists():
                    cycle_files = list(interaction_dir.glob("cycle_*.json"))
                    if cycle_files:
                        # Use the first cycle file (typically only one per interaction)
                        try:
                            with open(cycle_files[0], 'r') as f:
                                cycle_data = json.load(f)
                                cycle_file = cycle_files[0]
                                found = True
                                break
                        except Exception:
                            continue
            if found:
                break

    if not cycle_data:
        display_error(f"No scratchpad found for interaction: {interaction_id}")
        return

    try:
        # Extract ReAct cycle information
        cycle_id = cycle_data.get('cycle_id', 'Unknown')
        query = cycle_data.get('query', 'Unknown query')
        actions = cycle_data.get('actions', [])
        tools_executed = cycle_data.get('tools_executed', [])
        timestamp = cycle_data.get('timestamp', 'Unknown')

        # Use tools_executed if actions is empty (for backward compatibility)
        if not actions and tools_executed:
            actions = [{'tool_name': tool.get('name', 'unknown'), 'observation': tool.get('result', '')} for tool in tools_executed]

        # Get timing information from analysis data
        analysis = cycle_data.get('analysis', {})
        reasoning_time = analysis.get('reasoning_time', 0)

        # Display header
        short_id = search_id.replace('interaction_', '')
        print(f"\n{Colors.BRIGHT_CYAN}🧠 ReAct Scratchpad - {short_id}{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")

        # Query
        print(f"\n{Colors.BRIGHT_BLUE}📋 Query: {Colors.WHITE}{query}{Colors.RESET}")

        # Actions or response content
        if actions:
            print(f"\n{Colors.BRIGHT_YELLOW}⚡ Actions:{Colors.RESET}")
            for i, action in enumerate(actions, 1):
                tool_name = action.get('tool_name', 'unknown')
                observation = action.get('observation', '')
                print(f"\n{Colors.BRIGHT_MAGENTA}  {i}. {tool_name}{Colors.RESET}")
                if observation:
                    print(f"     {Colors.GREEN}→ {observation[:100]}{'...' if len(observation) > 100 else ''}{Colors.RESET}")
        else:
            # Show response content when no actions are captured
            response_content = cycle_data.get('response_content', '')
            if response_content and 'tool_call' in response_content:
                print(f"\n{Colors.BRIGHT_YELLOW}⚡ Tool calls found in response:{Colors.RESET}")
                import re
                tool_calls = re.findall(r'<\|tool_call\|>(.*?)</\|tool_call\|>', response_content, re.DOTALL)
                for i, call in enumerate(tool_calls, 1):
                    try:
                        import json
                        tool_data = json.loads(call)
                        tool_name = tool_data.get('name', 'unknown')
                        print(f"  {i}. {tool_name}")
                    except:
                        print(f"  {i}. Tool call found")
            elif response_content:
                print(f"\n{Colors.GREEN}📝 Response: {response_content[:150]}...{Colors.RESET}")
            else:
                print(f"\n{Colors.DIM}No data recorded{Colors.RESET}")

        print()

    except Exception as e:
        display_error(f"Error reading interaction data: {str(e)}")


def _parse_reasoning_phases(think_content: str) -> list:
    """Parse thinking content into structured reasoning phases."""
    if not think_content:
        return []
    
    phases = []
    
    # Try to identify different reasoning patterns
    paragraphs = [p.strip() for p in think_content.split('\n\n') if p.strip()]
    
    for i, paragraph in enumerate(paragraphs):
        # Identify the type of reasoning phase
        if any(keyword in paragraph.lower() for keyword in ['first', 'initially', 'start', 'begin']):
            phase_title = "Initial Analysis"
        elif any(keyword in paragraph.lower() for keyword in ['need to', 'should', 'must', 'have to']):
            phase_title = "Action Planning"
        elif any(keyword in paragraph.lower() for keyword in ['because', 'since', 'therefore', 'so']):
            phase_title = "Reasoning"
        elif any(keyword in paragraph.lower() for keyword in ['check', 'verify', 'confirm', 'validate']):
            phase_title = "Verification"
        elif any(keyword in paragraph.lower() for keyword in ['conclude', 'final', 'result', 'answer']):
            phase_title = "Conclusion"
        else:
            phase_title = f"Consideration {i+1}"
        
        phases.append({
            'title': phase_title,
            'content': paragraph
        })
    
    return phases  # Return ALL phases without any limit for complete verbatim content


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