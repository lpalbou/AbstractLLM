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
    """Save enhanced interaction context with structured ReAct cycle data."""
    
    if not response.react_cycle_id:
        return ""
    
    # Create interaction context file
    context_file = f"/tmp/alma_interaction_{response.react_cycle_id}.json"
    
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
    
    # Comprehensive context data
    context = {
        "cycle_id": response.react_cycle_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response_content": response.content,
        "model": response.model,
        "usage": response.usage,
        "reasoning_time": response.total_reasoning_time,
        "facts_extracted": response.facts_extracted,
        "scratchpad_file": response.scratchpad_file,
        
        # Enhanced structured data
        "react_cycle": react_cycle,
        "structured_thinking": structured_thinking,
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
        with open(context_file, 'w') as f:
            json.dump(context, f, indent=2)
        return context_file
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
    """Display SOTA ReAct scratchpad following proper methodology."""
    from abstractllm.utils.display import display_info, display_error, Colors
    import re
    
    context_file = f"/tmp/alma_interaction_{cycle_id}.json"
    
    try:
        with open(context_file, 'r') as f:
            context = json.load(f)
        
        # Extract key information
        query = context.get('query', 'Unknown query')
        response_content = context.get('response_content', '')
        tools_executed = context.get('tools_executed', [])
        reasoning_time = context.get('reasoning_time', 0)
        timestamp = context.get('timestamp', 'Unknown')
        
        # Parse ReAct components from response content
        think_content = ""
        final_response = ""
        
        if response_content:
            # Extract thinking process and final response
            think_match = re.search(r'<think>(.*?)</think>', response_content, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                final_response = response_content.split('</think>')[-1].strip()
            else:
                # If no think tags, treat entire content as response
                final_response = response_content.strip()
        
        # Display header
        short_id = cycle_id.replace('cycle_', '')
        print(f"\n{Colors.BRIGHT_CYAN}🧠 ReAct Scratchpad - Interaction {short_id}{Colors.RESET}")
        # Add spacing for better readability
        print()
        
        # 1. ORIGINAL QUERY
        print(f"\n{Colors.BRIGHT_BLUE}📋 ORIGINAL QUERY{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
        print(f"{Colors.WHITE}{query}{Colors.RESET}")
        
        # 2. REASONING PROCESS (THINK Phase)
        # Try to use structured thinking data if available, otherwise parse from content
        structured_thinking = context.get('structured_thinking', {})
        reasoning_steps = structured_thinking.get('phases', [])
        
        if not reasoning_steps and think_content:
            # Fallback to parsing from content
            reasoning_steps = _parse_reasoning_phases(think_content)
        
        if reasoning_steps:
            print(f"\n{Colors.BRIGHT_YELLOW}🤔 THINK - Reasoning Process{Colors.RESET}")
            print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
            
            # Show reasoning summary if available
            reasoning_summary = structured_thinking.get('summary', '')
            if reasoning_summary:
                print(f"\n{Colors.BRIGHT_WHITE}📝 Summary:{Colors.RESET} {Colors.DIM}{reasoning_summary}{Colors.RESET}")
            
            for i, step in enumerate(reasoning_steps, 1):
                step_title = step.get('title', f'Reasoning Step {i}')
                step_content = step.get('content', '')
                
                if step_content:
                    print(f"\n{Colors.BRIGHT_YELLOW}  Step {i}: {step_title}{Colors.RESET}")
                    print(f"  {Colors.DIM}{'┌' + '─' * 50}{Colors.RESET}")
                    
                    # Format reasoning content with proper indentation
                    for line in step_content.split('\n'):
                        if line.strip():
                            print(f"  {Colors.DIM}│{Colors.RESET} {line.strip()}")
                    
                    print(f"  {Colors.DIM}{'└' + '─' * 50}{Colors.RESET}")
        elif think_content:
            # Raw thinking content fallback
            print(f"\n{Colors.BRIGHT_YELLOW}🤔 THINK - Reasoning Process{Colors.RESET}")
            print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
            print(f"{Colors.DIM}{think_content}{Colors.RESET}")
        
        # 3. ACTION-OBSERVATION CYCLES
        if tools_executed:
            print(f"\n{Colors.BRIGHT_MAGENTA}⚡ ACT-OBSERVE Cycles{Colors.RESET}")
            print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
            
            for i, tool in enumerate(tools_executed, 1):
                tool_name = tool.get('name', 'unknown_action')
                tool_args = tool.get('arguments', {})
                tool_result = tool.get('result', 'No result available')
                
                print(f"\n{Colors.BRIGHT_MAGENTA}  Cycle {i}:{Colors.RESET}")
                print(f"  {Colors.BRIGHT_GREEN}🎯 ACT:{Colors.RESET} {Colors.BRIGHT_CYAN}{tool_name}{Colors.RESET}")
                
                # Show tool arguments if available
                if tool_args and isinstance(tool_args, dict):
                    for arg_name, arg_value in tool_args.items():
                        if isinstance(arg_value, str) and len(arg_value) > 100:
                            arg_display = f"{arg_value[:100]}..."
                        else:
                            arg_display = str(arg_value)
                        print(f"    {Colors.DIM}├─ {arg_name}:{Colors.RESET} {arg_display}")
                
                print(f"\n  {Colors.BRIGHT_BLUE}👁️  OBSERVE:{Colors.RESET}")
                print(f"  {Colors.DIM}{'┌' + '─' * 60}{Colors.RESET}")
                
                # Format observation with proper indentation
                if isinstance(tool_result, str):
                    result_lines = tool_result.split('\n')
                    for j, line in enumerate(result_lines):
                        if j < 20:  # Limit to first 20 lines
                            print(f"  {Colors.DIM}│{Colors.RESET} {line}")
                        elif j == 20:
                            remaining = len(result_lines) - 20
                            print(f"  {Colors.DIM}│{Colors.RESET} {Colors.DIM}... ({remaining} more lines){Colors.RESET}")
                            break
                else:
                    print(f"  {Colors.DIM}│{Colors.RESET} {str(tool_result)}")
                
                print(f"  {Colors.DIM}{'└' + '─' * 60}{Colors.RESET}")
        
        # 4. FINAL RESPONSE
        if final_response:
            print(f"\n{Colors.BRIGHT_GREEN}✨ FINAL RESPONSE{Colors.RESET}")
            print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
            print(f"{Colors.WHITE}{final_response}{Colors.RESET}")
        
        # 5. SUMMARY AND INSIGHTS
        print(f"\n{Colors.BRIGHT_WHITE}📊 INTERACTION SUMMARY{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 40}{Colors.RESET}")
        
        # Use analysis data if available, otherwise calculate
        analysis = context.get('analysis', {})
        
        total_reasoning_steps = analysis.get('reasoning_depth', len(_parse_reasoning_phases(think_content)) if think_content else 0)
        total_actions = analysis.get('tool_usage', len(tools_executed))
        complexity_score = analysis.get('complexity_score', 0.0)
        
        summary_items = [
            f"Reasoning Steps: {total_reasoning_steps}",
            f"Actions Taken: {total_actions}",
            f"Complexity: {complexity_score:.1f}/1.0",
            f"Processing Time: {reasoning_time:.2f}s" if reasoning_time else "Processing Time: Unknown",
            f"Success: {'✅ Yes' if final_response else '❌ No response'}"
        ]
        
        for item in summary_items:
            print(f"  {Colors.DIM}•{Colors.RESET} {item}")
        
        # Success indicators
        success_indicators = analysis.get('success_indicators', {})
        if success_indicators:
            indicators_text = []
            if success_indicators.get('has_definitive_answer'):
                indicators_text.append("✅ Definitive answer")
            if success_indicators.get('used_tools_successfully'):
                indicators_text.append("🔧 Tool success")
            if success_indicators.get('showed_reasoning'):
                indicators_text.append("🧠 Showed reasoning")
            if success_indicators.get('provided_examples'):
                indicators_text.append("📝 Provided examples")
            
            if indicators_text:
                print(f"\n  {Colors.BRIGHT_GREEN}Quality Indicators:{Colors.RESET} {' | '.join(indicators_text)}")
        
        # Resolution strategy
        if tools_executed:
            strategy = "Tool-assisted problem solving"
            if len(tools_executed) == 1:
                strategy += f" (single {tools_executed[0].get('name', 'action')})"
            else:
                strategy += f" (multi-step: {' → '.join([t.get('name', 'action') for t in tools_executed])})"
        elif think_content or reasoning_steps:
            strategy = "Direct reasoning without external tools"
        else:
            strategy = "Simple response generation"
        
        print(f"\n  {Colors.BRIGHT_YELLOW}🎯 Resolution Strategy:{Colors.RESET} {strategy}")
        
        # Metadata footer
        print(f"\n{Colors.DIM}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.DIM}Interaction ID: {short_id} | Timestamp: {timestamp} | Duration: {reasoning_time:.2f}s{Colors.RESET}")
        # Add spacing for better readability
        print()
        
    except FileNotFoundError:
        display_error(f"Interaction {cycle_id.replace('cycle_', '')} not found")
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
    
    return phases[:10]  # Limit to 10 phases for readability


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