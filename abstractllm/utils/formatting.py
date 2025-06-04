"""
Formatting utilities for AbstractLLM responses and statistics.
"""

import re
from typing import Dict, Any, Tuple

# ANSI color codes for response formatting
RED_BOLD = '\033[1m\033[31m'    # Red bold
GREY_ITALIC = '\033[3m\033[90m'  # Grey italic
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
RESET = '\033[0m'               # Reset formatting


def parse_response_content(content: str) -> Tuple[str, str]:
    """
    Parse response content to extract think tags and clean content.
    
    Args:
        content: Raw response content that may contain <think>...</think> tags
        
    Returns:
        tuple: (think_content, clean_content)
    """
    think_content = ""
    clean_content = content
    
    # Extract <think> content using regex
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, content, re.DOTALL)
    
    if think_matches:
        think_content = think_matches[0].strip()
        # Remove all <think>...</think> blocks from the main content
        clean_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
    
    return think_content, clean_content


def format_response_display(response) -> None:
    """
    Format and display a response with proper styling.
    
    Args:
        response: GenerateResponse object or string
    """
    if hasattr(response, 'content'):
        content = response.content
        
        # Parse the content
        think_content, clean_content = parse_response_content(content)
        
        # Display think content in grey italic if present
        if think_content:
            print(f"\n{GREY_ITALIC}<think>")
            print(think_content)
            print(f"</think>{RESET}\n")
        
        # Display clean content
        if clean_content:
            print(clean_content)
        
        # Display metadata in blue italic if available
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            model_name = getattr(response, 'model', 'unknown')
            
            metadata_parts = []
            if 'completion_tokens' in usage:
                metadata_parts.append(f"{usage['completion_tokens']} tokens")
            if 'time' in usage:
                metadata_parts.append(f"{usage['time']:.2f}s")
            metadata_parts.append(f"model: {model_name}")
            
            if metadata_parts:
                print(f"\n{BLUE_ITALIC}[{' | '.join(metadata_parts)}]{RESET}")
    else:
        # Handle string responses
        think_content, clean_content = parse_response_content(str(response))
        
        if think_content:
            print(f"\n{GREY_ITALIC}<think>")
            print(think_content)
            print(f"</think>{RESET}\n")
        
        if clean_content:
            print(clean_content)


def format_stats_display(stats: Dict[str, Any]) -> str:
    """
    Format session statistics for display.
    
    Args:
        stats: Dictionary containing session statistics from Session.get_stats()
        
    Returns:
        Formatted string for display
    """
    output = []
    
    # Session Info
    session_info = stats.get("session_info", {})
    output.append(f"{BLUE_ITALIC}📊 Session Statistics{RESET}")
    output.append(f"Session ID: {session_info.get('id', 'N/A')}")
    output.append(f"Created: {session_info.get('created_at', 'N/A')}")
    output.append(f"Duration: {session_info.get('duration_hours', 0):.2f} hours")
    output.append(f"Has System Prompt: {session_info.get('has_system_prompt', False)}")
    
    # Message Stats
    msg_stats = stats.get("message_stats", {})
    output.append(f"\n{BLUE_ITALIC}💬 Message Statistics{RESET}")
    output.append(f"Total Messages: {msg_stats.get('total_messages', 0)}")
    
    by_role = msg_stats.get("by_role", {})
    for role, count in by_role.items():
        output.append(f"  {role.title()}: {count}")
    
    output.append(f"Total Characters: {msg_stats.get('total_characters', 0):,}")
    output.append(f"Average Message Length: {msg_stats.get('average_message_length', 0):.1f} chars")
    
    # Tool Stats
    tool_stats = stats.get("tool_stats", {})
    output.append(f"\n{BLUE_ITALIC}🔧 Tool Statistics{RESET}")
    output.append(f"Tools Available: {tool_stats.get('tools_available', 0)}")
    output.append(f"Total Tool Calls: {tool_stats.get('total_tool_calls', 0)}")
    output.append(f"Successful: {tool_stats.get('successful_tool_calls', 0)}")
    output.append(f"Failed: {tool_stats.get('failed_tool_calls', 0)}")
    
    success_rate = tool_stats.get('tool_success_rate', 0)
    output.append(f"Success Rate: {success_rate:.1%}")
    
    unique_tools = tool_stats.get('unique_tools_used', [])
    if unique_tools:
        output.append(f"Tools Used: {', '.join(unique_tools)}")
    
    # Provider Info
    provider_info = stats.get("provider_info", {})
    output.append(f"\n{BLUE_ITALIC}🤖 Provider Information{RESET}")
    output.append(f"Current Provider: {provider_info.get('current_provider', 'None')}")
    
    capabilities = provider_info.get('provider_capabilities', [])
    if capabilities:
        output.append(f"Capabilities: {', '.join(str(cap) for cap in capabilities)}")
    
    return "\n".join(output) 