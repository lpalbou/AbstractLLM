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
    
    # Token Stats
    token_stats = stats.get("token_stats", {})
    if token_stats and token_stats.get("total_tokens", 0) > 0:
        output.append(f"\n{BLUE_ITALIC}🪙 Token Statistics{RESET}")
        output.append(f"Total Tokens: {token_stats.get('total_tokens', 0):,}")
        output.append(f"  Prompt: {token_stats.get('total_prompt_tokens', 0):,}")
        output.append(f"  Completion: {token_stats.get('total_completion_tokens', 0):,}")
        
        messages_with_usage = token_stats.get('messages_with_usage', 0)
        if messages_with_usage > 0:
            avg_prompt = token_stats.get('average_prompt_tokens', 0)
            avg_completion = token_stats.get('average_completion_tokens', 0)
            output.append(f"Average per Message: {avg_prompt:.1f} prompt, {avg_completion:.1f} completion")
            output.append(f"Messages with Usage Data: {messages_with_usage}")
        
        # Add TPS information
        total_time = token_stats.get('total_time', 0)
        if total_time > 0:
            avg_total_tps = token_stats.get('average_total_tps', 0)
            avg_prompt_tps = token_stats.get('average_prompt_tps', 0)
            avg_completion_tps = token_stats.get('average_completion_tps', 0)
            
            output.append(f"Performance:")
            output.append(f"  Total: {avg_total_tps:.1f} tokens/sec")
            output.append(f"  Prompt: {avg_prompt_tps:.1f} tokens/sec")
            output.append(f"  Completion: {avg_completion_tps:.1f} tokens/sec")
            output.append(f"Total Generation Time: {total_time:.2f} seconds")
        
        # Show by provider breakdown if available
        by_provider = token_stats.get('by_provider', {})
        if by_provider:
            output.append(f"By Provider:")
            for provider_name, provider_stats in by_provider.items():
                total = provider_stats.get('total_tokens', 0)
                messages = provider_stats.get('messages', 0)
                provider_tps = provider_stats.get('average_tps', 0)
                provider_time = provider_stats.get('total_time', 0)
                
                if provider_tps > 0:
                    output.append(f"  {provider_name.title()}: {total:,} tokens ({messages} messages, {provider_tps:.1f} tokens/sec)")
                else:
                    output.append(f"  {provider_name.title()}: {total:,} tokens ({messages} messages)")
    
    # Provider Info
    provider_info = stats.get("provider_info", {})
    output.append(f"\n{BLUE_ITALIC}🤖 Provider Information{RESET}")
    output.append(f"Current Provider: {provider_info.get('current_provider', 'None')}")
    
    capabilities = provider_info.get('provider_capabilities', [])
    if capabilities:
        output.append(f"Capabilities: {', '.join(str(cap) for cap in capabilities)}")
    
    return "\n".join(output) 