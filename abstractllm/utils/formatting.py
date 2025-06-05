"""
Formatting utilities for AbstractLLM responses and statistics.
"""

import re
from typing import Dict, Any, Tuple, List

# ANSI color codes for response formatting
RED_BOLD = '\033[1m\033[31m'    # Red bold
GREY_ITALIC = '\033[3m\033[90m'  # Grey italic
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
RESET = '\033[0m'               # Reset formatting


def parse_response_content(content: str) -> Tuple[str, str]:
    """
    Parse response content to extract think tags and clean content.
    
    Args:
        content: Raw response content that may contain <think>...</think> and <answer>...</answer> tags
        
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
    
    # Remove <answer>...</answer> tags but keep the content inside
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_matches = re.findall(answer_pattern, clean_content, re.DOTALL)
    
    if answer_matches:
        # Replace <answer>...</answer> with just the content inside
        for match in answer_matches:
            clean_content = re.sub(answer_pattern, match.strip(), clean_content, count=1, flags=re.DOTALL)
    
    # Remove any remaining answer tags (opening/closing separately)
    clean_content = re.sub(r'</?answer>', '', clean_content).strip()
    
    # Clean up any extra whitespace or newlines
    clean_content = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_content).strip()
    
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
            
            # Format context tokens with thousands separator
            if 'prompt_tokens' in usage:
                context_tokens = usage['prompt_tokens']
                formatted_context = f"{context_tokens:,}".replace(',', ' ')
                metadata_parts.append(f"context : {formatted_context} tk")
            
            # Format generated tokens
            if 'completion_tokens' in usage:
                generated_tokens = usage['completion_tokens']
                metadata_parts.append(f"generated : {generated_tokens} tk")
            
            # Add timing
            if 'time' in usage:
                time_taken = usage['time']
                metadata_parts.append(f"time : {time_taken:.1f}s")
                
                # Calculate and add speed (tokens per second)
                if 'completion_tokens' in usage and usage['completion_tokens'] > 0 and time_taken > 0:
                    speed = usage['completion_tokens'] / time_taken
                    metadata_parts.append(f"speed : {speed:.1f} tk/s")
            
            # Add tool usage if available (check for tool_calls in response)
            tool_count = 0
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_count = len(response.tool_calls)
            # Also check if it's a ToolCallRequest
            elif hasattr(response, '__class__') and 'ToolCall' in str(response.__class__):
                tool_count = len(getattr(response, 'tool_calls', []))
            
            if tool_count > 0:
                metadata_parts.append(f"tools : {tool_count} used")
            
            # Add model name
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


def format_last_interactions(interactions: List[Dict[str, Any]]) -> str:
    """
    Format last interactions data for display.
    
    Args:
        interactions: List of interaction dictionaries from Session.get_last_interactions()
        
    Returns:
        Formatted string for display
    """
    if not interactions:
        return f"{BLUE_ITALIC}No conversation history available{RESET}"
    
    result = [f"\n{BLUE_ITALIC}📜 Last {len(interactions)} Interaction(s){RESET}"]
    
    for idx, interaction in enumerate(interactions, 1):
        # Show interaction number and timestamp
        user_info = interaction.get("user", {})
        timestamp = user_info.get("timestamp")
        if timestamp:
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = "Unknown time"
        
        result.append(f"\n--- Interaction #{idx} ({timestamp_str}) ---")
        
        # Show user message
        if "user" in interaction:
            result.append(f"\n{BLUE_ITALIC}You:{RESET}")
            result.append(interaction["user"]["content"])
        
        # Show assistant response with thinking separated
        if "assistant" in interaction:
            result.append(f"\n{BLUE_ITALIC}Assistant:{RESET}")
            
            assistant_content = interaction["assistant"]["content"]
            think_content, clean_content = parse_response_content(assistant_content)
            
            # Display think content in grey italic if present
            if think_content:
                result.append(f"\n{GREY_ITALIC}<think>")
                result.append(think_content)
                result.append(f"</think>{RESET}\n")
            
            # Display clean content
            if clean_content:
                result.append(clean_content)
            
            # Show metadata if available
            assistant_metadata = interaction["assistant"].get("metadata", {})
            if assistant_metadata and assistant_metadata.get("usage"):
                usage = assistant_metadata["usage"]
                model_name = assistant_metadata.get("model", "unknown")
                
                metadata_parts = []
                
                # Format context tokens with thousands separator
                if 'prompt_tokens' in usage:
                    context_tokens = usage['prompt_tokens']
                    formatted_context = f"{context_tokens:,}".replace(',', ' ')
                    metadata_parts.append(f"context : {formatted_context} tk")
                
                # Format generated tokens
                if 'completion_tokens' in usage:
                    generated_tokens = usage['completion_tokens']
                    metadata_parts.append(f"generated : {generated_tokens} tk")
                
                # Add timing
                if 'time' in usage:
                    time_taken = usage['time']
                    metadata_parts.append(f"time : {time_taken:.1f}s")
                    
                    # Calculate and add speed (tokens per second)
                    if 'completion_tokens' in usage and usage['completion_tokens'] > 0 and time_taken > 0:
                        speed = usage['completion_tokens'] / time_taken
                        metadata_parts.append(f"speed : {speed:.1f} tk/s")
                
                # Add model name
                metadata_parts.append(f"model: {model_name}")
                
                if metadata_parts:
                    result.append(f"\n{BLUE_ITALIC}[{' | '.join(metadata_parts)}]{RESET}")
        
        # Show tool messages if any
        if "tools" in interaction:
            for tool_msg in interaction["tools"]:
                result.append(f"\n{GREY_ITALIC}Tool Result:{RESET}")
                result.append(tool_msg["content"])
    
    result.append(f"\n{BLUE_ITALIC}--- End of Last {len(interactions)} Interaction(s) ---{RESET}")
    return "\n".join(result)


def format_system_prompt_info(info: Dict[str, Any]) -> str:
    """
    Format system prompt information for display.
    
    Args:
        info: System prompt info dictionary from Session.get_system_prompt_info()
        
    Returns:
        Formatted string for display
    """
    if not info.get("has_system_prompt"):
        return "No system prompt is currently set."
    
    char_count = info.get("character_count", 0)
    line_count = info.get("line_count", 0)
    system_prompt = info.get("system_prompt", "")
    
    result = [f"Current System Prompt ({char_count} characters, {line_count} lines):\n"]
    result.append(system_prompt)
    
    return "\n".join(result)


def format_update_result(result: Dict[str, Any]) -> str:
    """
    Format system prompt update result for display.
    
    Args:
        result: Update result dictionary from Session.update_system_prompt()
        
    Returns:
        Formatted string for display
    """
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        return f"{RED_BOLD}Error: {error}{RESET}"
    
    old_length = result.get("old_length", 0)
    new_length = result.get("new_length", 0)
    
    return (f"✅ System prompt updated successfully\n"
            f"📝 Length: {old_length} → {new_length} characters\n"
            f"💡 The new system prompt will apply to future conversations")


def format_tools_list(tools: List[Dict[str, Any]]) -> str:
    """
    Format tools list for display.
    
    Args:
        tools: List of tool dictionaries from Session.get_tools_list()
        
    Returns:
        Formatted string for display
    """
    result = [f"\n{BLUE_ITALIC}🔧 Available Tools{RESET}"]
    result.append("\nThis session has access to the following tools:")
    
    if tools:
        for idx, tool in enumerate(tools, 1):
            name = tool.get("name", "Unknown")
            description = tool.get("description", "No description available")
            parameters = tool.get("parameters", {})
            source = tool.get("source", "Unknown")
            
            # Get parameter info if available
            if isinstance(parameters, dict) and "properties" in parameters:
                param_names = list(parameters["properties"].keys())
                param_str = f"({', '.join(param_names)})" if param_names else "()"
            else:
                param_str = "()" if source == "Function" else ""
            
            result.append(f"  {idx}. {name}{param_str}")
            
            # Clean up description - take first line only
            if description:
                clean_desc = description.strip().split('\n')[0]
                result.append(f"     {clean_desc}")
            
            result.append("")
    else:
        result.append("  No tools are currently available in this session.")
    
    result.append("\n💡 Usage Tips:")
    result.append("  • Tools are called automatically when needed")
    result.append("  • Example: 'List all Python files in the current directory'")
    result.append("  • Example: 'Read the contents of README.md'")
    result.append("  • The agent will choose and execute the right tools")
    
    result.append("\n📚 More Tools:")
    result.append("  See abstractllm.tools.common_tools for additional tools:")
    result.append("  • search_files, write_file, update_file")
    result.append("  • search_internet, fetch_url, fetch_and_parse_html")
    result.append("  • execute_command, ask_user_multiple_choice")
    
    return "\n".join(result)


def format_provider_switch_result(result: Dict[str, Any]) -> str:
    """
    Format provider switch result for display.
    
    Args:
        result: Switch result dictionary from Session.switch_provider()
        
    Returns:
        Formatted string for display
    """
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        return f"{RED_BOLD}Error: {error}{RESET}"
    
    old_provider = result.get("old_provider")
    old_model = result.get("old_model")
    new_provider = result.get("new_provider")
    new_model = result.get("new_model")
    
    old_info = f"{old_provider}:{old_model}" if old_provider and old_model else "None"
    new_info = f"{new_provider}:{new_model}" if new_provider and new_model else "None"
    
    return f"✅ Switched from {old_info} to {new_info}"


def format_provider_info(info: Dict[str, Any]) -> str:
    """
    Format provider information for display.
    
    Args:
        info: Provider info dictionary from Session.get_provider_info()
        
    Returns:
        Formatted string for display
    """
    if not info.get("has_provider"):
        return "No provider is currently configured."
    
    provider_name = info.get("provider_name", "Unknown")
    model_name = info.get("model_name", "Unknown")
    capabilities = info.get("capabilities", [])
    
    result = [f"Current Provider: {provider_name}"]
    result.append(f"Current Model: {model_name}")
    
    if capabilities:
        result.append(f"Capabilities: {', '.join(capabilities)}")
    
    return "\n".join(result) 