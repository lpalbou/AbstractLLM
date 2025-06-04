#!/usr/bin/env python3
"""
Minimal ALMA (AbstractLLM Agent) implementation with file reading capability.
Uses the simplest approach to tool calling with an interactive REPL.

# Requirements
- AbstractLLM: pip install abstractllm[anthropic]
- Anthropic API key: export ANTHROPIC_API_KEY=your_api_key_here
"""

from abstractllm import create_llm
from abstractllm.session import Session
import os
import logging
import re

# ANSI color codes for logging
class LogColors:
    RED = '\033[91m'        # Error
    YELLOW = '\033[93m'     # Warning  
    GREEN = '\033[92m'      # Tool execution
    CYAN = '\033[96m'       # Info
    BLUE = '\033[94m'       # Debug
    MAGENTA = '\033[95m'    # Results
    BOLD = '\033[1m'        # Bold
    RESET = '\033[0m'       # Reset

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level and content."""
    
    def format(self, record):
        message = record.getMessage()
        
        # Only show essential tool execution messages
        if "Executing tool call:" in message and "with args:" in message:
            # Extract tool name and args for concise display
            import re
            match = re.search(r'Executing tool call: (\w+) with args: ({.*})', message)
            if match:
                tool_name = match.group(1)
                args = match.group(2)
                timestamp = self.formatTime(record, '%H:%M:%S')
                return f"{LogColors.GREEN}{timestamp} TOOL: {tool_name}({args}){LogColors.RESET}"
                
        elif "Tool execution successful:" in message:
            # Parse the enhanced success message
            import re
            # Match pattern: "Tool execution successful: tool_name (123 chars, ~30 tokens)"
            enhanced_match = re.search(r'Tool execution successful: (\w+) \((\d+) chars, ~(\d+) tokens\)', message)
            if enhanced_match:
                tool_name = enhanced_match.group(1)
                char_count = enhanced_match.group(2)
                token_count = enhanced_match.group(3)
                timestamp = self.formatTime(record, '%H:%M:%S')
                return f"{LogColors.GREEN}{timestamp} SUCCESS: {tool_name} ({char_count} chars, ~{token_count} tokens){LogColors.RESET}"
            else:
                # Check if this is a simple "Tool execution successful: tool_name" message (no parentheses)
                # These are uninformative and should be filtered out
                if not '(' in message and not ')' in message:
                    # This is a simple success message without details, filter it out
                    return None
                else:
                    # This might be a different kind of success message with details, show it
                    tool_name = message.split("Tool execution successful: ")[-1]
                    timestamp = self.formatTime(record, '%H:%M:%S')
                    return f"{LogColors.GREEN}{timestamp} SUCCESS: {tool_name}{LogColors.RESET}"
            
        elif record.levelno == logging.WARNING:
            timestamp = self.formatTime(record, '%H:%M:%S')
            return f"{LogColors.YELLOW}{LogColors.BOLD}{timestamp} WARNING: {message}{LogColors.RESET}"
            
        elif record.levelno >= logging.ERROR:
            timestamp = self.formatTime(record, '%H:%M:%S')
            return f"{LogColors.RED}{LogColors.BOLD}{timestamp} ERROR: {message}{LogColors.RESET}"
            
        # Filter out all other verbose logs
        return None

# Set up colored logging
def setup_colored_logging():
    """Set up colored logging for better visibility."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    # Filter out None messages
    class FilterNoneHandler(logging.StreamHandler):
        def emit(self, record):
            msg = self.format(record)
            if msg is not None:
                super().emit(record)
    
    console_handler = FilterNoneHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

# Set up colored logging
setup_colored_logging()

# ANSI color codes for response formatting
RED_BOLD = '\033[1m\033[31m'    # Red bold
GREY_ITALIC = '\033[3m\033[90m'  # Grey italic
BLUE_ITALIC = '\033[3m\033[34m'  # Blue italic
RESET = '\033[0m'               # Reset formatting

class VerboseSession(Session):
    """Session class with enhanced tool execution logging and token tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_call_count = 0
        # Track token usage across the session
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_generation_time = 0.0
        self.response_count = 0
    
    def add_message(self, role, content, name=None, tool_results=None, metadata=None):
        """Override to capture usage data from assistant messages."""
        # Check if this is a GenerateResponse with usage data
        if hasattr(content, 'usage') and content.usage:
            usage = content.usage
            # Track cumulative token usage
            self.total_prompt_tokens += usage.get('prompt_tokens', 0)
            self.total_completion_tokens += usage.get('completion_tokens', 0)
            self.total_generation_time += usage.get('time', 0.0)
            self.response_count += 1
            
            # Store usage data in message metadata
            if metadata is None:
                metadata = {}
            metadata['usage'] = usage
            
            # Extract just the content from GenerateResponse
            content_text = getattr(content, 'content', str(content))
        else:
            content_text = content
        
        # Call parent method with text content
        return super().add_message(role, content_text, name, tool_results, metadata)
    
    def generate(self, *args, **kwargs):
        """Override to capture token usage from responses."""
        # Call parent method
        response = super().generate(*args, **kwargs)
        
        # If we got a GenerateResponse, capture its usage
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            # Track cumulative token usage
            self.total_prompt_tokens += usage.get('prompt_tokens', 0)
            self.total_completion_tokens += usage.get('completion_tokens', 0)
            self.total_generation_time += usage.get('time', 0.0)
            self.response_count += 1
            
            # Find the last assistant message and add usage to its metadata
            for message in reversed(self.messages):
                if message.role == "assistant":
                    if 'usage' not in message.metadata:
                        message.metadata['usage'] = usage
                    break
        
        return response
    
    def execute_tool_call(self, tool_call, tool_functions):
        """Override to add detailed tool execution logging."""
        self.tool_call_count += 1
        
        # Execute the tool using parent method
        result = super().execute_tool_call(tool_call, tool_functions)
        
        # Add enhanced success logging with details
        if result and not result.get("error"):
            output = result.get("output", "")
            if isinstance(output, str):
                char_count = len(output)
                # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                estimated_tokens = char_count // 4
                
                # Create enhanced success message
                import logging
                logger = logging.getLogger("abstractllm.session")
                logger.info(f"Tool execution successful: {tool_call.name} "
                           f"({char_count} chars, ~{estimated_tokens} tokens)")
            else:
                # For non-string outputs, just log the tool name
                import logging
                logger = logging.getLogger("abstractllm.session")
                logger.info(f"Tool execution successful: {tool_call.name}")
        
        return result
    
    def get_stats(self):
        """Override to include enhanced token usage statistics."""
        # Get base stats from parent
        stats = super().get_stats()
        
        # Add comprehensive token usage stats
        stats["token_stats"] = {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_generation_time": self.total_generation_time,
            "response_count": self.response_count,
            "average_prompt_tokens": (self.total_prompt_tokens / self.response_count) if self.response_count > 0 else 0,
            "average_completion_tokens": (self.total_completion_tokens / self.response_count) if self.response_count > 0 else 0,
            "average_response_time": (self.total_generation_time / self.response_count) if self.response_count > 0 else 0,
            "tokens_per_second": (self.total_completion_tokens / self.total_generation_time) if self.total_generation_time > 0 else 0
        }
        
        # Extract token usage from individual messages
        message_token_details = []
        for message in self.messages:
            if message.role == "assistant" and message.metadata.get('usage'):
                usage = message.metadata['usage']
                message_token_details.append({
                    "timestamp": message.timestamp.isoformat(),
                    "prompt_tokens": usage.get('prompt_tokens', 0),
                    "completion_tokens": usage.get('completion_tokens', 0),
                    "total_tokens": usage.get('total_tokens', 0),
                    "time": usage.get('time', 0.0),
                    "content_length": len(message.content)
                })
        
        stats["token_stats"]["message_details"] = message_token_details
        
        return stats

def read_file(file_path: str, should_read_entire_file: bool = True, start_line_one_indexed: int = 1, end_line_one_indexed_inclusive: int = None) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        should_read_entire_file: Whether to read the entire file (default: True)
        start_line_one_indexed: Starting line number (1-indexed, default: 1)
        end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive, default: None for end of file)
    
    Returns:
        File contents or error message
    """
    try:
        with open(file_path, 'r') as f:
            if should_read_entire_file:
                return f.read()
            else:
                # Read specific line range
                lines = f.readlines()
                
                # Convert to 0-indexed for Python
                start_idx = max(0, start_line_one_indexed - 1)
                
                if end_line_one_indexed_inclusive is None:
                    end_idx = len(lines)
                else:
                    end_idx = min(len(lines), end_line_one_indexed_inclusive)
                
                # Extract the requested lines
                selected_lines = lines[start_idx:end_idx]
                
                # Add line numbers for clarity
                result_lines = []
                for i, line in enumerate(selected_lines, start=start_line_one_indexed):
                    result_lines.append(f"{i:4d}: {line.rstrip()}")
                
                return "\n".join(result_lines)
                
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def parse_response_content(content: str):
    """
    Parse response content to extract think tags and clean content.
    
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

def format_response_display(response):
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

def format_stats_display(stats: dict) -> str:
    """
    Format session statistics for display.
    
    Args:
        stats: Dictionary containing session statistics
        
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
    
    # Token Usage Stats
    token_stats = stats.get("token_stats", {})
    if token_stats:
        output.append(f"\n{BLUE_ITALIC}🪙 Token Usage Statistics{RESET}")
        output.append(f"Total Prompt Tokens: {token_stats.get('total_prompt_tokens', 0):,}")
        output.append(f"Total Completion Tokens: {token_stats.get('total_completion_tokens', 0):,}")
        output.append(f"Total Tokens: {token_stats.get('total_tokens', 0):,}")
        output.append(f"Response Count: {token_stats.get('response_count', 0)}")
        
        if token_stats.get('response_count', 0) > 0:
            output.append(f"Average Prompt Tokens: {token_stats.get('average_prompt_tokens', 0):.1f}")
            output.append(f"Average Completion Tokens: {token_stats.get('average_completion_tokens', 0):.1f}")
            output.append(f"Average Response Time: {token_stats.get('average_response_time', 0):.2f}s")
            
        if token_stats.get('tokens_per_second', 0) > 0:
            output.append(f"Average Generation Speed: {token_stats.get('tokens_per_second', 0):.1f} tokens/sec")
        
        total_time = token_stats.get('total_generation_time', 0)
        if total_time > 0:
            output.append(f"Total Generation Time: {total_time:.2f}s")
    
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

def main():    
    # Initialize the provider with the model - this is the key step
    # The Session will use this provider's model by default
    
    model_name = "cogito"
    model_name = "qwen2.5"
    #provider = create_llm("ollama", model=model_name)
    provider = create_llm("mlx", 
                         model="mlx-community/Qwen3-30B-A3B-4bit",
                         #model="mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
                         max_tokens=4096)  # Set default max_tokens


    # TEST WITH ANTHROPIC
    # provider = create_llm("anthropic", 
    #                    model="claude-3-5-haiku-20241022")

    # TEST WITH OPENAI
    # provider = create_llm("openai", 
    #                     model="gpt-4o")

    # TEST WITH MLX (Apple Silicon only)
    # provider = create_llm("mlx", 
    #                     model="mlx-community/Qwen3-30B-A3B-4bit")

    # Create session with the provider and tool function
    # Use our enhanced VerboseSession for better tool logging
    session = VerboseSession(
        system_prompt="""You are a capable agent that EXECUTES tools and follows instructions directly. When a user asks you to follow instructions from a document, you should READ the document and then EXECUTE the steps as written, not just summarize them.

CRITICAL: When you need to read a file, you MUST actually call the read_file tool. Do NOT show code examples or pseudo-code. EXECUTE the tool call immediately.

For example, if you need to read a file, you should:
- Actually call read_file(file_path="path/to/file", should_read_entire_file=True)
- NOT show: ```python read_file(...)``` 
- NOT explain what you would do
- JUST DO IT

You have access to these tools:
- read_file(file_path, should_read_entire_file=True, start_line_one_indexed=1, end_line_one_indexed_inclusive=None)

When following multi-step procedures:
1. Read the instructions first by CALLING read_file
2. Execute each step that requires a tool call by CALLING the tools
3. Continue to the next step based on the results
4. Complete the entire procedure unless instructed otherwise

You are an ACTION-TAKING agent, not just an advisor. Take action immediately when requested.""",
        provider=provider,
        tools=[read_file]  # Function is automatically registered
    )
    
    print("\nMinimal ALMA - Type '/exit', '/quit', or '/q' to quit")
    print("Example: 'Read the file README.md and summarize it'")
    print("Commands: /stats, /save <filename>, /load <filename>")
    print("Type '/help' for more information")
    
    # Simple REPL loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit command
            if user_input.lower() in ["/exit", "/quit", "/q", "exit", "quit", "bye"]:
                print("Goodbye!")
                break
                
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Handle special commands
            if user_input.startswith("/"):
                command_parts = user_input.split()
                command = command_parts[0].lower()
                
                if command == "/stats":
                    # Show session statistics
                    try:
                        stats = session.get_stats()
                        print(f"\n{format_stats_display(stats)}")
                    except Exception as e:
                        print(f"{RED_BOLD}Error getting stats: {str(e)}{RESET}")
                    continue
                
                elif command == "/save":
                    # Save session to file
                    if len(command_parts) < 2:
                        print(f"{RED_BOLD}Usage: /save <filename>{RESET}")
                        continue
                    
                    filename = command_parts[1]
                    if not filename.endswith('.json'):
                        filename += '.json'
                    
                    try:
                        session.save(filename)
                        print(f"{LogColors.GREEN}Session saved to {filename}{LogColors.RESET}")
                    except Exception as e:
                        print(f"{RED_BOLD}Error saving session: {str(e)}{RESET}")
                    continue
                
                elif command == "/load":
                    # Load session from file
                    if len(command_parts) < 2:
                        print(f"{RED_BOLD}Usage: /load <filename>{RESET}")
                        continue
                    
                    filename = command_parts[1]
                    if not filename.endswith('.json'):
                        filename += '.json'
                    
                    try:
                        # Load session and replace current session
                        from abstractllm.session import Session
                        new_session = Session.load(filename, provider=provider)
                        
                        # Transfer tools from current session to loaded session
                        new_session.tools = session.tools
                        new_session._tool_implementations = session._tool_implementations
                        
                        session = new_session
                        print(f"{LogColors.GREEN}Session loaded from {filename}{LogColors.RESET}")
                        
                        # Show basic info about loaded session
                        stats = session.get_stats()
                        msg_count = stats["message_stats"]["total_messages"]
                        duration = stats["session_info"]["duration_hours"]
                        print(f"{BLUE_ITALIC}Loaded session: {msg_count} messages, {duration:.2f} hours{RESET}")
                        
                    except FileNotFoundError:
                        print(f"{RED_BOLD}Error: File '{filename}' not found{RESET}")
                    except Exception as e:
                        print(f"{RED_BOLD}Error loading session: {str(e)}{RESET}")
                    continue
                
                elif command == "/help":
                    # Show help information
                    print(f"\n{BLUE_ITALIC}💡 ALMA Session Management Help{RESET}")
                    print(f"\n{LogColors.GREEN}Available Commands:{LogColors.RESET}")
                    print(f"  /stats                  - Show session statistics")
                    print(f"  /save <filename>        - Save current session to file")
                    print(f"  /load <filename>        - Load session from file")
                    print(f"  /help                   - Show this help message")
                    print(f"  /exit, /quit, /q        - Exit ALMA")
                    
                    print(f"\n{LogColors.GREEN}Session Features:{LogColors.RESET}")
                    print(f"  • Conversation history is automatically saved")
                    print(f"  • Tool calls and results are tracked")
                    print(f"  • Sessions persist system prompts and metadata")
                    print(f"  • JSON format allows easy inspection and sharing")
                    
                    print(f"\n{LogColors.GREEN}Example Usage:{LogColors.RESET}")
                    print(f"  /save my_conversation     - Saves to 'my_conversation.json'")
                    print(f"  /load my_conversation     - Loads from 'my_conversation.json'")
                    print(f"  /stats                    - Shows message counts, tool usage, etc.")
                    
                    print(f"\n{LogColors.GREEN}Tool Support:{LogColors.RESET}")
                    print(f"  Available tools: read_file")
                    print(f"  Example: 'Read the README.md file and summarize it'")
                    continue
                
                else:
                    print(f"{RED_BOLD}Unknown command: {command}{RESET}")
                    print("Available commands: /stats, /save <filename>, /load <filename>, /help, /exit")
                    continue
            
            # Generate response with tool support
            print(f"\nAssistant:")
            
            # Use the unified generate method - trust AbstractLLM to handle everything
            response = session.generate(
                prompt=user_input,
                max_tool_calls=25  # Limit tool calls to avoid infinite loops
            )
            
            # Simply display the response - trust AbstractLLM formatting
            print()  # Add spacing
            format_response_display(response)
            
        except EOFError:
            # Handle EOF gracefully (Ctrl+D or redirected input)
            print("\nReceived EOF (Ctrl+D). Goodbye!")
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nReceived interrupt (Ctrl+C). Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 