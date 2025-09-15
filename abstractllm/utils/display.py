"""
SOTA CLI display utilities for beautiful agent responses.

This module provides enhanced formatting for LLM responses with:
- Rich typography and colors
- Tool execution visualization
- Metrics dashboard
- Interaction tracking
"""

import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import math

# Rich color codes and formatting
class Colors:
    """ANSI color codes for rich terminal output."""
    
    # Base colors
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class Symbols:
    """Unicode symbols for enhanced visual display."""
    
    # Progress and status
    CHECKMARK = '✅'
    CROSS = '❌'
    WARNING = '⚠️'
    INFO = 'ℹ️'
    ROCKET = '🚀'
    GEAR = '⚙️'
    BRAIN = '🧠'
    SPARKLES = '✨'
    CHART = '📊'
    CLOCK = '⏱️'
    CHAT = '💬'
    
    # Arrows and connectors
    ARROW_RIGHT = '→'
    ARROW_DOWN = '↓'
    ARROW_UP = '↑'
    DOUBLE_ARROW = '⇒'
    
    # Shapes and separators
    DOT = '•'
    DIAMOND = '◆'
    SQUARE = '▪'
    CIRCLE = '●'
    TRIANGLE = '▲'
    
    # Special
    LIGHTNING = '⚡'
    FIRE = '🔥'
    STAR = '⭐'
    TARGET = '🎯'
    KEY = '🔑'
    LINK = '🔗'


def supports_color() -> bool:
    """Check if terminal supports color output."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def colorize(text: str, color: str, bold: bool = False, italic: bool = False) -> str:
    """Apply color and formatting to text."""
    if not supports_color():
        return text
    
    formatting = ""
    if bold:
        formatting += Colors.BOLD
    if italic:
        formatting += Colors.ITALIC
    
    return f"{formatting}{color}{text}{Colors.RESET}"


def create_divider(width: int = 60, char: str = "─", color: str = Colors.BRIGHT_BLACK) -> str:
    """Create a visual divider line."""
    return colorize(char * width, color)


def create_box(content: str, width: int = 60, padding: int = 1) -> str:
    """Create a boxed content area."""
    lines = content.split('\n')
    box = []
    
    # Top border
    box.append(f"┌{'─' * (width - 2)}┐")
    
    # Empty padding
    for _ in range(padding):
        box.append(f"│{' ' * (width - 2)}│")
    
    # Content lines
    for line in lines:
        # Wrap long lines
        if len(line) > width - 4:
            words = line.split(' ')
            current_line = ""
            for word in words:
                if len(current_line + word) <= width - 4:
                    current_line += word + " "
                else:
                    if current_line:
                        box.append(f"│ {current_line.strip():<{width-4}} │")
                    current_line = word + " "
            if current_line:
                box.append(f"│ {current_line.strip():<{width-4}} │")
        else:
            box.append(f"│ {line:<{width-4}} │")
    
    # Empty padding
    for _ in range(padding):
        box.append(f"│{' ' * (width - 2)}│")
    
    # Bottom border
    box.append(f"└{'─' * (width - 2)}┘")
    
    return '\n'.join(box)


def format_tools_execution(tools_executed: List[Dict[str, Any]]) -> str:
    """Format tool execution details beautifully."""
    if not tools_executed:
        return ""
    
    output = []
    output.append(colorize(f"\n{Symbols.GEAR} Tool Execution Trace", Colors.BRIGHT_CYAN, bold=True))
    output.append(create_divider(50, "─", Colors.CYAN))
    
    for i, tool in enumerate(tools_executed, 1):
        name = tool.get('name', 'unknown')
        success = tool.get('success', False)
        exec_time = tool.get('execution_time', 0)
        
        status_symbol = Symbols.CHECKMARK if success else Symbols.CROSS
        status_color = Colors.BRIGHT_GREEN if success else Colors.BRIGHT_RED
        
        # Tool header
        tool_header = f"{status_symbol} {colorize(name, Colors.BRIGHT_BLUE, bold=True)}"
        if exec_time:
            tool_header += colorize(f" ({exec_time*1000:.1f}ms)", Colors.DIM)
        output.append(f"  {i}. {tool_header}")
        
        # Arguments
        args = tool.get('arguments', {})
        if args:
            args_str = ", ".join([f"{k}={v}" for k, v in args.items() if v is not None])
            if args_str:
                output.append(colorize(f"     {Symbols.ARROW_RIGHT} {args_str}", Colors.BRIGHT_BLACK))
        
        # Result preview (first 100 chars)
        result = tool.get('result', '')
        if result and isinstance(result, str):
            preview = result.replace('\n', ' ').strip()
            if len(preview) > 80:
                preview = preview[:80] + "..."
            output.append(colorize(f"     {Symbols.DOUBLE_ARROW} {preview}", Colors.GREEN))
    
    return '\n'.join(output)


def format_metrics_line(response: Any) -> str:
    """Create a single compact italic metrics line."""
    from datetime import datetime

    # Always show response ID if available, even without usage data
    if not hasattr(response, 'react_cycle_id') or not response.react_cycle_id:
        return ""

    usage = response.usage if hasattr(response, 'usage') else None
    metrics_parts = []

    # Add timestamp first
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    metrics_parts.append(f"Time: {timestamp}")

    # Interaction ID - always show if available
    cycle_id = response.react_cycle_id[-8:]  # Last 8 chars for display
    metrics_parts.append(f"ID: {cycle_id}")
    
    # Token metrics - only if usage data available
    if usage and isinstance(usage, dict):
        # Handle different provider field names
        prompt_tokens = usage.get('prompt_tokens', usage.get('input_tokens', 0))
        completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens if prompt_tokens and completion_tokens else 0)
        
        if prompt_tokens and completion_tokens and total_tokens:
            metrics_parts.append(f"Tokens: {prompt_tokens}→{completion_tokens} ({total_tokens} total)")
        elif total_tokens:
            metrics_parts.append(f"Tokens: {total_tokens}")
    
    # Speed calculation - check multiple sources for timing information
    reasoning_time = None
    if hasattr(response, 'total_reasoning_time') and response.total_reasoning_time:
        reasoning_time = response.total_reasoning_time
    elif usage and 'total_time' in usage and usage['total_time']:
        reasoning_time = usage['total_time']
    
    if reasoning_time and usage:
        # Get completion tokens with fallback to output_tokens (Anthropic)
        completion_tokens = usage.get('completion_tokens', usage.get('output_tokens', 0))
        if reasoning_time > 0 and completion_tokens > 0:
            tokens_per_second = completion_tokens / reasoning_time
            metrics_parts.append(f"Speed: {tokens_per_second:.1f} tk/s")
        metrics_parts.append(f"Time: {reasoning_time:.2f}s")
    elif reasoning_time:
        # Show timing even without token data
        metrics_parts.append(f"Time: {reasoning_time:.2f}s")
    
    # Tools used
    if hasattr(response, 'tools_executed') and response.tools_executed:
        tools_count = len(response.tools_executed)
        metrics_parts.append(f"Tools: {tools_count}")
    
    # Scratchpad reference - always show since we have the ID (use short format)
    short_id = cycle_id.replace('cycle_', '') if cycle_id.startswith('cycle_') else cycle_id
    scratchpad_note = f" | /scratch {short_id} for details"
    
    metrics_line = " | ".join(metrics_parts) + scratchpad_note
    return colorize(f"  {metrics_line}", Colors.BRIGHT_BLUE, italic=True)


def format_metrics_summary(response: Any) -> str:
    """Create a beautiful metrics summary dashboard."""
    if not hasattr(response, 'usage') or not response.usage:
        return ""
    
    usage = response.usage
    output = []
    
    # Header
    output.append(f"\n{colorize(f'{Symbols.CHART} Interaction Metrics', Colors.BRIGHT_YELLOW, bold=True)}")
    output.append(create_divider(70, "─", Colors.YELLOW))
    
    # Create metrics grid
    metrics = []
    
    # Interaction ID
    if hasattr(response, 'react_cycle_id') and response.react_cycle_id:
        cycle_id = response.react_cycle_id[-8:]  # Last 8 chars for display
        metrics.append(f"{colorize('ID:', Colors.BRIGHT_BLUE)} {colorize(cycle_id, Colors.WHITE, bold=True)}")
    
    # Token metrics
    if 'total_tokens' in usage:
        total_tokens = usage['total_tokens']
        metrics.append(f"{colorize('Tokens:', Colors.BRIGHT_GREEN)} {colorize(str(total_tokens), Colors.WHITE, bold=True)}")
    
    if 'completion_tokens' in usage:
        completion_tokens = usage['completion_tokens']
        metrics.append(f"{colorize('Generated:', Colors.BRIGHT_CYAN)} {colorize(str(completion_tokens), Colors.WHITE, bold=True)}")
    
    # Speed calculation
    if hasattr(response, 'total_reasoning_time') and response.total_reasoning_time:
        reasoning_time = response.total_reasoning_time
        if 'completion_tokens' in usage and reasoning_time > 0:
            tokens_per_second = usage['completion_tokens'] / reasoning_time
            metrics.append(f"{colorize('Speed:', Colors.BRIGHT_MAGENTA)} {colorize(f'{tokens_per_second:.1f} tk/s', Colors.WHITE, bold=True)}")
        
        metrics.append(f"{colorize('Time:', Colors.BRIGHT_RED)} {colorize(f'{reasoning_time:.2f}s', Colors.WHITE, bold=True)}")
    
    # Tools used
    if hasattr(response, 'tools_executed') and response.tools_executed:
        tools_count = len(response.tools_executed)
        metrics.append(f"{colorize('Tools:', Colors.BRIGHT_YELLOW)} {colorize(str(tools_count), Colors.WHITE, bold=True)}")
    
    # Model info
    if hasattr(response, 'model') and response.model:
        model_name = response.model
        metrics.append(f"{colorize('Model:', Colors.BRIGHT_BLACK)} {colorize(model_name, Colors.WHITE, bold=True)}")
    
    # Scratchpad reference
    if hasattr(response, 'scratchpad_file') and response.scratchpad_file:
        metrics.append(f"{colorize('Scratchpad:', Colors.BRIGHT_BLUE)} {colorize('Available', Colors.WHITE, bold=True)}")
    
    # Display metrics in a grid (2 columns)
    for i in range(0, len(metrics), 2):
        line_metrics = metrics[i:i+2]
        if len(line_metrics) == 2:
            output.append(f"  {line_metrics[0]:<35} {line_metrics[1]}")
        else:
            output.append(f"  {line_metrics[0]}")
    
    # Add helpful commands
    if hasattr(response, 'react_cycle_id') and response.react_cycle_id:
        output.append("")
        commands = [
            f"{colorize('facts(', Colors.DIM)}{colorize(cycle_id, Colors.BRIGHT_BLUE)}{colorize(')', Colors.DIM)} - View extracted facts",
            f"{colorize('scratchpad(', Colors.DIM)}{colorize(cycle_id, Colors.BRIGHT_BLUE)}{colorize(')', Colors.DIM)} - View reasoning trace"
        ]
        for cmd in commands:
            output.append(colorize(f"  {Symbols.KEY} {cmd}", Colors.DIM, italic=True))
    
    return '\n'.join(output)


def display_response(response: Any, show_content: bool = True) -> None:
    """Display a GenerateResponse with beautiful formatting."""
    
    # Main content
    if show_content and hasattr(response, 'content') and response.content:
        content = response.content
        print(f"\n{colorize(f'{Symbols.SPARKLES} Response', Colors.BRIGHT_GREEN, bold=True)}")
        print(create_divider(60, "─", Colors.GREEN))
        
        # Format content with proper wrapping
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                print(f"  {line}")
            else:
                print()
    
    # Tool execution trace
    if hasattr(response, 'tools_executed') and response.tools_executed:
        print(format_tools_execution(response.tools_executed))
    
    # Single-line metrics summary (compact version)
    metrics_line = format_metrics_line(response)
    if metrics_line:
        print(f"\n{metrics_line}")
    
    # Add spacing after response for better readability
    print()


def display_error(error: str, details: Optional[str] = None) -> None:
    """Display error message with proper formatting."""
    print(f"\n{colorize(f'{Symbols.CROSS} Error', Colors.BRIGHT_RED, bold=True)}")
    print(create_divider(50, "─", Colors.RED))
    print(f"  {colorize(error, Colors.RED)}")
    
    if details:
        print(f"  {colorize('Details:', Colors.BRIGHT_RED)} {colorize(details, Colors.DIM)}")
    
    # Add spacing after error for better readability
    print()


def display_thinking(message: str) -> None:
    """Display a 'thinking' message."""
    print(f"{colorize(f'{Symbols.BRAIN} {message}', Colors.BRIGHT_BLUE, italic=True)}")


def display_success(message: str) -> None:
    """Display a success message."""
    print(f"{colorize(f'{Symbols.CHECKMARK} {message}', Colors.BRIGHT_GREEN, bold=True)}")


def display_info(message: str) -> None:
    """Display an info message."""
    print(f"{colorize(f'{Symbols.INFO} {message}', Colors.BRIGHT_CYAN)}")