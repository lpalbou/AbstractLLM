"""
Universal tool support for AbstractLLM.

This package provides a unified tool system that works across all models
and providers, whether they have native tool APIs or require prompting.

Key components:
- Core types (ToolDefinition, ToolCall, ToolResult)
- Universal handler for all models
- Architecture-based parsing and formatting
- Tool registry for managing available tools

Example usage:
```python
from abstractllm.tools import create_handler, register

# Register a tool
@register
def search_web(query: str) -> str:
    '''Search the web for information.'''
    return f"Results for: {query}"

# Create handler for a model
handler = create_handler("gpt-4")

# Get tool prompt for prompted models
tool_prompt = handler.format_tools_prompt([search_web])

# Or prepare tools for native API
native_tools = handler.prepare_tools_for_native([search_web])

# Parse response for tool calls
response = "I'll search for weather information. <function_call>{"name": "search_web", "arguments": {"query": "current weather"}}</function_call>"
parsed = handler.parse_response(response, mode="prompted")

# Execute tools if needed
if parsed.has_tool_calls():
    from abstractllm.tools import execute_tools
    results = execute_tools(parsed.tool_calls)
    formatted = handler.format_tool_results(results, mode="prompted")
```
"""

# Core types
from abstractllm.tools.core import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolCallResponse,
    ToolCallRequest,  # Legacy alias for ToolCallResponse
    function_to_tool_definition
)

# Handler
from abstractllm.tools.handler import (
    UniversalToolHandler,
    create_handler
)

# Parser functions
from abstractllm.tools.parser import (
    detect_tool_calls,
    parse_tool_calls,
    format_tool_prompt
)

# Registry
from abstractllm.tools.registry import (
    ToolRegistry,
    register,
    get_registry,
    execute_tool,
    execute_tools
)

# Enhanced features (conditionally imported)
try:
    from abstractllm.tools.enhanced import (
        tool,
        EnhancedToolDefinition,
        ToolChoice,
        ToolContext,
        ToolValidationError,
        inject_context,
        create_tool_from_function,
    )
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError:
    # Fallback if dependencies not available
    tool = register  # Use basic register as fallback
    ENHANCED_TOOLS_AVAILABLE = False

__all__ = [
    # Core types
    "ToolDefinition",
    "ToolCall", 
    "ToolResult",
    "ToolCallResponse",
    "ToolCallRequest",  # Legacy alias
    "function_to_tool_definition",
    
    # Handler
    "UniversalToolHandler",
    "create_handler",
    
    # Parser
    "detect_tool_calls",
    "parse_tool_calls",
    "format_tool_prompt",
    
    # Registry
    "ToolRegistry",
    "register",
    "get_registry",
    "execute_tool",
    "execute_tools",
    
    # Enhanced features
    "tool",  # Enhanced decorator
    "ENHANCED_TOOLS_AVAILABLE",  # Feature flag
]

# Conditionally add enhanced exports if available
if ENHANCED_TOOLS_AVAILABLE:
    __all__.extend([
        "EnhancedToolDefinition",
        "ToolChoice",
        "ToolContext", 
        "ToolValidationError",
        "inject_context",
        "create_tool_from_function",
    ])