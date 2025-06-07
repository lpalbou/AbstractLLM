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

# Prepare request with tools
request = handler.prepare_request(
    tools=[search_web],
    messages=[{"role": "user", "content": "What's the weather?"}]
)

# Parse response for tool calls
response = "I'll search for weather information. <function_call>{"name": "search_web", "arguments": {"query": "current weather"}}</function_call>"
parsed = handler.parse_response(response, mode=request["mode"])

# Execute tools if needed
if parsed.has_tool_calls():
    from abstractllm.tools import execute_tools
    results = execute_tools(parsed.tool_calls)
    formatted = handler.format_tool_results(results, mode=request["mode"])
```
"""

# Core types
from abstractllm.tools.core import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolCallResponse
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

__all__ = [
    # Core types
    "ToolDefinition",
    "ToolCall", 
    "ToolResult",
    "ToolCallResponse",
    
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
    "execute_tools"
]