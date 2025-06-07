"""
Tool registry and utility functions.

This module provides a registry for managing available tools and
utility functions for common operations.
"""

import logging
from typing import Dict, List, Callable, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from abstractllm.tools.core import ToolDefinition, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Features:
    - Register functions as tools
    - Execute tools with validation
    - Parallel tool execution support
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, Callable] = {}
        self._definitions: Dict[str, ToolDefinition] = {}
    
    def register(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Callable:
        """
        Register a function as a tool.
        
        Can be used as a decorator:
        ```
        @registry.register
        def my_tool(param: str) -> str:
            return f"Result: {param}"
        ```
        
        Or directly:
        ```
        registry.register(my_function, name="custom_name")
        ```
        
        Args:
            func: Function to register
            name: Optional custom name
            description: Optional custom description
            
        Returns:
            The original function (for decorator usage)
        """
        def decorator(f: Callable) -> Callable:
            tool_def = ToolDefinition.from_function(f)
            
            # Override name/description if provided
            if name:
                tool_def.name = name
            if description:
                tool_def.description = description
            
            self._tools[tool_def.name] = f
            self._definitions[tool_def.name] = tool_def
            
            logger.debug(f"Registered tool: {tool_def.name}")
            return f
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool function by name."""
        return self._tools.get(name)
    
    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._definitions.get(name)
    
    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tool definitions."""
        return list(self._definitions.values())
    
    def execute(
        self,
        tool_call: ToolCall,
        validate: bool = True,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """
        Execute a single tool call.
        
        Args:
            tool_call: Tool call to execute
            validate: Whether to validate arguments
            timeout: Optional execution timeout
            
        Returns:
            ToolResult with output or error
        """
        tool_func = self._tools.get(tool_call.name)
        if not tool_func:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=f"Unknown tool: {tool_call.name}"
            )
        
        try:
            # Validate arguments if requested
            if validate:
                tool_def = self._definitions[tool_call.name]
                # Simple validation - check required parameters
                required = tool_def.parameters.get("required", [])
                missing = [p for p in required if p not in tool_call.arguments]
                if missing:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        output=None,
                        error=f"Missing required parameters: {missing}"
                    )
            
            # Execute tool
            if timeout:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(tool_func, **tool_call.arguments)
                    output = future.result(timeout=timeout)
            else:
                output = tool_func(**tool_call.arguments)
            
            return ToolResult(
                tool_call_id=tool_call.id,
                output=output,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.name}: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=str(e)
            )
    
    def execute_parallel(
        self,
        tool_calls: List[ToolCall],
        max_workers: int = 5,
        timeout: Optional[float] = None
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls in parallel.
        
        Args:
            tool_calls: List of tool calls
            max_workers: Maximum parallel workers
            timeout: Optional timeout per tool
            
        Returns:
            List of results in same order as input
        """
        if not tool_calls:
            return []
        
        # Single tool - just execute directly
        if len(tool_calls) == 1:
            return [self.execute(tool_calls[0], timeout=timeout)]
        
        # Multiple tools - execute in parallel
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_call = {
                executor.submit(self.execute, tc, timeout=timeout): tc
                for tc in tool_calls
            }
            
            # Collect results
            for future in as_completed(future_to_call):
                tool_call = future_to_call[future]
                try:
                    result = future.result()
                    results[tool_call.id] = result
                except Exception as e:
                    logger.error(f"Error in parallel execution: {e}")
                    results[tool_call.id] = ToolResult(
                        tool_call_id=tool_call.id,
                        output=None,
                        error=str(e)
                    )
        
        # Return in original order
        return [results[tc.id] for tc in tool_calls]
    
    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        self._definitions.clear()


# Global registry instance
_global_registry = ToolRegistry()


def register(func: Optional[Callable] = None, **kwargs) -> Callable:
    """Register a tool in the global registry."""
    return _global_registry.register(func, **kwargs)


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def execute_tool(tool_call: ToolCall, **kwargs) -> ToolResult:
    """Execute a tool using the global registry."""
    return _global_registry.execute(tool_call, **kwargs)


def execute_tools(tool_calls: List[ToolCall], **kwargs) -> List[ToolResult]:
    """Execute multiple tools using the global registry."""
    return _global_registry.execute_parallel(tool_calls, **kwargs)