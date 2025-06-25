"""
Core types and interfaces for universal tool support.

This module provides the fundamental building blocks for tool support
across all models and providers.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import json


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by an LLM."""
    
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_function(cls, func: Callable) -> "ToolDefinition":
        """Create a tool definition from a Python function."""
        import inspect
        from typing import get_type_hints
        
        # Get function metadata
        name = func.__name__
        doc = inspect.getdoc(func) or ""
        description = doc.split('\n')[0] if doc else f"Function {name}"
        
        # Get parameters
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            # Get type
            param_type = hints.get(param_name, Any)
            json_type = _python_type_to_json(param_type)
            
            properties[param_name] = {"type": json_type}
            
            # Check if required
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        
        return cls(name=name, description=description, parameters=parameters)


@dataclass
class ToolCall:
    """A single tool call request."""
    
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            import uuid
            self.id = f"call_{uuid.uuid4().hex[:8]}"


@dataclass
class ToolResult:
    """Result from executing a tool."""
    
    tool_call_id: str
    output: Any
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if the tool call was successful."""
        return self.error is None


@dataclass
class ToolCallResponse:
    """Response containing tool calls from an LLM."""
    
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


def _python_type_to_json(py_type: Any) -> str:
    """Convert Python type to JSON Schema type."""
    type_map = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    
    # Handle Optional types
    origin = getattr(py_type, "__origin__", None)
    if origin is Union:
        args = py_type.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json(non_none[0])
    
    return type_map.get(py_type, "string")


# Legacy alias for backward compatibility
ToolCallRequest = ToolCallResponse


def function_to_tool_definition(func: Callable) -> ToolDefinition:
    """Convert a function to a tool definition (legacy function)."""
    return ToolDefinition.from_function(func)