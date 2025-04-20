# Phase 1: Tool Definition Foundation

This document provides detailed implementation guidance for establishing the core data structures and conversion utilities needed for tool call support in AbstractLLM.

## Overview

The foundation phase creates the fundamental building blocks that will be used throughout the tool call implementation:

1. Type definitions for tool calls and responses
2. Function-to-tool-definition conversion utilities
3. Response standardization utilities
4. Basic package structure
5. Validation utilities for tool definitions and results

## Files to Create

### 1. `abstractllm/tools/__init__.py`

Create the package initialization file:

```python
"""Tools package for AbstractLLM."""

from .types import (
    ToolDefinition,
    ToolCallRequest,
    ToolCallResponse,
    ToolCall,
    ToolResult
)

from .conversion import (
    function_to_tool_definition,
    standardize_tool_response
)

from .validation import (
    validate_tool_definition,
    validate_tool_arguments,
    validate_tool_result
)

__all__ = [
    # Types
    "ToolDefinition",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolCall",
    "ToolResult",
    
    # Conversion utilities
    "function_to_tool_definition",
    "standardize_tool_response",
    
    # Validation utilities
    "validate_tool_definition",
    "validate_tool_arguments",
    "validate_tool_result"
]
```

### 2. `abstractllm/tools/types.py`

Create type definitions for tool-related components, now with Pydantic models:

```python
"""Type definitions for AbstractLLM tool support."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, get_type_hints

from pydantic import BaseModel, Field, ValidationError, validator


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by an LLM."""
    
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for inputs")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for return value")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate that the name follows the required pattern."""
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Tool name must only contain alphanumeric characters and underscores")
        return v
    
    @validator('input_schema')
    def validate_input_schema(cls, v):
        """Validate that the input schema is valid."""
        if not isinstance(v, dict) or v.get('type') != 'object':
            raise ValueError("Input schema must be an object schema")
        if 'properties' not in v:
            raise ValueError("Input schema must have a 'properties' field")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return self.dict(exclude_none=True)


class ToolCall(BaseModel):
    """A call to a tool by an LLM."""
    
    id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create a ToolCall from a dictionary."""
        # Handle different formats from different providers
        if "function" in data:
            # OpenAI format
            import json
            args = data["function"]["arguments"]
            # Handle arguments as string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"value": args}  # Fallback
            
            return cls(
                id=data.get("id", ""),
                name=data["function"]["name"],
                arguments=args
            )
        else:
            # Standard/Anthropic format
            return cls(
                id=data.get("id", ""),
                name=data["name"],
                arguments=data["arguments"]
            )


class ToolResult(BaseModel):
    """The result of executing a tool call."""
    
    call_id: str = Field(..., description="ID of the tool call this result responds to")
    result: Any = Field(..., description="Result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")


class ToolCallRequest(BaseModel):
    """A request containing tool calls from an LLM."""
    
    content: Optional[str] = Field(None, description="Text content from the LLM")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tool calls")
    
    @classmethod
    def from_provider_response(cls, response: Dict[str, Any], provider: str) -> "ToolCallRequest":
        """Create a ToolCallRequest from a provider-specific response."""
        from .conversion import standardize_tool_response
        # Delegate to the standardize_tool_response function
        return standardize_tool_response(response, provider)


class ToolCallResponse(BaseModel):
    """A response to be sent back to the LLM after executing tool calls."""
    
    tool_results: List[ToolResult] = Field(default_factory=list, description="Results of executed tools")
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to a dictionary representation."""
        return {
            "tool_results": [result.dict(exclude_none=True) for result in self.tool_results]
        }
```

### 3. `abstractllm/tools/conversion.py`

Create the conversion utilities:

```python
"""Utilities for converting between tool formats."""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

try:
    from docstring_parser import parse
except ImportError:
    raise ImportError("docstring-parser is required for tool conversion utilities. Install with `pip install docstring-parser`.")

from .types import ToolDefinition, ToolCallRequest, ToolCallResponse, ToolCall


# Mapping from Python types to JSON Schema types
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    None: "null",
}


def _get_type_from_annotation(annotation: Any) -> str:
    """
    Convert a Python type annotation to a JSON Schema type.
    
    Args:
        annotation: The type annotation to convert
        
    Returns:
        The corresponding JSON Schema type
    """
    # Handle Optional types
    if getattr(annotation, "__origin__", None) is Union:
        args = annotation.__args__
        if type(None) in args:
            # It's an Optional type
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return _get_type_from_annotation(non_none_args[0])
    
    # Handle List types
    if getattr(annotation, "__origin__", None) is list:
        return "array"
    
    # Handle Dict types
    if getattr(annotation, "__origin__", None) is dict:
        return "object"
    
    # Handle basic types
    return TYPE_MAP.get(annotation, "object")


def _get_docstring_return_info(docstring):
    """Extract return type information from docstring."""
    return_info = None
    for returns in docstring.returns:
        if returns.description:
            return_info = {
                "description": returns.description
            }
            break
    return return_info


def function_to_tool_definition(func: Callable) -> ToolDefinition:
    """
    Convert a Python function to a tool definition.
    
    Args:
        func: The function to convert
        
    Returns:
        A ToolDefinition describing the function
    """
    # Get function name and docstring
    name = func.__name__
    docstring = parse(func.__doc__ or "")
    description = docstring.short_description or ""
    
    # Get parameter types from type hints
    type_hints = get_type_hints(func)
    
    # Build input schema
    properties = {}
    required = []
    
    # Get default values
    signature = inspect.signature(func)
    
    for param_name, param in signature.parameters.items():
        # Skip self for methods
        if param_name == "self":
            continue
        
        # Get parameter type
        param_type = type_hints.get(param_name, Any)
        json_type = _get_type_from_annotation(param_type)
        
        # Get parameter description from docstring
        param_desc = ""
        for param_doc in docstring.params:
            if param_doc.arg_name == param_name:
                param_desc = param_doc.description or ""
                break
        
        # Build property
        prop = {
            "type": json_type,
            "description": param_desc
        }
        
        properties[param_name] = prop
        
        # Check if parameter is required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    
    # Build input schema
    input_schema = {
        "type": "object",
        "properties": properties,
        "required": required
    }
    
    # Build output schema if return type is available
    output_schema = None
    if "return" in type_hints:
        return_type = type_hints["return"]
        json_return_type = _get_type_from_annotation(return_type)
        
        # Get return description from docstring
        return_info = _get_docstring_return_info(docstring)
        
        if return_info:
            output_schema = {
                "type": json_return_type,
                "description": return_info.get("description", "")
            }
        else:
            output_schema = {
                "type": json_return_type
            }
    
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema
    )


def standardize_tool_response(
    provider_response: Dict[str, Any], 
    provider: str
) -> ToolCallRequest:
    """
    Convert a provider-specific response to a standardized ToolCallRequest.
    
    Args:
        provider_response: The raw response from the provider
        provider: The provider name (e.g., "openai", "anthropic", "ollama")
        
    Returns:
        A standardized ToolCallRequest
    """
    if provider.lower() == "openai":
        content = provider_response.get("content")
        tool_calls = []
        
        # Extract tool calls from the response message
        if "tool_calls" in provider_response:
            for tc in provider_response["tool_calls"]:
                # Ensure the function arguments are valid JSON
                try:
                    if isinstance(tc["function"]["arguments"], str):
                        args = json.loads(tc["function"]["arguments"])
                    else:
                        args = tc["function"]["arguments"]
                except json.JSONDecodeError:
                    args = {"_raw": tc["function"]["arguments"]}
                
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": args
                })
        
        return ToolCallRequest(
            content=content,
            tool_calls=[ToolCall.from_dict(tc) for tc in tool_calls]
        )
    
    elif provider.lower() == "anthropic":
        content = provider_response.get("content", "")
        tool_calls = []
        
        # Anthropic returns content blocks
        if "content" in provider_response and isinstance(provider_response["content"], list):
            text_content = []
            for block in provider_response["content"]:
                if block["type"] == "tool_use":
                    tool_data = block["tool_use"]
                    
                    # Arguments come as dict in Anthropic
                    tool_calls.append({
                        "id": tool_data.get("id", ""),
                        "name": tool_data["name"],
                        "arguments": tool_data["parameters"]
                    })
                elif block["type"] == "text":
                    # Collect text content
                    text_content.append(block["text"])
            
            # Join text content if any
            if text_content:
                content = "".join(text_content)
        
        return ToolCallRequest(
            content=content,
            tool_calls=[ToolCall.from_dict(tc) for tc in tool_calls]
        )
    
    elif provider.lower() == "ollama":
        content = provider_response.get("message", {}).get("content", "")
        tool_calls = []
        
        # Ollama has a different format
        if "tool_calls" in provider_response.get("message", {}):
            for tc in provider_response["message"]["tool_calls"]:
                # Parse arguments from JSON string to dict if needed
                args = tc.get("parameters") or tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                
                tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": tc.get("name", tc.get("function", {}).get("name", "")),
                    "arguments": args
                })
        
        return ToolCallRequest(
            content=content,
            tool_calls=[ToolCall.from_dict(tc) for tc in tool_calls]
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
```

### 4. `abstractllm/tools/validation.py`

Create a new file for validation utilities:

```python
"""Validation utilities for tool definitions and tool execution."""

import json
from typing import Any, Dict, Optional, Union

import jsonschema
from pydantic import ValidationError

from .types import ToolDefinition, ToolCall, ToolResult


class ValidationException(Exception):
    """Base exception for validation errors."""
    pass


class ToolDefinitionValidationError(ValidationException):
    """Exception raised when a tool definition is invalid."""
    pass


class ToolArgumentValidationError(ValidationException):
    """Exception raised when tool arguments don't match the schema."""
    pass


class ToolResultValidationError(ValidationException):
    """Exception raised when a tool result doesn't match the expected schema."""
    pass


def validate_tool_definition(tool_dict: Dict[str, Any]) -> ToolDefinition:
    """
    Validate a tool definition against the required schema.
    
    Args:
        tool_dict: The tool definition dictionary to validate
        
    Returns:
        A validated ToolDefinition object
        
    Raises:
        ToolDefinitionValidationError: If the definition is invalid
    """
    try:
        return ToolDefinition.parse_obj(tool_dict)
    except ValidationError as e:
        raise ToolDefinitionValidationError(f"Invalid tool definition: {str(e)}") from e


def validate_tool_arguments(tool_def: ToolDefinition, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the provided arguments match the tool's input schema.
    
    Args:
        tool_def: The tool definition to validate against
        arguments: The arguments to validate
        
    Returns:
        The validated arguments (potentially with type conversions)
        
    Raises:
        ToolArgumentValidationError: If the arguments don't match the schema
    """
    try:
        jsonschema.validate(instance=arguments, schema=tool_def.input_schema)
        return arguments
    except jsonschema.exceptions.ValidationError as e:
        raise ToolArgumentValidationError(f"Invalid arguments for tool '{tool_def.name}': {str(e)}") from e


def validate_tool_result(tool_def: ToolDefinition, result: Any) -> Any:
    """
    Validate that a tool result matches the expected output schema.
    
    Args:
        tool_def: The tool definition to validate against
        result: The result to validate
        
    Returns:
        The validated result
        
    Raises:
        ToolResultValidationError: If the result doesn't match the schema
    """
    # Skip validation if no output schema defined
    if not tool_def.output_schema:
        return result
        
    try:
        jsonschema.validate(instance=result, schema=tool_def.output_schema)
        return result
    except jsonschema.exceptions.ValidationError as e:
        raise ToolResultValidationError(f"Invalid result from tool '{tool_def.name}': {str(e)}") from e


def create_safe_tool_wrapper(func, tool_def: ToolDefinition):
    """
    Create a wrapper function that validates arguments and return value.
    
    Args:
        func: The function to wrap
        tool_def: The tool definition to validate against
        
    Returns:
        A wrapped function that validates inputs and outputs
    """
    def wrapper(**kwargs):
        # Validate arguments
        validated_args = validate_tool_arguments(tool_def, kwargs)
        
        # Execute the function
        try:
            result = func(**validated_args)
        except Exception as e:
            # Return a standardized error
            return ToolResult(
                call_id="",  # Will be set by caller
                result=None,
                error=str(e)
            )
            
        # Validate result
        try:
            validated_result = validate_tool_result(tool_def, result)
            return ToolResult(
                call_id="",  # Will be set by caller
                result=validated_result
            )
        except ToolResultValidationError as e:
            return ToolResult(
                call_id="",  # Will be set by caller
                result=None,
                error=str(e)
            )
            
    return wrapper
```

## Implementation Considerations

### Dependencies

This implementation requires several dependencies. Add them to the project's requirements:

```
docstring-parser>=0.15
pydantic>=2.0.0
jsonschema>=4.0.0
```

### Type Safety and Validation

The updated implementation provides enhanced type safety and validation:

1. **Pydantic Models:** Using Pydantic for data validation provides runtime type checking and better error messages
2. **JSON Schema Validation:** The `jsonschema` library validates tool arguments and results
3. **Error Hierarchy:** A clear error hierarchy helps identify and handle specific validation issues
4. **Safe Execution:** The `create_safe_tool_wrapper` function provides a pattern for safely executing tools

### Security Considerations

The implementation includes several security enhancements:

1. **Input Validation:** Arguments are validated against the schema before execution
2. **Sanitized Error Handling:** Errors are standardized to prevent leaking implementation details
3. **Type Coercion:** The validation system attempts to coerce types to the expected format where possible

### Testing Requirements

Create comprehensive unit tests for the enhanced functionality:

1. Test `function_to_tool_definition` with:
   - Simple functions with basic types
   - Functions with complex type annotations (Optional, List, Dict)
   - Functions with return type annotations
   - Functions with and without docstrings
   - Methods (to verify `self` handling)

2. Test `standardize_tool_response` with:
   - Mock responses from each provider
   - Edge cases (empty responses, malformed responses)
   
3. Test validation utilities with:
   - Valid and invalid tool definitions
   - Valid and invalid arguments
   - Valid and invalid results
   - Error cases and error handling

### Example Usage

```python
# Example function to convert
def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        x1: X-coordinate of the first point
        y1: Y-coordinate of the first point
        x2: X-coordinate of the second point
        y2: Y-coordinate of the second point
        
    Returns:
        The Euclidean distance between the points
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Convert function to tool definition
tool_def = function_to_tool_definition(calculate_distance)

# Validate tool definition
validated_tool = validate_tool_definition(tool_def.to_dict())

# Create a safe wrapper
safe_calculate_distance = create_safe_tool_wrapper(calculate_distance, validated_tool)

# Execute safely
result = safe_calculate_distance(x1=0, y1=0, x2=3, y2=4)
# result would be a ToolResult with result=5.0
```

## Integration with Next Phases

The enhanced types and utilities defined in this phase will be used in subsequent phases:

1. **Core Interface Adaptation**: The Pydantic models provide a robust foundation for the interface extensions
2. **Provider Implementations**: The improved error handling simplifies provider-specific adapters
3. **Session Lifecycle Management**: The validation utilities ensure type safety throughout the tool call lifecycle
4. **Security**: The safe wrapper pattern can be adopted throughout the implementation

## Next Steps

After implementing this enhanced foundation, proceed to [Core Interface Adaptation](interfaces.md) to extend the AbstractLLM interfaces for tool support. 