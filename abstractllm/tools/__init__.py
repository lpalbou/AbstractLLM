"""
Tools package for AbstractLLM.

This package provides utilities for defining, converting, and validating tools/functions
that can be used with LLM providers that support tool calling.

Required dependencies:
    - docstring-parser: For extracting information from function docstrings
    - jsonschema: For validating tool definitions, arguments, and results
    - pydantic: For schema generation and validation

To install all required dependencies:
    pip install abstractllm[tools]
"""

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
    validate_tool_result,
    create_safe_tool_wrapper,
    ToolDefinitionValidationError,
    ToolArgumentValidationError,
    ToolResultValidationError,
    ValidationException
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
    "validate_tool_result",
    "create_safe_tool_wrapper",
    
    # Exceptions
    "ValidationException",
    "ToolDefinitionValidationError",
    "ToolArgumentValidationError",
    "ToolResultValidationError"
] 