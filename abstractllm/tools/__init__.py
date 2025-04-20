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