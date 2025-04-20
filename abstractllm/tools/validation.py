"""Validation utilities for tool definitions and tool execution."""

import json
import logging
from typing import Any, Dict, Optional, Union, Callable

try:
    import jsonschema
except ImportError:
    raise ImportError("jsonschema is required for tool validation. Install with `pip install jsonschema`.")

from .types import ToolDefinition, ToolCall, ToolResult

# Configure logger
logger = logging.getLogger("abstractllm.tools.validation")


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
    except Exception as e:
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


def create_safe_tool_wrapper(func: Callable, tool_def: ToolDefinition) -> Callable:
    """
    Create a wrapper function that validates arguments and return value.
    
    Args:
        func: The function to wrap
        tool_def: The tool definition to validate against
        
    Returns:
        A wrapped function that validates inputs and outputs
    """
    def wrapper(**kwargs) -> ToolResult:
        # Validate arguments
        try:
            validated_args = validate_tool_arguments(tool_def, kwargs)
        except ToolArgumentValidationError as e:
            return ToolResult(
                call_id="",  # Will be set by caller
                result=None,
                error=str(e)
            )
        
        # Execute the function
        try:
            result = func(**validated_args)
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_def.name}': {str(e)}")
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