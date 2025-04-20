"""Utilities for converting between tool formats."""

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

try:
    from docstring_parser import parse
except ImportError:
    raise ImportError("docstring-parser is required for tool conversion utilities. Install with `pip install docstring-parser`.")

from .types import ToolDefinition, ToolCallRequest, ToolCall

# Configure logger
logger = logging.getLogger("abstractllm.tools.conversion")

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
    
    # Check if returns is present and handle both iterable and single return cases
    if hasattr(docstring, 'returns'):
        if isinstance(docstring.returns, list) and docstring.returns:
            # Handle as a list
            for returns in docstring.returns:
                if returns.description:
                    return_info = {
                        "description": returns.description
                    }
                    break
        elif docstring.returns and hasattr(docstring.returns, 'description') and docstring.returns.description:
            # Handle as a single return object
            return_info = {
                "description": docstring.returns.description
            }
    
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
    from .types import ToolCallRequest, ToolCall
    
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
        content = ""
        tool_calls = []
        
        # Extract content first
        if isinstance(provider_response.get("message", {}), dict):
            content = provider_response["message"].get("content", "")
        
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
        logger.warning(f"Unsupported provider: {provider}. Returning empty ToolCallRequest.")
        return ToolCallRequest(content="", tool_calls=[]) 