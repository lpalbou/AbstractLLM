"""
Tests for the core tool functionality in AbstractLLM.

This module tests the foundation of the tool calling implementation, including:
- ToolDefinition, ToolCall, and ToolResult dataclasses
- function_to_tool_definition utility
- Standardization utilities for tool responses
- Validation utilities
"""

import json
import pytest
from typing import Dict, List, Optional, Union, Any

from abstractllm.tools import (
    ToolDefinition,
    ToolCall,
    ToolCallRequest,
    ToolCallResponse,
    ToolResult,
    function_to_tool_definition,
    standardize_tool_response,
    validate_tool_definition,
    validate_tool_arguments,
    validate_tool_result,
)


# Test functions for function_to_tool_definition

def calculator(operation: str, a: float, b: float) -> float:
    """Perform a basic calculation.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        The result of the calculation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """Get the current weather for a location.
    
    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        A dictionary with weather information
    """
    return {
        "location": location,
        "temperature": 22.5,
        "unit": unit,
        "condition": "Sunny",
        "humidity": 65,
    }


def complex_function(
    required_str: str,
    required_int: int,
    optional_float: Optional[float] = None,
    optional_list: Optional[List[str]] = None,
    optional_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A more complex function with various parameter types.
    
    Args:
        required_str: A required string parameter
        required_int: A required integer parameter
        optional_float: An optional float parameter
        optional_list: An optional list of strings
        optional_dict: An optional dictionary
        
    Returns:
        A dictionary containing all the provided parameters
    """
    result = {
        "required_str": required_str,
        "required_int": required_int,
    }
    
    if optional_float is not None:
        result["optional_float"] = optional_float
    
    if optional_list is not None:
        result["optional_list"] = optional_list
    
    if optional_dict is not None:
        result["optional_dict"] = optional_dict
    
    return result


# Unit Tests

class TestToolDefinition:
    """Tests for the ToolDefinition class."""
    
    def test_valid_tool_definition(self):
        """Test creating a valid ToolDefinition."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        )
        
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert "param1" in tool_def.input_schema["properties"]
    
    def test_invalid_name(self):
        """Test that invalid names are rejected."""
        with pytest.raises(ValueError):
            ToolDefinition(
                name="invalid name with spaces",
                description="Tool with invalid name",
                input_schema={
                    "type": "object",
                    "properties": {}
                }
            )
    
    def test_invalid_schema(self):
        """Test that invalid schemas are rejected."""
        with pytest.raises(ValueError):
            ToolDefinition(
                name="valid_name",
                description="Tool with invalid schema",
                input_schema={"type": "string"}  # Not an object schema
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        tool_def = ToolDefinition(
            name="dict_tool",
            description="A tool for testing dict conversion",
            input_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            }
        )
        
        tool_dict = tool_def.to_dict()
        assert isinstance(tool_dict, dict)
        assert tool_dict["name"] == "dict_tool"
        assert tool_dict["description"] == "A tool for testing dict conversion"
        assert "input_schema" in tool_dict
        assert "output_schema" in tool_dict


class TestToolCall:
    """Tests for the ToolCall class."""
    
    def test_create_tool_call(self):
        """Test creating a ToolCall directly."""
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={"param1": "value1"}
        )
        
        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments["param1"] == "value1"
    
    def test_from_dict_openai(self):
        """Test creating a ToolCall from an OpenAI-format dictionary."""
        openai_format = {
            "id": "call_456",
            "function": {
                "name": "test_tool",
                "arguments": '{"param1": "value1", "param2": 42}'
            }
        }
        
        tool_call = ToolCall.from_dict(openai_format)
        
        assert tool_call.id == "call_456"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments["param1"] == "value1"
        assert tool_call.arguments["param2"] == 42
    
    def test_from_dict_anthropic(self):
        """Test creating a ToolCall from an Anthropic-format dictionary."""
        anthropic_format = {
            "id": "call_789",
            "name": "test_tool",
            "arguments": {"param1": "value1", "param2": 42}
        }
        
        tool_call = ToolCall.from_dict(anthropic_format)
        
        assert tool_call.id == "call_789"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments["param1"] == "value1"
        assert tool_call.arguments["param2"] == 42
    
    def test_from_dict_invalid_json(self):
        """Test creating a ToolCall with invalid JSON arguments."""
        invalid_format = {
            "id": "call_invalid",
            "function": {
                "name": "test_tool",
                "arguments": "invalid json {["
            }
        }
        
        # Should not raise an exception, but should store the raw string
        tool_call = ToolCall.from_dict(invalid_format)
        
        assert tool_call.id == "call_invalid"
        assert tool_call.name == "test_tool"
        assert "value" in tool_call.arguments


class TestToolResult:
    """Tests for the ToolResult class."""
    
    def test_success_result(self):
        """Test creating a successful ToolResult."""
        tool_result = ToolResult(
            call_id="call_123",
            result={"status": "success", "data": "example"}
        )
        
        assert tool_result.call_id == "call_123"
        assert tool_result.result["status"] == "success"
        assert tool_result.error is None
    
    def test_error_result(self):
        """Test creating an error ToolResult."""
        tool_result = ToolResult(
            call_id="call_error",
            result=None,
            error="An error occurred during execution"
        )
        
        assert tool_result.call_id == "call_error"
        assert tool_result.result is None
        assert tool_result.error == "An error occurred during execution"


class TestFunctionToToolDefinition:
    """Tests for the function_to_tool_definition utility."""
    
    def test_simple_function(self):
        """Test converting a simple function to a tool definition."""
        def add(a: int, b: int) -> int:
            """Add two numbers.
            
            Args:
                a: First number
                b: Second number
                
            Returns:
                The sum of a and b
            """
            return a + b
        
        tool_def = function_to_tool_definition(add)
        
        assert tool_def.name == "add"
        assert "Add two numbers" in tool_def.description
        assert "properties" in tool_def.input_schema
        assert "a" in tool_def.input_schema["properties"]
        assert "b" in tool_def.input_schema["properties"]
        assert tool_def.input_schema["properties"]["a"]["type"] == "integer"
        assert tool_def.input_schema["properties"]["b"]["type"] == "integer"
        assert "a" in tool_def.input_schema["required"]
        assert "b" in tool_def.input_schema["required"]
        assert tool_def.output_schema["type"] == "integer"
    
    def test_calculator_function(self):
        """Test the calculator function conversion."""
        tool_def = function_to_tool_definition(calculator)
        
        assert tool_def.name == "calculator"
        assert "Perform a basic calculation" in tool_def.description
        assert len(tool_def.input_schema["properties"]) == 3
        assert tool_def.input_schema["properties"]["operation"]["type"] == "string"
        assert tool_def.input_schema["properties"]["a"]["type"] == "number"
        assert tool_def.input_schema["properties"]["b"]["type"] == "number"
        assert tool_def.output_schema["type"] == "number"
    
    def test_function_with_optional_params(self):
        """Test converting a function with optional parameters."""
        tool_def = function_to_tool_definition(get_weather)
        
        assert tool_def.name == "get_weather"
        assert "Get the current weather" in tool_def.description
        assert len(tool_def.input_schema["properties"]) == 2
        assert "location" in tool_def.input_schema["required"]
        assert "unit" not in tool_def.input_schema["required"]
        assert tool_def.output_schema["type"] == "object"
    
    def test_complex_function(self):
        """Test converting a complex function with various parameter types."""
        tool_def = function_to_tool_definition(complex_function)
        
        assert tool_def.name == "complex_function"
        assert len(tool_def.input_schema["properties"]) == 5
        assert "required_str" in tool_def.input_schema["required"]
        assert "required_int" in tool_def.input_schema["required"]
        assert "optional_float" not in tool_def.input_schema["required"]
        assert "optional_list" not in tool_def.input_schema["required"]
        assert "optional_dict" not in tool_def.input_schema["required"]
        assert tool_def.input_schema["properties"]["optional_list"]["type"] == "array"
        assert tool_def.input_schema["properties"]["optional_dict"]["type"] == "object"
    
    def test_function_without_docstring(self):
        """Test converting a function without a docstring."""
        def no_docs(x: int) -> int:
            return x * 2
        
        tool_def = function_to_tool_definition(no_docs)
        
        assert tool_def.name == "no_docs"
        assert tool_def.description == ""  # Empty description
        assert "x" in tool_def.input_schema["properties"]
        assert tool_def.input_schema["properties"]["x"]["type"] == "integer"


class TestStandardizeToolResponse:
    """Tests for the standardize_tool_response utility."""
    
    def test_openai_format(self):
        """Test standardizing an OpenAI format response."""
        openai_response = {
            "content": "I'll help you with that calculation.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"operation": "add", "a": 5, "b": 3}'
                    }
                }
            ]
        }
        
        result = standardize_tool_response(openai_response, "openai")
        
        assert result.content == "I'll help you with that calculation."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments["operation"] == "add"
        assert result.tool_calls[0].arguments["a"] == 5
        assert result.tool_calls[0].arguments["b"] == 3
    
    def test_anthropic_format(self):
        """Test standardizing an Anthropic format response."""
        anthropic_response = {
            "content": [
                {"type": "text", "text": "I'll help you with that calculation."},
                {
                    "type": "tool_use",
                    "tool_use": {
                        "id": "call_456",
                        "name": "calculator",
                        "parameters": {"operation": "add", "a": 5, "b": 3}
                    }
                }
            ]
        }
        
        result = standardize_tool_response(anthropic_response, "anthropic")
        
        assert "I'll help you with that calculation." in result.content
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_456"
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments["operation"] == "add"
        assert result.tool_calls[0].arguments["a"] == 5
        assert result.tool_calls[0].arguments["b"] == 3
    
    def test_ollama_format(self):
        """Test standardizing an Ollama format response."""
        ollama_response = {
            "message": {
                "content": "I'll help you with that calculation.",
                "tool_calls": [
                    {
                        "id": "call_789",
                        "name": "calculator",
                        "arguments": '{"operation": "add", "a": 5, "b": 3}'
                    }
                ]
            }
        }
        
        result = standardize_tool_response(ollama_response, "ollama")
        
        assert result.content == "I'll help you with that calculation."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_789"
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments["operation"] == "add"
        assert result.tool_calls[0].arguments["a"] == 5
        assert result.tool_calls[0].arguments["b"] == 3
    
    def test_invalid_provider(self):
        """Test with an invalid provider name."""
        invalid_response = {"content": "test"}
        result = standardize_tool_response(invalid_response, "invalid_provider")
        
        assert result.content == ""
        assert len(result.tool_calls) == 0


class TestValidationUtilities:
    """Tests for the validation utilities."""
    
    def test_validate_tool_definition(self):
        """Test validating a tool definition."""
        valid_def = {
            "name": "valid_tool",
            "description": "A valid tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        }
        
        result = validate_tool_definition(valid_def)
        assert isinstance(result, ToolDefinition)
        assert result.name == "valid_tool"
    
    def test_validate_invalid_tool_definition(self):
        """Test validating an invalid tool definition."""
        invalid_def = {
            "name": "invalid tool name",  # contains spaces
            "description": "An invalid tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        }
        
        with pytest.raises(Exception):
            validate_tool_definition(invalid_def)
    
    def test_validate_tool_arguments(self):
        """Test validating tool arguments."""
        tool_def = ToolDefinition(
            name="test_validation",
            description="A tool for testing validation",
            input_schema={
                "type": "object",
                "properties": {
                    "str_param": {"type": "string"},
                    "int_param": {"type": "integer"}
                },
                "required": ["str_param"]
            }
        )
        
        valid_args = {
            "str_param": "test",
            "int_param": 42
        }
        
        result = validate_tool_arguments(tool_def, valid_args)
        assert result == valid_args
    
    def test_validate_invalid_tool_arguments(self):
        """Test validating invalid tool arguments."""
        tool_def = ToolDefinition(
            name="test_validation",
            description="A tool for testing validation",
            input_schema={
                "type": "object",
                "properties": {
                    "str_param": {"type": "string"},
                    "int_param": {"type": "integer"}
                },
                "required": ["str_param", "int_param"]
            }
        )
        
        # Missing required parameter
        invalid_args = {
            "str_param": "test"
        }
        
        with pytest.raises(Exception):
            validate_tool_arguments(tool_def, invalid_args)
    
    def test_validate_tool_result(self):
        """Test validating a tool result."""
        tool_def = ToolDefinition(
            name="test_result_validation",
            description="A tool for testing result validation",
            input_schema={
                "type": "object",
                "properties": {}
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "value": {"type": "integer"}
                },
                "required": ["status", "value"]
            }
        )
        
        valid_result = {
            "status": "success",
            "value": 42
        }
        
        result = validate_tool_result(tool_def, valid_result)
        assert result == valid_result
    
    def test_validate_invalid_tool_result(self):
        """Test validating an invalid tool result."""
        tool_def = ToolDefinition(
            name="test_result_validation",
            description="A tool for testing result validation",
            input_schema={
                "type": "object",
                "properties": {}
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "value": {"type": "integer"}
                },
                "required": ["status", "value"]
            }
        )
        
        # Wrong type for value
        invalid_result = {
            "status": "success",
            "value": "not an integer"
        }
        
        with pytest.raises(Exception):
            validate_tool_result(tool_def, invalid_result)
    
    def test_validate_tool_result_no_schema(self):
        """Test validating a tool result with no output schema."""
        tool_def = ToolDefinition(
            name="test_no_schema",
            description="A tool with no output schema",
            input_schema={
                "type": "object",
                "properties": {}
            }
        )
        
        # Any result should be valid
        result = validate_tool_result(tool_def, "any result")
        assert result == "any result" 