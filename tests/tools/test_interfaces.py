"""
Tests for AbstractLLM interfaces with tool support.

This module tests the interface extensions for tool support, including:
- AbstractLLMInterface tool methods
- BaseProvider tool handling
- GenerateResponse tool functionality
- Error handling for tool-related operations
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional, Union

from abstractllm.interface import AbstractLLMInterface
from abstractllm.providers.base import BaseProvider
from abstractllm.types import GenerateResponse
from abstractllm.enums import ModelCapability
from abstractllm.tools import (
    ToolDefinition,
    ToolCall,
    ToolCallRequest,
    ToolResult,
    function_to_tool_definition,
)


# Test functions to use as tools
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


# Mock provider for testing
class MockProvider(BaseProvider):
    """Mock provider for testing tool support."""
    
    def __init__(self, config=None, supports_tools=True):
        super().__init__(config)
        self.supports_tools = supports_tools
        self.process_tools_called = False
        self.tools_processed = None
    
    def get_capabilities(self):
        capabilities = super().get_capabilities()
        if self.supports_tools:
            capabilities[ModelCapability.FUNCTION_CALLING] = True
        return capabilities
    
    def _process_tools(self, tools):
        self.process_tools_called = True
        self.tools_processed = tools
        return super()._process_tools(tools)
    
    def generate(self, prompt, system_prompt=None, files=None, stream=False, tools=None, **kwargs):
        self._validate_tool_support(tools)
        processed_tools = None
        if tools:
            processed_tools = self._process_tools(tools)
        
        # Mock a response with or without tool calls
        if tools and processed_tools:
            tool_calls = [
                ToolCall(
                    id="mock_call_123",
                    name=processed_tools[0].name,
                    arguments={"a": 5, "b": 3}
                )
            ]
            tool_call_request = ToolCallRequest(
                content="Using the provided tool",
                tool_calls=tool_calls
            )
            return GenerateResponse(
                content="I'll use the calculator tool",
                tool_calls=tool_call_request
            )
        else:
            return GenerateResponse(content="Response without tools")
    
    async def generate_async(self, prompt, system_prompt=None, files=None, stream=False, tools=None, **kwargs):
        # For simplicity, just call the sync version
        return self.generate(prompt, system_prompt, files, stream, tools, **kwargs)


# Helper for real OpenAI provider tests
def skip_if_no_openai_key():
    """Skip the test if OPENAI_API_KEY is not available."""
    if os.environ.get("OPENAI_API_KEY") is None:
        pytest.skip("OPENAI_API_KEY environment variable not set")


# Tests for AbstractLLMInterface extensions
class TestAbstractLLMInterface:
    """Tests for the AbstractLLMInterface with tool support."""
    
    def test_interface_signature(self):
        """Test that AbstractLLMInterface methods include tool parameters."""
        # Check that both generate and generate_async methods have tools parameter
        assert "tools" in AbstractLLMInterface.generate.__code__.co_varnames
        assert "tools" in AbstractLLMInterface.generate_async.__code__.co_varnames
    
    def test_interface_docstrings(self):
        """Test that AbstractLLMInterface docstrings document tool usage."""
        assert "tools" in AbstractLLMInterface.generate.__doc__
        assert "tools" in AbstractLLMInterface.generate_async.__doc__
    
    def test_capabilities_include_tool_support(self):
        """Test that capabilities include tool-related flags."""
        capabilities = AbstractLLMInterface.get_capabilities(None)
        assert ModelCapability.FUNCTION_CALLING in capabilities
        assert ModelCapability.TOOL_USE in capabilities


# Tests for BaseProvider tool handling
class TestBaseProviderTools:
    """Tests for the BaseProvider's tool handling implementation."""
    
    def test_validate_tool_support_with_supported_tools(self):
        """Test validating tool support when the provider supports tools."""
        provider = MockProvider(supports_tools=True)
        tool_def = function_to_tool_definition(calculator)
        
        # This should not raise an exception
        provider._validate_tool_support([tool_def])
    
    def test_validate_tool_support_with_unsupported_tools(self):
        """Test validating tool support when the provider does not support tools."""
        provider = MockProvider(supports_tools=False)
        tool_def = function_to_tool_definition(calculator)
        
        # This should raise a ValueError
        with pytest.raises(ValueError):
            provider._validate_tool_support([tool_def])
    
    def test_process_tools_with_function(self):
        """Test processing tools with a Python function."""
        provider = MockProvider()
        
        result = provider._process_tools([calculator])
        
        assert provider.process_tools_called
        assert len(result) == 1
        assert isinstance(result[0], ToolDefinition)
        assert result[0].name == "calculator"
    
    def test_process_tools_with_tool_definition(self):
        """Test processing tools with a ToolDefinition."""
        provider = MockProvider()
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
        
        result = provider._process_tools([tool_def])
        
        assert provider.process_tools_called
        assert len(result) == 1
        assert result[0] == tool_def
    
    def test_process_tools_with_dict(self):
        """Test processing tools with a dictionary."""
        provider = MockProvider()
        tool_dict = {
            "name": "dict_tool",
            "description": "A tool from dict",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            }
        }
        
        result = provider._process_tools([tool_dict])
        
        assert provider.process_tools_called
        assert len(result) == 1
        assert isinstance(result[0], ToolDefinition)
        assert result[0].name == "dict_tool"
    
    def test_process_tools_with_invalid_type(self):
        """Test processing tools with an invalid type."""
        provider = MockProvider()
        
        with pytest.raises(ValueError):
            provider._process_tools([123])  # Integer is not a valid tool type
    
    def test_generate_with_tools(self):
        """Test the generate method with tools."""
        provider = MockProvider()
        tool_def = function_to_tool_definition(calculator)
        
        response = provider.generate("Calculate 5 + 3", tools=[tool_def])
        
        assert provider.process_tools_called
        assert response.has_tool_calls()
        assert len(response.tool_calls.tool_calls) == 1
        assert response.tool_calls.tool_calls[0].name == "calculator"


# Tests for GenerateResponse tool functionality
class TestGenerateResponseTools:
    """Tests for the GenerateResponse with tool calls."""
    
    def test_has_tool_calls_true(self):
        """Test has_tool_calls when tool calls are present."""
        tool_calls = ToolCallRequest(
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="test_tool",
                    arguments={"param1": "value1"}
                )
            ]
        )
        
        response = GenerateResponse(content="Test", tool_calls=tool_calls)
        
        assert response.has_tool_calls()
    
    def test_has_tool_calls_false_when_none(self):
        """Test has_tool_calls when tool_calls is None."""
        response = GenerateResponse(content="Test", tool_calls=None)
        
        assert not response.has_tool_calls()
    
    def test_has_tool_calls_false_when_empty(self):
        """Test has_tool_calls when tool_calls list is empty."""
        tool_calls = ToolCallRequest(tool_calls=[])
        
        response = GenerateResponse(content="Test", tool_calls=tool_calls)
        
        assert not response.has_tool_calls()
    
    def test_response_with_both_content_and_tool_calls(self):
        """Test a response with both content and tool calls."""
        tool_calls = ToolCallRequest(
            content="I'll help you calculate this.",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="calculator",
                    arguments={"operation": "add", "a": 5, "b": 3}
                )
            ]
        )
        
        response = GenerateResponse(
            content="Response content",
            tool_calls=tool_calls
        )
        
        assert response.content == "Response content"
        assert response.has_tool_calls()
        assert response.tool_calls.content == "I'll help you calculate this."
        assert response.tool_calls.tool_calls[0].name == "calculator"


# Real provider tests with OpenAI (when available)
class TestRealProviderIntegration:
    """Integration tests with real provider implementations."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_openai_provider_tool_support(self):
        """Test OpenAI provider's tool support capability."""
        from abstractllm.providers.openai import OpenAIProvider
        
        provider = OpenAIProvider({"api_key": os.environ.get("OPENAI_API_KEY")})
        capabilities = provider.get_capabilities()
        
        assert capabilities.get(ModelCapability.FUNCTION_CALLING, False)
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_openai_tool_definition_conversion(self):
        """Test converting tool definitions to OpenAI format."""
        from abstractllm.providers.openai import OpenAIProvider
        
        provider = OpenAIProvider({"api_key": os.environ.get("OPENAI_API_KEY")})
        
        tool_def = function_to_tool_definition(calculator)
        processed_tools = provider._process_tools([tool_def])
        
        assert len(processed_tools) > 0
        assert "type" in processed_tools[0]
        assert processed_tools[0]["type"] == "function"
        assert "function" in processed_tools[0]
        assert processed_tools[0]["function"]["name"] == "calculator"
        assert "description" in processed_tools[0]["function"]
        assert "parameters" in processed_tools[0]["function"]


class TestErrorHandling:
    """Tests for tool-related error handling."""
    
    def test_error_with_invalid_tool_definition(self):
        """Test error handling with an invalid tool definition."""
        provider = MockProvider()
        
        with pytest.raises(ValueError):
            provider.generate("Test", tools=[{"invalid": "tool"}])
    
    def test_error_with_unsupported_provider(self):
        """Test error handling when tools are used with an unsupported provider."""
        provider = MockProvider(supports_tools=False)
        tool_def = function_to_tool_definition(calculator)
        
        with pytest.raises(ValueError):
            provider.generate("Test", tools=[tool_def])
    
    def test_tool_processing_without_tool_module(self):
        """Test error when the tools module is not available."""
        provider = MockProvider()
        
        # Patch TOOLS_AVAILABLE to False
        with patch("abstractllm.providers.base.TOOLS_AVAILABLE", False):
            with pytest.raises(ValueError, match="Tool support is not available"):
                provider._process_tools([calculator])
    
    def test_error_when_tool_names_collide(self):
        """Test error handling when two tools have the same name."""
        provider = MockProvider()
        
        def calculator2(x: int, y: int) -> int:
            """A different calculator function with the same name."""
            return x + y
        
        # Modify the name attribute of the function
        calculator2.__name__ = "calculator"
        
        tool1 = function_to_tool_definition(calculator)
        tool2 = function_to_tool_definition(calculator2)
        
        # This should work (the second tool will replace the first)
        result = provider._process_tools([tool1, tool2])
        assert len(result) == 2 