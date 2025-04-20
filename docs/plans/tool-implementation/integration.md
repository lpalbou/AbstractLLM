# Phase 5: Integration Testing

This document provides guidance for testing the complete tool calling implementation across all layers of the AbstractLLM framework.

## Overview

Integration testing ensures that all components of the tool calling implementation work correctly together, including:

1. Tool definitions and conversion
2. Provider-specific implementations
3. Session management with tools
4. End-to-end tool calling flows
5. Error handling and edge cases

## Test Strategy

### 1. Unit Tests for Each Component

First, implement unit tests for each component in isolation:

#### Tool Definition Tests

```python
# tests/test_tools.py

import json
import pytest
from typing import Any, Dict, List

from abstractllm.tools import (
    ToolDefinition,
    ToolCall,
    ToolCallRequest,
    function_to_tool_definition,
)

def test_function_to_tool_definition():
    """Test converting a Python function to a tool definition."""
    def add_numbers(a: int, b: int) -> int:
        """
        Add two numbers together.
        
        Args:
            a: The first number
            b: The second number
            
        Returns:
            The sum of the two numbers
        """
        return a + b
    
    tool_def = function_to_tool_definition(add_numbers)
    
    assert tool_def.name == "add_numbers"
    assert tool_def.description == "Add two numbers together."
    assert len(tool_def.parameters["properties"]) == 2
    assert "a" in tool_def.parameters["properties"]
    assert "b" in tool_def.parameters["properties"]
    assert tool_def.parameters["properties"]["a"]["type"] == "integer"
    assert tool_def.parameters["properties"]["b"]["type"] == "integer"
    assert "a" in tool_def.parameters["required"]
    assert "b" in tool_def.parameters["required"]

def test_tool_call_processing():
    """Test processing a tool call."""
    tool_call = ToolCall(
        id="call_123",
        name="add_numbers",
        arguments='{"a": 1, "b": 2}'
    )
    
    # Parse arguments
    args = json.loads(tool_call.arguments)
    
    assert args["a"] == 1
    assert args["b"] == 2
    
    # Execute the tool
    def add_numbers(a: int, b: int) -> int:
        return a + b
    
    result = add_numbers(**args)
    assert result == 3
```

#### Provider Implementation Tests

```python
# tests/providers/test_openai_tools.py

import json
import pytest
from unittest.mock import patch, MagicMock

from abstractllm.providers.openai import OpenAIProvider
from abstractllm.tools import ToolDefinition

def test_openai_convert_tool_definitions():
    """Test converting tool definitions to OpenAI format."""
    provider = OpenAIProvider(api_key="test_key")
    
    tool_def = ToolDefinition(
        name="get_weather",
        description="Get the current weather",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state"
                }
            },
            "required": ["location"]
        }
    )
    
    openai_tools = provider._convert_tool_definitions([tool_def])
    
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "get_weather"
    assert openai_tools[0]["function"]["description"] == "Get the current weather"
    assert "location" in openai_tools[0]["function"]["parameters"]["properties"]

@pytest.mark.asyncio
async def test_openai_extract_tool_calls():
    """Test extracting tool calls from OpenAI response."""
    provider = OpenAIProvider(api_key="test_key")
    
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4-0613",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll help you get the weather.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco, CA"}'
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    }
    
    tool_calls = provider._extract_tool_calls(mock_response)
    
    assert tool_calls is not None
    assert len(tool_calls.tool_calls) == 1
    assert tool_calls.tool_calls[0].id == "call_abc123"
    assert tool_calls.tool_calls[0].name == "get_weather"
    assert json.loads(tool_calls.tool_calls[0].arguments)["location"] == "San Francisco, CA"
```

#### Session with Tools Tests

```python
# tests/test_session_tools.py

import json
import pytest
from unittest.mock import patch, MagicMock

from abstractllm.session import Session
from abstractllm.tools import ToolDefinition, ToolCall, ToolCallRequest
from abstractllm.types import GenerateResponse

class MockProvider:
    @property
    def capabilities(self):
        from abstractllm.enums import Capability
        return [Capability.CHAT, Capability.FUNCTION_CALLING]
        
    def generate(self, **kwargs):
        # Mock tool calls in the response
        mock_response = GenerateResponse(
            provider=self,
            model="test-model",
            content="I'll help you get the weather.",
            tool_calls=ToolCallRequest(
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="get_weather",
                        arguments='{"location": "London"}'
                    )
                ]
            )
        )
        return mock_response

def test_session_with_tools():
    """Test a session with tools."""
    # Create a session with a mock provider
    provider = MockProvider()
    session = Session(provider=provider)
    
    # Define a tool
    def get_weather(location: str) -> str:
        return f"The weather in {location} is cloudy."
    
    # Add the tool to the session
    session.add_tool(get_weather)
    
    # Add a message
    session.add_message(role="user", content="What's the weather in London?")
    
    # Generate a response with tools
    response = session.generate_with_tools(
        tool_functions={"get_weather": get_weather}
    )
    
    # The weather function should have been called
    assert "The weather in London is cloudy" in response.content
```

### 2. Integration Tests for Provider Combinations

Test each provider with tool calling support:

```python
# tests/integration/test_openai_tools.py

import os
import pytest
from abstractllm import AbstractLLM, Session
from abstractllm.tools import function_to_tool_definition

# Skip if no API key
pytest.mark.skipif(
    "OPENAI_API_KEY" not in os.environ,
    reason="OPENAI_API_KEY environment variable not set"
)
def test_openai_tool_calls():
    """Test tool calls with OpenAI."""
    # Create an LLM instance
    llm = AbstractLLM.create("openai", api_key=os.environ["OPENAI_API_KEY"])
    
    # Define a tool function
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        """
        Get the current weather for a location.
        
        Args:
            location: The city and state, e.g., "San Francisco, CA"
            unit: The unit of temperature, either "celsius" or "fahrenheit"
            
        Returns:
            A string describing the current weather
        """
        return f"The weather in {location} is 22°{unit[0].upper()}"
    
    # Create a session with the tool
    session = Session(
        provider=llm.provider,
        tools=[function_to_tool_definition(get_current_weather)]
    )
    
    # Add a user message
    session.add_message(
        role="user", 
        content="What's the weather in London?"
    )
    
    # Generate a response with tools
    response = session.generate_with_tools(
        tool_functions={"get_current_weather": get_current_weather},
        model="gpt-4"
    )
    
    # Check that the response mentions London weather
    assert "weather in London" in response.content.lower()
```

### 3. End-to-End Workflow Tests

Test the complete workflow from tool definition to execution:

```python
# tests/integration/test_workflow.py

import os
import pytest
from abstractllm import AbstractLLM, Session
from abstractllm.tools import function_to_tool_definition

def test_multi_tool_workflow():
    """Test a workflow with multiple tools."""
    # Skip if no API key
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Create an LLM instance
    llm = AbstractLLM.create("openai", api_key=os.environ["OPENAI_API_KEY"])
    
    # Define tool functions
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        """
        Get the current weather for a location.
        
        Args:
            location: The city and state, e.g., "San Francisco, CA"
            unit: The unit of temperature, either "celsius" or "fahrenheit"
            
        Returns:
            A string describing the current weather
        """
        return f"The weather in {location} is 22°{unit[0].upper()}"
    
    def get_stock_price(symbol: str) -> str:
        """
        Get the current stock price for a symbol.
        
        Args:
            symbol: The stock symbol, e.g., "AAPL"
            
        Returns:
            A string describing the current stock price
        """
        return f"The current stock price of {symbol} is $150.75"
    
    # Create a session with tools
    session = Session(
        provider=llm.provider,
        tools=[
            function_to_tool_definition(get_current_weather),
            function_to_tool_definition(get_stock_price)
        ]
    )
    
    # Add a user message that requires both tools
    session.add_message(
        role="user", 
        content="What's the weather in London and the stock price for AAPL?"
    )
    
    # Generate a response with tools
    response = session.generate_with_tools(
        tool_functions={
            "get_current_weather": get_current_weather,
            "get_stock_price": get_stock_price
        },
        model="gpt-4"
    )
    
    # Check that the response mentions both London weather and AAPL stock
    assert "weather in London" in response.content.lower()
    assert "aapl" in response.content.lower() and "$150.75" in response.content
```

### 4. Error Handling Tests

Test how the system handles errors:

```python
# tests/test_error_handling.py

import json
import pytest
from unittest.mock import patch, MagicMock

from abstractllm.session import Session
from abstractllm.tools import ToolDefinition, ToolCall, ToolCallRequest
from abstractllm.types import GenerateResponse
from abstractllm.enums import Capability

class MockProvider:
    @property
    def capabilities(self):
        return [Capability.CHAT, Capability.FUNCTION_CALLING]
        
    def generate(self, **kwargs):
        # Mock tool calls in the response
        mock_response = GenerateResponse(
            provider=self,
            model="test-model",
            content="I'll help you get the weather.",
            tool_calls=ToolCallRequest(
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="get_weather",
                        arguments='{"location": "London"}'
                    )
                ]
            )
        )
        return mock_response

def test_missing_tool_function():
    """Test handling a missing tool function."""
    # Create a session with a mock provider
    provider = MockProvider()
    session = Session(provider=provider)
    
    # Add a tool definition but not the function
    tool_def = ToolDefinition(
        name="get_weather",
        description="Get the current weather",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state"
                }
            },
            "required": ["location"]
        }
    )
    session.add_tool(tool_def)
    
    # Add a message
    session.add_message(role="user", content="What's the weather in London?")
    
    # Generate a response with tools - should handle the missing function gracefully
    response = session.generate_with_tools(
        tool_functions={}  # Empty tool functions dict
    )
    
    # The response should contain an error message
    assert "error" in response.content.lower()

def test_invalid_arguments():
    """Test handling invalid arguments in a tool call."""
    # Create a session with a mock provider
    provider = MockProvider()
    session = Session(provider=provider)
    
    # Define a tool that expects a specific type
    def get_weather(location: str, temperature: int) -> str:
        return f"The weather in {location} is {temperature}°C."
    
    # Add the tool to the session
    session.add_tool(get_weather)
    
    # Override the provider's generate method to return invalid arguments
    def mock_generate(**kwargs):
        return GenerateResponse(
            provider=provider,
            model="test-model",
            content="I'll help you get the weather.",
            tool_calls=ToolCallRequest(
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="get_weather",
                        arguments='{"location": "London", "temperature": "warm"}'  # Invalid: temperature should be int
                    )
                ]
            )
        )
    
    provider.generate = mock_generate
    
    # Add a message
    session.add_message(role="user", content="What's the weather in London?")
    
    # Generate a response with tools - should handle the type error gracefully
    response = session.generate_with_tools(
        tool_functions={"get_weather": get_weather}
    )
    
    # The response should contain an error message
    assert "error" in response.content.lower()
```

### 5. Cross-Provider Tests

Test tool handling across different providers:

```python
# tests/integration/test_cross_provider.py

import os
import pytest
from abstractllm import AbstractLLM, Session
from abstractllm.tools import function_to_tool_definition

@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
def test_provider_tool_support(provider_name):
    """Test tool support across providers."""
    # Skip if API key not available
    api_key_var = f"{provider_name.upper()}_API_KEY"
    if api_key_var not in os.environ:
        pytest.skip(f"{api_key_var} environment variable not set")
    
    # Create an LLM instance
    llm = AbstractLLM.create(provider_name, api_key=os.environ[api_key_var])
    
    # Skip if provider doesn't support function calling
    from abstractllm.enums import Capability
    if Capability.FUNCTION_CALLING not in llm.provider.capabilities:
        pytest.skip(f"Provider {provider_name} does not support function calling")
    
    # Define a tool function
    def get_current_weather(location: str) -> str:
        """
        Get the current weather for a location.
        
        Args:
            location: The city and state, e.g., "San Francisco, CA"
            
        Returns:
            A string describing the current weather
        """
        return f"The weather in {location} is sunny."
    
    # Create a session with the tool
    session = Session(
        provider=llm.provider,
        tools=[function_to_tool_definition(get_current_weather)]
    )
    
    # Add a user message
    session.add_message(
        role="user", 
        content="What's the weather in Paris?"
    )
    
    # Get appropriate model for the provider
    model = None
    if provider_name == "openai":
        model = "gpt-4"
    elif provider_name == "anthropic":
        model = "claude-3-haiku-20240307"
    
    # Generate a response with tools
    response = session.generate_with_tools(
        tool_functions={"get_current_weather": get_current_weather},
        model=model
    )
    
    # Check that the response mentions Paris weather
    assert "paris" in response.content.lower()
    assert "sunny" in response.content.lower()
```

## Manual Testing Plan

In addition to automated tests, perform manual testing of the following scenarios:

1. **Multi-Turn Conversations with Tools**
   - Start a conversation that requires multiple tool calls
   - Verify tool calls are made correctly and results are incorporated

2. **Streaming with Tool Calls**
   - Test streaming responses with tool calls
   - Verify that tool calls are intercepted and processed correctly during streaming

3. **Tool Chaining**
   - Create a scenario where one tool call result leads to another tool call
   - Verify the correct sequence of tool calls and responses

4. **Edge Cases**
   - Test with very large tool results
   - Test with nested or complex tool arguments
   - Test with tools that have optional parameters

## Testing Against Known Issues

Test the implementation against known issues with similar implementations:

1. **Tool Call Streaming Issues**
   - Verify that tool calls are correctly processed during streaming
   - Check that streaming doesn't break when tool calls are present

2. **Argument Parsing**
   - Test with complex nested arguments
   - Test with arrays and other complex types

3. **Concurrent Tool Calls**
   - Test scenarios where multiple tool calls are made simultaneously
   - Verify that all tool calls are executed and results are correctly incorporated

## Documentation Testing

Ensure the documentation accurately reflects the implementation:

1. Check that all public API methods are documented
2. Verify that example code in the documentation works as expected
3. Test the README examples

## Next Steps

After completing integration testing:

1. Address any issues found during testing
2. Update documentation based on testing results
3. Prepare for release
4. Implement monitoring for tool usage in production

Proceed to [Final Checklist](checklist.md) for a final review before implementation. 