"""
Integration tests for AbstractLLM session-based tool functionality.

This module tests the Session class with tools, including:
- Session initialization with tool configurations
- Adding tools to sessions
- Full tool execution flow
- Multi-turn conversations with tool calls
- Error handling during tool execution
"""

import os
import time
import pytest
from typing import Dict, List, Any, Optional, Callable, Union
import json

from abstractllm.factory import create_llm
from abstractllm.session import Session
from abstractllm.tools import (
    ToolDefinition,
    ToolCall,
    ToolCallRequest,
    ToolResult,
    function_to_tool_definition,
)


# Real tool implementations for testing
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
    """Get weather information for a location.
    
    Args:
        location: City name or location
        unit: Temperature unit (celsius or fahrenheit)
        
    Returns:
        Dictionary with weather information
    """
    # Use a real weather API with rate limiting and caching
    import requests
    from functools import lru_cache
    
    @lru_cache(maxsize=100)
    def fetch_weather(loc, u):
        # Use a free weather API or a mock server for testing
        api_key = os.environ.get("WEATHER_API_KEY", "demo_key")
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={loc}&aqi=no"
        
        try:
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                # Fallback when API is unavailable or for CI environments
                return {
                    "temperature": 22.5 if u == "celsius" else 72.5,
                    "conditions": "Sunny",
                    "humidity": 45,
                    "location": loc,
                    "unit": u
                }
            data = response.json()
            return {
                "temperature": data["current"]["temp_c"] if u == "celsius" else data["current"]["temp_f"],
                "conditions": data["current"]["condition"]["text"],
                "humidity": data["current"]["humidity"],
                "location": loc,
                "unit": u
            }
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            # Fallback for any error
            return {
                "temperature": 22.5 if u == "celsius" else 72.5,
                "conditions": "Sunny",
                "humidity": 45,
                "location": loc,
                "unit": u
            }
    
    return fetch_weather(location, unit)


def slow_calculator(operation: str, a: float, b: float, delay: float = 1.0) -> float:
    """A deliberately slow calculator for testing timeouts.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        delay: How long to wait before returning (seconds)
        
    Returns:
        The result of the calculation
    """
    time.sleep(delay)
    return calculator(operation, a, b)


# Test Session initialization with tool configurations
class TestSessionInitialization:
    """Tests for initializing Sessions with tools."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_init_with_function_tools(self):
        """Test initializing a Session with function tools."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Initialize session with tools as functions
        session = Session(
            provider=provider,
            tools=[calculator, get_weather]
        )
        
        # Verify tools were properly added
        assert len(session.tools) == 2
        assert any(tool.name == "calculator" for tool in session.tools)
        assert any(tool.name == "get_weather" for tool in session.tools)
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_init_with_tool_definitions(self):
        """Test initializing a Session with ToolDefinition objects."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create tool definitions
        calculator_tool = function_to_tool_definition(calculator)
        weather_tool = function_to_tool_definition(get_weather)
        
        # Initialize session with tools as ToolDefinition objects
        session = Session(
            provider=provider,
            tools=[calculator_tool, weather_tool]
        )
        
        # Verify tools were properly added
        assert len(session.tools) == 2
        assert any(tool.name == "calculator" for tool in session.tools)
        assert any(tool.name == "get_weather" for tool in session.tools)
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_init_with_dict_tools(self):
        """Test initializing a Session with dictionary tool definitions."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create tool dictionaries
        calculator_dict = {
            "name": "calculator",
            "description": "Perform a basic calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
        
        # Initialize session with tools as dictionaries
        session = Session(
            provider=provider,
            tools=[calculator_dict]
        )
        
        # Verify tools were properly added
        assert len(session.tools) == 1
        assert session.tools[0].name == "calculator"


# Test adding tools to a Session
class TestSessionAddTool:
    """Tests for adding tools to Sessions."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_add_function_tool(self):
        """Test adding a function tool to a Session."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add a tool as a function
        session.add_tool(calculator)
        
        # Verify the tool was properly added
        assert len(session.tools) == 1
        assert session.tools[0].name == "calculator"
        assert "description" in session.tools[0].to_dict()
        assert "input_schema" in session.tools[0].to_dict()
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_add_tool_definition(self):
        """Test adding a ToolDefinition to a Session."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Create a tool definition
        calculator_tool = function_to_tool_definition(calculator)
        
        # Add the tool definition
        session.add_tool(calculator_tool)
        
        # Verify the tool was properly added
        assert len(session.tools) == 1
        assert session.tools[0].name == "calculator"
        assert session.tools[0] == calculator_tool
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_add_dict_tool(self):
        """Test adding a dictionary tool to a Session."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Create a tool dictionary
        calculator_dict = {
            "name": "calculator",
            "description": "Perform a basic calculation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
        
        # Add the tool dictionary
        session.add_tool(calculator_dict)
        
        # Verify the tool was properly added
        assert len(session.tools) == 1
        assert session.tools[0].name == "calculator"
        assert session.tools[0].description == "Perform a basic calculation"
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_add_multiple_tools(self):
        """Test adding multiple tools to a Session."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add multiple tools
        session.add_tool(calculator)
        session.add_tool(get_weather)
        
        # Verify the tools were properly added
        assert len(session.tools) == 2
        assert {tool.name for tool in session.tools} == {"calculator", "get_weather"}


# Test complete tool execution flow
@pytest.mark.api_call
class TestToolExecutionFlow:
    """Tests for the complete tool execution flow."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_simple_calculator_execution(self):
        """Test a simple calculator tool execution."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add calculator tool
        session.add_tool(calculator)
        
        # Add a user message
        session.add_message(
            role="user",
            content="What is 5 plus 3? Use the calculator tool."
        )
        
        # Generate a response with tools
        response = session.generate_with_tools(
            tool_functions={"calculator": calculator},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify the response
        assert response.content is not None
        assert "8" in response.content
        
        # Check session history for tool results
        messages = session.get_history()
        assert len(messages) >= 3  # User message + assistant response + (potentially) tool response
        
        # Find the assistant's message with the tool call
        assistant_msg = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify there was at least one tool call
        assert len(assistant_msg) > 0
        assert assistant_msg[0].tool_results is not None
        
        # Verify the tool call contained the operation and numbers
        tool_results = assistant_msg[0].tool_results
        assert len(tool_results) > 0
        assert any(
            "result" in result and result["result"] == 8 
            for result in tool_results
        )
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_weather_execution(self):
        """Test a weather tool execution."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add weather tool
        session.add_tool(get_weather)
        
        # Add a user message
        session.add_message(
            role="user",
            content="What's the weather like in San Francisco? Use the weather tool."
        )
        
        # Generate a response with tools
        response = session.generate_with_tools(
            tool_functions={"get_weather": get_weather},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify the response
        assert response.content is not None
        assert "San Francisco" in response.content
        
        # Check session history for tool results
        messages = session.get_history()
        assistant_msg = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify there was at least one tool call
        assert len(assistant_msg) > 0
        assert assistant_msg[0].tool_results is not None
        
        # Verify the tool results contain weather information
        tool_results = assistant_msg[0].tool_results
        assert len(tool_results) > 0
        assert any(
            "result" in result and 
            isinstance(result["result"], dict) and
            "temperature" in result["result"]
            for result in tool_results
        )
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_multiple_tool_execution(self):
        """Test execution with multiple tools available."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add multiple tools
        session.add_tool(calculator)
        session.add_tool(get_weather)
        
        # Add a user message
        session.add_message(
            role="user",
            content="What's 10 multiplied by 5? Also, what's the weather in London?"
        )
        
        # Generate a response with tools
        response = session.generate_with_tools(
            tool_functions={
                "calculator": calculator,
                "get_weather": get_weather
            },
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify the response
        assert response.content is not None
        assert "50" in response.content
        assert "London" in response.content
        
        # Check session history for tool results
        messages = session.get_history()
        assistant_msg = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify there was at least one tool call
        assert len(assistant_msg) > 0
        assert assistant_msg[0].tool_results is not None
        
        # Count the number of tool results
        tool_results = assistant_msg[0].tool_results
        assert len(tool_results) >= 1  # At least one tool should have been called


# Test multi-turn conversations with tool calls
@pytest.mark.api_call
class TestMultiTurnToolConversations:
    """Tests for multi-turn conversations with tool calls."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_two_turn_conversation(self):
        """Test a two-turn conversation with tool calls."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add calculator tool
        session.add_tool(calculator)
        
        # First turn
        session.add_message(
            role="user",
            content="What is 5 plus 3?"
        )
        
        response1 = session.generate_with_tools(
            tool_functions={"calculator": calculator},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify first response
        assert response1.content is not None
        assert "8" in response1.content
        
        # Second turn
        session.add_message(
            role="user",
            content="Now multiply that by 2."
        )
        
        response2 = session.generate_with_tools(
            tool_functions={"calculator": calculator},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify second response
        assert response2.content is not None
        assert "16" in response2.content
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_multi_tool_conversation(self):
        """Test a multi-turn conversation with different tools."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add multiple tools
        session.add_tool(calculator)
        session.add_tool(get_weather)
        
        # First turn - calculator
        session.add_message(
            role="user",
            content="What is 10 divided by 2?"
        )
        
        response1 = session.generate_with_tools(
            tool_functions={
                "calculator": calculator,
                "get_weather": get_weather
            },
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify first response
        assert response1.content is not None
        assert "5" in response1.content
        
        # Second turn - weather
        session.add_message(
            role="user",
            content="What's the weather like in Tokyo?"
        )
        
        response2 = session.generate_with_tools(
            tool_functions={
                "calculator": calculator,
                "get_weather": get_weather
            },
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify second response
        assert response2.content is not None
        assert "Tokyo" in response2.content
        
        # Check the conversation history
        messages = session.get_history()
        assert len(messages) >= 5  # 2 user messages + at least 3 assistant/system messages
        
        # Verify tool calls in the conversation
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        assert len(assistant_msgs) >= 1
        
        # Check that the tools were used appropriately
        tool_results = [result for msg in assistant_msgs for result in (msg.tool_results or [])]
        assert len(tool_results) >= 1


# Test error handling during tool execution
@pytest.mark.api_call
class TestToolExecutionErrors:
    """Tests for error handling during tool execution."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_divide_by_zero_error(self):
        """Test handling divide by zero errors."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add calculator tool
        session.add_tool(calculator)
        
        # Add a user message that will cause a divide by zero error
        session.add_message(
            role="user",
            content="What is 10 divided by 0? Use the calculator tool."
        )
        
        # Generate a response with tools
        response = session.generate_with_tools(
            tool_functions={"calculator": calculator},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify the error was handled properly
        assert response.content is not None
        assert "divide by zero" in response.content.lower() or "cannot divide" in response.content.lower()
        
        # Check the conversation history for error message
        messages = session.get_history()
        assistant_msg = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify there was at least one tool call
        assert len(assistant_msg) > 0
        assert assistant_msg[0].tool_results is not None
        
        # Verify the tool results contain an error message
        tool_results = assistant_msg[0].tool_results
        assert len(tool_results) > 0
        assert any(
            "error" in result and "divide by zero" in result["error"].lower()
            for result in tool_results
        )
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_invalid_operation_error(self):
        """Test handling invalid operation errors."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add calculator tool
        session.add_tool(calculator)
        
        # Add a user message that will cause an invalid operation error
        session.add_message(
            role="user",
            content="What is 10 modulo 3? Use the calculator tool."
        )
        
        # Generate a response with tools
        response = session.generate_with_tools(
            tool_functions={"calculator": calculator},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Check if the model used a valid operation or described the error
        assert response.content is not None
        
        # Some models might be smart enough to use a valid operation
        # Others might report the error
        messages = session.get_history()
        assistant_msg = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        if assistant_msg and assistant_msg[0].tool_results:
            tool_results = assistant_msg[0].tool_results
            if any("error" in result for result in tool_results):
                # If there was an error, verify it's related to invalid operation
                assert any(
                    "error" in result and "unknown operation" in result["error"].lower()
                    for result in tool_results
                )
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_missing_tool_function(self):
        """Test handling missing tool functions."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add calculator tool
        session.add_tool(calculator)
        
        # Add a user message
        session.add_message(
            role="user",
            content="What is 5 plus 3? Use the calculator tool."
        )
        
        # Generate a response with an empty tool functions dictionary
        response = session.generate_with_tools(
            tool_functions={},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify the error was handled properly
        assert response.content is not None
        assert "tool" in response.content.lower() and "not" in response.content.lower()
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_slow_tool_execution(self):
        """Test handling slow tool execution."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Add slow calculator tool
        session.add_tool(slow_calculator)
        
        # Add a user message
        session.add_message(
            role="user",
            content="What is 7 plus 8? Use the calculator."
        )
        
        # Generate a response with a slow tool
        response = session.generate_with_tools(
            tool_functions={"slow_calculator": slow_calculator},
            model="gpt-4"  # Use GPT-4 to ensure tool calls are supported
        )
        
        # Verify the response - tool should still execute successfully despite delay
        assert response.content is not None
        assert "15" in response.content 