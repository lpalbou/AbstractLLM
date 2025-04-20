"""
Integration tests for streaming tool calls in AbstractLLM.

This module tests streaming functionality with tool calls, including:
- Detecting tool calls during streaming
- Pausing streams for tool execution
- Resuming streams after tool execution
- Handling multiple tool calls in a single stream
- Error handling during streaming tool execution
"""

import os
import time
import pytest
from typing import Dict, List, Any, Callable, Generator

from abstractllm.factory import create_llm
from abstractllm.session import Session
from abstractllm.types import GenerateResponse


# Test tool implementations
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform a basic calculation.
    
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
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        A dictionary with weather information
    """
    # Simulate weather API call with realistic but fake data
    weather_data = {
        "San Francisco, CA": {"temp": 15, "condition": "Foggy"},
        "New York, NY": {"temp": 22, "condition": "Sunny"},
        "Chicago, IL": {"temp": 5, "condition": "Windy"},
        "Miami, FL": {"temp": 28, "condition": "Humid"},
        "Seattle, WA": {"temp": 10, "condition": "Rainy"},
    }
    
    # Get weather data or default
    data = weather_data.get(location, {"temp": 20, "condition": "Clear"})
    
    # Convert temperature if needed
    temp = data["temp"]
    if unit.lower() == "fahrenheit":
        temp = (temp * 9/5) + 32
    
    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
        "humidity": 65,
        "timestamp": time.time()
    }


def search_knowledge_base(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search a knowledge base for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        A list of matching documents
    """
    # Simulate knowledge base with some documents
    kb = [
        {"id": "doc1", "title": "Introduction to Python", "content": "Python is a high-level programming language..."},
        {"id": "doc2", "title": "Machine Learning Basics", "content": "Machine learning is a subset of AI..."},
        {"id": "doc3", "title": "Web Development", "content": "Web development involves creating websites..."},
        {"id": "doc4", "title": "Data Science Overview", "content": "Data science combines statistics and programming..."},
        {"id": "doc5", "title": "Cloud Computing", "content": "Cloud computing provides services over the internet..."},
    ]
    
    # Simple search implementation
    results = []
    for doc in kb:
        if query.lower() in doc["title"].lower() or query.lower() in doc["content"].lower():
            results.append(doc)
            if len(results) >= max_results:
                break
    
    return results


@pytest.mark.api_call
class TestStreamingBasics:
    """Tests for basic streaming functionality with tools."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_detect_tool_calls_during_streaming(self):
        """Test detection of tool calls during streaming."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[calculator]
        )
        
        # Add a user message
        session.add_message(
            role="user",
            content="What is 25 multiplied by 16?"
        )
        
        # Start streaming with tools
        tool_detected = False
        content_chunks = []
        tool_functions = {"calculator": calculator}
        
        # Use the streaming API
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model="gpt-4"
        ):
            # Track content chunks
            if isinstance(chunk, str):
                content_chunks.append(chunk)
            
            # Check if chunk contains tool call
            elif isinstance(chunk, dict) and "tool_call" in chunk:
                tool_detected = True
                # Execute the tool
                session.execute_tool_call(chunk["tool_call"], tool_functions)
        
        # Verify tool was detected in the stream
        assert tool_detected, "Tool call should be detected during streaming"
        
        # Verify content was received
        assert len(content_chunks) > 0, "Content chunks should be received"
        
        # Check session history for tool results
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify that tools were used and results added to history
        assert len(assistant_msgs) > 0
        if assistant_msgs and assistant_msgs[0].tool_results:
            tool_results = assistant_msgs[0].tool_results
            assert len(tool_results) > 0
    
    @pytest.mark.skipif(os.environ.get("ANTHROPIC_API_KEY") is None, 
                      reason="Anthropic API key not available")
    def test_anthropic_streaming_with_tools(self):
        """Test streaming with tools using Anthropic provider."""
        # Create a provider
        provider = create_llm("anthropic", api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[calculator, get_weather]
        )
        
        # Add a user message
        session.add_message(
            role="user",
            content="What's the weather in Seattle, and can you multiply the temperature by 2?"
        )
        
        # Track streaming behavior
        tool_detected = False
        content_chunks = []
        tool_functions = {
            "calculator": calculator,
            "get_weather": get_weather
        }
        
        # Use the streaming API
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model="claude-3-opus-20240229"  # Use a model that supports tools
        ):
            # Track content chunks
            if isinstance(chunk, str):
                content_chunks.append(chunk)
            
            # Check if chunk contains tool call
            elif isinstance(chunk, dict) and "tool_call" in chunk:
                tool_detected = True
                # Execute the tool
                session.execute_tool_call(chunk["tool_call"], tool_functions)
        
        # Verify tool was detected in the stream
        assert tool_detected, "Tool call should be detected during streaming"
        
        # Verify content was received
        assert len(content_chunks) > 0, "Content chunks should be received"
        
        # Check final content has temperature information
        full_content = "".join(content_chunks)
        assert "Seattle" in full_content or "temperature" in full_content
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_stream_pause_resume(self):
        """Test pausing and resuming a stream for tool execution."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[calculator, get_weather]
        )
        
        # Add a user message requiring multiple tools
        session.add_message(
            role="user",
            content="What's the weather in Miami and New York? What's the temperature difference?"
        )
        
        # Track streaming behavior
        tool_calls = []
        content_chunks = []
        resumed_after_tool = False
        tool_functions = {
            "calculator": calculator,
            "get_weather": get_weather
        }
        
        # Use the streaming API
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model="gpt-4"
        ):
            # Track content chunks
            if isinstance(chunk, str):
                content_chunks.append(chunk)
                # Check if we resumed after a tool call
                if tool_calls and not resumed_after_tool:
                    resumed_after_tool = True
            
            # Check if chunk contains tool call
            elif isinstance(chunk, dict) and "tool_call" in chunk:
                tool_calls.append(chunk["tool_call"])
                # Execute the tool
                session.execute_tool_call(chunk["tool_call"], tool_functions)
        
        # Verify at least one tool was called
        assert len(tool_calls) > 0, "At least one tool call should be detected"
        
        # Verify the stream resumed after tool execution
        assert resumed_after_tool, "Stream should resume after tool execution"
        
        # Verify content about both cities and temperature difference
        full_content = "".join(content_chunks)
        assert "Miami" in full_content and "New York" in full_content
        assert "difference" in full_content.lower() or "degrees" in full_content.lower()


@pytest.mark.api_call
class TestMultipleToolsStreaming:
    """Tests for handling multiple tool calls in a stream."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_multiple_tool_calls_in_stream(self):
        """Test handling multiple tool calls in a single stream."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[calculator, get_weather, search_knowledge_base]
        )
        
        # Add a user message requiring multiple tools
        session.add_message(
            role="user",
            content="What's the weather in Seattle? Also, can you find some information about Python programming and calculate 15 * 7?"
        )
        
        # Track streaming behavior
        tool_calls = []
        content_chunks = []
        tool_functions = {
            "calculator": calculator,
            "get_weather": get_weather,
            "search_knowledge_base": search_knowledge_base
        }
        
        # Use the streaming API
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model="gpt-4"
        ):
            # Track content chunks
            if isinstance(chunk, str):
                content_chunks.append(chunk)
            
            # Check if chunk contains tool call
            elif isinstance(chunk, dict) and "tool_call" in chunk:
                tool_calls.append(chunk["tool_call"])
                # Execute the tool
                session.execute_tool_call(chunk["tool_call"], tool_functions)
        
        # Verify multiple tools were called
        # Convert tool_calls to a set of unique tool names
        tool_names = set()
        for call in tool_calls:
            tool_names.add(call.name if hasattr(call, "name") else call["name"])
        
        # At least 2 different tools should be called
        assert len(tool_names) >= 2, "At least two different tools should be called"
        
        # Verify content includes information from all tools
        full_content = "".join(content_chunks)
        assert "Seattle" in full_content or "weather" in full_content
        assert "Python" in full_content or "programming" in full_content
        assert "15" in full_content and "7" in full_content and "105" in full_content
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_sequential_tool_dependencies(self):
        """Test tools that depend on results from previous tools in a stream."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[calculator, get_weather]
        )
        
        # Add a user message requiring sequential tool usage
        session.add_message(
            role="user",
            content="What's the temperature in Miami and Chicago? What's their average?"
        )
        
        # Track streaming behavior
        tool_calls = []
        content_chunks = []
        tool_functions = {
            "calculator": calculator,
            "get_weather": get_weather
        }
        
        # Use the streaming API
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model="gpt-4"
        ):
            # Track content chunks
            if isinstance(chunk, str):
                content_chunks.append(chunk)
            
            # Check if chunk contains tool call
            elif isinstance(chunk, dict) and "tool_call" in chunk:
                tool_calls.append(chunk["tool_call"])
                # Execute the tool
                session.execute_tool_call(chunk["tool_call"], tool_functions)
        
        # Verify correct sequence of tool calls (weather first, then calculator)
        weather_calls = 0
        calc_calls = 0
        
        for call in tool_calls:
            name = call.name if hasattr(call, "name") else call["name"]
            if name == "get_weather":
                weather_calls += 1
            elif name == "calculator":
                calc_calls += 1
        
        # Verify at least two weather calls (Miami and Chicago) and one calculator call (for average)
        assert weather_calls >= 2, "Should call weather tool at least twice"
        assert calc_calls >= 1, "Should call calculator tool at least once"
        
        # Verify content includes average temperature
        full_content = "".join(content_chunks)
        assert "average" in full_content.lower() and "temperature" in full_content.lower()


@pytest.mark.api_call
class TestStreamingErrorHandling:
    """Tests for error handling during streaming with tools."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_tool_error_during_streaming(self):
        """Test handling tool errors during streaming."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[calculator]
        )
        
        # Add a user message that will trigger an error
        session.add_message(
            role="user",
            content="Calculate 10 divided by 0."
        )
        
        # Track streaming behavior
        error_handled = False
        content_chunks = []
        tool_functions = {"calculator": calculator}
        
        # Use the streaming API
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model="gpt-4"
        ):
            # Track content chunks
            if isinstance(chunk, str):
                content_chunks.append(chunk)
                # Check if error is mentioned in the content
                if "error" in chunk.lower() or "divide by zero" in chunk.lower():
                    error_handled = True
            
            # Check if chunk contains tool call
            elif isinstance(chunk, dict) and "tool_call" in chunk:
                # Execute the tool (will result in error)
                session.execute_tool_call(chunk["tool_call"], tool_functions)
        
        # Verify error was handled
        assert error_handled or any("divide by zero" in chunk.lower() 
                                    or "cannot divide by zero" in chunk.lower() 
                                    for chunk in content_chunks), "Tool error should be handled in streaming"
        
        # Check session history for tool results with error
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify that the error was recorded in history
        assert len(assistant_msgs) > 0
        if assistant_msgs and assistant_msgs[0].tool_results:
            tool_results = assistant_msgs[0].tool_results
            assert any("error" in result for result in tool_results), "Tool error should be recorded in history"
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_recovery_after_tool_error(self):
        """Test recovery after a tool error during streaming."""
        # Create a provider
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[calculator, get_weather]
        )
        
        # Add a user message with multiple tasks, one that will error
        session.add_message(
            role="user",
            content="Calculate 10 divided by 0, and also tell me the weather in New York."
        )
        
        # Track streaming behavior
        error_occurred = False
        recovered_after_error = False
        weather_called_after_error = False
        content_chunks = []
        tool_functions = {
            "calculator": calculator,
            "get_weather": get_weather
        }
        
        # Use the streaming API
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model="gpt-4"
        ):
            # Track content chunks
            if isinstance(chunk, str):
                content_chunks.append(chunk)
                # Check if content indicates recovery after error
                if error_occurred and ("weather" in chunk.lower() or "New York" in chunk):
                    recovered_after_error = True
            
            # Check if chunk contains tool call
            elif isinstance(chunk, dict) and "tool_call" in chunk:
                tool_call = chunk["tool_call"]
                name = tool_call.name if hasattr(tool_call, "name") else tool_call["name"]
                
                # Execute the tool
                try:
                    session.execute_tool_call(tool_call, tool_functions)
                except Exception:
                    error_occurred = True
                
                # Check if weather is called after calculator error
                if error_occurred and name == "get_weather":
                    weather_called_after_error = True
        
        # Verify recovery occurred
        assert error_occurred, "Calculator error should occur"
        assert recovered_after_error or weather_called_after_error, "Should recover after error"
        
        # Verify content includes weather information despite calculator error
        full_content = "".join(content_chunks)
        assert "New York" in full_content or "weather" in full_content


@pytest.mark.api_call
class TestStreamingProviderSpecifics:
    """Tests for provider-specific streaming behaviors with tools."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_openai_streaming_format(self):
        """Test OpenAI-specific streaming format with tools."""
        # Create a provider directly (not through session)
        provider = create_llm("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Convert functions to tools
        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform a basic calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["add", "subtract", "multiply", "divide"],
                                "description": "The operation to perform"
                            },
                            "a": {
                                "type": "number",
                                "description": "First number"
                            },
                            "b": {
                                "type": "number",
                                "description": "Second number"
                            }
                        },
                        "required": ["operation", "a", "b"]
                    }
                }
            }
        ]
        
        # Generate with streaming directly
        stream = provider.generate(
            prompt="What is 25 times 4?",
            tools=tool_defs,
            stream=True,
            model="gpt-4"
        )
        
        tool_call_detected = False
        tool_call_id = None
        tool_call_name = None
        tool_call_args = None
        
        # Process the stream
        for chunk in stream:
            if chunk.tool_calls:
                tool_call_detected = True
                for tool_call in chunk.tool_calls.tool_calls:
                    tool_call_id = tool_call.id
                    tool_call_name = tool_call.name
                    tool_call_args = tool_call.arguments
                break
        
        # Verify tool call was detected with correct format
        assert tool_call_detected, "Tool call should be detected in OpenAI stream"
        assert tool_call_id is not None, "Tool call should have an ID"
        assert tool_call_name == "calculator", "Tool name should be 'calculator'"
        assert "operation" in tool_call_args, "Tool args should include 'operation'"
        assert "a" in tool_call_args and "b" in tool_call_args, "Tool args should include 'a' and 'b'"
    
    @pytest.mark.skipif(os.environ.get("ANTHROPIC_API_KEY") is None, 
                      reason="Anthropic API key not available")
    def test_anthropic_streaming_format(self):
        """Test Anthropic-specific streaming format with tools."""
        # Create a provider directly (not through session)
        provider = create_llm("anthropic", api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        # Convert functions to tools for Anthropic
        tool_defs = [
            {
                "name": "calculator",
                "description": "Perform a basic calculation",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The operation to perform"
                        },
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        ]
        
        # Generate with streaming directly
        stream = provider.generate(
            prompt="What is 36 divided by 4?",
            tools=tool_defs,
            stream=True,
            model="claude-3-opus-20240229"
        )
        
        tool_call_detected = False
        tool_call_name = None
        tool_call_args = None
        
        # Process the stream
        for chunk in stream:
            if chunk.tool_calls:
                tool_call_detected = True
                for tool_call in chunk.tool_calls.tool_calls:
                    tool_call_name = tool_call.name
                    tool_call_args = tool_call.arguments
                break
        
        # Verify tool call was detected with correct format
        assert tool_call_detected, "Tool call should be detected in Anthropic stream"
        assert tool_call_name == "calculator", "Tool name should be 'calculator'"
        assert "operation" in tool_call_args, "Tool args should include 'operation'"
        assert "a" in tool_call_args and "b" in tool_call_args, "Tool args should include 'a' and 'b'" 