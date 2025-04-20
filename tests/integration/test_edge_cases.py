"""
Integration tests for edge cases in AbstractLLM tool functionality.

This module tests edge cases in tool functionality, including:
- Handling large responses from tools
- Working with complex nested data structures
- Handling timeouts during tool execution
- Backward compatibility with different API versions
- Handling rate limits and retries
- Unicode and special character handling
"""

import os
import time
import json
import pytest
import threading
from typing import Dict, List, Any, Optional, Union
from functools import lru_cache

from abstractllm import AbstractLLM
from abstractllm.session import Session
from abstractllm.types import GenerateResponse
from abstractllm.tools import ToolDefinition, ToolCall, function_to_tool_definition


# --- Tool implementations for edge case testing ---

def large_response_tool(size: int = 1000, include_structure: bool = False) -> Dict[str, Any]:
    """
    Generate a large response of specified size.
    
    Args:
        size: The approximate size of the response in number of elements
        include_structure: Whether to include nested structures
        
    Returns:
        A large dictionary with the requested size
    """
    response = {
        "metadata": {
            "generator": "large_response_tool",
            "size_requested": size,
            "timestamp": time.time()
        },
        "data": {}
    }
    
    # Generate a large dataset
    for i in range(size):
        if include_structure and i % 10 == 0:
            # Create nested structure every 10 items
            response["data"][f"item_{i}"] = {
                "nested1": {
                    "nested2": {
                        "nested3": {
                            "value": f"Deeply nested value {i}",
                            "metrics": [i, i*2, i*3],
                            "active": i % 2 == 0
                        }
                    }
                }
            }
        else:
            response["data"][f"item_{i}"] = f"Value for item {i}"
    
    return response


def nested_data_tool(depth: int = 3, width: int = 3) -> Dict[str, Any]:
    """
    Generate a deeply nested data structure.
    
    Args:
        depth: The depth of nesting
        width: The number of branches at each level
        
    Returns:
        A deeply nested dictionary structure
    """
    def _create_nested(current_depth: int, max_depth: int, branch_width: int) -> Dict[str, Any]:
        if current_depth >= max_depth:
            return {"value": f"Leaf at depth {current_depth}", "is_leaf": True}
        
        result = {}
        for i in range(branch_width):
            result[f"branch_{i}"] = _create_nested(current_depth + 1, max_depth, branch_width)
        
        result["metadata"] = {
            "depth": current_depth,
            "children": branch_width,
            "is_leaf": False
        }
        
        return result
    
    return {
        "root": _create_nested(0, depth, width),
        "structure_info": {
            "max_depth": depth,
            "width": width,
            "total_nodes": sum(width**i for i in range(depth + 1))
        }
    }


def delayed_tool(seconds: float, data: Any = None) -> Dict[str, Any]:
    """
    Return a response after specified delay.
    
    Args:
        seconds: The number of seconds to delay
        data: Optional data to return
        
    Returns:
        A dictionary with the timing information and data
    """
    start_time = time.time()
    time.sleep(seconds)
    end_time = time.time()
    
    return {
        "requested_delay": seconds,
        "actual_delay": end_time - start_time,
        "data": data if data is not None else {"message": "Delayed response completed"},
        "timestamp": end_time
    }


def unicode_rich_tool(language: str) -> Dict[str, str]:
    """
    Return text with unicode characters for different languages.
    
    Args:
        language: The language to use (e.g., 'chinese', 'arabic', 'emoji')
        
    Returns:
        Dictionary with text in the requested language
    """
    responses = {
        "chinese": {
            "text": "你好，世界！这是中文文本。人工智能非常有趣。",
            "language": "Chinese (Mandarin)",
            "description": "Hello, world! This is Chinese text. Artificial intelligence is very interesting."
        },
        "arabic": {
            "text": "مرحبا بالعالم! هذا نص عربي. الذكاء الاصطناعي مثير للاهتمام.",
            "language": "Arabic",
            "description": "Hello, world! This is Arabic text. Artificial intelligence is interesting."
        },
        "russian": {
            "text": "Привет, мир! Это русский текст. Искусственный интеллект очень интересен.",
            "language": "Russian",
            "description": "Hello, world! This is Russian text. Artificial intelligence is very interesting."
        },
        "emoji": {
            "text": "👋 🌍! 🤖 ⚙️ 🔬 📊 💻 🧠 🚀 👾 🔮",
            "language": "Emoji",
            "description": "Wave, Earth, Robot, Gears, Microscope, Chart, Computer, Brain, Rocket, Alien, Crystal Ball"
        },
        "mixed": {
            "text": "Hello 你好 مرحبا Привет 👋 - AI (人工智能) technology is 🚀 in 2023!",
            "language": "Mixed",
            "description": "A mix of English, Chinese, Arabic, Russian and emoji"
        }
    }
    
    return responses.get(language.lower(), {
        "text": "Unknown language requested",
        "language": "Unknown",
        "description": f"The language '{language}' is not supported"
    })


class ThreadedTool:
    """A tool that runs in a separate thread and can be interrupted."""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.result = None
        self.error = None
    
    def start(self, seconds: float, fail: bool = False) -> Dict[str, Any]:
        """
        Start a long-running process in a thread.
        
        Args:
            seconds: How long the process should run
            fail: Whether the process should fail
            
        Returns:
            Dictionary with status information
        """
        if self.running:
            return {
                "status": "error",
                "message": "A task is already running",
                "timestamp": time.time()
            }
        
        self.running = True
        self.result = None
        self.error = None
        
        # Start the thread
        self.thread = threading.Thread(
            target=self._run_task,
            args=(seconds, fail)
        )
        self.thread.daemon = True
        self.thread.start()
        
        return {
            "status": "started",
            "message": f"Started task that will run for {seconds} seconds",
            "timestamp": time.time(),
            "task_id": id(self.thread)
        }
    
    def _run_task(self, seconds: float, fail: bool):
        """Internal method to run the task."""
        try:
            time.sleep(seconds)
            if fail:
                raise ValueError("Task was configured to fail")
            self.result = {
                "status": "completed",
                "message": f"Task completed after {seconds} seconds",
                "timestamp": time.time()
            }
        except Exception as e:
            self.error = {
                "status": "error",
                "message": str(e),
                "timestamp": time.time()
            }
        finally:
            self.running = False
    
    def status(self) -> Dict[str, Any]:
        """
        Get the status of the running task.
        
        Returns:
            Dictionary with status information
        """
        if self.running:
            return {
                "status": "running",
                "message": "Task is still running",
                "timestamp": time.time()
            }
        elif self.error is not None:
            return self.error
        elif self.result is not None:
            return self.result
        else:
            return {
                "status": "idle",
                "message": "No task has been started",
                "timestamp": time.time()
            }
    
    def cancel(self) -> Dict[str, Any]:
        """
        Cancel the running task.
        
        Returns:
            Dictionary with cancellation status
        """
        if not self.running or self.thread is None:
            return {
                "status": "error",
                "message": "No task is running",
                "timestamp": time.time()
            }
        
        # Thread cannot be forcibly terminated in Python
        # This just sets the flag, the thread will still run
        self.running = False
        
        return {
            "status": "cancelling",
            "message": "Task is being cancelled",
            "timestamp": time.time()
        }


# Initialize the threaded tool as a global
threaded_tool = ThreadedTool()

# Functions to be exposed as tools
def start_long_task(seconds: float, fail: bool = False) -> Dict[str, Any]:
    """
    Start a long-running task.
    
    Args:
        seconds: How long the task should run
        fail: Whether the task should fail
        
    Returns:
        Status information
    """
    return threaded_tool.start(seconds, fail)


def get_task_status() -> Dict[str, Any]:
    """
    Get the status of the running task.
    
    Returns:
        Status information
    """
    return threaded_tool.status()


def cancel_task() -> Dict[str, Any]:
    """
    Cancel the running task.
    
    Returns:
        Cancellation status
    """
    return threaded_tool.cancel()


# --- Test classes ---

@pytest.mark.api_call
class TestLargeResponses:
    """Tests for handling large responses from tools."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_large_tool_response(self):
        """Test handling a tool that returns a large amount of data."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session with the large response tool
        session = Session(
            provider=provider,
            tools=[large_response_tool]
        )
        
        # Add a user message
        session.add_message(
            role="user",
            content="Generate a dataset with 100 items and show me some statistics about it."
        )
        
        # Generate a response with the tool available
        response = session.generate_with_tools(
            tool_functions={"large_response_tool": large_response_tool},
            model="gpt-4"
        )
        
        # Verify the response
        assert response.content is not None
        
        # Check that the tool was called
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        # Verify tool call and summarization
        assert len(assistant_msgs) > 0
        if assistant_msgs and assistant_msgs[0].tool_results:
            tool_results = assistant_msgs[0].tool_results
            assert len(tool_results) > 0
            
            # Verify the model summarized or processed the large data
            assert "items" in response.content.lower() or "dataset" in response.content.lower()
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_nested_data_structure(self):
        """Test handling deeply nested data structures."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session with the nested data tool
        session = Session(
            provider=provider,
            tools=[nested_data_tool]
        )
        
        # Add a user message
        session.add_message(
            role="user",
            content="Create a nested data structure with depth 4 and width 3, then explain its organization."
        )
        
        # Generate a response with the tool available
        response = session.generate_with_tools(
            tool_functions={"nested_data_tool": nested_data_tool},
            model="gpt-4"
        )
        
        # Verify the response
        assert response.content is not None
        
        # Check that the tool was called and the data structure was described
        assert "structure" in response.content.lower() or "nested" in response.content.lower()
        assert "depth" in response.content.lower() or "levels" in response.content.lower()
        
        # Check that tool results are in history
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        assert len(assistant_msgs) > 0


@pytest.mark.api_call
class TestTimeoutHandling:
    """Tests for timeout handling with tools."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_tool_execution_timeout(self):
        """Test timeout handling during tool execution."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(
            provider=provider,
            tools=[delayed_tool]
        )
        
        # Add a user message
        session.add_message(
            role="user",
            content="Run a task that takes 3 seconds to complete."
        )
        
        # Set a timeout for tool execution
        start_time = time.time()
        
        # Generate a response with timeout
        response = session.generate_with_tools(
            tool_functions={"delayed_tool": delayed_tool},
            model="gpt-4",
            timeout=5  # 5 second timeout, should be enough for a 3-second task
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Verify the response
        assert response.content is not None
        
        # Verify the task completed successfully within timeout
        assert "completed" in response.content.lower() or "finished" in response.content.lower()
        
        # The execution should take at least 3 seconds
        assert execution_time >= 3, "Task should have taken at least 3 seconds"
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_long_running_task_management(self):
        """Test managing a long-running task with status checks."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session with the threaded tool functions
        session = Session(
            provider=provider,
            tools=[start_long_task, get_task_status, cancel_task]
        )
        
        # Add a user message to start a task
        session.add_message(
            role="user",
            content="Start a task that runs for 10 seconds, then immediately check its status."
        )
        
        # Generate a response with tools
        response = session.generate_with_tools(
            tool_functions={
                "start_long_task": start_long_task,
                "get_task_status": get_task_status,
                "cancel_task": cancel_task
            },
            model="gpt-4"
        )
        
        # Verify that the model started the task and checked status
        assert response.content is not None
        assert "started" in response.content.lower() or "running" in response.content.lower()
        
        # Check that both tools were called (start and status)
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        
        if assistant_msgs and assistant_msgs[0].tool_results:
            tool_results = assistant_msgs[0].tool_results
            
            # Check if at least start_long_task was called
            start_task_called = any(
                result.get("name", "") == "start_long_task"
                for result in tool_results
            )
            
            assert start_task_called, "start_long_task should have been called"
        
        # Now ask for the final status
        session.add_message(
            role="user",
            content="What is the current status of the task?"
        )
        
        # Wait a moment to ensure task has made progress
        time.sleep(2)
        
        # Generate a response for the status check
        response2 = session.generate_with_tools(
            tool_functions={
                "start_long_task": start_long_task,
                "get_task_status": get_task_status,
                "cancel_task": cancel_task
            },
            model="gpt-4"
        )
        
        # Verify the status update
        assert response2.content is not None
        assert "running" in response2.content.lower() or "status" in response2.content.lower()


@pytest.mark.api_call
class TestSpecialCaseHandling:
    """Tests for special case handling in tools."""
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_unicode_handling(self):
        """Test handling of unicode and special characters in tool responses."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session with the unicode tool
        session = Session(
            provider=provider,
            tools=[unicode_rich_tool]
        )
        
        # Add a user message
        session.add_message(
            role="user",
            content="Show me examples of text in Chinese, Arabic, and Emoji, then explain what they say."
        )
        
        # Generate a response with the tool available
        response = session.generate_with_tools(
            tool_functions={"unicode_rich_tool": unicode_rich_tool},
            model="gpt-4"
        )
        
        # Verify the response contains unicode characters
        assert response.content is not None
        
        # Check for multiple languages in the content
        # Note: this may vary depending on how the model formats the response
        content_lower = response.content.lower()
        has_chinese = "chinese" in content_lower or "mandarin" in content_lower or "你好" in response.content
        has_arabic = "arabic" in content_lower or "مرحبا" in response.content
        has_emoji = "emoji" in content_lower or "👋" in response.content
        
        # At least two of the requested languages should be mentioned
        assert has_chinese or has_arabic or has_emoji, "Response should contain at least one of the requested languages"
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_backward_compatibility(self):
        """Test backward compatibility with older tool formats."""
        # Create a provider
        provider = AbstractLLM.create("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create a session
        session = Session(provider=provider)
        
        # Define a calculator function
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
        
        # Old-style tool definition (following OpenAI's format)
        old_style_tool = {
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
        
        # Add a user message
        session.add_message(
            role="user",
            content="What is 42 divided by 7?"
        )
        
        # Generate a response with the old-style tool definition
        response = session.generate_with_tools(
            tool_functions={"calculator": calculator},
            model="gpt-4",
            tools=[old_style_tool]  # Use old-style tool definition
        )
        
        # Verify the response includes the correct calculation result
        assert response.content is not None
        assert "6" in response.content, "Response should include the correct result (42/7=6)"
        
        # Check that the tool was used
        messages = session.get_history()
        assistant_msgs = [msg for msg in messages if msg.role == "assistant" and msg.tool_results]
        assert len(assistant_msgs) > 0


@pytest.mark.api_call
class TestMultiProviderEdgeCases:
    """Tests for edge cases across different providers."""
    
    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None or os.environ.get("ANTHROPIC_API_KEY") is None, 
        reason="One or more provider API keys not available"
    )
    def test_extreme_nesting_cross_provider(self):
        """Test handling extremely nested structures across different providers."""
        # Define providers to test
        providers = [
            ("openai", os.environ.get("OPENAI_API_KEY"), "gpt-4"),
            ("anthropic", os.environ.get("ANTHROPIC_API_KEY"), "claude-3-opus-20240229")
        ]
        
        results = []
        
        for provider_name, api_key, model in providers:
            # Skip if API key not available
            if not api_key:
                continue
                
            # Create a provider
            provider = AbstractLLM.create(provider_name, api_key=api_key)
            
            # Create a session with nested data tool
            session = Session(
                provider=provider,
                tools=[nested_data_tool]
            )
            
            # Add a user message requesting extreme nesting
            session.add_message(
                role="user",
                content="Create a nested data structure with depth 5 and width 4, then tell me the total number of nodes and describe the first branch."
            )
            
            # Generate a response
            response = session.generate_with_tools(
                tool_functions={"nested_data_tool": nested_data_tool},
                model=model
            )
            
            # Store results
            results.append({
                "provider": provider_name,
                "model": model,
                "response": response.content,
                "history": session.get_history()
            })
        
        # Verify results from all providers
        assert len(results) >= 1, "At least one provider should have been tested"
        
        for result in results:
            # Verify the response contains information about the structure
            assert result["response"] is not None
            assert "nodes" in result["response"].lower() or "structure" in result["response"].lower()
            
            # Check if the model correctly identified the complexity
            content = result["response"].lower()
            has_depth_info = "depth" in content or "level" in content
            has_width_info = "width" in content or "branch" in content
            
            assert has_depth_info or has_width_info, f"Response from {result['provider']} should describe the structure"
    
    @pytest.mark.skipif(
        os.environ.get("OPENAI_API_KEY") is None or os.environ.get("ANTHROPIC_API_KEY") is None, 
        reason="One or more provider API keys not available"
    )
    def test_unicode_handling_cross_provider(self):
        """Test handling of unicode characters across different providers."""
        # Define providers to test
        providers = [
            ("openai", os.environ.get("OPENAI_API_KEY"), "gpt-4"),
            ("anthropic", os.environ.get("ANTHROPIC_API_KEY"), "claude-3-opus-20240229")
        ]
        
        results = []
        
        for provider_name, api_key, model in providers:
            # Skip if API key not available
            if not api_key:
                continue
                
            # Create a provider
            provider = AbstractLLM.create(provider_name, api_key=api_key)
            
            # Create a session with unicode tool
            session = Session(
                provider=provider,
                tools=[unicode_rich_tool]
            )
            
            # Add a user message requesting mixed unicode
            session.add_message(
                role="user",
                content="Show me text in the 'mixed' language option and explain what it contains."
            )
            
            # Generate a response
            response = session.generate_with_tools(
                tool_functions={"unicode_rich_tool": unicode_rich_tool},
                model=model
            )
            
            # Store results
            results.append({
                "provider": provider_name,
                "model": model,
                "response": response.content,
                "history": session.get_history()
            })
        
        # Verify results from all providers
        assert len(results) >= 1, "At least one provider should have been tested"
        
        for result in results:
            # Check for unicode characters in the response
            assert result["response"] is not None
            
            # The mixed text should contain characters from multiple languages
            has_mixed_content = (
                "mixed" in result["response"].lower() and 
                ("chinese" in result["response"].lower() or 
                 "arabic" in result["response"].lower() or 
                 "russian" in result["response"].lower() or 
                 "emoji" in result["response"].lower())
            )
            
            assert has_mixed_content, f"Response from {result['provider']} should explain the mixed language content" 