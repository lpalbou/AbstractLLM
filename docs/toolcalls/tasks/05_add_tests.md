# Task 5: Add Tests for Tool Call Flow

## Overview

This task focuses on creating comprehensive tests to verify that the tool call flow works correctly in AbstractLLM. Tests should cover both the proper LLM-first flow and detect any attempts at direct tool execution.

## Background

Proper testing is essential to ensure that:
1. All tool calls are initiated by the LLM, not by pattern matching in the agent code
2. Tool results are properly passed back to the LLM for processing
3. The flow works correctly for different query types and tools
4. Error cases are handled appropriately

## Required Changes

### 1. Create Unit Tests for Direct Tool Execution Detection

Create tests that detect and fail when direct tool execution is attempted:

```python
# File: tests/test_direct_execution.py

import pytest
import re
from unittest.mock import patch, MagicMock

from abstractllm.session import Session
from basic_agent import BasicAgent

def test_detect_direct_tool_execution():
    """Test that direct tool execution is not happening."""
    
    # Mock read_file so we can detect if it's called directly
    with patch("basic_agent.read_file") as mock_read_file:
        # Set up the agent
        agent = BasicAgent(provider_name="openai")
        
        # Run a query that might trigger direct tool execution
        agent.run("Please read the file test.txt")
        
        # The read_file function should never be called directly
        # It should only be called through the tool execution framework
        mock_read_file.assert_not_called()

def test_proper_flow_is_used():
    """Test that the proper LLM-first flow is used."""
    
    # Mock the session's generate_with_tools method
    with patch("abstractllm.session.Session.generate_with_tools") as mock_generate:
        # Configure the mock to return a response
        mock_response = MagicMock()
        mock_response.content = "This is a test response"
        mock_response.has_tool_calls.return_value = False
        mock_generate.return_value = mock_response
        
        # Set up the agent
        agent = BasicAgent(provider_name="openai")
        
        # Run a query
        result = agent.run("Please read the file test.txt")
        
        # Verify that generate_with_tools was called
        mock_generate.assert_called_once()
        
        # Verify that the result comes from the mock response
        assert result == "This is a test response"

def test_no_conditional_path_for_file_requests():
    """Test that there's no conditional path for file-related requests."""
    
    # Read the source code of basic_agent.py
    with open("basic_agent.py", "r") as f:
        source_code = f.read()
    
    # Look for patterns that suggest conditional handling based on query content
    file_conditionals = re.findall(r"if.*['\"]file['\"].*in.*query", source_code)
    read_conditionals = re.findall(r"if.*['\"]read['\"].*in.*query", source_code)
    
    # There should be no such conditionals in the code
    assert len(file_conditionals) == 0, f"Found file conditionals: {file_conditionals}"
    assert len(read_conditionals) == 0, f"Found read conditionals: {read_conditionals}"
```

### 2. Create Integration Tests for the LLM-First Flow

Create integration tests for the complete LLM-first flow:

```python
# File: tests/integration/test_llm_first_flow.py

import pytest
import os
from unittest.mock import patch, MagicMock

from abstractllm.tools import ToolCallRequest, ToolCall
from abstractllm.types import GenerateResponse
from basic_agent import BasicAgent

class TestLLMFirstFlow:
    """Test the LLM-first tool call flow."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        with patch("abstractllm.session.Session") as mock_session_cls:
            # Create mock session instance
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            
            # Configure the session manager to return our mock
            with patch("abstractllm.session.SessionManager") as mock_manager:
                manager_instance = MagicMock()
                manager_instance.get_session.return_value = mock_session
                mock_manager.return_value = manager_instance
                
                yield mock_session
    
    def test_llm_first_file_reading(self, mock_session):
        """Test that file reading goes through the LLM-first flow."""
        # Prepare mock response with tool calls
        tool_call = ToolCall(
            id="call_123",
            name="read_file",
            arguments={"file_path": "test.txt"}
        )
        tool_calls_request = ToolCallRequest(
            content="I'll read that file for you",
            tool_calls=[tool_call]
        )
        mock_response = GenerateResponse(
            content="I'll read that file for you",
            tool_calls=tool_calls_request
        )
        
        # Mock the tool result
        mock_result = "This is the content of test.txt"
        
        # Configure the mock session
        mock_session.generate_with_tools.return_value = mock_response
        
        # Mock execute_tool_call to return our result
        def mock_execute(tool_call, tool_functions):
            return {
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "output": mock_result,
                "error": None
            }
        mock_session.execute_tool_call.side_effect = mock_execute
        
        # Also mock the final response after tool execution
        final_response = GenerateResponse(
            content=f"The content of the file is: {mock_result}"
        )
        mock_session.generate_with_tools.side_effect = [mock_response, final_response]
        
        # Create the agent
        agent = BasicAgent(provider_name="mock")
        
        # Run the query
        result = agent.run("Please read the file test.txt")
        
        # Verify that generate_with_tools was called twice
        assert mock_session.generate_with_tools.call_count == 2
        
        # Verify that execute_tool_call was called
        mock_session.execute_tool_call.assert_called_once()
        
        # Verify that the final response is what we expect
        assert result == "The content of the file is: This is the content of test.txt"
    
    def test_streaming_tool_call(self, mock_session):
        """Test tool calls in streaming mode."""
        # This test requires a more complex setup to mock streaming
        
        # Mock the streaming generator
        def mock_streaming_generator():
            # First yield some content
            yield "I'll read the file "
            
            # Then yield a tool call
            yield {
                "call_id": "call_123",
                "name": "read_file",
                "arguments": {"file_path": "test.txt"},
                "output": "This is the content of test.txt",
                "error": None
            }
            
            # Then yield the rest of the content
            yield "The content of the file is: This is the content of test.txt"
        
        # Configure the mock session
        mock_session.generate_with_tools_streaming.return_value = mock_streaming_generator()
        
        # Create the agent
        agent = BasicAgent(provider_name="mock")
        
        # Mock print to capture output
        with patch("builtins.print") as mock_print:
            # Run the streaming query
            agent.run_streaming("Please read the file test.txt")
            
            # Check that print was called with the expected content
            mock_print.assert_any_call("I'll read the file ", end="", flush=True)
            mock_print.assert_any_call("\n[Executing tool: read_file]\n", flush=True)
            mock_print.assert_any_call("The content of the file is: This is the content of test.txt", end="", flush=True)
    
    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                       reason="OpenAI API key not available")
    def test_real_openai_tool_call(self):
        """Test with the real OpenAI API (requires API key)."""
        # Create the agent
        agent = BasicAgent(provider_name="openai")
        
        # Create a test file
        with open("test_file.txt", "w") as f:
            f.write("This is a test file.\nIt has some content.\nThree lines total.")
        
        try:
            # Run the query
            result = agent.run("What's in the file test_file.txt?")
            
            # Verify that the result contains the file content
            assert "test file" in result.lower()
            assert "three lines" in result.lower()
            
        finally:
            # Clean up the test file
            os.remove("test_file.txt")
```

### 3. Create Tests for Error Handling

Create tests for error handling in the tool call process:

```python
# File: tests/test_tool_error_handling.py

import pytest
from unittest.mock import patch, MagicMock

from abstractllm.tools import ToolCallRequest, ToolCall
from abstractllm.types import GenerateResponse
from basic_agent import BasicAgent

def test_tool_not_found_error():
    """Test error handling when a tool is not found."""
    # Prepare mock response with tool calls
    tool_call = ToolCall(
        id="call_123",
        name="nonexistent_tool",
        arguments={"param": "value"}
    )
    tool_calls_request = ToolCallRequest(
        content="I'll use the nonexistent_tool",
        tool_calls=[tool_call]
    )
    mock_response = GenerateResponse(
        content="I'll use the nonexistent_tool",
        tool_calls=tool_calls_request
    )
    
    # Configure the mock session
    with patch("abstractllm.session.Session") as mock_session_cls:
        # Create mock session instance
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        
        # Configure the session manager to return our mock
        with patch("abstractllm.session.SessionManager") as mock_manager:
            manager_instance = MagicMock()
            manager_instance.get_session.return_value = mock_session
            mock_manager.return_value = manager_instance
            
            # Mock the tool result with an error
            def mock_execute(tool_call, tool_functions):
                return {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": None,
                    "error": f"Tool '{tool_call.name}' not found"
                }
            mock_session.execute_tool_call.side_effect = mock_execute
            
            # Configure the initial and final responses
            mock_session.generate_with_tools.side_effect = [
                mock_response,  # Initial response with tool call
                GenerateResponse(content="I encountered an error: Tool 'nonexistent_tool' not found")  # Final response
            ]
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Run the query
            result = agent.run("Use a nonexistent tool")
            
            # Verify the error handling
            assert "error" in result.lower()
            assert "not found" in result

def test_tool_execution_error():
    """Test error handling when a tool execution fails."""
    # Prepare mock response with tool calls
    tool_call = ToolCall(
        id="call_123",
        name="read_file",
        arguments={"file_path": "nonexistent.txt"}
    )
    tool_calls_request = ToolCallRequest(
        content="I'll read that file for you",
        tool_calls=[tool_call]
    )
    mock_response = GenerateResponse(
        content="I'll read that file for you",
        tool_calls=tool_calls_request
    )
    
    # Configure the mock session
    with patch("abstractllm.session.Session") as mock_session_cls:
        # Create mock session instance
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        
        # Configure the session manager to return our mock
        with patch("abstractllm.session.SessionManager") as mock_manager:
            manager_instance = MagicMock()
            manager_instance.get_session.return_value = mock_session
            mock_manager.return_value = manager_instance
            
            # Mock the tool result with an error
            def mock_execute(tool_call, tool_functions):
                return {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": None,
                    "error": "Error reading file: File not found"
                }
            mock_session.execute_tool_call.side_effect = mock_execute
            
            # Configure the initial and final responses
            mock_session.generate_with_tools.side_effect = [
                mock_response,  # Initial response with tool call
                GenerateResponse(content="I couldn't read the file. Error: File not found")  # Final response
            ]
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Run the query
            result = agent.run("Please read the file nonexistent.txt")
            
            # Verify the error handling
            assert "couldn't read" in result.lower() or "error" in result.lower()
            assert "file not found" in result.lower()
```

### 4. Create Tests for Security Validation

Create tests for security validation in tool execution:

```python
# File: tests/test_tool_security.py

import pytest
import os
from unittest.mock import patch

from basic_agent import BasicAgent, is_safe_path

def test_safe_path_validation():
    """Test the is_safe_path function."""
    # Define allowed directories
    allowed_dirs = [
        os.path.abspath("data"),
        os.path.abspath("test_files")
    ]
    
    # Create test directories if they don't exist
    for dir_path in allowed_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        # Test valid paths
        assert is_safe_path(os.path.join("data", "file.txt"), allowed_dirs)
        assert is_safe_path(os.path.join("test_files", "test.txt"), allowed_dirs)
        
        # Test invalid paths
        assert not is_safe_path(os.path.join("..", "file.txt"), allowed_dirs)
        assert not is_safe_path("/etc/passwd", allowed_dirs)
        assert not is_safe_path("../../../etc/passwd", allowed_dirs)
        
        # Test path normalization
        assert not is_safe_path("data/../../../etc/passwd", allowed_dirs)
        assert is_safe_path("data/../data/file.txt", allowed_dirs)
    
    finally:
        # Clean up test directories
        for dir_path in allowed_dirs:
            if os.path.exists(dir_path):
                os.rmdir(dir_path)

def test_read_file_security():
    """Test that read_file enforces security."""
    # Get the current directory
    current_dir = os.getcwd()
    
    # Create a test file
    test_file_path = os.path.join(current_dir, "test_secure.txt")
    with open(test_file_path, "w") as f:
        f.write("This is a secure test file.")
    
    try:
        # Test reading a file in the allowed directory
        from basic_agent import read_file
        result = read_file(test_file_path)
        assert "secure test file" in result
        
        # Test reading a file outside allowed directories
        outside_path = os.path.join("..", "outside.txt")
        result = read_file(outside_path)
        assert "Error" in result
        assert "not allowed" in result
        
        # Test path traversal attempt
        traversal_path = os.path.join(current_dir, "..", "..", "etc", "passwd")
        result = read_file(traversal_path)
        assert "Error" in result
        assert "not allowed" in result
    
    finally:
        # Clean up the test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

def test_tool_execution_timeout():
    """Test that tool execution has timeout protection."""
    # Import the necessary functions
    from basic_agent import create_secure_tool_wrapper
    
    # Define a slow function
    def slow_function():
        import time
        time.sleep(10)  # Sleep for 10 seconds
        return "Done"
    
    # Wrap it with a 1-second timeout
    wrapped_function = create_secure_tool_wrapper(slow_function, max_execution_time=1)
    
    # Execute and check for timeout
    result = wrapped_function()
    assert "timeout" in result.lower()
```

### 5. Create Tests for Streaming Tool Calls

Create tests for streaming tool calls:

```python
# File: tests/test_streaming_tools.py

import pytest
from unittest.mock import patch, MagicMock, call

from abstractllm.tools import ToolCallRequest, ToolCall
from abstractllm.types import GenerateResponse
from basic_agent import BasicAgent

def test_streaming_tool_execution():
    """Test that tools are executed properly in streaming mode."""
    # Configure the mock session
    with patch("abstractllm.session.Session") as mock_session_cls:
        # Create mock session instance
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        
        # Configure the session manager to return our mock
        with patch("abstractllm.session.SessionManager") as mock_manager:
            manager_instance = MagicMock()
            manager_instance.get_session.return_value = mock_session
            mock_manager.return_value = manager_instance
            
            # Define the streaming response
            def mock_streaming():
                # Yield some content
                yield "I'll read the file for you"
                
                # Yield a tool call dict
                yield {
                    "call_id": "call_123",
                    "name": "read_file",
                    "arguments": {"file_path": "test.txt"},
                    "output": "This is the content of test.txt",
                    "error": None
                }
                
                # Yield more content
                yield " and here's what I found in the file: This is the content of test.txt"
            
            # Configure the mock to return our streaming generator
            mock_session.generate_with_tools_streaming.return_value = mock_streaming()
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Mock print to capture output
            with patch("builtins.print") as mock_print:
                # Run the query
                agent.run_streaming("Please read the file test.txt")
                
                # Verify that print was called with the expected content
                calls = [
                    call("I'll read the file for you", end="", flush=True),
                    call("\n[Executing tool: read_file]\n", flush=True),
                    call(" and here's what I found in the file: This is the content of test.txt", end="", flush=True),
                    call("\n")
                ]
                mock_print.assert_has_calls(calls, any_order=False)

def test_streaming_error_handling():
    """Test error handling in streaming mode."""
    # Configure the mock session
    with patch("abstractllm.session.Session") as mock_session_cls:
        # Create mock session instance
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        
        # Configure the session manager to return our mock
        with patch("abstractllm.session.SessionManager") as mock_manager:
            manager_instance = MagicMock()
            manager_instance.get_session.return_value = mock_session
            mock_manager.return_value = manager_instance
            
            # Make generate_with_tools_streaming raise an exception
            mock_session.generate_with_tools_streaming.side_effect = Exception("Test error")
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Mock print to capture output
            with patch("builtins.print") as mock_print:
                # Run the query
                agent.run_streaming("Please read the file test.txt")
                
                # Verify that print was called with the error message
                mock_print.assert_called_with("\nError: Error during streaming: Test error")
```

### 6. Create System-Level Tests

Create system-level tests for the entire agent system:

```python
# File: tests/system/test_tool_system.py

import pytest
import os
import subprocess
import re

@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                   reason="OpenAI API key not available")
def test_agent_system_file_reading():
    """Test the entire agent system for file reading."""
    # Create a test file
    with open("system_test.txt", "w") as f:
        f.write("This is a system test file.\nIt has some content.\nThree lines total.")
    
    try:
        # Run the agent as a subprocess
        cmd = ["python", "basic_agent.py", "--query", "Please read the file system_test.txt", "--debug"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check the output
        output = result.stdout
        
        # Verify the flow
        assert "STEP 1: USER→AGENT" in output
        assert "STEP 2: AGENT→LLM" in output
        assert "STEP 3: LLM→AGENT" in output
        
        # Look for either tool call steps or a direct response
        if "LLM requested" in output:
            assert "STEP 4: AGENT→TOOL" in output
            assert "STEP 5: TOOL→AGENT" in output
            assert "STEP 6: AGENT→LLM" in output
            assert "STEP 7: LLM→AGENT" in output
            assert "STEP 8: AGENT→USER" in output
        else:
            # If no tool call, should go directly to the final step
            assert "STEP 4: LLM→AGENT" in output
            assert "STEP 5: AGENT→USER" in output
        
        # Verify the content
        assert "system test file" in output.lower()
        
        # Verify the exit code
        assert result.returncode == 0
        
    finally:
        # Clean up the test file
        if os.path.exists("system_test.txt"):
            os.remove("system_test.txt")

@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                   reason="OpenAI API key not available")
def test_agent_system_nonexistent_file():
    """Test the system with a nonexistent file."""
    # Run the agent as a subprocess
    cmd = ["python", "basic_agent.py", "--query", "Please read the file nonexistent_file.txt", "--debug"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check the output
    output = result.stdout
    
    # Verify that the error is handled properly
    assert "STEP 1: USER→AGENT" in output
    assert "STEP 2: AGENT→LLM" in output
    
    # Expect an error message in the response
    assert "file not found" in output.lower() or "no such file" in output.lower() or "does not exist" in output.lower()
```

## Testing Strategy

The tests should be organized to cover:

1. **Unit tests**: Test individual components (tool validation, tool execution)
2. **Integration tests**: Test interactions between components (agent → LLM → tools)
3. **System tests**: Test the entire system end-to-end
4. **Security tests**: Test security measures in tool execution
5. **Error handling tests**: Test error scenarios

## Implementation Plan

1. Create a `tests` directory with appropriate subdirectories
2. Implement the unit tests first
3. Implement integration tests that use mocks
4. Implement system-level tests for end-to-end validation
5. Implement automated test runs as part of CI/CD

## Completion Criteria

- ✅ Unit tests verify that direct tool execution is not happening
- ✅ Integration tests verify the LLM-first flow
- ✅ Tests verify proper error handling
- ✅ Tests verify security measures
- ✅ Tests verify streaming tool execution
- ✅ System-level tests verify end-to-end functionality
- ✅ All tests pass with the fixed implementation 