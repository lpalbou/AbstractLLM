#!/usr/bin/env python3
"""
Tests for error handling in AbstractLLM tool execution.

These tests verify that error conditions during tool execution are properly 
handled and communicated back to the user through the LLM.
"""

import pytest
from unittest.mock import patch, MagicMock

from basic_agent import BasicAgent


def test_tool_not_found_error():
    """Test error handling when a tool is not found."""
    # Prepare mock response with tool calls
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.name = "nonexistent_tool"
    mock_tool_call.arguments = {"param": "value"}

    mock_tool_calls = MagicMock()
    mock_tool_calls.tool_calls = [mock_tool_call]
    
    mock_initial_response = MagicMock()
    mock_initial_response.content = "I'll use the nonexistent_tool"
    mock_initial_response.tool_calls = mock_tool_calls
    mock_initial_response.has_tool_calls.return_value = True
    
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
            
            # Mock execute_tool_calls to use our mock_execute for each tool call
            def mock_execute_calls(response, tool_functions):
                results = []
                for tool_call in response.tool_calls.tool_calls:
                    results.append(mock_execute(tool_call, tool_functions))
                return results
            mock_session.execute_tool_calls.side_effect = mock_execute_calls
            
            # Configure the initial and final responses
            mock_final_response = MagicMock()
            mock_final_response.content = "I encountered an error: Tool 'nonexistent_tool' not found"
            mock_final_response.has_tool_calls.return_value = False
            
            mock_session.generate_with_tools.side_effect = [
                mock_initial_response,  # Initial response with tool call
                mock_final_response     # Final response after error
            ]
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Run the query
            result = agent.run("Use a nonexistent tool")
            
            # Verify the error handling
            assert "error" in result.lower() or "not found" in result.lower()
            assert "nonexistent_tool" in result


def test_tool_execution_error():
    """Test error handling when a tool execution fails."""
    # Prepare mock response with tool calls
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.name = "read_file"
    mock_tool_call.arguments = {"file_path": "nonexistent.txt"}
    
    mock_tool_calls = MagicMock()
    mock_tool_calls.tool_calls = [mock_tool_call]
    
    mock_initial_response = MagicMock()
    mock_initial_response.content = "I'll read that file for you"
    mock_initial_response.tool_calls = mock_tool_calls
    mock_initial_response.has_tool_calls.return_value = True
    
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
            
            # Mock execute_tool_calls to use our mock_execute for each tool call
            def mock_execute_calls(response, tool_functions):
                results = []
                for tool_call in response.tool_calls.tool_calls:
                    results.append(mock_execute(tool_call, tool_functions))
                return results
            mock_session.execute_tool_calls.side_effect = mock_execute_calls
            
            # Configure the initial and final responses
            mock_final_response = MagicMock()
            mock_final_response.content = "I couldn't read the file. Error: File not found"
            mock_final_response.has_tool_calls.return_value = False
            
            mock_session.generate_with_tools.side_effect = [
                mock_initial_response,  # Initial response with tool call
                mock_final_response     # Final response after error
            ]
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Run the query
            result = agent.run("Please read the file nonexistent.txt")
            
            # Verify the error handling
            assert "couldn't read" in result.lower() or "error" in result.lower()
            assert "file not found" in result.lower()


def test_invalid_parameters_error():
    """Test error handling for invalid tool parameters."""
    # Prepare mock response with tool calls
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.name = "read_file"
    mock_tool_call.arguments = {"max_lines": -10}  # Missing required file_path and invalid max_lines
    
    mock_tool_calls = MagicMock()
    mock_tool_calls.tool_calls = [mock_tool_call]
    
    mock_initial_response = MagicMock()
    mock_initial_response.content = "I'll read that file for you"
    mock_initial_response.tool_calls = mock_tool_calls
    mock_initial_response.has_tool_calls.return_value = True
    
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
            
            # Mock the tool result with a parameter validation error
            def mock_execute(tool_call, tool_functions):
                return {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": None,
                    "error": "Invalid parameters: Missing required parameter 'file_path'"
                }
            mock_session.execute_tool_call.side_effect = mock_execute
            
            # Mock execute_tool_calls to use our mock_execute for each tool call
            def mock_execute_calls(response, tool_functions):
                results = []
                for tool_call in response.tool_calls.tool_calls:
                    results.append(mock_execute(tool_call, tool_functions))
                return results
            mock_session.execute_tool_calls.side_effect = mock_execute_calls
            
            # Configure the initial and final responses
            mock_final_response = MagicMock()
            mock_final_response.content = "I encountered a parameter error: Missing required parameter 'file_path'"
            mock_final_response.has_tool_calls.return_value = False
            
            mock_session.generate_with_tools.side_effect = [
                mock_initial_response,  # Initial response with tool call
                mock_final_response     # Final response after error
            ]
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Run the query
            result = agent.run("Please read a file")
            
            # Verify the error handling
            assert "parameter" in result.lower() or "error" in result.lower()
            assert "file_path" in result.lower()


def test_llm_provider_error():
    """Test error handling when the LLM provider has an error."""
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
            
            # Make generate_with_tools raise an exception
            mock_session.generate_with_tools.side_effect = Exception("API Error: Rate limit exceeded")
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Run the query
            result = agent.run("This will cause an error")
            
            # Verify the error handling
            assert "error" in result.lower()
            assert "rate limit" in result.lower() or "api error" in result.lower()


def test_sequential_tool_errors():
    """Test handling of multiple sequential tool errors."""
    # Prepare first mock response with a tool call that will error
    mock_tool_call_1 = MagicMock()
    mock_tool_call_1.id = "call_1"
    mock_tool_call_1.name = "read_file"
    mock_tool_call_1.arguments = {"file_path": "nonexistent1.txt"}
    
    mock_tool_calls_1 = MagicMock()
    mock_tool_calls_1.tool_calls = [mock_tool_call_1]
    
    mock_response_1 = MagicMock()
    mock_response_1.content = "I'll read the first file"
    mock_response_1.tool_calls = mock_tool_calls_1
    mock_response_1.has_tool_calls.return_value = True
    
    # Prepare second mock response with another tool call that will also error
    mock_tool_call_2 = MagicMock()
    mock_tool_call_2.id = "call_2"
    mock_tool_call_2.name = "read_file"
    mock_tool_call_2.arguments = {"file_path": "nonexistent2.txt"}
    
    mock_tool_calls_2 = MagicMock()
    mock_tool_calls_2.tool_calls = [mock_tool_call_2]
    
    mock_response_2 = MagicMock()
    mock_response_2.content = "Let me try another file"
    mock_response_2.tool_calls = mock_tool_calls_2
    mock_response_2.has_tool_calls.return_value = True
    
    # Prepare final mock response with no tool calls
    mock_response_3 = MagicMock()
    mock_response_3.content = "I couldn't read either file due to errors."
    mock_response_3.has_tool_calls.return_value = False
    
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
            
            # Mock the tool result with errors
            error_count = 0
            
            def mock_execute(tool_call, tool_functions):
                nonlocal error_count
                error_count += 1
                return {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": None,
                    "error": f"Error reading file {error_count}: File not found"
                }
            mock_session.execute_tool_call.side_effect = mock_execute
            
            # Mock execute_tool_calls to use our mock_execute for each tool call
            def mock_execute_calls(response, tool_functions):
                results = []
                for tool_call in response.tool_calls.tool_calls:
                    results.append(mock_execute(tool_call, tool_functions))
                return results
            mock_session.execute_tool_calls.side_effect = mock_execute_calls
            
            # Configure the sequence of responses
            mock_session.generate_with_tools.side_effect = [
                mock_response_1,  # First response with tool call
                mock_response_2,  # Second response with tool call after first error
                mock_response_3   # Final response after second error
            ]
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Run the query
            result = agent.run("Please read some files")
            
            # Verify that generate_with_tools was called 3 times
            assert mock_session.generate_with_tools.call_count == 3
            
            # Verify execute_tool_call was called twice (once for each error)
            assert mock_session.execute_tool_call.call_count == 2
            
            # Verify the final result contains the appropriate message
            assert "couldn't read" in result.lower() or "errors" in result.lower()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 