#!/usr/bin/env python3
"""
Integration tests for verifying the LLM-first tool call flow in AbstractLLM.

These tests ensure that tool calls are always initiated by the LLM, not by
pattern matching in the agent code, and that tool results are properly
passed back to the LLM for processing.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from abstractllm.session import Session
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
        # Create mock tool call objects
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.name = "read_file"
        mock_tool_call.arguments = {"file_path": "test.txt"}

        # Create mock tool calls container
        mock_tool_calls = MagicMock()
        mock_tool_calls.tool_calls = [mock_tool_call]
        
        # Prepare mock response with tool calls
        mock_initial_response = MagicMock()
        mock_initial_response.content = "I'll read that file for you"
        mock_initial_response.tool_calls = mock_tool_calls
        mock_initial_response.has_tool_calls.return_value = True
        
        # Mock the tool result
        mock_result = "This is the content of test.txt"
        
        # Mock the execute_tool_call method to return our result
        def mock_execute(tool_call, tool_functions):
            return {
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "output": mock_result,
                "error": None
            }
        mock_session.execute_tool_call.side_effect = mock_execute
        
        # Create mock final response after tool execution
        mock_final_response = MagicMock()
        mock_final_response.content = f"The content of the file is: {mock_result}"
        mock_final_response.has_tool_calls.return_value = False
        
        # Configure the generate_with_tools method to return our mock responses
        mock_session.generate_with_tools.side_effect = [mock_initial_response, mock_final_response]
        
        # Create the agent
        agent = BasicAgent(provider_name="mock")
        
        # Run the query
        result = agent.run("Please read the file test.txt")
        
        # Verify the result comes from the final LLM response
        assert result == "The content of the file is: This is the content of test.txt"
        
        # Verify that generate_with_tools was called twice
        assert mock_session.generate_with_tools.call_count == 2
        
        # Verify that execute_tool_call was called with the right arguments
        mock_session.execute_tool_call.assert_called_once()
        args, kwargs = mock_session.execute_tool_call.call_args
        assert args[0] == mock_tool_call
        
        # Verify that the second call to generate_with_tools happens after tool execution
        first_call_args, first_call_kwargs = mock_session.generate_with_tools.call_args_list[0]
        second_call_args, second_call_kwargs = mock_session.generate_with_tools.call_args_list[1]
        assert second_call_kwargs.get("prompt", "") == ""  # Empty prompt means using session history with tool results
    
    def test_single_call_when_no_tools_needed(self, mock_session):
        """Test that when no tools are needed, only one LLM call is made."""
        # Create mock response with no tool calls
        mock_response = MagicMock()
        mock_response.content = "I don't need any tools for this query"
        mock_response.has_tool_calls.return_value = False
        
        # Configure the mock session
        mock_session.generate_with_tools.return_value = mock_response
        
        # Create the agent
        agent = BasicAgent(provider_name="mock")
        
        # Run a query that shouldn't need tools
        result = agent.run("What is the weather like today?")
        
        # Verify that generate_with_tools was called only once
        mock_session.generate_with_tools.assert_called_once()
        
        # Verify that execute_tool_call was never called
        mock_session.execute_tool_call.assert_not_called()
        
        # Verify that the result comes from the mock response
        assert result == "I don't need any tools for this query"
    
    def test_multiple_tool_calls(self, mock_session):
        """Test handling multiple tool calls in a single response."""
        # Create mock tool call objects
        tool_call_1 = MagicMock()
        tool_call_1.id = "call_1"
        tool_call_1.name = "read_file"
        tool_call_1.arguments = {"file_path": "file1.txt"}
        
        tool_call_2 = MagicMock()
        tool_call_2.id = "call_2"
        tool_call_2.name = "read_file"
        tool_call_2.arguments = {"file_path": "file2.txt"}
        
        # Create mock tool calls container
        mock_tool_calls = MagicMock()
        mock_tool_calls.tool_calls = [tool_call_1, tool_call_2]
        
        # Prepare mock response with tool calls
        mock_initial_response = MagicMock()
        mock_initial_response.content = "I'll read those files for you"
        mock_initial_response.tool_calls = mock_tool_calls
        mock_initial_response.has_tool_calls.return_value = True
        
        # Mock the execute_tool_calls method 
        mock_session.execute_tool_calls.return_value = [
            {
                "call_id": "call_1",
                "name": "read_file",
                "arguments": {"file_path": "file1.txt"},
                "output": "Content of file1",
                "error": None
            },
            {
                "call_id": "call_2",
                "name": "read_file",
                "arguments": {"file_path": "file2.txt"},
                "output": "Content of file2",
                "error": None
            }
        ]
        
        # Create mock final response after tool execution
        mock_final_response = MagicMock()
        mock_final_response.content = "File1: Content of file1\nFile2: Content of file2"
        mock_final_response.has_tool_calls.return_value = False
        
        # Configure the generate_with_tools method to return our mock responses
        mock_session.generate_with_tools.side_effect = [mock_initial_response, mock_final_response]
        
        # Create the agent
        agent = BasicAgent(provider_name="mock")
        
        # Run the query
        result = agent.run("Please read files file1.txt and file2.txt")
        
        # Verify the result
        assert result == "File1: Content of file1\nFile2: Content of file2"
        
        # Verify that execute_tool_calls was called
        mock_session.execute_tool_calls.assert_called_once()
        
        # Verify that generate_with_tools was called twice
        assert mock_session.generate_with_tools.call_count == 2
    
    def test_streaming_tool_call(self, mock_session):
        """Test tool calls in streaming mode."""
        # Mock the streaming generator
        def mock_streaming_generator():
            # First yield some content
            yield "I'll read the file "
            
            # Then yield a tool call
            yield {
                "type": "tool_result",
                "tool_call": {
                    "call_id": "call_123",
                    "name": "read_file",
                    "arguments": {"file_path": "test.txt"},
                    "output": "This is the content of test.txt",
                    "error": None
                }
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
            mock_print.assert_any_call("\n")
    
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
            if os.path.exists("test_file.txt"):
                os.remove("test_file.txt")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 