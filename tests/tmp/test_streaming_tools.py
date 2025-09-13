#!/usr/bin/env python3
"""
Tests for streaming tool execution in AbstractLLM.

These tests verify that tools are properly executed during streaming responses
and that the streaming flow correctly handles tool execution and errors.
"""

import pytest
from unittest.mock import patch, MagicMock, call

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
                    "type": "tool_result",
                    "tool_call": {
                        "call_id": "call_123",
                        "name": "read_file",
                        "arguments": {"file_path": "test.txt"},
                        "output": "This is the content of test.txt",
                        "error": None
                    }
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
                mock_print.assert_any_call("\nError: Error during streaming: Test error")


def test_streaming_multiple_tool_calls():
    """Test handling multiple tool calls in streaming mode."""
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
            
            # Define the streaming response with multiple tool calls
            def mock_streaming():
                # Yield initial content
                yield "I'll check multiple files for you"
                
                # Yield first tool call
                yield {
                    "type": "tool_result",
                    "tool_call": {
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": {"file_path": "file1.txt"},
                        "output": "Content of file1",
                        "error": None
                    }
                }
                
                # Yield content after first tool call
                yield " First file contains: Content of file1."
                
                # Yield second tool call
                yield {
                    "type": "tool_result",
                    "tool_call": {
                        "call_id": "call_2",
                        "name": "read_file",
                        "arguments": {"file_path": "file2.txt"},
                        "output": "Content of file2",
                        "error": None
                    }
                }
                
                # Yield final content
                yield " Second file contains: Content of file2."
            
            # Configure the mock to return our streaming generator
            mock_session.generate_with_tools_streaming.return_value = mock_streaming()
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Mock print to capture output
            with patch("builtins.print") as mock_print:
                # Run the query
                agent.run_streaming("Please read files file1.txt and file2.txt")
                
                # Verify that print was called with the expected content
                calls = [
                    call("I'll check multiple files for you", end="", flush=True),
                    call("\n[Executing tool: read_file]\n", flush=True),
                    call(" First file contains: Content of file1.", end="", flush=True),
                    call("\n[Executing tool: read_file]\n", flush=True),
                    call(" Second file contains: Content of file2.", end="", flush=True),
                    call("\n")
                ]
                mock_print.assert_has_calls(calls, any_order=False)


def test_streaming_tool_error():
    """Test handling tool execution errors in streaming mode."""
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
            
            # Define the streaming response with a tool error
            def mock_streaming():
                # Yield initial content
                yield "I'll try to read the file for you"
                
                # Yield a tool call with an error
                yield {
                    "type": "tool_result",
                    "tool_call": {
                        "call_id": "call_123",
                        "name": "read_file",
                        "arguments": {"file_path": "nonexistent.txt"},
                        "output": None,
                        "error": "Error reading file: File not found"
                    }
                }
                
                # Yield content handling the error
                yield " I encountered an error: The file 'nonexistent.txt' could not be found."
            
            # Configure the mock to return our streaming generator
            mock_session.generate_with_tools_streaming.return_value = mock_streaming()
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Mock print to capture output
            with patch("builtins.print") as mock_print:
                # Run the query
                agent.run_streaming("Please read the file nonexistent.txt")
                
                # Verify that print was called with the expected content
                calls = [
                    call("I'll try to read the file for you", end="", flush=True),
                    call("\n[Executing tool: read_file]\n", flush=True),
                    call(" I encountered an error: The file 'nonexistent.txt' could not be found.", end="", flush=True),
                    call("\n")
                ]
                mock_print.assert_has_calls(calls, any_order=False)


def test_streaming_follow_up_response():
    """Test that follow-up responses are handled correctly in streaming mode."""
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
            
            # Track stream state
            stream_complete = False
            
            # Define the streaming response
            def mock_streaming():
                # Initial streaming generator
                # Yield initial content
                yield "I'll read the file for you"
                
                # Yield a tool call
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
                
                # Mark stream as complete
                nonlocal stream_complete
                stream_complete = True
                
                # The follow-up response would be a separate stream
                # but we can model it here since we control the generator
                yield "Based on the file content, I can tell you that test.txt contains information."
            
            # Configure the mock to return our streaming generator
            mock_session.generate_with_tools_streaming.return_value = mock_streaming()
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Mock print to capture output
            with patch("builtins.print") as mock_print:
                # Run the query
                agent.run_streaming("Please read the file test.txt and tell me about it")
                
                # Verify that the tool was executed
                assert stream_complete
                
                # Verify print calls
                mock_print.assert_any_call("I'll read the file for you", end="", flush=True)
                mock_print.assert_any_call("\n[Executing tool: read_file]\n", flush=True)
                mock_print.assert_any_call("Based on the file content, I can tell you that test.txt contains information.", end="", flush=True)


def test_session_message_addition_during_streaming():
    """Test that messages are correctly added to the session during streaming."""
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
                yield "Initial content"
                yield {
                    "type": "tool_result",
                    "tool_call": {
                        "call_id": "call_123",
                        "name": "read_file",
                        "arguments": {"file_path": "test.txt"},
                        "output": "File content",
                        "error": None
                    }
                }
                yield " and final content"
            
            # Configure the mock to return our streaming generator
            mock_session.generate_with_tools_streaming.return_value = mock_streaming()
            
            # Create the agent
            agent = BasicAgent(provider_name="mock")
            
            # Mock print to avoid console output
            with patch("builtins.print"):
                # Run the query
                agent.run_streaming("Test query")
                
                # Verify the session message additions
                # 1. User message
                mock_session.add_message.assert_any_call("user", "Test query")
                
                # 2. Assistant message (should include the complete response)
                complete_content = "Initial content and final content"
                # The exact call might vary depending on how agent.run_streaming is implemented
                calls = mock_session.add_message.call_args_list
                
                # Find the assistant message addition with the complete content
                assistant_message_found = False
                for call_args, call_kwargs in calls:
                    if call_args and call_args[0] == "assistant" and complete_content in call_args[1]:
                        assistant_message_found = True
                        break
                
                assert assistant_message_found, "Complete assistant message not added to session"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 