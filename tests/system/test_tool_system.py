#!/usr/bin/env python3
"""
System-level tests for tool execution in AbstractLLM.

These tests verify the entire agent system for proper tool call handling
using actual execution of the agent with real or simulated LLM providers.
"""

import pytest
import os
import subprocess
import re
import tempfile
from unittest.mock import patch

from basic_agent import BasicAgent


def create_test_file(content, filename="system_test.txt"):
    """Create a test file with the given content."""
    with open(filename, "w") as f:
        f.write(content)
    return os.path.abspath(filename)


def delete_test_file(filepath):
    """Delete a test file if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, 
                   reason="OpenAI API key not available")
def test_agent_system_file_reading():
    """Test the entire agent system for file reading."""
    # Create a test file
    test_file_content = "This is a system test file.\nIt has some content.\nThree lines total."
    test_file = create_test_file(test_file_content)
    
    try:
        # Run the agent as a subprocess
        cmd = ["python", "basic_agent.py", "--query", f"Please read the file {test_file}", "--debug"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check the output
        output = result.stdout
        
        # Verify the flow
        assert "USER→AGENT" in output, "User to agent flow not found in output"
        assert "AGENT→LLM" in output, "Agent to LLM flow not found in output"
        assert "LLM→AGENT" in output, "LLM to agent flow not found in output"
        
        # Look for either tool call steps or a direct response
        if "LLM requested" in output:
            assert "AGENT→TOOL" in output, "Agent to tool flow not found in output when tool was requested"
            assert "TOOL→AGENT" in output, "Tool to agent flow not found in output when tool was requested"
            assert "AGENT→LLM" in output, "Agent to LLM flow not found in output after tool execution"
        
        # Verify the content is in the response
        assert "system test file" in output.lower(), "Test file content not found in output"
        assert "three lines" in output.lower(), "Test file content not found in output"
        
        # Verify the exit code
        assert result.returncode == 0, f"Command exited with non-zero status: {result.returncode}"
        
    finally:
        # Clean up the test file
        delete_test_file(test_file)


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
    assert "USER→AGENT" in output, "User to agent flow not found in output"
    assert "AGENT→LLM" in output, "Agent to LLM flow not found in output"
    
    # Expect an error message in the response
    error_indicators = ["file not found", "no such file", "does not exist", "could not find", "unable to locate"]
    assert any(indicator in output.lower() for indicator in error_indicators), \
        "Error message about nonexistent file not found in output"


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None,
                   reason="OpenAI API key not available")
def test_agent_system_direct_class_usage():
    """Test the agent system using direct class instantiation."""
    # Create a test file
    test_file_content = "This is a direct test file.\nUsed for direct class testing.\nThree lines in total."
    test_file = create_test_file(test_file_content, "direct_test.txt")
    
    try:
        # Create and use agent directly
        agent = BasicAgent(provider_name="openai", debug=True)
        
        # Intercept logging output
        log_output = []
        
        def mock_log(step_number, step_name, message, level=None):
            log_output.append(f"STEP {step_number}: {step_name} - {message}")
        
        # Patch the log_step function
        with patch("basic_agent.log_step", side_effect=mock_log):
            # Run the query
            result = agent.run(f"What's in the file direct_test.txt?")
            
            # Verify the flow
            assert any("USER→AGENT" in log for log in log_output), "User to agent flow not found in logs"
            assert any("AGENT→LLM" in log for log in log_output), "Agent to LLM flow not found in logs"
            assert any("LLM→AGENT" in log for log in log_output), "LLM to agent flow not found in logs"
            
            # Verify the content is in the response
            assert "direct test file" in result.lower() or "direct test" in result.lower(), \
                "Test file content not found in result"
            assert "three lines" in result.lower(), "Test file content not found in result"
    
    finally:
        # Clean up the test file
        delete_test_file("direct_test.txt")


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None,
                   reason="OpenAI API key not available")
def test_agent_system_streaming():
    """Test the agent system in streaming mode."""
    # Create a test file
    test_file_content = "This is a streaming test file.\nUsed for streaming tests.\nThree lines in total."
    test_file = create_test_file(test_file_content, "streaming_test.txt")
    
    try:
        # Create agent
        agent = BasicAgent(provider_name="openai", debug=True)
        
        # Capture print output
        print_output = []
        
        def mock_print(*args, **kwargs):
            # Join the args with a space
            output = " ".join(str(arg) for arg in args)
            print_output.append(output)
        
        # Patch the print function
        with patch("builtins.print", side_effect=mock_print):
            # Run the streaming query
            agent.run_streaming(f"What's in the file streaming_test.txt?")
            
            # Verify the content is in the response
            all_output = " ".join(print_output)
            assert "streaming test" in all_output.lower(), "Test file content not found in streaming output"
            assert "three lines" in all_output.lower() or "3 lines" in all_output.lower(), \
                "Test file content not found in streaming output"
    
    finally:
        # Clean up the test file
        delete_test_file("streaming_test.txt")


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None,
                   reason="OpenAI API key not available")
def test_agent_system_security_boundary():
    """Test that the agent system enforces security boundaries for file access."""
    # Create a test file in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file outside the allowed directories
        outside_file = os.path.join(temp_dir, "outside.txt")
        with open(outside_file, "w") as f:
            f.write("This file is outside allowed directories.")
        
        # Create agent
        agent = BasicAgent(provider_name="openai", debug=True)
        
        # Run the query
        result = agent.run(f"Please read the file {outside_file}")
        
        # Verify that the security boundary was enforced
        assert "not allowed" in result.lower() or "security" in result.lower() or \
               "permission" in result.lower() or "access denied" in result.lower(), \
               "Security boundary message not found in result"
        
        # Verify that the file content is not in the result
        assert "outside allowed directories" not in result.lower(), \
               "File content from outside allowed directories found in result"


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None or os.environ.get("ANTHROPIC_API_KEY") is None,
    reason="Both OpenAI and Anthropic API keys required"
)
def test_agent_system_multiple_providers():
    """Test the agent system with multiple providers."""
    # Create a test file
    test_file_content = "This is a multi-provider test file.\nUsed for testing multiple providers.\nThree lines in total."
    test_file = create_test_file(test_file_content, "multi_provider_test.txt")
    
    try:
        # Test with OpenAI
        openai_agent = BasicAgent(provider_name="openai", debug=True)
        openai_result = openai_agent.run(f"What's in the file multi_provider_test.txt?")
        
        # Test with Anthropic
        anthropic_agent = BasicAgent(provider_name="anthropic", debug=True)
        anthropic_result = anthropic_agent.run(f"What's in the file multi_provider_test.txt?")
        
        # Verify both providers could read the file
        assert "multi-provider" in openai_result.lower() or "multi provider" in openai_result.lower(), \
               "OpenAI couldn't read the test file content"
        assert "multi-provider" in anthropic_result.lower() or "multi provider" in anthropic_result.lower(), \
               "Anthropic couldn't read the test file content"
    
    finally:
        # Clean up the test file
        delete_test_file("multi_provider_test.txt")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 