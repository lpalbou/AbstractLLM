"""
Test suite for the ALMA agent implementation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from alma import ALMA, read_file

@pytest.fixture
def test_file():
    """Create a temporary test file."""
    content = "This is a test file.\nIt has multiple lines.\nLine 3 here."
    filename = "test_alma_file.txt"
    with open(filename, "w") as f:
        f.write(content)
    yield filename
    # Clean up
    if os.path.exists(filename):
        os.remove(filename)


def test_read_file(test_file):
    """Test the read_file tool."""
    # Test normal reading
    content = read_file(test_file)
    assert "This is a test file." in content
    assert "Line 3 here." in content
    
    # Test with max_lines
    content = read_file(test_file, max_lines=2)
    assert "This is a test file." in content
    assert "Line 3 here." not in content
    
    # Test with non-existent file
    content = read_file("non_existent_file.txt")
    assert "Error reading file" in content
    
    # Test with file outside allowed directories
    content = read_file("/etc/passwd")
    assert "not allowed for security reasons" in content


def test_alma_initialization():
    """Test ALMA initialization."""
    # Test with default parameters
    with patch("alma.create_llm") as mock_create_llm:
        mock_provider = MagicMock()
        mock_create_llm.return_value = mock_provider
        
        agent = ALMA()
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-3-haiku-20240307"
        assert "alma-" in agent.session_id
        assert mock_create_llm.called
        
        # Check tool setup
        assert "read_file" in agent.tool_functions


@patch("alma.create_llm")
@patch("alma.SessionManager")
def test_alma_run_without_tools(mock_session_manager, mock_create_llm):
    """Test ALMA run method without tool calls."""
    # Set up mocks
    mock_provider = MagicMock()
    mock_create_llm.return_value = mock_provider
    
    mock_session = MagicMock()
    mock_session_manager_instance = MagicMock()
    mock_session_manager_instance.get_session.return_value = mock_session
    mock_session_manager.return_value = mock_session_manager_instance
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.content = "This is a test response without tool calls."
    mock_response.tool_calls = None
    mock_session.generate_with_tools.return_value = mock_response
    
    # Create agent and run query
    agent = ALMA()
    response = agent.run("What is the meaning of life?")
    
    # Verify
    assert response == "This is a test response without tool calls."
    mock_session.add_message.assert_called_with("user", "What is the meaning of life?")
    mock_session.generate_with_tools.assert_called_once()


@patch("alma.create_llm")
@patch("alma.SessionManager")
def test_alma_run_with_tools(mock_session_manager, mock_create_llm):
    """Test ALMA run method with tool calls."""
    # Set up mocks
    mock_provider = MagicMock()
    mock_create_llm.return_value = mock_provider
    
    mock_session = MagicMock()
    mock_session_manager_instance = MagicMock()
    mock_session_manager_instance.get_session.return_value = mock_session
    mock_session_manager.return_value = mock_session_manager_instance
    
    # Mock tool call
    mock_tool_call = MagicMock()
    mock_tool_call.name = "read_file"
    mock_tool_call.arguments = {"file_path": "test.txt"}
    
    # Mock tool calls collection
    mock_tool_calls = MagicMock()
    mock_tool_calls.tool_calls = [mock_tool_call]
    mock_tool_calls.has_tool_calls.return_value = True
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.content = "I read the file and found that it contains test content."
    mock_response.tool_calls = mock_tool_calls
    mock_session.generate_with_tools.return_value = mock_response
    
    # Create agent and run query
    agent = ALMA()
    response = agent.run("What's in the test.txt file?")
    
    # Verify
    assert response == "I read the file and found that it contains test content."
    mock_session.add_message.assert_called_with("user", "What's in the test.txt file?")
    mock_session.generate_with_tools.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", "test_alma.py"]) 