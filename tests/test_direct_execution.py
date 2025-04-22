#!/usr/bin/env python3
"""
Tests to verify that AbstractLLM is not using direct tool execution patterns.

These tests check for proper LLM-first flow and ensure that direct tool execution
based on pattern matching in user queries is not occurring.
"""

import pytest
import re
import os
from unittest.mock import patch, MagicMock

from abstractllm.session import Session
from basic_agent import BasicAgent, read_file


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
        mock_response.has_tool_calls = MagicMock(return_value=False)
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
    with open(os.path.join(os.path.dirname(__file__), "..", "basic_agent.py"), "r") as f:
        source_code = f.read()
    
    # Look for patterns that suggest conditional handling based on query content
    file_conditionals = re.findall(r"if.*['\"]file['\"].*in.*query", source_code)
    read_conditionals = re.findall(r"if.*['\"]read['\"].*in.*query", source_code)
    
    # There should be no such conditionals in the code
    assert len(file_conditionals) == 0, f"Found file conditionals: {file_conditionals}"
    assert len(read_conditionals) == 0, f"Found read conditionals: {read_conditionals}"


def test_no_should_use_tool_methods():
    """Test that there are no methods for determining tool use from query content."""
    
    # Read the source code of basic_agent.py
    with open(os.path.join(os.path.dirname(__file__), "..", "basic_agent.py"), "r") as f:
        source_code = f.read()
    
    # Look for patterns that suggest determining tool use from query
    should_use_patterns = [
        r"def\s+_?should_use_tool",
        r"def\s+_?detect_tool_use",
        r"def\s+_?extract_tool_from_query",
        r"def\s+_?query_needs_tool"
    ]
    
    for pattern in should_use_patterns:
        matches = re.findall(pattern, source_code)
        assert len(matches) == 0, f"Found suspicious method: {matches}"


def test_no_extracting_filenames_from_query():
    """Test that there's no pattern of extracting filenames directly from queries."""
    
    # Read the source code of basic_agent.py
    with open(os.path.join(os.path.dirname(__file__), "..", "basic_agent.py"), "r") as f:
        source_code = f.read()
    
    # Look for patterns that suggest extracting filenames
    extract_patterns = [
        r"extract_filename.*from.*query",
        r"filename\s*=.*query",
        r"parse_filename.*query"
    ]
    
    for pattern in extract_patterns:
        matches = re.search(pattern, source_code, re.IGNORECASE)
        assert matches is None, f"Found pattern for extracting filenames: {matches.group(0) if matches else None}"


def test_all_queries_flow_through_llm():
    """Test that all query handling paths flow through the LLM."""
    
    # Set up mock session
    mock_session = MagicMock()
    mock_manager = MagicMock()
    mock_manager.get_session.return_value = mock_session
    
    # Mock session manager
    with patch("abstractllm.session.SessionManager", return_value=mock_manager):
        # Mock generate_with_tools to track calls
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.has_tool_calls = MagicMock(return_value=False)
        mock_session.generate_with_tools.return_value = mock_response
        
        # Create agent
        agent = BasicAgent(provider_name="openai")
        
        # Test different types of queries
        test_queries = [
            "Hello, how are you?",                             # Regular query
            "Please read file.txt",                            # File query
            "What's in the test.txt file?",                    # Implicit file query
            "I need information from config.json",             # Implied file request
            "Read the contents of /etc/passwd",                # Security test
            "Read test.txt and tell me about it",              # Complex request
        ]
        
        # All queries should go through the same LLM path
        for query in test_queries:
            mock_session.generate_with_tools.reset_mock()
            agent.run(query)
            
            # Every query should result in exactly one call to generate_with_tools
            mock_session.generate_with_tools.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 