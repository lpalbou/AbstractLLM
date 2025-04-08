"""
Test utility functions.
"""

import os
from typing import List, Dict, Any, Union, Optional, Callable
import pytest
import re


def check_api_key(env_var_name: str) -> bool:
    """
    Check if the API key environment variable is set.
    
    Args:
        env_var_name: Name of the environment variable
        
    Returns:
        True if the environment variable is set
    """
    return bool(os.environ.get(env_var_name))


def skip_if_no_api_key(env_var_name: str) -> None:
    """
    Skip a test if the API key environment variable is not set.
    
    Args:
        env_var_name: Name of the environment variable
    """
    if not check_api_key(env_var_name):
        raise pytest.skip(f"Skipping test because {env_var_name} is not set")


def validate_response(response: str, expected_contains: List[str], case_sensitive: bool = False) -> bool:
    """
    Validate that the response contains at least one of the expected strings.
    
    Args:
        response: The response to validate
        expected_contains: List of strings that the response should contain
        case_sensitive: Whether the check should be case sensitive
        
    Returns:
        True if the response contains at least one of the expected strings
    """
    if not case_sensitive:
        response = response.lower()
        expected_contains = [e.lower() for e in expected_contains]
    
    return any(item in response for item in expected_contains)


def validate_not_contains(response: str, not_expected_contains: List[str], case_sensitive: bool = False) -> bool:
    """
    Validate that the response does not contain any of the unexpected strings.
    
    Args:
        response: The response to validate
        not_expected_contains: List of strings that the response should not contain
        case_sensitive: Whether the check should be case sensitive
        
    Returns:
        True if the response does not contain any of the unexpected strings
    """
    if not case_sensitive:
        response = response.lower()
        not_expected_contains = [e.lower() for e in not_expected_contains]
    
    return all(item not in response for item in not_expected_contains)


def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    This is a very rough approximation (~ 4 chars per token).
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Estimated number of tokens
    """
    # This is a very rough approximation
    return len(text) // 4


def collect_stream(generator: Any) -> str:
    """
    Collect all chunks from a streaming response generator.
    
    Args:
        generator: The generator yielding chunks
        
    Returns:
        Concatenated response
    """
    chunks = []
    for chunk in generator:
        chunks.append(chunk)
    return "".join(chunks)


async def collect_stream_async(generator: Any) -> str:
    """
    Collect all chunks from an async streaming response generator.
    
    Args:
        generator: The async generator yielding chunks
        
    Returns:
        Concatenated response
    """
    chunks = []
    async for chunk in generator:
        chunks.append(chunk)
    return "".join(chunks)


def check_order_in_response(response: str, expected_sequence: List[str]) -> bool:
    """
    Check if elements appear in the expected order in the response.
    
    Args:
        response: The response to check
        expected_sequence: Sequence of strings that should appear in order
        
    Returns:
        True if all elements appear in the expected order
    """
    last_pos = -1
    for item in expected_sequence:
        pos = response.find(item, last_pos + 1)
        if pos <= last_pos:
            return False
        last_pos = pos
    return True


def has_capability(capabilities: Dict[str, Any], capability_name: str) -> bool:
    """
    Check if a provider has a specific capability.
    
    Args:
        capabilities: Dictionary of capabilities
        capability_name: Name of the capability to check
        
    Returns:
        True if the provider has the capability
    """
    return bool(capabilities.get(capability_name)) 