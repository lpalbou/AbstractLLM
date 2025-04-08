"""
Test utility functions.
"""

import os
from typing import List, Dict, Any, Union, Optional, Callable
import pytest
import re
from abstractllm.providers.huggingface import DEFAULT_MODEL


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
    Skip the test if the API key environment variable is not set.
    
    Args:
        env_var_name: Name of the environment variable
    
    Raises:
        pytest.skip: If API key is not set
    """
    if not check_api_key(env_var_name):
        pytest.skip(f"{env_var_name} not set")


def preload_hf_model(provider=None, model_name: str = DEFAULT_MODEL) -> None:
    """
    Preload a HuggingFace model to avoid first inference delay.
    
    Args:
        provider: Optional provider instance to use
        model_name: Name of the model to preload
        
    Returns:
        None
    """
    try:
        if provider is None:
            import tempfile
            import os
            from abstractllm import create_llm, ModelParameter
            
            # Use a test-specific cache directory
            cache_dir = tempfile.mkdtemp(prefix="abstractllm_preload_")
            # Make sure directory exists
            os.makedirs(cache_dir, exist_ok=True)
            
            # Use distilgpt2 for testing - very small, reliable model
            test_model = "distilgpt2"
            
            provider = create_llm("huggingface", **{
                ModelParameter.MODEL: test_model,
                ModelParameter.DEVICE: "cpu",  # Force CPU for tests
                ModelParameter.CACHE_DIR: cache_dir,  # Use test-specific cache directory
                "load_timeout": 120,  # Longer timeout for downloads
                "trust_remote_code": True,  # Allow trusted code execution if needed
            })
        
        # Call model loading method
        if hasattr(provider, "load_model"):
            provider.load_model()
        elif hasattr(provider, "preload"):
            provider.preload()
            
        # Run a quick warmup if available
        if hasattr(provider, "warmup"):
            provider.warmup()
    except Exception as e:
        pytest.skip(f"Could not preload Hugging Face model: {e}")


def setup_hf_testing() -> None:
    """
    Set up environment for HuggingFace tests.
    
    This function is meant to be called at module level to prepare
    for HuggingFace tests. It:
    
    - Preloads the default HuggingFace model for faster tests
    - Runs basic checks to detect environment issues
    
    Returns:
        None
    """
    try:
        import torch
        import transformers
    except ImportError:
        pytest.skip("PyTorch or Transformers not installed", allow_module_level=True)
    
    # Set environment variables to help with testing
    os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Allow downloading if needed
    
    # Preload a small model
    try:
        preload_hf_model()
    except Exception as e:
        pytest.skip(f"Could not preload HuggingFace model: {e}", allow_module_level=True)


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