"""
Tests for the Anthropic provider.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from abstractllm import create_llm, ModelParameter

@pytest.fixture
def anthropic_llm():
    """Create Anthropic LLM instance for testing."""
    # If API key is available, use real provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return create_llm("anthropic", **{
            ModelParameter.API_KEY: api_key,
            ModelParameter.MODEL: "claude-3-haiku-20240307"  # Use a currently supported model
        })
    
    # Otherwise, create a mock provider
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Mock response from Anthropic"
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = iter(["Mock ", "streaming ", "response"])
    mock_llm.generate.side_effect = lambda prompt, system_prompt=None, stream=False, **kwargs: \
        mock_stream if stream else "Mock response from Anthropic"
    
    # Make MagicMock look like a real provider
    mock_capabilities = {
        "streaming": True,
        "max_tokens": 100000,
        "supports_system_prompt": True
    }
    mock_llm.get_capabilities.return_value = mock_capabilities
    
    return mock_llm

def test_generate(anthropic_llm):
    """Test basic text generation."""
    response = anthropic_llm.generate("Say hello")
    assert isinstance(response, str)
    assert len(response) > 0

def test_system_prompt(anthropic_llm):
    """Test generation with system prompt."""
    # This test will use mock if API key isn't available
    if isinstance(anthropic_llm, MagicMock):
        pytest.skip("Using mock - skipping detailed assertion")
        
    response = anthropic_llm.generate(
        "Tell me about yourself", 
        system_prompt="You are a professional chef. Always talk about cooking and food."
    )
    assert isinstance(response, str)
    assert len(response) > 0
    # Check if response contains cooking-related terms
    cooking_terms = ["chef", "cook", "food", "recipe"]
    assert any(term in response.lower() for term in cooking_terms)

def test_streaming(anthropic_llm):
    """Test streaming response generation."""
    stream = anthropic_llm.generate("Count from 1 to 5", stream=True)
    
    # Collect chunks from stream
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    
    # Check that we got at least one chunk
    assert len(chunks) > 0
    
    # Check that the combined response makes sense
    full_response = "".join(chunks)
    assert len(full_response) > 0
    
    # Skip number check if using mock
    if not isinstance(anthropic_llm, MagicMock):
        # Check if the response contains numbers 1-5
        for num in range(1, 6):
            assert str(num) in full_response 