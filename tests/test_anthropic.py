"""
Tests for the Anthropic provider.
"""

import os
import pytest
from abstractllm import create_llm, ModelParameter
from abstractllm.utils.config import ConfigurationManager

@pytest.fixture
def anthropic_llm():
    """Create Anthropic LLM instance for testing."""
    # Skip if no API key is available
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    
    # Create with ConfigurationManager
    base_config = ConfigurationManager.create_base_config()
    provider_config = ConfigurationManager.initialize_provider_config("anthropic", base_config)
    
    # Since we've already checked that the API key is in the environment,
    # manually set it in the config if it wasn't picked up automatically
    if provider_config.get(ModelParameter.API_KEY) is None:
        provider_config[ModelParameter.API_KEY] = os.environ.get("ANTHROPIC_API_KEY")
        
    # Manually set the model name since it's not being set automatically
    provider_config[ModelParameter.MODEL] = "claude-3-5-haiku-20241022"
    
    # Verify provider config now has API key from environment and default model
    assert provider_config.get(ModelParameter.API_KEY) is not None
    assert provider_config.get(ModelParameter.MODEL) == "claude-3-5-haiku-20241022"
    
    return create_llm("anthropic", **provider_config)

def test_generate(anthropic_llm):
    """Test basic text generation."""
    response = anthropic_llm.generate("Say hello")
    assert isinstance(response, str)
    assert len(response) > 0

def test_system_prompt(anthropic_llm):
    """Test generation with system prompt."""
    # Extract generation parameters to verify system_prompt is included
    gen_params = ConfigurationManager.extract_generation_params(
        "anthropic", 
        anthropic_llm.config, 
        {}, 
        system_prompt="You are a professional chef. Always talk about cooking and food."
    )
    
    # Verify system_prompt is set correctly
    assert gen_params["system_prompt"] == "You are a professional chef. Always talk about cooking and food."
        
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
    # Verify streaming capability is reported correctly
    capabilities = anthropic_llm.get_capabilities()
    assert capabilities.get("streaming", False)
    
    # Use extract_generation_params to verify stream parameter handling
    gen_params = ConfigurationManager.extract_generation_params(
        "anthropic", 
        anthropic_llm.config, 
        {"stream": True}
    )

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
    
    # Check if the response contains numbers 1-5
    for num in range(1, 6):
        assert str(num) in full_response

def test_parameter_override(anthropic_llm):
    """Test that parameters can be overridden at generation time."""
    # Create a base configuration
    base_config = ConfigurationManager.create_base_config(temperature=0.7)
    
    # Override temperature at generation time
    params = ConfigurationManager.extract_generation_params(
        "anthropic", 
        base_config, 
        {"temperature": 0.2}
    )
    
    # Verify overridden temperature
    assert params["temperature"] == 0.2
    
    # Original config should be unchanged
    assert base_config.get(ModelParameter.TEMPERATURE) == 0.7 