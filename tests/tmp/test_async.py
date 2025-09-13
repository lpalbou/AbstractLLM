"""
Tests for asynchronous functionality across providers.
"""

import os
import pytest
import asyncio
from abstractllm import create_llm, ModelCapability, ModelParameter
from tests.utils import collect_stream_async

@pytest.mark.asyncio
async def test_openai_async():
    """Test OpenAI async generation."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    llm = create_llm("openai")
    capabilities = llm.get_capabilities()
    
    # Skip if async is not supported
    if not capabilities.get(ModelCapability.ASYNC, False):
        pytest.skip("Provider does not support async generation")
    
    # Test basic async generation
    response = await llm.generate_async("What is 2+2?")
    assert isinstance(response, str)
    assert "4" in response.lower() or "four" in response.lower()
    
    # Test async streaming if supported
    if capabilities.get(ModelCapability.STREAMING, False):
        stream = await llm.generate_async("Count from 1 to 5", stream=True)
        response = await collect_stream_async(stream)
        assert isinstance(response, str)
        assert len(response) > 0
        # Check for numbers in response
        for num in range(1, 6):
            assert str(num) in response

@pytest.mark.asyncio
async def test_anthropic_async():
    """Test Anthropic async generation."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    llm = create_llm("anthropic", **{
        ModelParameter.API_KEY: api_key,
        ModelParameter.MODEL: "claude-3-haiku-20240307"  # Use a currently supported model
    })
    capabilities = llm.get_capabilities()
    
    # Skip if async is not supported
    if not capabilities.get(ModelCapability.ASYNC, False):
        pytest.skip("Provider does not support async generation")
    
    # Test basic async generation
    response = await llm.generate_async("What is 2+2?")
    assert isinstance(response, str)
    assert "4" in response.lower() or "four" in response.lower()
    
    # Test async streaming if supported
    if capabilities.get(ModelCapability.STREAMING, False):
        stream = await llm.generate_async("Count from 1 to 5", stream=True)
        response = await collect_stream_async(stream)
        assert isinstance(response, str)
        assert len(response) > 0
        # Check for numbers in response
        for num in range(1, 6):
            assert str(num) in response

@pytest.mark.asyncio
async def test_ollama_async():
    """Test Ollama async generation."""
    # Try to connect to Ollama
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            pytest.skip("Ollama API not accessible")
            
        # Check if at least one model is available
        models = response.json().get("models", [])
        if not models:
            pytest.skip("No Ollama models available")
            
        # Use the first available model
        model_name = models[0]["name"]
    except Exception:
        pytest.skip("Ollama API not accessible or other error")
    
    llm = create_llm("ollama", **{
        ModelParameter.MODEL: model_name,
        ModelParameter.BASE_URL: "http://localhost:11434"
    })
    capabilities = llm.get_capabilities()
    
    # Skip if async is not supported
    if not capabilities.get(ModelCapability.ASYNC, False):
        pytest.skip("Provider does not support async generation")
    
    # Test basic async generation
    response = await llm.generate_async("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test async streaming if supported
    if capabilities.get(ModelCapability.STREAMING, False):
        stream = await llm.generate_async("Hello, how are you?", stream=True)
        response = await collect_stream_async(stream)
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_multiple_async_calls():
    """Test multiple async calls in parallel."""
    # Skip if no OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    llm = create_llm("openai")
    capabilities = llm.get_capabilities()
    
    # Skip if async is not supported
    if not capabilities.get(ModelCapability.ASYNC, False):
        pytest.skip("Provider does not support async generation")
    
    # Run multiple requests in parallel
    prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Name three colors of the rainbow."
    ]
    
    # Gather all tasks
    tasks = [llm.generate_async(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    
    # Check that we got valid responses
    assert len(responses) == len(prompts)
    for response in responses:
        assert isinstance(response, str)
        assert len(response) > 0
    
    # Check first response contains '4' or 'four'
    assert "4" in responses[0].lower() or "four" in responses[0].lower()
    
    # Check second response contains 'Paris'
    assert "paris" in responses[1].lower()
    
    # Check third response contains at least one color
    colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "purple"]
    assert any(color in responses[2].lower() for color in colors) 