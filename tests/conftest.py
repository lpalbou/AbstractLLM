"""
pytest configuration file.
"""

import os
import pytest
from typing import Dict, Any, Generator

from abstractllm import create_llm, ModelParameter
from abstractllm.providers.openai import OpenAIProvider
from abstractllm.providers.anthropic import AnthropicProvider
from abstractllm.providers.ollama import OllamaProvider
from abstractllm.providers.huggingface import HuggingFaceProvider


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    
    Returns:
        OpenAI API key
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """
    Get the Anthropic API key from environment variables.
    
    Returns:
        Anthropic API key
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def openai_provider(openai_api_key) -> Generator[OpenAIProvider, None, None]:
    """
    Create an OpenAI provider for testing.
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        OpenAI provider instance
    """
    provider = create_llm("openai", **{
        ModelParameter.API_KEY: openai_api_key,
        ModelParameter.MODEL: "gpt-3.5-turbo"
    })
    yield provider


@pytest.fixture(scope="session")
def anthropic_provider(anthropic_api_key) -> Generator[AnthropicProvider, None, None]:
    """
    Create an Anthropic provider for testing.
    
    Args:
        anthropic_api_key: Anthropic API key
        
    Returns:
        Anthropic provider instance
    """
    provider = create_llm("anthropic", **{
        ModelParameter.API_KEY: anthropic_api_key,
        ModelParameter.MODEL: "claude-3-haiku-20240307"  # Use a currently supported model
    })
    yield provider


@pytest.fixture(scope="session")
def ollama_provider() -> Generator[OllamaProvider, None, None]:
    """
    Create an Ollama provider for testing.
    
    Returns:
        Ollama provider instance
    """
    # Skip test if Ollama is not running
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
        model_name = "llama2"  # Default, won't be used if skipped
    
    provider = create_llm("ollama", **{
        ModelParameter.BASE_URL: "http://localhost:11434",
        ModelParameter.MODEL: model_name
    })
    yield provider


@pytest.fixture(scope="session")
def huggingface_provider() -> Generator[HuggingFaceProvider, None, None]:
    """
    Create a HuggingFace provider for testing.
    
    Returns:
        HuggingFace provider instance
    """
    try:
        # Small model for quick testing
        provider = create_llm("huggingface", **{
            ModelParameter.MODEL: "distilgpt2",  # Small model for testing
            ModelParameter.DEVICE: "cpu"  # Run on CPU to ensure it works everywhere
        })
        yield provider
    except Exception as e:
        pytest.skip(f"Could not load HuggingFace model: {e}")


@pytest.fixture(params=["openai_provider", "anthropic_provider", "ollama_provider", "huggingface_provider"])
def any_provider(request) -> Generator[Any, None, None]:
    """
    Parametrized fixture that returns each provider.
    This lets us run the same test against all providers.
    
    Args:
        request: pytest request object
        
    Returns:
        Provider instance
    """
    try:
        yield request.getfixturevalue(request.param)
    except pytest.skip.Exception:
        pytest.skip(f"Skipping {request.param} tests") 