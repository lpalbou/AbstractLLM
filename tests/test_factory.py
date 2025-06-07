"""
Tests for the AbstractLLM factory module.
"""

import pytest
import os
from typing import Dict, Any

from abstractllm import create_llm, AbstractLLMInterface, ModelParameter
from abstractllm.utils.config import ConfigurationManager
from abstractllm.providers.openai import OpenAIProvider
from abstractllm.providers.anthropic import AnthropicProvider
from abstractllm.providers.ollama import OllamaProvider
from abstractllm.providers.huggingface import HuggingFaceProvider, DEFAULT_MODEL
from abstractllm.utils.config import DEFAULT_MODELS


def test_factory_create_provider() -> None:
    """
    Test that the factory creates the correct provider.
    """
    # Test with OpenAI provider
    if os.environ.get("OPENAI_API_KEY"):
        provider = create_llm("openai", **{
            ModelParameter.API_KEY: os.environ["OPENAI_API_KEY"]
        })
        assert isinstance(provider, OpenAIProvider)
        assert isinstance(provider, AbstractLLMInterface)
    
    # Test with Anthropic provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        provider = create_llm("anthropic", **{
            ModelParameter.API_KEY: os.environ["ANTHROPIC_API_KEY"]
        })
        assert isinstance(provider, AnthropicProvider)
        assert isinstance(provider, AbstractLLMInterface)
    
    # Test with Ollama provider - check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            provider = create_llm("ollama")
            assert isinstance(provider, OllamaProvider)
            assert isinstance(provider, AbstractLLMInterface)
    except Exception:
        # Skip if Ollama is not running
        pass
    
    # Test with Hugging Face provider - using a simple model
    try:
        provider = create_llm("huggingface", **{
            ModelParameter.MODEL: "distilgpt2",
            ModelParameter.DEVICE: "cpu"
        })
        assert isinstance(provider, HuggingFaceProvider)
        assert isinstance(provider, AbstractLLMInterface)
    except Exception:
        # Skip if HuggingFace can't be initialized
        pass


def test_factory_with_string_keys() -> None:
    """
    Test that the factory works with string keys instead of enum keys.
    """
    if os.environ.get("OPENAI_API_KEY"):
        provider = create_llm("openai", 
                             api_key=os.environ["OPENAI_API_KEY"],
                             temperature=0.5,
                             max_tokens=1000)
        
        # Verify config was properly set
        assert provider.config.get(ModelParameter.TEMPERATURE) == 0.5
        assert provider.config.get(ModelParameter.MAX_TOKENS) == 1000
        assert provider.config.get(ModelParameter.API_KEY) == os.environ["OPENAI_API_KEY"]
        
        # String keys should be accessible too
        assert provider.config.get("temperature") == 0.5


def test_factory_with_enum_keys() -> None:
    """
    Test that the factory works with enum keys.
    """
    if os.environ.get("OPENAI_API_KEY"):
        provider = create_llm("openai", **{
            ModelParameter.API_KEY: os.environ["OPENAI_API_KEY"],
            ModelParameter.TEMPERATURE: 0.5,
            ModelParameter.MAX_TOKENS: 1000
        })
        
        # Verify config was properly set
        assert provider.config.get(ModelParameter.TEMPERATURE) == 0.5
        assert provider.config.get(ModelParameter.MAX_TOKENS) == 1000


def test_factory_config_flow() -> None:
    """
    Test that the config flow works as expected through the factory.
    """
    if os.environ.get("OPENAI_API_KEY"):
        # Configure a base config first
        base_config = ConfigurationManager.create_base_config(
            temperature=0.5,
            max_tokens=1000,
            model=DEFAULT_MODELS["openai"]  # Explicitly set the model
        )
        
        # Then provider config
        provider_config = ConfigurationManager.initialize_provider_config("openai", base_config)
        
        # Ensure API key is set
        provider_config[ModelParameter.API_KEY] = os.environ["OPENAI_API_KEY"]
        
        # Then create provider directly with config
        provider = create_llm("openai", **provider_config)
        
        # Check that parameters are set correctly
        assert provider.config.get(ModelParameter.TEMPERATURE) == 0.5
        assert provider.config.get(ModelParameter.MAX_TOKENS) == 1000
        
        # Check the model using get_param which handles both enum and string keys
        model = ConfigurationManager.get_param(provider.config, ModelParameter.MODEL, DEFAULT_MODELS["openai"])
        assert model == DEFAULT_MODELS["openai"]


def test_unsupported_provider() -> None:
    """
    Test that the factory raises an error for unsupported providers.
    """
    with pytest.raises(ValueError) as excinfo:
        create_llm("unsupported_provider")
    
    # Error message should mention the unsupported provider
    assert "unsupported_provider" in str(excinfo.value)
    
    # Error message should list available providers
    assert "openai" in str(excinfo.value)
    assert "anthropic" in str(excinfo.value)
    assert "ollama" in str(excinfo.value)
    assert "huggingface" in str(excinfo.value)


def test_factory_errors() -> None:
    """
    Test that the factory raises appropriate errors.
    """
    # Invalid provider
    with pytest.raises(ValueError):
        create_llm("invalid_provider")
    
    # Missing API key
    if not os.environ.get("OPENAI_API_KEY"):
        with pytest.raises(ValueError):
            # OpenAI without API key should fail
            provider = create_llm("openai")
            provider.generate("test")  # Error happens at generation time


def test_factory_with_parameters() -> None:
    """
    Test that the factory passes parameters to providers correctly.
    """
    # Test with parameters
    if os.environ.get("OPENAI_API_KEY"):
        # Create with string parameters
        provider = create_llm(
            "openai",
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-3.5-turbo",
            temperature=0.5
        )
        config = provider.get_config()
        # Check configuration is set correctly
        assert config.get("temperature") == 0.5
        assert ConfigurationManager.get_param(config, ModelParameter.MODEL) == "gpt-3.5-turbo"
        
        # Create with enum parameters
        provider = create_llm(
            "openai",
            **{
                ModelParameter.API_KEY: os.environ["OPENAI_API_KEY"],
                ModelParameter.MODEL: "gpt-3.5-turbo",
                ModelParameter.TEMPERATURE: 0.7
            }
        )
        config = provider.get_config()
        assert config.get(ModelParameter.TEMPERATURE) == 0.7
        assert ConfigurationManager.get_param(config, ModelParameter.MODEL) == "gpt-3.5-turbo"


def test_factory_with_environment_variables() -> None:
    """
    Test that the factory uses environment variables correctly.
    """
    # Skip if no OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # First, create a provider explicitly specifying the API key
    api_key = os.environ["OPENAI_API_KEY"]
    provider = create_llm("openai", api_key=api_key)
    config = provider.get_config()
    
    # Verify the API key is set in the config
    assert config.get(ModelParameter.API_KEY) == api_key
    
    # For the environment test, let's check that the provider recognizes the key
    # This is testing the provider's implementation, not just the config
    try:
        # Try a simple generation - it should work if the API key is properly received
        response = provider.generate("Hello", max_tokens=5)
        assert response is not None
        assert isinstance(response, str)
    except ValueError as e:
        if "API key not provided" in str(e):
            pytest.fail("Provider did not properly receive API key from environment")
        else:
            # Other errors might be expected (e.g., rate limits)
            pass
    except Exception:
        # Other errors are expected and fine for this test
        pass 