"""
Tests for the AbstractLLM factory module.
"""

import pytest
import os
from typing import Dict, Any

from abstractllm import create_llm, AbstractLLMInterface, ModelParameter
from abstractllm.providers.openai import OpenAIProvider
from abstractllm.providers.anthropic import AnthropicProvider
from abstractllm.providers.ollama import OllamaProvider
from abstractllm.providers.huggingface import HuggingFaceProvider, DEFAULT_MODEL


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
    
    # Test with Hugging Face provider - only if test environment flag is set
    if os.environ.get("TEST_HUGGINGFACE", "false").lower() == "true":
        provider = create_llm("huggingface", **{
            ModelParameter.MODEL: DEFAULT_MODEL,
            ModelParameter.DEVICE: "cpu"
        })
        assert isinstance(provider, HuggingFaceProvider)
        assert isinstance(provider, AbstractLLMInterface)
    
    # Test with Ollama provider - only if test environment flag is set
    if os.environ.get("TEST_OLLAMA", "false").lower() == "true":
        try:
            provider = create_llm("ollama")
            assert isinstance(provider, OllamaProvider)
            assert isinstance(provider, AbstractLLMInterface)
        except:
            # Skip if Ollama is not running
            pass


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
        assert config.get("model") == "gpt-3.5-turbo"
        assert config.get("temperature") == 0.5
        
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
        assert config.get(ModelParameter.MODEL) == "gpt-3.5-turbo"
        assert config.get(ModelParameter.TEMPERATURE) == 0.7


def test_factory_with_environment_variables() -> None:
    """
    Test that the factory uses environment variables correctly.
    """
    # Test with OpenAI API key from environment
    if os.environ.get("OPENAI_API_KEY"):
        # Save original and remove from environment temporarily
        original_key = os.environ["OPENAI_API_KEY"]
        temp_key = original_key
        
        try:
            # Test with missing env var
            os.environ.pop("OPENAI_API_KEY")
            
            # Should fail without API key
            provider = create_llm("openai")
            with pytest.raises(ValueError):
                provider.generate("test")
            
            # Restore and test with env var
            os.environ["OPENAI_API_KEY"] = temp_key
            provider = create_llm("openai")
            
            # Call generate to ensure the API key is obtained from the environment
            # This will make the provider read the API key from the environment
            response = provider.generate("test")
            
            # Now check if the API key is in the config
            config = provider.get_config()
            assert config.get(ModelParameter.API_KEY) == temp_key
        finally:
            # Restore original environment
            os.environ["OPENAI_API_KEY"] = original_key 