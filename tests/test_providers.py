"""
Tests for provider-specific features.
"""

import pytest
import os
from typing import Dict, Any, List, Union, Generator

from abstractllm import ModelParameter, ModelCapability
from abstractllm.providers.openai import OpenAIProvider
from abstractllm.providers.anthropic import AnthropicProvider
from abstractllm.providers.ollama import OllamaProvider
from abstractllm.providers.huggingface import HuggingFaceProvider
from tests.utils import validate_response, validate_not_contains, has_capability, count_tokens
from tests.examples.test_prompts import VISION_PROMPT, FUNCTION_CALLING_PROMPT


def test_openai_specific_features(openai_provider: OpenAIProvider) -> None:
    """
    Test OpenAI-specific features.
    
    Args:
        openai_provider: OpenAI provider instance
    """
    # Check capabilities
    capabilities = openai_provider.get_capabilities()
    
    # Test with a simple prompt to verify connection
    response = openai_provider.generate("What is 2+2?")
    assert isinstance(response, str)
    assert validate_response(response, ["4", "four"])
    
    # Test with different model if capability allows
    original_model = openai_provider.config.get(ModelParameter.MODEL)
    try:
        # Try GPT-4 if available
        if os.environ.get("TEST_GPT4", "false").lower() == "true":
            openai_provider.set_config(**{ModelParameter.MODEL: "gpt-4"})
            response = openai_provider.generate("What is 2+2?")
            assert validate_response(response, ["4", "four"])
    finally:
        # Restore original model
        openai_provider.set_config(**{ModelParameter.MODEL: original_model})
    
    # Test vision if capability is available
    if has_capability(capabilities, ModelCapability.VISION) and os.environ.get("TEST_VISION", "false").lower() == "true":
        vision_prompt = VISION_PROMPT["prompt"]
        image_url = VISION_PROMPT["image_url"]
        expected_contains = VISION_PROMPT["expected_contains"]
        
        # This will need to be adjusted based on actual OpenAI vision API implementation
        response = openai_provider.generate(
            vision_prompt,
            **{
                "model": "gpt-4-vision-preview",
                "images": [{"url": image_url, "detail": "low"}]
            }
        )
        
        assert validate_response(response, expected_contains)


def test_anthropic_specific_features(anthropic_provider: AnthropicProvider) -> None:
    """
    Test Anthropic-specific features.
    
    Args:
        anthropic_provider: Anthropic provider instance
    """
    # Check capabilities
    capabilities = anthropic_provider.get_capabilities()
    
    # Test with a simple prompt to verify connection
    response = anthropic_provider.generate("What is 2+2?")
    assert isinstance(response, str)
    assert validate_response(response, ["4", "four"])
    
    # Test with different model if capability allows
    original_model = anthropic_provider.config.get(ModelParameter.MODEL)
    try:
        # Try newest Claude model if available
        if os.environ.get("TEST_CLAUDE3", "false").lower() == "true":
            anthropic_provider.set_config(**{ModelParameter.MODEL: "claude-3-opus-20240229"})
            response = anthropic_provider.generate("What is 2+2?")
            assert validate_response(response, ["4", "four"])
    finally:
        # Restore original model
        anthropic_provider.set_config(**{ModelParameter.MODEL: original_model})
    
    # Test vision if capability is available
    if has_capability(capabilities, ModelCapability.VISION) and os.environ.get("TEST_VISION", "false").lower() == "true":
        vision_prompt = VISION_PROMPT["prompt"]
        image_url = VISION_PROMPT["image_url"]
        expected_contains = VISION_PROMPT["expected_contains"]
        
        # This will need to be adjusted based on actual Anthropic vision API implementation
        response = anthropic_provider.generate(
            vision_prompt,
            **{
                "model": "claude-3-opus-20240229",
                "images": [{"url": image_url}]
            }
        )
        
        assert validate_response(response, expected_contains)


def test_ollama_specific_features(ollama_provider: OllamaProvider) -> None:
    """
    Test Ollama-specific features.
    
    Args:
        ollama_provider: Ollama provider instance
    """
    # Test with a simple prompt to verify connection
    response = ollama_provider.generate("What is 2+2?")
    assert isinstance(response, str)
    assert validate_response(response, ["4", "four"])
    
    # Test available models
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models and len(models) > 1:
                # Try with a different model
                original_model = ollama_provider.config.get(ModelParameter.MODEL)
                try:
                    new_model = next((m["name"] for m in models if m["name"] != original_model), None)
                    if new_model:
                        ollama_provider.set_config(**{ModelParameter.MODEL: new_model})
                        response = ollama_provider.generate("What is 2+2?")
                        assert validate_response(response, ["4", "four"])
                finally:
                    # Restore original model
                    ollama_provider.set_config(**{ModelParameter.MODEL: original_model})
    except Exception:
        pass  # Skip model switching test if API call fails


def test_huggingface_specific_features(huggingface_provider: HuggingFaceProvider) -> None:
    """
    Test HuggingFace-specific features.
    
    Args:
        huggingface_provider: HuggingFace provider instance
    """
    # Test with a simple prompt
    response = huggingface_provider.generate("The capital of France is")
    assert isinstance(response, str)
    assert len(response) > 0  # Small models might not give factual answers
    
    # Test cache management functions if available
    if hasattr(huggingface_provider, "list_cached_models") and os.environ.get("TEST_HF_CACHE", "false").lower() == "true":
        try:
            cached_models = HuggingFaceProvider.list_cached_models()
            assert isinstance(cached_models, list)
            
            # Don't actually clear cache in tests unless specifically requested
            # and only clear a test model
            if os.environ.get("TEST_HF_CACHE_CLEAR", "false").lower() == "true" and os.environ.get("TEST_HF_CACHE_MODEL"):
                test_model = os.environ.get("TEST_HF_CACHE_MODEL")
                HuggingFaceProvider.clear_model_cache(model_name=test_model)
        except ImportError:
            # Skip if huggingface_hub not installed
            pass


@pytest.mark.parametrize("provider_fixture", ["openai_provider", "anthropic_provider"])
def test_cloud_provider_error_handling(request, provider_fixture):
    """
    Test error handling for cloud providers.
    
    Args:
        request: pytest request object
        provider_fixture: Provider fixture name
    """
    try:
        provider = request.getfixturevalue(provider_fixture)
    except pytest.skip.Exception:
        pytest.skip(f"Skipping {provider_fixture} tests")
        return
    
    # Invalid parameter test
    try:
        # Use a parameter we know is invalid
        response = provider.generate("This is a test", invalid_parameter=123)
        # Some providers may ignore invalid parameters instead of raising errors
    except Exception as e:
        # We expect an error, so this is actually success
        assert "invalid" in str(e).lower() or "unknown" in str(e).lower() or "unexpected" in str(e).lower()
    
    # Test with invalid model
    original_model = provider.config.get(ModelParameter.MODEL)
    try:
        provider.set_config(**{ModelParameter.MODEL: "non-existent-model-123"})
        with pytest.raises(Exception):
            provider.generate("This is a test")
    except Exception as e:
        # Expected error
        pass
    finally:
        # Restore original model
        provider.set_config(**{ModelParameter.MODEL: original_model}) 