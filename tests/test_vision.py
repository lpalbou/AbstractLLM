"""
Tests for vision capabilities in AbstractLLM providers.
"""

import os
import pytest
from typing import Dict, Any, List, Union, Generator
import unittest.mock

from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.providers.openai import OpenAIProvider, VISION_CAPABLE_MODELS as OPENAI_VISION_MODELS
from abstractllm.providers.anthropic import AnthropicProvider, VISION_CAPABLE_MODELS as ANTHROPIC_VISION_MODELS
from abstractllm.providers.ollama import OllamaProvider, VISION_CAPABLE_MODELS as OLLAMA_VISION_MODELS
from abstractllm.utils.image import format_image_for_provider
from tests.utils import validate_response, validate_not_contains, has_capability


# Skip the entire module if vision testing is not enabled
def setup_module(module):
    """Set up the vision test module."""
    if os.environ.get("TEST_VISION", "false").lower() != "true":
        pytest.skip("Vision tests disabled (set TEST_VISION=true to enable)", allow_module_level=True)


def test_vision_capability_detection():
    """Test that vision capabilities are correctly detected for different models."""
    # Test OpenAI models
    for model in OPENAI_VISION_MODELS:
        provider = create_llm("openai", **{
            ModelParameter.MODEL: model
        })
        capabilities = provider.get_capabilities()
        assert has_capability(capabilities, ModelCapability.VISION)
    
    # Test non-vision OpenAI model
    provider = create_llm("openai", **{
        ModelParameter.MODEL: "gpt-3.5-turbo"
    })
    capabilities = provider.get_capabilities()
    assert not has_capability(capabilities, ModelCapability.VISION)
    
    # Test Anthropic models
    for model in ANTHROPIC_VISION_MODELS:
        provider = create_llm("anthropic", **{
            ModelParameter.MODEL: f"{model}-20240620"  # Add a version suffix
        })
        capabilities = provider.get_capabilities()
        assert has_capability(capabilities, ModelCapability.VISION)
    
    # Test Ollama models (using name detection)
    for model in OLLAMA_VISION_MODELS:
        provider = create_llm("ollama", **{
            ModelParameter.MODEL: model
        })
        capabilities = provider.get_capabilities()
        assert has_capability(capabilities, ModelCapability.VISION)


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not set")
@pytest.mark.skipif(os.environ.get("TEST_VISION_API", "false").lower() != "true", 
                   reason="Vision API tests disabled (set TEST_VISION_API=true to enable)")
def test_openai_vision_generation():
    """Test OpenAI vision generation."""
    vision_model = "gpt-4o"
    
    # Create provider with vision-capable model
    provider = create_llm("openai", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.MAX_TOKENS: 100  # Keep response short for testing
    })
    
    # Skip if no API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Test image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg/800px-Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg"
    
    prompt = "What landmark is shown in this image?"
    response = provider.generate(prompt, image=image_url)
    
    assert validate_response(response, ["Eiffel", "Tower", "Paris"])


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
@pytest.mark.skipif(os.environ.get("TEST_VISION_API", "false").lower() != "true", 
                   reason="Vision API tests disabled (set TEST_VISION_API=true to enable)")
def test_anthropic_vision_generation():
    """Test Anthropic vision generation."""
    vision_model = "claude-3-5-sonnet-20240620"
    
    # Create provider with vision-capable model
    provider = create_llm("anthropic", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.MAX_TOKENS: 100  # Keep response short for testing
    })
    
    # Skip if no API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    
    # Test image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg/800px-Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg"
    
    prompt = "What landmark is shown in this image?"
    response = provider.generate(prompt, image=image_url)
    
    assert validate_response(response, ["Eiffel", "Tower", "Paris"])


@pytest.mark.skipif(os.environ.get("TEST_VISION_API", "false").lower() != "true", 
                   reason="Vision API tests disabled (set TEST_VISION_API=true to enable)")
def test_ollama_vision_generation():
    """Test Ollama vision generation."""
    # Try to find a vision-capable Ollama model
    vision_model = None
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Look for vision-capable models
            for model in models:
                model_name = model.get("name", "")
                if "vision" in model_name.lower() or "janus" in model_name.lower():
                    vision_model = model_name
                    break
    except Exception:
        pytest.skip("Ollama not available or error checking models")
    
    if not vision_model:
        pytest.skip("No vision-capable Ollama model found")
    
    # Create provider with vision-capable model
    provider = create_llm("ollama", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.MAX_TOKENS: 100  # Keep response short for testing
    })
    
    # Test image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg/800px-Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg"
    
    prompt = "What landmark is shown in this image?"
    response = provider.generate(prompt, image=image_url)
    
    assert validate_response(response, ["Eiffel", "Tower", "Paris"])


def test_image_format_conversion():
    """Test image format conversion functions."""
    # Test URL format conversion for each provider
    image_url = "https://example.com/image.jpg"
    
    # OpenAI format
    openai_format = format_image_for_provider(image_url, "openai")
    assert openai_format["type"] == "image_url"
    assert openai_format["image_url"]["url"] == image_url
    assert openai_format["image_url"]["detail"] == "auto"
    
    # Anthropic format
    anthropic_format = format_image_for_provider(image_url, "anthropic")
    assert anthropic_format["type"] == "image"
    assert anthropic_format["source"]["type"] == "url"
    assert anthropic_format["source"]["url"] == image_url
    
    # Ollama format
    ollama_format = format_image_for_provider(image_url, "ollama")
    assert ollama_format["url"] == image_url


@unittest.mock.patch("abstractllm.providers.openai.OpenAIProvider.generate")
@unittest.mock.patch("abstractllm.utils.image.preprocess_image_inputs")
def test_vision_input_preprocessing(mock_preprocess, mock_generate):
    """Test that image inputs are correctly preprocessed before being sent to the provider."""
    # Set up the mocks
    mock_preprocess.return_value = {"messages": [{"role": "user", "content": [{"type": "text", "text": "test"}, {"type": "image_url", "image_url": {"url": "test_url"}}]}]}
    mock_generate.return_value = "Test response"
    
    # Create provider with vision-capable model
    provider = create_llm("openai", **{
        ModelParameter.MODEL: "gpt-4o"
    })
    
    # Call generate with an image
    provider.generate("What's in this image?", image="test_url")
    
    # Verify that preprocess_image_inputs was called with the correct parameters
    mock_preprocess.assert_called_once()
    args, _ = mock_preprocess.call_args
    assert args[1] == "openai"  # The provider name should be passed to preprocess_image_inputs
    
    # Verify that the generate method was called with the preprocessed parameters
    mock_generate.assert_called_once() 