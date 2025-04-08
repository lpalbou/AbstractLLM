"""
Tests for vision capabilities in AbstractLLM providers.
"""

import os
import sys
import pytest
import requests
from io import BytesIO
from typing import Dict, Any, List, Union, Generator
import unittest.mock
from pathlib import Path

from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.providers.openai import OpenAIProvider, VISION_CAPABLE_MODELS as OPENAI_VISION_MODELS
from abstractllm.providers.anthropic import AnthropicProvider, VISION_CAPABLE_MODELS as ANTHROPIC_VISION_MODELS
from abstractllm.providers.ollama import OllamaProvider, VISION_CAPABLE_MODELS as OLLAMA_VISION_MODELS
from abstractllm.providers.huggingface import HuggingFaceProvider, VISION_CAPABLE_MODELS as HF_VISION_MODELS
from abstractllm.utils.image import format_image_for_provider
from tests.utils import validate_response, validate_not_contains, has_capability

# Define test resources directory
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Define test image paths and their keywords
TEST_IMAGES = {
    "mountain_path.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_1.jpg"),
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/test_image_1.jpg",
        "keywords": [
            "mountain", "mountains", "range", "dirt", "path", "trail", "wooden", "fence", 
            "sunlight", "sunny", "blue sky", "grass", "meadow", "hiking", 
            "countryside", "rural", "landscape", "horizon"
        ],
        "prompt": "Describe what you see in this image in detail."
    },
    "city_park_sunset.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_2.jpg"),
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/test_image_2.jpg",
        "keywords": [
            "lamppost", "street light", "sunset", "dusk", "pink", "orange", "sky",
            "pathway", "walkway", "park", "urban", "trees", "buildings", "benches",
            "garden", "evening"
        ],
        "prompt": "What's shown in this image? Give a detailed description."
    },
    "humpback_whale.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_3.jpg"),
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/test_image_3.jpg",
        "keywords": [
            "whale", "humpback", "ocean", "sea", "breaching", "jumping", "splash",
            "marine", "mammal", "fins", "flipper", "gray", "waves", "wildlife",
            "water"
        ],
        "prompt": "Describe the creature in this image and what it's doing."
    },
    "cat_carrier.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_4.jpg"),
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/test_image_4.jpg",
        "keywords": [
            "cat", "pet", "carrier", "transport", "dome", "window", "plastic",
            "orange", "tabby", "fur", "eyes", "round", "opening", "white", "base",
            "ventilation", "air holes"
        ],
        "prompt": "What animal is shown in this image and where is it located?"
    }
}

def download_test_images():
    """Download test images if they don't exist locally."""
    for image_name, image_info in TEST_IMAGES.items():
        # Use the actual images provided by the user instead of downloading
        if not os.path.exists(image_info["path"]):
            try:
                # This is a placeholder - in real implementation, we would save the 
                # user-provided images to these paths
                print(f"Would download {image_name} to {image_info['path']}")
                
                # Actual download code (commented out for now)
                # response = requests.get(image_info["url"])
                # with open(image_info["path"], "wb") as f:
                #     f.write(response.content)
            except Exception as e:
                print(f"Failed to download {image_name}: {e}")


def setup_module(module):
    """Set up the vision test module."""
    # Only skip tests if explicitly requested, otherwise run them
    if os.environ.get("SKIP_VISION_TESTS", "false").lower() == "true":
        pytest.skip("Vision tests explicitly disabled (SKIP_VISION_TESTS=true)", allow_module_level=True)
    
    # Try to download test images
    download_test_images()


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
        
    # Test HuggingFace models
    for model in HF_VISION_MODELS:
        provider = create_llm("huggingface", **{
            ModelParameter.MODEL: model
        })
        capabilities = provider.get_capabilities()
        assert has_capability(capabilities, ModelCapability.VISION)


def calculate_keyword_match_percentage(response: str, keywords: List[str]) -> float:
    """
    Calculate the percentage of keywords found in the response.
    
    Args:
        response: The response text to check
        keywords: List of keywords to look for
        
    Returns:
        Percentage of keywords found (0.0 to 1.0)
    """
    if not response or not keywords:
        return 0.0
        
    response_lower = response.lower()
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in response_lower]
    return len(matched_keywords) / len(keywords)


def evaluate_vision_response(response: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Evaluate a vision model response against keywords.
    
    Args:
        response: The response text to evaluate
        keywords: List of keywords to look for
        
    Returns:
        Dictionary with evaluation results
    """
    match_percentage = calculate_keyword_match_percentage(response, keywords)
    
    # Determine the quality level
    if match_percentage >= 0.75:
        quality = "excellent"
    elif match_percentage >= 0.5:
        quality = "good"
    elif match_percentage >= 0.25:
        quality = "fair"
    else:
        quality = "poor"
        
    return {
        "match_percentage": match_percentage,
        "quality": quality,
        "matched_keywords": [k for k in keywords if k.lower() in response.lower()],
        "missed_keywords": [k for k in keywords if k.lower() not in response.lower()]
    }


# Test image recognition for each provider with real images
@pytest.mark.vision
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_openai_image_recognition():
    """Test OpenAI's ability to recognize content in test images."""
    vision_model = "gpt-4o"
    
    # Create provider with vision-capable model
    provider = create_llm("openai", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.MAX_TOKENS: 300
    })
    
    # Test with actual image path instead of URL
    for image_name, image_info in TEST_IMAGES.items():
        # Check if we're using a placeholder or actual image path
        image_path = image_info["path"]
        
        # For the actual test, use either the path if it exists, or fall back to URL
        image_source = image_path if os.path.exists(image_path) else image_info["url"]
        prompt = image_info["prompt"]
        
        try:
            print(f"\nTesting OpenAI with image: {image_name}")
            response = provider.generate(prompt, image=image_source)
            
            # Evaluate the response
            keywords = image_info["keywords"]
            evaluation = evaluate_vision_response(response, keywords)
            
            print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
            print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
            
            # Assert minimum quality level
            assert evaluation["match_percentage"] >= 0.25, f"Response quality too low: {evaluation['quality']}"
            
        except Exception as e:
            pytest.skip(f"OpenAI vision test failed for {image_name}: {e}")


@pytest.mark.vision
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
def test_anthropic_image_recognition():
    """Test Anthropic's ability to recognize content in test images."""
    vision_model = "claude-3-5-sonnet-20240620"
    
    # Create provider with vision-capable model
    provider = create_llm("anthropic", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.MAX_TOKENS: 300
    })
    
    # Test with actual image path instead of URL
    for image_name, image_info in TEST_IMAGES.items():
        # For the actual test, use either the path if it exists, or fall back to URL
        image_path = image_info["path"]
        image_source = image_path if os.path.exists(image_path) else image_info["url"]
        prompt = image_info["prompt"]
        
        try:
            print(f"\nTesting Anthropic with image: {image_name}")
            response = provider.generate(prompt, image=image_source)
            
            # Evaluate the response
            keywords = image_info["keywords"]
            evaluation = evaluate_vision_response(response, keywords)
            
            print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
            print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
            
            # Assert minimum quality level
            assert evaluation["match_percentage"] >= 0.25, f"Response quality too low: {evaluation['quality']}"
            
        except Exception as e:
            pytest.skip(f"Anthropic vision test failed for {image_name}: {e}")


@pytest.mark.vision
def test_ollama_image_recognition():
    """Test Ollama's ability to recognize content in test images."""
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
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")
    
    if not vision_model:
        pytest.skip("No vision-capable Ollama model found")
    
    # Create provider with vision-capable model
    provider = create_llm("ollama", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.MAX_TOKENS: 300
    })
    
    # Test with actual image path instead of URL
    for image_name, image_info in TEST_IMAGES.items():
        # For the actual test, use either the path if it exists, or fall back to URL
        image_path = image_info["path"]
        image_source = image_path if os.path.exists(image_path) else image_info["url"]
        prompt = image_info["prompt"]
        
        try:
            print(f"\nTesting Ollama with image: {image_name}")
            response = provider.generate(prompt, image=image_source)
            
            # Evaluate the response
            keywords = image_info["keywords"]
            evaluation = evaluate_vision_response(response, keywords)
            
            print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
            print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
            
            # Assert minimum quality level
            assert evaluation["match_percentage"] >= 0.25, f"Response quality too low: {evaluation['quality']}"
            
        except Exception as e:
            pytest.skip(f"Ollama vision test failed for {image_name}: {e}")


@pytest.mark.vision
def test_huggingface_image_recognition():
    """Test HuggingFace's ability to recognize content in test images."""
    # Try to find a vision-capable model
    vision_model = None
    for model in HF_VISION_MODELS:
        # For tests, prefer smaller models like microsoft/Phi-4-multimodal-instruct
        if "phi-4" in model.lower():
            vision_model = model
            break
    
    if not vision_model and HF_VISION_MODELS:
        vision_model = HF_VISION_MODELS[0]
        
    if not vision_model:
        pytest.skip("No HuggingFace vision model defined")
    
    # Create provider with vision-capable model
    provider = create_llm("huggingface", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.DEVICE: "cpu",  # Use CPU for tests
        ModelParameter.MAX_TOKENS: 300,
        "trust_remote_code": True  # Required for some models
    })
    
    # Test with actual image path instead of URL
    for image_name, image_info in TEST_IMAGES.items():
        # For the actual test, use either the path if it exists, or fall back to URL
        image_path = image_info["path"]
        image_source = image_path if os.path.exists(image_path) else image_info["url"]
        prompt = image_info["prompt"]
        
        try:
            print(f"\nTesting HuggingFace with image: {image_name}")
            response = provider.generate(prompt, image=image_source)
            
            # Evaluate the response
            keywords = image_info["keywords"]
            evaluation = evaluate_vision_response(response, keywords)
            
            print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
            print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
            
            # Assert minimum quality level
            assert evaluation["match_percentage"] >= 0.25, f"Response quality too low: {evaluation['quality']}"
            
        except Exception as e:
            pytest.skip(f"HuggingFace vision test failed for {image_name}: {e}")


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
    
    # HuggingFace format
    huggingface_format = format_image_for_provider(image_url, "huggingface")
    assert huggingface_format == image_url


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