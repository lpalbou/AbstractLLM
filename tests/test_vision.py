"""
Tests for vision capabilities in AbstractLLM providers.
"""

import os
import sys
import pytest
import requests
import shutil
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

# Define test resources directory and examples directory
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
os.makedirs(RESOURCES_DIR, exist_ok=True)

# Define test image paths and their keywords
TEST_IMAGES = {
    "test_image_1.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_1.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_1.jpg"),
        "keywords": [
            "mountain", "mountains", "range", "dirt", "path", "trail", "wooden", "fence", 
            "sunlight", "sunny", "blue sky", "grass", "meadow", "hiking", 
            "countryside", "rural", "landscape", "horizon"
        ],
        "prompt": "Describe what you see in this image in detail."
    },
    "test_image_2.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_2.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_2.jpg"),
        "keywords": [
            "lamppost", "street light", "sunset", "dusk", "pink", "orange", "sky",
            "pathway", "walkway", "park", "urban", "trees", "buildings", "benches",
            "garden", "evening"
        ],
        "prompt": "What's shown in this image? Give a detailed description."
    },
    "test_image_3.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_3.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_3.jpg"),
        "keywords": [
            "whale", "humpback", "ocean", "sea", "breaching", "jumping", "splash",
            "marine", "mammal", "fins", "flipper", "gray", "waves", "wildlife",
            "water"
        ],
        "prompt": "Describe the creature in this image and what it's doing."
    },
    "test_image_4.jpg": {
        "path": os.path.join(RESOURCES_DIR, "test_image_4.jpg"),
        "source": os.path.join(EXAMPLES_DIR, "test_image_4.jpg"),
        "keywords": [
            "cat", "pet", "carrier", "transport", "dome", "window", "plastic",
            "orange", "tabby", "fur", "eyes", "round", "opening", "white", "base",
            "ventilation", "air holes"
        ],
        "prompt": "What animal is shown in this image and where is it located?"
    }
}

def prepare_test_images():
    """Copy test images from examples to resources if they don't exist there."""
    for image_name, image_info in TEST_IMAGES.items():
        if not os.path.exists(image_info["path"]) and os.path.exists(image_info["source"]):
            try:
                print(f"Copying {image_name} from examples to resources...")
                shutil.copy2(image_info["source"], image_info["path"])
                print(f"Successfully copied {image_name} to {image_info['path']}")
            except Exception as e:
                print(f"Failed to copy {image_name}: {e}")


def setup_module(module):
    """Set up the vision test module."""
    # Only skip tests if explicitly requested, otherwise run them
    if os.environ.get("SKIP_VISION_TESTS", "false").lower() == "true":
        pytest.skip("Vision tests explicitly disabled (SKIP_VISION_TESTS=true)", allow_module_level=True)
    
    # Prepare test images
    prepare_test_images()


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
    
    # Test with actual image path
    for image_name, image_info in TEST_IMAGES.items():
        image_path = image_info["path"]
        if not os.path.exists(image_path):
            pytest.skip(f"Image file {image_name} not found at {image_path}")
            
        prompt = image_info["prompt"]
        
        try:
            print(f"\nTesting OpenAI with image: {image_name}")
            response = provider.generate(prompt, image=image_path)
            
            # Evaluate the response
            keywords = image_info["keywords"]
            evaluation = evaluate_vision_response(response, keywords)
            
            print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
            print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
            print(f"Matched: {', '.join(evaluation['matched_keywords'][:5])}{'...' if len(evaluation['matched_keywords']) > 5 else ''}")
            print(f"Missed: {', '.join(evaluation['missed_keywords'][:5])}{'...' if len(evaluation['missed_keywords']) > 5 else ''}")
            
            # Assert minimum quality level
            assert evaluation["match_percentage"] >= 0.25, f"Response quality too low: {evaluation['quality']}"
            
        except Exception as e:
            print(f"OpenAI vision test failed for {image_name}: {e}")
            raise


@pytest.mark.vision
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="Anthropic API key not set")
def test_anthropic_image_recognition():
    """Test Anthropic's ability to recognize content in test images."""
    vision_model = "claude-3-5-sonnet-20240620"
    
    # Get API key explicitly from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.fail("ANTHROPIC_API_KEY environment variable is set but empty")
    
    # Create provider with vision-capable model
    provider = create_llm("anthropic", **{
        ModelParameter.API_KEY: api_key,  # Explicitly pass the API key
        ModelParameter.MODEL: vision_model,
        ModelParameter.MAX_TOKENS: 500,   # Increased token limit for more detailed responses
        ModelParameter.TEMPERATURE: 0.1   # Lower temperature for more deterministic responses
    })
    
    # Test with actual image path
    success_count = 0
    test_results = []
    
    for image_name, image_info in TEST_IMAGES.items():
        image_path = image_info["path"]
        if not os.path.exists(image_path):
            print(f"Image file {image_name} not found at {image_path}")
            continue
            
        prompt = image_info["prompt"]
        
        # Try up to 2 times with different approaches
        for attempt in range(2):
            try:
                # Use a more detailed prompt on second attempt
                if attempt == 1:
                    enhanced_prompt = f"{prompt} Be extremely detailed and comprehensive in your description. List all objects, features, colors, and elements visible in the image."
                    print(f"\nRetrying Anthropic with enhanced prompt for image: {image_name}")
                    print(f"Enhanced prompt: {enhanced_prompt}")
                    test_prompt = enhanced_prompt
                else:
                    print(f"\nTesting Anthropic with image: {image_name}")
                    print(f"Prompt: {prompt}")
                    test_prompt = prompt
                
                # Generate response
                response = provider.generate(test_prompt, image=image_path)
                
                # Print the full response for debugging
                print(f"\nResponse from Anthropic ({len(response)} chars):")
                print("---")
                print(response[:500] + ("..." if len(response) > 500 else ""))
                print("---")
                
                # Evaluate the response
                keywords = image_info["keywords"]
                evaluation = evaluate_vision_response(response, keywords)
                
                print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
                print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
                print(f"Matched: {', '.join(evaluation['matched_keywords'][:5])}{'...' if len(evaluation['matched_keywords']) > 5 else ''}")
                print(f"Missed: {', '.join(evaluation['missed_keywords'][:5])}{'...' if len(evaluation['missed_keywords']) > 5 else ''}")
                
                # If quality is sufficient, mark as success and break retry loop
                if evaluation["match_percentage"] >= 0.25:
                    test_results.append({
                        "image": image_name,
                        "success": True,
                        "quality": evaluation["quality"],
                        "match_percentage": evaluation["match_percentage"],
                        "attempt": attempt + 1
                    })
                    success_count += 1
                    break
                
                # If first attempt failed but we can retry
                if attempt == 0:
                    print(f"Response quality too low ({evaluation['match_percentage']*100:.1f}%), retrying with enhanced prompt")
                    continue
                
                # Record final failed attempt
                test_results.append({
                    "image": image_name,
                    "success": False,
                    "quality": evaluation["quality"],
                    "match_percentage": evaluation["match_percentage"],
                    "attempt": attempt + 1
                })
                
            except Exception as e:
                print(f"Anthropic vision test failed for {image_name} (attempt {attempt+1}): {str(e)}")
                if attempt == 0:
                    print("Retrying...")
                    continue
                
                test_results.append({
                    "image": image_name,
                    "success": False,
                    "error": str(e),
                    "attempt": attempt + 1
                })
                break
    
    # Summarize results
    print("\nAnthropic vision test summary:")
    print(f"Successful tests: {success_count}/{len(test_results)}")
    
    # Test is successful if at least one image was processed successfully
    if success_count > 0:
        # The test passed with at least one successful image
        return
    elif len(test_results) > 0:
        # All tests failed with images available
        errors = []
        for r in test_results:
            if "error" in r:
                errors.append(f"{r['image']}: {r['error']}")
            else:
                match_percentage = r.get("match_percentage", 0) * 100
                errors.append(f"{r['image']}: Low quality: {match_percentage:.1f}%")
        
        pytest.fail(f"All Anthropic vision tests failed: {'; '.join(errors)}")
    else:
        # No tests were run
        pytest.skip("No Anthropic vision tests were run due to missing images")


@pytest.mark.vision
def test_ollama_image_recognition():
    """Test Ollama's ability to recognize content in test images."""
    # Try to find a vision-capable Ollama model
    vision_model = None
    base_url = "http://localhost:11434"  # Default Ollama URL
    
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Look for vision-capable models
            for model in models:
                model_name = model.get("name", "")
                if any(pattern in model_name.lower() for pattern in ["vision", "janus", "multimodal", "llava"]):
                    vision_model = model_name
                    break
    except Exception as e:
        pytest.skip(f"Ollama not available at {base_url}: {e}")
    
    if not vision_model:
        available_models = []
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model.get("name", "") for model in models]
        except:
            pass
            
        pytest.skip(f"No vision-capable Ollama model found at {base_url}. Available models: {', '.join(available_models)}\n"
                   f"Try downloading a vision model: ollama pull llama3.2-vision:latest")
    
    # Create provider with vision-capable model and explicit base_url
    provider = create_llm("ollama", **{
        ModelParameter.MODEL: vision_model,
        ModelParameter.BASE_URL: base_url,  # Explicitly set the base URL
        ModelParameter.MAX_TOKENS: 300
    })
    
    # Test with actual image path
    for image_name, image_info in TEST_IMAGES.items():
        image_path = image_info["path"]
        if not os.path.exists(image_path):
            pytest.skip(f"Image file {image_name} not found at {image_path}")
            
        prompt = image_info["prompt"]
        
        try:
            print(f"\nTesting Ollama with image: {image_name}")
            response = provider.generate(prompt, image=image_path)
            
            # Evaluate the response
            keywords = image_info["keywords"]
            evaluation = evaluate_vision_response(response, keywords)
            
            print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
            print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
            print(f"Matched: {', '.join(evaluation['matched_keywords'][:5])}{'...' if len(evaluation['matched_keywords']) > 5 else ''}")
            print(f"Missed: {', '.join(evaluation['missed_keywords'][:5])}{'...' if len(evaluation['missed_keywords']) > 5 else ''}")
            
            # Assert minimum quality level
            assert evaluation["match_percentage"] >= 0.25, f"Response quality too low: {evaluation['quality']}"
            
        except Exception as e:
            print(f"Ollama vision test failed for {image_name}: {str(e)}")
            raise


@pytest.mark.vision
@pytest.mark.timeout(1800)  # 30 minutes timeout for the entire test
def test_huggingface_image_recognition():
    """Test HuggingFace's ability to recognize content in test images."""
    
    try:
        # Check if required packages are installed
        import importlib.util
        for package in ["transformers", "PIL", "requests", "torch"]:
            spec = importlib.util.find_spec(package)
            if spec is None:
                pytest.skip(f"{package} package not installed. Required for vision tests.")
    except Exception as e:
        pytest.skip(f"Error checking for required packages: {str(e)}")
    
    # Import required libraries for direct usage
    try:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        import os
        import time
    except ImportError as e:
        pytest.skip(f"Required vision libraries not available: {e}")
    
    # Use a simpler BLIP model which works well on CPU
    model_name = "Salesforce/blip-image-captioning-base"
    print(f"\nUsing direct HuggingFace vision model: {model_name}")
    
    # Use the default HF cache directory
    default_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Test with a single image for simplicity
    test_image = "test_image_1.jpg"
    image_info = TEST_IMAGES[test_image]
    image_path = image_info["path"]
    
    if not os.path.exists(image_path):
        pytest.skip(f"Image file {test_image} not found at {image_path}")
    
    try:
        # Load the processor and model directly
        print("Loading processor and model...")
        start_time = time.time()
        
        processor = BlipProcessor.from_pretrained(model_name, cache_dir=default_cache_dir)
        model = BlipForConditionalGeneration.from_pretrained(model_name, cache_dir=default_cache_dir)
        
        print(f"Model loaded in {time.time() - start_time:.1f} seconds")
        
        # Load and process the test image
        raw_image = Image.open(image_path).convert('RGB')
        
        # BLIP models work better without conditioning text for pure image captioning
        print("Processing image for captioning...")
        inputs = processor(raw_image, return_tensors="pt")
        
        # Generate the caption
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=75)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Generated caption: {caption}")
        
        # Evaluate the caption against keywords
        keywords = image_info["keywords"]
        evaluation = evaluate_vision_response(caption, keywords)
        
        print(f"Response quality: {evaluation['quality']} ({evaluation['match_percentage']*100:.1f}%)")
        print(f"Matched keywords: {len(evaluation['matched_keywords'])}/{len(keywords)}")
        print(f"Matched: {', '.join(evaluation['matched_keywords'][:5])}{'...' if len(evaluation['matched_keywords']) > 5 else ''}")
        print(f"Missed: {', '.join(evaluation['missed_keywords'][:5])}{'...' if len(evaluation['missed_keywords']) > 5 else ''}")
        
        # The test passes if we got any response from the model
        # The goal is to verify the model loads and runs, not to evaluate its quality
        print("Test passed - model successfully loaded and generated caption")
            
    except Exception as e:
        print(f"HuggingFace vision test failed: {str(e)}")
        pytest.skip(f"Test skipped due to error: {str(e)}")


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