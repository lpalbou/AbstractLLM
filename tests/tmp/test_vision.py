"""Tests for vision capabilities of various providers."""

import os
import pytest
import requests
import logging
from abstractllm import create_llm, ModelCapability
from abstractllm.media import MediaFactory

# Configure logger
logger = logging.getLogger(__name__)

# Test image paths and expected keywords
TEST_IMAGES = {
    "mountain_path": {
        "path": "tests/examples/mountain_path.jpg",
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/mountain_path.jpg",
        "keywords": [
            "mountains", "path", "fence", "sunlight", "sky", "clouds", "hiking", 
            "trail", "wooden fence", "meadow", "landscape", "nature", "dirt road", 
            "mountain range", "sunny day"
        ]
    },
    "urban_sunset": {
        "path": "tests/examples/urban_sunset.jpg",
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/urban_sunset.jpg",
        "keywords": [
            "sunset", "street lamps", "pathway", "trees", "buildings", "urban", 
            "city", "dusk", "pink sky", "lampposts", "sidewalk", "park", 
            "evening", "autumn", "architecture"
        ]
    },
    "whale": {
        "path": "tests/examples/whale.jpg",
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/whale.jpg",
        "keywords": [
            "whale", "humpback", "ocean", "water", "splash", "marine", "wildlife",
            "breaching", "sea", "waves", "mammal", "nature", "aquatic", 
            "motion", "dramatic"
        ]
    },
    "space_cat": {
        "path": "tests/examples/space_cat.jpg",
        "url": "https://raw.githubusercontent.com/lpalbou/abstractllm/refs/heads/main/tests/examples/space_cat.jpg",
        "keywords": [
            "cat", "dome", "spaceship", "astronaut", "helmet", "white", "funny",
            "pet", "feline", "porthole", "viewing window", "whimsical", 
            "curious", "space theme", "futuristic"
        ]
    }
}

def ensure_test_images_exist():
    """Ensure all test images exist and are accessible."""
    # Check local images
    for image_info in TEST_IMAGES.values():
        if not os.path.exists(image_info["path"]):
            pytest.skip(f"Test image not found: {image_info['path']}")
            
    # Check remote images with proper headers
    headers = {
        "User-Agent": "AbstractLLM/1.0 (https://github.com/lpalbou/abstractllm)"
    }
    for image_info in TEST_IMAGES.values():
        try:
            response = requests.head(image_info["url"], headers=headers)
            response.raise_for_status()
        except (requests.RequestException, Exception) as e:
            pytest.skip(f"Remote test image not accessible: {image_info['url']} - {str(e)}")

def verify_keyword_match(extracted_keywords: str, expected_keywords: list, min_matches: int = 2) -> bool:
    """Verify that enough expected keywords are found in the extracted keywords."""
    extracted_lower = extracted_keywords.lower()
    matches = sum(1 for keyword in expected_keywords if keyword.lower() in extracted_lower)
    return matches >= min_matches

def test_openai_vision():
    """Test OpenAI's vision capabilities."""
    ensure_test_images_exist()
    llm = create_llm("openai", capabilities=[ModelCapability.VISION])
    
    # Test local images
    for image_name, image_info in TEST_IMAGES.items():
        image = MediaFactory.from_source(image_info["path"])
        response = llm.generate("Extract keywords describing this image:", image=image)
        assert verify_keyword_match(response, image_info["keywords"])
        
    # Test remote images
    for image_name, image_info in TEST_IMAGES.items():
        image = MediaFactory.from_source(image_info["url"])
        response = llm.generate("Extract keywords describing this image:", image=image)
        assert verify_keyword_match(response, image_info["keywords"])

def test_anthropic_vision():
    """Test Anthropic's vision capabilities."""
    ensure_test_images_exist()
    llm = create_llm("anthropic", capabilities=[ModelCapability.VISION])
    
    # Test local images
    for image_name, image_info in TEST_IMAGES.items():
        image = MediaFactory.from_source(image_info["path"])
        response = llm.generate("Extract keywords describing this image:", image=image)
        assert verify_keyword_match(response, image_info["keywords"])
        
    # Test remote images
    for image_name, image_info in TEST_IMAGES.items():
        image = MediaFactory.from_source(image_info["url"])
        response = llm.generate("Extract keywords describing this image:", image=image)
        assert verify_keyword_match(response, image_info["keywords"])

def test_ollama_vision():
    """Test Ollama's vision capabilities."""
    ensure_test_images_exist()
    llm = create_llm("ollama", capabilities=[ModelCapability.VISION])
    
    # Test local images
    for image_name, image_info in TEST_IMAGES.items():
        image = MediaFactory.from_source(image_info["path"])
        response = llm.generate("Extract keywords describing this image:", image=image)
        assert verify_keyword_match(response, image_info["keywords"])
        
    # Test remote images
    for image_name, image_info in TEST_IMAGES.items():
        image = MediaFactory.from_source(image_info["url"])
        response = llm.generate("Extract keywords describing this image:", image=image)
        assert verify_keyword_match(response, image_info["keywords"])

def test_huggingface_vision():
    """Test HuggingFace's vision capabilities."""
    ensure_test_images_exist()
    llm = create_llm("huggingface", capabilities=[ModelCapability.VISION])
    
    # Test with first image only (most HF models only support one image)
    image_info = next(iter(TEST_IMAGES.values()))
    
    # Test with local image
    image = MediaFactory.from_source(image_info["path"])
    response = llm.generate("Extract keywords describing this image:", image=image)
    assert verify_keyword_match(response, image_info["keywords"])
    
    # Test with remote image
    image = MediaFactory.from_source(image_info["url"])
    response = llm.generate("Extract keywords describing this image:", image=image)
    assert verify_keyword_match(response, image_info["keywords"])
    
    # Test with multiple images (should raise warning)
    with pytest.warns(UserWarning, match="Most HuggingFace models only support one image"):
        images = [MediaFactory.from_source(info["path"]) for info in list(TEST_IMAGES.values())[:2]]
        response = llm.generate("Extract keywords describing these images:", images=images)
        # Should still get valid response for the first image
        assert verify_keyword_match(response, list(TEST_IMAGES.values())[0]["keywords"])

def test_multiple_providers():
    """Test using multiple providers with the same images."""
    ensure_test_images_exist()
    
    providers = ["openai", "anthropic", "ollama", "huggingface"]
    image_info = next(iter(TEST_IMAGES.values()))  # Use first image for consistency
    
    # Test with local image
    image = MediaFactory.from_source(image_info["path"])
    
    for provider in providers:
        llm = create_llm(provider, capabilities=[ModelCapability.VISION])
        try:
            response = llm.generate("Extract keywords describing this image:", image=image)
            assert verify_keyword_match(response, image_info["keywords"])
        except Exception as e:
            logger.warning(f"Provider {provider} failed with error: {str(e)}")
            continue
    
    # Test with remote image
    image = MediaFactory.from_source(image_info["url"])
    
    for provider in providers:
        llm = create_llm(provider, capabilities=[ModelCapability.VISION])
        try:
            response = llm.generate("Extract keywords describing this image:", image=image)
            assert verify_keyword_match(response, image_info["keywords"])
        except Exception as e:
            logger.warning(f"Provider {provider} failed with error: {str(e)}")
            continue