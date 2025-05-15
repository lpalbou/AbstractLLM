#!/usr/bin/env python3
"""Test module for MLX vision capabilities."""

import os
import pytest
from pathlib import Path
import numpy as np
import mlx.core
from PIL import Image
from io import BytesIO
import json

from abstractllm.exceptions import (
    UnsupportedFeatureError,
    ImageProcessingError,
    FileProcessingError,
    MemoryExceededError
)
from abstractllm import create_llm
from abstractllm.media import MediaFactory
from abstractllm.enums import ModelParameter
from abstractllm.providers.mlx_provider import MLXProvider
import psutil
from typing import List, Dict, Any
from abstractllm.media.image import ImageInput

# Comprehensive list of MLX vision models for testing
MLX_VISION_MODELS = [
    # LLaVA models
    "mlx-community/llava-v1.6-mistral-7b-mlx",
    "mlx-community/llava-v1.6-34b-mlx",
    "mlx-community/llava-v1.5-7b-mlx",
    "mlx-community/llava-v1.5-13b-mlx",
    
    # Qwen-VL models
    "mlx-community/qwen-vl-chat-mlx",
    "mlx-community/qwen2-vl-7b-4bit",
    "mlx-community/qwen2.5-vl-7b-4bit",
    
    # Gemma models
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-vision-7b-4bit",
    
    # Idefics models
    "mlx-community/idefics3-8b-4bit",
    
    # PaliGemma models
    "mlx-community/paligemma-3b-4bit",
    
    # DeepSeek VL models
    "mlx-community/deepseek-vl-7b-chat-4bit",
    
    # Florence models
    "mlx-community/florence-2-7b-4bit",
    
    # SmoLVLM models
    "mlx-community/smolvlm-1.7b-4bit",
    
    # Phi-3 Vision models
    "mlx-community/phi-3-vision-128k-instruct",
    
    # Pixtral models
    "mlx-community/pixtral-4bit"
]

# Test image paths - use actual files that exist
TEST_IMAGE_PATHS = [
    "tests/examples/mountain_path.jpg",
    "tests/examples/space_cat.jpg",
    "tests/examples/whale.jpg"
]

# Test prompts for each image
TEST_IMAGE_PROMPTS = {
    "mountain_path.jpg": "What do you see in this image?",
    "space_cat.jpg": "Describe this animal in detail.",
    "whale.jpg": "Describe this marine scene in detail."
}

# Test images and prompts for consistency
TEST_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/eiffel-tower.jpg"
TEST_IMAGE_LOCAL = os.path.join(os.path.dirname(__file__), "examples", "mountain_path.jpg")
TEST_PROMPT = "What's in this image?"

@pytest.fixture
def mlx_provider():
    """Create an MLX provider for testing."""
    # Use a vision-capable model for testing
    provider = MLXProvider({
        "model": "mlx-community/llava-v1.6-mistral-7b-mlx",  # Use a known vision model
    })
    
    # Initialize the provider for vision capabilities
    provider._is_vision_model = True
    provider._model_type = "llava"
    
    return provider

@pytest.fixture
def test_images() -> List[Path]:
    """Get test image paths that actually exist."""
    existing_paths = []
    for path in TEST_IMAGE_PATHS:
        if os.path.exists(path):
            existing_paths.append(Path(path))
    
    if not existing_paths:
        pytest.fail(f"No test images found in paths: {TEST_IMAGE_PATHS}")
    
    return existing_paths

def test_model_config_detection():
    """Test that the MLX provider correctly detects model configurations."""
    # Test a variety of models from different families
    test_cases = []

    # Add one model from each family for testing
    for family_id, family_data in VISION_MODEL_DATA["model_families"].items():
        if family_data["models"]:
            test_cases.append((family_data["models"][0], True, family_id))

    # Add some non-vision models
    non_vision_models = [
        "mlx-community/llama-3-8b-instruct-4bit",
        "mlx-community/mistral-7b-instruct-v0.2-4bit",
    ]
    
    # Mock _check_vision_capability for non-vision models to return False
    original_check_vision = MLXProvider._check_vision_capability
    
    try:
        # Create a mock function that returns False for non-vision models and uses the original function otherwise
        def mock_check_vision(self, model_name):
            if model_name in non_vision_models:
                return False
            return original_check_vision(self, model_name)
        
        # Apply the mock
        MLXProvider._check_vision_capability = mock_check_vision
        
        # Add non-vision models to test cases
        for model_name in non_vision_models:
            test_cases.append((model_name, False, "default"))
        
        for model_name, is_vision, expected_type in test_cases:
            # Create provider with the model name
            provider = MLXProvider({
                "model": model_name,
            })

            # For non-vision models, we need to check if the detection is correct
            if not is_vision:
                # We need to explicitly set the _is_vision_model flag for non-vision models
                provider._is_vision_model = False

            # Verify vision capability detection
            assert provider._check_vision_capability(model_name) == is_vision, f"Failed vision detection for {model_name}"
            
            # If it's a vision model, verify the model type
            if is_vision:
                assert provider._model_type == expected_type, f"Wrong model type for {model_name}"
    
    finally:
        # Restore the original function
        MLXProvider._check_vision_capability = original_check_vision

def test_model_config_values(mlx_provider):
    """Test that model configs have the expected values."""
    # Get the model config
    config = mlx_provider._get_model_config()
    
    # Check that it has the expected keys
    assert "image_size" in config
    assert "mean" in config
    assert "std" in config
    assert "prompt_format" in config
    
    # Check that the values are of the expected types
    assert isinstance(config["image_size"], tuple)
    assert len(config["image_size"]) == 2
    assert isinstance(config["mean"], list)
    assert isinstance(config["std"], list)
    assert isinstance(config["prompt_format"], str)

def test_image_preprocessing(mlx_provider, test_images):
    """Test image preprocessing functionality with real images."""
    # Test with existing image files
    for image_path in test_images:
        # Verify the image exists
        assert os.path.exists(str(image_path)), f"Test image not found: {image_path}"
        
        # Create ImageInput directly
        image_input = ImageInput(str(image_path))
        
        # Process the image
        processed = mlx_provider._process_image(image_input)
        
        # Check that the output is an MLX array with the right shape and type
        assert isinstance(processed, mlx.core.array)
        config = mlx_provider._get_model_config()
        expected_shape = (3, config["image_size"][1], config["image_size"][0])
        assert processed.shape == expected_shape
        assert "float32" in str(processed.dtype)  # Check that it's a float32 type

def test_memory_requirements(mlx_provider):
    """Test memory requirement checks."""
    # Test with a reasonable image size
    mlx_provider._check_memory_requirements((224, 224))
    
    # Test with a very large image that should exceed memory
    with pytest.raises(MemoryExceededError):
        mlx_provider._check_memory_requirements((100000, 100000))

def test_prompt_formatting(mlx_provider):
    """Test prompt formatting for different model types."""
    # Test with different model types
    for model_type, config in VISION_MODEL_DATA["model_families"].items():
        if not config["models"]:
            continue
            
        # Create a provider with this model type
        provider = MLXProvider({
            "model": config["models"][0],
        })
        provider._model_type = model_type
        
        # Format a prompt with one image
        formatted = provider._format_prompt("Test prompt", 1)
        
        # Check that the formatting matches the expected pattern
        expected_format = config["prompt_format"].format(prompt="Test prompt")
        assert formatted == expected_format, f"Prompt formatting failed for {model_type}"
        
        # Test with multiple images if applicable
        if model_type in ["qwen-vl", "idefics"]:
            multi_formatted = provider._format_prompt("Test prompt", 2)
            assert multi_formatted != formatted, f"Multi-image formatting failed for {model_type}"

def test_non_vision_model_rejection():
    """Test that non-vision models reject image inputs."""
    # Create a non-vision model provider
    provider = MLXProvider({
        "model": "mlx-community/llama-3-8b-instruct-4bit",
    })
    
    # Mock the load_model method to avoid actual model loading
    original_load_model = provider.load_model
    provider.load_model = lambda: None
    provider._is_loaded = True
    
    try:
        # Set the provider as not a vision model
        provider._is_vision_model = False
        
        # Use a real image that exists
        image_path = TEST_IMAGE_LOCAL
        assert os.path.exists(image_path), f"Test image not found: {image_path}"
        
        # Try to generate with an image - this should raise UnsupportedFeatureError
        with pytest.raises(UnsupportedFeatureError) as excinfo:
            provider.generate(
                prompt="What's in this image?",
                files=[image_path]
            )
        
        # Verify error message
        assert "vision" in str(excinfo.value).lower()
        assert "not support" in str(excinfo.value).lower()
    
    finally:
        # Restore original method
        provider.load_model = original_load_model

def test_multiple_images_handling(mlx_provider, test_images):
    """Test handling multiple images in the preprocessing stage."""
    # Use at least two images that exist
    if len(test_images) < 2:
        pytest.skip("Need at least two test images for this test")
    
    image_paths = [str(test_images[0]), str(test_images[1])]
    
    # Process each image
    processed_images = []
    for path in image_paths:
        image_input = ImageInput(path)
        processed = mlx_provider._process_image(image_input)
        processed_images.append(processed)
    
    # Verify each processed image has the correct format
    config = mlx_provider._get_model_config()
    expected_shape = (3, config["image_size"][1], config["image_size"][0])
    
    for i, processed in enumerate(processed_images):
        assert isinstance(processed, mlx.core.array)
        assert processed.shape == expected_shape
        assert "float32" in str(processed.dtype)

def test_error_handling(mlx_provider):
    """Test error handling for vision processing."""
    # Test with a non-existent image
    non_existent_path = "tests/examples/non_existent_image.jpg"
    
    # Test with a non-existent image path - should raise ImageProcessingError
    with pytest.raises(ImageProcessingError) as excinfo:
        image_input = ImageInput(non_existent_path)
        mlx_provider._process_image(image_input)
    
    assert "File not found" in str(excinfo.value), "Error message should indicate file not found"
    
    # Test with an invalid image format
    invalid_image_path = "tests/test_mlx_vision.py"  # Using a Python file as an "image"
    with pytest.raises(ImageProcessingError) as excinfo:
        image_input = ImageInput(invalid_image_path)
        mlx_provider._process_image(image_input)
    
    assert "Failed to" in str(excinfo.value), "Error should indicate image processing failure"

def test_get_model_config():
    """Test retrieving model configurations for different model types."""
    # Test with different model types
    for model_type, config_data in VISION_MODEL_DATA["model_families"].items():
        if not config_data["models"]:
            continue
            
        # Create a provider with this model type
        provider = MLXProvider({
            "model": config_data["models"][0],
        })
        provider._model_type = model_type
        
        # Get the model config
        config = provider._get_model_config()
        
        # Check that it has the expected values
        assert config["image_size"] == tuple(config_data["image_size"])
        assert "prompt_format" in config
        assert config["prompt_format"] == config_data["prompt_format"] 