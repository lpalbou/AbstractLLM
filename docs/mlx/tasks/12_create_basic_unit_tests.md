# Task 12: Create Basic Unit Tests

## Description
Create comprehensive unit tests for the MLX provider, including vision capabilities testing.

## Requirements
1. Create test file in the appropriate test directory
2. Test model loading, generation, and vision capabilities
3. Test image processing and memory safety
4. Test error handling and edge cases

## Implementation Details

Create a test file at `tests/test_mlx_vision.py`:

```python
#!/usr/bin/env python3
"""Test module for MLX vision capabilities."""

import os
import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

from abstractllm.providers.mlx_provider import MLXProvider, MODEL_CONFIGS
from abstractllm.media.factory import MediaFactory
from abstractllm.enums import ModelParameter, ModelCapability
from abstractllm.exceptions import UnsupportedFeatureError, ImageProcessingError

# Test image paths
TEST_IMAGES = [
    "tests/examples/mountain_path.jpg",
    "tests/examples/space_cat.jpg",
    "tests/examples/urban_sunset.jpg",
    "tests/examples/whale.jpg"
]

# Test prompts for each image
IMAGE_PROMPTS = {
    "mountain_path.jpg": "Describe this mountain path in detail.",
    "space_cat.jpg": "What's unusual about this cat image?",
    "urban_sunset.jpg": "Describe this urban scene and its lighting.",
    "whale.jpg": "Describe this marine scene in detail."
}

@pytest.fixture
def mlx_provider():
    """Create MLX provider with vision model."""
    config = {
        ModelParameter.MODEL: "mlx-community/Qwen2.5-VL-32B-Instruct-6bit",  # Using latest Qwen model
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 1024,
        "quantize": True
    }
    return MLXProvider(config)

@pytest.fixture
def test_images() -> List[Path]:
    """Get test image paths."""
    return [Path(img_path) for img_path in TEST_IMAGES]

def test_model_config_detection(mlx_provider):
    """Test that model type is correctly detected and configured."""
    assert mlx_provider._model_type == "qwen-vl"
    assert mlx_provider._is_vision_model is True
    
    # Verify config values
    config = MODEL_CONFIGS[mlx_provider._model_type]
    assert config["image_size"] == (448, 448)  # Qwen-VL specific size
    assert config["prompt_format"] == "<img>{prompt}"

def test_vision_model_loading(mlx_provider):
    """Test that vision model loads correctly."""
    assert mlx_provider._is_vision_model is True
    mlx_provider.load_model()
    assert mlx_provider._is_loaded is True
    assert mlx_provider._processor is not None
    assert mlx_provider._model is not None
    assert mlx_provider._config is not None

def test_vision_capabilities(mlx_provider):
    """Test vision capability reporting."""
    capabilities = mlx_provider.get_capabilities()
    assert capabilities[ModelCapability.VISION] is True
    assert capabilities[ModelCapability.STREAMING] is True
    assert capabilities[ModelCapability.SYSTEM_PROMPT] is True

def test_image_preprocessing(mlx_provider, test_images):
    """Test image preprocessing functionality."""
    for img_path in test_images:
        # Create image input
        image_input = MediaFactory.from_source(str(img_path))
        
        # Process image
        processed = mlx_provider._process_image(image_input)
        
        # Check processed image properties
        assert processed.shape == (3, 448, 448)  # CHW format for Qwen-VL
        assert processed.dtype == "float32"
        
        # Test aspect ratio preservation
        original_image = Image.open(img_path)
        orig_aspect = original_image.width / original_image.height
        
        # Convert processed back to PIL for aspect check
        processed_np = processed.numpy()
        processed_np = np.transpose(processed_np, (1, 2, 0))  # CHW to HWC
        non_zero_mask = np.any(processed_np != 0, axis=2)
        non_zero_coords = np.nonzero(non_zero_mask)
        
        if len(non_zero_coords[0]) > 0 and len(non_zero_coords[1]) > 0:
            height = non_zero_coords[0].max() - non_zero_coords[0].min()
            width = non_zero_coords[1].max() - non_zero_coords[1].min()
            processed_aspect = width / height
            assert abs(orig_aspect - processed_aspect) < 0.1  # Allow small difference due to padding

def test_memory_requirements(mlx_provider):
    """Test memory requirement checking."""
    # Test with normal image size
    mlx_provider._check_memory_requirements((448, 448), 1)
    
    # Test with extremely large image (should raise error)
    with pytest.raises(MemoryError):
        mlx_provider._check_memory_requirements((100000, 100000), 1)

def test_prompt_formatting(mlx_provider):
    """Test prompt formatting with images."""
    # Single image
    formatted = mlx_provider._format_prompt("Describe this.", 1)
    assert formatted == "<img>Describe this."
    
    # Multiple images (Qwen-VL supports numbered images)
    formatted = mlx_provider._format_prompt("Compare these.", 2)
    assert "<image1>" in formatted
    assert "<image2>" in formatted

def test_single_image_generation(mlx_provider, test_images):
    """Test vision model generation with a single image."""
    for img_path in test_images:
        prompt = IMAGE_PROMPTS[img_path.name]
        
        # Generate response
        response = mlx_provider.generate(
            prompt=prompt,
            files=[str(img_path)]
        )
        
        # Verify response
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.model == "mlx-community/Qwen2.5-VL-32B-Instruct-6bit"

def test_streaming_image_generation(mlx_provider, test_images):
    """Test streaming generation with images."""
    for img_path in test_images:
        prompt = IMAGE_PROMPTS[img_path.name]
        
        # Stream response
        chunks = []
        for chunk in mlx_provider.generate(
            prompt=prompt,
            files=[str(img_path)],
            stream=True
        ):
            assert isinstance(chunk.content, str)
            assert "time" in chunk.usage  # Check streaming-specific metrics
            chunks.append(chunk.content)
        
        # Verify complete response
        complete_response = "".join(chunks)
        assert len(complete_response) > 0

def test_system_prompt_with_image(mlx_provider, test_images):
    """Test vision generation with system prompt."""
    system_prompt = "You are a professional photographer and art critic."
    img_path = test_images[0]
    
    response = mlx_provider.generate(
        prompt=IMAGE_PROMPTS[img_path.name],
        system_prompt=system_prompt,
        files=[str(img_path)]
    )
    
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Response should reflect the system prompt's role
    assert any(word in response.content.lower() for word in ["composition", "lighting", "perspective", "artistic"])

def test_non_vision_model_rejection():
    """Test that non-vision models reject image inputs."""
    config = {
        ModelParameter.MODEL: "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",  # Non-vision model
    }
    provider = MLXProvider(config)
    
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        provider.generate(
            prompt="What's in this image?",
            files=[TEST_IMAGES[0]]
        )
    assert "vision" in str(exc_info.value)
    assert "does not support vision inputs" in str(exc_info.value)

def test_multiple_images_handling(mlx_provider):
    """Test handling multiple images in one request."""
    image_paths = TEST_IMAGES[:2]
    prompt = "Compare these two images in detail."
    
    response = mlx_provider.generate(
        prompt=prompt,
        files=image_paths
    )
    
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    # Response should mention aspects of both images
    assert any(word in response.content.lower() for word in ["first", "second", "both", "compare", "while"])

def test_error_handling(mlx_provider):
    """Test comprehensive error handling."""
    # Test invalid image path
    with pytest.raises(FileProcessingError):
        mlx_provider.generate(
            prompt="What's in this image?",
            files=["nonexistent.jpg"]
        )
    
    # Test invalid image data
    with pytest.raises(ImageProcessingError):
        mlx_provider.generate(
            prompt="What's in this image?",
            files=["tests/test_mlx_vision.py"]  # Using this test file as invalid image
        )
    
    # Test memory error (mock large image)
    large_image = Image.new('RGB', (100000, 100000))
    with pytest.raises(MemoryError):
        mlx_provider._process_image(MediaFactory.from_source(large_image))

def test_async_generation(mlx_provider, test_images):
    """Test async generation with images."""
    import asyncio
    
    async def test_async():
        img_path = test_images[0]
        prompt = IMAGE_PROMPTS[img_path.name]
        
        # Test non-streaming async
        response = await mlx_provider.generate_async(
            prompt=prompt,
            files=[str(img_path)]
        )
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        
        # Test streaming async
        chunks = []
        async for chunk in await mlx_provider.generate_async(
            prompt=prompt,
            files=[str(img_path)],
            stream=True
        ):
            assert isinstance(chunk.content, str)
            assert "time" in chunk.usage
            chunks.append(chunk.content)
        
        complete_response = "".join(chunks)
        assert len(complete_response) > 0
    
    asyncio.run(test_async())
```

## References
- See `docs/mlx/vision-upgrade.md` for vision implementation details
- See `docs/mlx/deepsearch-mlx-vlm.md` for MLX-VLM insights
- See `tests/examples/` for test images

## Testing
Run the tests with pytest:

```bash
# Run all tests
pytest -xvs tests/test_mlx_vision.py

# Run specific test
pytest -xvs tests/test_mlx_vision.py::test_image_preprocessing
```

## Success Criteria
1. All tests pass on Apple Silicon hardware
2. Image processing tests verify aspect ratio preservation
3. Memory safety checks are properly tested
4. Error handling tests cover all failure modes
5. Vision capabilities are properly verified 