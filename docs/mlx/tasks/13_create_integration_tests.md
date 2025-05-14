# Task 13: Create Integration Tests

## Description
Create comprehensive integration tests for the MLX provider, focusing on end-to-end functionality including vision capabilities, performance, and error handling.

## Requirements
1. Test end-to-end functionality with real MLX models
2. Test vision model integration
3. Test media factory integration
4. Test error handling and recovery
5. Test performance and memory usage
6. Test caching utilities
7. Test CLI tools

## Implementation Details

### 1. Vision Integration Tests

Create test file at `tests/integration/test_mlx_vision_integration.py`:

```python
#!/usr/bin/env python3
"""Integration tests for MLX vision capabilities."""

import os
import pytest
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import psutil
import asyncio
from typing import List, Dict, Any

from abstractllm import create_llm
from abstractllm.enums import ModelParameter, ModelCapability
from abstractllm.media.factory import MediaFactory
from abstractllm.exceptions import (
    UnsupportedFeatureError,
    ImageProcessingError,
    MemoryError,
    ModelLoadError
)

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

# Test models
VISION_MODELS = [
    "mlx-community/Qwen2.5-VL-32B-Instruct-6bit",
    "mlx-community/llava-v1.6-34b-mlx",
    "mlx-community/kimi-vl-70b-instruct-mlx"
]

@pytest.fixture(scope="module")
def test_images() -> List[Path]:
    """Get test image paths."""
    return [Path(img_path) for img_path in TEST_IMAGES]

@pytest.fixture(params=VISION_MODELS)
def vision_model(request):
    """Create vision model for testing."""
    config = {
        ModelParameter.MODEL: request.param,
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 1024,
        "quantize": True
    }
    return create_llm("mlx", **config)

def test_model_loading_memory(vision_model):
    """Test model loading and memory usage."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    vision_model.load_model()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Check memory increase is reasonable (less than 16GB)
    assert memory_increase < 16 * 1024 * 1024 * 1024
    
    # Verify model is loaded
    assert vision_model._is_loaded
    assert vision_model._model is not None
    assert vision_model._processor is not None

def test_media_factory_integration(vision_model, test_images):
    """Test integration with media factory."""
    for img_path in test_images:
        # Test different input formats
        # 1. Path string
        media1 = MediaFactory.from_source(str(img_path))
        assert media1 is not None
        
        # 2. Path object
        media2 = MediaFactory.from_source(img_path)
        assert media2 is not None
        
        # 3. PIL Image
        with Image.open(img_path) as img:
            media3 = MediaFactory.from_source(img)
            assert media3 is not None
        
        # 4. Numpy array
        img_array = np.array(Image.open(img_path))
        media4 = MediaFactory.from_source(img_array)
        assert media4 is not None
        
        # Test processing through vision model
        processed = vision_model._process_image(media1)
        assert processed is not None
        assert processed.shape[-2:] == vision_model._get_model_config()["image_size"]

def test_end_to_end_vision(vision_model, test_images):
    """Test end-to-end vision capabilities."""
    for img_path in test_images:
        prompt = IMAGE_PROMPTS[img_path.name]
        
        # Test non-streaming generation
        response = vision_model.generate(
            prompt=prompt,
            files=[str(img_path)]
        )
        
        assert response.content is not None
        assert len(response.content) > 0
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        
        # Test streaming generation
        chunks = []
        for chunk in vision_model.generate(
            prompt=prompt,
            files=[str(img_path)],
            stream=True
        ):
            assert chunk.content is not None
            chunks.append(chunk.content)
        
        complete_response = "".join(chunks)
        assert len(complete_response) > 0

def test_multiple_images_end_to_end(vision_model):
    """Test end-to-end with multiple images."""
    image_paths = TEST_IMAGES[:2]
    prompt = "Compare these two images in detail."
    
    # Test with different combinations of image inputs
    responses = []
    
    # 1. Path strings
    response1 = vision_model.generate(
        prompt=prompt,
        files=image_paths
    )
    responses.append(response1.content)
    
    # 2. Mixed Path and PIL Image
    with Image.open(image_paths[0]) as img:
        response2 = vision_model.generate(
            prompt=prompt,
            files=[img, image_paths[1]]
        )
        responses.append(response2.content)
    
    # Verify responses are meaningful and different
    assert all(len(r) > 0 for r in responses)
    assert responses[0] != responses[1]  # Responses should vary

def test_error_recovery(vision_model):
    """Test error handling and recovery."""
    # 1. Test recovery from invalid image
    with pytest.raises(ImageProcessingError):
        vision_model.generate(
            prompt="What's in this image?",
            files=["invalid.jpg"]
        )
    
    # Should still work with valid image after error
    response = vision_model.generate(
        prompt=IMAGE_PROMPTS["mountain_path.jpg"],
        files=[TEST_IMAGES[0]]
    )
    assert response.content is not None
    
    # 2. Test recovery from memory error
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
        # Create large image that should trigger memory error
        large_image = Image.new('RGB', (10000, 10000))
        large_image.save(tmp.name)
        
        with pytest.raises(MemoryError):
            vision_model.generate(
                prompt="What's in this image?",
                files=[tmp.name]
            )
    
    # Should still work with normal image after memory error
    response = vision_model.generate(
        prompt=IMAGE_PROMPTS["mountain_path.jpg"],
        files=[TEST_IMAGES[0]]
    )
    assert response.content is not None

def test_concurrent_processing(vision_model, test_images):
    """Test concurrent image processing."""
    async def process_image(img_path):
        prompt = IMAGE_PROMPTS[img_path.name]
        return await vision_model.generate_async(
            prompt=prompt,
            files=[str(img_path)]
        )
    
    async def run_concurrent():
        tasks = [process_image(img_path) for img_path in test_images]
        responses = await asyncio.gather(*tasks)
        return responses
    
    responses = asyncio.run(run_concurrent())
    
    assert len(responses) == len(test_images)
    assert all(r.content is not None for r in responses)
    assert all(len(r.content) > 0 for r in responses)

def test_performance_metrics(vision_model, test_images):
    """Test performance metrics collection."""
    img_path = test_images[0]
    prompt = IMAGE_PROMPTS[img_path.name]
    
    # Test streaming metrics
    chunks = list(vision_model.generate(
        prompt=prompt,
        files=[str(img_path)],
        stream=True
    ))
    
    # Verify timing metrics
    assert all("time" in chunk.usage for chunk in chunks)
    times = [chunk.usage["time"] for chunk in chunks]
    assert all(isinstance(t, (int, float)) for t in times)
    assert all(t >= 0 for t in times)
    
    # Test token metrics
    response = vision_model.generate(
        prompt=prompt,
        files=[str(img_path)]
    )
    assert "prompt_tokens" in response.usage
    assert "completion_tokens" in response.usage
    assert response.usage["prompt_tokens"] > 0
    assert response.usage["completion_tokens"] > 0

def test_model_specific_features(vision_model):
    """Test model-specific features and configurations."""
    model_type = vision_model._model_type
    config = vision_model._get_model_config()
    
    # Test image size requirements
    img_size = config["image_size"]
    with Image.new('RGB', img_size) as test_img:
        response = vision_model.generate(
            prompt="Describe this test image.",
            files=[test_img]
        )
        assert response.content is not None
    
    # Test prompt format
    if "prompt_format" in config:
        formatted = vision_model._format_prompt("Test prompt", 1)
        assert formatted.startswith(config["prompt_format"].replace("{prompt}", ""))

def test_system_prompt_integration(vision_model, test_images):
    """Test system prompt integration with vision."""
    system_prompts = [
        "You are a professional photographer analyzing images.",
        "You are an art critic evaluating visual compositions.",
        "You are a nature expert identifying flora and fauna."
    ]
    
    img_path = test_images[0]
    prompt = IMAGE_PROMPTS[img_path.name]
    
    responses = []
    for system_prompt in system_prompts:
        response = vision_model.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            files=[str(img_path)]
        )
        responses.append(response.content)
    
    # Verify responses reflect different system prompts
    assert len(set(responses)) == len(system_prompts)  # Responses should be different
    assert all(len(r) > 0 for r in responses)

def test_media_preprocessing_pipeline(vision_model, test_images):
    """Test the complete media preprocessing pipeline."""
    img_path = test_images[0]
    
    # Test with different preprocessing options
    with Image.open(img_path) as img:
        # 1. Original size
        media1 = MediaFactory.from_source(img)
        processed1 = vision_model._process_image(media1)
        
        # 2. Resized image
        resized = img.resize((224, 224))
        media2 = MediaFactory.from_source(resized)
        processed2 = vision_model._process_image(media2)
        
        # 3. Grayscale image
        gray = img.convert('L')
        media3 = MediaFactory.from_source(gray)
        processed3 = vision_model._process_image(media3)
        
        # All should be processed to correct format
        assert all(p.shape[0] == 3 for p in [processed1, processed2, processed3])
        assert all(p.shape[1:] == vision_model._get_model_config()["image_size"] 
                  for p in [processed1, processed2, processed3])

def test_resource_cleanup(vision_model, test_images):
    """Test resource cleanup after processing."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Process multiple images
    for img_path in test_images:
        response = vision_model.generate(
            prompt=IMAGE_PROMPTS[img_path.name],
            files=[str(img_path)]
        )
        assert response.content is not None
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Check memory usage hasn't grown significantly
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Should not have a large memory leak (allow for some overhead)
    assert memory_increase < 1 * 1024 * 1024 * 1024  # 1GB limit

### 2. Performance Tests

Create test file at `tests/integration/test_mlx_performance.py`:

```python
"""Integration tests for MLX provider performance."""

import pytest
import time
import psutil
import gc
from pathlib import Path

from abstractllm import create_llm
from abstractllm.enums import ModelParameter

def test_model_loading_performance():
    """Test model loading performance and memory usage."""
    # Record initial state
    initial_memory = psutil.Process().memory_info().rss
    start_time = time.time()
    
    # Create and load model
    llm = create_llm(
        "mlx",
        model="mlx-community/phi-2"  # Small model for testing
    )
    llm.load_model()
    
    # Check metrics
    load_time = time.time() - start_time
    memory_used = psutil.Process().memory_info().rss - initial_memory
    
    print(f"Load time: {load_time:.2f}s")
    print(f"Memory used: {memory_used / 1024**2:.1f}MB")
    
    # Basic assertions
    assert load_time < 60  # Should load within 60 seconds
    assert memory_used < 8 * 1024**3  # Should use less than 8GB

def test_caching_performance():
    """Test model caching performance."""
    llm = create_llm("mlx")
    
    # First generation (cold start)
    start_time = time.time()
    response1 = llm.generate("Test prompt")
    cold_time = time.time() - start_time
    
    # Second generation (warm start)
    start_time = time.time()
    response2 = llm.generate("Test prompt")
    warm_time = time.time() - start_time
    
    # Warm start should be faster
    assert warm_time < cold_time
    print(f"Cold start: {cold_time:.2f}s")
    print(f"Warm start: {warm_time:.2f}s")

def test_memory_cleanup():
    """Test memory cleanup after processing."""
    llm = create_llm("mlx")
    
    # Record initial memory
    initial_memory = psutil.Process().memory_info().rss
    
    # Generate multiple responses
    for _ in range(5):
        response = llm.generate("Test prompt")
        gc.collect()  # Force garbage collection
    
    # Check final memory
    final_memory = psutil.Process().memory_info().rss
    memory_diff = final_memory - initial_memory
    
    # Should not have significant memory growth
    assert memory_diff < 1 * 1024**3  # Less than 1GB growth
    print(f"Memory growth: {memory_diff / 1024**2:.1f}MB")

def test_concurrent_performance():
    """Test performance with concurrent operations."""
    import asyncio
    
    async def run_concurrent(num_requests: int):
        llm = create_llm("mlx")
        start_time = time.time()
        
        # Create multiple requests
        tasks = [
            llm.generate_async("Test prompt")
            for _ in range(num_requests)
        ]
        
        # Run concurrently
        responses = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        return total_time
    
    # Test with different numbers of concurrent requests
    for num_requests in [2, 4, 8]:
        total_time = asyncio.run(run_concurrent(num_requests))
        print(f"{num_requests} concurrent requests: {total_time:.2f}s")

### 3. Caching Tests

Create test file at `tests/integration/test_mlx_caching.py`:

```python
"""Integration tests for MLX provider caching."""

import pytest
import os
from pathlib import Path

from abstractllm import create_llm
from abstractllm.providers.mlx_provider import MLXProvider

def test_model_caching():
    """Test model caching functionality."""
    # List initial cached models
    initial_models = MLXProvider.list_cached_models()
    
    # Create and use a model
    llm = create_llm(
        "mlx",
        model="mlx-community/phi-2"
    )
    llm.generate("Test prompt")
    
    # Check cache
    cached_models = MLXProvider.list_cached_models()
    assert len(cached_models) > len(initial_models)
    
    # Clear specific model
    MLXProvider.clear_model_cache("mlx-community/phi-2")
    
    # Verify removal
    final_models = MLXProvider.list_cached_models()
    assert len(final_models) == len(initial_models)

def test_cache_limits():
    """Test cache size limits."""
    # Create multiple models
    models = [
        "mlx-community/phi-2",
        "mlx-community/mistral-7b-v0.1",
        "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX"
    ]
    
    for model in models:
        llm = create_llm("mlx", model=model)
        llm.generate("Test prompt")
    
    # Check cache size
    cached_models = MLXProvider.list_cached_models()
    assert len(cached_models) <= MLXProvider._max_cached_models

def test_cache_persistence():
    """Test cache persistence across sessions."""
    # Create and use a model
    llm = create_llm("mlx")
    llm.generate("Test prompt")
    
    # Get cache info
    cache_info = MLXProvider.get_cache_info()
    
    # Create new provider
    llm2 = create_llm("mlx")
    
    # Should use cached model
    start_time = time.time()
    llm2.generate("Test prompt")
    load_time = time.time() - start_time
    
    # Should be faster than cold start
    assert load_time < 5  # Arbitrary threshold

### 4. CLI Integration Tests

Create test file at `tests/integration/test_mlx_cli.py`:

```python
"""Integration tests for MLX provider CLI tools."""

import pytest
import subprocess
import json
from pathlib import Path

def run_cli(command: str) -> subprocess.CompletedProcess:
    """Run CLI command and return result."""
    return subprocess.run(
        command.split(),
        capture_output=True,
        text=True
    )

def test_basic_generation():
    """Test basic text generation via CLI."""
    result = run_cli(
        "abstractllm mlx generate -m mlx-community/phi-2 -p 'Hello'"
    )
    assert result.returncode == 0
    assert len(result.stdout) > 0

def test_vision_generation():
    """Test vision capabilities via CLI."""
    result = run_cli(
        "abstractllm mlx generate "
        "-m mlx-community/Qwen2.5-VL-32B-Instruct-6bit "
        "-p 'What is this?' "
        "-i tests/examples/mountain_path.jpg"
    )
    assert result.returncode == 0
    assert len(result.stdout) > 0

def test_streaming_output():
    """Test streaming output via CLI."""
    result = run_cli(
        "abstractllm mlx generate "
        "-m mlx-community/phi-2 "
        "-p 'Tell me a story' "
        "--stream"
    )
    assert result.returncode == 0
    assert len(result.stdout) > 0

def test_system_check():
    """Test system compatibility check."""
    result = run_cli("abstractllm mlx system-check")
    assert result.returncode == 0
    
    # Parse JSON output
    info = json.loads(result.stdout)
    assert "platform" in info
    assert "processor" in info
    assert "compatible" in info

def test_model_listing():
    """Test model listing functionality."""
    result = run_cli("abstractllm mlx list-models")
    assert result.returncode == 0
    
    # Parse JSON output
    models = json.loads(result.stdout)
    assert "text_models" in models
    assert "vision_models" in models

## References
- See `docs/mlx/vision-upgrade.md` for vision implementation details
- See `docs/mlx/mlx_integration_architecture.md` for architectural guidance
- See MLX documentation for performance guidelines

## Testing
Run the integration tests:

```bash
# Run all integration tests
pytest tests/integration/

# Run specific test suite
pytest tests/integration/test_mlx_vision_integration.py
pytest tests/integration/test_mlx_performance.py
pytest tests/integration/test_mlx_caching.py
pytest tests/integration/test_mlx_cli.py
```

## Success Criteria
1. All integration tests pass
2. Vision capabilities work correctly
3. Performance meets expectations
4. Caching works efficiently
5. CLI tools function properly
6. Memory usage is within limits
7. Error handling works as expected 