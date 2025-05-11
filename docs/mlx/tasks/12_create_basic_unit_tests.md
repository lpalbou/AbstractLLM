# Task 12: Create Basic Unit Tests

## Description
Create a basic set of unit tests for the MLX provider to verify its functionality using pytest.

## Requirements
1. Create a test file in the appropriate test directory
2. Include tests for initialization, model loading, and generation
3. Add platform-specific skipping for non-Apple Silicon platforms
4. Test key functionality like streaming and system prompts

## Implementation Details

Create a test file at `tests/providers/test_mlx_provider.py`:

```python
"""
Tests for the MLX provider.

These tests will only run on Apple Silicon hardware.
"""

import pytest
import platform
from pathlib import Path
import os

# Skip all tests if not on macOS with Apple Silicon
is_macos = platform.system().lower() == "darwin"
is_arm = platform.processor() == "arm" 
pytestmark = pytest.mark.skipif(
    not (is_macos and is_arm),
    reason="MLX tests require macOS with Apple Silicon"
)

# Try to import MLX, skip if not available
try:
    import mlx.core
    import mlx_lm
except ImportError:
    pytestmark = pytest.mark.skip(reason="MLX dependencies not available")

from abstractllm import create_llm, ModelParameter


@pytest.fixture
def mlx_provider():
    """Fixture to create a basic MLX provider instance for testing."""
    provider = create_llm("mlx", **{
        ModelParameter.MODEL: "mlx-community/phi-2",  # Small model for quick testing
        ModelParameter.MAX_TOKENS: 100  # Small limit for tests
    })
    return provider


def test_mlx_provider_initialization(mlx_provider):
    """Test MLX provider initialization."""
    # Check initialization
    assert mlx_provider is not None
    
    # Check capabilities
    capabilities = mlx_provider.get_capabilities()
    assert capabilities is not None
    assert "streaming" in capabilities or ModelParameter.STREAMING in capabilities


def test_mlx_model_loading(mlx_provider):
    """Test model loading."""
    # Force model loading if not already loaded
    if hasattr(mlx_provider, 'load_model'):
        mlx_provider.load_model()
    
    # Model should be loaded now
    assert hasattr(mlx_provider, '_model')
    assert mlx_provider._model is not None
    
    assert hasattr(mlx_provider, '_tokenizer')
    assert mlx_provider._tokenizer is not None


def test_mlx_text_generation(mlx_provider):
    """Test basic text generation."""
    # Generate text
    response = mlx_provider.generate("Hello, world!")
    
    # Check response
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0
    assert response.model is not None
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens > 0


def test_mlx_system_prompt(mlx_provider):
    """Test generation with system prompt."""
    # Generate with system prompt
    system_prompt = "You are a helpful assistant."
    response = mlx_provider.generate("Hello!", system_prompt=system_prompt)
    
    # Check response
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0


def test_mlx_streaming(mlx_provider):
    """Test streaming generation."""
    # Generate with streaming
    chunks = list(mlx_provider.generate("Tell me a short story.", stream=True))
    
    # Check chunks
    assert len(chunks) > 0
    
    # Check final chunk
    final_chunk = chunks[-1]
    assert final_chunk.text is not None
    assert len(final_chunk.text) > 0


@pytest.mark.parametrize(
    "model_name", 
    ["mlx-community/phi-2", "mlx-community/mistral-7b-v0.1"]
)
def test_different_models(model_name):
    """Test different models can be loaded."""
    llm = create_llm("mlx", **{
        ModelParameter.MODEL: model_name,
        ModelParameter.MAX_TOKENS: 50
    })
    
    # Verify model name is set correctly
    assert llm.config_manager.get_param(ModelParameter.MODEL) == model_name


def test_error_on_tool_use(mlx_provider):
    """Test error handling when using unsupported features."""
    from abstractllm.exceptions import UnsupportedFeatureError
    
    # Tools are not supported, should raise an error
    with pytest.raises(UnsupportedFeatureError):
        mlx_provider.generate("What's the weather?", tools=[
            {"type": "function", "function": {"name": "get_weather"}}
        ])


def test_text_file_handling(mlx_provider, tmp_path):
    """Test text file handling capability."""
    # Create a temporary text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test content for MLX processing.")
    
    # Generate with file input
    response = mlx_provider.generate(
        "What's in the file?", 
        files=[test_file]
    )
    
    # Check response
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0
```

## References
- See pytest documentation: https://docs.pytest.org/
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_usage_examples.md` for usage patterns to test

## Testing
Run the tests with pytest:

```bash
# Run all tests
pytest -xvs tests/providers/test_mlx_provider.py

# Run specific test
pytest -xvs tests/providers/test_mlx_provider.py::test_mlx_text_generation
``` 