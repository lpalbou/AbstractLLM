# Task 15: Create Integration Tests

## Description
Create integration tests for the MLX provider to verify it works correctly with the broader AbstractLLM ecosystem.

## Requirements
1. Create an integration test file in the appropriate test directory
2. Test integration with AbstractLLM's factory system
3. Test the provider with various parameter combinations
4. Include tests for error cases and graceful fallbacks

## Implementation Details

Create a test file at `tests/integration/test_mlx_integration.py`:

```python
"""
Integration tests for the MLX provider.

These tests verify that the MLX provider integrates properly with
the AbstractLLM ecosystem.
"""

import pytest
import platform
import os
from pathlib import Path

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

from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.exceptions import UnsupportedFeatureError

def test_factory_integration():
    """Test integration with AbstractLLM factory."""
    # Create provider using factory
    llm = create_llm("mlx")
    
    # Verify it's the correct type
    assert llm.__class__.__name__ == "MLXProvider"
    
    # Verify it has the correct model
    if hasattr(llm, 'config_manager'):
        model = llm.config_manager.get_param(ModelParameter.MODEL)
        assert model is not None
        assert isinstance(model, str)

def test_parameter_integration():
    """Test parameter passing through AbstractLLM factory."""
    # Create provider with custom parameters
    custom_params = {
        ModelParameter.TEMPERATURE: 0.8,
        ModelParameter.MAX_TOKENS: 1000,
        ModelParameter.TOP_P: 0.95,
        "cache_dir": "~/custom_cache",
        "quantize": False
    }
    
    llm = create_llm("mlx", **custom_params)
    
    # Verify parameters were set correctly
    if hasattr(llm, 'config_manager'):
        for key, value in custom_params.items():
            param_value = llm.config_manager.get_param(key)
            assert param_value == value

def test_create_with_different_models():
    """Test creating the provider with different models."""
    # Test with a small model
    llm1 = create_llm("mlx", model="mlx-community/phi-2")
    
    # Test with a different model
    llm2 = create_llm("mlx", model="mlx-community/mistral-7b-v0.1")
    
    # Verify they have different models
    if hasattr(llm1, 'config_manager') and hasattr(llm2, 'config_manager'):
        model1 = llm1.config_manager.get_param(ModelParameter.MODEL)
        model2 = llm2.config_manager.get_param(ModelParameter.MODEL)
        assert model1 != model2

def test_capability_integration():
    """Test capability reporting integration."""
    llm = create_llm("mlx")
    
    # Get capabilities
    capabilities = llm.get_capabilities()
    
    # Check required capabilities are present and have correct types
    assert ModelCapability.STREAMING in capabilities
    assert isinstance(capabilities[ModelCapability.STREAMING], bool)
    
    assert ModelCapability.MAX_TOKENS in capabilities
    assert isinstance(capabilities[ModelCapability.MAX_TOKENS], int)
    
    assert ModelCapability.SYSTEM_PROMPT in capabilities
    assert isinstance(capabilities[ModelCapability.SYSTEM_PROMPT], bool)
    
    assert ModelCapability.ASYNC in capabilities
    assert isinstance(capabilities[ModelCapability.ASYNC], bool)
    
    assert ModelCapability.VISION in capabilities
    assert isinstance(capabilities[ModelCapability.VISION], bool)

def test_unsupported_features():
    """Test proper error handling for unsupported features."""
    llm = create_llm("mlx")
    
    # Try to use tool calling, which should not be supported
    with pytest.raises(UnsupportedFeatureError):
        llm.generate("What's the weather?", tools=[
            {"type": "function", "function": {"name": "get_weather"}}
        ])

def test_file_handling_integration():
    """Test integration with file handling system."""
    llm = create_llm("mlx")
    
    # Create a temporary text file
    temp_file = Path("temp_test_file.txt")
    with open(temp_file, "w") as f:
        f.write("This is test content.")
    
    try:
        # Test text file processing
        response = llm.generate("Summarize the file:", files=[temp_file])
        
        # Verify response contains something
        assert response is not None
        assert response.text is not None
        assert len(response.text) > 0
    finally:
        # Clean up
        if temp_file.exists():
            os.unlink(temp_file)
```

## References
- See AbstractLLM's existing integration tests for reference
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_integration_architecture.md` for architectural guidance

## Testing
Run the integration tests with:

```bash
pytest -xvs tests/integration/test_mlx_integration.py
``` 