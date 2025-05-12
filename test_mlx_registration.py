"""
Tests for MLX provider registration.

These tests verify that the MLX provider is properly registered
and that appropriate error messages are shown on unsupported platforms.
"""

import platform
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

from abstractllm.providers.registry import register_mlx_provider
from abstractllm.factory import create_llm, get_llm_providers
from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.enums import ModelCapability

# Check if we're on Apple Silicon
is_macos = platform.system().lower() == "darwin"
is_arm = platform.processor() == "arm"
is_apple_silicon = is_macos and is_arm

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not is_apple_silicon,
    reason="MLX tests require macOS with Apple Silicon"
)

class TestMLXRegistration:
    """Tests for MLX provider registration."""
    
    def test_mlx_in_available_providers(self):
        """Test that MLX is listed in available providers on Apple Silicon."""
        # Check if MLX is in the list of available providers
        providers = get_llm_providers()
        assert "mlx" in providers, "MLX provider should be available on Apple Silicon"
    
    def test_mlx_provider_registration(self):
        """Test that MLX provider registers successfully on Apple Silicon."""
        # Register the MLX provider
        result = register_mlx_provider()
        assert result is True, "MLX provider should register successfully on Apple Silicon"
    
    def test_create_mlx_provider(self):
        """Test creating an MLX provider instance."""
        # Create an MLX provider
        llm = create_llm("mlx")
        assert llm is not None, "Should be able to create MLX provider"
        assert isinstance(llm, MLXProvider), "Should create an instance of MLXProvider"
    
    def test_mlx_provider_capabilities(self):
        """Test that MLX provider reports correct capabilities."""
        # Create an MLX provider
        llm = create_llm("mlx")
        
        # Check capabilities
        capabilities = llm.get_capabilities()
        assert capabilities is not None, "Should return capabilities"
        assert capabilities.get(ModelCapability.STREAMING) is True, "Should support streaming"
        assert capabilities.get(ModelCapability.SYSTEM_PROMPT) is True, "Should support system prompts"
        assert capabilities.get(ModelCapability.ASYNC) is True, "Should support async"
    
    def test_list_cached_models(self):
        """Test listing cached models."""
        # List cached models
        cached_models = MLXProvider.list_cached_models()
        
        # Verify the result is a list
        assert isinstance(cached_models, list), "Should return a list of cached models"
        
        # If models are cached, verify their structure
        if cached_models:
            model = cached_models[0]
            assert "name" in model, "Model info should include name"
            assert "size" in model, "Model info should include size"
            assert "last_used" in model, "Model info should include last_used"
            assert "implementation" in model, "Model info should include implementation"
            assert model["implementation"] == "mlx", "Implementation should be 'mlx'"
    
    def test_clear_model_cache(self):
        """Test clearing model cache."""
        # Add a test model to the cache
        MLXProvider._model_cache = {
            "test-model": (object(), object(), 1234567890.0)
        }
        
        # Verify the model is in the cache
        assert "test-model" in MLXProvider._model_cache, "Test model should be in cache"
        
        # Clear the cache
        MLXProvider.clear_model_cache()
        
        # Verify the cache is empty
        assert len(MLXProvider._model_cache) == 0, "Cache should be empty after clearing"
    
    def test_selective_cache_clearing(self):
        """Test selectively clearing the model cache."""
        # Add test models to the cache
        model1 = object()
        model2 = object()
        MLXProvider._model_cache = {
            "model1": (model1, object(), 1234567890.0),
            "model2": (model2, object(), 1234567891.0)
        }
        
        # Verify both models are in the cache
        assert "model1" in MLXProvider._model_cache, "model1 should be in cache"
        assert "model2" in MLXProvider._model_cache, "model2 should be in cache"
        
        # Clear only model1
        MLXProvider.clear_model_cache("model1")
        
        # Verify model1 is removed but model2 remains
        assert "model1" not in MLXProvider._model_cache, "model1 should be removed from cache"
        assert "model2" in MLXProvider._model_cache, "model2 should still be in cache" 