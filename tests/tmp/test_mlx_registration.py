"""
Tests for MLX provider registration.

These tests verify that the MLX provider is properly registered
and that appropriate error messages are shown on unsupported platforms.
"""

import platform
import pytest
from unittest.mock import patch, MagicMock

from abstractllm.providers.registry import register_mlx_provider
from abstractllm.factory import create_llm, get_llm_providers

# Check if we're on Apple Silicon
is_macos = platform.system().lower() == "darwin"
is_arm = platform.processor() == "arm"
is_apple_silicon = is_macos and is_arm

class TestMLXRegistration:
    """Tests for MLX provider registration."""
    
    def test_mlx_in_available_providers(self):
        """Test that MLX is listed in available providers on Apple Silicon."""
        # Skip if not on Apple Silicon
        if not is_apple_silicon:
            pytest.skip("Test requires Apple Silicon")
            
        # Check if MLX is in the list of available providers
        providers = get_llm_providers()
        assert "mlx" in providers, "MLX provider should be available on Apple Silicon"
    
    @pytest.mark.skipif(is_apple_silicon, reason="Test only relevant on non-Apple Silicon")
    def test_mlx_not_available_on_non_apple_silicon(self):
        """Test that MLX is not available on non-Apple Silicon platforms."""
        # This test should only run on non-Apple Silicon platforms
        with patch("platform.system", return_value="Linux"):
            with patch("platform.processor", return_value="x86_64"):
                # Attempt to register MLX provider
                result = register_mlx_provider()
                assert result is False, "MLX provider should not register on non-Apple Silicon"
    
    def test_mlx_registration_with_dependencies(self):
        """Test MLX registration with all dependencies available."""
        # Mock platform to be Apple Silicon
        with patch("platform.system", return_value="darwin"):
            with patch("platform.processor", return_value="arm"):
                # Mock MLX dependencies
                with patch.dict("sys.modules", {
                    "mlx.core": MagicMock(),
                    "mlx_lm": MagicMock()
                }):
                    # Attempt to register MLX provider
                    result = register_mlx_provider()
                    assert result is True, "MLX provider should register with all dependencies"
    
    def test_mlx_registration_missing_dependencies(self):
        """Test MLX registration with missing dependencies."""
        # Mock platform to be Apple Silicon
        with patch("platform.system", return_value="darwin"):
            with patch("platform.processor", return_value="arm"):
                # Mock the import of mlx.core to raise ImportError
                with patch("abstractllm.providers.registry.MLX_AVAILABLE", False):
                    # Attempt to register MLX provider
                    result = register_mlx_provider()
                    assert result is False, "MLX provider should not register with missing dependencies"
    
    @pytest.mark.skipif(not is_apple_silicon, reason="Test only relevant on Apple Silicon")
    def test_create_llm_with_mlx(self):
        """Test creating an MLX provider instance on Apple Silicon."""
        try:
            import mlx.core
            import mlx_lm
            # If we got here, MLX is available
            llm = create_llm("mlx")
            assert llm is not None, "Should be able to create MLX provider on Apple Silicon"
        except ImportError:
            pytest.skip("MLX dependencies not available")
    
    def test_create_llm_platform_error(self):
        """Test that creating an MLX provider on non-Apple Silicon raises appropriate error."""
        # Mock platform to be non-Apple Silicon
        with patch("platform.system", return_value="Linux"):
            with patch("platform.processor", return_value="x86_64"):
                # Attempt to create MLX provider
                with pytest.raises(ValueError) as excinfo:
                    create_llm("mlx")
                # Check error message
                assert "MLX provider requires macOS with Apple Silicon" in str(excinfo.value)
    
    def test_create_llm_dependency_error(self):
        """Test that creating an MLX provider with missing dependencies raises appropriate error."""
        # Mock platform to be Apple Silicon
        with patch("platform.system", return_value="darwin"):
            with patch("platform.processor", return_value="arm"):
                # Mock dependency check to fail
                with patch("abstractllm.factory._check_dependency", return_value=False):
                    # Attempt to create MLX provider
                    with pytest.raises(ImportError) as excinfo:
                        create_llm("mlx")
                    # Check error message
                    assert "Missing required dependencies for MLX provider" in str(excinfo.value)
                    assert "pip install 'abstractllm[mlx]'" in str(excinfo.value) 