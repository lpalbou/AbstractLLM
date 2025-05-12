"""
Tests for the MLX provider's media handling functionality.

These tests will only run on Apple Silicon hardware with MLX installed.
"""

import os
import platform
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

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
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MLX_AVAILABLE,
    reason="MLX dependencies not available"
)

from abstractllm import create_llm
from abstractllm.enums import ModelParameter, ModelCapability
from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.exceptions import UnsupportedFeatureError, FileProcessingError


class TestMLXMediaHandling:
    """Tests for the MLX provider's media handling functionality."""

    @pytest.fixture
    def mlx_provider(self):
        """Return an initialized MLX provider."""
        return MLXProvider({
            ModelParameter.MODEL: "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-6bit"
        })

    @pytest.fixture
    def text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("This is a test text file content.")
        temp_path = Path(f.name)
        yield temp_path
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)

    def test_process_text_file(self, mlx_provider, text_file):
        """Test processing a text file."""
        # Test with the _process_files method directly
        result = mlx_provider._process_files("Original prompt: ", [text_file])
        
        # Verify result contains both original prompt and file content
        assert "Original prompt: " in result
        assert "This is a test text file content." in result
        assert f"Content from file '{text_file.name}'" in result

    def test_nonexistent_file(self, mlx_provider):
        """Test handling of nonexistent files."""
        # Use a path that definitely doesn't exist
        non_existent_path = Path("/path/to/nonexistent/file.txt")
        
        with pytest.raises(FileProcessingError) as excinfo:
            mlx_provider._process_files("Test prompt", [non_existent_path])
        
        assert "File not found" in str(excinfo.value)

    @patch('abstractllm.providers.mlx_provider.MediaFactory')
    def test_vision_capability_check(self, mock_factory, mlx_provider):
        """Test checking vision capability when processing image files."""
        # Create a mock image input
        mock_media_input = type('MockMediaInput', (), {'media_type': 'image'})
        mock_factory.from_source.return_value = mock_media_input
        
        # Set up a dummy path that exists
        with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
            image_path = Path(f.name)
            
            # Test with vision capability disabled
            mlx_provider._is_vision_model = False
            
            with pytest.raises(UnsupportedFeatureError) as excinfo:
                mlx_provider._process_files("Test prompt", [image_path])
            
            assert "vision" in str(excinfo.value)
            
            # Test with vision capability enabled
            mlx_provider._is_vision_model = True
            
            # Should not raise an exception
            result = mlx_provider._process_files("Test prompt", [image_path])
            assert "Test prompt" in result

    def test_multiple_files(self, mlx_provider):
        """Test processing multiple files."""
        # Create multiple temporary text files
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"Content of file {i}")
                files.append(Path(f.name))
        
        try:
            # Process all files
            result = mlx_provider._process_files("Multiple files test: ", files)
            
            # Verify all file contents are in the result
            assert "Multiple files test: " in result
            for i in range(3):
                assert f"Content of file {i}" in result
        finally:
            # Clean up
            for f in files:
                if f.exists():
                    os.unlink(f) 