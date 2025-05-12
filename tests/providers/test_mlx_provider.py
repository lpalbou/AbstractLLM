"""
Tests for the MLX provider.

These tests will only run on Apple Silicon hardware with MLX installed.
"""

import platform
import pytest
import asyncio
from typing import AsyncGenerator

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


class TestMLXProvider:
    """Tests for the MLX provider."""

    @pytest.fixture
    def test_model(self):
        """Return a suitable test model."""
        # Use a small model for quick testing
        return "mlx-community/phi-2"
    
    @pytest.fixture
    def mlx_llm(self, test_model):
        """Return an initialized MLX provider."""
        return create_llm("mlx", **{
            ModelParameter.MODEL: test_model,
            ModelParameter.MAX_TOKENS: 100  # Small limit for faster tests
        })
    
    def test_initialization(self, mlx_llm):
        """Test MLX provider initialization."""
        assert mlx_llm is not None
        assert mlx_llm.get_capabilities().get(ModelCapability.ASYNC) is True
    
    def test_sync_generation(self, mlx_llm):
        """Test synchronous text generation."""
        response = mlx_llm.generate("Hello, world!")
        assert response is not None
        assert response.text is not None
        assert len(response.text) > 0
    
    def test_sync_streaming(self, mlx_llm):
        """Test synchronous streaming generation."""
        chunks = list(mlx_llm.generate("Hello, world!", stream=True))
        assert len(chunks) > 0
        assert all(chunk.text for chunk in chunks)
        # Last chunk should have the complete text
        assert len(chunks[-1].text) > 0
    
    @pytest.mark.asyncio
    async def test_async_generation(self, mlx_llm):
        """Test asynchronous text generation."""
        response = await mlx_llm.generate_async("Hello, world!")
        assert response is not None
        assert response.text is not None
        assert len(response.text) > 0
    
    @pytest.mark.asyncio
    async def test_async_streaming(self, mlx_llm):
        """Test asynchronous streaming generation."""
        chunks = []
        async_gen = await mlx_llm.generate_async("Hello, world!", stream=True)
        
        # Verify it returns an AsyncGenerator
        assert isinstance(async_gen, AsyncGenerator)
        
        # Collect chunks
        async for chunk in async_gen:
            chunks.append(chunk)
        
        assert len(chunks) > 0
        assert all(chunk.text for chunk in chunks)
        # Last chunk should have the complete text
        assert len(chunks[-1].text) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_generation(self, mlx_llm):
        """Test concurrent async generations."""
        # Create multiple generation tasks
        prompts = ["Hello", "Hi there", "Good day"]
        tasks = [
            mlx_llm.generate_async(prompt) 
            for prompt in prompts
        ]
        
        # Run concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert len(results) == len(prompts)
        assert all(response.text for response in results)

    @pytest.fixture
    def vision_model(self):
        """Return a vision-capable model name."""
        return "mlx-community/llava-1.5-7b-mlx"

    def test_capabilities(self, mlx_llm):
        """Test capability reporting."""
        capabilities = mlx_llm.get_capabilities()
        
        # Check basic capabilities
        assert capabilities.get(ModelCapability.STREAMING) is True
        assert capabilities.get(ModelCapability.SYSTEM_PROMPT) is True
        assert capabilities.get(ModelCapability.ASYNC) is True
        assert capabilities.get(ModelCapability.FUNCTION_CALLING) is False
        assert capabilities.get(ModelCapability.TOOL_USE) is False
        
        # Check max tokens
        assert capabilities.get(ModelCapability.MAX_TOKENS) > 0
        
        # Non-vision model should report False for vision capability
        assert capabilities.get(ModelCapability.VISION) is False

    def test_vision_capability_detection(self, test_model, vision_model):
        """Test vision capability detection."""
        # Access the MLX provider directly to test internal methods
        from abstractllm.providers.mlx_provider import MLXProvider
        
        provider = MLXProvider({ModelParameter.MODEL: test_model})
        
        # Test non-vision model
        assert provider._check_vision_capability(test_model) is False
        
        # Test vision models
        assert provider._check_vision_capability(vision_model) is True
        assert provider._check_vision_capability("some-model-with-vision-capability") is True
        assert provider._check_vision_capability("clip-model-example") is True
        assert provider._check_vision_capability("regular-text-model") is False 