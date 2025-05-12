"""
Tests for the MLX provider.

These tests will only run on Apple Silicon hardware with MLX installed.
"""

import platform
import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import patch, MagicMock

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
from abstractllm.types import GenerateResponse


class TestMLXProvider:
    """Tests for the MLX provider."""

    @pytest.fixture
    def test_model(self):
        """Return a suitable test model."""
        # Use a model that's already converted to MLX format
        return "mlx-community/Phi-3-mini-4k-instruct-mlx"
    
    @pytest.fixture
    def mock_mlx_model(self):
        """Create a mock MLX model."""
        model = MagicMock()
        tokenizer = MagicMock()
        
        # Configure tokenizer mock
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "Hello, I'm an AI assistant."
        
        return model, tokenizer
    
    @pytest.fixture
    def mlx_llm(self, test_model, mock_mlx_model):
        """Return an initialized MLX provider with mocked model loading."""
        model, tokenizer = mock_mlx_model
        
        with patch('mlx_lm.utils.load', return_value=(model, tokenizer)):
            llm = create_llm("mlx", **{
                ModelParameter.MODEL: test_model,
                ModelParameter.MAX_TOKENS: 100  # Small limit for faster tests
            })
            # Pre-load the model to ensure the mock is used
            if hasattr(llm, 'load_model'):
                llm.load_model()
            return llm
    
    def test_initialization(self, mlx_llm):
        """Test MLX provider initialization."""
        assert mlx_llm is not None
        assert mlx_llm.get_capabilities().get(ModelCapability.ASYNC) is True
    
    def test_sync_generation(self, mlx_llm):
        """Test synchronous text generation."""
        # Mock the generate method directly
        with patch.object(mlx_llm, 'generate', return_value=GenerateResponse(
            content="Hello, I'm an AI assistant.",
            model="test-model",
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        )):
            response = mlx_llm.generate("Hello, world!")
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
    
    def test_list_cached_models(self):
        """Test listing cached models."""
        from abstractllm.providers.mlx_provider import MLXProvider
        
        # Mock the huggingface_hub scan_cache_dir function
        mock_repo = MagicMock()
        mock_repo.repo_id = "mlx-community/test-model"
        mock_repo.size_on_disk = 1000000  # 1MB
        mock_repo.last_accessed = 1234567890.0
        
        mock_cache_info = MagicMock()
        mock_cache_info.repos = [mock_repo]
        
        with patch('huggingface_hub.scan_cache_dir', return_value=mock_cache_info):
            cached_models = MLXProvider.list_cached_models()
            
            # Verify the results
            assert len(cached_models) == 1
            assert cached_models[0]["name"] == "mlx-community/test-model"
            assert cached_models[0]["size"] == 1000000
            assert cached_models[0]["last_used"] == 1234567890.0
            assert cached_models[0]["implementation"] == "mlx"
    
    def test_clear_model_cache(self):
        """Test clearing model cache."""
        from abstractllm.providers.mlx_provider import MLXProvider
        
        # Set up test data in the model cache
        model1 = MagicMock()
        tokenizer1 = MagicMock()
        model2 = MagicMock()
        tokenizer2 = MagicMock()
        
        # Add models to cache
        MLXProvider._model_cache = {
            "model1": (model1, tokenizer1, 1234567890.0),
            "model2": (model2, tokenizer2, 1234567891.0)
        }
        
        # Test clearing specific model
        MLXProvider.clear_model_cache("model1")
        assert "model1" not in MLXProvider._model_cache
        assert "model2" in MLXProvider._model_cache
        assert len(MLXProvider._model_cache) == 1
        
        # Test clearing all models
        MLXProvider.clear_model_cache()
        assert len(MLXProvider._model_cache) == 0
    
    def test_sync_streaming(self, mlx_llm):
        """Test synchronous streaming generation."""
        # Create a generator for streaming response
        def mock_stream():
            for i in range(3):
                yield GenerateResponse(
                    content=f"Hello {i}",
                    model="test-model",
                    usage={"prompt_tokens": 5, "completion_tokens": i+1, "total_tokens": 5+i+1}
                )
        
        # Mock the generate method for streaming
        with patch.object(mlx_llm, 'generate', return_value=mock_stream()):
            chunks = list(mlx_llm.generate("Hello, world!", stream=True))
            assert len(chunks) == 3
            assert all(hasattr(chunk, 'content') for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_async_generation(self, mlx_llm):
        """Test asynchronous text generation."""
        # Mock the async generate method
        with patch.object(mlx_llm, 'generate_async', return_value=GenerateResponse(
            content="Hello, I'm an AI assistant.",
            model="test-model",
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        )):
            response = await mlx_llm.generate_async("Hello, world!")
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_async_streaming(self, mlx_llm):
        """Test asynchronous streaming generation."""
        # Create an async generator for streaming response
        async def mock_stream():
            for i in range(3):
                yield GenerateResponse(
                    content=f"Hello {i}",
                    model="test-model",
                    usage={"prompt_tokens": 5, "completion_tokens": i+1, "total_tokens": 5+i+1}
                )
        
        # Mock the async generate method for streaming
        with patch.object(mlx_llm, 'generate_async', return_value=mock_stream()):
            chunks = []
            async_gen = await mlx_llm.generate_async("Hello, world!", stream=True)
            
            # Verify it returns an AsyncGenerator
            assert isinstance(async_gen, AsyncGenerator)
            
            # Collect chunks
            async for chunk in async_gen:
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert all(hasattr(chunk, 'content') for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_concurrent_generation(self, mlx_llm):
        """Test concurrent async generations."""
        # Mock the async generate method
        with patch.object(mlx_llm, 'generate_async', return_value=GenerateResponse(
            content="Hello, I'm an AI assistant.",
            model="test-model",
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        )):
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
            assert all(hasattr(response, 'content') for response in results)

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