"""
Tests for the MLX provider functionality.

These tests verify that the MLX provider can properly load models and generate text.
"""

import platform
import pytest
from pathlib import Path
import os
import time

from abstractllm import create_llm, ModelParameter, ModelCapability

# Check if we're on Apple Silicon
is_macos = platform.system().lower() == "darwin"
is_arm = platform.processor() == "arm"
is_apple_silicon = is_macos and is_arm

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    not is_apple_silicon,
    reason="MLX tests require macOS with Apple Silicon"
)

# Try to import MLX, skip if not available
try:
    import mlx.core
    import mlx_lm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="MLX dependencies not available")

# Use a small model that is known to be compatible with MLX
TEST_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit-MLX"

class TestMLXProvider:
    """Tests for MLX provider functionality."""
    
    @pytest.fixture(scope="module")
    def mlx_llm(self):
        """Create and cache an MLX provider instance for the tests."""
        # Use a small model with limited tokens for faster testing
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: TEST_MODEL,
            ModelParameter.MAX_TOKENS: 100
        })
        # Pre-load the model to avoid loading it multiple times
        if hasattr(llm, 'load_model'):
            llm.load_model()
        return llm
    
    def test_model_loading(self, mlx_llm):
        """Test that the model loads successfully."""
        assert mlx_llm._is_loaded, "Model should be loaded"
        assert mlx_llm._model is not None, "Model should not be None"
        assert mlx_llm._tokenizer is not None, "Tokenizer should not be None"
    
    def test_basic_generation(self, mlx_llm):
        """Test basic text generation."""
        prompt = "Hello, my name is"
        response = mlx_llm.generate(prompt)
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        # Check that usage statistics are present
        assert hasattr(response, 'usage'), "Response should have usage attribute"
        assert response.usage is not None, "Response should have usage statistics"
        assert response.usage.get("prompt_tokens", 0) > 0, "Should report prompt tokens"
        assert response.usage.get("completion_tokens", 0) > 0, "Should report completion tokens"
        assert response.usage.get("total_tokens", 0) > 0, "Should report total tokens"
    
    def test_system_prompt(self, mlx_llm):
        """Test generation with a system prompt."""
        system_prompt = "You are a helpful AI assistant that speaks like a pirate."
        user_prompt = "Tell me about the weather."
        
        response = mlx_llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
    
    def test_streaming_generation(self, mlx_llm):
        """Test streaming text generation."""
        prompt = "Count from 1 to 5."
        
        # Collect streaming chunks
        chunks = []
        for chunk in mlx_llm.generate(prompt, stream=True):
            chunks.append(chunk)
            
        # Check that we got chunks
        assert len(chunks) > 0, "Should return streaming chunks"
        
        # Check the final chunk
        final_chunk = chunks[-1]
        assert hasattr(final_chunk, 'content'), "Final chunk should have content attribute"
        assert final_chunk.content is not None, "Final chunk should have content"
        assert len(final_chunk.content) > 0, "Final chunk content should not be empty"
    
    @pytest.mark.asyncio
    async def test_async_generation(self, mlx_llm):
        """Test async text generation."""
        prompt = "What is the capital of France?"
        
        # Test async generation
        response = await mlx_llm.generate_async(prompt)
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
    
    @pytest.mark.asyncio
    async def test_async_streaming(self, mlx_llm):
        """Test async streaming text generation."""
        prompt = "List three colors."
        
        # Collect streaming chunks
        chunks = []
        async for chunk in await mlx_llm.generate_async(prompt, stream=True):
            chunks.append(chunk)
            
        # Check that we got chunks
        assert len(chunks) > 0, "Should return streaming chunks"
        
        # Check the final chunk
        final_chunk = chunks[-1]
        assert hasattr(final_chunk, 'content'), "Final chunk should have content attribute"
        assert final_chunk.content is not None, "Final chunk should have content"
        assert len(final_chunk.content) > 0, "Final chunk content should not be empty"
    
    def test_text_file_processing(self, mlx_llm, tmp_path):
        """Test processing a text file."""
        # Create a temporary text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is a test file with some content.")
        
        prompt = "Summarize the content of the file:"
        
        # Generate with the file
        response = mlx_llm.generate(prompt, files=[text_file])
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
    
    def test_model_caching(self):
        """Test that models are properly cached."""
        # Clear the cache first
        from abstractllm.providers.mlx_provider import MLXProvider
        MLXProvider.clear_model_cache()
        
        # Create a provider and load the model
        llm1 = create_llm("mlx", model=TEST_MODEL)
        llm1.generate("Hello")  # This will load the model
        
        # Check that the model is in the cache
        assert TEST_MODEL in MLXProvider._model_cache, "Model should be in cache"
        
        # Create another provider with the same model
        start_time = time.time()
        llm2 = create_llm("mlx", model=TEST_MODEL)
        llm2.generate("World")  # This should use the cached model
        load_time = time.time() - start_time
        
        # The second load should be faster than initial load, but may still take time
        # due to inference overhead
        assert load_time < 15.0, "Loading from cache should be reasonably fast" 