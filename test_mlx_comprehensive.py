"""
Comprehensive tests for the MLX provider.

These tests verify all aspects of the MLX provider functionality
using real models and real examples.
"""

import pytest
import platform
import tempfile
import os
from pathlib import Path
import time
import asyncio
import shutil

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

# Use the specified model for all tests
TEXT_MODEL = "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit"
VISION_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"

class TestMLXProviderComprehensive:
    """Comprehensive tests for MLX provider functionality."""
    
    @pytest.fixture(scope="module")
    def text_llm(self):
        """Create and cache a text MLX provider instance for the tests."""
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: TEXT_MODEL,
            ModelParameter.MAX_TOKENS: 100  # Limit tokens for faster tests
        })
        # Pre-load the model to avoid loading it multiple times
        if hasattr(llm, 'load_model'):
            llm.load_model()
        return llm
    
    @pytest.fixture(scope="module")
    def vision_llm(self):
        """Create and cache a vision MLX provider instance for the tests."""
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: VISION_MODEL,
            ModelParameter.MAX_TOKENS: 100  # Limit tokens for faster tests
        })
        # Pre-load the model to avoid loading it multiple times
        if hasattr(llm, 'load_model'):
            llm.load_model()
        return llm
    
    @pytest.fixture
    def test_files(self):
        """Create test files for file processing tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a text file
            text_file = Path(temp_dir) / "test.txt"
            text_file.write_text("This is a test file with some sample content for testing.")
            
            # Create a markdown file
            md_file = Path(temp_dir) / "test.md"
            md_file.write_text("# Test Markdown\n\nThis is a **markdown** file with some _formatting_.")
            
            # Create a JSON file
            json_file = Path(temp_dir) / "test.json"
            json_file.write_text('{"name": "Test", "value": 42, "items": ["one", "two", "three"]}')
            
            # Create a Python file
            py_file = Path(temp_dir) / "test.py"
            py_file.write_text('def hello():\n    print("Hello, world!")\n\nif __name__ == "__main__":\n    hello()')
            
            # Create an image file if needed for vision tests
            image_file = Path(temp_dir) / "test_image.jpg"
            # Use a small sample image or create one
            try:
                # Try to download a sample image
                import urllib.request
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/llava/examples/rocket.jpg",
                    image_file
                )
            except:
                # Fallback: create a simple image
                try:
                    from PIL import Image
                    img = Image.new('RGB', (100, 100), color='red')
                    img.save(image_file)
                except:
                    # If PIL is not available, just create an empty file
                    image_file.touch()
            
            yield {
                "text": text_file,
                "markdown": md_file,
                "json": json_file,
                "python": py_file,
                "image": image_file,
                "temp_dir": temp_dir
            }
    
    def test_model_loading(self, text_llm):
        """Test that the model loads successfully."""
        assert text_llm._is_loaded, "Model should be loaded"
        assert text_llm._model is not None, "Model should not be None"
        assert text_llm._tokenizer is not None, "Tokenizer should not be None"
    
    def test_basic_generation(self, text_llm):
        """Test basic text generation."""
        prompt = "Hello, my name is"
        response = text_llm.generate(prompt)
        
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
        
        print(f"\nGenerated response: {response.content}")
    
    def test_system_prompt(self, text_llm):
        """Test generation with a system prompt."""
        system_prompt = "You are a helpful AI assistant that speaks like a pirate."
        user_prompt = "Tell me about the weather."
        
        response = text_llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nGenerated response with system prompt: {response.content}")
    
    def test_streaming_generation(self, text_llm):
        """Test streaming text generation."""
        prompt = "Count from 1 to 5."
        
        # Collect streaming chunks
        chunks = []
        for chunk in text_llm.generate(prompt, stream=True):
            chunks.append(chunk)
            print(f"Chunk: {chunk.content}", end="", flush=True)
            
        # Check that we got chunks
        assert len(chunks) > 0, "Should return streaming chunks"
        
        # Check the final chunk
        final_chunk = chunks[-1]
        assert hasattr(final_chunk, 'content'), "Final chunk should have content attribute"
        assert final_chunk.content is not None, "Final chunk should have content"
        assert len(final_chunk.content) > 0, "Final chunk content should not be empty"
        
        print(f"\nFinal chunk content: {final_chunk.content}")
    
    @pytest.mark.asyncio
    async def test_async_generation(self, text_llm):
        """Test async text generation."""
        prompt = "What is the capital of France?"
        
        # Test async generation
        response = await text_llm.generate_async(prompt)
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nAsync generated response: {response.content}")
    
    @pytest.mark.asyncio
    async def test_async_streaming(self, text_llm):
        """Test async streaming text generation."""
        prompt = "List three colors."
        
        # Collect streaming chunks
        chunks = []
        async for chunk in await text_llm.generate_async(prompt, stream=True):
            chunks.append(chunk)
            print(f"Async chunk: {chunk.content}", end="", flush=True)
            
        # Check that we got chunks
        assert len(chunks) > 0, "Should return streaming chunks"
        
        # Check the final chunk
        final_chunk = chunks[-1]
        assert hasattr(final_chunk, 'content'), "Final chunk should have content attribute"
        assert final_chunk.content is not None, "Final chunk should have content"
        assert len(final_chunk.content) > 0, "Final chunk content should not be empty"
        
        print(f"\nFinal async chunk content: {final_chunk.content}")
    
    def test_text_file_processing(self, text_llm, test_files):
        """Test processing a text file."""
        prompt = "Summarize the content of the file:"
        
        # Generate with the file
        response = text_llm.generate(prompt, files=[test_files["text"]])
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nResponse with text file: {response.content}")
    
    def test_markdown_file_processing(self, text_llm, test_files):
        """Test processing a markdown file."""
        prompt = "Explain the structure of this markdown file:"
        
        # Generate with the file
        response = text_llm.generate(prompt, files=[test_files["markdown"]])
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nResponse with markdown file: {response.content}")
    
    def test_json_file_processing(self, text_llm, test_files):
        """Test processing a JSON file."""
        prompt = "What values are in this JSON file?"
        
        # Generate with the file
        response = text_llm.generate(prompt, files=[test_files["json"]])
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nResponse with JSON file: {response.content}")
    
    def test_python_file_processing(self, text_llm, test_files):
        """Test processing a Python file."""
        prompt = "What does this Python code do?"
        
        # Generate with the file
        response = text_llm.generate(prompt, files=[test_files["python"]])
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nResponse with Python file: {response.content}")
    
    def test_multiple_file_processing(self, text_llm, test_files):
        """Test processing multiple files together."""
        prompt = "Compare the content of these files:"
        
        # Generate with multiple files
        response = text_llm.generate(prompt, files=[
            test_files["text"],
            test_files["markdown"],
            test_files["json"]
        ])
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nResponse with multiple files: {response.content}")
    
    def test_model_caching(self):
        """Test that models are properly cached."""
        # Clear the cache first
        from abstractllm.providers.mlx_provider import MLXProvider
        MLXProvider.clear_model_cache()
        
        # Create a provider and load the model
        llm1 = create_llm("mlx", model=TEXT_MODEL)
        llm1.generate("Hello")  # This will load the model
        
        # Check that the model is in the cache
        assert TEXT_MODEL in MLXProvider._model_cache, "Model should be in cache"
        
        # Create another provider with the same model
        start_time = time.time()
        llm2 = create_llm("mlx", model=TEXT_MODEL)
        llm2.generate("World")  # This should use the cached model
        load_time = time.time() - start_time
        
        # The second load should be faster than initial load
        print(f"\nSecond load time: {load_time:.2f} seconds")
        
        # We're not testing exact timing as it can vary by machine,
        # just that the model is properly retrieved from cache
        assert TEXT_MODEL in MLXProvider._model_cache, "Model should still be in cache after second use"
    
    def test_capabilities(self, text_llm):
        """Test the capabilities reporting."""
        capabilities = text_llm.get_capabilities()
        
        # Check basic capabilities
        assert capabilities.get(ModelCapability.STREAMING) is True, "Should support streaming"
        assert capabilities.get(ModelCapability.SYSTEM_PROMPT) is True, "Should support system prompts"
        assert capabilities.get(ModelCapability.ASYNC) is True, "Should support async"
        assert capabilities.get(ModelCapability.MAX_TOKENS) > 0, "Should report max tokens"
        
        # Check unsupported capabilities
        assert capabilities.get(ModelCapability.FUNCTION_CALLING) is False, "Should not support function calling"
        assert capabilities.get(ModelCapability.TOOL_USE) is False, "Should not support tool use"
        
        print(f"\nReported capabilities: {capabilities}")
        
    def test_vision_capabilities(self, vision_llm, test_files):
        """Test vision capabilities with an image."""
        # Skip if PIL is not available
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available for vision tests")
            
        prompt = "Describe this image:"
        
        # Generate with the image file
        response = vision_llm.generate(prompt, files=[test_files["image"]])
        
        # Check that we got a response
        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Response should have content attribute"
        assert response.content is not None, "Response should have content"
        assert len(response.content) > 0, "Response content should not be empty"
        
        print(f"\nResponse with image: {response.content}") 