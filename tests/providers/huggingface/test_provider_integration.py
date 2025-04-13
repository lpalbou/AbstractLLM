"""Integration tests for the HuggingFace provider."""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil
import os

from abstractllm.providers.huggingface.provider import HuggingFaceProvider
from abstractllm.providers.huggingface.model_types import ModelArchitecture
from abstractllm.exceptions import ResourceError
from abstractllm.media.image import ImageInput
from abstractllm.media.text import TextInput
from abstractllm.enums import ModelParameter

# Skip tests if torch not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)

@pytest.fixture(scope="module")
def test_image(temp_dir):
    """Create a test image file."""
    from PIL import Image
    import numpy as np
    
    # Create a simple test image
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    path = os.path.join(temp_dir, "test.jpg")
    img.save(path)
    return path

@pytest.fixture(scope="module")
def test_text(temp_dir):
    """Create a test text file."""
    path = os.path.join(temp_dir, "test.txt")
    with open(path, "w") as f:
        f.write("This is a test document.")
    return path

class TestTextGeneration:
    """Test text generation capabilities."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the provider for each test."""
        self.provider = HuggingFaceProvider({
            ModelParameter.MODEL: "microsoft/phi-2",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 100,
            "device_map": "auto"
        })
        yield
        self.provider.cleanup()
    
    def test_basic_generation(self):
        """Test basic text generation."""
        response = self.provider.generate("Hello, how are you?")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_streaming(self):
        """Test streaming generation."""
        chunks = []
        for chunk in self.provider.generate(
            "Tell me a story.",
            stream=True
        ):
            chunks.append(chunk)
            
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)
    
    def test_system_prompt(self):
        """Test generation with system prompt."""
        response = self.provider.generate(
            "What is 2+2?",
            system_prompt="You are a math tutor."
        )
        assert isinstance(response, str)
        assert len(response) > 0

class TestVisionCapabilities:
    """Test vision model capabilities."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the provider for each test."""
        self.provider = HuggingFaceProvider({
            ModelParameter.MODEL: "Salesforce/blip-image-captioning-base",
            "device_map": "auto"
        })
        yield
        self.provider.cleanup()
    
    def test_image_captioning(self, test_image):
        """Test image captioning."""
        response = self.provider.generate(
            "Describe this image.",
            files=[test_image]
        )
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_multiple_images(self, test_image):
        """Test handling multiple images."""
        with pytest.raises(Exception):
            self.provider.generate(
                "Describe these images.",
                files=[test_image, test_image]
            )

class TestDocumentQA:
    """Test document question answering capabilities."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the provider for each test."""
        self.provider = HuggingFaceProvider({
            ModelParameter.MODEL: "microsoft/layoutlmv3-base",
            "device_map": "auto"
        })
        yield
        self.provider.cleanup()
    
    def test_document_qa(self, test_text):
        """Test document question answering."""
        response = self.provider.generate(
            "What is mentioned in the document?",
            files=[test_text]
        )
        assert isinstance(response, str)
        assert len(response) > 0

class TestResourceManagement:
    """Test resource management capabilities."""
    
    def test_gpu_memory_limit(self):
        """Test GPU memory limit enforcement."""
        provider = HuggingFaceProvider({
            ModelParameter.MODEL: "microsoft/phi-2",
            "device_map": "cuda",
            "max_memory": {
                "cuda:0": "1MB"  # Unreasonably small
            }
        })
        
        with pytest.raises(ResourceError) as exc_info:
            provider.load_model()
        assert "Insufficient GPU memory" in str(exc_info.value)
    
    def test_cpu_memory_limit(self):
        """Test CPU memory limit enforcement."""
        provider = HuggingFaceProvider({
            ModelParameter.MODEL: "microsoft/phi-2",
            "device_map": "cpu",
            "max_memory": {
                "cpu": "1TB"  # Unreasonably large
            }
        })
        
        with pytest.raises(ResourceError) as exc_info:
            provider.load_model()
        assert "Insufficient CPU memory" in str(exc_info.value)
    
    def test_cleanup(self):
        """Test resource cleanup."""
        provider = HuggingFaceProvider({
            ModelParameter.MODEL: "microsoft/phi-2",
            "device_map": "auto"
        })
        
        # Generate some text to load the model
        provider.generate("Hello")
        assert provider._pipeline is not None
        
        # Clean up
        provider.cleanup()
        assert provider._pipeline is None
        
        # Should be able to generate again
        response = provider.generate("Hello again")
        assert isinstance(response, str)

class TestEndToEndWorkflows:
    """Test end-to-end workflows."""
    
    def test_text_generation_workflow(self):
        """Test complete text generation workflow."""
        provider = HuggingFaceProvider({
            ModelParameter.MODEL: "microsoft/phi-2",
            "device_map": "auto"
        })
        
        try:
            # Get model recommendations
            recs = provider.get_model_recommendations("text-generation")
            assert len(recs) > 0
            
            # Generate with system prompt
            response = provider.generate(
                "Write a haiku.",
                system_prompt="You are a poet."
            )
            assert isinstance(response, str)
            
            # Stream response
            chunks = list(provider.generate(
                "Tell me a story.",
                stream=True
            ))
            assert len(chunks) > 0
            
            # Check capabilities
            caps = provider.get_capabilities()
            assert "text_generation" in str(caps)
            
        finally:
            provider.cleanup()
    
    def test_multimodal_workflow(self, test_image, test_text):
        """Test multimodal workflow."""
        provider = HuggingFaceProvider({
            ModelParameter.MODEL: "Salesforce/blip-image-captioning-base",
            "device_map": "auto"
        })
        
        try:
            # Image captioning
            response = provider.generate(
                "Describe this image in detail.",
                files=[test_image]
            )
            assert isinstance(response, str)
            
            # Get capabilities
            caps = provider.get_capabilities()
            assert "image_to_text" in str(caps)
            
        finally:
            provider.cleanup()
    
    @pytest.mark.asyncio
    async def test_async_workflow(self):
        """Test asynchronous workflow."""
        provider = HuggingFaceProvider({
            ModelParameter.MODEL: "microsoft/phi-2",
            "device_map": "auto"
        })
        
        try:
            # Basic async generation
            response = await provider.generate_async(
                "Hello, how are you?"
            )
            assert isinstance(response, str)
            
            # Async streaming
            chunks = []
            async for chunk in provider.generate_async(
                "Tell me a story.",
                stream=True
            ):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            
        finally:
            provider.cleanup() 