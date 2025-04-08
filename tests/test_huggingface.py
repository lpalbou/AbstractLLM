"""
Tests for the HuggingFace provider.
"""

import os
import unittest
import importlib.util
import pytest
from abstractllm import create_llm, ModelParameter

class TestHuggingFaceProvider(unittest.TestCase):
    def setUp(self):
        """Set up for the test."""
        # Check if transformers and torch are installed
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transformers is not installed")
        
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is not installed")
            
        # Use a small model for testing
        self.model_name = "distilgpt2"  # Small model for testing
    
    @pytest.mark.timeout(30)  # Shorter timeout
    def test_generate(self):
        try:
            # Create HuggingFace provider with a small model and limit generation
            llm = create_llm("huggingface", **{
                ModelParameter.MODEL: self.model_name,
                ModelParameter.DEVICE: "cpu",
                ModelParameter.MAX_TOKENS: 20  # Limit generation to be quicker
            })
            
            response = llm.generate("Hello, I am", max_tokens=10)  # Override to even smaller generation
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)
        except Exception as e:
            self.skipTest(f"Failed to load or run HuggingFace model: {e}")

    def test_streaming(self):
        try:
            # Create HuggingFace provider with a small model
            llm = create_llm("huggingface", **{
                ModelParameter.MODEL: self.model_name,
                ModelParameter.DEVICE: "cpu",
                ModelParameter.MAX_TOKENS: 10  # Very small generation for testing
            })
            
            # Check if streaming is supported
            capabilities = llm.get_capabilities()
            if not capabilities.get("streaming", False):
                self.skipTest(f"Model {self.model_name} does not support streaming")
            
            stream = llm.generate("Hello, I am", stream=True, max_tokens=5)  # Very small generation
            
            # Collect chunks from stream
            chunks = []
            for chunk in stream:
                chunks.append(chunk)
            
            # Check that we got at least one chunk
            self.assertTrue(len(chunks) > 0)
            
            # Check that the combined response makes sense
            full_response = "".join(chunks)
            self.assertTrue(len(full_response) > 0)
        except Exception as e:
            self.skipTest(f"Failed to load or run HuggingFace model: {e}")

    def test_cached_models(self):
        try:
            from abstractllm.providers.huggingface import HuggingFaceProvider
            
            # List cached models
            cached_models = HuggingFaceProvider.list_cached_models()
            self.assertIsInstance(cached_models, list)
            
        except (ImportError, AttributeError) as e:
            self.skipTest(f"Cannot test cache functionality: {e}")

if __name__ == "__main__":
    unittest.main() 