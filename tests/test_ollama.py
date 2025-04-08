"""
Tests for the Ollama provider.
"""

import os
import unittest
import requests
from abstractllm import create_llm, ModelParameter
from tests.utils import skip_if_no_api_key

class TestOllamaProvider(unittest.TestCase):
    def setUp(self):
        """Set up for the test."""
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                self.skipTest("Ollama API not accessible")
                
            # Check if at least one model is available
            models = response.json().get("models", [])
            if not models:
                self.skipTest("No Ollama models available")
                
            # Use the first available model
            self.model_name = models[0]["name"]
        except Exception:
            self.skipTest("Ollama API not accessible or other error")
            self.model_name = None  # Won't be used if skipped
    
    def test_generate(self):
        # Create Ollama provider with the first available model
        llm = create_llm("ollama", **{
            ModelParameter.BASE_URL: "http://localhost:11434",
            ModelParameter.MODEL: self.model_name
        })
        
        response = llm.generate("Say hello")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_streaming(self):
        # Create Ollama provider with the first available model
        llm = create_llm("ollama", **{
            ModelParameter.BASE_URL: "http://localhost:11434",
            ModelParameter.MODEL: self.model_name
        })
        
        stream = llm.generate("Count from 1 to 5", stream=True)
        
        # Collect chunks from stream
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        
        # Check that we got at least one chunk
        self.assertTrue(len(chunks) > 0)
        
        # Check that the combined response makes sense
        full_response = "".join(chunks)
        self.assertTrue(len(full_response) > 0)

    def test_system_prompt_if_supported(self):
        # Create Ollama provider with the first available model
        llm = create_llm("ollama", **{
            ModelParameter.BASE_URL: "http://localhost:11434",
            ModelParameter.MODEL: self.model_name
        })
        
        # Check if system prompts are supported
        capabilities = llm.get_capabilities()
        if not capabilities.get("supports_system_prompt", False):
            self.skipTest(f"Model {self.model_name} does not support system prompts")
            
        response = llm.generate(
            "Tell me about yourself", 
            system_prompt="You are a professional chef. Always talk about cooking and food."
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == "__main__":
    unittest.main() 