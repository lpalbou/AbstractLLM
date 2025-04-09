"""
Tests for the Ollama provider.
"""

import os
import unittest
import requests
from abstractllm import create_llm, ModelParameter
from abstractllm.utils.config import ConfigurationManager
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
            
            # Create a base configuration using ConfigurationManager
            self.base_config = ConfigurationManager.create_base_config()
            
            # Initialize provider-specific configuration
            self.provider_config = ConfigurationManager.initialize_provider_config("ollama", self.base_config)
            
            # Override with the detected model
            self.provider_config[ModelParameter.MODEL] = self.model_name
            
            # Verify the Ollama-specific configuration
            self.assertEqual(self.provider_config.get(ModelParameter.BASE_URL), "http://localhost:11434")
            
        except Exception:
            self.skipTest("Ollama API not accessible or other error")
            self.model_name = None  # Won't be used if skipped
    
    def test_generate(self):
        # Create Ollama provider with the first available model
        llm = create_llm("ollama", **self.provider_config)
        
        # Verify that generate uses the correct parameters
        gen_params = ConfigurationManager.extract_generation_params(
            "ollama", 
            llm.config, 
            {}
        )
        
        # Check that the model name is correct
        self.assertEqual(gen_params["model"], self.model_name)
        
        # Test actual generation
        response = llm.generate("Say hello")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_streaming(self):
        # Create Ollama provider with the first available model
        llm = create_llm("ollama", **self.provider_config)
        
        # Verify streaming capability is reported correctly
        capabilities = llm.get_capabilities()
        self.assertTrue(capabilities.get("streaming", False))
        
        # Use extract_generation_params to verify stream parameter handling
        gen_params = ConfigurationManager.extract_generation_params(
            "ollama", 
            llm.config, 
            {"stream": True}
        )
        
        # Test actual streaming
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
        llm = create_llm("ollama", **self.provider_config)
        
        # Check if system prompts are supported
        capabilities = llm.get_capabilities()
        if not capabilities.get("supports_system_prompt", False):
            self.skipTest(f"Model {self.model_name} does not support system prompts")
        
        # Test system prompt parameter extraction
        system_prompt = "You are a professional chef. Always talk about cooking and food."
        gen_params = ConfigurationManager.extract_generation_params(
            "ollama", 
            llm.config, 
            {}, 
            system_prompt=system_prompt
        )
        
        # Verify system_prompt is set correctly
        self.assertEqual(gen_params["system_prompt"], system_prompt)
        
        # Test actual generation with system prompt
        response = llm.generate(
            "Tell me about yourself", 
            system_prompt=system_prompt
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
    def test_parameter_override(self):
        """Test that parameters can be overridden at generation time."""
        # Create Ollama provider with default temperature
        config = self.provider_config.copy()
        config[ModelParameter.TEMPERATURE] = 0.7
        
        llm = create_llm("ollama", **config)
        
        # Override temperature at generation time
        params = ConfigurationManager.extract_generation_params(
            "ollama", 
            llm.config, 
            {"temperature": 0.2}
        )
        
        # Verify overridden temperature
        self.assertEqual(params["temperature"], 0.2)
        
        # Original config should be unchanged
        self.assertEqual(llm.config.get(ModelParameter.TEMPERATURE), 0.7)
        
        # Test actual generation with overridden parameter
        response = llm.generate("Say hello", temperature=0.2)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == "__main__":
    unittest.main() 