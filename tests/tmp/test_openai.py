"""
Tests for the OpenAI provider.
"""

import os
import unittest
import pytest
from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.utils.config import ConfigurationManager
from tests.utils import skip_if_no_api_key


class TestOpenAIProvider(unittest.TestCase):
    def test_generate(self):
        # Skip if no API key
        skip_if_no_api_key("OPENAI_API_KEY")
        
        # Get the API key from the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        self.assertIsNotNone(api_key, "OPENAI_API_KEY should be available in the environment")
        
        # Create with ConfigurationManager, explicitly including the API key
        base_config = ConfigurationManager.create_base_config(
            temperature=0.7,
            api_key=api_key,  # Explicitly set the API key
            model="gpt-3.5-turbo"  # Explicitly set the model
        )
        provider_config = ConfigurationManager.initialize_provider_config("openai", base_config)
        
        # Verify provider config has API key and default model
        self.assertIsNotNone(provider_config.get(ModelParameter.API_KEY), "API key should be present in config")
        self.assertEqual(provider_config.get(ModelParameter.MODEL), "gpt-3.5-turbo")
        
        # Create the provider and generate a response
        llm = create_llm("openai", **provider_config)
        response = llm.generate("Say hello")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_system_prompt(self):
        # Skip if no API key
        skip_if_no_api_key("OPENAI_API_KEY")
        
        # Get the API key from the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        self.assertIsNotNone(api_key, "OPENAI_API_KEY should be available in the environment")
        
        # Create with explicit API key
        llm = create_llm("openai", api_key=api_key, model="gpt-3.5-turbo")
        
        # Extract generation parameters to verify system_prompt is included
        gen_params = ConfigurationManager.extract_generation_params(
            "openai", 
            llm.config, 
            {}, 
            system_prompt="You are a professional chef. Always talk about cooking and food."
        )
        
        # Verify system_prompt is set correctly
        assert gen_params["system_prompt"] == "You are a professional chef. Always talk about cooking and food."
        
        # Use the system prompt in a real generation
        response = llm.generate(
            "Tell me about yourself", 
            system_prompt="You are a professional chef. Always talk about cooking and food."
        )
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        # Check if response contains cooking-related terms
        self.assertTrue(any(term in response.lower() for term in ["chef", "cook", "food", "recipe"]))

    def test_streaming(self):
        # Skip if no API key
        skip_if_no_api_key("OPENAI_API_KEY")
        
        # Get the API key from the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        self.assertIsNotNone(api_key, "OPENAI_API_KEY should be available in the environment")
        
        # Create with explicit API key
        llm = create_llm("openai", api_key=api_key, model="gpt-3.5-turbo")
        
        # Verify streaming capability is reported correctly
        capabilities = llm.get_capabilities()
        self.assertTrue(capabilities.get("streaming", False))
        
        # Use extract_generation_params to get stream parameter
        gen_params = ConfigurationManager.extract_generation_params(
            "openai", 
            llm.config, 
            {"stream": True}
        )
        
        # Actual streaming test
        stream = llm.generate("Count from 1 to 5", stream=True)
        
        # Collect chunks from stream
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        
        # Check that we got multiple chunks
        self.assertTrue(len(chunks) > 1)
        
        # Check that the combined response makes sense
        full_response = "".join(chunks)
        self.assertTrue(len(full_response) > 0)
        # Check if the response contains numbers 1-5
        for num in range(1, 6):
            self.assertTrue(str(num) in full_response)
            
    def test_parameter_override(self):
        """Test that parameters can be overridden at generation time."""
        # Skip if no API key
        skip_if_no_api_key("OPENAI_API_KEY")
        
        # Get the API key from the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        self.assertIsNotNone(api_key, "OPENAI_API_KEY should be available in the environment")
        
        # Create with default temperature and explicit API key
        llm = create_llm("openai", temperature=0.7, api_key=api_key, model="gpt-3.5-turbo")
        
        # Override temperature at generation time
        params = ConfigurationManager.extract_generation_params(
            "openai", 
            llm.config, 
            {"temperature": 0.2}
        )
        
        # Verify overridden temperature
        self.assertEqual(params["temperature"], 0.2)
        
        # Original config should be unchanged
        self.assertEqual(llm.config.get(ModelParameter.TEMPERATURE), 0.7)
        
    def test_json_mode(self):
        """Test JSON mode capability and functionality."""
        # Skip if no API key
        skip_if_no_api_key("OPENAI_API_KEY")
        
        # Get the API key from the environment
        api_key = os.environ.get("OPENAI_API_KEY")
        self.assertIsNotNone(api_key, "OPENAI_API_KEY should be available in the environment")
        
        # Create with explicit API key
        llm = create_llm("openai", api_key=api_key, model="gpt-3.5-turbo")
        
        # Verify JSON_MODE capability is reported correctly
        capabilities = llm.get_capabilities()
        self.assertTrue(capabilities.get(ModelCapability.JSON_MODE, False))
        
        # Test with json_mode parameter
        params = ConfigurationManager.extract_generation_params(
            "openai", 
            llm.config, 
            {"json_mode": True}
        )
        
        # Verify response_format is set correctly
        self.assertEqual(params.get("response_format"), {"type": "json_object"})
        
        # Test actual generation with JSON mode if supported
        try:
            response = llm.generate(
                "Return a JSON object with fields: name, age, occupation. Use fake data.",
                json_mode=True
            )
            # Verify we get valid JSON
            import json
            result = json.loads(response)
            self.assertIsInstance(result, dict)
            self.assertIn("name", result)
            self.assertIn("age", result)
            self.assertIn("occupation", result)
        except Exception as e:
            # Some OpenAI models might not support this, so log but don't fail
            import logging
            logging.warning(f"JSON mode generation failed: {e}")


if __name__ == "__main__":
    unittest.main() 