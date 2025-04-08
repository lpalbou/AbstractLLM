"""
Tests for the OpenAI provider.
"""

import os
import unittest
from abstractllm import create_llm
from tests.utils import skip_if_no_api_key

class TestOpenAIProvider(unittest.TestCase):
    def test_generate(self):
        # Skip if no API key
        skip_if_no_api_key("OPENAI_API_KEY")
        
        llm = create_llm("openai")
        response = llm.generate("Say hello")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_system_prompt(self):
        # Skip if no API key
        skip_if_no_api_key("OPENAI_API_KEY")
        
        llm = create_llm("openai")
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
        
        llm = create_llm("openai")
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

if __name__ == "__main__":
    unittest.main() 