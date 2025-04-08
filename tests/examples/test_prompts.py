"""
Test prompts with deterministic expected outcomes.

These prompts are designed to have clear, deterministic responses
that we can test against regardless of the LLM provider.
"""

# Simple factual prompts that should have consistent answers across providers
FACTUAL_PROMPTS = [
    {
        "prompt": "What is 2+2?",
        "expected_contains": ["4", "four"],  # The response should contain at least one of these
        "description": "Basic arithmetic"
    },
    {
        "prompt": "What is the capital of France?",
        "expected_contains": ["Paris"],
        "description": "Basic geography"
    },
    {
        "prompt": "Who wrote Romeo and Juliet?",
        "expected_contains": ["Shakespeare", "William Shakespeare"],
        "description": "Famous author"
    }
]

# Prompts for system prompt testing
SYSTEM_PROMPT_TESTS = [
    {
        "prompt": "Tell me about yourself.",
        "system_prompt": "You are a professional chef. Always talk about cooking and food. Never mention that you are an AI, artificial intelligence, language model, or assistant. If asked about your identity, only discuss your role as a chef.",
        "expected_contains": ["chef", "cook", "food", "recipe", "restaurant", "culinary"],
        "not_expected_contains": ["AI", "artificial intelligence", "language model"],
        "description": "Chef persona"
    },
    {
        "prompt": "Tell me about yourself.",
        "system_prompt": "You are a mathematician. Always use mathematical terms and concepts in your responses. Never mention that you are an AI, artificial intelligence, language model, or assistant. If asked about your identity, only discuss your role as a mathematician.",
        "expected_contains": ["math", "equation", "theorem", "calculation", "number", "formula"],
        "not_expected_contains": ["AI", "artificial intelligence", "language model"],
        "description": "Mathematician persona"
    }
]

# Prompts for streaming testing
STREAMING_TEST_PROMPTS = [
    {
        "prompt": "Count from 1 to 10.",
        "min_chunks": 3,  # Should receive at least this many chunks
        "expected_sequence": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "description": "Simple counting stream"
    },
    {
        "prompt": "Write a short poem about the ocean.",
        "min_chunks": 5,
        "description": "Creative content streaming"
    }
]

# Prompts for capability testing
FUNCTION_CALLING_PROMPT = {
    "prompt": "What's the weather in New York?",
    "description": "Basic function calling test - should trigger weather lookup"
}

VISION_PROMPT = {
    "prompt": "Look at this image and describe what you see.",
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg/800px-Eiffel_Tower_from_the_Tour_Montparnasse_1.jpg",
    "expected_contains": ["Eiffel", "tower", "Paris"],
    "description": "Basic vision capability test"
}

# Prompts for different parameter settings
PARAMETER_TEST_PROMPTS = [
    {
        "prompt": "Write a creative story about a dragon.",
        "parameters": {"temperature": 0.9},
        "description": "High temperature creativity test"
    },
    {
        "prompt": "What is photosynthesis?",
        "parameters": {"temperature": 0.1},
        "description": "Low temperature factual test"
    }
]

# Longer prompts to test max_tokens and truncation
LONG_CONTEXT_PROMPT = {
    "prompt": "Provide a detailed explanation of the theory of relativity, including both special and general relativity. "
              "Explain the key concepts, mathematical foundations, experimental evidence, and modern applications. "
              "Also discuss how Einstein developed these theories and their impact on physics.",
    "parameters": {"max_tokens": 500},
    "expected_tokens_range": (450, 600),  # Response should be in this token range
    "description": "Long-form response with token limit"
}

# Prompts for error handling testing
ERROR_TEST_PROMPTS = [
    {
        "prompt": "This is a valid prompt but we'll use invalid parameters.",
        "parameters": {"invalid_param": "value"},
        "expected_error": True,
        "description": "Invalid parameter test"
    }
] 