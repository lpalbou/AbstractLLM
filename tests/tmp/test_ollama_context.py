#!/usr/bin/env python3
"""Test that Ollama is now using the correct context size."""

import logging
from abstractllm import create_llm
from abstractllm.architectures.detection import get_context_length

# Enable logging to see the context size being set
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(message)s'
)

# Check what context length should be for the model
model_name = "qwen3:30b-a3b-q4_K_M"
expected_context = get_context_length(model_name)
print(f"Expected context length for {model_name}: {expected_context}")

# Create Ollama provider
print("\nCreating Ollama provider...")
llm = create_llm(
    provider="ollama",
    model=model_name,
    temperature=0.7
)

# Test with a prompt - should see log message about context size
print("\nTesting generation (watch for context size log)...")
response = llm.generate("Hello, can you help me with a task?")
print(f"\nResponse: {response.content if hasattr(response, 'content') else response}")

# Test with tools to ensure they work with larger context
from abstractllm.tools.common_tools import list_files

print("\n\nTesting with tools (should work better with larger context)...")
response = llm.generate(
    prompt="List all Python files in the current directory",
    tools=[list_files]
)
print(f"Response type: {type(response)}")
print(f"Has tool calls: {response.has_tool_calls() if hasattr(response, 'has_tool_calls') else 'N/A'}")