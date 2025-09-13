#!/usr/bin/env python3
"""Test Ollama logging with tools to ensure it captures all details."""

from abstractllm import create_llm
from abstractllm.tools import register
from abstractllm.utils.logging import configure_logging

# Configure logging to capture everything
configure_logging(
    log_dir="logs",
    console_level=None,  # Disable console to focus on files
    file_level=10  # DEBUG level
)

# Define a simple tool
@register
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny and 72°F."

# Create Ollama provider
llm = create_llm("ollama", model="qwen3:30b-a3b-q4_K_M")

# Test with tools and system prompt
response = llm.generate(
    prompt="What's the weather like in Paris?",
    system_prompt="You are a helpful weather assistant. Always use the get_weather tool when asked about weather.",
    tools=[get_weather]
)

print(f"Response: {response.content}")
if hasattr(response, 'tool_calls') and response.tool_calls:
    print(f"Tool calls: {response.tool_calls}")

print("\nCheck the logs directory for the detailed interaction log.")