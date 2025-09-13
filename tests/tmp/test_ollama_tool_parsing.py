#!/usr/bin/env python3
"""Test to debug why Ollama is not executing tools properly."""

import logging
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.tools.common_tools import list_files

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Filter out some noisy loggers
logging.getLogger("urllib3").setLevel(logging.INFO)

# Create Ollama provider
print("Creating Ollama provider...")
llm = create_llm(
    provider="ollama",
    model="qwen3:30b-a3b-q4_K_M",
    temperature=0.7
)

# Test 1: Direct tool use without Session
print("\n=== Test 1: Direct provider tool use ===")
response = llm.generate(
    prompt="List all files in the current directory ./",
    tools=[list_files]
)
print(f"Response type: {type(response)}")
print(f"Has tool_calls attr: {hasattr(response, 'tool_calls')}")
print(f"Has has_tool_calls method: {hasattr(response, 'has_tool_calls')}")
if hasattr(response, 'has_tool_calls'):
    print(f"has_tool_calls(): {response.has_tool_calls()}")
if hasattr(response, 'content'):
    print(f"Content: {response.content[:200]}...")
else:
    print(f"Response: {response[:200]}...")

# Test 2: Session with strong tool prompt
print("\n\n=== Test 2: Session with strong tool prompt ===")
session = Session(
    provider=llm,
    tools=[list_files],
    system_prompt="""You are a tool-using assistant. When asked to list files or perform any action, you MUST use the appropriate tool.

CRITICAL: You have access to the list_files tool. When asked to list files, you MUST call it using the exact format:
<|tool_call|>
{"name": "list_files", "arguments": {"directory_path": "./"}}
</|tool_call|>

DO NOT ask for clarification. DO NOT describe what you would do. EXECUTE THE TOOL."""
)

response = session.generate("List all files in the current directory")
print(f"Response: {response}")