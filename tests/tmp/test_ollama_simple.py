#!/usr/bin/env python3
"""Simple test to see if Ollama executes tools properly."""

import logging
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.tools.common_tools import list_files

# Enable INFO level logging only
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(message)s'
)

# Create provider and session
llm = create_llm("ollama", model="qwen3:30b-a3b-q4_K_M")
session = Session(provider=llm, tools=[list_files])

# Test simple prompts
print("\n=== Test 1: List files in ./ ===")
response = session.generate("List all files in ./")
print(f"Response: {response}")

print("\n\n=== Test 2: List Python files ===")
response = session.generate("List all Python files in the current directory")
print(f"Response: {response}")

print("\n\n=== Test 3: Check messages ===")
print(f"Total messages: {len(session.messages)}")
for i, msg in enumerate(session.messages[-6:]):  # Last 6 messages
    content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
    print(f"{i}: {msg.role}: {content}")