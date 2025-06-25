#!/usr/bin/env python3
"""Test script to verify Ollama tool support is working correctly."""

import logging
from abstractllm import create_llm
from abstractllm.session import Session

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Define test tools
def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    return f"Contents of {file_path}: This is a test file."

def list_files(directory_path: str, recursive: bool = False) -> str:
    """List files in a directory."""
    return f"Files in {directory_path} (recursive={recursive}): file1.txt, file2.py, file3.md"

# Create Ollama provider
print("Creating Ollama provider...")
llm = create_llm(
    provider="ollama",
    model="qwen3:30b-a3b-q4_K_M",
    temperature=0.7
)

# Test 1: Direct tool use
print("\n=== Test 1: Direct Tool Use ===")
response = llm.generate(
    prompt="Please list files in /tmp directory",
    tools=[list_files]
)
print(f"Response type: {type(response)}")
print(f"Response: {response}")

# Test 2: Session with tools
print("\n\n=== Test 2: Session with Tools ===")
session = Session(provider=llm, tools=[read_file, list_files])
response = session.generate("Please read the file /Users/albou/projects/mnemosyne/memory/Core/Suggestions/Awareness_Selection_Protocol.md")
print(f"Response: {response}")