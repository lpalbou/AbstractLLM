#!/usr/bin/env python3
"""
Debug script to understand what type of response is returned from session.generate_with_tools()
"""

from abstractllm import create_llm
from abstractllm.session import Session
import os
import sys

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Initialize provider and session
provider = create_llm("ollama", model="cogito")
session = Session(
    system_prompt="You are a helpful assistant.",
    provider=provider,
    tools=[read_file]
)

# Get a simple query from command line or use default
query = sys.argv[1] if len(sys.argv) > 1 else "Who are you?"

print(f"\nSending query: {query}")
print("-" * 50)

# Generate response
response = session.generate_with_tools(prompt=query)

# Debug response type
print(f"Response type: {type(response)}")
print(f"Response has 'content' attribute: {hasattr(response, 'content')}")
print(f"Response is string: {isinstance(response, str)}")
print("-" * 50)

# Print raw response
print("Raw response:")
print(response)
print("-" * 50)

# Try to access content attribute if present
if hasattr(response, 'content'):
    print("Response.content:")
    print(response.content)
else:
    print("Response has no content attribute") 