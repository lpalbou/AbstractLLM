#!/usr/bin/env python3
"""Test to reproduce the exact issue the user reported."""

import logging
from abstractllm import create_llm
from abstractllm.session import Session

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Create Ollama provider
print("Creating Ollama provider...")
llm = create_llm(
    provider="ollama",
    model="qwen3:30b-a3b-q4_K_M",
    temperature=0.7
)

# Create session
session = Session(provider=llm)

# Test conversation
print("\n=== Testing conversation memory ===")

# First message
print("\nUser: My favorite fruit is banana.")
response1 = session.generate("My favorite fruit is banana.")
print(f"Assistant: {response1}")

# Second message - asking about last message
print("\nUser: what was my last message ?")
response2 = session.generate("what was my last message ?")
print(f"Assistant: {response2}")

# Third message - asking about favorite fruit
print("\nUser: What is my favorite fruit?")
response3 = session.generate("What is my favorite fruit?")
print(f"Assistant: {response3}")

# Check the actual messages in the session
print("\n=== Session messages ===")
for i, msg in enumerate(session.messages):
    print(f"{i}: {msg.role}: {msg.content[:100]}...")