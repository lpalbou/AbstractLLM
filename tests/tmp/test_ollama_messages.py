#!/usr/bin/env python3
"""Test to verify if Ollama is actually receiving messages/chat history."""

import json
import logging
from abstractllm import create_llm

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Intercept the actual HTTP request to see what's being sent
import requests
original_post = requests.post

def intercepted_post(url, **kwargs):
    """Intercept and log the actual request being sent to Ollama."""
    print("\n" + "="*80)
    print(f"INTERCEPTED REQUEST TO: {url}")
    print("="*80)
    
    if 'json' in kwargs:
        request_data = kwargs['json']
        print("REQUEST DATA:")
        print(json.dumps(request_data, indent=2))
        
        # Check messages
        if 'messages' in request_data:
            print(f"\nMESSAGES COUNT: {len(request_data['messages'])}")
            for i, msg in enumerate(request_data['messages']):
                print(f"\nMessage {i}:")
                print(f"  Role: {msg.get('role')}")
                print(f"  Content: {msg.get('content')[:200]}..." if len(msg.get('content', '')) > 200 else f"  Content: {msg.get('content')}")
    
    print("="*80 + "\n")
    
    # Call the original function
    return original_post(url, **kwargs)

# Monkey patch requests.post
requests.post = intercepted_post

# Test 1: Simple generation without messages
print("\n\n" + "="*60)
print("TEST 1: Simple generation (no messages)")
print("="*60)

llm = create_llm("ollama", model="qwen3:30b-a3b-q4_K_M")
response = llm.generate("Hello, who are you?")
print(f"Response: {response}")

# Test 2: Generation with messages (conversation history)
print("\n\n" + "="*60)
print("TEST 2: Generation with messages (conversation history)")
print("="*60)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My favorite color is blue."},
    {"role": "assistant", "content": "That's nice! Blue is a beautiful color."},
    {"role": "user", "content": "What is my favorite color?"}
]

response = llm.generate(
    prompt="What is my favorite color?",
    messages=messages
)
print(f"Response: {response}")

# Test 3: Testing with Session (which uses messages internally)
print("\n\n" + "="*60)
print("TEST 3: Using Session (which should maintain conversation)")
print("="*60)

from abstractllm.session import Session

session = Session(provider=llm)
response1 = session.generate("My name is Bob.")
print(f"Response 1: {response1}")

response2 = session.generate("What is my name?")
print(f"Response 2: {response2}")

# Restore original requests.post
requests.post = original_post