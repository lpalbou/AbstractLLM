#!/usr/bin/env python3
"""
Debug the Session tool call conversation format to understand why infinite loops occur.
"""

from abstractllm import create_llm
from abstractllm.session import Session
import time
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def main():
    """Debug Session tool calling to understand the infinite loop."""
    
    print("🔍 Debugging Session Tool Call Conversation Format")
    print("=" * 60)
    
    # Create MLX provider and session
    provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
    session = Session(
        system_prompt="You are a helpful assistant.",
        provider=provider,
        tools=[calculate_math]
    )
    
    print("\n🔧 Testing with max_tool_calls=2 to see conversation structure")
    start_time = time.time()
    
    # Generate with debug output
    response = session.generate(
        prompt="What is 25 * 4 + 7?",
        max_tool_calls=2  # Limit to 2 to see the conversation structure
    )
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Time: {elapsed:.1f}s")
    
    print("\n📜 Final Conversation History:")
    for i, msg in enumerate(session.messages):
        print(f"  {i+1}. Role: {msg.role}")
        print(f"     Content: {msg.content[:100]}...")
        if hasattr(msg, 'tool_results') and msg.tool_results:
            print(f"     Tool Results: {msg.tool_results}")
        print()
    
    print("\n📨 Messages Formatted for MLX Provider:")
    formatted_messages = session.get_messages_for_provider("mlx")
    for i, msg in enumerate(formatted_messages):
        print(f"  {i+1}. {msg}")
        print()
    
    if hasattr(response, 'content'):
        print(f"\n📝 Final Response: {response.content[:200]}...")
    else:
        print(f"\n📝 Final Response: {str(response)[:200]}...")
    
    print("\n" + "=" * 60)
    print("🎯 Analysis: Check if conversation format prevents tool loops")
    
if __name__ == "__main__":
    main() 