#!/usr/bin/env python3
"""
Debug Ollama tool calling to see what broke.
"""

from abstractllm import create_llm
from abstractllm.session import Session
import time
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def main():
    """Test Ollama tool calling to see what's broken."""
    
    print("🔍 Testing Ollama Tool Calling")
    print("=" * 50)
    
    # Test 1: Direct provider tool calling
    print("\n🧪 Test 1: Direct Ollama Provider Tool Calling")
    provider = create_llm("ollama", model="cogito:8b")
    
    try:
        response = provider.generate(
            prompt="What is 25 * 4 + 7?",
            tools=[calculate_math],
            max_tokens=500
        )
        print(f"✅ Direct tool calling response: {response}")
        if hasattr(response, 'has_tool_calls') and response.has_tool_calls():
            print("🔧 Tool calls detected in direct response")
        else:
            print("❌ No tool calls detected in direct response")
    except Exception as e:
        print(f"❌ Direct tool calling failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Session-based tool calling
    print("\n🧪 Test 2: Session-based Ollama Tool Calling")
    session = Session(
        system_prompt="You are a helpful assistant. Use tools when needed.",
        provider=provider,
        tools=[calculate_math]
    )
    
    try:
        start_time = time.time()
        response = session.generate(
            prompt="What is 25 * 4 + 7?",
            max_tool_calls=2  # Limit to see what happens
        )
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📝 Session response: {response}")
        
        print("\n📜 Conversation History:")
        for i, msg in enumerate(session.messages):
            print(f"  {i+1}. Role: {msg.role}")
            print(f"     Content: {msg.content[:100]}...")
            if hasattr(msg, 'tool_results') and msg.tool_results:
                print(f"     Tool Results: {msg.tool_results}")
            print()
            
    except Exception as e:
        print(f"❌ Session tool calling failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check what messages are sent to Ollama
    print("\n🧪 Test 3: Check Ollama Message Format")
    try:
        formatted_messages = session.get_messages_for_provider("ollama")
        print("📨 Messages sent to Ollama:")
        for i, msg in enumerate(formatted_messages):
            print(f"  {i+1}. {msg}")
            print()
    except Exception as e:
        print(f"❌ Message formatting failed: {e}")

if __name__ == "__main__":
    main() 