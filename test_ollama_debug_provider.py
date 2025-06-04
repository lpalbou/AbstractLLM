#!/usr/bin/env python3
"""
Debug the Ollama provider to see exactly where the tool call response is lost.
"""

from abstractllm import create_llm
import logging

# Enable all debug logging
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
    """Debug Ollama provider generate method."""
    
    print("🔍 Debugging Ollama Provider Generate Method")
    print("=" * 60)
    
    provider = create_llm("ollama", model="cogito:8b")
    
    # Add detailed logging to see exactly what's happening
    print("\n🧪 Calling provider.generate with tools...")
    
    response = provider.generate(
        prompt="What is 25 * 4 + 7?",
        tools=[calculate_math],
        max_tokens=500
    )
    
    print(f"\n📝 Final response type: {type(response)}")
    print(f"📝 Final response: {response}")
    
    # If we can access the internal methods, let's check them
    if hasattr(provider, '_check_for_tool_calls'):
        print(f"\n🔧 Provider has _check_for_tool_calls method")
        
    if hasattr(provider, '_extract_tool_calls'):
        print(f"🔧 Provider has _extract_tool_calls method")

if __name__ == "__main__":
    main() 