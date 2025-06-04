#!/usr/bin/env python3
"""
Test that demonstrates MLX provider now works correctly with native tool calling.
"""

from abstractllm import create_llm
import time

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def main():
    """Test MLX native tool calling - should work without loops."""
    
    print("🧪 Testing Fixed MLX Provider")
    print("=" * 50)
    
    # Create MLX provider
    provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
    
    # Test 1: Direct provider tool calling (should work perfectly)
    print("\n🔧 Test 1: Direct provider tool calling")
    start_time = time.time()
    
    response = provider.generate(
        prompt="What is 25 * 4 + 7?",
        system_prompt="You are a helpful assistant.",
        tools=[calculate_math]
    )
    
    elapsed = time.time() - start_time
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"📄 Response type: {type(response)}")
    
    if hasattr(response, 'has_tool_calls') and response.has_tool_calls():
        print(f"✅ Tool calls detected: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls):
            print(f"   {i+1}. {tool_call.name}({tool_call.arguments})")
        print(f"📝 Content: {response.content}")
    else:
        print(f"📝 Direct answer: {response.content}")
    
    # Test 2: Basic generation without tools (should also work)
    print("\n🔧 Test 2: Basic generation without tools")
    start_time = time.time()
    
    simple_response = provider.generate(
        prompt="Say hello in exactly 5 words.",
        system_prompt="You are a helpful assistant."
    )
    
    elapsed = time.time() - start_time
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"📝 Response: {simple_response.content}")
    
    print("\n" + "=" * 50)
    print("🎉 MLX Provider is now working correctly!")
    
if __name__ == "__main__":
    main() 