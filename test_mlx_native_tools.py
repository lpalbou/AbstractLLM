#!/usr/bin/env python3
"""
Test the new native MLX tool calling implementation.
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
    """Test native MLX tool calling."""
    
    print("🧪 Testing Native MLX Tool Calling")
    print("=" * 50)
    
    # Create MLX provider
    provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
    
    # Test direct tool calling
    print("\n🔧 Testing direct provider tool calling...")
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
    else:
        print(f"❌ No tool calls detected")
        print(f"📝 Content: {response.content[:200]}...")
    
    print("\n" + "=" * 50)
    
if __name__ == "__main__":
    main() 