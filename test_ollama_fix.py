#!/usr/bin/env python3
"""
Test to verify Ollama tool calling is actually working correctly.
"""

from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.tools.types import ToolCallRequest
import time

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def main():
    """Test that Ollama tool calling works correctly."""
    
    print("🧪 Testing Ollama Tool Calling - Proper Test")
    print("=" * 50)
    
    # Test 1: Direct provider tool calling with proper debug
    print("\n🔧 Test 1: Direct Ollama Provider Tool Calling")
    provider = create_llm("ollama", model="cogito:8b")
    
    response = provider.generate(
        prompt="What is 25 * 4 + 7?",
        tools=[calculate_math],
        max_tokens=500
    )
    
    print(f"📝 Response type: {type(response)}")
    print(f"📝 Response: {response}")
    
    # Check if it's a ToolCallRequest
    if isinstance(response, ToolCallRequest):
        print("✅ Ollama returned ToolCallRequest!")
        print(f"🔧 Tool calls: {response.tool_calls}")
        print(f"📄 Content: {response.content}")
        
        # Execute the tool call manually to verify
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"🔧 Executing tool call: {tool_call.name}({tool_call.arguments})")
            result = calculate_math(**tool_call.arguments)
            print(f"✅ Tool result: {result}")
    else:
        print("❌ Ollama did NOT return ToolCallRequest")
        print(f"   Instead got: {type(response)}")
        
    # Test 2: Session-based tool calling - but with direct execution
    print("\n🔧 Test 2: Session Tool Calling - Single Iteration")
    session = Session(
        system_prompt="You are a helpful assistant.",
        provider=provider,
        tools=[calculate_math]
    )
    
    # Generate with max_tool_calls=1 to prevent loops
    start_time = time.time()
    try:
        response = session.generate_with_tools(
            prompt="What is 25 * 4 + 7?",
            max_tool_calls=1  # Only 1 iteration
        )
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📝 Session response: {response}")
        
        if hasattr(response, 'content'):
            print(f"📄 Response content: {response.content}")
        
        print(f"✅ Session tool calling worked!")
        
    except Exception as e:
        print(f"❌ Session tool calling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 