#!/usr/bin/env python3
"""
Detailed debug of Ollama tool calling to see request/response format.
"""

from abstractllm import create_llm
import json
import requests

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def main():
    """Test Ollama tool calling request/response format."""
    
    print("🔍 Detailed Ollama Tool Calling Debug")
    print("=" * 50)
    
    # Test 1: Check what the provider sends to Ollama
    print("\n🧪 Test 1: Inspecting Ollama Provider Tool Request")
    provider = create_llm("ollama", model="cogito:8b")
    
    # Create tools list
    tools = [calculate_math]
    
    # Check tool processing
    processed_tools = provider._process_tools(tools)
    print(f"📋 Processed tools: {json.dumps(processed_tools, indent=2)}")
    
    # Check what request would be sent
    request_data = provider._prepare_request_for_chat(
        model="cogito:8b",
        prompt="What is 25 * 4 + 7?",
        system_prompt="You are a helpful assistant.",
        processed_files=[],
        processed_tools=processed_tools,
        temperature=0.7,
        max_tokens=500,
        stream=False
    )
    print(f"\n📤 Request data that would be sent: {json.dumps(request_data, indent=2)}")
    
    # Test 2: Manual API call to see Ollama's response
    print("\n🧪 Test 2: Manual API call to Ollama")
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=request_data
        )
        response.raise_for_status()
        data = response.json()
        print(f"📥 Raw Ollama response: {json.dumps(data, indent=2)}")
        
        # Check for tool calls
        if "message" in data and "tool_calls" in data.get("message", {}):
            print("✅ Tool calls found in response!")
            print(f"🔧 Tool calls: {data['message']['tool_calls']}")
        else:
            print("❌ No tool calls found in response")
            
    except Exception as e:
        print(f"❌ Manual API call failed: {e}")
    
    # Test 3: Check what cogito model supports
    print(f"\n🧪 Test 3: Check cogito:8b capabilities")
    print(f"🔧 Tool support check: {provider._supports_tool_calls()}")
    print(f"📋 Model in tool-capable list: {'cogito:8b' in provider.TOOL_CALL_CAPABLE_MODELS}")
    
    # Test 4: Try with a different prompt that's more explicit about tool format
    print(f"\n🧪 Test 4: Test with explicit tool format instruction")
    explicit_request = request_data.copy()
    explicit_request["messages"][0]["content"] = "You are a helpful assistant. When you need to use tools, respond in this exact format: {'tool_calls': [{'name': 'tool_name', 'parameters': {'arg': 'value'}}]}"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=explicit_request
        )
        response.raise_for_status()
        data = response.json()
        print(f"📥 Response with explicit instructions: {data['message']['content'][:200]}...")
        
    except Exception as e:
        print(f"❌ Explicit instruction test failed: {e}")

if __name__ == "__main__":
    main() 