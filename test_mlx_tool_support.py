#!/usr/bin/env python3
"""
Test if MLX models actually support native tool calling.
"""

from abstractllm import create_llm
import inspect

def main():
    """Test MLX tool calling support at the processor level."""
    
    print("🔍 Testing MLX Native Tool Calling Support")
    print("=" * 50)
    
    # Create MLX provider
    provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
    provider.load_model()
    
    print(f"📦 Model: {provider._model}")
    print(f"🔧 Processor: {provider._processor}")
    print(f"📋 Processor type: {type(provider._processor)}")
    
    # Check if apply_chat_template supports tools
    if hasattr(provider._processor, 'apply_chat_template'):
        template_method = provider._processor.apply_chat_template
        sig = inspect.signature(template_method)
        print(f"\n📝 apply_chat_template signature: {sig}")
        print(f"🔧 Parameters: {list(sig.parameters.keys())}")
        
        # Test if tools parameter is supported
        if 'tools' in sig.parameters:
            print("✅ Tools parameter is supported in apply_chat_template")
            
            # Test with a simple tool
            messages = [{"role": "user", "content": "What is 2+2?"}]
            tools = [{
                "name": "calculate",
                "description": "Calculate math",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }]
            
            try:
                result = provider._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tools=tools
                )
                print(f"✅ Tool template generation successful")
                print(f"📝 Generated prompt: {result[:200]}...")
            except Exception as e:
                print(f"❌ Tool template generation failed: {e}")
        else:
            print("❌ Tools parameter NOT supported in apply_chat_template")
    else:
        print("❌ apply_chat_template method not found")
    
    # Test what the model actually generates when asked to use tools
    print(f"\n🧪 Testing model output with tool prompt")
    
    tool_prompt = """You are a helpful assistant with access to tools.

Available tools:
- calculate_math(expression: str) -> str: Calculate a mathematical expression

To use a tool, respond with: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

User: What is 25 * 4 + 7?
Assistant:"""
    
    try:
        # Generate using simple text generation (no tools)
        response = provider.generate(
            prompt="What is 25 * 4 + 7? Use the calculate_math tool if available.",
            system_prompt="You are a helpful assistant. If tools are available, use this format: <tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>",
            max_tokens=200
        )
        
        print(f"📝 Model response: {response.content}")
        
        # Check if it contains tool call format
        if "<tool_call>" in response.content and "</tool_call>" in response.content:
            print("✅ Model generated tool call format!")
        else:
            print("❌ Model did NOT generate tool call format")
            
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Conclusion: Check if MLX supports native tool calling")

if __name__ == "__main__":
    main() 