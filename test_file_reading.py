#!/usr/bin/env python3
"""
Test tool calling with file reading - a proper test case where models MUST use tools.
"""

from abstractllm import create_llm
from abstractllm.session import Session
import time
import os

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def list_files(directory: str = ".") -> str:
    """List files in a directory."""
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def main():
    """Test file reading tools with both providers."""
    
    print("📁 Testing Tool Calling with File Reading")
    print("=" * 60)
    
    # Create a test file
    test_content = """# Test Document
This is a test document for tool calling.
It contains multiple lines and some important information.
The capital of France is Paris.
The year is 2025.
"""
    
    with open("test_file.txt", "w") as f:
        f.write(test_content)
    
    print("✅ Created test_file.txt")
    
    # Test both providers
    providers = [
        ("ollama", "cogito:8b"),
        ("mlx", "mlx-community/Qwen3-30B-A3B-4bit")
    ]
    
    for provider_name, model_name in providers:
        print(f"\n🧪 Testing {provider_name.upper()} Provider ({model_name})")
        print("-" * 50)
        
        try:
            # Create provider
            provider = create_llm(provider_name, model=model_name)
            
            # Test 1: Direct provider tool calling
            print(f"\n🔧 Test 1: Direct {provider_name.upper()} Tool Calling")
            start_time = time.time()
            
            response = provider.generate(
                prompt="Please read the file 'test_file.txt' and tell me what's in it",
                system_prompt="You are a helpful assistant. When you need to read files, use the read_file tool.",
                tools=[read_file, list_files],
                max_tokens=500
            )
            
            elapsed = time.time() - start_time
            print(f"⏱️  Time: {elapsed:.1f}s")
            print(f"📝 Response type: {type(response)}")
            
            if hasattr(response, 'tool_calls'):
                print(f"✅ {provider_name.upper()} returned ToolCallRequest!")
                print(f"🔧 Tool calls: {len(response.tool_calls)}")
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"   {i+1}. {tool_call.name}({tool_call.arguments})")
            else:
                print(f"❌ {provider_name.upper()} did NOT return ToolCallRequest")
                print(f"📄 Response preview: {str(response)[:200]}...")
            
            # Test 2: Session-based tool calling
            print(f"\n🔧 Test 2: {provider_name.upper()} Session Tool Calling")
            session = Session(
                system_prompt="You are a helpful assistant. When you need to read files, use the read_file tool.",
                provider=provider,
                tools=[read_file, list_files]
            )
            
            start_time = time.time()
            try:
                response = session.generate(
                    prompt="Read test_file.txt and summarize its contents",
                    max_tool_calls=3
                )
                elapsed = time.time() - start_time
                
                print(f"⏱️  Time: {elapsed:.1f}s")
                print(f"📝 Session response: {str(response)[:300]}...")
                
                # Check conversation history for tool usage
                tool_calls_found = 0
                for msg in session.messages:
                    if hasattr(msg, 'tool_results') and msg.tool_results:
                        tool_calls_found += len(msg.tool_results)
                
                if tool_calls_found > 0:
                    print(f"✅ Session used {tool_calls_found} tool call(s)")
                else:
                    print(f"❌ Session did not use any tools")
                    
            except Exception as e:
                print(f"❌ Session test failed: {e}")
            
        except Exception as e:
            print(f"❌ {provider_name.upper()} provider failed: {e}")
            continue
    
    # Cleanup
    if os.path.exists("test_file.txt"):
        os.remove("test_file.txt")
        print(f"\n🧹 Cleaned up test_file.txt")

if __name__ == "__main__":
    main() 