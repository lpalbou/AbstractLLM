#!/usr/bin/env python3
"""
Focused test for debugging specific provider issues.
"""

from abstractllm import create_llm
import time
import os

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "pow": pow})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def main():
    """Test specific provider issues."""
    
    print("🔧 Focused Provider Testing")
    print("=" * 40)
    
    # Create test file
    with open("numbers.txt", "w") as f:
        f.write("10\n20\n30\n40\n50")
    
    print("✅ Created numbers.txt")
    
    # Test Ollama with different prompting strategies
    provider = create_llm("ollama", model="cogito:8b")
    tools = [read_file, write_file, calculate_math]
    
    strategies = [
        {
            "name": "Direct Command",
            "prompt": "Execute this: read numbers.txt, calculate the sum, write to result.txt",
            "system_prompt": "You are a helpful assistant. Use tools to execute commands."
        },
        {
            "name": "Step-by-step",
            "prompt": "Step 1: Read numbers.txt. Step 2: Calculate sum. Step 3: Write to result.txt",
            "system_prompt": "You are a helpful assistant. Execute each step using the appropriate tool."
        },
        {
            "name": "Imperative",
            "prompt": "Read the file numbers.txt using read_file tool",
            "system_prompt": "You are a helpful assistant. Use the exact tools specified."
        }
    ]
    
    for strategy in strategies:
        print(f"\n🧪 Strategy: {strategy['name']}")
        print("-" * 30)
        
        start_time = time.time()
        response = provider.generate(
            prompt=strategy["prompt"],
            system_prompt=strategy["system_prompt"],
            tools=tools,
            max_tokens=500
        )
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📝 Response type: {type(response).__name__}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"✅ Tool calls: {len(response.tool_calls)}")
            for i, tc in enumerate(response.tool_calls):
                print(f"   {i+1}. {tc.name}({tc.arguments})")
        else:
            print(f"❌ No tool calls")
            print(f"📄 Response: {str(response)[:150]}...")
    
    # Cleanup
    if os.path.exists("numbers.txt"):
        os.remove("numbers.txt")
    if os.path.exists("result.txt"):
        os.remove("result.txt")
    
    print(f"\n🧹 Cleanup complete")

if __name__ == "__main__":
    main() 