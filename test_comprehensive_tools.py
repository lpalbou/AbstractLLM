#!/usr/bin/env python3
"""
Comprehensive tool calling test with multiple scenarios.
Tests both Ollama and MLX providers with various tool types.
"""

from abstractllm import create_llm
from abstractllm.session import Session
import time
import os
import math

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
        # Only allow safe mathematical operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "pow": pow})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def list_directory(path: str = ".") -> str:
    """List files and directories in the specified path."""
    try:
        items = os.listdir(path)
        return "\n".join(sorted(items))
    except Exception as e:
        return f"Error listing directory: {str(e)}"

def get_file_info(file_path: str) -> str:
    """Get information about a file."""
    try:
        stat = os.stat(file_path)
        size = stat.st_size
        modified = time.ctime(stat.st_mtime)
        return f"File: {file_path}\nSize: {size} bytes\nModified: {modified}"
    except Exception as e:
        return f"Error getting file info: {str(e)}"

def run_test_scenario(provider_name: str, model_name: str, scenario_name: str, 
                     prompt: str, expected_tools: list, system_prompt: str = None):
    """Run a single test scenario and return results."""
    print(f"\n🧪 {scenario_name}")
    print("-" * 40)
    
    try:
        provider = create_llm(provider_name, model=model_name)
        
        tools = [read_file, write_file, calculate_math, list_directory, get_file_info]
        
        # Test direct provider tool calling
        start_time = time.time()
        response = provider.generate(
            prompt=prompt,
            system_prompt=system_prompt or "You are a helpful assistant. Use tools when needed.",
            tools=tools,
            max_tokens=800
        )
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📝 Response type: {type(response).__name__}")
        
        # Check if tools were called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"✅ Tool calls detected: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"   {i+1}. {tool_call.name}({tool_call.arguments})")
            
            # Check if expected tools were used
            used_tools = [tc.name for tc in response.tool_calls]
            if any(expected in used_tools for expected in expected_tools):
                print(f"✅ Expected tools used: {expected_tools}")
                return True
            else:
                print(f"⚠️  Expected {expected_tools}, got {used_tools}")
                return False
        else:
            print(f"❌ No tool calls detected")
            print(f"📄 Response preview: {str(response)[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run comprehensive tool calling tests."""
    
    print("🔧 Comprehensive Tool Calling Tests")
    print("=" * 60)
    
    # Create test files
    test_data = {
        "numbers.txt": "10\n20\n30\n40\n50",
        "info.txt": "This is a test document.\nIt contains information about AbstractLLM.\nVersion: 1.0",
        "data.csv": "name,age,city\nAlice,25,NYC\nBob,30,LA\nCharlie,35,Chicago"
    }
    
    for filename, content in test_data.items():
        with open(filename, "w") as f:
            f.write(content)
    
    print(f"✅ Created test files: {', '.join(test_data.keys())}")
    
    # Test scenarios
    scenarios = [
        {
            "name": "File Reading",
            "prompt": "Read the file 'info.txt' and tell me what's in it",
            "expected_tools": ["read_file"],
            "system_prompt": "You are a helpful assistant. When you need to read files, use the read_file tool."
        },
        {
            "name": "Math Calculation", 
            "prompt": "Calculate 25 * 4 + 7",
            "expected_tools": ["calculate_math"],
            "system_prompt": "You are a helpful assistant. When you need to perform calculations, use the calculate_math tool."
        },
        {
            "name": "File Operations",
            "prompt": "Read numbers.txt, calculate the sum of the numbers, and write the result to result.txt",
            "expected_tools": ["read_file", "write_file", "calculate_math"],
            "system_prompt": "You are a helpful assistant. Use the read_file tool to read files, calculate_math for calculations, and write_file to save results. Execute these operations step by step."
        },
        {
            "name": "Directory Listing",
            "prompt": "Show me all files in the current directory",
            "expected_tools": ["list_directory"],
            "system_prompt": "You are a helpful assistant. When asked to list files or directories, use the list_directory tool."
        }
    ]
    
    # Test providers
    providers = [
        ("ollama", "cogito:8b"),
        ("mlx", "mlx-community/Qwen3-30B-A3B-4bit")
    ]
    
    results = {}
    
    for provider_name, model_name in providers:
        print(f"\n🤖 Testing {provider_name.upper()} Provider ({model_name})")
        print("=" * 50)
        
        results[provider_name] = {}
        
        for scenario in scenarios:
            success = run_test_scenario(
                provider_name=provider_name,
                model_name=model_name,
                scenario_name=scenario["name"],
                prompt=scenario["prompt"],
                expected_tools=scenario["expected_tools"],
                system_prompt=scenario["system_prompt"]
            )
            results[provider_name][scenario["name"]] = success
    
    # Summary
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    for provider_name in results:
        print(f"\n{provider_name.upper()} Provider:")
        total_tests = len(results[provider_name])
        passed_tests = sum(results[provider_name].values())
        
        for scenario_name, success in results[provider_name].items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {scenario_name}: {status}")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"  Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    # Cleanup
    for filename in test_data.keys():
        if os.path.exists(filename):
            os.remove(filename)
    
    # Remove result.txt if created
    if os.path.exists("result.txt"):
        os.remove("result.txt")
    
    print(f"\n🧹 Cleaned up test files")

if __name__ == "__main__":
    main() 