#!/usr/bin/env python3
"""
Debug script to see what MLX is actually generating for tool calls.
"""

from abstractllm import create_llm

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def main():
    """Debug MLX responses."""
    
    print("🔍 MLX Response Debug")
    print("=" * 50)
    
    # Create MLX provider
    provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
    
    # Test cases
    test_cases = [
        "What is 25 * 4 + 7?",
        "Read the file README.md",
        "Calculate 15 * 8"
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {prompt}")
        print("-" * 40)
        
        # Generate with tools in system prompt (like Session does)
        system_prompt = (
            "You are a helpful assistant that can use tools. "
            "When asked to perform tasks, use the appropriate tools and provide clear, concise responses.\n\n"
            "Available Tools:\n"
            "Tool: read_file\nDescription: Read the contents of a file.\n\n"
            "Tool: calculate_math\nDescription: Calculate a mathematical expression safely.\n\n"
            "To use a tool, respond in the format:\n"
            "```\nAction: tool_name\nAction Input: {\"parameter\": \"value\"}\n```"
        )
        
        response = provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            tools=[read_file, calculate_math]
        )
        
        print(f"📄 Full Response:")
        print(response.content)
        print(f"\n🔍 Response Type: {type(response)}")
        print(f"🔍 Has tool_calls attr: {hasattr(response, 'tool_calls')}")
        if hasattr(response, 'tool_calls'):
            print(f"🔍 tool_calls value: {response.tool_calls}")
        print(f"🔍 has_tool_calls(): {response.has_tool_calls()}")
        
        # Test our parsing method directly
        from abstractllm.providers.mlx_provider import MLXProvider
        mlx_provider = MLXProvider()
        tools = [read_file, calculate_math]
        parsed = mlx_provider._parse_tool_calls_from_response(response.content, tools)
        
        print(f"🔍 Our parser result: {parsed}")
        if parsed:
            print(f"🔍 Parsed tool calls: {[tc.name for tc in parsed.tool_calls]}")

if __name__ == "__main__":
    main() 