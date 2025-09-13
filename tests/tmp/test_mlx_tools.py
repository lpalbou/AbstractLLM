#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to verify MLX provider tools are working.
"""

import sys
from abstractllm import create_session
from abstractllm.tools.common_tools import read_file, list_files, search_files

def test_mlx_with_tools():
    """Test MLX provider with tools."""
    print("=" * 60)
    print("Testing MLX Provider with Tools")
    print("=" * 60)
    
    # Create session with tools
    session = create_session(
        "mlx",
        model="Qwen/Qwen3-14B-MLX-4bit",
        tools=[read_file, list_files, search_files],
        system_prompt="You are a helpful AI assistant.",
        max_tokens=500,
        temperature=0.0  # Low temperature for consistent behavior
    )
    
    print("✅ Session created with 3 tools")
    
    # Test 1: Simple list files
    print("\n📝 Test 1: List files")
    prompt1 = "List the files in the current directory"
    print(f"Prompt: {prompt1}")
    
    try:
        response1 = session.generate(
            prompt=prompt1,
            tools=[read_file, list_files, search_files],
            max_tool_calls=5
        )
        print(f"Response preview: {str(response1)[:200]}...")
        print("✅ Test 1 passed - model used tools")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Read specific file
    print("\n📝 Test 2: Read file")
    prompt2 = "Read the first 5 lines of the README.md file"
    print(f"Prompt: {prompt2}")
    
    try:
        response2 = session.generate(
            prompt=prompt2,
            tools=[read_file, list_files, search_files],
            max_tool_calls=5
        )
        print(f"Response preview: {str(response2)[:200]}...")
        print("✅ Test 2 passed - model read file")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! MLX tools are working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_mlx_with_tools()
    sys.exit(0 if success else 1)