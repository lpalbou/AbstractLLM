#!/usr/bin/env python3
"""
Debug script to see what tools are being passed to MLX provider.
"""

from abstractllm import create_llm
from abstractllm.session import Session

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def main():
    """Debug tool passing."""
    
    print("🔍 Tool Passing Debug")
    print("=" * 50)
    
    # Create MLX provider and Session like in the test
    provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
    
    session = Session(
        system_prompt="You are a helpful assistant.",
        provider=provider,
        tools=[read_file]  # Function is automatically registered
    )
    
    print(f"📋 Session.tools: {session.tools}")
    print(f"📋 Type of session.tools: {type(session.tools)}")
    if session.tools:
        print(f"📋 Type of first tool: {type(session.tools[0])}")
        print(f"📋 First tool attributes: {dir(session.tools[0])}")
        if hasattr(session.tools[0], 'name'):
            print(f"📋 First tool name: {session.tools[0].name}")
        if hasattr(session.tools[0], 'description'):
            print(f"📋 First tool description: {session.tools[0].description}")
        if hasattr(session.tools[0], 'to_dict'):
            print(f"📋 First tool as dict: {session.tools[0].to_dict()}")

if __name__ == "__main__":
    main() 