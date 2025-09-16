#!/usr/bin/env python3
"""
Test script to verify ReAct streaming behavior fix.
This should now execute one tool at a time instead of parallel execution.
"""
import asyncio
import logging
from abstractllm.tools import tool
from abstractllm import create_session, create_llm

# Set up logging to see the ReAct fix in action
logging.basicConfig(level=logging.INFO)

@tool
def read_file(file_path: str) -> str:
    """Read a file and return its contents."""
    print(f"🔧 TOOL EXECUTED: read_file({file_path})")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return f"File content from {file_path}: {content[:100]}..."
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"

@tool
def list_files(directory: str) -> str:
    """List files in a directory."""
    print(f"🔧 TOOL EXECUTED: list_files({directory})")
    import os
    try:
        files = os.listdir(directory)
        return f"Files in {directory}: {files[:10]}"
    except Exception as e:
        return f"Error listing {directory}: {str(e)}"

@tool
def analyze_project() -> str:
    """Analyze the current project structure."""
    print(f"🔧 TOOL EXECUTED: analyze_project()")
    return "Project analysis: AbstractLLM unified interface library"

async def test_streaming_react_behavior():
    """Test that streaming mode executes tools sequentially."""
    print("🧪 Testing ReAct Streaming Behavior Fix")
    print("=" * 50)

    # Create LLM and session
    llm = create_llm("ollama", model="qwen3-coder:30b")
    session = create_session(
        provider=llm,
        tools=[read_file, list_files, analyze_project],
        enable_memory=True
    )

    # Test query that might trigger multiple tool calls
    test_query = """Please analyze the current project by:
1. Reading the README.md file
2. Listing files in the abstractllm directory
3. Reading the pyproject.toml file
4. Analyzing the project structure

Be thorough and methodical in your approach."""

    print(f"Query: {test_query}")
    print("-" * 50)
    print("Expected behavior:")
    print("✅ Execute ONE tool at a time")
    print("✅ Wait for observation before next tool")
    print("✅ Follow Think→Act→Observe→Think pattern")
    print("=" * 50)

    # Stream the response
    print("🎯 Streaming Response:")
    async for chunk in session.generate_async(test_query, stream=True):
        if isinstance(chunk, str):
            print(chunk, end='', flush=True)
        elif isinstance(chunk, dict) and chunk.get('type') == 'tool_result':
            tool_info = chunk.get('tool_call', {})
            print(f"\n📊 Tool Result: {tool_info.get('name')} executed")

    print("\n" + "=" * 50)
    print("✅ Test completed! Check above for sequential tool execution.")

if __name__ == "__main__":
    asyncio.run(test_streaming_react_behavior())