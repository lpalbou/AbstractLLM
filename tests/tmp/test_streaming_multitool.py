#!/usr/bin/env python3
"""
Test script to verify streaming mode with multiple consecutive tool calls.
This tests the fix for the issue where streaming mode would hang after
the first tool call when the LLM wanted to call multiple tools.
"""

import sys
import logging
from abstractllm import Session
from abstractllm.tools import tool

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Create test tools that simulate the scenario from the log file
@tool
def list_files(directory_path: str = ".") -> str:
    """List files in a directory."""
    return """Files in '.' matching '*':
  📄 README.md (16,849 bytes)
  📄 pyproject.toml (2,973 bytes)
  📄 alma-simple.py (13,407 bytes)
  📁 abstractllm/
  📁 tests/"""

@tool
def read_file(file_path: str, should_read_entire_file: bool = True) -> str:
    """Read the contents of a file."""
    if "README.md" in file_path:
        return """# AbstractLLM
A unified interface for Large Language Models with memory, reasoning, and tool capabilities.
Version: 1.0.1"""
    elif "pyproject.toml" in file_path:
        return """[project]
name = "abstractllm"
version = "1.0.1"
description = "Unified LLM interface"""
    else:
        return f"Contents of {file_path}"

def test_streaming_multi_tool():
    """Test streaming mode with multiple tool calls."""
    print("\n" + "="*80)
    print("Testing Streaming Mode with Multiple Tool Calls")
    print("="*80 + "\n")

    # Create session with tools
    session = Session(
        provider="ollama",
        system_prompt="You are a helpful assistant with tool access.",
        tools=[list_files, read_file],
        max_tool_calls=5
    )

    # Test prompt that should trigger multiple tool calls
    prompt = "List the local files and then read README.md and pyproject.toml to understand this project"

    print(f"Prompt: {prompt}\n")
    print("Streaming response:")
    print("-" * 40)

    tool_count = 0
    text_buffer = ""

    try:
        # Generate streaming response
        for chunk in session.generate(prompt=prompt, stream=True):
            if isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                tool_count += 1
                tool_name = chunk["tool_call"].get("name", "unknown")
                print(f"\n[TOOL EXECUTED #{tool_count}: {tool_name}]")
            elif isinstance(chunk, str):
                text_buffer += chunk
                print(chunk, end="", flush=True)
            elif hasattr(chunk, "content"):
                text_buffer += chunk.content
                print(chunk.content, end="", flush=True)

        print("\n" + "-" * 40)
        print(f"\nTest Results:")
        print(f"  ✅ Streaming completed successfully")
        print(f"  ✅ Total tools executed: {tool_count}")
        print(f"  ✅ Response length: {len(text_buffer)} characters")

        if tool_count >= 2:
            print(f"  ✅ PASS: Multiple tools executed in streaming mode ({tool_count} tools)")
            return True
        else:
            print(f"  ❌ FAIL: Expected multiple tools, but only {tool_count} executed")
            return False

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_non_streaming_multi_tool():
    """Test non-streaming mode with multiple tool calls (baseline)."""
    print("\n" + "="*80)
    print("Testing Non-Streaming Mode with Multiple Tool Calls (Baseline)")
    print("="*80 + "\n")

    # Create session with tools
    session = Session(
        provider="ollama",
        system_prompt="You are a helpful assistant with tool access.",
        tools=[list_files, read_file],
        max_tool_calls=5
    )

    # Test prompt that should trigger multiple tool calls
    prompt = "List the local files and then read README.md and pyproject.toml to understand this project"

    print(f"Prompt: {prompt}\n")

    try:
        # Generate non-streaming response
        response = session.generate(prompt=prompt, stream=False)

        # Count tools executed based on session messages
        tool_messages = [m for m in session.messages if m.role.value == "tool"]
        tool_count = len(tool_messages)

        print(f"Response: {response[:200]}..." if len(str(response)) > 200 else f"Response: {response}")
        print(f"\nTest Results:")
        print(f"  ✅ Non-streaming completed successfully")
        print(f"  ✅ Total tool messages: {tool_count}")

        if tool_count >= 2:
            print(f"  ✅ PASS: Multiple tools executed in non-streaming mode ({tool_count} tools)")
            return True
        else:
            print(f"  ❌ FAIL: Expected multiple tools, but only {tool_count} executed")
            return False

    except Exception as e:
        print(f"\n❌ Error during non-streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔧 Testing AbstractLLM Streaming Multi-Tool Fix")
    print("=" * 80)

    # Check if Ollama is available
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Warning: Ollama not running. Please start Ollama first.")
            print("   Run: ollama serve")
            sys.exit(1)
    except FileNotFoundError:
        print("⚠️  Warning: Ollama not installed. Please install Ollama first.")
        sys.exit(1)

    # Run tests
    baseline_pass = test_non_streaming_multi_tool()
    streaming_pass = test_streaming_multi_tool()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if baseline_pass and streaming_pass:
        print("✅ ALL TESTS PASSED: Streaming multi-tool support is working!")
    elif baseline_pass and not streaming_pass:
        print("⚠️  PARTIAL SUCCESS: Non-streaming works but streaming still has issues")
    else:
        print("❌ TESTS FAILED: Multi-tool support needs more work")

    sys.exit(0 if (baseline_pass and streaming_pass) else 1)