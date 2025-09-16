#!/usr/bin/env python3
"""
Unit test to verify the streaming multi-tool fix without requiring Ollama.
This simulates the exact scenario from the user's log file.
"""

import logging
from unittest.mock import Mock, MagicMock, patch
from abstractllm import Session
from abstractllm.tools import ToolCall, ToolCallResponse
from abstractllm.types import GenerateResponse

# Alias for compatibility
ToolCallRequest = ToolCallResponse

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def create_mock_provider_with_multi_tools():
    """Create a mock provider that simulates multiple tool calls in streaming."""
    mock_provider = Mock()

    # First stream: initial response with first tool call
    def first_stream(*args, **kwargs):
        # Initial text
        yield "I'll help you list the local files. Let me use the available tool to retrieve that information for you.\n"

        # Tool call for list_files
        tool_request = ToolCallRequest()
        tool_request.tool_calls = [
            ToolCall(id="call_1", name="list_files", arguments={"directory_path": "."})
        ]
        yield tool_request

    # Second stream: follow-up with second tool call
    def second_stream(*args, **kwargs):
        # Response after first tool
        yield "Now let me read the README.md file to understand this project better.\n"

        # Tool call for read_file
        tool_request = ToolCallRequest()
        tool_request.tool_calls = [
            ToolCall(id="call_2", name="read_file", arguments={"file_path": "README.md", "should_read_entire_file": True})
        ]
        yield tool_request

    # Third stream: follow-up with third tool call
    def third_stream(*args, **kwargs):
        # Response after second tool
        yield "Let me also check the pyproject.toml file for more details.\n"

        # Tool call for read_file
        tool_request = ToolCallRequest()
        tool_request.tool_calls = [
            ToolCall(id="call_3", name="read_file", arguments={"file_path": "pyproject.toml", "should_read_entire_file": True})
        ]
        yield tool_request

    # Fourth stream: final response with no more tools
    def final_stream(*args, **kwargs):
        # Final summary after all tools
        yield "Based on my analysis of the files:\n"
        yield "- This is the AbstractLLM project\n"
        yield "- Version 1.0.1\n"
        yield "- It provides a unified interface for Large Language Models\n"
        yield "- Includes memory, reasoning, and tool capabilities\n"

    # Set up the mock to return different streams on each call
    stream_sequence = [first_stream, second_stream, third_stream, final_stream]
    call_count = [0]

    def generate_mock(*args, **kwargs):
        if kwargs.get('stream'):
            # Return the appropriate stream based on call count
            stream_func = stream_sequence[min(call_count[0], len(stream_sequence) - 1)]
            call_count[0] += 1
            return stream_func(*args, **kwargs)
        else:
            # Non-streaming response
            response = GenerateResponse(
                content="Here are the files and project information.",
                raw_response=None
            )
            return response

    mock_provider.generate = generate_mock
    mock_provider.config_manager = Mock()
    mock_provider.config_manager.get_param = Mock(return_value="mock-model")

    return mock_provider

def test_streaming_with_multiple_tools():
    """Test that streaming mode can handle multiple consecutive tool calls."""
    print("\n" + "="*80)
    print("Testing Streaming Mode - Multiple Consecutive Tool Calls")
    print("="*80 + "\n")

    # Create mock tool functions
    tool_functions = {
        "list_files": lambda directory_path=".": "Files: README.md, pyproject.toml, abstractllm/",
        "read_file": lambda file_path, should_read_entire_file=True: f"Content of {file_path}"
    }

    # Create session with mock provider
    mock_provider = create_mock_provider_with_multi_tools()
    session = Session(provider=mock_provider)

    # Track what happens during streaming
    chunks_received = []
    tools_executed = []

    try:
        # Generate streaming response
        print("Starting streaming generation...")
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            prompt="List files and read key project files"
        ):
            if isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                tool_name = chunk["tool_call"].get("name", "unknown")
                tools_executed.append(tool_name)
                print(f"  ✓ Tool executed: {tool_name}")
            elif isinstance(chunk, str):
                chunks_received.append(chunk)
                if len(chunk.strip()) > 0:
                    print(f"  → Text: {chunk.strip()[:60]}...")

        print(f"\n📊 Results:")
        print(f"  • Total text chunks: {len(chunks_received)}")
        print(f"  • Total tools executed: {len(tools_executed)}")
        print(f"  • Tools: {', '.join(tools_executed)}")

        # Check if all expected tools were executed
        expected_tools = ["list_files", "read_file", "read_file"]
        if tools_executed == expected_tools:
            print(f"\n✅ SUCCESS: All {len(expected_tools)} tools executed in correct order!")
            print("   The streaming multi-tool fix is working correctly.")
            return True
        else:
            print(f"\n❌ FAILURE: Expected {expected_tools}, got {tools_executed}")
            return False

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_max_tool_calls_limit():
    """Test that max_tool_calls limit is respected in streaming mode."""
    print("\n" + "="*80)
    print("Testing Max Tool Calls Limit in Streaming Mode")
    print("="*80 + "\n")

    # Create mock tool functions
    tool_functions = {
        "list_files": lambda directory_path=".": "Files: README.md, pyproject.toml",
        "read_file": lambda file_path, should_read_entire_file=True: f"Content of {file_path}"
    }

    # Create session with mock provider and limit to 2 tool calls
    mock_provider = create_mock_provider_with_multi_tools()
    session = Session(provider=mock_provider, max_tool_calls=2)

    tools_executed = []

    try:
        # Generate streaming response with tool call limit
        print("Starting streaming with max_tool_calls=2...")
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            prompt="List files and read multiple files",
            max_tool_calls=2
        ):
            if isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                tool_name = chunk["tool_call"].get("name", "unknown")
                tools_executed.append(tool_name)
                print(f"  ✓ Tool executed: {tool_name}")

        print(f"\n📊 Results:")
        print(f"  • Tools executed: {len(tools_executed)} (limit was 2)")
        print(f"  • Tools: {', '.join(tools_executed)}")

        if len(tools_executed) <= 2:
            print("\n✅ SUCCESS: Tool call limit was respected!")
            return True
        else:
            print(f"\n❌ FAILURE: Executed {len(tools_executed)} tools, but limit was 2")
            return False

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔧 Unit Testing AbstractLLM Streaming Multi-Tool Fix")
    print("=" * 80)
    print("This test simulates the exact scenario from the user's log file:")
    print("- LLM calls list_files first")
    print("- Then calls read_file for README.md")
    print("- Then calls read_file for pyproject.toml")
    print("- Finally provides a summary")

    # Run tests
    test1_pass = test_streaming_with_multiple_tools()
    test2_pass = test_max_tool_calls_limit()

    # Summary
    print("\n" + "=" * 80)
    print("UNIT TEST SUMMARY")
    print("=" * 80)

    if test1_pass and test2_pass:
        print("✅ ALL UNIT TESTS PASSED!")
        print("\nThe streaming multi-tool fix successfully:")
        print("  1. Handles multiple consecutive tool calls in streaming mode")
        print("  2. Respects the max_tool_calls limit")
        print("  3. Processes follow-up responses that contain additional tool calls")
        print("\nThis resolves the issue where streaming would hang after the first tool.")
    else:
        print("❌ Some unit tests failed. The fix may need adjustment.")

    import sys
    sys.exit(0 if (test1_pass and test2_pass) else 1)