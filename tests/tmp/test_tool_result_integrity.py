#!/usr/bin/env python3
"""
Test script to reproduce and diagnose tool result mismatch issues.

This test specifically checks if there are issues with:
1. Tool result content accuracy
2. Context contamination between tool calls
3. Streaming mode tool result delivery
4. Memory system interference with fresh tool results
"""

import asyncio
import logging
import tempfile
import os
from pathlib import Path
from abstractllm import create_session, create_llm
from abstractllm.tools import tool

# Set up detailed logging to track tool execution
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@tool
def read_test_file(file_path: str) -> str:
    """Read a test file and return its contents with identifying markers."""
    logger.info(f"🔧 TOOL EXECUTING: read_test_file({file_path})")

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Add unique markers to identify which file was actually read
        result = f"=== TOOL RESULT FOR {file_path} ===\n{content}\n=== END {file_path} ==="
        logger.info(f"🔧 TOOL RESULT LENGTH: {len(result)} chars")
        logger.info(f"🔧 TOOL RESULT PREVIEW: {result[:100]}...")
        return result

    except Exception as e:
        error_msg = f"Error reading {file_path}: {str(e)}"
        logger.error(f"🔧 TOOL ERROR: {error_msg}")
        return error_msg

async def test_tool_result_integrity():
    """Test whether tool results are delivered correctly without contamination."""

    print("🧪 Testing Tool Result Integrity")
    print("=" * 60)

    # Create test files with unique, identifiable content
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = {}

        # Create 3 test files with distinct content
        for i in range(1, 4):
            file_path = Path(temp_dir) / f"test_file_{i}.txt"
            content = f"This is UNIQUE CONTENT from test_file_{i}.txt\n" + \
                     f"FILE_{i}_MARKER_CONTENT\n" + \
                     f"If you see this, you successfully read file {i}\n" + \
                     f"UNIQUE_IDENTIFIER_FILE_{i}" * 3

            with open(file_path, 'w') as f:
                f.write(content)

            test_files[f"file_{i}"] = str(file_path)
            print(f"📄 Created test file {i}: {file_path}")

        print("-" * 60)

        # Create LLM session
        llm = create_llm("ollama", model="qwen3-coder:30b")
        session = create_session(
            provider=llm,
            tools=[read_test_file],
            enable_memory=True
        )

        # Test query that reads files sequentially
        test_query = f"""Please read these three test files in order and tell me what each one contains:

1. First, read: {test_files['file_1']}
2. Then read: {test_files['file_2']}
3. Finally read: {test_files['file_3']}

For each file, explicitly state:
- Which file you are reading
- What unique content identifier you found
- Whether the content matches the expected file

Be very specific about what content you actually received from each tool call."""

        print(f"🎯 Query: {test_query}")
        print("=" * 60)
        print("📊 Expected behavior:")
        print("✅ File 1 should return content with 'FILE_1_MARKER_CONTENT'")
        print("✅ File 2 should return content with 'FILE_2_MARKER_CONTENT'")
        print("✅ File 3 should return content with 'FILE_3_MARKER_CONTENT'")
        print("❌ Any file returning content from a different file indicates the bug")
        print("=" * 60)

        # Test in streaming mode
        print("🎭 STREAMING MODE TEST:")
        print("-" * 30)

        response_content = ""
        async for chunk in session.generate_async(test_query, stream=True):
            if isinstance(chunk, str):
                print(chunk, end='', flush=True)
                response_content += chunk
            elif isinstance(chunk, dict) and chunk.get('type') == 'tool_result':
                tool_info = chunk.get('tool_call', {})
                tool_output = tool_info.get('output', '')
                tool_name = tool_info.get('name', 'unknown')

                print(f"\n🔍 TOOL RESULT RECEIVED:")
                print(f"   Tool: {tool_name}")
                print(f"   Output Preview: {tool_output[:150]}...")
                print(f"   Contains FILE_1_MARKER: {'FILE_1_MARKER_CONTENT' in tool_output}")
                print(f"   Contains FILE_2_MARKER: {'FILE_2_MARKER_CONTENT' in tool_output}")
                print(f"   Contains FILE_3_MARKER: {'FILE_3_MARKER_CONTENT' in tool_output}")
                print("-" * 30)

        print("\n" + "=" * 60)
        print("🔍 INTEGRITY CHECK:")

        # Check if the LLM mentioned any content mismatches
        if "wrong" in response_content.lower() or "different" in response_content.lower():
            print("❌ POTENTIAL MISMATCH DETECTED - LLM mentioned wrong/different content!")
        else:
            print("✅ No obvious mismatch indicators found in response")

        # Check conversation history for proper tool messages
        print("\n📋 CONVERSATION HISTORY CHECK:")
        for i, msg in enumerate(session.conversation_history[-10:]):  # Last 10 messages
            print(f"  Message {i}: {msg.role} - {msg.content[:50]}...")
            if msg.role == 'tool':
                # Check if tool message content matches expected file
                if hasattr(msg, 'name'):
                    print(f"    Tool: {msg.name}")
                print(f"    Content markers: "
                      f"F1={str('FILE_1_MARKER' in msg.content)}, "
                      f"F2={str('FILE_2_MARKER' in msg.content)}, "
                      f"F3={str('FILE_3_MARKER' in msg.content)}")

if __name__ == "__main__":
    asyncio.run(test_tool_result_integrity())