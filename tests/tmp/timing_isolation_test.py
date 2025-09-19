#!/usr/bin/env python3
"""
Timing isolation test to determine if the issue is:
1. LLM starts thinking before receiving full observation
2. Tool results arrive but get overwritten by context
3. Memory system provides stale content instead of fresh results

This test adds explicit timing markers and delays to isolate the issue.
"""

import asyncio
import tempfile
import logging
import time
from pathlib import Path
from abstractllm import create_session, create_llm
from abstractllm.tools import tool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

@tool
def read_file_with_timing(file_path: str, should_read_entire_file: bool = True,
                         start_line_one_indexed: int = 1,
                         end_line_one_indexed_inclusive: int = None) -> str:
    """Read file with explicit timing markers to track observation delivery."""

    start_time = time.time()
    execution_id = f"EXEC_{int(start_time * 1000) % 100000}"

    logger.info(f"🕐 {execution_id} STARTED: read_file({file_path})")

    # Add a small delay to simulate processing time
    time.sleep(0.1)

    try:
        path = Path(file_path)
        if not path.exists():
            error = f"TIMING_TEST_ERROR: File '{file_path}' does not exist ({execution_id})"
            logger.error(f"🕐 {execution_id} ERROR: {error}")
            return error

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create result with explicit timing information
        end_time = time.time()
        timing_header = f"""
=== TIMING ANALYSIS FOR {execution_id} ===
Requested file: {file_path}
Execution start: {start_time:.3f}
Execution end: {end_time:.3f}
Duration: {(end_time - start_time):.3f}s
Timestamp: {time.strftime('%H:%M:%S.%f')[:-3]}
==========================================
"""

        # Add unique execution markers
        result = f"{timing_header}\n\nFILE_CONTENT_START_{execution_id}\n{content}\nFILE_CONTENT_END_{execution_id}"

        logger.info(f"🕐 {execution_id} COMPLETED: {len(result)} chars returned")
        logger.info(f"🕐 {execution_id} CONTENT_MARKER: FILE_CONTENT_START_{execution_id}")

        return result

    except Exception as e:
        error = f"TIMING_TEST_EXCEPTION: {str(e)} ({execution_id})"
        logger.error(f"🕐 {execution_id} EXCEPTION: {error}")
        return error


async def test_observation_timing():
    """Test if LLM receives observations at the correct time or gets contaminated."""

    print("🕐 TIMING & OBSERVATION DELIVERY TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create files with timing-specific content
        file1 = temp_path / "first_file.txt"
        file1_content = """FIRST_FILE_UNIQUE_CONTENT_MARKER_12345
This is the first file that should be read.
If you see this content when requesting second_file.txt, there is contamination!
TIMING_TEST_FILE_1_CONTENT"""

        file2 = temp_path / "second_file.txt"
        file2_content = """SECOND_FILE_UNIQUE_CONTENT_MARKER_67890
This is the second file that should be read.
If you see this content when requesting first_file.txt, there is contamination!
TIMING_TEST_FILE_2_CONTENT"""

        with open(file1, 'w') as f:
            f.write(file1_content)
        with open(file2, 'w') as f:
            f.write(file2_content)

        print(f"📁 Created test files:")
        print(f"   File 1: {file1}")
        print(f"   File 2: {file2}")
        print("-" * 60)

        # Create session
        llm = create_llm("ollama", model="qwen3-coder:30b")
        session = create_session(
            provider=llm,
            tools=[read_file_with_timing],
            enable_memory=True
        )

        # Test query with explicit timing verification
        test_query = f"""I need you to read two files in sequence and verify that you receive the correct content for each.

Step 1: Read the first file: {file1}
- You should receive content containing "FIRST_FILE_UNIQUE_CONTENT_MARKER_12345"
- You should receive content containing "TIMING_TEST_FILE_1_CONTENT"
- Look for the execution ID in format "EXEC_XXXXX"

Step 2: After receiving the first file result, read the second file: {file2}
- You should receive content containing "SECOND_FILE_UNIQUE_CONTENT_MARKER_67890"
- You should receive content containing "TIMING_TEST_FILE_2_CONTENT"
- Look for a different execution ID

CRITICAL TIMING TEST:
- For each file, tell me the EXACT execution ID you received (EXEC_XXXXX)
- Tell me what unique content marker you actually found
- If you receive content from the wrong file, this indicates contamination
- If you don't see the timing headers, this indicates delivery issues

DO NOT start thinking about the second file until you have fully processed the first file's observation!"""

        print("🎯 Test Query (Timing Focus):")
        print(test_query)
        print("=" * 60)

        # Track all chunks and timing
        tool_calls_made = []
        tool_results_received = []
        response_chunks = []
        start_test_time = time.time()

        print(f"⏱️  Test started at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        print("-" * 60)

        async for chunk in session.generate_async(test_query, stream=True):
            current_time = time.time() - start_test_time

            if isinstance(chunk, str):
                response_chunks.append((current_time, chunk))
                # Check if LLM mentions timing/execution IDs
                if "EXEC_" in chunk:
                    print(f"⏱️  [{current_time:.3f}s] LLM mentions execution ID: {chunk.strip()}")
                elif "FIRST_FILE" in chunk or "SECOND_FILE" in chunk:
                    print(f"⏱️  [{current_time:.3f}s] LLM mentions file marker: {chunk.strip()}")
                else:
                    print(f"⏱️  [{current_time:.3f}s] Response: {chunk}", end='', flush=True)

            elif isinstance(chunk, dict) and chunk.get('type') == 'tool_result':
                tool_call = chunk.get('tool_call', {})
                tool_args = tool_call.get('arguments', {})
                tool_output = tool_call.get('output', '')
                requested_file = tool_args.get('file_path', 'unknown')

                # Extract execution ID from result
                execution_id = "UNKNOWN"
                if "EXEC_" in tool_output:
                    import re
                    match = re.search(r'EXEC_(\d+)', tool_output)
                    if match:
                        execution_id = f"EXEC_{match.group(1)}"

                tool_result_record = {
                    'timestamp': current_time,
                    'requested_file': requested_file,
                    'execution_id': execution_id,
                    'output_length': len(tool_output),
                    'contains_first_marker': "FIRST_FILE_UNIQUE_CONTENT_MARKER" in tool_output,
                    'contains_second_marker': "SECOND_FILE_UNIQUE_CONTENT_MARKER" in tool_output,
                    'contains_timing_header': "TIMING ANALYSIS" in tool_output
                }
                tool_results_received.append(tool_result_record)

                print(f"\n⏱️  [{current_time:.3f}s] 🔍 TOOL RESULT RECEIVED:")
                print(f"   Execution ID: {execution_id}")
                print(f"   Requested: {requested_file}")
                print(f"   Output length: {tool_result_record['output_length']} chars")
                print(f"   Has timing header: {tool_result_record['contains_timing_header']}")
                print(f"   Contains first marker: {tool_result_record['contains_first_marker']}")
                print(f"   Contains second marker: {tool_result_record['contains_second_marker']}")

                # Check for contamination
                if "first_file" in requested_file.lower():
                    if tool_result_record['contains_second_marker']:
                        print("   🚨 CONTAMINATION: Requested first file but got second file content!")
                    elif not tool_result_record['contains_first_marker']:
                        print("   🚨 MISSING CONTENT: Requested first file but no first file marker!")

                elif "second_file" in requested_file.lower():
                    if tool_result_record['contains_first_marker']:
                        print("   🚨 CONTAMINATION: Requested second file but got first file content!")
                    elif not tool_result_record['contains_second_marker']:
                        print("   🚨 MISSING CONTENT: Requested second file but no second file marker!")

                print("-" * 40)

        print(f"\n⏱️  Test completed at: {time.strftime('%H:%M:%S.%f')[:-3]}")
        print("=" * 60)
        print("🔍 TIMING & CONTAMINATION ANALYSIS:")
        print(f"Total tool results: {len(tool_results_received)}")

        for i, result in enumerate(tool_results_received, 1):
            print(f"\nTool call {i} (at {result['timestamp']:.3f}s):")
            print(f"   File: {result['requested_file']}")
            print(f"   Execution ID: {result['execution_id']}")
            print(f"   Has timing header: {result['contains_timing_header']}")
            print(f"   First marker: {result['contains_first_marker']}")
            print(f"   Second marker: {result['contains_second_marker']}")

            # Contamination analysis
            filename = Path(result['requested_file']).stem
            if "first" in filename:
                expected_marker = result['contains_first_marker']
                unexpected_marker = result['contains_second_marker']
            else:
                expected_marker = result['contains_second_marker']
                unexpected_marker = result['contains_first_marker']

            if not expected_marker:
                print(f"   ❌ ISSUE: Missing expected content marker!")
            if unexpected_marker:
                print(f"   🚨 CONTAMINATION: Contains unexpected marker!")
            if not result['contains_timing_header']:
                print(f"   ⚠️  WARNING: Missing timing header - delivery issue?")


async def test_memory_context_interference():
    """Test if memory context interferes with fresh tool results."""

    print("\n🧠 MEMORY CONTEXT INTERFERENCE TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create file
        test_file = temp_path / "test_file.txt"
        test_content = "FRESH_CONTENT_FROM_ACTUAL_FILE_EXECUTION"

        with open(test_file, 'w') as f:
            f.write(test_content)

        # Create session with memory
        llm = create_llm("ollama", model="qwen3-coder:30b")
        session = create_session(
            provider=llm,
            tools=[read_file_with_timing],
            enable_memory=True
        )

        # First, populate memory with some content
        await session.generate_async("I am working on a project about CACHED_CONTENT_FROM_MEMORY_SYSTEM", stream=False)

        # Now request to read the file
        test_query = f"""Please read this file: {test_file}

You should receive content that says "FRESH_CONTENT_FROM_ACTUAL_FILE_EXECUTION".

If you receive anything mentioning "CACHED_CONTENT_FROM_MEMORY_SYSTEM" instead,
this indicates memory system contamination of fresh tool results."""

        print(f"Memory interference test - requesting: {test_file}")
        print("-" * 60)

        async for chunk in session.generate_async(test_query, stream=True):
            if isinstance(chunk, str):
                print(chunk, end='', flush=True)
                if "CACHED_CONTENT_FROM_MEMORY_SYSTEM" in chunk:
                    print("\n🚨 MEMORY CONTAMINATION: LLM mentioned cached content!")
                elif "FRESH_CONTENT_FROM_ACTUAL_FILE_EXECUTION" in chunk:
                    print("\n✅ FRESH CONTENT: LLM correctly received actual file content!")
            elif isinstance(chunk, dict) and chunk.get('type') == 'tool_result':
                tool_output = chunk.get('tool_call', {}).get('output', '')

                has_fresh = "FRESH_CONTENT_FROM_ACTUAL_FILE_EXECUTION" in tool_output
                has_cached = "CACHED_CONTENT_FROM_MEMORY_SYSTEM" in tool_output

                print(f"\n🔍 Tool result analysis:")
                print(f"   Contains fresh content: {has_fresh}")
                print(f"   Contains cached content: {has_cached}")

                if has_cached:
                    print("   🚨 MEMORY CONTAMINATION DETECTED!")
                elif has_fresh:
                    print("   ✅ Fresh content delivered correctly!")
                else:
                    print("   ⚠️  Neither fresh nor cached content found - unexpected result!")


if __name__ == "__main__":
    asyncio.run(test_observation_timing())
    await asyncio.sleep(1)
    await test_memory_context_interference()