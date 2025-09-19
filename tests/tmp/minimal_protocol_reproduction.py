#!/usr/bin/env python3
"""
Minimal reproduction test for the protocol→files contamination issue.

This reproduces the exact pattern:
1. Create a "protocol" file that instructs reading other files step by step
2. Create target files with unique, identifiable content
3. Follow the protocol and observe if we get contaminated results

This will help isolate whether the issue is:
- Context contamination from memory system
- Timing issues with observation delivery
- Tool execution returning wrong content
"""

import asyncio
import tempfile
import logging
from pathlib import Path
from abstractllm import create_session, create_llm
from abstractllm.tools import tool

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@tool
def read_file(file_path: str, should_read_entire_file: bool = True,
              start_line_one_indexed: int = 1,
              end_line_one_indexed_inclusive: int = None) -> str:
    """Read a file with detailed logging to track what's actually returned."""

    logger.info(f"🔧 TOOL CALLED: read_file({file_path})")
    logger.info(f"   Params: entire_file={should_read_entire_file}, start={start_line_one_indexed}, end={end_line_one_indexed_inclusive}")

    try:
        path = Path(file_path)
        if not path.exists():
            error = f"Error: File '{file_path}' does not exist"
            logger.error(f"🔧 TOOL ERROR: {error}")
            return error

        with open(path, 'r', encoding='utf-8') as f:
            if should_read_entire_file:
                content = f.read()
                line_count = len(content.splitlines())
                result = f"File: {file_path} ({line_count} lines)\n\n{content}"
            else:
                lines = f.readlines()
                total_lines = len(lines)
                start_idx = max(0, start_line_one_indexed - 1)
                end_idx = min(total_lines, end_line_one_indexed_inclusive or total_lines)
                selected_lines = lines[start_idx:end_idx]
                result = f"File: {file_path} (lines {start_line_one_indexed}-{end_idx}, {len(selected_lines)} lines shown)\n\n{''.join(selected_lines)}"

        # Add unique fingerprint to track this specific execution
        fingerprint = f"TOOL_EXECUTION_ID_{hash(file_path + str(should_read_entire_file))}_{id(content)}"
        result_with_fingerprint = f"{result}\n\n--- {fingerprint} ---"

        logger.info(f"🔧 TOOL RESULT: {len(result_with_fingerprint)} chars")
        logger.info(f"🔧 TOOL FINGERPRINT: {fingerprint}")
        logger.info(f"🔧 CONTENT PREVIEW: {result_with_fingerprint[:100]}...")

        return result_with_fingerprint

    except Exception as e:
        error = f"Error reading {file_path}: {str(e)}"
        logger.error(f"🔧 TOOL EXCEPTION: {error}")
        return error


async def test_protocol_contamination():
    """Test the exact protocol→files pattern that triggers contamination."""

    print("🧪 PROTOCOL CONTAMINATION REPRODUCTION TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create protocol file that instructs reading other files step by step
        protocol_content = """# Test Protocol for File Reading

## Purpose
This protocol will test if file reading results get contaminated by previous content.

## Step-by-Step Instructions

### Step 1: Read Identity File
Execute this exact tool call:
read_file(file_path="{identity_file}", should_read_entire_file=false, start_line_one_indexed=1, end_line_one_indexed_inclusive=10)

### Step 2: Process Identity Information
After reading the identity file, you should see IDENTITY_UNIQUE_CONTENT_MARKER.

### Step 3: Read Values File
Execute this exact tool call:
read_file(file_path="{values_file}", should_read_entire_file=false, start_line_one_indexed=1, end_line_one_indexed_inclusive=10)

### Step 4: Process Values Information
After reading the values file, you should see VALUES_UNIQUE_CONTENT_MARKER.

### Step 5: Read Context File
Execute this exact tool call:
read_file(file_path="{context_file}", should_read_entire_file=true)

### Step 6: Final Verification
Report what unique content markers you found in each file:
- Identity file should contain: IDENTITY_UNIQUE_CONTENT_MARKER
- Values file should contain: VALUES_UNIQUE_CONTENT_MARKER
- Context file should contain: CONTEXT_UNIQUE_CONTENT_MARKER

CRITICAL: If any file returns content from a different file, report this as contamination!
"""

        # Create target files with unique, identifiable content
        identity_file = temp_path / "Identity.md"
        identity_content = """# Identity File

IDENTITY_UNIQUE_CONTENT_MARKER - If you see this, you successfully read the Identity file.

This file contains identity information that should NEVER appear when reading other files.
If you see this content when requesting Values.md or Context.md, there is contamination!

IDENTITY_FINGERPRINT_CONTENT_12345
"""

        values_file = temp_path / "Values.md"
        values_content = """# Values File

VALUES_UNIQUE_CONTENT_MARKER - If you see this, you successfully read the Values file.

This file contains values information that should NEVER appear when reading other files.
If you see this content when requesting Identity.md or Context.md, there is contamination!

VALUES_FINGERPRINT_CONTENT_67890
"""

        context_file = temp_path / "Context.md"
        context_content = """# Current Context File

CONTEXT_UNIQUE_CONTENT_MARKER - If you see this, you successfully read the Context file.

This file contains context information that should NEVER appear when reading other files.
If you see this content when requesting Identity.md or Values.md, there is contamination!

CONTEXT_FINGERPRINT_CONTENT_ABCDEF
"""

        # Write all files
        with open(identity_file, 'w') as f:
            f.write(identity_content)
        with open(values_file, 'w') as f:
            f.write(values_content)
        with open(context_file, 'w') as f:
            f.write(context_content)

        # Create protocol with file paths filled in
        protocol_with_paths = protocol_content.format(
            identity_file=str(identity_file),
            values_file=str(values_file),
            context_file=str(context_file)
        )

        protocol_file = temp_path / "Protocol.md"
        with open(protocol_file, 'w') as f:
            f.write(protocol_with_paths)

        print(f"📁 Created test files in: {temp_dir}")
        print(f"   Protocol: {protocol_file}")
        print(f"   Identity: {identity_file}")
        print(f"   Values: {values_file}")
        print(f"   Context: {context_file}")
        print("-" * 60)

        # Create session with memory enabled (this might be the source of contamination)
        llm = create_llm("ollama", model="qwen3-coder:30b")
        session = create_session(
            provider=llm,
            tools=[read_file],
            enable_memory=True  # CRITICAL: This enables the memory system that may cause contamination
        )

        # The test query that reproduces the issue
        test_query = f"""Please read and follow the protocol {protocol_file} step by step.
Do not skip any step and trust the process.

IMPORTANT: For each file you read, explicitly tell me:
1. What file you REQUESTED to read
2. What unique content marker you ACTUALLY received
3. Whether the content matches the expected file

If you receive content from a different file than requested, this indicates contamination!"""

        print("🎯 Test Query:")
        print(test_query)
        print("=" * 60)
        print("📊 Expected Behavior:")
        print("✅ Protocol file → Should contain protocol instructions")
        print("✅ Identity file → Should contain IDENTITY_UNIQUE_CONTENT_MARKER")
        print("✅ Values file → Should contain VALUES_UNIQUE_CONTENT_MARKER")
        print("✅ Context file → Should contain CONTEXT_UNIQUE_CONTENT_MARKER")
        print("❌ Any cross-contamination indicates the bug!")
        print("=" * 60)

        # Run the test in streaming mode
        tool_executions = []
        response_chunks = []

        async for chunk in session.generate_async(test_query, stream=True):
            if isinstance(chunk, str):
                response_chunks.append(chunk)
                print(chunk, end='', flush=True)
            elif isinstance(chunk, dict) and chunk.get('type') == 'tool_result':
                tool_call = chunk.get('tool_call', {})
                tool_name = tool_call.get('name', 'unknown')
                tool_output = tool_call.get('output', '')
                tool_args = tool_call.get('arguments', {})

                # Track this tool execution
                execution_record = {
                    'tool_name': tool_name,
                    'requested_file': tool_args.get('file_path', 'unknown'),
                    'output': tool_output,
                    'timestamp': 'now'
                }
                tool_executions.append(execution_record)

                print(f"\n🔍 TOOL EXECUTION #{len(tool_executions)}:")
                print(f"   Requested: {execution_record['requested_file']}")
                print(f"   Output length: {len(tool_output)} chars")

                # Check for contamination markers
                has_identity = "IDENTITY_UNIQUE_CONTENT_MARKER" in tool_output
                has_values = "VALUES_UNIQUE_CONTENT_MARKER" in tool_output
                has_context = "CONTEXT_UNIQUE_CONTENT_MARKER" in tool_output
                has_protocol = "Step-by-Step Instructions" in tool_output

                print(f"   Contains Identity marker: {has_identity}")
                print(f"   Contains Values marker: {has_values}")
                print(f"   Contains Context marker: {has_context}")
                print(f"   Contains Protocol content: {has_protocol}")

                # Detect contamination
                requested_file = execution_record['requested_file'].lower()
                if "identity" in requested_file and not has_identity:
                    print("   ❌ CONTAMINATION: Requested Identity but no Identity marker!")
                elif "values" in requested_file and not has_values:
                    print("   ❌ CONTAMINATION: Requested Values but no Values marker!")
                elif "context" in requested_file and not has_context:
                    print("   ❌ CONTAMINATION: Requested Context but no Context marker!")
                elif "protocol" in requested_file and not has_protocol:
                    print("   ❌ CONTAMINATION: Requested Protocol but no Protocol content!")

                print("-" * 40)

        print("\n" + "=" * 60)
        print("🔍 CONTAMINATION ANALYSIS SUMMARY:")
        print(f"Total tool executions: {len(tool_executions)}")

        for i, exec_record in enumerate(tool_executions, 1):
            print(f"\nExecution {i}: {exec_record['requested_file']}")
            output = exec_record['output']

            # Analyze what content was actually returned
            content_analysis = {
                'identity': "IDENTITY_UNIQUE_CONTENT_MARKER" in output,
                'values': "VALUES_UNIQUE_CONTENT_MARKER" in output,
                'context': "CONTEXT_UNIQUE_CONTENT_MARKER" in output,
                'protocol': "Step-by-Step Instructions" in output
            }

            print(f"   Returned content: {content_analysis}")

            # Check if this looks like contamination
            filename = Path(exec_record['requested_file']).stem.lower()
            expected_marker = f"{filename.upper()}_UNIQUE_CONTENT_MARKER"

            if expected_marker.replace("_UNIQUE_CONTENT_MARKER", "") in ["identity", "values", "context"]:
                if not content_analysis[filename]:
                    print(f"   🚨 CONTAMINATION DETECTED: Expected {filename} content but didn't find marker!")


if __name__ == "__main__":
    asyncio.run(test_protocol_contamination())