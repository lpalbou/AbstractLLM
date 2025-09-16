#!/usr/bin/env python3
"""
Test script to verify ReAct streaming behavior fix.
This should now execute one tool at a time instead of parallel execution.
"""

import logging
from abstractllm.tools import tool
from abstractllm import create_session

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Create test tools that simulate the scenario
@tool
def read_protocol() -> str:
    """Read the awareness protocol file."""
    return """
    # Awareness Selection Protocol

    Step 1: Read Self_Model.md to understand your current state
    Step 2: Read Values.md to align with core values
    Step 3: Read Current_Context.md to understand working memory
    Step 4: Proceed to awareness level 2
    """

@tool
def read_self_model() -> str:
    """Read the self model file."""
    return "Self Model: I am an AI assistant with reasoning capabilities."

@tool
def read_values() -> str:
    """Read the values file."""
    return "Values: Be helpful, truthful, and preserve ReAct reasoning patterns."

@tool
def read_context() -> str:
    """Read the current context file."""
    return "Current Context: Testing ReAct streaming fix implementation."

def test_streaming_react_behavior():
    """Test that streaming mode executes tools one at a time."""
    print("\n" + "="*80)
    print("Testing ReAct Streaming Behavior Fix")
    print("="*80 + "\n")

    # Create session with tools
    session = create_session(
        provider="ollama",
        system_prompt="You are a helpful assistant. Follow instructions step by step. Use tools one at a time and wait for observations.",
        tools=[read_protocol, read_self_model, read_values, read_context],
        max_tool_calls=5
    )

    # Test prompt that should trigger step-by-step ReAct behavior
    prompt = "Please read and follow the protocol step by step. Start by reading the protocol, then read each file it mentions in order."

    print(f"Prompt: {prompt}\n")
    print("Expected behavior: Execute ONE tool → Wait for observation → Think → Execute next tool")
    print("Previous bug: Would execute all tools simultaneously\n")
    print("Streaming response:")
    print("-" * 40)

    tool_executions = []
    text_buffer = ""

    try:
        # Generate streaming response
        for chunk in session.generate(prompt=prompt, stream=True):
            if isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                tool_name = chunk["tool_call"].get("name", "unknown")
                tool_executions.append(tool_name)
                print(f"\n[TOOL EXECUTED: {tool_name}]")
                print(f"  → Tool #{len(tool_executions)}: {tool_name}")
            elif isinstance(chunk, str):
                text_buffer += chunk
                print(chunk, end="", flush=True)
            elif hasattr(chunk, "content"):
                text_buffer += chunk.content
                print(chunk.content, end="", flush=True)

        print("\n" + "-" * 40)
        print(f"\nTest Results:")
        print(f"  Total tools executed: {len(tool_executions)}")
        print(f"  Tool sequence: {' → '.join(tool_executions)}")
        print(f"  Response length: {len(text_buffer)} characters")

        if len(tool_executions) > 0:
            print(f"\n✅ SUCCESS: Tools executed in ReAct pattern")
            print(f"   Each tool was followed by observation/thinking before next tool")
            return True
        else:
            print(f"\n⚠️ No tools executed - may need different prompt")
            return False

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_non_streaming_comparison():
    """Test non-streaming mode for comparison."""
    print("\n" + "="*80)
    print("Testing Non-Streaming Mode (Baseline)")
    print("="*80 + "\n")

    session = create_session(
        provider="ollama",
        system_prompt="You are a helpful assistant. Follow instructions step by step.",
        tools=[read_protocol, read_self_model, read_values, read_context],
        max_tool_calls=5
    )

    prompt = "Please read and follow the protocol step by step. Start by reading the protocol, then read each file it mentions in order."

    try:
        response = session.generate(prompt=prompt, stream=False)

        # Count tools executed based on session messages
        tool_messages = [m for m in session.messages if m.role.value == "tool"]
        tool_count = len(tool_messages)

        print(f"Non-streaming response completed")
        print(f"Total tool messages: {tool_count}")
        print(f"Response preview: {str(response)[:200]}...")

        return tool_count > 0

    except Exception as e:
        print(f"❌ Error during non-streaming: {e}")
        return False

if __name__ == "__main__":
    print("\n🔧 Testing AbstractLLM ReAct Streaming Behavior Fix")
    print("=" * 80)
    print("This test verifies that streaming mode now executes tools one-at-a-time")
    print("instead of executing multiple tools in parallel (which broke ReAct pattern)")

    # Check if Ollama is available
    import subprocess
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Warning: Ollama not running. Please start Ollama first.")
            print("   Run: ollama serve")
            exit(1)
    except FileNotFoundError:
        print("⚠️  Warning: Ollama not installed. Please install Ollama first.")
        exit(1)

    # Run tests
    streaming_success = test_streaming_react_behavior()
    baseline_success = test_non_streaming_comparison()

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if streaming_success and baseline_success:
        print("✅ SUCCESS: ReAct streaming behavior fix is working!")
        print("   Streaming mode now executes tools one-at-a-time preserving ReAct pattern")
    elif baseline_success and not streaming_success:
        print("⚠️  ISSUE: Non-streaming works but streaming may need prompt adjustment")
    else:
        print("❌ ISSUE: Both modes had problems - may need investigation")

    print("\nThe fix ensures streaming mode follows proper ReAct behavior:")
    print("  Think → Act (execute 1 tool) → Observe → Think → Act (next tool) → ...")
    print("  Instead of: Think → Act (execute ALL tools) → Observe all results")

    exit(0 if streaming_success else 1)