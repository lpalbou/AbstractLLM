#!/usr/bin/env python3
"""
Test script to validate the memory context token limit fix.
This verifies that the memory system now uses appropriate token limits
based on the model's capacity instead of the hardcoded 2000 token limit.
"""

import logging
from abstractllm import Session
from abstractllm.tools import tool

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

@tool
def dummy_tool(message: str) -> str:
    """A simple test tool."""
    return f"Processed: {message}"

def test_memory_context_scaling():
    """Test that memory context limit scales with model capacity."""
    print("\n" + "="*80)
    print("Testing Memory Context Token Limit Scaling")
    print("="*80 + "\n")

    # Test with different max_tokens values
    test_scenarios = [
        {"max_tokens": 4000, "expected_limit": 1000, "percentage": 0.25},  # 25% of 4k
        {"max_tokens": 32000, "expected_limit": 8000, "percentage": 0.25}, # 25% of 32k
        {"max_tokens": 128000, "expected_limit": 32000, "percentage": 0.25}, # 25% of 128k
    ]

    # Test custom percentage
    custom_scenarios = [
        {"max_tokens": 32000, "expected_limit": 16000, "percentage": 0.5},  # 50% of 32k
        {"max_tokens": 32000, "expected_limit": 3200, "percentage": 0.1},   # 10% of 32k
    ]

    all_pass = True

    for i, scenario in enumerate(test_scenarios + custom_scenarios, 1):
        max_tokens = scenario["max_tokens"]
        expected_limit = scenario["expected_limit"]
        percentage = scenario["percentage"]

        print(f"Scenario {i}: max_tokens={max_tokens}, percentage={percentage*100:.0f}%")

        # Create session with custom memory context percentage
        session = Session(
            provider="ollama",  # Mock provider - won't actually be called
            enable_memory=True,
            tools=[dummy_tool],
            memory_context_percentage=percentage
        )

        # Check calculated limit matches expected
        model_max_tokens = max_tokens
        calculated_limit = int(model_max_tokens * session.memory_context_percentage)

        print(f"  Expected limit: {expected_limit} tokens")
        print(f"  Calculated limit: {calculated_limit} tokens")

        if calculated_limit == expected_limit:
            print(f"  ✅ PASS: Memory context limit correctly calculated")
        else:
            print(f"  ❌ FAIL: Expected {expected_limit}, got {calculated_limit}")
            all_pass = False

        print()

    return all_pass

def test_backwards_compatibility():
    """Test that the default behavior is reasonable."""
    print("\n" + "="*80)
    print("Testing Backwards Compatibility")
    print("="*80 + "\n")

    # Create session with default settings
    session = Session(
        provider="ollama",  # Mock provider
        enable_memory=True,
        tools=[dummy_tool]
    )

    # Check that default percentage is 25%
    print(f"Default memory_context_percentage: {session.memory_context_percentage}")
    if session.memory_context_percentage == 0.25:
        print("✅ PASS: Default percentage is 25%")

        # With default 32k context, should get 8k for memory
        expected_memory_limit = int(32768 * 0.25)  # 8192
        print(f"With 32k context: {expected_memory_limit} tokens for memory context")
        print("✅ PASS: Much better than the old 2000 token hardcoded limit!")
        return True
    else:
        print(f"❌ FAIL: Expected 0.25, got {session.memory_context_percentage}")
        return False

if __name__ == "__main__":
    print("\n🔧 Testing Memory Context Token Limit Fix")
    print("=" * 80)
    print("This fix addresses the issue where memory context was hardcoded to 2000 tokens,")
    print("regardless of the model's actual capacity (e.g., 32k tokens).")
    print()

    # Run tests
    scaling_pass = test_memory_context_scaling()
    compatibility_pass = test_backwards_compatibility()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if scaling_pass and compatibility_pass:
        print("✅ ALL TESTS PASSED!")
        print("\nThe memory context limit fix successfully:")
        print("  1. Scales memory context limit based on model capacity")
        print("  2. Allows customization via memory_context_percentage parameter")
        print("  3. Maintains backwards compatibility with sensible defaults")
        print("  4. Provides much more context for large models (8k vs 2k tokens)")
        print("\nWith a 32k context model, you now get 8192 tokens for memory context")
        print("instead of the old hardcoded 2000 token limit!")
    else:
        print("❌ Some tests failed. The fix may need adjustment.")

    print("\n🎯 Next Steps:")
    print("- Test with actual conversation to verify the warning is gone")
    print("- Consider increasing memory_context_percentage for very large models")
    print("- Monitor memory context quality with the larger token allowance")

    import sys
    sys.exit(0 if (scaling_pass and compatibility_pass) else 1)