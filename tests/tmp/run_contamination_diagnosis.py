#!/usr/bin/env python3
"""
Master test runner to systematically diagnose the tool result contamination issue.

This will run multiple focused tests to isolate:
1. Protocol-based contamination (the exact pattern from your log)
2. Timing issues with observation delivery
3. Memory system interference
4. Context/scratchpad structure issues

Run this to get a comprehensive diagnosis of the root cause.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Import our test modules
from minimal_protocol_reproduction import test_protocol_contamination
from timing_isolation_test import test_observation_timing, test_memory_context_interference

# Configure logging for diagnosis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./contamination_diagnosis.log')
    ]
)

logger = logging.getLogger("contamination_diagnosis")

async def run_comprehensive_diagnosis():
    """Run all diagnostic tests to identify the contamination source."""

    print("🔬 COMPREHENSIVE CONTAMINATION DIAGNOSIS")
    print("=" * 80)
    print("This will run multiple focused tests to isolate the exact cause of")
    print("tool result contamination in streaming ReAct mode.")
    print("=" * 80)

    # Test 1: Protocol-based contamination (exact reproduction)
    print("\n" + "🧪 TEST 1: PROTOCOL-BASED CONTAMINATION")
    print("-" * 50)
    print("This reproduces the exact pattern from your log:")
    print("1. Read protocol file with instructions")
    print("2. Follow instructions to read other files step by step")
    print("3. Check if subsequent file reads get contaminated")
    print("-" * 50)

    try:
        await test_protocol_contamination()
        print("\n✅ Protocol contamination test completed")
    except Exception as e:
        print(f"\n❌ Protocol contamination test failed: {e}")
        logger.error(f"Protocol test failed: {e}")

    # Test 2: Timing and observation delivery
    print("\n" + "🕐 TEST 2: TIMING & OBSERVATION DELIVERY")
    print("-" * 50)
    print("This tests if the LLM starts thinking before receiving full observations")
    print("or if tool results arrive but get overwritten by context")
    print("-" * 50)

    try:
        await test_observation_timing()
        print("\n✅ Timing & observation test completed")
    except Exception as e:
        print(f"\n❌ Timing & observation test failed: {e}")
        logger.error(f"Timing test failed: {e}")

    # Test 3: Memory system interference
    print("\n" + "🧠 TEST 3: MEMORY SYSTEM INTERFERENCE")
    print("-" * 50)
    print("This tests if the memory system provides cached content")
    print("instead of fresh tool execution results")
    print("-" * 50)

    try:
        await test_memory_context_interference()
        print("\n✅ Memory interference test completed")
    except Exception as e:
        print(f"\n❌ Memory interference test failed: {e}")
        logger.error(f"Memory test failed: {e}")

    print("\n" + "=" * 80)
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 80)
    print("Check the output above for:")
    print("  🚨 CONTAMINATION DETECTED - Wrong file content delivered")
    print("  ⚠️  WARNING - Missing expected content")
    print("  🕐 TIMING ISSUE - Observation delivery problems")
    print("  🧠 MEMORY CONTAMINATION - Cached content interference")
    print("")
    print("Detailed logs saved to: ./contamination_diagnosis.log")
    print("=" * 80)


async def quick_diagnosis():
    """Quick focused test based on your specific issue."""

    print("⚡ QUICK DIAGNOSIS: Protocol → Files Contamination")
    print("=" * 60)

    # Just run the protocol test since that's the exact issue
    try:
        await test_protocol_contamination()
    except Exception as e:
        print(f"❌ Quick diagnosis failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Running quick diagnosis...")
        asyncio.run(quick_diagnosis())
    else:
        print("Running comprehensive diagnosis...")
        print("(Use --quick for focused test only)")
        asyncio.run(run_comprehensive_diagnosis())