#!/usr/bin/env python
"""
Simple demonstration of accessing ReAct agent telemetry.

Shows how easy it is to get comprehensive information about what the agent did.
"""

from abstractllm.factory_enhanced import create_enhanced_session
from abstractllm.tools.common_tools import read_file, list_files, search_files

def demo_simple_access():
    """Demonstrate simple telemetry access."""
    
    # Create session
    session = create_enhanced_session(
        provider="mlx",
        model="mlx-community/GLM-4.5-Air-4bit",
        enable_memory=True,
        tools=[read_file, list_files, search_files]
    )
    
    # Run query
    response = session.generate("What files are here and what do they tell you about this project?")
    
    print("🎯 SIMPLE TELEMETRY ACCESS")
    print("=" * 50)
    
    # 1. What tools were executed?
    print(f"🔧 Tools executed: {', '.join(response.get_tools_executed())}")
    
    # 2. How long did it take?  
    print(f"⏱️  Total time: {response.total_reasoning_time:.1f}s")
    
    # 3. What facts were learned?
    facts = response.get_facts_extracted()
    print(f"🧠 Facts extracted: {len(facts)}")
    for i, fact in enumerate(facts[:3], 1):  # Show first 3
        print(f"   {i}. {fact}")
    
    # 4. How did it reason?
    print(f"\n💭 Reasoning trace available: {'Yes' if response.get_scratchpad_trace() else 'No'}")
    if response.get_scratchpad_trace():
        # Show just the summary line
        trace_lines = response.get_scratchpad_trace().split('\n')
        summary_line = next((line for line in trace_lines if line.startswith('Query:')), 'No summary')
        print(f"   {summary_line}")
    
    # 5. One-liner summary
    print(f"\n📊 Summary: {response.get_summary()}")
    
    print(f"\n✨ Response length: {len(response.content)} characters")
    print("\n🎉 All telemetry is now easily accessible!")

if __name__ == "__main__":
    demo_simple_access()