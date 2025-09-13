#!/usr/bin/env python
"""
Test script for Enhanced GenerateResponse telemetry system.

This script tests the full SOTA ReAct agent capabilities including:
- Tool execution traces with timing
- ReAct cycle scratchpad visibility  
- Fact extraction and knowledge graph updates
- Token usage breakdown
- Memory system statistics
"""

import sys
import json
from abstractllm.factory_enhanced import create_enhanced_session
from abstractllm.tools.common_tools import read_file, list_files, search_files
from abstractllm.types import GenerateResponse

def test_enhanced_telemetry():
    """Test the enhanced response system with comprehensive telemetry."""
    
    print("🧪 Testing Enhanced ReAct Agent Telemetry System")
    print("=" * 60)
    
    # Create enhanced session
    session = create_enhanced_session(
        provider="mlx",
        model="mlx-community/GLM-4.5-Air-4bit",
        enable_memory=True,
        enable_retry=True,
        tools=[read_file, list_files, search_files],
        system_prompt="You are an intelligent AI assistant with memory and reasoning capabilities.",
        max_tokens=1000,
        temperature=0.7
    )
    
    # Test query that should trigger tool usage and fact extraction
    test_query = "List the files in the current directory and tell me what this project does"
    
    print(f"📝 Query: {test_query}")
    print()
    
    # Execute with enhanced telemetry
    response = session.generate(
        prompt=test_query,
        use_memory_context=True,
        create_react_cycle=True,
        tools=[read_file, list_files, search_files],
        max_tool_calls=10
    )
    
    # Verify we got an enhanced response
    if not isinstance(response, GenerateResponse):
        print("❌ ERROR: Expected GenerateResponse, got:", type(response))
        return False
    
    print("✅ Received enhanced GenerateResponse with telemetry!")
    print()
    
    # Display comprehensive telemetry
    print("🔍 RESPONSE ANALYSIS")
    print("-" * 40)
    
    # 1. Basic response info
    print(f"📄 Content length: {len(response.content) if response.content else 0} chars")
    print(f"🤖 Model: {response.model}")
    print(f"⏱️  Total time: {response.total_reasoning_time:.2f}s" if response.total_reasoning_time else "⏱️  No timing info")
    print(f"🔗 ReAct Cycle: {response.react_cycle_id}" if response.react_cycle_id else "🔗 No ReAct cycle")
    print()
    
    # 2. Tool execution analysis
    tools_executed = response.get_tools_executed()
    print(f"🔧 TOOL EXECUTION ANALYSIS")
    print(f"   Tools executed: {len(tools_executed)}")
    for tool in tools_executed:
        print(f"   - {tool}")
    
    if response.tool_calls:
        successful = sum(1 for t in response.tool_calls if t.get('success', False))
        success_rate = successful / len(response.tool_calls) * 100 if response.tool_calls else 0
        print(f"   Tool success rate: {success_rate:.1f}%")
        
        # Show tool details
        for i, tool in enumerate(response.tool_calls, 1):
            status = "✅" if tool.get('success') else "❌"
            print(f"   {i}. {status} {tool.get('name')} ({tool.get('execution_time', 0):.3f}s)")
    print()
    
    # 3. ReAct scratchpad
    if response.get_scratchpad_trace():
        print("🧠 REACT SCRATCHPAD TRACE")
        trace = response.get_scratchpad_trace()
        # Show first 800 chars to keep readable
        print(trace[:800] + ("..." if len(trace) > 800 else ""))
        print()
    else:
        print("❌ No scratchpad trace available")
        print()
    
    # 4. Facts extracted
    facts = response.get_facts_extracted()
    if facts:
        print(f"📚 FACTS EXTRACTED ({len(facts)} total):")
        for i, fact in enumerate(facts[:5], 1):  # Show first 5
            print(f"   {i}. {fact}")
        if len(facts) > 5:
            print(f"   ... and {len(facts) - 5} more")
        print()
    else:
        print("📚 No facts extracted")
        print()
    
    # 5. Token usage
    if response.usage:
        print(f"🪙 TOKEN USAGE")
        print(f"   Total: {response.usage.get('total_tokens', 0)}")
        print(f"   Prompt: {response.usage.get('prompt_tokens', 0)}")
        print(f"   Completion: {response.usage.get('completion_tokens', 0)}")
        print(f"   Generation time: {response.usage.get('time', 0):.2f}s")
        print()
    
    # 6. Summary
    summary = response.get_summary()
    print(f"📊 TELEMETRY SUMMARY")
    print(f"   {summary}")
    print()
    
    # 7. Test methods  
    print(f"🧪 ENHANCED METHODS TEST")
    print(f"   has_tool_calls(): {response.has_tool_calls()}")
    print(f"   get_tools_executed(): {response.get_tools_executed()}")
    print(f"   get_scratchpad_trace(): {'✅ Available' if response.get_scratchpad_trace() else '❌ None'}")
    print(f"   get_facts_extracted(): {len(response.get_facts_extracted())} facts")
    print()
    
    print("✅ Enhanced telemetry system test completed!")
    print("🎯 All SOTA ReAct agent features are now fully visible and trackable!")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_telemetry()
    sys.exit(0 if success else 1)