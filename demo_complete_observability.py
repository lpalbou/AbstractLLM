#!/usr/bin/env python
"""
Demonstration of COMPLETE ReAct Agent Observability with Real-Time Events.

This script shows:
- Complete scratchpad trace (NO truncation)
- Real-time phase change triggers
- Persistent scratchpad storage in session folder
- Event listeners for visual cues
- Cycle-by-cycle breakdown with full details

Based on SOTA 2024-2025 AI agent observability practices.
"""

import time
from pathlib import Path
from abstractllm.factory_enhanced import create_enhanced_session
from abstractllm.tools.common_tools import read_file, list_files, search_files
from abstractllm.scratchpad_manager import ReActPhase, CyclePhaseEvent

def create_visual_event_listeners():
    """Create event listeners that provide visual cues during agent reasoning."""
    
    def on_cycle_start(event: CyclePhaseEvent):
        print(f"\n🚀 [CYCLE START] {event.cycle_id}")
        print(f"   Query: {event.content}")
        print(f"   Timestamp: {event.timestamp}")
        
    def on_thinking(event: CyclePhaseEvent):
        print(f"\n💭 [THINKING] Iteration {event.iteration}")
        print(f"   Content: {event.content}")
        
    def on_acting(event: CyclePhaseEvent):
        tool_name = event.metadata.get('tool_name', 'unknown')
        print(f"\n🔧 [ACTING] Calling tool: {tool_name}")
        print(f"   Args: {event.metadata.get('tool_args', {})}")
        
    def on_observing(event: CyclePhaseEvent):
        success = "✅" if event.metadata.get('success', False) else "❌"
        exec_time = event.metadata.get('execution_time', 0)
        print(f"\n👁️  [OBSERVING] {success} Tool result received")
        print(f"   Execution time: {exec_time:.6f}s")
        
    def on_final_answer(event: CyclePhaseEvent):
        print(f"\n✅ [FINAL ANSWER] Generated")
        print(f"   Success: {event.metadata.get('success', False)}")
        
    def on_cycle_complete(event: CyclePhaseEvent):
        iterations = event.metadata.get('total_iterations', 0)
        success = "✅" if event.metadata.get('success', False) else "❌"
        print(f"\n🏁 [CYCLE COMPLETE] {success}")
        print(f"   Total iterations: {iterations}")
        print(f"   Cycle ID: {event.cycle_id}")
        
    def on_error(event: CyclePhaseEvent):
        print(f"\n❌ [ERROR] {event.content}")
        error_details = event.metadata.get('error_details')
        if error_details:
            print(f"   Details: {error_details}")
    
    return {
        ReActPhase.CYCLE_START: on_cycle_start,
        ReActPhase.THINKING: on_thinking,
        ReActPhase.ACTING: on_acting,
        ReActPhase.OBSERVING: on_observing,
        ReActPhase.FINAL_ANSWER: on_final_answer,
        ReActPhase.CYCLE_COMPLETE: on_cycle_complete,
        ReActPhase.ERROR: on_error
    }

def demo_complete_observability():
    """Demonstrate complete ReAct agent observability."""
    
    print("🎯 COMPLETE REACT AGENT OBSERVABILITY DEMO")
    print("=" * 60)
    print("Features:")
    print("- Complete scratchpad trace (NO truncation)")
    print("- Real-time phase change events")
    print("- Persistent storage in session memory folder")
    print("- Visual cues for each reasoning phase")
    print("=" * 60)
    
    # Create session with memory folder
    memory_folder = Path("./demo_memory")
    session = create_enhanced_session(
        provider="mlx",
        model="mlx-community/GLM-4.5-Air-4bit",
        enable_memory=True,
        persist_memory=memory_folder / "session.pkl",
        tools=[read_file, list_files, search_files]
    )
    
    print(f"\n📁 Memory folder: {memory_folder.absolute()}")
    
    # Set up event listeners for real-time visual cues
    event_listeners = create_visual_event_listeners()
    
    # Subscribe to all phase events
    for phase, callback in event_listeners.items():
        session.scratchpad.subscribe_to_events(phase, callback)
    
    print("\n🎭 Event listeners registered for all ReAct phases")
    print("\n" + "="*60)
    print("🤖 STARTING AGENT REASONING...")
    print("="*60)
    
    # Execute query with complete observability
    query = "List the files here, read the README, and tell me what this project does in detail"
    
    print(f"\n📝 User Query: {query}")
    
    # Generate response (events will fire in real-time)
    start_time = time.time()
    response = session.generate(
        prompt=query,
        use_memory_context=True,
        create_react_cycle=True,
        tools=[read_file, list_files, search_files],
        max_tool_calls=10
    )
    end_time = time.time()
    
    print("\n" + "="*60)
    print("🎉 AGENT REASONING COMPLETE")
    print("="*60)
    
    # Display complete results
    print(f"\n📊 COMPLETE OBSERVABILITY RESULTS")
    print(f"   Total time: {end_time - start_time:.2f}s")
    print(f"   ReAct cycle: {response.react_cycle_id}")
    print(f"   Tools executed: {len(response.get_tools_executed())}")
    print(f"   Facts extracted: {len(response.get_facts_extracted())}")
    
    # Access complete scratchpad
    scratchpad_file = response.get_complete_scratchpad_file()
    print(f"\n📄 Complete scratchpad saved to: {scratchpad_file}")
    
    if scratchpad_file and Path(scratchpad_file).exists():
        file_size = Path(scratchpad_file).stat().st_size
        print(f"   File size: {file_size:,} bytes")
        print(f"   🔍 COMPLETE trace with NO truncation available")
    
    # Show first part of complete trace
    complete_trace = response.get_scratchpad_trace()
    if complete_trace:
        print(f"\n📝 COMPLETE SCRATCHPAD PREVIEW (first 2000 chars):")
        print("-" * 60)
        print(complete_trace[:2000])
        if len(complete_trace) > 2000:
            print(f"\n... ({len(complete_trace) - 2000:,} more characters in complete file)")
            print(f"💡 Access full trace via: response.get_scratchpad_trace()")
    
    # Get cycle summary
    cycle_summary = response.get_cycle_summary()
    if cycle_summary:
        print(f"\n📈 CYCLE SUMMARY:")
        print(f"   Start time: {cycle_summary.get('start_time')}")
        print(f"   End time: {cycle_summary.get('end_time')}")
        print(f"   Total entries: {cycle_summary.get('total_entries')}")
        print(f"   Iterations: {cycle_summary.get('iterations')}")
        print(f"   Tools used: {cycle_summary.get('tools_used')}")
        print(f"   Success: {cycle_summary.get('success')}")
    
    # Show final response
    print(f"\n🎯 FINAL RESPONSE:")
    print("-" * 60)
    print(response.content[:500] + ("..." if len(response.content) > 500 else ""))
    
    print(f"\n✨ OBSERVABILITY FEATURES DEMONSTRATED:")
    print(f"   ✅ Real-time phase change events")
    print(f"   ✅ Complete scratchpad persistence")  
    print(f"   ✅ Visual cues during reasoning")
    print(f"   ✅ Cycle-by-cycle breakdown")
    print(f"   ✅ NO truncation of agent traces")
    print(f"   ✅ Event subscription for external monitoring")
    
    return response

def demo_event_monitoring():
    """Demonstrate advanced event monitoring capabilities."""
    
    print(f"\n🎭 ADVANCED EVENT MONITORING DEMO")
    print("=" * 40)
    
    # Create session
    session = create_enhanced_session(
        provider="mlx", 
        model="mlx-community/GLM-4.5-Air-4bit",
        enable_memory=True
    )
    
    # Advanced event monitoring
    thinking_events = []
    tool_events = []
    
    def capture_thinking(event):
        thinking_events.append(event)
        print(f"💭 Captured thought: {event.content[:50]}...")
        
    def capture_tools(event):
        tool_events.append(event)
        tool_name = event.metadata.get('tool_name', 'unknown')
        print(f"🔧 Captured tool call: {tool_name}")
    
    # Subscribe to specific events
    session.scratchpad.subscribe_to_events(ReActPhase.THINKING, capture_thinking)
    session.scratchpad.subscribe_to_events(ReActPhase.ACTING, capture_tools)
    
    # Use context manager for monitoring
    with session.scratchpad.monitor_phase(ReActPhase.OBSERVING) as observations:
        response = session.generate("What's in this directory?", tools=[list_files])
    
    print(f"\n📊 EVENT MONITORING RESULTS:")
    print(f"   Thinking events captured: {len(thinking_events)}")
    print(f"   Tool events captured: {len(tool_events)}")
    print(f"   Observation events: {len(observations)}")
    
    return response

if __name__ == "__main__":
    print("Starting complete observability demonstration...")
    
    # Demo 1: Complete observability with real-time events
    response = demo_complete_observability()
    
    # Demo 2: Advanced event monitoring
    response2 = demo_event_monitoring()
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print(f"   All agent reasoning is now fully observable!")
    print(f"   Complete scratchpad files created in ./demo_memory/")
    print(f"   Real-time events demonstrated!")
    print(f"   SOTA observability achieved! 🚀")