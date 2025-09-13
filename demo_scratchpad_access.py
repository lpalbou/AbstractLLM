#!/usr/bin/env python
"""
Simple demonstration of accessing the COMPLETE scratchpad.
Shows how users can get full visibility into agent reasoning.
"""

from pathlib import Path
from abstractllm.factory_enhanced import create_enhanced_session
from abstractllm.tools.common_tools import list_files

def demo_scratchpad_access():
    """Demonstrate complete scratchpad access."""
    
    print("🔍 COMPLETE SCRATCHPAD ACCESS DEMO")
    print("=" * 50)
    
    # Create session
    session = create_enhanced_session(
        provider="mlx",
        model="mlx-community/GLM-4.5-Air-4bit",
        enable_memory=True,
        tools=[list_files]
    )
    
    # Simple query
    response = session.generate("List files here")
    
    print(f"✅ Query completed!")
    print(f"📂 Scratchpad file: {response.scratchpad_file}")
    print(f"🔗 ReAct cycle: {response.react_cycle_id}")
    
    # Access complete scratchpad
    if response.scratchpad_file and Path(response.scratchpad_file).exists():
        file_size = Path(response.scratchpad_file).stat().st_size
        print(f"📊 File size: {file_size:,} bytes")
        
        # Show complete trace
        complete_trace = response.get_scratchpad_trace()
        print(f"📝 Complete trace length: {len(complete_trace):,} characters")
        
        print("\n" + "="*50)
        print("🧠 COMPLETE AGENT REASONING TRACE:")
        print("="*50)
        print(complete_trace)
        print("="*50)
        
        # Access via scratchpad manager
        if response.scratchpad_manager:
            cycle_summary = response.scratchpad_manager.get_cycle_summary(response.react_cycle_id)
            print(f"\n📈 CYCLE SUMMARY:")
            for key, value in cycle_summary.items():
                print(f"   {key}: {value}")
    
    return response

if __name__ == "__main__":
    demo_scratchpad_access()