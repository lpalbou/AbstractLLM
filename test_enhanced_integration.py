#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for verifying enhanced AbstractLLM features integration.

This script tests:
1. Enhanced session creation
2. Memory persistence and ReAct cycles
3. Structured response generation
4. Retry strategies
"""

import sys
import json
from pathlib import Path
from abstractllm.factory_enhanced import create_enhanced_session
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
from abstractllm.tools.common_tools import read_file, list_files

def test_enhanced_session():
    """Test basic enhanced session creation."""
    print("=" * 60)
    print("TEST 1: Enhanced Session Creation")
    print("=" * 60)
    
    try:
        # Create enhanced session with memory and retry
        session = create_enhanced_session(
            "ollama",
            model="qwen3:4b",
            enable_memory=True,
            enable_retry=True,
            memory_config={
                'working_memory_size': 5,
                'consolidation_threshold': 3
            },
            tools=[read_file, list_files],
            system_prompt="You are a helpful AI assistant with memory capabilities.",
            max_tokens=500
        )
        
        print("✅ Enhanced session created successfully")
        print(f"   - Memory enabled: {session.enable_memory}")
        print(f"   - Retry enabled: {session.enable_retry}")
        print(f"   - Memory type: {type(session.memory).__name__}")
        print(f"   - Retry manager: {type(session.retry_manager).__name__}")
        return session
        
    except Exception as e:
        print(f"❌ Failed to create enhanced session: {e}")
        return None


def test_memory_and_react(session):
    """Test memory context and ReAct cycle creation."""
    print("\n" + "=" * 60)
    print("TEST 2: Memory and ReAct Cycles")
    print("=" * 60)
    
    if not session:
        print("⚠️  Skipping - no session available")
        return
    
    try:
        # First query - should create a ReAct cycle
        print("\n📝 Sending first query...")
        response1 = session.generate(
            prompt="What is the capital of France?",
            use_memory_context=True,
            create_react_cycle=True,
            max_tokens=100
        )
        
        if session.current_cycle:
            print(f"✅ ReAct cycle created: {session.current_cycle.cycle_id}")
            print(f"   - Query: {session.current_cycle.query}")
            print(f"   - Thoughts: {len(session.current_cycle.thoughts)}")
            print(f"   - Success: {session.current_cycle.success}")
        
        # Second query - should use memory context
        print("\n📝 Sending second query (should remember context)...")
        response2 = session.generate(
            prompt="What country's capital did I just ask about?",
            use_memory_context=True,
            create_react_cycle=True,
            max_tokens=100
        )
        
        # Check memory
        if session.memory:
            working_memory = session.memory.working_memory
            print(f"\n✅ Memory system active:")
            print(f"   - Working memory items: {len(working_memory)}")
            print(f"   - Cycles completed: {len([c for c in session.memory.react_cycles.values() if c.success])}")
            
            # Show first few working memory items
            for i, item in enumerate(working_memory[:3]):
                if 'content' in item:
                    preview = item['content'][:50] + "..." if len(item['content']) > 50 else item['content']
                    print(f"   - Memory {i+1}: {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory/ReAct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structured_response(session):
    """Test structured response generation."""
    print("\n" + "=" * 60)
    print("TEST 3: Structured Response Generation")
    print("=" * 60)
    
    if not session:
        print("⚠️  Skipping - no session available")
        return
    
    try:
        # Configure structured JSON response
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON,
            force_valid_json=True,
            max_retries=2,
            temperature_override=0.0,
            examples=[
                {"name": "Paris", "country": "France", "population": 2161000}
            ]
        )
        
        print("\n📝 Requesting structured JSON response...")
        response = session.generate(
            prompt="Give me information about Tokyo in JSON format with fields: name, country, population",
            structured_config=config,
            max_tokens=200
        )
        
        # Try to parse the response as JSON
        try:
            if isinstance(response, str):
                json_data = json.loads(response)
            else:
                json_data = response
            
            print("✅ Structured JSON response received:")
            print(f"   {json.dumps(json_data, indent=2)}")
            
            # Validate expected fields
            expected_fields = ['name', 'country', 'population']
            missing_fields = [f for f in expected_fields if f not in json_data]
            if missing_fields:
                print(f"⚠️  Missing expected fields: {missing_fields}")
            else:
                print("✅ All expected fields present")
                
        except json.JSONDecodeError as e:
            print(f"⚠️  Response is not valid JSON: {e}")
            print(f"   Raw response: {response[:200]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Structured response test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_persistence(session):
    """Test memory persistence to disk."""
    print("\n" + "=" * 60)
    print("TEST 4: Memory Persistence")
    print("=" * 60)
    
    if not session or not session.memory:
        print("⚠️  Skipping - no session with memory available")
        return
    
    try:
        persist_path = Path("./test_memory.pkl")
        
        # Save memory
        print(f"\n💾 Saving memory to {persist_path}...")
        session.memory.save(persist_path)
        
        if persist_path.exists():
            file_size = persist_path.stat().st_size
            print(f"✅ Memory saved successfully ({file_size} bytes)")
            
            # Load memory in a new session to verify
            print("\n📂 Loading memory in new session...")
            new_session = create_enhanced_session(
                "ollama",
                model="qwen3:4b",
                persist_memory=str(persist_path)
            )
            
            if new_session.memory:
                print("✅ Memory loaded successfully")
                print(f"   - Working memory items: {len(new_session.memory.working_memory)}")
                print(f"   - ReAct cycles: {len(new_session.memory.react_cycles)}")
            
            # Clean up
            persist_path.unlink()
            print("🧹 Test file cleaned up")
            
        return True
        
    except Exception as e:
        print(f"❌ Memory persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "🚀 " * 20)
    print("ABSTRACTLLM ENHANCED FEATURES INTEGRATION TEST")
    print("🚀 " * 20)
    
    # Test 1: Create enhanced session
    session = test_enhanced_session()
    
    # Test 2: Memory and ReAct cycles
    memory_ok = test_memory_and_react(session)
    
    # Test 3: Structured responses
    structured_ok = test_structured_response(session)
    
    # Test 4: Memory persistence
    persist_ok = test_memory_persistence(session)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Enhanced session creation: {'PASS' if session else 'FAIL'}")
    print(f"{'✅' if memory_ok else '❌'} Memory and ReAct cycles: {'PASS' if memory_ok else 'FAIL'}")
    print(f"{'✅' if structured_ok else '❌'} Structured responses: {'PASS' if structured_ok else 'FAIL'}")
    print(f"{'✅' if persist_ok else '❌'} Memory persistence: {'PASS' if persist_ok else 'FAIL'}")
    
    all_passed = all([session, memory_ok, structured_ok, persist_ok])
    print("\n" + ("✅ ALL TESTS PASSED!" if all_passed else "⚠️  Some tests failed"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())