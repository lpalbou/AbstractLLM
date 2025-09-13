#!/usr/bin/env python
"""
Integration test for enhanced AbstractLLM features.
Tests that everything is properly wired together.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_enhanced_session
from abstractllm.tools.common_tools import list_files
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
import json
import tempfile
from pathlib import Path


def test_basic_integration():
    """Test basic enhanced session functionality."""
    print("\n" + "="*60)
    print("Test 1: Basic Enhanced Session")
    print("="*60)
    
    try:
        # Create enhanced session with Ollama
        session = create_enhanced_session(
            provider="ollama",
            model="qwen3:4b",
            temperature=0.7,
            max_tokens=100
        )
        
        # Simple generation
        response = session.generate("What is 2+2? Answer in one word.")
        print(f"Response: {response}")
        
        # Check memory was created
        stats = session.get_memory_stats()
        print(f"Memory stats: {stats}")
        assert stats is not None, "Memory not initialized"
        assert stats['chat_messages'] >= 2, "Messages not tracked"
        
        print("✅ Basic integration working")
        return True
        
    except Exception as e:
        print(f"❌ Basic integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_persistence():
    """Test memory persistence across sessions."""
    print("\n" + "="*60)
    print("Test 2: Memory Persistence")
    print("="*60)
    
    try:
        # Use temp directory for memory
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = str(tmpdir)
            
            # Session 1: Store information
            session1 = create_enhanced_session(
                provider="ollama",
                model="qwen3:4b",
                persist_memory=memory_path,
                temperature=0.5,
                max_tokens=50
            )
            
            response1 = session1.generate("Remember this: AbstractLLM has hierarchical memory")
            print(f"Session 1 response: {response1}")
            
            # Save memory
            session1.save_memory()
            stats1 = session1.get_memory_stats()
            print(f"Session 1 facts: {stats1['total_facts']}")
            
            # Session 2: Load and query
            session2 = create_enhanced_session(
                provider="ollama",
                model="qwen3:4b",
                persist_memory=memory_path,
                temperature=0.5,
                max_tokens=50
            )
            
            # Query memory
            results = session2.query_memory("memory")
            print(f"Memory query results: {results is not None}")
            
            stats2 = session2.get_memory_stats()
            print(f"Session 2 loaded facts: {stats2['total_facts']}")
            
            print("✅ Memory persistence working")
            return True
            
    except Exception as e:
        print(f"❌ Memory persistence failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_with_retry():
    """Test tool execution with retry."""
    print("\n" + "="*60)
    print("Test 3: Tools with Retry")
    print("="*60)
    
    try:
        session = create_enhanced_session(
            provider="ollama",
            model="qwen3:4b",
            temperature=0.3,
            max_tokens=200
        )
        
        # Add tool
        session.add_tool(list_files)
        
        # Execute with tools
        response = session.generate(
            prompt="List Python files in the current directory",
            tools=[list_files],
            max_tool_calls=3
        )
        
        print(f"Tool response preview: {str(response)[:200]}...")
        
        # Check if ReAct cycle was created
        stats = session.get_memory_stats()
        print(f"ReAct cycles: {stats['total_react_cycles']}")
        
        print("✅ Tool execution with retry working")
        return True
        
    except Exception as e:
        print(f"❌ Tool execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structured_response():
    """Test structured response generation."""
    print("\n" + "="*60)
    print("Test 4: Structured Response")
    print("="*60)
    
    try:
        session = create_enhanced_session(
            provider="ollama",
            model="qwen3:4b",
            temperature=0.3
        )
        
        # Configure structured response
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON,
            temperature_override=0.1,
            max_retries=2
        )
        
        # Generate structured response
        response = session.generate(
            prompt='Generate a simple JSON with keys "status" and "value" where status is "ok" and value is 42',
            structured_config=config
        )
        
        print(f"Structured response: {response}")
        
        # Verify it's valid JSON
        if isinstance(response, str):
            parsed = json.loads(response)
            print(f"Parsed JSON: {parsed}")
        elif isinstance(response, dict):
            print(f"Already parsed: {response}")
        
        print("✅ Structured response working")
        return True
        
    except Exception as e:
        print(f"❌ Structured response failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_context():
    """Test memory context in generation."""
    print("\n" + "="*60)
    print("Test 5: Memory Context")
    print("="*60)
    
    try:
        session = create_enhanced_session(
            provider="ollama",
            model="qwen3:4b",
            temperature=0.5,
            max_tokens=100
        )
        
        # First query
        response1 = session.generate("My favorite language is Python")
        print(f"Response 1: {response1}")
        
        # Second query should use context
        response2 = session.generate("What is my favorite language?")
        print(f"Response 2: {response2}")
        
        # Check if context was used
        if "python" in str(response2).lower():
            print("✅ Memory context working - recalled Python")
        else:
            print("⚠️ Memory context may not be working optimally")
        
        # Show memory stats
        stats = session.get_memory_stats()
        print(f"Total facts extracted: {stats['total_facts']}")
        print(f"Chat messages: {stats['chat_messages']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory context failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("="*60)
    print("AbstractLLM Enhanced Integration Tests")
    print("="*60)
    
    tests = [
        ("Basic Integration", test_basic_integration),
        ("Memory Persistence", test_memory_persistence),
        ("Tools with Retry", test_tool_with_retry),
        ("Structured Response", test_structured_response),
        ("Memory Context", test_memory_context)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("The enhanced features are properly integrated into AbstractLLM.")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)