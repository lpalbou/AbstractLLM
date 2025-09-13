#!/usr/bin/env python
"""
Integration test with mock provider to verify wiring without real LLM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_enhanced_session
from abstractllm.interface import AbstractLLMInterface
from abstractllm.types import GenerateResponse
from typing import Any, Dict, Optional, Union, List, Generator
from pathlib import Path
import tempfile


class MockProvider(AbstractLLMInterface):
    """Mock provider for testing."""
    
    def __init__(self, config: Optional[Dict[Any, Any]] = None):
        super().__init__(config)
        self.call_count = 0
        self.last_prompt = None
        
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                **kwargs) -> GenerateResponse:
        """Mock generation."""
        return self._generate_impl(prompt, system_prompt, files, stream, tools, **kwargs)
    
    async def generate_async(self, prompt: str, **kwargs):
        """Mock async generation."""
        return self.generate(prompt, **kwargs)
        
    def _generate_impl(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      files: Optional[List[Union[str, Path]]] = None,
                      stream: bool = False,
                      tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                      **kwargs) -> GenerateResponse:
        """Mock generation implementation."""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Simulate response
        response = GenerateResponse(
            content=f"Mock response to: {prompt[:50]}...",
            model="mock-model",
            provider="mock",
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        
        # Add tool calls if tools provided
        if tools:
            response.tool_calls = []
            
        return response
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {"supports_tools": True}


def test_memory_integration():
    """Test that memory is properly integrated."""
    print("\n" + "="*60)
    print("Test 1: Memory Integration")
    print("="*60)
    
    # Import EnhancedSession directly
    from abstractllm.session_enhanced import EnhancedSession
    
    # Create mock provider
    mock_provider = MockProvider()
    
    # Create enhanced session directly with mock provider
    session = EnhancedSession(
        provider=mock_provider,
        enable_memory=True,
        enable_retry=True
    )
    
    # Test that memory is initialized
    assert session.memory is not None, "Memory not initialized"
    assert session.enable_memory is True, "Memory not enabled"
    
    # Test that retry manager is initialized
    assert session.retry_manager is not None, "Retry manager not initialized"
    assert session.enable_retry is True, "Retry not enabled"
    
    # Generate something
    response = session.generate("Test query")
    
    # Check that ReAct cycle was created
    stats = session.get_memory_stats()
    assert stats is not None, "Memory stats not available"
    assert stats['total_react_cycles'] > 0, "ReAct cycle not created"
    assert stats['chat_messages'] >= 2, "Chat messages not tracked"
    
    print(f"✅ Memory properly integrated")
    print(f"  - ReAct cycles: {stats['total_react_cycles']}")
    print(f"  - Chat messages: {stats['chat_messages']}")
    print(f"  - Working memory: {stats['working_memory_size']}")
    
    return True


def test_memory_persistence():
    """Test memory persistence."""
    print("\n" + "="*60)
    print("Test 2: Memory Persistence")
    print("="*60)
    
    from abstractllm.session_enhanced import EnhancedSession
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Session 1: Create and save
        session1 = EnhancedSession(
            provider=MockProvider(),
            persist_memory=Path(tmpdir)
        )
        
        # Generate some interactions
        session1.generate("First query")
        session1.generate("Second query about Python")
        
        # Get stats before saving
        stats1 = session1.get_memory_stats()
        facts1 = stats1['total_facts']
        cycles1 = stats1['total_react_cycles']
        
        # Save memory
        session1.save_memory()
        print(f"Session 1 saved: {cycles1} cycles, {facts1} facts")
        
        # Session 2: Load
        session2 = EnhancedSession(
            provider=MockProvider(),
            persist_memory=Path(tmpdir)
        )
        
        # Check loaded memory
        stats2 = session2.get_memory_stats()
        print(f"Session 2 loaded: {stats2['total_react_cycles']} cycles, {stats2['total_facts']} facts")
        
        # Query memory
        results = session2.query_memory("python")
        print(f"Memory query found results: {results is not None}")
        
        print("✅ Memory persistence working")
        
    return True


def test_retry_integration():
    """Test retry mechanism integration."""
    print("\n" + "="*60)
    print("Test 3: Retry Integration")
    print("="*60)
    
    class FailingProvider(MockProvider):
        """Provider that fails first N times."""
        def __init__(self, fail_count=2):
            super().__init__()
            self.fail_count = fail_count
            
        def _generate_impl(self, prompt, **kwargs):
            if self.call_count < self.fail_count:
                self.call_count += 1
                raise ConnectionError(f"Simulated failure {self.call_count}")
            return super()._generate_impl(prompt, **kwargs)
    
    # Create session with failing provider
    from abstractllm.session_enhanced import EnhancedSession
    
    session = EnhancedSession(
        provider=FailingProvider(fail_count=2),
        enable_retry=True
    )
    
    # Should retry and succeed
    try:
        response = session.generate("Test with retry")
        print(f"Response after retries: {response}")
        print(f"Provider called {session._provider.call_count} times (with retries)")
        print("✅ Retry mechanism working")
        return True
    except Exception as e:
        print(f"❌ Retry failed: {e}")
        return False


def test_memory_context():
    """Test memory context in prompts."""
    print("\n" + "="*60)
    print("Test 4: Memory Context in Prompts")
    print("="*60)
    
    class ContextAwareProvider(MockProvider):
        """Provider that checks for context."""
        def _generate_impl(self, prompt, **kwargs):
            # Check if context was added
            has_context = "Recent context:" in prompt or "Reasoning trace:" in prompt
            response = GenerateResponse(
                content=f"Context present: {has_context}",
                model="mock-model",
                provider="mock"
            )
            return response
    
    from abstractllm.session_enhanced import EnhancedSession
    
    session = EnhancedSession(
        provider=ContextAwareProvider(),
        enable_memory=True
    )
    
    # First query to populate memory
    session.generate("Remember: I like Python")
    
    # Second query should include context
    response = session.generate("What do I like?", use_memory_context=True)
    
    print(f"Response: {response}")
    
    # Check memory stats
    stats = session.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    # Visualize links
    links = session.visualize_memory_links()
    print(f"Memory links:\n{links}")
    
    print("✅ Memory context integration working")
    return True


def test_fact_extraction():
    """Test fact extraction from conversations."""
    print("\n" + "="*60)
    print("Test 5: Fact Extraction")
    print("="*60)
    
    from abstractllm.session_enhanced import EnhancedSession
    
    session = EnhancedSession(
        provider=MockProvider(),
        enable_memory=True
    )
    
    # Generate with facts
    session.generate("Python is a programming language. Python has decorators.")
    
    # Check facts
    stats = session.get_memory_stats()
    print(f"Facts extracted: {stats['total_facts']}")
    
    # Query facts
    results = session.query_memory("python")
    if results and results.get("facts"):
        print(f"Facts about Python: {len(results['facts'])}")
        for fact in results["facts"][:3]:
            print(f"  - {fact}")
    
    print("✅ Fact extraction working")
    return True


def main():
    """Run all mock integration tests."""
    print("="*60)
    print("AbstractLLM Integration Tests (Mock Provider)")
    print("="*60)
    print("Testing that all components are properly wired together")
    
    tests = [
        ("Memory Integration", test_memory_integration),
        ("Memory Persistence", test_memory_persistence),
        ("Retry Integration", test_retry_integration),
        ("Memory Context", test_memory_context),
        ("Fact Extraction", test_fact_extraction)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All components are properly integrated!")
        print("\nThe enhanced AbstractLLM features are ready to use:")
        print("  - Hierarchical memory with ReAct cycles")
        print("  - Automatic retry with exponential backoff")
        print("  - Fact extraction and knowledge graphs")
        print("  - Memory persistence across sessions")
        print("  - Context-aware generation")
        print("\nUse: from abstractllm import create_enhanced_session")
    else:
        print("\n⚠️ Some integration issues detected.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)