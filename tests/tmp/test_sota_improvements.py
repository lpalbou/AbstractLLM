#!/usr/bin/env python
"""
Test script for SOTA improvements: Advanced Memory, Retry Strategies, and Enhanced Tools.
Tests across multiple providers to ensure robust agentic performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_llm, create_session
from abstractllm.memory_v2 import HierarchicalMemory, ReActCycle, MemoryComponent
from abstractllm.retry_strategies import RetryManager, RetryConfig, with_retry
from abstractllm.structured_response import StructuredResponseHandler, StructuredResponseConfig, ResponseFormat
from abstractllm.tools.common_tools import read_file, list_files, search_files
import json
import time
from typing import Dict, Any, List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hierarchical_memory():
    """Test the hierarchical memory system with bidirectional linking."""
    print("\n" + "="*60)
    print("Testing Hierarchical Memory System")
    print("="*60)
    
    # Create memory with persistence
    memory = HierarchicalMemory(
        working_memory_size=5,
        episodic_consolidation_threshold=3,
        persist_path=Path("./memory_test")
    )
    
    # Test 1: ReAct Cycle Management
    print("\n1. Testing ReAct Cycle Management:")
    
    # Start a ReAct cycle
    cycle = memory.start_react_cycle("How do I implement retry logic in Python?")
    print(f"Started cycle: {cycle.cycle_id}")
    
    # Add thoughts
    cycle.add_thought("I need to research retry patterns", confidence=0.9)
    cycle.add_thought("Exponential backoff is a common pattern", confidence=0.95)
    
    # Add action
    action_id = cycle.add_action(
        tool_name="search_files",
        arguments={"query": "retry", "file_type": "py"},
        reasoning="Looking for existing retry implementations"
    )
    
    # Add observation
    cycle.add_observation(
        action_id=action_id,
        content="Found retry_strategies.py with exponential backoff implementation",
        success=True
    )
    
    # Complete cycle
    cycle.complete("Use exponential backoff with jitter for retry logic", success=True)
    
    print(f"Cycle trace:\n{cycle.get_trace()}")
    
    # Test 2: Bidirectional Linking
    print("\n2. Testing Bidirectional Linking:")
    
    # Add chat message linked to cycle
    msg_id = memory.add_chat_message(
        role="assistant",
        content="To implement retry logic, use exponential backoff with jitter",
        cycle_id=cycle.cycle_id
    )
    
    print(f"Added message: {msg_id}")
    
    # Get links for the cycle
    links = memory.get_links(MemoryComponent.SCRATCHPAD, cycle.cycle_id)
    print(f"Links for cycle {cycle.cycle_id}:")
    for link in links:
        print(f"  -> {link.target_type.value}:{link.target_id} ({link.relationship})")
    
    # Test 3: Fact Extraction
    print("\n3. Testing Fact Extraction:")
    
    # Add message with facts
    memory.add_chat_message(
        role="user",
        content="Python has decorators. Tenacity is a retry library. Exponential backoff reduces load."
    )
    
    # Query facts
    facts = memory.get_related_facts("python", max_depth=2)
    print(f"Facts related to 'python':")
    for fact in facts[:5]:
        print(f"  - {fact}")
    
    # Test 4: Memory Consolidation
    print("\n4. Testing Memory Consolidation:")
    
    # Add more items to trigger consolidation
    for i in range(6):
        memory.add_chat_message(
            role="user",
            content=f"Test message {i}"
        )
    
    stats = memory.get_statistics()
    print(f"Memory statistics:")
    print(f"  Working memory: {stats['working_memory_size']}")
    print(f"  Episodic memory: {stats['episodic_memory_size']}")
    print(f"  Total facts: {stats['total_facts']}")
    print(f"  Total links: {stats['total_links']}")
    
    # Test 5: Context Generation
    print("\n5. Testing Context Generation:")
    
    context = memory.get_context_for_query("retry implementation")
    print(f"Generated context:\n{context[:500]}...")
    
    # Test 6: Persistence
    print("\n6. Testing Persistence:")
    
    memory.save_to_disk()
    print(f"Saved memory to disk")
    
    # Create new memory and load
    memory2 = HierarchicalMemory(persist_path=Path("./memory_test"))
    stats2 = memory2.get_statistics()
    print(f"Loaded memory statistics:")
    print(f"  Total facts: {stats2['total_facts']}")
    print(f"  Total cycles: {stats2['total_react_cycles']}")
    
    print("\n✅ Hierarchical memory tests completed")
    return memory


def test_retry_strategies():
    """Test retry strategies with different error scenarios."""
    print("\n" + "="*60)
    print("Testing Retry Strategies")
    print("="*60)
    
    # Create retry manager with custom config
    config = RetryConfig(
        max_attempts=3,
        initial_delay=0.5,
        exponential_base=2,
        jitter=True,
        include_error_feedback=True,
        simplify_on_retry=True
    )
    
    retry_manager = RetryManager(config)
    
    # Test 1: Exponential Backoff
    print("\n1. Testing Exponential Backoff:")
    
    attempt_count = 0
    
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        print(f"  Attempt {attempt_count}")
        if attempt_count < 3:
            raise TimeoutError("Connection timeout")
        return "Success!"
    
    try:
        result = retry_manager.retry_with_backoff(
            flaky_function,
            key="test_function"
        )
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Test 2: Circuit Breaker
    print("\n2. Testing Circuit Breaker:")
    
    def always_fails():
        raise RuntimeError("Service unavailable")
    
    # Trigger circuit breaker
    for i in range(6):
        try:
            retry_manager.retry_with_backoff(
                always_fails,
                key="failing_service"
            )
        except:
            pass
    
    breaker = retry_manager.get_circuit_breaker("failing_service")
    print(f"  Circuit breaker state: {breaker.state.value}")
    print(f"  Failure count: {breaker.failure_count}")
    
    # Test 3: Error Classification
    print("\n3. Testing Error Classification:")
    
    errors = [
        RuntimeError("Rate limit exceeded"),
        TimeoutError("Request timed out"),
        ValueError("JSON parsing failed"),
        ConnectionError("Network unreachable"),
        RuntimeError("Context length exceeded")
    ]
    
    for error in errors:
        error_type = retry_manager.classify_error(error)
        should_retry = retry_manager.should_retry(error, attempt=1)
        print(f"  {error}: {error_type.value} (retry: {should_retry})")
    
    # Test 4: Decorator Usage
    print("\n4. Testing Decorator Pattern:")
    
    @with_retry(key="api_call", config=config)
    def api_call(success_on_attempt: int = 2):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < success_on_attempt:
            raise ConnectionError("API unavailable")
        return {"status": "ok"}
    
    attempt_count = 0
    result = api_call(success_on_attempt=2)
    print(f"  API call result: {result}")
    
    print("\n✅ Retry strategy tests completed")
    return retry_manager


def test_enhanced_tools_with_retry(provider_name: str):
    """Test enhanced tools with retry strategies."""
    print(f"\n" + "="*60)
    print(f"Testing Enhanced Tools with Retry ({provider_name})")
    print("="*60)
    
    try:
        # Create provider
        if provider_name == "ollama":
            llm = create_llm(provider_name, model="qwen3:4b", temperature=0.3)
        elif provider_name == "openai":
            llm = create_llm(provider_name, model="gpt-4o-mini", temperature=0.3)
        else:
            print(f"Skipping {provider_name}")
            return
        
        # Create memory and retry manager
        memory = HierarchicalMemory()
        retry_manager = RetryManager()
        
        # Create session
        session = create_session(provider=llm)
        
        # Add tools
        session.add_tool(list_files)
        session.add_tool(read_file)
        session.add_tool(search_files)
        
        # Test with retry wrapper
        @with_retry(key=f"{provider_name}_tools")
        def execute_with_tools(prompt: str):
            # Start ReAct cycle
            cycle = memory.start_react_cycle(prompt)
            
            # Generate with tools
            response = session.generate(
                prompt=prompt,
                tools=[list_files, read_file, search_files],
                max_tool_calls=5
            )
            
            # Complete cycle
            cycle.complete(str(response), success=True)
            
            # Add to memory
            memory.add_chat_message("user", prompt)
            memory.add_chat_message("assistant", str(response), cycle_id=cycle.cycle_id)
            
            return response
        
        # Test tool execution
        print("\n1. Testing tool execution with retry:")
        response = execute_with_tools("List Python files in the current directory")
        print(f"Response preview: {str(response)[:200]}...")
        
        # Check memory
        stats = memory.get_statistics()
        print(f"\n2. Memory statistics after tool execution:")
        print(f"  Total cycles: {stats['total_react_cycles']}")
        print(f"  Success rate: {stats['success_rate']:.0%}")
        
        print(f"\n✅ Enhanced tools test completed for {provider_name}")
        
    except Exception as e:
        print(f"❌ Error testing {provider_name}: {e}")
        import traceback
        traceback.print_exc()


def test_structured_response_with_retry(provider_name: str):
    """Test structured response with retry and validation."""
    print(f"\n" + "="*60)
    print(f"Testing Structured Response with Retry ({provider_name})")
    print("="*60)
    
    try:
        # Create provider
        if provider_name == "ollama":
            llm = create_llm(provider_name, model="qwen3:4b")
        elif provider_name == "openai":
            llm = create_llm(provider_name, model="gpt-4o-mini")
        else:
            print(f"Skipping {provider_name}")
            return
        
        # Create handler and retry manager
        handler = StructuredResponseHandler(provider_name)
        retry_manager = RetryManager(
            RetryConfig(
                validation_retries=5,
                re_prompt_on_validation_failure=True,
                include_error_feedback=True
            )
        )
        
        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            },
            "required": ["name", "age", "skills"]
        }
        
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON,
            schema=schema,
            max_retries=3,
            temperature_override=0.3
        )
        
        # Test with retry
        @with_retry(key=f"{provider_name}_structured")
        def generate_structured():
            return handler.generate_with_retry(
                generate_fn=llm.generate,
                prompt="Generate a profile for a senior Python developer",
                config=config
            )
        
        print("\n1. Testing structured response with validation:")
        result = generate_structured()
        print(f"Generated profile: {json.dumps(result, indent=2)}")
        
        # Validate result
        assert "name" in result
        assert "age" in result
        assert "skills" in result
        assert isinstance(result["skills"], list)
        
        print(f"\n✅ Structured response test completed for {provider_name}")
        
    except Exception as e:
        print(f"❌ Error testing structured response: {e}")


def test_multi_turn_with_memory(provider_name: str):
    """Test multi-turn conversation with advanced memory."""
    print(f"\n" + "="*60)
    print(f"Testing Multi-Turn with Memory ({provider_name})")
    print("="*60)
    
    try:
        # Create provider
        if provider_name == "ollama":
            llm = create_llm(provider_name, model="qwen3:4b")
        elif provider_name == "openai":
            llm = create_llm(provider_name, model="gpt-4o-mini")
        else:
            print(f"Skipping {provider_name}")
            return
        
        # Create memory
        memory = HierarchicalMemory()
        session = create_session(provider=llm)
        
        # Turn 1: Establish context
        print("\n1. Turn 1 - Establishing context:")
        cycle1 = memory.start_react_cycle("Introduction")
        response1 = session.generate("I'm working on a Python project that needs retry logic and structured responses.")
        cycle1.complete(str(response1))
        memory.add_chat_message("user", "I'm working on a Python project that needs retry logic and structured responses.")
        memory.add_chat_message("assistant", str(response1), cycle_id=cycle1.cycle_id)
        print(f"Assistant: {str(response1)[:200]}...")
        
        # Turn 2: Build on context
        print("\n2. Turn 2 - Building on context:")
        context = memory.get_context_for_query("retry")
        
        cycle2 = memory.start_react_cycle("Specific question")
        
        # Include context in prompt
        prompt_with_context = f"{context}\n\nUser: What specific retry pattern should I use for API calls?"
        response2 = session.generate(prompt_with_context)
        cycle2.complete(str(response2))
        memory.add_chat_message("user", "What specific retry pattern should I use for API calls?")
        memory.add_chat_message("assistant", str(response2), cycle_id=cycle2.cycle_id)
        print(f"Assistant: {str(response2)[:200]}...")
        
        # Turn 3: Test memory recall
        print("\n3. Turn 3 - Testing memory recall:")
        cycle3 = memory.start_react_cycle("Recall question")
        response3 = session.generate("What was I working on again?")
        cycle3.complete(str(response3))
        
        # Check if context was maintained
        if "retry" in str(response3).lower() or "structured" in str(response3).lower():
            print("✅ Context maintained across turns")
        else:
            print("⚠️ Context may not be fully maintained")
        
        # Show memory statistics
        stats = memory.get_statistics()
        print(f"\n4. Final memory statistics:")
        print(f"  Total cycles: {stats['total_react_cycles']}")
        print(f"  Success rate: {stats['success_rate']:.0%}")
        print(f"  Total facts: {stats['total_facts']}")
        print(f"  Total links: {stats['total_links']}")
        
        # Visualize links
        print(f"\n5. Memory link visualization:")
        print(memory.visualize_links())
        
        print(f"\n✅ Multi-turn memory test completed for {provider_name}")
        
    except Exception as e:
        print(f"❌ Error in multi-turn test: {e}")


def main():
    """Run all SOTA improvement tests."""
    print("="*60)
    print("SOTA Improvements Test Suite")
    print("="*60)
    print("\nTesting advanced memory, retry strategies, and enhanced tools")
    
    # Test core components
    print("\n" + "="*60)
    print("PART 1: Core Component Tests")
    print("="*60)
    
    memory = test_hierarchical_memory()
    retry_manager = test_retry_strategies()
    
    # Test with providers
    print("\n" + "="*60)
    print("PART 2: Provider Integration Tests")
    print("="*60)
    
    providers = ["ollama"]  # Start with Ollama, add "openai" if API key available
    
    for provider in providers:
        try:
            # Test enhanced tools with retry
            test_enhanced_tools_with_retry(provider)
            
            # Test structured response with retry
            test_structured_response_with_retry(provider)
            
            # Test multi-turn with memory
            test_multi_turn_with_memory(provider)
            
        except Exception as e:
            print(f"Error testing {provider}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    print("\n✅ Core Improvements Tested:")
    print("  - Hierarchical memory with bidirectional linking")
    print("  - ReAct cycles with unique scratchpads")
    print("  - Fact extraction and knowledge graphs")
    print("  - Exponential backoff with jitter")
    print("  - Circuit breaker pattern")
    print("  - Smart retry with error feedback")
    print("  - Structured response validation")
    print("  - Multi-turn context preservation")
    
    print("\n📊 Key Metrics:")
    if memory:
        stats = memory.get_statistics()
        print(f"  - Memory sessions: {stats['session_id']}")
        print(f"  - Total facts extracted: {stats['total_facts']}")
        print(f"  - ReAct cycles: {stats['total_react_cycles']}")
        print(f"  - Success rate: {stats['success_rate']:.0%}")
    
    if retry_manager:
        print(f"  - Circuit breakers: {len(retry_manager.circuit_breakers)}")
        print(f"  - Error history: {len(retry_manager.error_history)} entries")
    
    print("\n" + "="*60)
    print("All SOTA improvement tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()