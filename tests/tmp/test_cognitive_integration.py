#!/usr/bin/env python3
"""
Test Cognitive Integration

This script tests whether our cognitive abstractions are properly integrated
and working with AbstractLLM.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_cognitive_imports():
    """Test if cognitive modules can be imported and initialized"""
    print("🧪 Testing cognitive module imports...")

    try:
        from abstractllm.cognitive import Summarizer, FactsExtractor, ValueResonance
        print("✅ Core cognitive classes imported successfully")

        # Test ValueResonance with default values
        value_evaluator = ValueResonance()
        print(f"✅ ValueResonance created with default values: {value_evaluator.ai_core_values}")

        # Test ValueResonance with custom values
        custom_evaluator = ValueResonance(ai_core_values=['helpfulness', 'creativity', 'accuracy'])
        print(f"✅ ValueResonance created with custom values: {custom_evaluator.ai_core_values}")

        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_memory_patching():
    """Test if memory system is patched with cognitive fact extraction"""
    print("\n🧪 Testing memory system patching...")

    try:
        from abstractllm.memory import HierarchicalMemory

        # Create memory instance
        memory = HierarchicalMemory()

        # Test fact extraction with some content
        test_content = "Python is a programming language. FastAPI is a web framework. FastAPI uses Python."
        facts = memory.extract_facts(test_content, "test", "test_001")

        print(f"✅ Extracted {len(facts)} facts from test content")

        # Check if we got cognitive-style facts
        if facts:
            first_fact = facts[0]
            print(f"   Sample fact: {first_fact.subject} --[{first_fact.predicate}]--> {first_fact.object}")

            # Check if extraction method indicates cognitive processing
            if hasattr(first_fact, 'extraction_method') and 'cognitive' in first_fact.extraction_method:
                print("✅ Memory is using cognitive fact extraction")
                return True
            else:
                print("⚠️ Memory might still be using basic extraction")
                return False
        else:
            print("⚠️ No facts extracted")
            return False

    except Exception as e:
        print(f"❌ Memory patching test failed: {e}")
        return False

def test_value_resonance():
    """Test value resonance functionality"""
    print("\n🧪 Testing value resonance functionality...")

    try:
        from abstractllm.cognitive import ValueResonance

        # Create evaluator
        evaluator = ValueResonance()

        # Test evaluation
        test_interaction = """
        User: Can you help me write better code?
        Assistant: I'd be happy to help you write better code! Here are some key principles:
        1. Write clear, readable code with good variable names
        2. Add comments to explain complex logic
        3. Follow consistent formatting and style
        4. Write tests for your functions
        5. Keep functions small and focused
        """

        # Note: This will only work if granite3.3:2b is available
        print("   Attempting value evaluation (requires granite3.3:2b)...")
        assessment = evaluator.evaluate_interaction(test_interaction)

        print(f"✅ Value resonance evaluation successful!")
        print(f"   Overall resonance: {assessment.overall_resonance:.2f}")
        print(f"   Number of value evaluations: {len(assessment.evaluations)}")

        # Show sample evaluation
        if assessment.evaluations:
            sample_eval = assessment.evaluations[0]
            print(f"   Sample: {sample_eval.format_output()}")

        return True

    except Exception as e:
        print(f"❌ Value resonance test failed: {e}")
        print("   (This is expected if granite3.3:2b is not available)")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Cognitive Integration for AbstractLLM")
    print("=" * 50)

    results = []

    # Test imports
    results.append(test_cognitive_imports())

    # Test memory patching
    results.append(test_memory_patching())

    # Test value resonance (may fail without granite3.3:2b)
    results.append(test_value_resonance())

    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    print(f"   Passed: {sum(results)}/{len(results)} tests")

    if all(results):
        print("✅ All cognitive integration tests passed!")
    elif results[0] and results[1]:  # At least imports and memory work
        print("✅ Core cognitive integration working!")
        print("ℹ️ Value resonance test may require granite3.3:2b model")
    else:
        print("❌ Some critical tests failed")

    return all(results[:2])  # Success if imports and memory work

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)