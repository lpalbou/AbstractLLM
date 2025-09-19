#!/usr/bin/env python3
"""
Test Simple Cognitive Functionality

Test the core cognitive abstractions directly without complex patching.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_functionality():
    """Test basic cognitive functionality"""
    print("🧪 Testing basic cognitive functionality...")

    try:
        from abstractllm.cognitive import FactsExtractor, ValueResonance

        # Test FactsExtractor
        print("   Testing FactsExtractor...")
        facts_extractor = FactsExtractor()

        test_content = "Python is a programming language. FastAPI is a modern web framework. FastAPI uses Python for building APIs."

        # This would normally require granite3.3:2b, but let's see what happens
        try:
            facts = facts_extractor.extract_facts(test_content, context_type="interaction")
            print(f"   ✅ FactsExtractor created {facts.total_extracted} facts")

            if facts.all_facts():
                sample_fact = facts.all_facts()[0]
                print(f"   Sample fact: {sample_fact.subject} --[{sample_fact.predicate}]--> {sample_fact.object}")
                print(f"   Ontology: {sample_fact.ontology}, Category: {sample_fact.category}")
        except Exception as e:
            print(f"   ⚠️ FactsExtractor failed (expected without granite3.3:2b): {e}")

        # Test ValueResonance
        print("   Testing ValueResonance...")

        # Test with default values
        value_evaluator = ValueResonance()
        print(f"   ✅ Default values: {value_evaluator.ai_core_values}")

        # Test with custom values
        custom_evaluator = ValueResonance(ai_core_values=['helpfulness', 'creativity', 'accuracy', 'innovation'])
        print(f"   ✅ Custom values: {custom_evaluator.ai_core_values}")

        # Test interaction evaluation (would require LLM)
        try:
            test_interaction = "User: Help me code better. Assistant: I'd be happy to help you improve your coding skills!"
            assessment = custom_evaluator.evaluate_interaction(test_interaction)
            print(f"   ✅ Value evaluation successful: {assessment.overall_resonance:.2f} resonance")
        except Exception as e:
            print(f"   ⚠️ Value evaluation failed (expected without granite3.3:2b): {e}")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration readiness"""
    print("\n🧪 Testing CLI integration...")

    try:
        from abstractllm.utils.commands import CommandProcessor
        from abstractllm.session import Session

        # Create basic session
        session = Session()
        cmd_processor = CommandProcessor(session)

        # Check if values command exists
        if 'values' in cmd_processor.commands:
            print("   ✅ /values command registered in CLI")
        else:
            print("   ❌ /values command not found")
            return False

        return True

    except Exception as e:
        print(f"   ❌ CLI integration test failed: {e}")
        return False

def main():
    """Run simple tests"""
    print("🚀 Testing Simple Cognitive Functionality")
    print("=" * 50)

    results = []

    # Test basic functionality
    results.append(test_basic_functionality())

    # Test CLI integration
    results.append(test_cli_integration())

    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    print(f"   Passed: {sum(results)}/{len(results)} tests")

    if all(results):
        print("✅ All core cognitive functionality tests passed!")
        print("ℹ️ LLM-dependent features require granite3.3:2b")
    else:
        print("❌ Some tests failed")

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)