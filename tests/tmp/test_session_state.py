#!/usr/bin/env python3
"""
Test what changes between first and second generation.
"""

import sys
sys.path.insert(0, '.')

from abstractllm.factory import create_session

def test_session_state():
    """Test what changes in session state between generations."""

    print("🔍 Testing Session State Changes")
    print("=" * 60)

    # Create ONE session
    config = {
        'model': 'qwen3:4b',
        'temperature': 0.0,
        'seed': 123,
        'max_tokens': 50,
        'enable_memory': True,  # Keep memory enabled to match CLI
        'tools': [],
        'system_prompt': "You are an AI assistant."
    }

    session = create_session("ollama", **config)
    prompt = "who are you?"

    print("📊 Testing SAME session, TWO consecutive calls:")
    print()

    # First call
    print("🔄 Call 1:")
    response1 = session.generate(prompt)
    content1 = response1.content if hasattr(response1, 'content') else str(response1)
    print(f"  Response: {content1[:100]}...")

    # Check session state
    if hasattr(session, 'memory') and session.memory:
        print(f"  Memory after call 1: {len(session.memory.working_memory)} items in working memory")
        if hasattr(session.memory, 'current_cycle'):
            print(f"  Current cycle: {session.memory.current_cycle}")

    # Second call - SAME session
    print("\n🔄 Call 2 (same session):")
    response2 = session.generate(prompt)
    content2 = response2.content if hasattr(response2, 'content') else str(response2)
    print(f"  Response: {content2[:100]}...")

    # Check session state again
    if hasattr(session, 'memory') and session.memory:
        print(f"  Memory after call 2: {len(session.memory.working_memory)} items in working memory")

    # Compare
    print(f"\n{'='*60}")
    if content1 == content2:
        print("✅ Identical responses from same session")
    else:
        print("❌ Different responses from same session!")

        # Find differences
        min_len = min(len(content1), len(content2))
        for i in range(min_len):
            if content1[i] != content2[i]:
                print(f"\nFirst difference at position {i}:")
                print(f"  Call 1: '{content1[max(0,i-20):i+20]}'")
                print(f"  Call 2: '{content2[max(0,i-20):i+20]}'")
                break

    # Now test with TWO separate sessions
    print(f"\n{'='*60}")
    print("📊 Testing TWO separate sessions:")
    print()

    session1 = create_session("ollama", **config)
    print("🔄 Session 1:")
    resp1 = session1.generate(prompt)
    cont1 = resp1.content if hasattr(resp1, 'content') else str(resp1)
    print(f"  Response: {cont1[:100]}...")

    session2 = create_session("ollama", **config)
    print("\n🔄 Session 2:")
    resp2 = session2.generate(prompt)
    cont2 = resp2.content if hasattr(resp2, 'content') else str(resp2)
    print(f"  Response: {cont2[:100]}...")

    print(f"\n{'='*60}")
    if cont1 == cont2:
        print("✅ Identical responses from separate sessions")
    else:
        print("❌ Different responses from separate sessions!")

if __name__ == "__main__":
    test_session_state()