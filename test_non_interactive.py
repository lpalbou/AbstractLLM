#!/usr/bin/env python3
"""
Test script for MLX provider in non-interactive mode.
"""

from abstractllm import create_llm

def test_mlx_provider():
    """Test MLX provider basic functionality."""
    
    print("🧪 Testing MLX Provider Configuration")
    print("=" * 40)
    
    try:
        # Create MLX provider with proper configuration
        provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
        
        print("✅ Provider created successfully")
        
        # Test basic generation with explicit parameters
        print("\n📝 Testing basic generation...")
        response = provider.generate(
            prompt="What is 2+2? Think step by step.",
            max_tokens=512,
            temperature=0.1
        )
        
        print(f"Response type: {type(response)}")
        print(f"Content: {response.content[:200]}...")
        
        # Test with system prompt
        print("\n🎯 Testing with system prompt...")
        response2 = provider.generate(
            prompt="Explain gravity in simple terms.",
            system_prompt="You are a physics teacher explaining concepts to students.",
            max_tokens=512,
            temperature=0.3
        )
        
        print(f"Response 2 type: {type(response2)}")
        print(f"Content: {response2.content[:200]}...")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mlx_provider() 