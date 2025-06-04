#!/usr/bin/env python3
"""
Simple test for MLX provider to verify it works correctly.
"""

from abstractllm import create_llm

def main():
    """Test basic MLX provider functionality."""
    
    print("🧪 Testing MLX Provider")
    print("=" * 30)
    
    try:
        # Create MLX provider
        provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
        
        # Test basic generation
        print("📝 Testing basic text generation...")
        response = provider.generate(
            prompt="What is 2+2? Explain your reasoning.",
            max_tokens=1024
        )
        
        print(f"✅ Response type: {type(response)}")
        print(f"📄 Response: {str(response)[:200]}...")
        
        # Test with system prompt
        print("\n📝 Testing with system prompt...")
        response2 = provider.generate(
            prompt="Tell me about artificial intelligence.",
            system_prompt="You are a helpful AI assistant.",
            max_tokens=1024
        )
        
        print(f"✅ Response type: {type(response2)}")
        print(f"📄 Response: {str(response2)[:200]}...")
        
        print(f"\n✅ MLX Provider tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 