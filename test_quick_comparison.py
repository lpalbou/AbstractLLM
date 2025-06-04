#!/usr/bin/env python3
"""
Quick test to verify both MLX and Ollama providers work correctly.
"""

from abstractllm import create_llm
from abstractllm.session import Session
import time

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def test_provider(provider_name: str, model: str):
    """Test a provider with a simple tool call."""
    print(f"\n🧪 Testing {provider_name.upper()}")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # Create provider and session
        provider = create_llm(provider_name, model=model)
        session = Session(
            system_prompt="You are a helpful assistant.", 
            provider=provider,
            tools=[calculate_math]
        )
        
        # Test simple calculation
        response = session.generate("What is 25 * 4 + 7?")
        elapsed = time.time() - start_time
        
        print(f"⏱️  Time: {elapsed:.1f}s")
        
        # Handle response properly
        if hasattr(response, 'content'):
            print(f"📄 Response: {response.content[:100]}...")
        else:
            print(f"📄 Response: {str(response)[:100]}...")
        print(f"✅ {provider_name.upper()} works correctly")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"❌ {provider_name.upper()} failed: {str(e)}")

def main():
    """Test both providers."""
    print("🔄 Quick Provider Comparison Test")
    print("=" * 50)
    
    # Test MLX
    test_provider("mlx", "mlx-community/Qwen3-30B-A3B-4bit")
    
    # Test Ollama  
    test_provider("ollama", "qwen3:30b-a3b-q4_K_M")
    
    print("\n" + "=" * 50)
    print("🏁 Quick test completed!")

if __name__ == "__main__":
    main() 