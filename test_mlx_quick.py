#!/usr/bin/env python
"""Quick test for MLX model loading."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_llm
import time

print("Testing MLX with Qwen3 (known working model)...")
start = time.time()

try:
    # Use Qwen3 which we know works
    llm = create_llm(
        "mlx",
        model="mlx-community/Qwen3-30B-A3B-4bit",
        temperature=0.7,
        max_tokens=50
    )
    
    print(f"Model loaded in {time.time() - start:.2f}s")
    
    # Quick test
    response = llm.generate("Say hello in one word")
    print(f"Response: {response}")
    print("✅ MLX provider working with Qwen3")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()