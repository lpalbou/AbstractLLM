# Task 13: Add Usage Examples Documentation

## Description
Create comprehensive examples demonstrating both synchronous and asynchronous usage of the MLX provider.

## Requirements
1. Create example files for both sync and async usage
2. Include examples for text generation, vision, and streaming
3. Add comments explaining key concepts
4. Keep examples simple and focused
5. Include error handling examples

## Implementation Details

### 1. Basic Examples (`examples/mlx_basic_examples.py`)

```python
"""
MLX Provider Basic Usage Examples

This file demonstrates basic usage of AbstractLLM with the MLX provider
on Apple Silicon devices.

Requirements:
- macOS with Apple Silicon (M1/M2/M3)
- AbstractLLM installed with MLX dependencies: pip install ".[mlx]"
"""

from abstractllm import create_llm, ModelParameter
from pathlib import Path

def example_basic_generation():
    """Basic text generation with MLX."""
    print("\n=== Basic Generation ===")
    
    # Create an LLM with MLX provider
    llm = create_llm("mlx")
    
    # Generate text
    response = llm.generate("Explain what MLX is in simple terms.")
    
    print(f"Response: {response.content}")
    print(f"Token stats: {response.usage}")

def example_vision_generation():
    """Basic vision capabilities with MLX."""
    print("\n=== Vision Generation ===")
    
    # Create a vision-capable LLM
    llm = create_llm(
        "mlx",
        model="mlx-community/Qwen2.5-VL-32B-Instruct-6bit"
    )
    
    # Generate description from image
    response = llm.generate(
        prompt="What's in this image?",
        files=["path/to/image.jpg"]
    )
    
    print(f"Response: {response.content}")

def example_streaming():
    """Streaming generation example."""
    print("\n=== Streaming Generation ===")
    
    llm = create_llm("mlx")
    
    # Stream response chunks
    for chunk in llm.generate(
        "Explain quantum computing",
        stream=True
    ):
        print(chunk.content, end="", flush=True)
    print()

def example_with_system_prompt():
    """Using system prompts."""
    print("\n=== System Prompt ===")
    
    llm = create_llm("mlx")
    
    response = llm.generate(
        prompt="Write a function to calculate fibonacci numbers.",
        system_prompt="You are an expert Python programmer who writes clean, efficient code."
    )
    
    print(f"Response: {response.content}")

def example_error_handling():
    """Demonstrating error handling."""
    print("\n=== Error Handling ===")
    
    llm = create_llm("mlx")
    
    try:
        # Try to use vision with non-vision model
        response = llm.generate(
            prompt="What's in this image?",
            files=["image.jpg"]
        )
    except UnsupportedFeatureError as e:
        print(f"Expected error: {e}")
    
    try:
        # Try to process invalid image
        response = llm.generate(
            prompt="Describe this.",
            files=["nonexistent.jpg"]
        )
    except FileProcessingError as e:
        print(f"Expected error: {e}")

if __name__ == "__main__":
    # Print platform information
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    try:
        example_basic_generation()
        example_vision_generation()
        example_streaming()
        example_with_system_prompt()
        example_error_handling()
    except Exception as e:
        print(f"Error: {e}")
        print("Note: MLX provider requires macOS with Apple Silicon.")
```

### 2. Async Examples (`examples/mlx_async_examples.py`)

```python
"""
MLX Provider Async Usage Examples

This file demonstrates asynchronous usage of AbstractLLM with the MLX provider.

Requirements:
- macOS with Apple Silicon (M1/M2/M3)
- AbstractLLM installed with MLX dependencies: pip install ".[mlx]"
"""

import asyncio
from typing import AsyncGenerator
from abstractllm import create_llm, ModelParameter

async def example_basic_async():
    """Basic async text generation."""
    print("\n=== Basic Async Generation ===")
    
    llm = create_llm("mlx")
    
    response = await llm.generate_async(
        "Explain what MLX is in simple terms."
    )
    
    print(f"Response: {response.content}")

async def example_async_vision():
    """Async vision generation."""
    print("\n=== Async Vision Generation ===")
    
    llm = create_llm(
        "mlx",
        model="mlx-community/Qwen2.5-VL-32B-Instruct-6bit"
    )
    
    response = await llm.generate_async(
        prompt="What's in this image?",
        files=["path/to/image.jpg"]
    )
    
    print(f"Response: {response.content}")

async def example_async_streaming():
    """Async streaming generation."""
    print("\n=== Async Streaming ===")
    
    llm = create_llm("mlx")
    
    # Get async generator
    stream = await llm.generate_async(
        "Explain quantum computing",
        stream=True
    )
    
    # Stream response chunks
    async for chunk in stream:
        print(chunk.content, end="", flush=True)
    print()

async def example_concurrent_generations():
    """Running multiple generations concurrently."""
    print("\n=== Concurrent Generations ===")
    
    llm = create_llm("mlx")
    
    # Create multiple tasks
    tasks = [
        llm.generate_async("What is AI?"),
        llm.generate_async("What is machine learning?"),
        llm.generate_async("What is deep learning?")
    ]
    
    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    
    for i, response in enumerate(responses):
        print(f"\nResponse {i+1}: {response.content[:100]}...")

async def example_async_error_handling():
    """Demonstrating async error handling."""
    print("\n=== Async Error Handling ===")
    
    llm = create_llm("mlx")
    
    try:
        # Try to use vision with non-vision model
        await llm.generate_async(
            prompt="What's in this image?",
            files=["image.jpg"]
        )
    except UnsupportedFeatureError as e:
        print(f"Expected error: {e}")

async def main():
    """Run all async examples."""
    try:
        await example_basic_async()
        await example_async_vision()
        await example_async_streaming()
        await example_concurrent_generations()
        await example_async_error_handling()
    except Exception as e:
        print(f"Error: {e}")
        print("Note: MLX provider requires macOS with Apple Silicon.")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. CLI Examples

```bash
# Basic text generation
abstractllm mlx generate -m mlx-community/phi-2 -p "Hello, world!"

# Vision analysis
abstractllm mlx generate \
    -m mlx-community/Qwen2.5-VL-32B-Instruct-6bit \
    -p "What's in this image?" \
    -i image.jpg

# Streaming with system prompt
abstractllm mlx generate \
    -m mlx-community/Qwen2.5-VL-32B-Instruct-6bit \
    -p "Analyze this artwork." \
    -s "You are an art critic." \
    -i artwork.jpg \
    --stream

# Check system compatibility
abstractllm mlx system-check

# List available models
abstractllm mlx list-models
```

## References
- See MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See MLX Vision Upgrade Guide: `docs/mlx/vision-upgrade.md`
- See MLX documentation: https://github.com/ml-explore/mlx

## Testing
Run the example scripts:

```bash
# Run basic examples
python examples/mlx_basic_examples.py

# Run async examples
python examples/mlx_async_examples.py
```

## Success Criteria
1. All examples run successfully
2. Examples cover all major functionality
3. Error handling is demonstrated
4. Documentation is clear and helpful
5. Examples are properly organized and focused 