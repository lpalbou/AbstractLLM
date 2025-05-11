# Task 14: Add Async Usage Examples Documentation

## Description
Create an examples file demonstrating how to use the MLX provider with async/await.

## Requirements
1. Create a `examples/mlx_async_examples.py` file
2. Include examples for basic async generation and streaming
3. Add comments explaining async patterns
4. Show proper error handling with asyncio

## Implementation Details

Create a file at `examples/mlx_async_examples.py`:

```python
"""
MLX Provider Async Usage Examples

This file demonstrates how to use AbstractLLM with the MLX provider
in asynchronous environments.

Requirements:
- macOS with Apple Silicon (M1/M2/M3)
- AbstractLLM installed with MLX dependencies: pip install ".[mlx]"
"""

import asyncio
from typing import AsyncGenerator
from abstractllm import create_llm, ModelParameter

async def example_basic_async_generation():
    """
    Basic async text generation with MLX.
    """
    print("\n=== Basic Async Generation ===")
    
    # Create an LLM with MLX provider
    llm = create_llm("mlx")
    
    # Generate text asynchronously
    response = await llm.generate_async("Explain what MLX is in simple terms.")
    
    # Display the response
    print(f"Response: {response.text}")
    print(f"Token stats: {response.prompt_tokens} prompt tokens, " 
          f"{response.completion_tokens} completion tokens")


async def example_async_streaming():
    """
    Async streaming generation for real-time responses.
    """
    print("\n=== Async Streaming Generation ===")
    
    # Create an LLM
    llm = create_llm("mlx")
    
    # Generate with async streaming
    print("Streaming response:")
    
    # Make sure we receive the AsyncGenerator
    stream_gen = await llm.generate_async(
        "Explain the importance of asynchronous programming.", 
        stream=True
    )
    
    # Iterate through chunks as they arrive
    async for chunk in stream_gen:
        print(chunk.text, end="", flush=True)
        # Simulate some processing time
        await asyncio.sleep(0.001)
    
    print("\n")  # Add newline at the end


async def example_concurrent_generations():
    """
    Running multiple generations concurrently.
    """
    print("\n=== Concurrent Generations ===")
    
    # Create an LLM
    llm = create_llm("mlx")
    
    # Define several prompts
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "What are neural networks?"
    ]
    
    # Create tasks for concurrent execution
    tasks = [
        llm.generate_async(prompt) 
        for prompt in prompts
    ]
    
    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    
    # Display the responses
    for i, response in enumerate(responses):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {response.text[:100]}...")  # Show first 100 chars


async def example_async_with_system_prompt():
    """
    Using system prompts with async generation.
    """
    print("\n=== Async with System Prompt ===")
    
    # Create an LLM
    llm = create_llm("mlx")
    
    # Define a system prompt
    system_prompt = "You are an expert programmer who explains code in simple terms."
    
    # Generate text with system prompt asynchronously
    response = await llm.generate_async(
        "Explain how asyncio works in Python.",
        system_prompt=system_prompt
    )
    
    # Display the response
    print(f"Response: {response.text}")


async def main():
    """Run all examples."""
    # Print platform information
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Run examples
    try:
        await example_basic_async_generation()
        await example_async_streaming()
        await example_concurrent_generations()
        await example_async_with_system_prompt()
    except Exception as e:
        print(f"Error: {e}")
        print("Note: MLX provider requires macOS with Apple Silicon.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
```

## References
- Reference the MLX Provider Usage Guide: `docs/mlx/mlx_usage_examples.md`
- See AbstractLLM's existing async examples (if any)
- Python asyncio documentation: https://docs.python.org/3/library/asyncio.html

## Testing
Run the examples script to ensure it works correctly:

```bash
python examples/mlx_async_examples.py
``` 