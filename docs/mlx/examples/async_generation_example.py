"""
Example of using asynchronous generation with the MLX provider.

This example demonstrates:
1. Basic asynchronous text generation
2. Asynchronous streaming generation
3. Concurrent generation with multiple prompts
"""

import asyncio
from abstractllm import create_llm


async def async_generation_example():
    """Demonstrate basic async generation."""
    llm = create_llm("mlx")  # Uses default model
    
    print("\n=== Basic Asynchronous Generation ===")
    response = await llm.generate_async("Explain what MLX is in one sentence")
    print(f"Response: {response.text}")
    print(f"Generated {response.completion_tokens} tokens")


async def async_streaming_example():
    """Demonstrate async streaming generation."""
    llm = create_llm("mlx")
    
    print("\n=== Asynchronous Streaming Generation ===")
    print("Response: ", end="", flush=True)
    
    # Get async generator
    async_gen = await llm.generate_async(
        "Count from 1 to 5 and explain why each number is interesting",
        stream=True
    )
    
    # Stream the response chunks
    async for chunk in async_gen:
        print(chunk.text, end="", flush=True)
    print()  # Add a newline at the end


async def concurrent_generation_example():
    """Demonstrate concurrent generation with multiple prompts."""
    llm = create_llm("mlx")
    
    print("\n=== Concurrent Generation ===")
    
    # Define multiple prompts
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms",
        "What are neural networks?"
    ]
    
    # Create tasks for concurrent execution
    tasks = [llm.generate_async(prompt) for prompt in prompts]
    
    # Execute all tasks concurrently
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()
    
    # Print results
    for i, response in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {response.text}")
    
    print(f"\nConcurrent execution time: {end_time - start_time:.2f} seconds")
    
    # Compare with sequential execution
    start_time = asyncio.get_event_loop().time()
    for prompt in prompts:
        await llm.generate_async(prompt)
    end_time = asyncio.get_event_loop().time()
    
    print(f"Sequential execution would take: {end_time - start_time:.2f} seconds")


async def main():
    """Run all examples."""
    await async_generation_example()
    await async_streaming_example()
    await concurrent_generation_example()


if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main()) 