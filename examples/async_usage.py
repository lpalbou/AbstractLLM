#!/usr/bin/env python
"""
Async usage examples for AbstractLLM.

This script demonstrates how to use the async functionality of AbstractLLM.
"""

import os
import asyncio
import logging
from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.utils.logging import setup_logging

async def generate_responses(llm, prompts):
    """Generate responses for multiple prompts in parallel."""
    tasks = [llm.generate_async(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

async def streaming_example(llm, prompt):
    """Demonstrate async streaming."""
    print(f"\nStreaming response for: {prompt}")
    stream = await llm.generate_async(prompt, stream=True)
    
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()

async def main():
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Skip if no API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping examples (OPENAI_API_KEY not set)")
        return
    
    # Create the LLM instance
    llm = create_llm("openai", **{
        ModelParameter.MODEL: "gpt-3.5-turbo",
        ModelParameter.TEMPERATURE: 0.7
    })
    
    # Check if async is supported
    capabilities = llm.get_capabilities()
    if not capabilities.get(ModelCapability.ASYNC, False):
        print("This provider doesn't support async operations")
        return
    
    # Example 1: Basic async generation
    prompt = "What is the capital of France?"
    print(f"\nAsync generation for prompt: {prompt}")
    response = await llm.generate_async(prompt)
    print(f"Response: {response}")
    
    # Example 2: Parallel generation
    prompts = [
        "What is 2+2?",
        "Name three colors of the rainbow",
        "What's the tallest mountain in the world?"
    ]
    
    print("\nGenerating multiple responses in parallel...")
    start_time = asyncio.get_event_loop().time()
    
    responses = await generate_responses(llm, prompts)
    
    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response {i+1}: {response}")
    
    print(f"\nParallel generation completed in {duration:.2f} seconds")
    
    # Example 3: Async streaming if supported
    if capabilities.get(ModelCapability.STREAMING, False):
        await streaming_example(llm, "Count from 1 to 10, with a brief pause after each number.")

if __name__ == "__main__":
    asyncio.run(main()) 