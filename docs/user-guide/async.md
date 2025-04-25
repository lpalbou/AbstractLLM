# Asynchronous Generation

This guide covers asynchronous generation in AbstractLLM, which allows you to generate responses in non-blocking ways, enabling more efficient handling of multiple requests and integration with async frameworks.

## Prerequisites

To use async features, ensure you're familiar with:
- Python's async/await syntax
- Asynchronous programming principles
- The basics of AbstractLLM (see [Basic Generation](basic-generation.md))

## Basic Async Generation

The simplest way to generate responses asynchronously is with the `generate_async` method:

```python
import asyncio
from abstractllm import create_llm

async def generate_response():
    # Create a provider
    llm = create_llm("openai", model="gpt-4")
    
    # Generate asynchronously
    response = await llm.generate_async("Explain quantum computing briefly.")
    return response

# Run the async function
response = asyncio.run(generate_response())
print(response)
```

## Async Streaming

You can combine async with streaming for real-time processing of responses:

```python
import asyncio
from abstractllm import create_llm

async def stream_async():
    llm = create_llm("anthropic", model="claude-3-opus-20240229")
    
    print("Streaming response: ", end="", flush=True)
    async for chunk in llm.generate_async(
        "Write a short story about a space explorer.",
        stream=True
    ):
        print(chunk, end="", flush=True)
    print("\nDone!")

asyncio.run(stream_async())
```

## Multiple Concurrent Requests

Async really shines when handling multiple requests concurrently:

```python
import asyncio
from abstractllm import create_llm

async def generate_multiple():
    llm = create_llm("openai", model="gpt-4")
    
    # Define multiple prompts
    prompts = [
        "Write a haiku about mountains.",
        "Explain how rockets work.",
        "Create a short joke about programming."
    ]
    
    # Create async tasks for each prompt
    tasks = [llm.generate_async(prompt) for prompt in prompts]
    
    # Execute all tasks concurrently
    responses = await asyncio.gather(*tasks)
    
    # Print results
    for i, response in enumerate(responses):
        print(f"Response {i+1}:\n{response}\n")

asyncio.run(generate_multiple())
```

## Async Sessions

You can use async methods with sessions for stateful conversations:

```python
import asyncio
from abstractllm import create_llm
from abstractllm.session import Session

async def async_conversation():
    provider = create_llm("openai", model="gpt-4")
    session = Session(
        system_prompt="You are a helpful assistant.",
        provider=provider
    )
    
    # Add a user message
    session.add_message("user", "Tell me about black holes.")
    
    # Generate the first response asynchronously
    response1 = await session.generate_async()
    print("First response:", response1)
    
    # Continue the conversation
    session.add_message("user", "How do they affect time?")
    response2 = await session.generate_async()
    print("Second response:", response2)

asyncio.run(async_conversation())
```

## Async Tool Calls

Async generation also works with tool calling:

```python
import asyncio
from abstractllm import create_llm
from abstractllm.session import Session

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulated weather data
    return f"The weather in {location} is currently sunny and 72°F."

async def async_tool_usage():
    provider = create_llm("openai", model="gpt-4")
    session = Session(
        system_prompt="You are a helpful assistant that can check the weather.",
        provider=provider,
        tools=[get_weather]
    )
    
    # Generate a response with tool usage asynchronously
    response = await session.generate_with_tools_async(
        "What's the weather like in San Francisco?"
    )
    print(response)

asyncio.run(async_tool_usage())
```

## Integration with Async Web Frameworks

AbstractLLM's async capabilities make it easy to integrate with async web frameworks like FastAPI:

```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from abstractllm import create_llm

app = FastAPI()
llm = create_llm("openai", model="gpt-4")

class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: str = None

@app.post("/generate")
async def generate(request: GenerationRequest):
    response = await llm.generate_async(
        prompt=request.prompt,
        system_prompt=request.system_prompt
    )
    return {"response": response}

@app.post("/stream")
async def stream(request: GenerationRequest):
    async def generate_stream():
        async for chunk in llm.generate_async(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            stream=True
        ):
            yield f"data: {chunk}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
```

## Provider-Specific Async Considerations

Different providers have varying async support:

- **OpenAI**: Fully async-compatible, efficient for concurrent requests
- **Anthropic**: Good async support with efficient streaming
- **Ollama**: Async support depends on your local setup and model size
- **HuggingFace**: Async support varies by model type and implementation

## Error Handling in Async Context

Error handling in async functions requires proper try/except blocks:

```python
import asyncio
from abstractllm import create_llm
from abstractllm.exceptions import AbstractLLMError

async def safe_generate():
    llm = create_llm("openai", model="gpt-4")
    
    try:
        response = await llm.generate_async("Explain neural networks.")
        return response
    except AbstractLLMError as e:
        print(f"Generation error: {str(e)}")
        return f"Error occurred: {str(e)}"

result = asyncio.run(safe_generate())
```

## Performance Optimization

### Rate Limiting

When making many async requests, consider implementing rate limiting to avoid API rate limits:

```python
import asyncio
import time
from abstractllm import create_llm

class RateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60 / calls_per_minute
        self.last_call_time = 0
    
    async def wait(self):
        now = time.time()
        time_since_last_call = now - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last_call)
        
        self.last_call_time = time.time()

async def rate_limited_generation(prompts):
    llm = create_llm("openai", model="gpt-4")
    limiter = RateLimiter(calls_per_minute=20)  # Adjust based on your API limits
    
    results = []
    for prompt in prompts:
        await limiter.wait()
        response = await llm.generate_async(prompt)
        results.append(response)
    
    return results
```

### Connection Pooling

For high-volume applications, consider connection pooling:

```python
import aiohttp
import asyncio
from abstractllm import create_llm

async def generate_with_session(session, prompt):
    # Create LLM with custom session
    llm = create_llm("openai", model="gpt-4", http_session=session)
    return await llm.generate_async(prompt)

async def main():
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]
    
    # Create a shared HTTP session with connection pooling
    async with aiohttp.ClientSession() as session:
        tasks = [generate_with_session(session, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result[:50]}...")

asyncio.run(main())
```

## Conclusion

Asynchronous generation in AbstractLLM provides significant performance benefits for applications that need to handle multiple requests efficiently. By leveraging async/await syntax, you can build non-blocking applications that make optimal use of resources while waiting for LLM responses.

For more advanced usage, consider combining async generation with other AbstractLLM features like [Streaming](streaming.md), [Sessions](sessions.md), and [Tool Calls](tools.md). 