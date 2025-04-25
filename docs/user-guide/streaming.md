# Streaming Responses in AbstractLLM

Streaming allows you to receive and process LLM responses in real-time as they're being generated, rather than waiting for the complete response. This guide covers how to implement streaming in your AbstractLLM applications.

## Basic Streaming

### Stream Text Responses

The most basic way to stream responses is with the `generate_stream` method:

```python
from abstractllm import create_llm

provider = create_llm("openai", model="gpt-4")

# Get a stream of response chunks
for chunk in provider.generate_stream("Write a short poem about mountains."):
    # Print each chunk as it arrives
    print(chunk, end="", flush=True)
```

This will print each chunk of the response as it's received, creating a typing-like effect.

### Stream with Sessions

You can also stream responses within a session to maintain conversation history:

```python
from abstractllm import create_llm
from abstractllm.session import Session

provider = create_llm("openai", model="gpt-4")
session = Session(provider=provider)

# Add a user message
session.add_message("user", "Tell me about quantum physics.")

# Stream the response
for chunk in session.generate_stream():
    print(chunk, end="", flush=True)
    
# The complete response is automatically added to the session history
```

## Advanced Streaming

### Streaming with Custom Processing

You can process each chunk as it arrives:

```python
from abstractllm import create_llm

provider = create_llm("openai", model="gpt-4")

total_tokens = 0
word_count = 0

for chunk in provider.generate_stream("Explain neural networks briefly."):
    # Count tokens and words
    total_tokens += 1
    word_count += len(chunk.split())
    
    # Print with formatting
    print(chunk, end="", flush=True)

print(f"\n\nStats: {total_tokens} tokens, {word_count} words")
```

### Async Streaming

For asynchronous applications, use the async streaming API:

```python
import asyncio
from abstractllm import create_llm

async def stream_async():
    provider = create_llm("openai", model="gpt-4")
    
    async for chunk in provider.agenerate_stream("Write a haiku about programming."):
        print(chunk, end="", flush=True)
    
    print("\nDone!")

# Run the async function
asyncio.run(stream_async())
```

### Streaming with Tools

When using tools, you can stream the response and observe tool calls as they happen:

```python
from abstractllm import create_llm
from abstractllm.tool import Tool, ToolCall

def calculator(expression):
    """Evaluates a mathematical expression."""
    try:
        return eval(expression)
    except:
        return "Error evaluating expression"

# Create a provider with tool support
provider = create_llm("openai", model="gpt-4")

# Define tools
tools = [
    Tool(
        name="calculator",
        description="Evaluates a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        function=calculator
    )
]

# Stream with tool support
prompt = "What is 123 * 456? Then add 789 to the result."

for chunk in provider.generate_stream_with_tools(prompt, tools=tools):
    if isinstance(chunk, str):
        # Regular text chunk
        print(chunk, end="", flush=True)
    elif isinstance(chunk, ToolCall):
        # Tool call
        print(f"\n[Tool Call: {chunk.name}({chunk.arguments})]", flush=True)
        print(f"[Tool Result: {chunk.result}]\n", flush=True)
```

## Implementation Details

### Chunk Format

Different providers return chunks in different formats. AbstractLLM normalizes these differences so you always get simple text chunks when streaming.

Internally, this works by:

1. Converting provider-specific streaming formats to a common format
2. Extracting just the text content from each chunk
3. Handling special tokens or formatting specific to each provider

### Buffer Options

You can control how streaming chunks are buffered:

```python
from abstractllm import create_llm

provider = create_llm("openai", model="gpt-4")

# Stream with buffer control
for chunk in provider.generate_stream(
    "List five facts about space.",
    buffer_size=10,  # Buffer up to 10 chunks before yielding
    min_chunk_size=5  # Only yield chunks of at least 5 characters
):
    print(chunk, end="", flush=True)
```

This can be useful for creating a smoother streaming experience or reducing the number of UI updates in a front-end application.

## Provider-Specific Considerations

Different LLM providers handle streaming in slightly different ways:

### OpenAI

OpenAI returns small chunks, often at the token level. Their streaming is reliable and consistent.

```python
from abstractllm import create_llm

provider = create_llm("openai", model="gpt-4")

for chunk in provider.generate_stream("Tell me a short story."):
    print(chunk, end="", flush=True)
```

### Anthropic

Anthropic's Claude typically streams larger chunks than OpenAI.

```python
from abstractllm import create_llm

provider = create_llm("anthropic", model="claude-3-opus-20240229")

for chunk in provider.generate_stream("Tell me a short story."):
    print(chunk, end="", flush=True)
```

### Local Models (Ollama)

Local models through Ollama also support streaming:

```python
from abstractllm import create_llm

provider = create_llm("ollama", model="llama3")

for chunk in provider.generate_stream("Tell me a short story."):
    print(chunk, end="", flush=True)
```

## Use Cases for Streaming

### Real-time Chat Interfaces

Streaming is ideal for chat applications to provide a more responsive user experience:

```python
from abstractllm import create_llm
from abstractllm.session import Session

provider = create_llm("openai", model="gpt-4")
session = Session(provider=provider)

# In a real application, this would be in a chat UI
def chat_loop():
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
            
        session.add_message("user", user_input)
        
        print("AI: ", end="", flush=True)
        for chunk in session.generate_stream():
            print(chunk, end="", flush=True)
        print("\n")

chat_loop()
```

### Progress Indicators

You can use streaming to show progress while generating long responses:

```python
from abstractllm import create_llm
import time

provider = create_llm("openai", model="gpt-4")

prompt = "Write a detailed 500-word essay about climate change."

start_time = time.time()
total_chars = 0

print("Generating essay...")
print("-" * 50)

for chunk in provider.generate_stream(prompt):
    print(chunk, end="", flush=True)
    total_chars += len(chunk)
    
    # Update progress every 100 characters
    if total_chars % 100 == 0:
        elapsed = time.time() - start_time
        chars_per_second = total_chars / elapsed
        print(f"\n[Progress: {total_chars} chars, {chars_per_second:.2f} chars/sec]", end="\r", flush=True)

elapsed = time.time() - start_time
print(f"\n{'-' * 50}")
print(f"Generation complete: {total_chars} characters in {elapsed:.2f} seconds")
```

### Early Stopping

Streaming allows you to stop generation early if needed:

```python
from abstractllm import create_llm

provider = create_llm("openai", model="gpt-4")

target_length = 100
result = ""

print("Starting generation (will stop after ~100 characters)...")

for chunk in provider.generate_stream("Explain the theory of relativity in detail."):
    result += chunk
    print(chunk, end="", flush=True)
    
    # Stop after we reach the target length
    if len(result) >= target_length:
        print("\n\nReached target length. Stopping early.")
        break

print("\nDone!")
```

## Error Handling with Streaming

It's important to handle errors that may occur during streaming:

```python
from abstractllm import create_llm
from abstractllm.exceptions import AbstractLLMException

provider = create_llm("openai", model="gpt-4")

try:
    for chunk in provider.generate_stream("Write a short story."):
        print(chunk, end="", flush=True)
except AbstractLLMException as e:
    print(f"\nError during streaming: {str(e)}")
```

## Performance Optimization

When working with streaming responses, consider these performance tips:

1. **Buffer Appropriately**: Adjust buffer sizes based on your use case
2. **Process Efficiently**: Do minimal processing in the streaming loop
3. **Use Async for Web Applications**: Use async streaming for web servers
4. **Limit Concurrent Streams**: Be mindful of how many concurrent streams you're maintaining

Example of efficient async streaming for a web application:

```python
import asyncio
from abstractllm import create_llm

async def stream_handler(prompt):
    provider = create_llm("openai", model="gpt-4")
    result = []
    
    try:
        async for chunk in provider.agenerate_stream(prompt):
            # Store chunks in memory
            result.append(chunk)
            # In a real app, you'd send each chunk to the client
            # e.g., via websockets
            
        return "".join(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage in an async context
async def main():
    tasks = [
        stream_handler("Write about AI"),
        stream_handler("Write about ML"),
        stream_handler("Write about NLP")
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"Result length: {len(result)}")

# Run the async function
asyncio.run(main())
```

## Streaming with Custom Callbacks

You can implement custom callbacks for advanced streaming scenarios:

```python
from abstractllm import create_llm
from typing import Callable

def stream_with_callback(provider, prompt, callback: Callable[[str], None]):
    """Stream with a custom callback for each chunk."""
    for chunk in provider.generate_stream(prompt):
        # Process each chunk with the callback
        callback(chunk)

# Example usage
provider = create_llm("openai", model="gpt-4")

def my_processor(chunk):
    # Custom processing logic
    processed = chunk.upper()  # Just an example - convert to uppercase
    print(processed, end="", flush=True)

stream_with_callback(provider, "Tell me a joke.", my_processor)
```

## Conclusion

Streaming responses in AbstractLLM provides a more interactive and responsive experience for users. It allows for real-time feedback, early stopping, and efficient handling of large responses. Whether you're building a chat interface, a content generation tool, or a complex AI application, streaming can significantly enhance the user experience. 