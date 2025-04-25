# Anthropic Provider

The Anthropic provider in AbstractLLM integrates with Anthropic's Claude models, known for their robust reasoning, safety features, and long context windows. This guide covers authentication, configuration options, supported models, and provider-specific features.

## Setup

### Authentication

To use the Anthropic provider, you need an Anthropic API key:

```python
from abstractllm import create_llm

# Using environment variable (recommended)
# Set ANTHROPIC_API_KEY in your environment
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Explicit API key
llm = create_llm(
    "anthropic", 
    model="claude-3-haiku-20240307", 
    api_key="your-anthropic-api-key-here"
)
```

### Installation

To use the Anthropic provider, install AbstractLLM with Anthropic dependencies:

```bash
pip install abstractllm[anthropic]
```

## Supported Models

The Anthropic provider supports these Claude models with different capabilities:

| Model | Description | Special Capabilities |
|-------|-------------|---------------------|
| claude-3-opus-20240229 | Most powerful model with advanced reasoning | Tool use, Vision |
| claude-3-sonnet-20240229 | Balanced model for most tasks | Tool use, Vision |
| claude-3-haiku-20240307 | Fast, efficient model for simple tasks | Tool use, Vision |
| claude-2.1 | Previous generation model | Long context window |
| claude-2.0 | Previous generation model | Long context window |
| claude-instant-1.2 | Faster, lower-cost model | General text tasks |

For the most current list of available models, refer to the [Anthropic Models documentation](https://docs.anthropic.com/claude/docs/models-overview).

## Basic Usage

### Simple Text Generation

```python
from abstractllm import create_llm

llm = create_llm("anthropic", model="claude-3-sonnet-20240229")
response = llm.generate("What is the capital of France?")
print(response)  # Paris
```

### Streaming Responses

```python
from abstractllm import create_llm

llm = create_llm("anthropic", model="claude-3-haiku-20240307")
for chunk in llm.generate("Explain quantum computing", stream=True):
    print(chunk, end="", flush=True)
```

## Provider-Specific Features

### Vision Capabilities

Use Claude models to analyze images:

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create LLM with a Claude vision-capable model
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Process a single image
image = ImageInput.from_file("path/to/image.jpg")
response = llm.generate("What's in this image?", media=[image])

# Process multiple images
image1 = ImageInput.from_file("path/to/image1.jpg")
image2 = ImageInput.from_url("https://example.com/image2.jpg")
response = llm.generate(
    "Compare these two images. What are the differences?", 
    media=[image1, image2]
)
```

### Tool Use

Define and use tools with Claude models:

```python
from abstractllm import create_llm, ToolDefinition

# Define a tool
get_weather = ToolDefinition(
    name="get_weather",
    description="Get the current weather in a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit"
            }
        },
        "required": ["location"]
    }
)

# Create LLM with tool(s)
llm = create_llm("anthropic", model="claude-3-opus-20240229", tools=[get_weather])

# Function to be called by the model
def get_weather_function(location, unit="celsius"):
    # In a real app, you would call a weather API here
    temp = 22 if unit == "celsius" else 72
    return f"The weather in {location} is sunny and {temp} degrees {unit}."

# Handle the response with potential tool calls
response = llm.generate(
    "What's the weather like in Boston?",
    tool_handler=lambda tool_name, args: get_weather_function(**args) if tool_name == "get_weather" else None
)

print(response)
```

### Long Context Window

Take advantage of Claude's long context window for processing large documents:

```python
from abstractllm import create_llm

llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Reading a long document (up to 200K tokens depending on the model)
with open("path/to/large_document.txt", "r") as f:
    document = f.read()

response = llm.generate(
    f"Please summarize the following document into key points:\n\n{document}"
)

print(response)
```

## Configuration Options

The Anthropic provider supports these configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | The Anthropic model to use | "claude-3-sonnet-20240229" |
| `api_key` | Anthropic API key | From environment |
| `temperature` | Controls randomness (0.0 to 1.0) | 0.7 |
| `max_tokens` | Maximum tokens to generate | 1024 |
| `top_p` | Nucleus sampling parameter | 1.0 |
| `top_k` | Limits to top K token choices | None |
| `tools` | List of tool definitions | None |
| `stream` | Whether to stream responses | False |
| `base_url` | Custom API base URL | None |
| `max_retries` | Maximum number of retries for API errors | 3 |
| `retry_wait_time` | Wait time between retries in seconds | 2.0 |
| `system_prompt` | System prompt to guide model behavior | None |
| `timeout` | Request timeout in seconds | None |

## Session Management

Maintain conversation context with sessions:

```python
from abstractllm import create_llm

llm = create_llm("anthropic", model="claude-3-sonnet-20240229")
session = llm.create_session(system_prompt="You are a helpful assistant specializing in scientific topics.")

# First interaction
response1 = session.generate("What is dark matter?")
print(response1)

# Follow-up question (context is preserved)
response2 = session.generate("How does it differ from dark energy?")
print(response2)

# Continue conversation
response3 = session.generate("What are the leading theories about its composition?")
print(response3)

# Get the complete conversation history
conversation = session.get_history()
print(f"Conversation has {len(conversation)} messages")
```

## Asynchronous Usage

Use the Anthropic provider asynchronously:

```python
import asyncio
from abstractllm import create_llm

async def main():
    llm = create_llm("anthropic", model="claude-3-haiku-20240307")
    
    # Async generation
    response = await llm.generate_async("What is the capital of Japan?")
    print(response)
    
    # Async streaming
    async for chunk in llm.generate_async("Explain nuclear fusion", stream=True):
        print(chunk, end="", flush=True)
    
    # Process multiple prompts concurrently
    tasks = [
        llm.generate_async("What is the capital of France?"),
        llm.generate_async("What is the capital of Germany?"),
        llm.generate_async("What is the capital of Italy?")
    ]
    results = await asyncio.gather(*tasks)
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")

asyncio.run(main())
```

## Error Handling

Handle Anthropic API-specific errors:

```python
from abstractllm import create_llm
from abstractllm.errors import (
    ProviderAPIError, 
    RateLimitError, 
    InvalidAPIKeyError, 
    ContentFilterError,
    TokenLimitError
)

try:
    llm = create_llm("anthropic", model="claude-3-sonnet-20240229")
    response = llm.generate("Write a comprehensive analysis of nuclear fusion technology")
except InvalidAPIKeyError:
    print("Your API key is invalid. Check your credentials")
except RateLimitError:
    print("Rate limit exceeded. Try again later or adjust your request volume")
except TokenLimitError:
    print("The prompt is too long for the model's context window")
except ContentFilterError as e:
    print(f"Content filter triggered: {e}")
    # Handle by rephrasing or providing alternative content
    safe_response = llm.generate("Explain computer security principles and the importance of ethical security practices")
    print("Alternative safe response:", safe_response)
except ProviderAPIError as e:
    print(f"API error: {e}")
```

## Cost Management

Manage token usage and costs:

```python
from abstractllm import create_llm

# Create LLM with token tracking
llm = create_llm("anthropic", model="claude-3-sonnet-20240229", track_tokens=True)

# Generate response
response = llm.generate("Explain the theory of relativity")

# Get token usage information
token_usage = llm.get_last_token_usage()
print(f"Prompt tokens: {token_usage.prompt_tokens}")
print(f"Completion tokens: {token_usage.completion_tokens}")
print(f"Total tokens: {token_usage.total_tokens}")

# For estimating costs (example rates, check current Anthropic pricing)
prompt_cost = token_usage.prompt_tokens * 0.000015  # $0.015 per 1K tokens (Sonnet)
completion_cost = token_usage.completion_tokens * 0.000045  # $0.045 per 1K tokens (Sonnet)
total_cost = prompt_cost + completion_cost
print(f"Estimated cost: ${total_cost:.6f}")
```

## Advanced Configurations

### Custom Base URL

Connect to a compatible API or proxy:

```python
from abstractllm import create_llm

llm = create_llm(
    "anthropic",
    model="claude-3-sonnet-20240229",
    base_url="https://your-custom-endpoint.com/v1"
)
```

### Proxy Usage

Configure a proxy for API requests:

```python
from abstractllm import create_llm

llm = create_llm(
    "anthropic",
    model="claude-3-sonnet-20240229",
    http_proxy="http://your-proxy-server:port",
    https_proxy="https://your-proxy-server:port"
)
```

### System Prompt Configuration

Use system prompts to guide model behavior:

```python
from abstractllm import create_llm

# Set system prompt during LLM creation
llm = create_llm(
    "anthropic",
    model="claude-3-opus-20240229",
    system_prompt="You are Claude, an AI assistant built by Anthropic. Answer all questions factually and refuse to engage with harmful or illegal requests."
)

# Create a session with a specific system prompt
session = llm.create_session(
    system_prompt="You are a scientific assistant specializing in physics and astronomy. Provide detailed, technical answers with references to scientific literature when possible."
)
```

### Request Retry Configuration

Configure automatic retries for network issues:

```python
from abstractllm import create_llm

llm = create_llm(
    "anthropic",
    model="claude-3-sonnet-20240229",
    max_retries=5,
    retry_wait_time=2.0  # seconds
)
```

## Best Practices

- **API Key Security**: Never hardcode API keys in your code. Use environment variables or secure credential storage.
- **Model Selection**: 
  - Use `claude-3-haiku` for high-volume, simple tasks where speed is important
  - Use `claude-3-sonnet` for a balance of quality and cost
  - Use `claude-3-opus` for complex reasoning, sensitive applications, or when highest quality is needed
- **System Prompts**: Use system prompts to establish consistent behavior across interactions
- **Token Management**: 
  - Leverage Claude's long context window for large documents
  - Be mindful of costs when processing large amounts of text
- **Error Handling**: Implement robust error handling, particularly for content filters and rate limits
- **Streaming**: Use streaming for longer responses to improve user experience

## Common Issues

### Content Filter Handling

When encountering content filter issues:

```python
from abstractllm import create_llm
from abstractllm.errors import ContentFilterError

llm = create_llm("anthropic", model="claude-3-sonnet-20240229")

try:
    # A potentially sensitive topic
    response = llm.generate("Explain how to hack into a computer system")
    print(response)
except ContentFilterError as e:
    print(f"Content filter triggered: {e}")
    # Handle by rephrasing or providing alternative content
    safe_response = llm.generate("Explain computer security principles and the importance of ethical security practices")
    print("Alternative safe response:", safe_response)
```

### Rate Limits

If encountering rate limits:

```python
from abstractllm import create_llm
import time
from abstractllm.errors import RateLimitError

def generate_with_backoff(llm, prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            return llm.generate(prompt)
        except RateLimitError:
            wait_time = (2 ** attempt) * 1.5  # Exponential backoff
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")

llm = create_llm("anthropic", model="claude-3-sonnet-20240229")
response = generate_with_backoff(llm, "What is quantum physics?")
```

### Context Window Optimization

For large documents that might exceed context windows:

```python
from abstractllm import create_llm

llm = create_llm("anthropic", model="claude-3-opus-20240229")

def process_large_document(document, max_chunk_size=100000):
    # Simple approach: if document fits in context, process it directly
    if len(document) <= max_chunk_size:
        return llm.generate(f"Summarize this document:\n\n{document}")
    
    # For very large documents, use a chunking approach
    chunks = []
    for i in range(0, len(document), max_chunk_size):
        chunk = document[i:i + max_chunk_size]
        chunks.append(chunk)
    
    # Process each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        summary = llm.generate(
            f"This is part {i+1} of {len(chunks)} of a larger document. " +
            f"Summarize this part:\n\n{chunk}"
        )
        summaries.append(summary)
    
    # Combine summaries
    if len(summaries) == 1:
        return summaries[0]
    
    combined_summaries = "\n\n".join([f"Part {i+1} summary: {summary}" 
                                       for i, summary in enumerate(summaries)])
    
    final_summary = llm.generate(
        f"These are summaries of different parts of a document. " +
        f"Create a cohesive overall summary:\n\n{combined_summaries}"
    )
    
    return final_summary

# Example usage
with open("very_large_document.txt", "r") as f:
    document = f.read()

summary = process_large_document(document)
print(summary)
```

## Provider Implementation Details

For developers interested in how the Anthropic provider is implemented in AbstractLLM, see the [Provider Implementations](../architecture/providers.md) documentation. 