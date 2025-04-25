# OpenAI Provider

The OpenAI provider in AbstractLLM enables you to integrate with OpenAI's models, including GPT-4, GPT-3.5, and newer models. This guide covers authentication, configuration options, supported models, and provider-specific features.

## Setup

### Authentication

To use the OpenAI provider, you need an OpenAI API key:

```python
from abstractllm import create_llm

# Using environment variable (recommended)
# Set OPENAI_API_KEY in your environment
llm = create_llm("openai", model="gpt-3.5-turbo")

# Explicit API key
llm = create_llm(
    "openai", 
    model="gpt-3.5-turbo", 
    api_key="your-openai-api-key-here"
)

# Using organization ID
llm = create_llm(
    "openai",
    model="gpt-3.5-turbo",
    organization="your-organization-id"  # Optional
)
```

### Installation

To use the OpenAI provider, install AbstractLLM with OpenAI dependencies:

```bash
pip install abstractllm[openai]
```

## Supported Models

The OpenAI provider supports these models with different capabilities:

| Model | Description | Special Capabilities |
|-------|-------------|---------------------|
| gpt-4o | Latest multimodal model with vision and tool capabilities | Tool use, Vision |
| gpt-4-turbo | GPT-4 Turbo with improved performance | Tool use |
| gpt-4-vision-preview | GPT-4 with vision capabilities | Vision, Tool use |
| gpt-4 | Powerful model for complex tasks | Tool use |
| gpt-3.5-turbo | Cost-effective model for general tasks | Tool use |
| gpt-3.5-turbo-16k | GPT-3.5 with extended context window | Tool use |

For the most current list of available models, refer to the [OpenAI API documentation](https://platform.openai.com/docs/models).

## Basic Usage

### Simple Text Generation

```python
from abstractllm import create_llm

llm = create_llm("openai", model="gpt-3.5-turbo")
response = llm.generate("What is the capital of France?")
print(response)  # Paris
```

### Streaming Responses

```python
from abstractllm import create_llm

llm = create_llm("openai", model="gpt-3.5-turbo")
for chunk in llm.generate("Explain quantum computing", stream=True):
    print(chunk, end="", flush=True)
```

## Provider-Specific Features

### Vision Capabilities

Use OpenAI models to analyze images:

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create LLM with a vision-capable model
llm = create_llm("openai", model="gpt-4-vision-preview")

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

# Control image detail level to optimize token usage
high_detail_image = ImageInput.from_file(
    "path/to/complex_image.jpg", 
    detail="high"
)
low_detail_image = ImageInput.from_file(
    "path/to/simple_image.jpg", 
    detail="low"
)
```

### Tool Use (Function Calling)

Define and use tools that OpenAI models can call:

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
llm = create_llm("openai", model="gpt-3.5-turbo", tools=[get_weather])

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

### JSON Mode

Enable structured JSON outputs:

```python
from abstractllm import create_llm

llm = create_llm("openai", model="gpt-3.5-turbo", response_format={"type": "json_object"})

json_response = llm.generate(
    "Provide a list of the top 3 largest countries by area with their populations"
)
print(json_response)  # Returns a properly formatted JSON string
```

## Configuration Options

The OpenAI provider supports these configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | The OpenAI model to use | "gpt-3.5-turbo" |
| `api_key` | OpenAI API key | From environment |
| `organization` | OpenAI organization ID | From environment |
| `temperature` | Controls randomness (0.0 to 2.0) | 0.7 |
| `max_tokens` | Maximum tokens to generate | None |
| `top_p` | Nucleus sampling parameter | 1.0 |
| `frequency_penalty` | Penalizes token repetition | 0.0 |
| `presence_penalty` | Penalizes tokens based on their presence in the prompt | 0.0 |
| `response_format` | Format of the model response (json_object, text) | None |
| `seed` | Random number seed for reproducibility | None |
| `tools` | List of tool definitions | None |
| `tool_choice` | Control how/when tools are called | "auto" |
| `stream` | Whether to stream responses | False |
| `base_url` | Custom API base URL | None |
| `max_retries` | Maximum number of retries for API errors | 3 |
| `retry_wait_time` | Wait time between retries in seconds | 2.0 |
| `timeout` | Request timeout in seconds | None |

## Session Management

Maintain conversation context with sessions:

```python
from abstractllm import create_llm

llm = create_llm("openai", model="gpt-3.5-turbo")
session = llm.create_session()

# First interaction
response1 = session.generate("What are the main renewable energy sources?")
print(response1)

# Follow-up question (context is preserved)
response2 = session.generate("Which one has seen the most growth in recent years?")
print(response2)

# Continue conversation
response3 = session.generate("What are some challenges with implementing it at scale?")
print(response3)

# Get the complete conversation history
conversation = session.get_history()
print(f"Conversation has {len(conversation)} messages")
```

## Model Selection and Management

Use different models within the OpenAI provider:

```python
from abstractllm import create_llm

# Cost-effective model for simpler tasks
simple_llm = create_llm("openai", model="gpt-3.5-turbo")
simple_response = simple_llm.generate("Summarize the water cycle")

# More capable model for complex reasoning
advanced_llm = create_llm("openai", model="gpt-4")
complex_response = advanced_llm.generate(
    "Analyze the potential economic impacts of quantum computing on cybersecurity"
)

# Vision-capable model
vision_llm = create_llm("openai", model="gpt-4-vision-preview")
```

## Asynchronous Usage

Use the OpenAI provider asynchronously:

```python
import asyncio
from abstractllm import create_llm

async def main():
    llm = create_llm("openai", model="gpt-3.5-turbo")
    
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

Handle OpenAI API-specific errors:

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
    llm = create_llm("openai", model="gpt-3.5-turbo")
    response = llm.generate("Write a very long story about a space adventure")
except InvalidAPIKeyError:
    print("Your API key is invalid. Check your credentials")
except RateLimitError:
    print("Rate limit exceeded. Try again later or adjust your request volume")
except TokenLimitError:
    print("The prompt is too long for the model's context window")
except ContentFilterError as e:
    print(f"Content filter triggered: {e}. The prompt may contain unsafe content")
except ProviderAPIError as e:
    print(f"API error: {e}")
```

## Cost Management

Manage token usage and costs:

```python
from abstractllm import create_llm

# Create LLM with token tracking
llm = create_llm("openai", model="gpt-3.5-turbo", track_tokens=True)

# Generate response
response = llm.generate("Explain the theory of relativity")

# Get token usage information
token_usage = llm.get_last_token_usage()
print(f"Prompt tokens: {token_usage.prompt_tokens}")
print(f"Completion tokens: {token_usage.completion_tokens}")
print(f"Total tokens: {token_usage.total_tokens}")

# For estimating costs (example rates, check current OpenAI pricing)
prompt_cost = token_usage.prompt_tokens * 0.0000015  # $0.0015 per 1K tokens
completion_cost = token_usage.completion_tokens * 0.000002  # $0.002 per 1K tokens
total_cost = prompt_cost + completion_cost
print(f"Estimated cost: ${total_cost:.6f}")
```

## Advanced Configurations

### Custom Base URL

Connect to a compatible API or proxy:

```python
from abstractllm import create_llm

llm = create_llm(
    "openai",
    model="gpt-3.5-turbo",
    base_url="https://your-custom-endpoint.com/v1"
)
```

### Proxy Usage

Configure a proxy for API requests:

```python
from abstractllm import create_llm

llm = create_llm(
    "openai",
    model="gpt-3.5-turbo",
    http_proxy="http://your-proxy-server:port",
    https_proxy="https://your-proxy-server:port"
)
```

### Request Retry Configuration

Configure automatic retries for network issues:

```python
from abstractllm import create_llm

llm = create_llm(
    "openai",
    model="gpt-3.5-turbo",
    max_retries=5,
    retry_wait_time=2.0  # seconds
)
```

## Best Practices

- **API Key Security**: Never hardcode API keys in your code. Use environment variables or secure credential storage.
- **Model Selection**: 
  - Use `gpt-3.5-turbo` for most general tasks to optimize cost
  - Use `gpt-4` or `gpt-4o` for complex reasoning and challenging tasks
  - Use `gpt-4-vision-preview` for image analysis
- **Token Management**: Monitor and optimize token usage to control costs.
  - Use shorter prompts when possible
  - For multi-turn conversations, consider periodically summarizing the context
- **Error Handling**: Implement robust error handling, particularly for rate limits and context window limits.
- **Streaming**: Use streaming for longer responses to improve user experience.
- **Prompt Design**: Structure your prompts clearly with specific instructions for better results.

## Common Issues

### Context Window Management

When working with large inputs:

```python
from abstractllm import create_llm

# For handling larger inputs, use models with bigger context windows
llm = create_llm("openai", model="gpt-3.5-turbo-16k")  # 16K context window

# For extremely long inputs, consider chunking or summarization
def process_large_document(document, chunk_size=10000):
    chunks = [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]
    summaries = []
    
    for chunk in chunks:
        summary = llm.generate(f"Summarize the following text concisely:\n\n{chunk}")
        summaries.append(summary)
    
    if len(summaries) > 1:
        final_summary = llm.generate(
            "Combine these summaries into a coherent overall summary:\n\n" + 
            "\n\n".join([f"Summary {i+1}: {s}" for i, s in enumerate(summaries)])
        )
        return final_summary
    else:
        return summaries[0]
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

llm = create_llm("openai", model="gpt-3.5-turbo")
response = generate_with_backoff(llm, "What is quantum physics?")
```

## Provider Implementation Details

For developers interested in how the OpenAI provider is implemented in AbstractLLM, see the [Provider Implementations](../architecture/providers.md) documentation. 