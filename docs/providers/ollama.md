# Ollama Provider

The Ollama provider in AbstractLLM enables you to use local LLMs through [Ollama](https://ollama.ai/), allowing you to run models on your own hardware. This guide covers setup, configuration, model options, and provider-specific features.

## Setup

### Prerequisites

Before using the Ollama provider, you need to:

1. Install Ollama following the instructions on the [official website](https://ollama.ai/download)
2. Pull the models you want to use with the `ollama pull` command

### Installation

To use the Ollama provider, install AbstractLLM with the Ollama dependencies:

```bash
pip install abstractllm[ollama]
```

### Creating an Instance

Create an instance of the Ollama provider:

```python
from abstractllm import create_llm

# Basic configuration with default local Ollama server
llm = create_llm("ollama", model="llama2")

# With custom server URL
llm = create_llm(
    "ollama",
    model="mistral",
    api_base="http://custom-ollama-server:11434"
)

# With additional parameters
llm = create_llm(
    "ollama",
    model="llama2",
    temperature=0.7,
    max_tokens=2000,
    system_prompt="You are a helpful AI assistant."
)
```

## Supported Models

The Ollama provider supports any model available through Ollama. Some popular models include:

| Model | Description | Special Capabilities |
|-------|-------------|---------------------|
| llama2 | Meta's Llama 2 model | General text generation |
| llama2:13b | Larger Llama 2 variant | Enhanced reasoning |
| llama2:70b | Largest Llama 2 variant | Advanced reasoning |
| mistral | Mistral 7B model | Good performance-to-size ratio |
| mixtral | Mixtral 8x7B MoE model | High capability at reasonable speed |
| phi2 | Microsoft's Phi-2 model | Compact but capable |
| gemma | Google's Gemma model | Efficient text generation |
| codellama | Specialized for code generation | Code completion and explanation |
| llava | Multimodal model with vision capabilities | Vision (image understanding) |
| bakllava | Multimodal model based on Llama | Vision (image understanding) |

You can see all available models with `ollama list` in your terminal or by checking the [Ollama Library](https://ollama.ai/library).

## Basic Usage

### Simple Text Generation

```python
from abstractllm import create_llm

llm = create_llm("ollama", model="mistral")
response = llm.generate("What is the capital of France?")
print(response)  # Paris
```

### Streaming Responses

```python
from abstractllm import create_llm

llm = create_llm("ollama", model="llama2")
for chunk in llm.generate("Explain quantum computing", stream=True):
    print(chunk, end="", flush=True)
```

## Provider-Specific Features

### Vision Capabilities

Use vision-capable models like LLaVA or BakLLaVa:

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create LLM with a vision-capable model
llm = create_llm("ollama", model="llava")

# Process an image
image = ImageInput.from_file("path/to/image.jpg")
response = llm.generate("What's in this image?", media=[image])

print(response)
```

### JSON Mode

Some models support structured JSON output:

```python
from abstractllm import create_llm

llm = create_llm("ollama", model="llama2", format="json")

prompt = """
Generate a list of 3 fictional characters with their name, age, and occupation.
Return the result as a JSON array.
"""

response = llm.generate(prompt)
print(response)
# Expected structured JSON response
```

### Tool Use

For models that support tool use (function calling):

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
            }
        },
        "required": ["location"]
    }
)

# Create LLM with a tool-capable model
llm = create_llm("ollama", model="mixtral", tools=[get_weather])

# Function to be called by the model
def get_weather_function(location):
    return f"The weather in {location} is sunny and 72 degrees."

# Handle the response with potential tool calls
response = llm.generate(
    "What's the weather like in Boston?",
    tool_handler=lambda tool_name, args: get_weather_function(**args) if tool_name == "get_weather" else None
)

print(response)
```

Note: Tool use capability varies by model and may require fine-tuning or specific model variants.

## Configuration Options

The Ollama provider supports these configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | The model to use | "llama2" |
| `temperature` | Controls randomness (0.0 to 1.0) | 0.7 |
| `max_tokens` | Maximum tokens to generate | 512 |
| `top_p` | Nucleus sampling parameter | 1.0 |
| `top_k` | Limits token selection to top K options | 40 |
| `system_prompt` | System instructions for the model | None |
| `api_base` | Ollama server URL | "http://localhost:11434" |
| `format` | Response format (e.g., "json") | None |
| `mirostat` | Enable Mirostat sampling technique | 0 |
| `mirostat_eta` | Mirostat learning rate | 0.1 |
| `mirostat_tau` | Mirostat target entropy | 5.0 |
| `num_ctx` | Size of context window | Model-dependent |
| `num_gpu` | Number of GPUs to use | -1 (auto) |
| `num_thread` | Number of CPU threads to use | -1 (auto) |
| `repeat_last_n` | Look back size for repetition penalty | 64 |
| `repeat_penalty` | Penalty for repetitions | 1.1 |
| `seed` | Random seed for reproducibility | -1 (random) |
| `stop` | Sequences where generation should stop | [] |

## Session Management

Create a session for maintaining conversation history:

```python
from abstractllm import create_llm

llm = create_llm("ollama", model="llama2")
session = llm.create_session(system_prompt="You are a helpful assistant.")

# First interaction
response1 = session.generate("What is machine learning?")
print(response1)

# Follow-up question (context is preserved)
response2 = session.generate("Give me some example applications.")
print(response2)
```

## Asynchronous Usage

Use the Ollama provider asynchronously:

```python
import asyncio
from abstractllm import create_llm

async def main():
    llm = create_llm("ollama", model="mistral")
    
    # Async generation
    response = await llm.generate_async("What is the capital of Japan?")
    print(response)
    
    # Async streaming
    async for chunk in llm.generate_async("Explain nuclear fusion", stream=True):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Error Handling

Handle Ollama-specific errors:

```python
from abstractllm import create_llm
from abstractllm.errors import ProviderAPIError, ModelNotFoundError

try:
    llm = create_llm("ollama", model="nonexistent-model")
    response = llm.generate("Hello")
except ModelNotFoundError:
    print("Model not found. Try pulling it with 'ollama pull model-name'")
except ProviderAPIError as e:
    if "connection" in str(e).lower():
        print("Cannot connect to Ollama. Is the server running?")
    else:
        print(f"API error: {e}")
```

## Performance Optimization

Optimize performance for Ollama models:

```python
from abstractllm import create_llm

# For faster inference on GPU
llm = create_llm(
    "ollama",
    model="llama2",
    num_gpu=1,  # Use 1 GPU
    num_ctx=2048,  # Smaller context window for speed
    temperature=0.0  # Deterministic output (faster)
)

# For lower memory usage
llm = create_llm(
    "ollama",
    model="phi2",  # Smaller model
    num_gpu=0,  # CPU only
    num_thread=4  # Limit CPU threads
)
```

## Local Model Management

AbstractLLM integrates with Ollama's model management:

```python
from abstractllm import create_llm
from abstractllm.providers.ollama import OllamaProvider

# Check available models
models = OllamaProvider.list_models(api_base="http://localhost:11434")
print(f"Available models: {models}")

# Check if a model exists
has_model = OllamaProvider.has_model("llama2", api_base="http://localhost:11434")
if not has_model:
    # Pull the model using subprocess or direct API call
    import subprocess
    subprocess.run(["ollama", "pull", "llama2"])
```

## Best Practices

- **Hardware Requirements**: Ensure your hardware meets the model's requirements. Larger models (13B+) typically need more RAM and benefit from GPU acceleration.
- **Model Selection**: Choose smaller models (7B or less) for faster responses with limited hardware.
- **Batching**: Process multiple requests in batches rather than one at a time for better throughput.
- **Context Window Management**: Keep prompts concise to avoid hitting context window limits, especially with smaller models.
- **Temperature Setting**: Use lower temperature (0.0-0.3) for factual responses and higher (0.7-1.0) for creative content.
- **Local Networking**: When running in container environments, ensure proper network configuration to connect to the Ollama service.

## Common Issues

### Connection Problems

If you encounter connection issues:

```python
import requests
from abstractllm import create_llm

# Check if Ollama server is running
try:
    requests.get("http://localhost:11434/api/tags")
    print("Ollama server is running")
except:
    print("Cannot connect to Ollama server. Is it running?")
    print("Start it with 'ollama serve' or check installation")

# Try with explicit timeout settings
llm = create_llm(
    "ollama", 
    model="llama2",
    request_timeout=60  # Set longer timeout for slow models
)
```

### Out of Memory Errors

For models that exceed your hardware capabilities:

```python
from abstractllm import create_llm

# Use a smaller model
llm = create_llm("ollama", model="phi2")  # Instead of llama2:70b

# Or reduce context window size
llm = create_llm(
    "ollama",
    model="llama2",
    num_ctx=1024  # Smaller context window
)
```

### Slow First Generation

The first generation with a model can be slow due to model loading:

```python
from abstractllm import create_llm
import time

llm = create_llm("ollama", model="llama2")

# Pre-load the model with a simple query
start = time.time()
_ = llm.generate("hi")
print(f"First generation took {time.time() - start:.2f} seconds")

# Subsequent queries should be faster
start = time.time()
_ = llm.generate("What is the capital of France?")
print(f"Second generation took {time.time() - start:.2f} seconds")
```

## Provider Implementation Details

For developers interested in how the Ollama provider is implemented in AbstractLLM, see the [Provider Implementations](../architecture/providers.md) documentation. 