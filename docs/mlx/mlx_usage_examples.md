# MLX Provider Usage Examples

This document provides examples of how to use the MLX provider in AbstractLLM.

## Prerequisites

To use the MLX provider, you need:

1. An Apple Silicon Mac (M1/M2/M3 series)
2. macOS operating system
3. Python 3.8 or higher
4. MLX and MLX-LM packages installed

## Installation

```bash
pip install abstractllm
pip install mlx mlx-lm
```

## Basic Usage

### Simple Text Generation

```python
from abstractllm import create_llm

# Create an MLX provider with default settings
llm = create_llm("mlx")

# Generate text
response = llm.generate("Explain what MLX is in a few sentences.")
print(response.content)
```

### Specifying a Model

```python
from abstractllm import create_llm, ModelParameter

# Create an MLX provider with a specific model
llm = create_llm("mlx", **{
    ModelParameter.MODEL: "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit",
    ModelParameter.TEMPERATURE: 0.7,
    ModelParameter.MAX_TOKENS: 500
})

# Generate text
response = llm.generate("Write a short poem about autumn.")
print(response.content)
```

### Streaming Generation

```python
from abstractllm import create_llm

# Create an MLX provider
llm = create_llm("mlx")

# Generate text with streaming
for chunk in llm.generate("Tell me a short story.", stream=True):
    print(chunk.content, end="", flush=True)
print()
```

### Using System Prompts

```python
from abstractllm import create_llm

# Create an MLX provider
llm = create_llm("mlx")

# Generate text with a system prompt
response = llm.generate(
    prompt="What's the best way to learn programming?",
    system_prompt="You are a helpful programming tutor with 20 years of experience."
)
print(response.content)
```

### Processing Files

```python
from abstractllm import create_llm
from pathlib import Path

# Create an MLX provider
llm = create_llm("mlx")

# Generate text with a file
response = llm.generate(
    prompt="Explain what this code does:",
    files=[Path("example.py")]
)
print(response.content)
```

### Async Generation

```python
import asyncio
from abstractllm import create_llm

async def generate_async():
    # Create an MLX provider
    llm = create_llm("mlx")
    
    # Generate text asynchronously
    response = await llm.generate_async("What is the capital of France?")
    print(response.content)
    
    # Stream text asynchronously
    async for chunk in await llm.generate_async("Count from 1 to 5.", stream=True):
        print(chunk.content, end="", flush=True)
    print()

# Run the async function
asyncio.run(generate_async())
```

## Advanced Usage

### Checking Model Capabilities

```python
from abstractllm import create_llm, ModelCapability

# Create an MLX provider
llm = create_llm("mlx")

# Check capabilities
capabilities = llm.get_capabilities()
print(f"Supports streaming: {capabilities.get(ModelCapability.STREAMING)}")
print(f"Supports system prompts: {capabilities.get(ModelCapability.SYSTEM_PROMPT)}")
print(f"Supports vision: {capabilities.get(ModelCapability.VISION)}")
print(f"Max tokens: {capabilities.get(ModelCapability.MAX_TOKENS)}")
```

### Model Caching

MLX provider automatically caches models in memory to speed up subsequent runs. You can clear the cache if needed:

```python
from abstractllm import create_llm
from abstractllm.providers.mlx_provider import MLXProvider

# Clear all cached models
MLXProvider.clear_model_cache()

# Clear a specific model
MLXProvider.clear_model_cache("mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit")
```

### Listing Cached Models

```python
from abstractllm.providers.mlx_provider import MLXProvider

# List cached models
cached_models = MLXProvider.list_cached_models()
for model in cached_models:
    print(f"Model: {model['name']}, Size: {model['size']}")
```

## Command Line Example

The AbstractLLM package includes an example script for the MLX provider. You can run it as follows:

```bash
# Navigate to the examples directory
cd examples

# Run the example script
python mlx_example.py --prompt "Explain quantum computing in simple terms."

# Use streaming
python mlx_example.py --prompt "Tell me a story." --stream

# Specify a model
python mlx_example.py --model "mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit" --prompt "Hello!"

# Include a file
python mlx_example.py --prompt "Explain this code:" --file path/to/file.py
```

## Performance Tips

1. **Model Size**: Smaller models run faster but may produce lower quality results.
2. **Quantization**: Quantized models (4-bit, 8-bit) use less memory and run faster.
3. **Max Tokens**: Limit the max_tokens parameter to speed up generation.
4. **Caching**: The first run will be slower as the model loads, subsequent runs will be faster.

## Troubleshooting

1. **ImportError**: Ensure you have installed `mlx` and `mlx-lm` packages.
2. **UnsupportedFeatureError**: Ensure you're running on an Apple Silicon Mac.
3. **ModelNotFoundError**: Check that the model name is correct and available on Hugging Face.
4. **Memory Issues**: Try using a smaller or more quantized model.

## Available Models

Some models that work well with the MLX provider:

- `mlx-community/Josiefied-Qwen3-8B-abliterated-v1-4bit` (text generation)
- `mlx-community/Llama-3.1-8B-Instruct` (text generation)
- `mlx-community/gemma-3-27b-it-qat-3bit` (vision capabilities)

For a complete list, see the [MLX Community on Hugging Face](https://huggingface.co/mlx-community). 