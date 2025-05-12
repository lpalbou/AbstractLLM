# MLX Provider

The MLX provider leverages Apple's [MLX framework](https://github.com/ml-explore/mlx) for efficient language model inference on Apple Silicon devices. This provider enables you to run LLMs locally on your Mac with M1/M2/M3 chips, taking advantage of the unified memory architecture and Metal GPU acceleration.

## Overview

MLX is a framework for machine learning on Apple Silicon, developed by Apple's machine learning research team. It provides significant performance benefits through:

1. **Unified Memory Architecture**: Arrays live in shared memory, enabling seamless operations across CPU and GPU without data transfers
2. **Metal GPU Acceleration**: Optimized for Apple's Metal framework
3. **Lazy Computation**: Operations are only materialized when needed, improving efficiency
4. **Dynamic Graph Construction**: No slow recompilations when changing function argument shapes

## Installation

The MLX provider requires macOS with Apple Silicon (M1/M2/M3 chips):

```bash
pip install "abstractllm[mlx]"
```

This will install the required dependencies:
- `mlx`: Core MLX framework
- `mlx-lm`: Language model support
- Supporting libraries like `huggingface_hub` and `transformers`

## Basic Usage

```python
from abstractllm import create_llm

# Create an MLX-powered LLM
llm = create_llm("mlx", model="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX")

# Generate text with the same API as other providers
response = llm.generate("Explain the benefits of Apple Silicon for machine learning")
print(response.content)
```

## Supported Models

The MLX provider supports models that have been converted to the MLX format. Many popular models are available through the [mlx-community](https://huggingface.co/mlx-community) organization on Hugging Face:

- `mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX`
- `mlx-community/Phi-3-mini-4k-instruct-mlx`
- `mlx-community/qwen2.5-coder-14b-instruct-abliterated`
- `mlx-community/Mistral-7B-Instruct-v0.2`
- `mlx-community/llava-1.5-7b-mlx` (vision capability)

The first time you use a model, it will be downloaded from Hugging Face and cached locally.

## Configuration Options

```python
llm = create_llm(
    "mlx",
    model="mlx-community/Phi-3-mini-4k-instruct-mlx",  # Model ID from Hugging Face
    temperature=0.7,                                   # Temperature for sampling (0.0 to 1.0)
    max_tokens=4096,                                   # Maximum tokens to generate
    top_p=0.9,                                         # Top-p sampling parameter
    cache_dir=None,                                    # Custom cache directory (optional)
    quantize=True                                      # Whether to use quantized models
)
```

## Features

### Streaming Support

```python
# Generate with streaming for real-time responses
for chunk in llm.generate("Explain quantum computing", stream=True):
    print(chunk.content, end="", flush=True)
```

### Asynchronous Support

```python
import asyncio

async def generate_async():
    response = await llm.generate_async("What is the meaning of life?")
    print(response.content)

asyncio.run(generate_async())
```

### System Prompts

```python
# Using system prompts to guide the model's behavior
response = llm.generate(
    "Explain how recursion works in Python",
    system_prompt="You are an expert programmer who explains code in simple terms."
)
```

### Vision Capabilities

Some MLX models support vision capabilities:

```python
# Check if the model supports vision
if llm.get_capabilities().get(ModelCapability.VISION):
    # Process an image
    response = llm.generate(
        "What's in this image?",
        files=["path/to/image.jpg"]
    )
```

### Model Caching

The MLX provider implements both disk-based caching (using Hugging Face's cache) and in-memory caching for efficient model switching:

```python
# List cached models
from abstractllm.providers.mlx_provider import MLXProvider
cached_models = MLXProvider.list_cached_models()
for model in cached_models:
    print(f"Model: {model['name']}, Size: {model['size']}, Last used: {model['last_used']}")

# Clear specific model from memory
MLXProvider.clear_model_cache("mlx-community/Phi-3-mini-4k-instruct-mlx")

# Clear all models from memory
MLXProvider.clear_model_cache()
```

## Performance Considerations

For optimal performance with the MLX provider:

1. **Use quantized models**: 4-bit quantized models offer the best balance of performance and quality
2. **Prefer smaller models**: Smaller models like Phi-3-mini run faster on lower-end Apple Silicon
3. **Manage memory usage**: The provider caches models in memory, but you can clear the cache when needed
4. **Batch processing**: For multiple generations, reuse the same model rather than switching

## Limitations

- **Platform-specific**: Only works on macOS with Apple Silicon (M1/M2/M3)
- **No tool calling**: The MLX provider does not currently support function calling or tool use
- **Limited vision support**: Only certain models support vision capabilities

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'mlx'**
   - Solution: Install the MLX dependencies with `pip install "abstractllm[mlx]"`

2. **UnsupportedFeatureError: MLX provider requires Apple Silicon**
   - Solution: The MLX provider only works on Macs with Apple Silicon

3. **Failed to load model**
   - Solution: Check your internet connection, as the model needs to be downloaded from Hugging Face

4. **Out of memory errors**
   - Solution: Try a smaller model or clear the model cache with `MLXProvider.clear_model_cache()`

## Examples

### Basic Chat Interface

```python
from abstractllm import create_llm

def simple_chat():
    llm = create_llm("mlx", model="mlx-community/Phi-3-mini-4k-instruct-mlx")
    
    print("Chat with the MLX-powered AI (type 'exit' to quit):")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        response = llm.generate(user_input)
        print(f"\nAI: {response.content}")

# Run the chat interface
simple_chat()
```

### Processing Text Files

```python
from abstractllm import create_llm
from pathlib import Path

# Create an MLX-powered LLM
llm = create_llm("mlx", model="mlx-community/Phi-3-mini-4k-instruct-mlx")

# Process a text file
text_file = Path("document.txt")
response = llm.generate(
    "Summarize this document",
    files=[text_file]
)

print(response.content)
```

## Resources

- [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- [MLX-LM GitHub Repository](https://github.com/ml-explore/mlx-lm)
- [MLX Community Models on Hugging Face](https://huggingface.co/mlx-community)
- [AbstractLLM MLX Integration Documentation](../mlx/mlx_integration_architecture.md) 