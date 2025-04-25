# HuggingFace Provider

The HuggingFace provider in AbstractLLM enables integration with thousands of open-source models through the HuggingFace Hub or local model files. This guide covers setup, configuration options, supported models, and provider-specific features.

## Setup

### Authentication

To use the HuggingFace provider with the Inference API or private models, you need a HuggingFace API token:

```python
from abstractllm import create_llm

# Using environment variable (recommended)
# Set HUGGINGFACE_API_KEY in your environment
llm = create_llm("huggingface", model="google/gemma-2b")

# Explicit API key
llm = create_llm(
    "huggingface", 
    model="google/gemma-2b", 
    api_key="your-huggingface-api-token-here"
)
```

For local models, no API key is required.

### Installation

To use the HuggingFace provider, install AbstractLLM with HuggingFace dependencies:

```bash
pip install abstractllm[huggingface]
```

This will install PyTorch, Transformers, and other required dependencies.

## Supported Models

The HuggingFace provider supports a wide range of models:

| Mode | Description | Examples |
|------|-------------|----------|
| Local Pipeline | Load and run models locally using HuggingFace pipelines | `meta-llama/Llama-2-7b`, `google/gemma-2b` |
| Local GGUF | Run optimized GGUF models locally | `TheBloke/Llama-2-7B-GGUF` |
| Inference API | Use models via HuggingFace's hosted Inference API | Any model on HuggingFace Hub |
| Endpoint API | Connect to custom HuggingFace Endpoints | Your custom endpoints |

## Basic Usage

### Local Model Usage

```python
from abstractllm import create_llm

# Use a local model via pipeline
llm = create_llm(
    "huggingface", 
    model="google/gemma-2b",
    device="cuda"  # Use GPU if available, or "cpu" for CPU
)

response = llm.generate("Explain quantum computing in simple terms.")
print(response)
```

### Inference API Usage

```python
from abstractllm import create_llm

# Use a model via the Inference API
llm = create_llm(
    "huggingface", 
    model="microsoft/phi-2",
    use_inference_api=True,
    api_key="your-huggingface-api-token-here"
)

response = llm.generate("Write a short poem about the ocean.")
print(response)
```

### GGUF Model Usage

```python
from abstractllm import create_llm

# Use a GGUF quantized model
llm = create_llm(
    "huggingface", 
    model_path="/path/to/model.gguf",
    model_type="gguf"
)

response = llm.generate("Create a short story about robots.")
print(response)
```

## Provider-Specific Features

### Model Loading Options

HuggingFace offers several options for loading models:

```python
from abstractllm import create_llm

# Load with 4-bit quantization for memory efficiency
llm = create_llm(
    "huggingface", 
    model="meta-llama/Llama-2-7b-hf",
    quantization="4bit",
    device_map="auto"  # Automatically manage model across available devices
)

# Load with specific configuration
llm = create_llm(
    "huggingface", 
    model="google/gemma-2b",
    model_kwargs={
        "trust_remote_code": True,
        "revision": "main",
        "torch_dtype": "bfloat16"
    }
)
```

### Generation Parameters

HuggingFace supports various generation parameters:

```python
from abstractllm import create_llm

llm = create_llm("huggingface", model="mistralai/Mistral-7B-v0.1")

# Advanced generation parameters
response = llm.generate(
    "Write a story about a magical forest.",
    temperature=0.7,        # Controls randomness
    max_tokens=500,         # Maximum length of generation
    top_p=0.95,             # Nucleus sampling parameter
    top_k=50,               # Top-k sampling parameter
    repetition_penalty=1.1, # Penalize repetitive tokens
    do_sample=True,         # Use sampling instead of greedy decoding
    num_return_sequences=1  # Number of completions to generate
)
```

### Pipeline Configuration

Configure text generation pipelines:

```python
from abstractllm import create_llm

# Use a specific pipeline configuration
llm = create_llm(
    "huggingface", 
    model="mistralai/Mistral-7B-v0.1",
    pipeline_kwargs={
        "return_full_text": False,  # Don't include prompt in output
        "clean_up_tokenization_spaces": True,
        "truncate": True
    }
)
```

## Configuration Options

The HuggingFace provider supports these configuration options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Model ID on HuggingFace Hub | None (required) |
| `model_path` | Path to local model file | None |
| `model_type` | Type of model ("pipeline", "gguf", etc.) | Auto-detected |
| `api_key` | HuggingFace API token | From environment |
| `use_inference_api` | Whether to use Inference API | False |
| `device` | Device to run model on ("cuda", "cpu", etc.) | "cuda" if available |
| `device_map` | Device mapping for large models | None |
| `quantization` | Quantization method for efficiency | None |
| `temperature` | Controls randomness (0.0 to 1.0) | 0.7 |
| `max_tokens` | Maximum tokens to generate | 512 |
| `model_kwargs` | Additional kwargs for model loading | {} |
| `pipeline_kwargs` | Additional kwargs for pipeline | {} |
| `inference_api_url` | Custom Inference API URL | Default HF Inference API |

## Session Management

Maintain conversation context with sessions:

```python
from abstractllm import create_llm
from abstractllm.session import Session

llm = create_llm("huggingface", model="mistralai/Mistral-7B-v0.1")
session = Session(provider=llm)

# First interaction
response1 = session.generate("What is machine learning?")
print(response1)

# Follow-up question
response2 = session.generate("Give some practical applications of it.")
print(response2)
```

## Asynchronous Usage

Use the HuggingFace provider asynchronously:

```python
import asyncio
from abstractllm import create_llm

async def main():
    llm = create_llm("huggingface", model="mistralai/Mistral-7B-v0.1")
    
    # Async generation
    response = await llm.generate_async("What is the capital of France?")
    print(response)
    
    # Process multiple prompts concurrently (if using Inference API)
    if llm.config_manager.get_param("use_inference_api"):
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

Note: Asynchronous usage with local models may not provide performance benefits and depends on the implementation.

## Error Handling

Handle HuggingFace-specific errors:

```python
from abstractllm import create_llm
from abstractllm.errors import (
    ProviderAPIError, 
    ModelNotFoundError,
    ResourceExhaustedError,
    InvalidArgumentError
)

try:
    llm = create_llm("huggingface", model="mistralai/Mistral-7B-v0.1")
    response = llm.generate("Explain neural networks.")
except ModelNotFoundError:
    print("The specified model could not be found. Check the model ID or path.")
except ResourceExhaustedError:
    print("Insufficient resources (memory/GPU) to load the model.")
except InvalidArgumentError as e:
    print(f"Invalid argument: {e}")
except ProviderAPIError as e:
    print(f"API error: {e}")
```

## Performance Considerations

### Memory Management

Managing memory usage for large models:

```python
from abstractllm import create_llm

# Load with 8-bit quantization for reduced memory usage
llm = create_llm(
    "huggingface", 
    model="meta-llama/Llama-2-70b-hf",
    quantization="8bit",  # or "4bit" for even lower memory usage
    device_map="auto"     # Automatically distribute across GPUs
)
```

### CPU vs. GPU Performance

Performance varies significantly between CPU and GPU:

```python
from abstractllm import create_llm

# CPU usage (slower but no GPU required)
llm_cpu = create_llm(
    "huggingface", 
    model="google/gemma-2b", 
    device="cpu"
)

# GPU usage (much faster with compatible GPU)
llm_gpu = create_llm(
    "huggingface", 
    model="google/gemma-2b", 
    device="cuda",
    model_kwargs={"torch_dtype": "float16"}  # Using lower precision for better performance
)
```

## Common Issues

### Out of Memory Errors

If you encounter out of memory errors:

```python
from abstractllm import create_llm

# Solutions for OOM errors
llm = create_llm(
    "huggingface",
    model="meta-llama/Llama-2-7b-hf",
    # Choose one or more of these options:
    quantization="4bit",             # Reduce precision
    model_kwargs={"device_map": "auto"},  # Split across multiple GPUs
    max_tokens=256                   # Limit context size
)
```

### Slow Performance

For slow performance issues:

```python
from abstractllm import create_llm

# Use a smaller or more optimized model
llm = create_llm(
    "huggingface", 
    model="microsoft/phi-2",  # Smaller model with good performance
    model_kwargs={
        "torch_dtype": "float16",  # Use lower precision
        "low_cpu_mem_usage": True  # Optimize for lower memory usage
    }
)
```

### Model Compatibility

Not all models work with all features:

```python
from abstractllm import create_llm, ModelCapability

llm = create_llm("huggingface", model="mistralai/Mistral-7B-v0.1")
capabilities = llm.get_capabilities()

# Check capabilities before using features
if capabilities.get(ModelCapability.STREAMING):
    # Use streaming...
else:
    # Use non-streaming alternative...
```

## Best Practices

- **Model Selection**: 
  - Choose smaller models (2B-7B parameters) for faster inference 
  - Use quantized models to reduce memory requirements
  - Consider latency vs. quality tradeoffs for your application

- **Resource Management**:
  - Monitor GPU memory usage with `nvidia-smi` or similar tools
  - Use appropriate batch sizes for your hardware
  - Release resources when not in use with explicit model unloading

- **Production Deployment**:
  - Consider using dedicated inference servers like TGI or vLLM instead of direct model loading for production
  - Implement proper error handling and fallbacks
  - Monitor performance and resource usage

## Conclusion

The HuggingFace provider in AbstractLLM offers unparalleled flexibility in model selection and deployment options. Whether you need to run models locally on your hardware, access hosted models via the Inference API, or work with custom endpoints, the HuggingFace provider provides a consistent interface that integrates seamlessly with the rest of AbstractLLM.

For more advanced usage and detailed API documentation, see the [HuggingFace Provider API Reference](../api-reference/index.md) and the [HuggingFace Provider Implementation](../architecture/providers.md) documentation. 