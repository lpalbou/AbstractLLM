# MLX Provider Usage Guide

The MLX provider in AbstractLLM allows you to run inference on Apple Silicon devices using the MLX framework.

## Requirements

- macOS running on Apple Silicon (M1/M2/M3)
- Python 3.8+
- MLX and related packages installed

## Installation

To use the MLX provider, you need to install the required packages:

```bash
# Install MLX and related packages
pip install mlx mlx-lm mlx-vlm

# For vision models, PyTorch is also required
pip install torch
```

## Basic Usage

```python
from abstractllm import create_llm
from abstractllm.enums import ModelParameter

# Create an MLX-based LLM
llm = create_llm("mlx", **{
    ModelParameter.MODEL: "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
    ModelParameter.MAX_TOKENS: 100,
    ModelParameter.TEMPERATURE: 0.7
})

# Generate text
response = llm.generate("What is the capital of France?")
print(response.content)

# Stream generation
for chunk in llm.generate("Tell me about Paris.", stream=True):
    print(chunk.content, end="", flush=True)
```

## Vision Models

The MLX provider also supports vision models:

```python
from abstractllm import create_llm
from abstractllm.enums import ModelParameter

# Create a vision-capable MLX-based LLM
llm = create_llm("mlx", **{
    ModelParameter.MODEL: "mlx-community/llava-1.5-7b-4bit",
    ModelParameter.MAX_TOKENS: 100
})

# Generate text based on an image
response = llm.generate("What's in this image?", files=["path/to/image.jpg"])
print(response.content)
```

## Available Models

### Text Models

- `mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX`
- `mlx-community/Phi-3-mini-4k-instruct-4bit`
- `mlx-community/mistral-7b-instruct-v0.2-4bit`
- And many others from the mlx-community organization on Hugging Face

### Vision Models

- `mlx-community/llava-1.5-7b-4bit`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`
- `mlx-community/deepseek-vl2-4bit`
- `mlx-community/Kimi-VL-A3B-Thinking-4bit`
- `mlx-community/nanoLLaVA-4bit`

## Limitations

- The MLX provider only works on macOS with Apple Silicon.
- Vision models require PyTorch to be installed.
- Some models may have specific requirements or limitations.

## Troubleshooting

### Vision Models Not Working

If you encounter errors with vision models such as:

```
Failed to process inputs with error: unsupported operand type(s) for //: 'int' and 'NoneType'. Please install PyTorch and try again.
```

Make sure you have PyTorch installed:

```bash
pip install torch
```

### Model Not Found

If you see errors about models not being found, check that:

1. You're using the correct model ID
2. You have internet access to download the model
3. You have sufficient disk space for the model

You can list available models from the mlx-community organization on Hugging Face:

```python
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(author='mlx-community')
for model in models[:10]:  # Show first 10 models
    print(model.id)
``` 