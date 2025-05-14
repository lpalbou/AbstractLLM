# Task 16: Create MLX Provider Documentation

## Description
Create comprehensive documentation for the MLX provider in AbstractLLM's documentation system.

## Requirements
1. Create a new documentation file in the appropriate docs directory
2. Include information about installation, basic usage, and configuration
3. Document platform requirements and dependencies
4. Provide troubleshooting tips

## Implementation Details

Create a file at `docs/providers/mlx.md` (or appropriate location based on the existing documentation structure):

```markdown
# MLX Provider

The MLX provider enables efficient language model inference on Apple Silicon devices using Apple's [MLX framework](https://github.com/ml-explore/mlx).

## Platform Requirements

- macOS (running on Apple Silicon: M1/M2/M3 series chips)
- Python 3.8 or newer

## Installation

Install AbstractLLM with MLX support:

```bash
pip install "abstractllm[mlx]"
```

This will install the required dependencies:
- mlx: Apple's machine learning framework
- mlx-lm: MLX utilities for language models
- huggingface_hub: For model discovery and caching

## Basic Usage

```python
from abstractllm import create_llm

# Create an LLM with MLX provider
llm = create_llm("mlx")

# Generate text
response = llm.generate("What is MLX?")
print(response.text)
```

## Configuration

The MLX provider supports the following configuration options:

| Parameter     | Type    | Default                                         | Description                              |
|---------------|---------|------------------------------------------------|------------------------------------------|
| model         | str     | "mlx-community/qwen2.5-coder-14b-instruct-abliterated" | Model identifier from Hugging Face    |
| temperature   | float   | 0.7                                            | Generation temperature (randomness)      |
| max_tokens    | int     | 4096                                           | Maximum tokens to generate               |
| top_p         | float   | 0.9                                            | Nucleus sampling parameter              |
| cache_dir     | str     | None (uses default HF cache)                   | Custom cache directory                  |
| quantize      | bool    | True                                           | Whether to use quantized models         |

Example with custom configuration:

```python
from abstractllm import create_llm, ModelParameter

llm = create_llm(
    "mlx",
    **{
        ModelParameter.MODEL: "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
        ModelParameter.TEMPERATURE: 0.8,
        ModelParameter.MAX_TOKENS: 2048,
        ModelParameter.TOP_P: 0.95,
        "cache_dir": "~/custom_cache"
    }
)
```

## Supported Models

The MLX provider supports models that have been converted to the MLX format. These models typically have "mlx" in their repository name on Hugging Face.

Some recommended models:

- **mlx-community/phi-2**: Small but capable model (good for testing)
- **mlx-community/qwen2.5-coder-14b-instruct-abliterated**: Code-specialized model
- **mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX**: Good all-around model

## Streaming Support

The MLX provider supports streaming responses:

```python
for chunk in llm.generate("Explain quantum computing", stream=True):
    print(chunk.text, end="", flush=True)
```

## Async Support

For use in asynchronous applications:

```python
import asyncio

async def generate_async():
    llm = create_llm("mlx")
    response = await llm.generate_async("What is async programming?")
    print(response.text)

asyncio.run(generate_async())
```

## File Handling

The MLX provider supports processing text files:

```python
from pathlib import Path

llm = create_llm("mlx")
response = llm.generate("Summarize this file:", files=[Path("document.txt")])
print(response.text)
```

## Vision Support

Some MLX models support vision capabilities. Vision support will be automatically detected based on the model name:

```python
from abstractllm import create_llm, ModelCapability

llm = create_llm("mlx", model="mlx-community/llava-1.5-7b-mlx")

# Check if vision is supported
if llm.get_capabilities().get(ModelCapability.VISION, False):
    response = llm.generate("What's in this image?", files=["image.jpg"])
    print(response.text)
```

## Troubleshooting

### Common Issues

1. **ImportError for MLX**: Ensure you've installed with MLX dependencies: `pip install "abstractllm[mlx]"`

2. **Not on Apple Silicon**: The MLX provider only works on macOS with Apple Silicon. You'll see an error if trying to use it on other platforms.

3. **Model Not Found**: Ensure the model exists on Hugging Face Hub and is compatible with MLX.

4. **Memory Issues**: Large models may require significant memory. Try using a smaller or quantized model.

### Checking System Compatibility

To verify your system is compatible:

```python
import platform

is_macos = platform.system().lower() == "darwin"
is_arm = platform.processor() == "arm"

print(f"Is macOS: {is_macos}")
print(f"Is Apple Silicon: {is_arm}")
print(f"MLX compatible: {is_macos and is_arm}")
```

## Further Resources

- [MLX GitHub Repository](https://github.com/ml-explore/mlx)
- [MLX-LM GitHub Repository](https://github.com/ml-explore/mlx-lm)
- [Hugging Face MLX Models](https://huggingface.co/models?other=mlx)
```

## References
- See AbstractLLM's existing provider documentation
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- Reference the MLX Usage Examples: `docs/mlx/mlx_usage_examples.md`

## Testing
1. Verify all documentation sections are complete and accurate
2. Ensure code examples run correctly on Apple Silicon
3. Check for formatting issues in rendered documentation
``` 