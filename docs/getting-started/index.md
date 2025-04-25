# Getting Started with AbstractLLM

This guide will help you get up and running with AbstractLLM quickly. We'll cover installation, basic setup, and your first interactions with various LLM providers.

## Installation

AbstractLLM is designed with a modular approach to dependencies. You can install only what you need based on the providers you plan to use.

### Basic Installation

For the core functionality without any specific provider:

```bash
pip install abstractllm
```

### Provider-Specific Installation

Install with specific provider dependencies:

```bash
# OpenAI
pip install "abstractllm[openai]"

# Anthropic (Claude)
pip install "abstractllm[anthropic]"

# Ollama (local models)
pip install "abstractllm[ollama]"

# HuggingFace (includes PyTorch)
pip install "abstractllm[huggingface]"
```

### Multiple Providers

You can combine multiple providers:

```bash
pip install "abstractllm[openai,anthropic]"
```

### All Dependencies

To install all dependencies:

```bash
pip install "abstractllm[all]"
```

### Tool Calling Support

To use the tool/function calling capabilities:

```bash
pip install "abstractllm[tools]"
```

## Basic Usage

### Creating an LLM Instance

The main entry point to AbstractLLM is the `create_llm()` function, which returns a provider instance based on the specified provider name.

```python
from abstractllm import create_llm

# Create an OpenAI provider
openai_llm = create_llm(
    "openai", 
    model="gpt-4",
    api_key="your-api-key"  # Optional: can also use OPENAI_API_KEY env variable
)

# Create an Anthropic provider
anthropic_llm = create_llm(
    "anthropic", 
    model="claude-3-opus-20240229",
    api_key="your-api-key"  # Optional: can also use ANTHROPIC_API_KEY env variable
)

# Create an Ollama provider
ollama_llm = create_llm(
    "ollama", 
    model="llama3"
)

# Create a HuggingFace provider
hf_llm = create_llm(
    "huggingface", 
    model="google/gemma-2b"
)
```

### Generating Text

To generate text from a prompt:

```python
# Simple generation
response = openai_llm.generate("Explain quantum computing in simple terms.")
print(response)

# With system prompt
response = anthropic_llm.generate(
    prompt="What are the benefits of exercise?",
    system_prompt="You are a helpful fitness expert."
)
print(response)

# With additional parameters
response = ollama_llm.generate(
    prompt="Write a short poem about the ocean.",
    temperature=0.7,
    max_tokens=100
)
print(response)
```

### Streaming Responses

To stream responses as they're generated:

```python
# Stream responses
for chunk in openai_llm.generate(
    prompt="Tell me a long story about a space adventure.",
    stream=True
):
    print(chunk, end="", flush=True)
```

### Async Generation

AbstractLLM supports asynchronous generation:

```python
import asyncio

async def generate_async():
    response = await anthropic_llm.generate_async(
        prompt="What are the key principles of machine learning?"
    )
    print(response)

asyncio.run(generate_async())
```

## API Key Management

AbstractLLM supports two ways to provide API keys:

1. **Environment Variables**:
   - OpenAI: `OPENAI_API_KEY`
   - Anthropic: `ANTHROPIC_API_KEY`

2. **Explicit Parameter**:
   ```python
   llm = create_llm("openai", api_key="your-api-key")
   ```

## Next Steps

Now that you have AbstractLLM installed and can generate basic responses, you can explore more advanced features:

- [Working with Sessions](../user-guide/sessions.md)
- [Using Vision Capabilities](../user-guide/vision.md)
- [Implementing Tool Calls](../user-guide/tools.md)
- [Provider Interchangeability](../user-guide/interchangeability.md)
- [Configuration Options](../user-guide/configuration.md) 