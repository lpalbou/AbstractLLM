# AbstractLLM Usage Guide

This guide demonstrates how to use the AbstractLLM package in your applications.

## Installation

```bash
# Basic installation
pip install abstractllm

# Install with specific provider dependencies
pip install abstractllm[openai]     # For OpenAI
pip install abstractllm[anthropic]  # For Anthropic
pip install abstractllm[huggingface]  # For Hugging Face

# Install with all provider dependencies
pip install abstractllm[all]
```

## Basic Usage

Using AbstractLLM is straightforward. Here's a simple example:

```python
from abstractllm import create_llm

# Create an LLM instance
llm = create_llm("openai", api_key="your-api-key")

# Generate a response
response = llm.generate("Explain quantum computing in simple terms.")
print(response)
```

## Using Different Providers

AbstractLLM supports multiple LLM providers. You can easily switch between them:

### OpenAI

```python
# Using environment variable for API key
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

llm = create_llm("openai", model="gpt-4")
response = llm.generate("What are the benefits of exercise?")
```

### Anthropic

```python
llm = create_llm("anthropic", 
                api_key="your-api-key",
                model="claude-3-opus-20240229")
response = llm.generate("Explain the concept of recursion.")
```

### Ollama

```python
# Using locally running Ollama
llm = create_llm("ollama", 
                base_url="http://localhost:11434",
                model="llama2")
response = llm.generate("How do solar panels work?")
```

### Hugging Face

```python
# Using a Hugging Face model
llm = create_llm("huggingface", 
                model="google/gemma-7b",
                load_in_8bit=True)
response = llm.generate("What are some renewable energy sources?")
```

## Configuring LLM Behavior

You can configure the LLM's behavior in several ways:

### Setting Configuration at Initialization

```python
llm = create_llm("openai", 
                api_key="your-api-key",
                model="gpt-4",
                temperature=0.3,
                system_prompt="You are a helpful scientific assistant.")
```

### Updating Configuration Later

```python
llm = create_llm("openai")

# Update configuration
llm.set_config({
    "temperature": 0.5,
    "max_tokens": 500,
    "system_prompt": "You are a friendly assistant."
})
```

### Per-Request Configuration

```python
# Base configuration
llm = create_llm("anthropic", temperature=0.7)

# Override for a specific request
creative_response = llm.generate(
    "Write a poem about the ocean",
    temperature=0.9
)

# Technical request with different parameters
technical_response = llm.generate(
    "Explain how HTTPS works",
    temperature=0.2,
    system_prompt="You are a technical expert. Be precise and comprehensive."
)
```

## Working with System Prompts

System prompts help shape the LLM's personality and behavior:

```python
# Set system prompt in configuration
llm = create_llm("openai", system_prompt="You are a helpful medical assistant. Always clarify that you're not a doctor.")

# Or provide it for a specific request
response = llm.generate(
    "What are the symptoms of the flu?",
    system_prompt="You are a helpful medical assistant. Always clarify that you're not a doctor."
)
```

## Checking Provider Capabilities

Different providers have different capabilities:

```python
llm = create_llm("openai")
capabilities = llm.get_capabilities()
print(capabilities)
# Example output: {'streaming': True, 'max_tokens': 4096, 'supports_system_prompt': True}

# Check if a specific capability is supported
if capabilities["supports_system_prompt"]:
    # Use system prompt
    pass
```

## Error Handling

It's important to handle potential errors:

```python
from abstractllm import create_llm

try:
    llm = create_llm("openai")
    response = llm.generate("Tell me about quantum physics")
except ImportError as e:
    print(f"Missing dependency: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Generation error: {e}")
```

## Logging

AbstractLLM includes built-in logging that you can configure:

```python
import logging
from abstractllm.utils.logging import setup_logging

# Set up logging with desired level
setup_logging(level=logging.DEBUG)

# Now all requests and responses will be logged
llm = create_llm("openai")
response = llm.generate("Hello, world!")
```

## Advanced: Using Multiple Providers

You can use multiple providers in the same application:

```python
# Create instances for different providers
openai_llm = create_llm("openai")
anthropic_llm = create_llm("anthropic")
ollama_llm = create_llm("ollama")

# Use each for different purposes
summary = openai_llm.generate("Summarize this article: " + article_text)
analysis = anthropic_llm.generate("Analyze the ethical implications of: " + topic)
quick_response = ollama_llm.generate("What's the capital of France?")
```

## Advanced: Fallback Chains

Implement fallback chains for reliability:

```python
def generate_with_fallback(prompt, providers=None):
    """Generate with fallback to alternative providers."""
    if providers is None:
        providers = [
            ("openai", {"model": "gpt-3.5-turbo"}),
            ("anthropic", {"model": "claude-instant-1"}),
            ("ollama", {"model": "llama2"})
        ]
    
    last_error = None
    for provider_name, config in providers:
        try:
            llm = create_llm(provider_name, **config)
            return llm.generate(prompt)
        except Exception as e:
            last_error = e
            continue
    
    # If all providers failed
    raise Exception(f"All providers failed. Last error: {last_error}")

# Example usage
try:
    response = generate_with_fallback("Explain how rainbows form")
    print(response)
except Exception as e:
    print(f"Error: {e}")
```

## Hugging Face Model Management

When using the Hugging Face provider, models are automatically cached to avoid repeated downloads:

```python
# Models are cached by default in ~/.cache/abstractllm/models
llm = create_llm("huggingface", model="google/gemma-7b")
```

### Custom Cache Location

You can specify a custom cache location:

```python
llm = create_llm("huggingface", 
                model="google/gemma-7b",
                cache_dir="/path/to/custom/cache")
```

### Listing Cached Models

You can view all cached models:

```python
from abstractllm.providers.huggingface import HuggingFaceProvider

models = HuggingFaceProvider.list_cached_models()
for model in models:
    print(f"{model['name']} - {model['size']} - Last used: {model['last_used']}")
```

### Managing the Cache

Clear specific models or the entire cache:

```python
# Clear a specific model
HuggingFaceProvider.clear_model_cache("google/gemma-7b")

# Clear the entire cache
HuggingFaceProvider.clear_model_cache()
```

This guide covers the basic and advanced usage patterns for AbstractLLM. By providing a consistent interface across different LLM providers, AbstractLLM makes it easy to experiment with different models and build robust LLM-powered applications.