# AbstractLLM

[![PyPI version](https://badge.fury.io/py/abstractllm.svg)](https://badge.fury.io/py/abstractllm)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, unified interface for interacting with multiple Large Language Model providers.

[THIS IS A WORK IN PROGRESS, STAY TUNED !]

## Features

- 🔄 **Unified API**: Consistent interface for OpenAI, Anthropic, Ollama, and Hugging Face models
- 🔌 **Provider Agnostic**: Switch between providers with minimal code changes
- 🎛️ **Configurable**: Flexible configuration at initialization or per-request
- 📝 **System Prompts**: Standardized handling of system prompts across providers
- 📊 **Capabilities Inspection**: Query models for their capabilities
- 📝 **Logging**: Built-in request and response logging

## Installation

```bash
# Basic installation
pip install abstractllm

# With provider-specific dependencies
pip install abstractllm[openai]
pip install abstractllm[anthropic]
pip install abstractllm[huggingface]

# All dependencies
pip install abstractllm[all]
```

## Quick Start

```python
from abstractllm import create_llm

# Create an LLM instance
llm = create_llm("openai", api_key="your-api-key")

# Generate a response
response = llm.generate("Explain quantum computing in simple terms.")
print(response)
```

## Supported Providers

### OpenAI

```python
llm = create_llm("openai", 
                api_key="your-api-key",
                model="gpt-4")
```

### Anthropic

```python
llm = create_llm("anthropic", 
                api_key="your-api-key",
                model="claude-3-opus-20240229")
```

### Ollama

```python
llm = create_llm("ollama", 
                base_url="http://localhost:11434",
                model="llama2")
```

### Hugging Face

```python
llm = create_llm("huggingface", 
                model="google/gemma-7b")
```

## Configuration

You can configure the LLM's behavior in several ways:

```python
# At initialization
llm = create_llm("openai", temperature=0.7, system_prompt="You are a helpful assistant.")

# Update later
llm.set_config({"temperature": 0.5})

# Per-request
response = llm.generate("Hello", temperature=0.9)
```

## System Prompts

System prompts help shape the model's personality and behavior:

```python
llm = create_llm("openai", system_prompt="You are a helpful scientific assistant.")

# Or for a specific request
response = llm.generate("What is quantum entanglement?", 
                     system_prompt="You are a physics professor explaining to a high school student.")
```

## Capabilities

Check what capabilities a provider supports:

```python
capabilities = llm.get_capabilities()
print(capabilities)
# Example: {'streaming': True, 'max_tokens': 4096, 'supports_system_prompt': True}
```

## Logging

AbstractLLM includes built-in logging:

```python
import logging
from abstractllm.utils.logging import setup_logging

# Set up logging with desired level
setup_logging(level=logging.DEBUG)
```

## Advanced Usage

See the [Usage Guide](https://github.com/lpalbou/abstractllm/blob/main/docs/usage.md) for advanced usage patterns, including:

- Using multiple providers
- Implementing fallback chains
- Error handling
- And more

## Contributing

Contributions are welcome! 
Read more about how to contribute in the [CONTRIBUTING](CONTRIBUTING.md) file.
Please feel free to submit a [Pull Request](https://github.com/lpalbou/abstractllm/pulls).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.