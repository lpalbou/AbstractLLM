# AbstractLLM Documentation

**Version: 0.5.3**

Welcome to AbstractLLM, a lightweight Python library that provides a unified interface for interacting with multiple Large Language Model (LLM) providers. This documentation will help you get started with AbstractLLM and understand its capabilities, architecture, and usage patterns.

## 🚨 Important Note

AbstractLLM is currently a **Work In Progress** and is **not yet safe to use except for testing**. The API may change significantly before the first stable release.

## Key Features

- **Unified API**: Consistent interface for OpenAI, Anthropic, Ollama, and Hugging Face models
- **Provider Agnostic**: Switch between providers with minimal code changes
- **Configurable**: Flexible configuration at initialization or per-request
- **System Prompts**: Standardized handling of system prompts across providers
- **Vision Capabilities**: Support for multimodal models with image inputs
- **Capabilities Inspection**: Query models for their capabilities
- **Logging**: Built-in request and response logging
- **Type-Safe Parameters**: Enum-based parameters for enhanced IDE support
- **Provider Chains**: Create fallback chains and load balancing across providers
- **Session Management**: Maintain conversation context when switching providers
- **Tool Call Support**: Unified interface for tool/function calling capabilities
- **Unified Error Handling**: Consistent error handling across all providers

## Documentation Sections

- **[Getting Started](getting-started/index.md)**: Installation and basic usage
- **[User Guide](user-guide/index.md)**: Detailed guidance for using AbstractLLM
- **[Architecture](architecture/index.md)**: Understanding AbstractLLM's design
- **[API Reference](api-reference/index.md)**: Detailed API documentation
- **[Provider Guide](providers/index.md)**: Provider-specific documentation
- **[Advanced Features](advanced-features/index.md)**: Advanced usage patterns
- **[Examples](examples/index.md)**: Code examples for common use cases
- **[Contributing](contributing/index.md)**: Guide for contributors

## Quick Installation

```bash
# Basic installation (core functionality only)
pip install "abstractllm

# Provider-specific installations"
pip install "abstractllm[openai]"       # OpenAI API
pip install "abstractllm[anthropic]"    # Anthropic/Claude API
pip install "abstractllm[huggingface]"  # HuggingFace models (includes torch)
pip install "abstractllm[ollama]"       # Ollama API
pip install "abstractllm[tools]"        # Tool calling functionality

# Multiple providers
pip install "abstractllm[openai,anthropic]"

# All dependencies
pip install "abstractllm[all]"
```

## Quick Example

```python
from abstractllm import create_llm

# Create an LLM instance
llm = create_llm("openai", model="gpt-4")

# Generate a response
response = llm.generate("Hello, world!")
print(response)

# Switch to a different provider
anthropic_llm = create_llm("anthropic", model="claude-3-opus-20240229")
response = anthropic_llm.generate("Tell me about yourself.")
print(response)
```

## Status

AbstractLLM is in active development. See the [status report](../reports/status-2023-11-07.md) for the current state of the project.

## License

[MIT License](https://github.com/lpalbou/abstractllm/blob/main/LICENSE) 