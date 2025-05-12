# AbstractLLM

[![PyPI version](https://badge.fury.io/py/abstractllm.svg)](https://badge.fury.io/py/abstractllm)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, unified interface for interacting with multiple Large Language Model providers.

Version: 0.5.3

**IMPORTANT**: This is a Work In Progress. Things evolve rapidly. The library is not yet safe to use except for testing.

## Table of Contents

- [Features](#features)
- [Documentation](#documentation)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Example Implementations](#example-implementations)
- [Command-Line Examples](#command-line-examples)
- [Provider Support](#provider-support)
- [Contributing](#contributing)
- [License](#license)

## Features

- 🔄 **Unified API**: Consistent interface for OpenAI, Anthropic, Ollama, Hugging Face, and MLX models
- 🔌 **Provider Agnostic**: Switch between providers with minimal code changes
- 🎛️ **Configurable**: Flexible configuration at initialization or per-request
- 📝 **System Prompts**: Standardized handling of system prompts across providers
- 🖼️ **Vision Capabilities**: Support for multimodal models with image inputs
- 📊 **Capabilities Inspection**: Query models for their capabilities
- 📝 **Logging**: Built-in request and response logging
- 🔤 **Type-Safe Parameters**: Enum-based parameters for enhanced IDE support and error prevention
- 🔄 **Provider Chains**: Create fallback chains and load balancing across multiple providers
- 💬 **Session Management**: Maintain conversation context when switching between providers
- 🛠️ **Tool Calling**: Unified interface for function/tool calling capabilities across providers
- 🛑 **Unified Error Handling**: Consistent error handling across all providers
- 🍎 **Apple Silicon Optimization**: Support for MLX models on Apple Silicon devices

## Documentation

AbstractLLM now has comprehensive documentation covering all aspects of the library:

- **[Getting Started](docs/getting-started/index.md)**: Installation and basic usage
- **[User Guide](docs/user-guide/index.md)**:
  - [Basic Text Generation](docs/user-guide/basic-generation.md)
  - [Working with Sessions](docs/user-guide/sessions.md)
  - [Tool Calling](docs/user-guide/tools.md)
  - [Vision Capabilities](docs/user-guide/vision.md)
  - [Streaming Responses](docs/user-guide/streaming.md)
  - [Asynchronous Generation](docs/user-guide/async.md)
  - [Error Handling](docs/user-guide/error-handling.md)
  - [Provider Interchangeability](docs/user-guide/interchangeability.md)
  - [Logging and Debugging](docs/user-guide/logging.md)
- **[Provider Guide](docs/providers/index.md)**:
  - [OpenAI](docs/providers/openai.md)
  - [Anthropic](docs/providers/anthropic.md)
  - [Ollama](docs/providers/ollama.md)
  - [HuggingFace](docs/providers/huggingface.md)
- **[Advanced Features](docs/advanced-features/index.md)**:
  - [Provider-Specific Features](docs/advanced-features/provider-specific.md)
  - [Custom Providers](docs/advanced-features/custom-providers.md)
  - [Security Best Practices](docs/advanced-features/security.md)
  - [Performance Optimization](docs/advanced-features/performance.md)
  - [Multi-Modal Content](docs/advanced-features/multimodal.md)
  - [Tool Interfaces](docs/advanced-features/tools.md)
- **[Architecture](docs/architecture/index.md)**:
  - [Provider System](docs/architecture/providers.md)
  - [Media System](docs/architecture/media.md)
  - [Tool System](docs/architecture/tools.md)
  - [Configuration](docs/architecture/configuration.md)
  - [Error Handling](docs/architecture/error-handling.md)
- **[API Reference](docs/api-reference/index.md)**: Detailed API documentation
- **[Examples](docs/examples/index.md)**: Code examples for common use cases
- **[Contributing](docs/contributing/index.md)**: Guide for contributors

## Quick Start

```python
from abstractllm import create_llm

# Create an LLM instance
llm = create_llm("openai", api_key="your-api-key")

# Generate a response
response = llm.generate("Explain quantum computing in simple terms.")
print(response)

# Switch to a different provider
anthropic_llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")
response = anthropic_llm.generate("Tell me about yourself.")
print(response)
```

### Basic Tool Calling

```python
from abstractllm import create_llm
from abstractllm.session import Session

# Define your tool function
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Simulated weather data
    return f"The weather in {location} is currently sunny and 72°F."

# Create a provider and session
provider = create_llm("openai", model="gpt-4")
session = Session(
    system_prompt="You are a helpful assistant that can check the weather.",
    provider=provider,
    tools=[get_weather]  # Tool function is automatically registered
)

# Generate with tool support
response = session.generate_with_tools("What's the weather like in San Francisco?")
print(response.content)
```

For more detailed examples and advanced features, please see the [documentation](#documentation).

## Installation

```bash
# Basic installation (core functionality only)
pip install abstractllm

# Provider-specific installations
pip install "abstractllm[openai]"       # OpenAI API
pip install "abstractllm[anthropic]"    # Anthropic/Claude API
pip install "abstractllm[huggingface]"  # HuggingFace models (includes torch)
pip install "abstractllm[ollama]"       # Ollama API
pip install "abstractllm[mlx]"          # MLX support for Apple Silicon
pip install "abstractllm[tools]"        # Tool calling functionality

# Multiple providers
pip install "abstractllm[openai,anthropic]"

# All dependencies (excluding MLX which is platform-specific)
pip install "abstractllm[all]"
```

### Important: Provider Dependencies

Each provider requires specific dependencies to function:

- **OpenAI**: Requires the `openai` package
- **Anthropic**: Requires the `anthropic` package
- **HuggingFace**: Requires `torch`, `transformers`, and `huggingface-hub` 
- **Ollama**: Requires `requests` for sync and `aiohttp` for async operations
- **MLX**: Requires `mlx` and `mlx-lm` (Apple Silicon only)
- **Tool Calling**: Requires `docstring-parser`, `jsonschema`, and `pydantic`

If you try to use a provider without its dependencies, you'll get a clear error message telling you which package to install.

### Platform-Specific Support

- **MLX Provider**: Only available on macOS with Apple Silicon (M1/M2/M3 chips)
  ```bash
  # Install MLX support on Apple Silicon
  pip install "abstractllm[mlx]"
  ```

### Recommended Installation

For most users, we recommend installing at least one provider along with the base package:

```bash
# For just OpenAI support
pip install "abstractllm[openai]"

# For OpenAI and tool calling support
pip install "abstractllm[openai,tools]"

# For Apple Silicon users wanting local inference
pip install "abstractllm[mlx]"

# For all providers and tools (most comprehensive)
pip install "abstractllm[all]"
```

## Example Implementations

AbstractLLM includes two example implementations that demonstrate how to use the library:

### query.py - Simple Command-Line Interface

A straightforward example showing how to use AbstractLLM for basic queries:

```bash
# Basic text generation
python query.py "What is the capital of France?" --provider anthropic --model claude-3-5-haiku-20241022

# Processing a text file
python query.py "Summarize this text" -f tests/examples/test_file2.txt --provider anthropic

# Analyzing an image
python query.py "Describe this image" -f tests/examples/mountain_path.jpg --provider openai --model gpt-4o
```

### alma.py - Tool-Enhanced Agent

ALMA (Abstract Language Model Agent) demonstrates how to build a tool-enabled agent using AbstractLLM's tool calling capabilities:

```bash
# Basic usage
python alma.py --query "Tell me about the current time" --provider anthropic

# File reading and command execution with streaming output
python alma.py --query "Read the file at tests/examples/test_file2.txt and tell me what it's about" --stream

# Interactive mode with detailed logging
python alma.py --verbose
```

ALMA supports powerful tools:
- File reading with `read_file`
- Command execution with `execute_command`

## Command-Line Examples

### Text Generation
```bash
# Using OpenAI with logging
python query.py "what is AI ?" --provider openai --log-dir ./logs --log-level DEBUG --console-output

# Using Anthropic with custom log directory
python query.py "what is AI ?" --provider anthropic --log-dir /var/log/myapp/llm

# Using Ollama with debug logging
python query.py "what is AI ?" --provider ollama --log-level DEBUG

# Using HuggingFace with GGUF model
python query.py "what is AI ?" --provider huggingface --model https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_L.gguf
```

### Text File Analysis
```bash
# Using OpenAI
python query.py "describe the content of this file ?" -f tests/examples/test_data.csv --provider openai  

# Using Anthropic
python query.py "describe the content of this file ?" -f tests/examples/test_data.csv --provider anthropic
```

### Image Analysis
```bash
# Using Anthropic with Claude 3
python query.py "describe this image with a set of keywords" -f tests/examples/mountain_path.jpg --provider anthropic --model claude-3-5-sonnet-20241022

# Using Ollama with LLaVA
python query.py "describe this image with a set of keywords" -f tests/examples/mountain_path.jpg --provider ollama --model llama3.2-vision:latest

# Using OpenAI with GPT-4 Vision
python query.py "describe this image with a set of keywords" -f tests/examples/mountain_path.jpg --provider openai --model gpt-4o
```

## Provider Support

AbstractLLM supports multiple LLM providers, each with its own unique features and capabilities:

### OpenAI
```python
llm = create_llm("openai", model="gpt-4", api_key="your-api-key")
```

Key features:
- GPT-4 and GPT-3.5 models
- Vision capabilities with GPT-4o
- Function calling/tool use
- Token-by-token streaming
- JSON mode

### Anthropic
```python
llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022", api_key="your-api-key")
```

Key features:
- Claude 3 and Claude 3.5 models
- Very large context windows (up to 100K tokens)
- Vision capabilities
- Tool use with Claude 3
- Strong reasoning and instruction following

### Ollama
```python
llm = create_llm("ollama", model="llama3", api_base="http://localhost:11434")
```

Key features:
- Local model deployment
- No data sharing with external services
- Support for various open-source models (Llama, Mistral, etc.)
- Custom model support

### HuggingFace
```python
llm = create_llm("huggingface", model="ibm-granite/granite-3.2-2b-instruct")
```

Key features:
- Support for thousands of open-source models
- Local model deployment
- Integration with HuggingFace Hub
- Support for GGUF quantized models
- Custom model fine-tuning

For detailed information about each provider, see the [Provider Guide](docs/providers/index.md).

## Contributing

Contributions are welcome! Read more about how to contribute in the [Contributing Guide](docs/contributing/index.md) or the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Please feel free to submit a [Pull Request](https://github.com/lpalbou/abstractllm/pulls).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.