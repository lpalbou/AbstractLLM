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
- 🖼️ **Vision Capabilities**: Support for multimodal models with image inputs
- 📊 **Capabilities Inspection**: Query models for their capabilities
- 📝 **Logging**: Built-in request and response logging
- 🔤 **Type-Safe Parameters**: Enum-based parameters for enhanced IDE support and error prevention

## Installation

### Setting up a Virtual Environment

You can use either conda or venv to create a virtual environment:

#### Using conda
```bash
# Create a new conda environment
conda create -n abstractllm python=3.8
# Activate the environment
conda activate abstractllm
```

#### Using venv
```bash
# Create a new virtual environment
python -m venv abstractllm-env
# Activate the environment (Linux/Mac)
source abstractllm-env/bin/activate
# Activate the environment (Windows)
.\abstractllm-env\Scripts\activate
```

### Installing the Package

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

## Type-Safe Parameters with Enums

AbstractLLM provides enums for type-safe parameter settings:

```python
from abstractllm import create_llm, ModelParameter, ModelCapability

# Create LLM with enum parameters
llm = create_llm("openai", 
                **{
                    ModelParameter.API_KEY: "your-api-key",
                    ModelParameter.MODEL: "gpt-4",
                    ModelParameter.TEMPERATURE: 0.7
                })

# Check capabilities with enums
capabilities = llm.get_capabilities()
if capabilities[ModelCapability.STREAMING]:
    # Use streaming...
    pass
```

## Supported Providers

### OpenAI

```python
from abstractllm import create_llm, ModelParameter

llm = create_llm("openai", 
                **{
                    ModelParameter.API_KEY: "your-api-key",
                    ModelParameter.MODEL: "gpt-4"
                })
```

### Anthropic

```python
from abstractllm import create_llm, ModelParameter

llm = create_llm("anthropic", 
                **{
                    ModelParameter.API_KEY: "your-api-key",
                    ModelParameter.MODEL: "claude-3-opus-20240229"
                })
```

### Ollama

```python
from abstractllm import create_llm, ModelParameter

llm = create_llm("ollama", 
                **{
                    ModelParameter.BASE_URL: "http://localhost:11434",
                    ModelParameter.MODEL: "llama2"
                })
```

### Hugging Face

```python
from abstractllm import create_llm, ModelParameter

llm = create_llm("huggingface", 
                **{
                    ModelParameter.MODEL: "google/gemma-7b",
                    ModelParameter.LOAD_IN_8BIT: True,
                    ModelParameter.DEVICE_MAP: "auto"
                })
```

## Configuration

You can configure the LLM's behavior in several ways:

```python
from abstractllm import create_llm, ModelParameter

# Using string keys (backwards compatible)
llm = create_llm("openai", temperature=0.7, system_prompt="You are a helpful assistant.")

# Using enum keys (type-safe)
llm = create_llm("openai", **{
    ModelParameter.TEMPERATURE: 0.5,
    ModelParameter.SYSTEM_PROMPT: "You are a helpful assistant."
})

# Update later with enums
llm.update_config({ModelParameter.TEMPERATURE: 0.5})

# Update with kwargs
llm.set_config(temperature=0.9)

# Per-request
response = llm.generate("Hello", temperature=0.9)
```

## System Prompts

System prompts help shape the model's personality and behavior:

```python
from abstractllm import create_llm, ModelParameter

# Using string keys
llm = create_llm("openai", system_prompt="You are a helpful scientific assistant.")

# Using enum keys
llm = create_llm("openai", **{
    ModelParameter.SYSTEM_PROMPT: "You are a helpful scientific assistant."
})

# Or for a specific request
response = llm.generate(
    "What is quantum entanglement?", 
    system_prompt="You are a physics professor explaining to a high school student."
)
```

## Vision Capabilities

AbstractLLM supports vision capabilities for models that can process images:

```python
from abstractllm import create_llm, ModelParameter, ModelCapability

# Create an LLM instance with a vision-capable model
llm = create_llm("openai", **{
    ModelParameter.MODEL: "gpt-4o",  # Vision-capable model
})

# Check if vision is supported
capabilities = llm.get_capabilities()
if capabilities.get(ModelCapability.VISION):
    # Use vision capabilities
    image_url = "https://example.com/image.jpg"
    response = llm.generate("What's in this image?", image=image_url)
    print(response)
    
    # You can also use local image files
    local_image = "/path/to/image.jpg"
    response = llm.generate("Describe this image", image=local_image)
    
    # Or multiple images
    images = ["https://example.com/image1.jpg", "/path/to/image2.jpg"]
    response = llm.generate("Compare these images", images=images)
```

Supported vision models include:
- OpenAI: `gpt-4-vision-preview`, `gpt-4-turbo`, `gpt-4o`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-3.5-sonnet`, `claude-3.5-haiku`
- Ollama: `llama3.2-vision`, `deepseek-janus-pro`

See the [Vision Capabilities Guide](docs/vision_guide.md) for more details.

## Capabilities

Check what capabilities a provider supports:

```python
from abstractllm import create_llm, ModelCapability

llm = create_llm("openai")
capabilities = llm.get_capabilities()

# Check using string keys
if capabilities["streaming"]:
    print("Streaming is supported!")
    
# Check using enum keys (type-safe)
if capabilities[ModelCapability.STREAMING]:
    print("Streaming is supported!")
    
if capabilities[ModelCapability.VISION]:
    print("Vision capabilities are supported!")
```

## Logging

AbstractLLM includes built-in logging with hierarchical configuration:

```python
import logging
from abstractllm.utils.logging import setup_logging

# Set up logging with desired level
setup_logging(level=logging.INFO)

# Set up logging with different levels for providers
setup_logging(level=logging.INFO, provider_level=logging.DEBUG)

# Now all requests and responses will be logged
llm = create_llm("openai")
response = llm.generate("Hello, world!")
```

The logging system provides:

- **INFO level**: Basic operation logging (queries being made, generation starting/completing)
- **DEBUG level**: Detailed information including parameters, prompts, URLs, and responses
- **Provider-specific loggers**: Each provider class uses its own logger (e.g., `abstractllm.providers.openai.OpenAIProvider`)
- **Security-conscious logging**: API keys are never logged, even at DEBUG level

## Testing

AbstractLLM includes a comprehensive test suite that tests all aspects of the library with real implementations (no mocks).

### Development Setup

For development and testing, it's recommended to install the package in development mode:

```bash
# Clone the repository
git clone https://github.com/lpalbou/abstractllm.git
cd abstractllm

# Install the package in development mode
pip install -e .

# Install test dependencies
pip install -r requirements-test.txt
```

This installs the package in "editable" mode, meaning changes to the source code will be immediately available without reinstalling.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run only tests for specific providers
pytest tests/ -m openai
pytest tests/ -m anthropic
pytest tests/ -m huggingface
pytest tests/ -m ollama
pytest tests/ -m vision

# Run specific test
python -m pytest tests/test_vision_captions.py::test_caption_quality -v --log-cli-level=INFO

# Run tests with coverage report
pytest tests/ --cov=abstractllm --cov-report=term
```

### Environment Variables for Testing

The test suite uses these environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `TEST_GPT4`: Set to "true" to enable GPT-4 tests
- `TEST_CLAUDE3`: Set to "true" to enable Claude 3 tests
- `TEST_VISION`: Set to "true" to enable vision capability tests
- `TEST_HUGGINGFACE`: Set to "true" to enable HuggingFace-specific tests
- `TEST_OLLAMA`: Set to "true" to enable Ollama-specific tests
- `TEST_HF_CACHE`: Set to "true" to enable HuggingFace cache management tests

To run the test script:

```bash
./run_tests.sh
```

## Advanced Usage

See the [Usage Guide](https://github.com/lpalbou/abstractllm/blob/main/docs/usage.md) for advanced usage patterns, including:

- Using multiple providers
- Implementing fallback chains
- Error handling
- Streaming responses
- Async generation
- And more

## Contributing

Contributions are welcome! 
Read more about how to contribute in the [CONTRIBUTING](CONTRIBUTING.md) file.
Please feel free to submit a [Pull Request](https://github.com/lpalbou/abstractllm/pulls).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.