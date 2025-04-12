# AbstractLLM

[![PyPI version](https://badge.fury.io/py/abstractllm.svg)](https://badge.fury.io/py/abstractllm)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, unified interface for interacting with multiple Large Language Model providers.

Version: 0.4.5

## Features

- 🔄 **Unified API**: Consistent interface for OpenAI, Anthropic, Ollama, and Hugging Face models
- 🔌 **Provider Agnostic**: Switch between providers with minimal code changes
- 🎛️ **Configurable**: Flexible configuration at initialization or per-request
- 📝 **System Prompts**: Standardized handling of system prompts across providers
- 🖼️ **Vision Capabilities**: Support for multimodal models with image inputs
- 📊 **Capabilities Inspection**: Query models for their capabilities
- 📝 **Logging**: Built-in request and response logging
- 🔤 **Type-Safe Parameters**: Enum-based parameters for enhanced IDE support and error prevention
- 🔄 **Provider Chains**: Create fallback chains and load balancing across multiple providers
- 💬 **Session Management**: Maintain conversation context when switching between providers
- 🛑 **Unified Error Handling**: Consistent error handling across all providers

## Command-Line Examples

### Text Generation
```bash
# Using OpenAI with logging
python query.py "what is AI ?" --provider openai --log-dir ./logs --log-level DEBUG --console-output

# Using Anthropic with custom log directory
python query.py "what is AI ?" --provider anthropic --log-dir /var/log/myapp/llm

# Using Ollama with debug logging
python query.py "what is AI ?" --provider ollama --log-level DEBUG
```

### Text File Analysis
```bash
# Using OpenAI
python query.py "describe the content of this file ?" -f tests/examples/test_data.csv --provider openai  

# Using Anthropic
python query.py "describe the content of this file ?" -f tests/examples/test_data.csv --provider anthropic

# Using Ollama
python query.py "describe the content of this file ?" -f tests/examples/test_data.csv --provider ollama  
```

### Image Analysis
```bash
# Using Anthropic with Claude 3
python query.py "describe this image with a set of keywords" -f tests/examples/mountain_path.jpg --provider anthropic --model claude-3-5-sonnet-20241022

# Using Ollama with LLaVA
python query.py "describe this image with a set of keywords" -f tests/examples/mountain_path.jpg --provider ollama --model llama3.2-vision:latest

# Using OpenAI with GPT-4 Vision
python query.py "describe this image with a set of keywords" -f tests/examples/mountain_path.jpg --provider openai  
```

### Logging Configuration

The command-line tool supports flexible logging configuration:

```bash
# Basic logging (to logs/ directory)
python query.py "Hello" --provider openai

# Custom log directory
python query.py "Hello" --provider openai --log-dir /path/to/logs

# Debug level logging
python query.py "Hello" --provider openai --log-level DEBUG

# Force console output with file logging
python query.py "Hello" --provider openai --console-output

# Full logging configuration
python query.py "Hello" --provider openai \
    --log-dir /var/log/myapp/llm \
    --log-level DEBUG \
    --console-output
```

The logging system provides:
- Request/response logging in JSON format
- Automatic log directory creation
- Log rotation support
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Optional console output alongside file logging
- Secure handling of sensitive data (API keys never logged)

Log files are organized as follows:
- `abstractllm_YYYYMMDD_HHMMSS.log`: Main log file with all events
- `{provider}_request_YYYYMMDD_HHMMSS.json`: Individual request details
- `{provider}_response_YYYYMMDD_HHMMSS.json`: Individual response details

## Important Notes

### HuggingFace Support (Work in Progress)
The HuggingFace provider is currently under active development. While basic functionality is implemented, you may encounter some limitations and issues:
- Device handling (CPU/CUDA/MPS) is being refined
- Some model architectures may not work as expected
- Vision model support is experimental
- Memory management is being optimized

We recommend using the OpenAI, Anthropic, or Ollama providers for production use while HuggingFace support is being finalized.

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

# Using a HuggingFace model directly
llm = create_llm("huggingface", 
                **{
                    ModelParameter.MODEL: "ibm-granite/granite-3.2-2b-instruct",
                    # Device will be automatically detected (CUDA, MPS, or CPU)
                    ModelParameter.DEVICE: "auto",
                    ModelParameter.TEMPERATURE: 0.7
                })

# Using a pre-quantized GGUF model from HuggingFace
llm = create_llm("huggingface", 
                **{
                    ModelParameter.MODEL: "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_L.gguf",
                    # Device will be automatically detected for GGUF models
                    ModelParameter.DEVICE: "auto"
                })

# Using on-the-fly quantization
llm = create_llm("huggingface", 
                **{
                    ModelParameter.MODEL: "microsoft/Phi-4-mini-instruct",
                    ModelParameter.LOAD_IN_4BIT: True,  # Enable 4-bit quantization
                    ModelParameter.DEVICE_MAP: "auto"  # Automatic device mapping
                })
```

The HuggingFace provider supports:
- Automatic device detection (CUDA for NVIDIA GPUs, MPS for Apple Silicon, CPU fallback)
- Direct loading of HuggingFace models
- Loading pre-quantized GGUF models
- On-the-fly quantization (4-bit and 8-bit)
- Automatic device mapping for large models

Command-line examples:
```bash
# Using a HuggingFace model directly
python query.py "what is AI ?" --provider huggingface --model ibm-granite/granite-3.2-2b-instruct

# Using a pre-quantized GGUF model
python query.py "what is AI ?" --provider huggingface --model https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_L.gguf

# Using a higher quality quantized model
python query.py "what is AI ?" --provider huggingface --model https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q6_K_L.gguf
```

#### Important Note About HuggingFace Models

Many models on HuggingFace Hub require accepting a license before use. This is especially true for popular models like:
- Meta's Llama models
- Google's Gemma models
- Mistral models
- And many others

To use these models:
   - Visit the model's page on HuggingFace Hub (e.g., https://huggingface.co/meta-llama/Llama-2-7b)
   - Sign in with your HuggingFace account
   - Click "Agree and access repository" to accept the license
   - Some models (like Llama) may require additional approval from the model owners
   - Wait for approval if required
   - After accepting the license and receiving any necessary approvals, you can use the model
   - The script will guide you through this process if you haven't completed these steps


## Configuration

You can configure the LLM's behavior in several ways:

```python
from abstractllm import create_llm, ModelParameter

# Using string keys (backwards compatible)
llm = create_llm("openai", temperature=0.7, system_prompt="You are a helpful assistant.")

# Using enum keys (type-safe)
llm = create_llm("openai", **{
    ModelParameter.TEMPERATURE: 0.5,
    ModelParameter.SYSTEM_PROMPT: "You are a helpful scientific assistant."
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

## Provider Chains

AbstractLLM supports creating chains of providers with fallback capabilities to ensure robust operation:

```python
from abstractllm.chain import create_fallback_chain, create_capability_chain, create_load_balanced_chain

# Create a fallback chain that tries providers in sequence
chain = create_fallback_chain(
    providers=["openai", "anthropic", "ollama"],
    max_retries=2
)

# Generate with automatic fallback if a provider fails
response = chain.generate("Explain quantum computing in simple terms.")

# Create a chain that selects providers based on capabilities
vision_chain = create_capability_chain(
    required_capabilities=[ModelCapability.VISION],
    preferred_providers=["openai", "anthropic"]
)

# Generate with a provider that supports vision
image_url = "https://example.com/image.jpg"
response = vision_chain.generate("What's in this image?", image=image_url)

# Create a load-balanced chain for distributing requests
balanced_chain = create_load_balanced_chain(
    providers=["openai", "anthropic", "ollama"]
)

# Requests will be distributed across providers
response1 = balanced_chain.generate("What is AI?")
response2 = balanced_chain.generate("What is machine learning?")
```

## Session Management

AbstractLLM includes session management for maintaining conversation context even when switching providers:

```python
from abstractllm.session import Session, SessionManager

# Create a session with a system prompt
session = Session(
    system_prompt="You are a helpful assistant specializing in physics.",
    provider="openai"
)

# Send a message using the default provider
response = session.send("What is the theory of relativity?")
print(f"OpenAI: {response}")

# Switch providers for the next message while maintaining context
response = session.send(
    "Can you explain it in simpler terms?",
    provider="anthropic"
)
print(f"Anthropic: {response}")

# Save the session for later
session.save("physics_session.json")

# Later, load the session and continue
loaded_session = Session.load("physics_session.json")
response = loaded_session.send("How is this related to quantum mechanics?")

# Managing multiple sessions
manager = SessionManager(sessions_dir="my_sessions")
physics_session = manager.create_session(
    system_prompt="You are a physics professor.",
    provider="openai"
)
history_session = manager.create_session(
    system_prompt="You are a historian.",
    provider="anthropic"
)

# Use different sessions for different topics
physics_response = physics_session.send("What is quantum entanglement?")
history_response = history_session.send("Tell me about ancient Egypt.")

# Save all sessions
manager.save_all()
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

## Error Handling

AbstractLLM provides a unified error handling system across all providers:

```python
from abstractllm import create_llm
from abstractllm.exceptions import (
    AbstractLLMError,
    AuthenticationError,
    QuotaExceededError,
    ContextWindowExceededError
)

try:
    llm = create_llm("openai", api_key="invalid-key")
    response = llm.generate("Hello")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Try with a different key or provider
except QuotaExceededError as e:
    print(f"Quota exceeded: {e}")
    # Implement rate limiting or fallback to another provider
except ContextWindowExceededError as e:
    print(f"Context window exceeded: {e}")
    # Implement chunking or summarization
except AbstractLLMError as e:
    print(f"Generic error: {e}")
    # Handle all other AbstractLLM errors
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