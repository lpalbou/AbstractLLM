# AbstractLLM Examples

This directory contains example scripts demonstrating how to use AbstractLLM in various scenarios.

## Available Examples

| Example | Description |
| ------- | ----------- |
| [basic_usage.py](basic_usage.py) | Basic usage with different providers (OpenAI, Anthropic, Ollama) |
| [async_usage.py](async_usage.py) | Asynchronous operations and parallel generation |

## Running the Examples

Before running the examples, make sure:

1. You have installed AbstractLLM (either from PyPI or in development mode)
2. You have set up any required API keys as environment variables:
   - `OPENAI_API_KEY` for OpenAI examples
   - `ANTHROPIC_API_KEY` for Anthropic examples

### Basic Usage

```bash
# Install required dependencies
pip install abstractllm[openai,anthropic]

# Run basic usage example
python examples/basic_usage.py
```

### Async Usage

```bash
# Install required dependencies
pip install abstractllm[openai]

# Run async usage example
python examples/async_usage.py
```

## Setting API Keys

For convenience during development, you can set API keys in a `.env` file:

```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
echo "ANTHROPIC_API_KEY=your-api-key-here" >> .env

# Install python-dotenv
pip install python-dotenv

# Then, at the beginning of your scripts:
from dotenv import load_dotenv
load_dotenv()
```

## Using Ollama

For Ollama examples, you need to:

1. [Install Ollama](https://ollama.ai/download)
2. Run the Ollama server: `ollama serve`
3. Pull at least one model: `ollama pull llama2` 