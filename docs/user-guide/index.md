# AbstractLLM User Guide

This user guide provides detailed information about AbstractLLM's features and how to use them effectively in your applications.

## Core Concepts

### The Provider System

AbstractLLM is built around a provider-based architecture. Each provider (OpenAI, Anthropic, Ollama, HuggingFace) implements a common interface but handles the details of communicating with their respective APIs.

```python
from abstractllm import create_llm

# Create a provider instance
llm = create_llm("openai", model="gpt-4")

# Use the provider
response = llm.generate("Hello, world!")
```

### Configuration Management

AbstractLLM provides a flexible configuration system that allows you to set parameters at different levels:

```python
# Global configuration at creation time
llm = create_llm(
    "anthropic", 
    model="claude-3-opus-20240229",
    temperature=0.7,
    max_tokens=1000
)

# Per-request configuration
response = llm.generate(
    prompt="Generate a short poem.",
    temperature=0.9,  # Override for this request only
    max_tokens=500    # Override for this request only
)
```

### Capability Detection

You can check what capabilities a model supports before using them:

```python
from abstractllm import create_llm, ModelCapability

llm = create_llm("openai", model="gpt-4")
capabilities = llm.get_capabilities()

if capabilities.get(ModelCapability.VISION):
    print("Vision is supported!")
if capabilities.get(ModelCapability.STREAMING):
    print("Streaming is supported!")
if capabilities.get(ModelCapability.FUNCTION_CALLING):
    print("Function calling is supported!")
```

## Features

### [Basic Text Generation](basic-generation.md)

Learn how to generate text responses using various models and providers.

### [Working with Sessions](sessions.md)

Understand how to use sessions for maintaining conversation context and state.

### [Vision Capabilities](vision.md)

Discover how to work with images and vision-capable models.

### [Tool Calls](tools.md)

Learn how to integrate external tools with LLMs for more powerful applications.

### [Streaming Responses](streaming.md)

Understand how to stream responses in real-time for a better user experience.

### [Asynchronous Generation](async.md)

Learn how to use asynchronous generation for improved performance.

### [Error Handling](error-handling.md)

Discover how to handle errors properly in your AbstractLLM applications.

### [Provider Interchangeability](interchangeability.md)

Learn how to switch between providers and create fallback chains.

### [Logging and Debugging](logging.md)

Understand how to configure logging for debugging and auditing.

## Advanced Topics

### [Provider-Specific Features](provider-specific.md)

Learn about provider-specific features and how to use them.

### [Custom Providers](custom-providers.md)

Discover how to create your own provider implementations.

### [Security Best Practices](security.md)

Understand how to implement security best practices in your applications.

### [Performance Optimization](performance.md)

Learn how to optimize your AbstractLLM applications for better performance.

### [Multi-Modal Content](multimodal.md)

Understand how to work with multi-modal content beyond just text and images.

## Examples

Check out our [examples directory](../examples/index.md) for practical, ready-to-use examples of AbstractLLM in action. 