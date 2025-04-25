# Advanced Features

This section covers advanced features and techniques for using AbstractLLM in more complex scenarios.

## Table of Contents

- [Provider-Specific Features](#provider-specific-features)
- [Custom Providers](#custom-providers) 
- [Security Best Practices](#security-best-practices)
- [Performance Optimization](#performance-optimization)
- [Multi-Modal Content](#multi-modal-content)

## Provider-Specific Features

Each LLM provider offers unique features that go beyond AbstractLLM's unified interface. This section explains how to access and utilize provider-specific features when needed.

```python
from abstractllm import create_llm

# OpenAI: Using function_call parameter to force a specific tool call
llm = create_llm("openai", model="gpt-4")
response = llm.generate(
    "What's the weather in Boston?",
    tools=[get_weather],
    # OpenAI-specific parameter
    function_call={"name": "get_weather"}
)

# Anthropic: Using metadata for additional context
llm = create_llm("anthropic", model="claude-3-opus-20240229")
response = llm.generate(
    "Summarize the conversation.",
    # Anthropic-specific parameter
    metadata={"conversation_id": "abc123"}
)

# HuggingFace: Using model-specific generation parameters
llm = create_llm("huggingface", model="mistralai/Mistral-7B-Instruct-v0.1")
response = llm.generate(
    "Write a poem about artificial intelligence.",
    # HuggingFace-specific parameters
    do_sample=True,
    top_k=50,
    repetition_penalty=1.2
)
```

For more details, see each provider's specific documentation:

- [OpenAI Provider](../providers/openai.md)
- [Anthropic Provider](../providers/anthropic.md)
- [HuggingFace Provider](../providers/huggingface.md)
- [Ollama Provider](../providers/ollama.md)

## Custom Providers

You can extend AbstractLLM with custom provider implementations to support additional LLM services or specialized integrations.

To create a custom provider:

1. Subclass the `AbstractLLMInterface`
2. Implement the required methods
3. Register your provider

Here's a simple example:

```python
from abstractllm.interface import AbstractLLMInterface
from abstractllm.configuration import ConfigurationManager
from abstractllm.factory import register_provider
from typing import Dict, Any, Optional, Union, Generator

class CustomProvider(AbstractLLMInterface):
    """Custom provider implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the custom provider."""
        super().__init__(config)
        # Your provider-specific initialization
        
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[list] = None,
                stream: bool = False,
                tools: Optional[list] = None,
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response."""
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Your provider implementation
        # ...
        
        return "Your custom response"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return capabilities of the custom provider."""
        return {
            "streaming": True,
            "max_tokens": 2048,
            "system_prompt": True,
            "async": False,
            "function_calling": False,
            "tool_use": False,
            "vision": False
        }

# Register your provider
register_provider("custom", CustomProvider)

# Use your provider
from abstractllm import create_llm
llm = create_llm("custom", param1="value1")
```

For a more complete example, see the [Custom Provider Implementation Guide](custom-providers.md).

## Security Best Practices

When working with AbstractLLM, especially in production environments, follow these security best practices:

1. **API Key Management**
   - Use environment variables or secure credential storage
   - Rotate API keys regularly
   - Use different API keys for development and production

2. **Tool Call Security**
   - Apply strict validation to tool inputs
   - Implement timeouts and resource limits
   - Use the principle of least privilege

3. **Content Filtering**
   - Implement content filtering for user inputs
   - Process responses for sensitive information
   - Apply proper sanitization

4. **Authentication and Authorization**
   - Secure your application endpoints
   - Implement rate limiting
   - Use HTTPS for all API communications

Example of secure tool implementation:

```python
from abstractllm import create_llm, ToolDefinition
from abstractllm.session import Session
import os
import re

def secure_read_file(file_path: str) -> str:
    """Securely read a file with proper validation."""
    # Validate file path
    if not is_safe_path(file_path):
        return "Error: Invalid file path."
    
    # Apply timeout and error handling
    try:
        with open(file_path, 'r', timeout=5) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def is_safe_path(file_path: str) -> bool:
    """Check if a file path is safe."""
    # Normalize path
    abs_path = os.path.abspath(os.path.normpath(file_path))
    
    # Check for allowed directories
    allowed_dirs = ["/safe/path", "/another/safe/path"]
    for allowed_dir in allowed_dirs:
        if abs_path.startswith(allowed_dir):
            return True
    
    return False

# Create provider and session
llm = create_llm("openai", model="gpt-4")
session = Session(provider=llm, tools=[secure_read_file])
```

For more security best practices, see the [Security Guide](security.md).

## Performance Optimization

Optimize AbstractLLM for performance in production environments:

1. **Model Selection**
   - Choose appropriate models for the task
   - Consider smaller models for simple tasks
   - Balance quality vs. performance

2. **Caching**
   - Implement response caching for common queries
   - Cache embeddings for vector search
   - Consider using Redis or similar for distributed caching

3. **Asynchronous Processing**
   - Use `generate_async()` for concurrent requests
   - Implement proper concurrency control
   - Consider batch processing when appropriate

4. **Resource Management**
   - Monitor token usage
   - Implement graceful degradation
   - Consider serverless architectures for scaling

Example of performance optimization:

```python
import asyncio
import hashlib
import json
import redis
from abstractllm import create_llm

# Create Redis client for caching
redis_client = redis.Redis(host="localhost", port=6379, db=0)

async def cached_generate(llm, prompt, **kwargs):
    """Generate response with caching."""
    # Create cache key from prompt and kwargs
    cache_key = hashlib.sha256(
        (prompt + json.dumps(kwargs, sort_keys=True)).encode()
    ).hexdigest()
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return cached.decode("utf-8")
    
    # Generate response
    response = await llm.generate_async(prompt, **kwargs)
    
    # Cache response (with expiration)
    redis_client.setex(cache_key, 3600, response)  # 1 hour expiration
    
    return response

async def main():
    # Create LLM
    llm = create_llm("openai", model="gpt-3.5-turbo")
    
    # Process multiple requests concurrently with caching
    tasks = [
        cached_generate(llm, "What is the capital of France?"),
        cached_generate(llm, "What is the capital of Germany?"),
        cached_generate(llm, "What is the capital of Italy?")
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

# Run the async function
asyncio.run(main())
```

For more performance optimization techniques, see the [Performance Guide](performance.md).

## Multi-Modal Content

Work with multiple content types (text, images, audio) in your AbstractLLM applications:

1. **Mixed Media Inputs**
   - Combine text and images in prompts
   - Process multiple images in a single request
   - Work with different image formats and sources

2. **Advanced Media Processing**
   - Control image detail levels
   - Optimize media for token efficiency
   - Process media in different formats

Example of multi-modal processing:

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput
from PIL import Image

# Create provider with vision capabilities
llm = create_llm("openai", model="gpt-4o")

# Load and process images
image1 = ImageInput.from_file("image1.jpg")
image2 = ImageInput.from_file("image2.jpg")

# Modify image size to optimize token usage
image1.resize(max_width=800, max_height=600)

# Generate with multiple images
response = llm.generate(
    "Compare these two images and tell me what's different.",
    media=[image1, image2]
)
print(response)
```

For more details on working with multi-modal content, see the [Multi-Modal Guide](multimodal.md). 