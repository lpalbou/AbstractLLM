# Provider-Specific Features

While AbstractLLM provides a unified interface for interacting with various LLM providers, each provider offers unique features and capabilities that may not be exposed through the standard interface. This guide explains how to access and utilize these provider-specific features when needed.

## Understanding Provider-Specific Features

Provider-specific features are parameters, functionalities, or capabilities that:

1. Are unique to a particular provider
2. May not have a standard equivalent in other providers
3. Are passed through to the underlying API when using that provider

AbstractLLM's design allows you to pass provider-specific parameters directly to the underlying API while maintaining a consistent interface for common operations.

## Using Provider-Specific Parameters

To use provider-specific parameters, simply include them as keyword arguments when calling the `generate` method:

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("openai", model="gpt-4")

# Use standard parameters and provider-specific parameters
response = llm.generate(
    "Generate a creative story.",
    # Standard AbstractLLM parameters
    max_tokens=500,
    temperature=0.7,
    # OpenAI-specific parameter
    frequency_penalty=0.5
)
```

AbstractLLM will:
1. Use the standard parameters it recognizes
2. Pass through any provider-specific parameters to the underlying API

## Provider-Specific Features by Provider

### OpenAI

OpenAI offers several unique parameters and features:

#### Function Calling Control

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("openai", model="gpt-4")

# Define tools
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny."

def get_time(timezone: str) -> str:
    """Get the current time in a specific timezone."""
    return f"The current time in {timezone} is 12:00 PM."

# Control which function gets called
response = llm.generate(
    "What's the weather in New York?",
    tools=[get_weather, get_time],
    # OpenAI-specific parameter to force a particular function
    function_call={"name": "get_weather"}
)
```

#### Response Format Control

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("openai", model="gpt-4")

# Get JSON response
response = llm.generate(
    "List the top 3 planets by size in our solar system.",
    # OpenAI-specific parameter
    response_format={"type": "json_object"}
)
```

#### Logit Bias

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("openai", model="gpt-4")

# Adjust token probabilities
response = llm.generate(
    "Write a short poem.",
    # OpenAI-specific parameter
    logit_bias={4148: 10}  # Increase probability of "love"
)
```

### Anthropic

Anthropic's Claude models offer unique features:

#### System Prompt as Parameter

While AbstractLLM standardizes system prompt as a parameter, Anthropic has additional options:

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Use both standard system prompt and anthropic-specific parameters
response = llm.generate(
    "Tell me about AI safety.",
    system_prompt="You are a helpful assistant focused on AI safety.",
    # Anthropic-specific parameter
    system={"type": "expert", "domain": "AI safety"}
)
```

#### Metadata

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Use metadata for tracking
response = llm.generate(
    "Summarize our conversation.",
    # Anthropic-specific parameter
    metadata={
        "conversation_id": "abc123",
        "user_id": "user456",
        "session_type": "support"
    }
)
```

#### Stop Sequences

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Use custom stop sequences
response = llm.generate(
    "Write a dialogue between two people.",
    # Anthropic-specific parameter
    stop_sequences=["Person 1:", "THE END"]
)
```

### HuggingFace

HuggingFace models offer extensive generation parameters:

#### Generation Strategy Parameters

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("huggingface", model="mistralai/Mistral-7B-Instruct-v0.1")

# Use HuggingFace-specific generation parameters
response = llm.generate(
    "Write a creative short story.",
    # HuggingFace-specific parameters
    do_sample=True,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    num_return_sequences=1
)
```

#### Pipeline Configuration

```python
from abstractllm import create_llm

# Create the provider with pipeline-specific configuration
llm = create_llm(
    "huggingface", 
    model="mistralai/Mistral-7B-Instruct-v0.1",
    # HuggingFace-specific parameters
    pipeline_kwargs={
        "torch_dtype": "float16",
        "device_map": "auto",
        "trust_remote_code": True
    }
)

response = llm.generate("Explain quantum computing.")
```

### Ollama

Ollama offers configuration for local model usage:

#### Model Parameters

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("ollama", model="llama2")

# Use Ollama-specific parameters
response = llm.generate(
    "Explain how batteries work.",
    # Ollama-specific parameters
    num_ctx=4096,
    num_gpu=1,
    num_thread=4,
    repeat_penalty=1.1,
    temperature=0.7,
    top_k=40,
    top_p=0.9
)
```

#### Model Loading Options

```python
from abstractllm import create_llm

# Create the provider with model loading options
llm = create_llm(
    "ollama", 
    model="llama2",
    # Ollama-specific parameters
    mmap=True,
    numa=False
)

response = llm.generate("What is machine learning?")
```

## Checking Provider Capabilities

To check what capabilities a provider supports, use the `get_capabilities` method:

```python
from abstractllm import create_llm

# Create the provider
llm = create_llm("openai", model="gpt-4")

# Check capabilities
capabilities = llm.get_capabilities()
print(capabilities)
```

This will return a dictionary with capabilities such as:
```
{
    "streaming": True, 
    "max_tokens": 4096, 
    "system_prompt": True, 
    "async": True, 
    "function_calling": True, 
    "tool_use": True, 
    "vision": True
}
```

## Best Practices

When using provider-specific features:

1. **Check Documentation**: Always refer to the provider's documentation for the most up-to-date parameters.

2. **Fallback Options**: If your application needs to work with multiple providers, implement fallback logic for provider-specific features.

3. **Validation**: AbstractLLM does not validate provider-specific parameters. If you pass an invalid parameter, the provider's API will return an error.

4. **Cross-Provider Compatibility**: When possible, prefer AbstractLLM's standardized parameters for better cross-provider compatibility.

5. **Feature Detection**: Use the `get_capabilities` method to detect what features a provider supports before using them.

## Example: Provider-Agnostic Code with Fallbacks

Here's how to write code that uses provider-specific features with fallbacks:

```python
from abstractllm import create_llm

def generate_with_features(provider_name, model, prompt):
    """Generate response with provider-specific features when available."""
    llm = create_llm(provider_name, model=model)
    capabilities = llm.get_capabilities()
    
    kwargs = {
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    # Add provider-specific features when available
    if provider_name == "openai":
        kwargs["frequency_penalty"] = 0.5
    elif provider_name == "anthropic":
        kwargs["metadata"] = {"session_type": "example"}
    elif provider_name == "huggingface":
        kwargs["do_sample"] = True
        kwargs["top_k"] = 50
    
    return llm.generate(prompt, **kwargs)

# Try with different providers
openai_response = generate_with_features("openai", "gpt-4", "Write a poem.")
anthropic_response = generate_with_features("anthropic", "claude-3-opus-20240229", "Write a poem.")
huggingface_response = generate_with_features("huggingface", "mistralai/Mistral-7B-Instruct-v0.1", "Write a poem.")
```

## Advanced: Creating Wrapper Methods

For frequently used provider-specific features, consider creating wrapper methods:

```python
from abstractllm import create_llm
from typing import Any, Dict, Optional

class EnhancedLLM:
    def __init__(self, provider: str, model: str, **kwargs):
        self.llm = create_llm(provider, model=model, **kwargs)
        self.provider = provider
    
    def generate_json(self, prompt: str, **kwargs) -> str:
        """Generate JSON-formatted responses when supported."""
        if self.provider == "openai":
            kwargs["response_format"] = {"type": "json_object"}
        elif self.provider == "anthropic":
            # Claude doesn't have a direct JSON parameter, so use a system instruction
            system = kwargs.get("system_prompt", "")
            kwargs["system_prompt"] = system + " Return your response in valid JSON format."
        
        return self.llm.generate(prompt, **kwargs)
    
    def generate_with_bias(self, prompt: str, bias_tokens: Dict[int, int], **kwargs) -> str:
        """Generate with token bias when supported."""
        if self.provider == "openai":
            kwargs["logit_bias"] = bias_tokens
        # Other providers may have similar features with different parameter names
        
        return self.llm.generate(prompt, **kwargs)

# Use the enhanced wrapper
enhanced_llm = EnhancedLLM("openai", "gpt-4")
json_response = enhanced_llm.generate_json("List the top 5 programming languages.")
```

## Conclusion

Provider-specific features allow you to leverage the unique capabilities of each LLM provider while maintaining a consistent interface through AbstractLLM. By understanding how to access and use these features, you can build applications that take full advantage of each provider's strengths.

For more information on specific providers, refer to their individual documentation:

- [OpenAI Provider Documentation](../providers/openai.md)
- [Anthropic Provider Documentation](../providers/anthropic.md)
- [HuggingFace Provider Documentation](../providers/huggingface.md)
- [Ollama Provider Documentation](../providers/ollama.md) 