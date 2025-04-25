# Basic Text Generation

This guide explains how to use AbstractLLM for basic text generation across different providers. You'll learn how to set up providers, generate responses, and customize the generation process.

## Creating a Provider

The first step is to create an LLM provider using the `create_llm` function:

```python
from abstractllm import create_llm

# Create an OpenAI provider
openai_llm = create_llm(
    "openai", 
    model="gpt-4",
    api_key="your-api-key"  # Optional: can also use OPENAI_API_KEY env variable
)

# Create an Anthropic provider
anthropic_llm = create_llm(
    "anthropic", 
    model="claude-3-opus-20240229",
    api_key="your-api-key"  # Optional: can also use ANTHROPIC_API_KEY env variable
)

# Create an Ollama provider (local models)
ollama_llm = create_llm(
    "ollama", 
    model="llama3"
)

# Create a HuggingFace provider
hf_llm = create_llm(
    "huggingface", 
    model="google/gemma-2b"
)
```

## Simple Text Generation

The simplest way to generate text is with the `generate` method:

```python
# Basic generation with a prompt
response = openai_llm.generate("Explain quantum computing in simple terms.")
print(response)
```

## Using System Prompts

System prompts help set the context or role for the model:

```python
# Generation with a system prompt
response = anthropic_llm.generate(
    prompt="What are the health benefits of regular exercise?",
    system_prompt="You are a certified personal trainer and nutrition expert."
)
print(response)
```

## Customizing Generation Parameters

You can customize various generation parameters:

```python
# Customizing generation parameters
response = ollama_llm.generate(
    prompt="Write a creative short story about a robot.",
    temperature=0.8,     # Controls randomness (0.0 to 1.0)
    max_tokens=500,      # Maximum response length
    top_p=0.95,          # Nucleus sampling parameter
    frequency_penalty=0.5  # Reduces repetition
)
print(response)
```

## Provider-Specific Parameters

Each provider accepts specific parameters:

```python
# OpenAI-specific parameters
openai_response = openai_llm.generate(
    prompt="Suggest names for a tech startup.",
    temperature=0.8,
    max_tokens=100,
    presence_penalty=0.2,
    top_p=0.95
)

# Anthropic-specific parameters
anthropic_response = anthropic_llm.generate(
    prompt="Write a poem about autumn.",
    temperature=0.7,
    max_tokens=300
)

# Ollama-specific parameters
ollama_response = ollama_llm.generate(
    prompt="Explain how a car engine works.",
    num_predict=500,
    temperature=0.8,
    mirostat=2
)
```

## Working with Multiple Turns

For simple multi-turn conversations without using sessions:

```python
# First turn
response1 = openai_llm.generate("Tell me about Mars.")
print("Response 1:", response1)

# Second turn (including previous context)
response2 = openai_llm.generate(
    "What about its moons?",
    messages=[
        {"role": "user", "content": "Tell me about Mars."},
        {"role": "assistant", "content": response1},
        {"role": "user", "content": "What about its moons?"}
    ]
)
print("Response 2:", response2)
```

For more complex multi-turn conversations, see the [Sessions guide](sessions.md).

## Handling Generation Errors

Always handle potential errors during generation:

```python
from abstractllm.exceptions import AbstractLLMError

try:
    response = openai_llm.generate("Tell me about quantum physics.")
    print(response)
except AbstractLLMError as e:
    print(f"Generation error: {str(e)}")
```

For more details on error handling, see the [Error Handling guide](error-handling.md).

## Checking Generation Compatibility

You can check if a model supports basic text generation with the capabilities API:

```python
from abstractllm import create_llm, ModelCapability

llm = create_llm("openai", model="gpt-4")
capabilities = llm.get_capabilities()

# All models support basic text generation, but max token length may vary
max_tokens = capabilities.get(ModelCapability.MAX_TOKENS)
print(f"This model supports up to {max_tokens} tokens in its response")
```

## Performance Considerations

For basic text generation:

1. **Model Selection**: Smaller models are faster and cheaper but may produce lower quality responses.
2. **Token Length**: Shorter prompts and responses use fewer tokens and are processed faster.
3. **Temperature**: Lower temperature values (0.0-0.3) produce more deterministic responses, which may be faster to generate.

## Provider Comparison

| Provider | Strengths | Considerations |
|----------|-----------|----------------|
| OpenAI | Strong general capabilities, consistent responses | API costs, rate limits |
| Anthropic | Excellent for complex reasoning, longer contexts | Higher latency for some tasks |
| Ollama | Local deployment, no data sharing, customizable | Quality depends on model, requires hardware |
| HuggingFace | Wide model selection, local deployment options | Variable quality, may require more tuning |

## Next Steps

After mastering basic generation, you can explore:

- [Working with Sessions](sessions.md) for multi-turn conversations
- [Streaming Responses](streaming.md) for real-time generation
- [Asynchronous Generation](async.md) for concurrent requests
- [Tool Calls](tools.md) for augmenting the LLM with external functions
- [Vision Capabilities](vision.md) for working with image inputs 