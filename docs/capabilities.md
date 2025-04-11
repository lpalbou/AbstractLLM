# Model Capabilities in AbstractLLM

## Overview

AbstractLLM provides a comprehensive capability inspection system through the `ModelCapability` enum and the `get_capabilities()` method. This system allows you to dynamically check what features are available for each provider and model combination.

## Core Capabilities

| Capability | Description | OpenAI | Anthropic | HuggingFace | Ollama |
|------------|-------------|---------|-----------|-------------|---------|
| `STREAMING` | Stream responses | ✅ | ✅ | ✅ | ✅ |
| `VISION` | Process images | ✅* | ✅* | ✅* | ✅* |
| `SYSTEM_PROMPT` | Support system prompts | ✅ | ✅ | ✅ | ✅ |
| `ASYNC` | Async generation | ✅ | ✅ | ✅ | ✅ |
| `FUNCTION_CALLING` | Function/tool use | ✅ | ❌ | ❌ | ❌ |
| `JSON_MODE` | Structured JSON output | ✅ | ✅ | ❌ | ❌ |
| `MULTI_TURN` | Conversation history | ✅ | ✅ | ❌ | ❌ |

\* Vision support depends on the specific model being used

## Checking Capabilities

### Basic Capability Check

```python
from abstractllm import create_llm, ModelCapability

# Create provider instance
llm = create_llm("openai", model="gpt-4o")

# Get all capabilities
capabilities = llm.get_capabilities()

# Check specific capabilities
if capabilities[ModelCapability.VISION]:
    # Process images
    response = llm.generate("Describe this:", files=["image.jpg"])

if capabilities[ModelCapability.STREAMING]:
    # Use streaming
    for chunk in llm.generate("Tell me a story", stream=True):
        print(chunk, end="")
```

### Vision Capabilities

Each provider has specific models that support vision:

```python
# OpenAI Vision Models
llm = create_llm("openai", model="gpt-4o")  # Latest GPT-4 with vision
llm = create_llm("openai", model="gpt-4-vision-preview")  # Preview version

# Anthropic Vision Models
llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")  # Latest Claude
llm = create_llm("anthropic", model="claude-3-opus-20240229")  # High capability

# HuggingFace Vision Models
llm = create_llm("huggingface", model="Salesforce/blip-image-captioning-base")
llm = create_llm("huggingface", model="liuhaotian/llava-v1.5-7b")

# Ollama Vision Models
llm = create_llm("ollama", model="llama3.2-vision:latest")
llm = create_llm("ollama", model="bakllava:latest")
```

### Streaming Support

All providers support streaming, but with different implementations:

```python
# Synchronous streaming
for chunk in llm.generate("Tell me a story", stream=True):
    print(chunk, end="")

# Asynchronous streaming
async for chunk in llm.generate_async("Tell me a story", stream=True):
    print(chunk, end="")
```

### Function Calling

Currently supported by OpenAI:

```python
# Define functions
functions = [{
    "name": "get_weather",
    "description": "Get the weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}]

# Use function calling
response = llm.generate(
    "What's the weather in Paris?",
    functions=functions,
    function_call="auto"
)
```

### JSON Mode

For providers that support structured JSON output:

```python
# OpenAI JSON mode
response = llm.generate(
    "List three colors with their hex codes",
    response_format={"type": "json_object"}
)

# Anthropic JSON mode
response = llm.generate(
    "List three colors with their hex codes",
    system_prompt="Return response as valid JSON"
)
```

## Provider-Specific Capabilities

### OpenAI

```python
llm = create_llm("openai", model="gpt-4o")
capabilities = llm.get_capabilities()

# Available capabilities
print(f"""
Vision Support: {capabilities[ModelCapability.VISION]}
Max Tokens: {capabilities[ModelCapability.MAX_TOKENS]}
Function Calling: {capabilities[ModelCapability.FUNCTION_CALLING]}
JSON Mode: {capabilities[ModelCapability.JSON_MODE]}
""")
```

### Anthropic

```python
llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")
capabilities = llm.get_capabilities()

# Claude-specific capabilities
print(f"""
Vision Support: {capabilities[ModelCapability.VISION]}
Max Tokens: {capabilities[ModelCapability.MAX_TOKENS]}
Multi-turn: {capabilities[ModelCapability.MULTI_TURN]}
JSON Mode: {capabilities[ModelCapability.JSON_MODE]}
""")
```

### HuggingFace

```python
llm = create_llm("huggingface", model="microsoft/phi-2")
capabilities = llm.get_capabilities()

# Local model capabilities
print(f"""
Device Support: {capabilities["device_support"]}
Quantization: {capabilities["quantization_support"]}
Vision Support: {capabilities[ModelCapability.VISION]}
""")
```

### Ollama

```python
llm = create_llm("ollama", model="llama3.2-vision:latest")
capabilities = llm.get_capabilities()

# Ollama-specific capabilities
print(f"""
Vision Support: {capabilities[ModelCapability.VISION]}
GPU Acceleration: {capabilities["gpu_acceleration"]}
Quantization: {capabilities["quantization_support"]}
""")
```

## Capability-Aware Code

Write code that adapts to available capabilities:

```python
from abstractllm import create_llm, ModelCapability
from typing import Optional, List, Union
from pathlib import Path

def process_with_llm(
    llm,
    prompt: str,
    files: Optional[List[Union[str, Path]]] = None,
    stream: bool = False,
    json_output: bool = False
) -> Union[str, Generator[str, None, None]]:
    """Process input with capability-aware handling."""
    capabilities = llm.get_capabilities()
    
    # Handle files if provided
    if files and not capabilities[ModelCapability.VISION]:
        print("Warning: Provider does not support vision. Ignoring files.")
        files = None
    
    # Handle JSON output
    if json_output and not capabilities[ModelCapability.JSON_MODE]:
        print("Warning: JSON mode not supported. Using standard output.")
        json_output = False
    
    # Handle streaming
    if stream and not capabilities[ModelCapability.STREAMING]:
        print("Warning: Streaming not supported. Using standard generation.")
        stream = False
    
    # Generate response with appropriate parameters
    kwargs = {
        "stream": stream,
        "files": files,
    }
    
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    
    return llm.generate(prompt, **kwargs)
```

## Best Practices

1. **Always Check Capabilities**
   ```python
   capabilities = llm.get_capabilities()
   if not capabilities[ModelCapability.VISION]:
       raise UnsupportedFeatureError("Vision not supported")
   ```

2. **Provide Fallbacks**
   ```python
   if not capabilities[ModelCapability.JSON_MODE]:
       # Use system prompt to request structured output
       system_prompt = "Format response as JSON"
   ```

3. **Handle Model-Specific Features**
   ```python
   if capabilities[ModelCapability.FUNCTION_CALLING]:
       # Use function calling
       kwargs["functions"] = functions
   else:
       # Use text-based function simulation
       prompt = format_function_as_text(prompt, functions)
   ```

4. **Check Token Limits**
   ```python
   max_tokens = capabilities[ModelCapability.MAX_TOKENS]
   if max_tokens and token_count > max_tokens:
       raise ContextWindowExceededError(max_tokens, token_count)
   ```

## Future Capabilities

Planned capability enhancements:

1. **Audio Processing**
   - Speech-to-text
   - Text-to-speech
   - Audio analysis

2. **Advanced Vision**
   - Object detection
   - Image generation
   - Video processing

3. **Enhanced Tools**
   - Code execution
   - Database integration
   - External API access

4. **Memory and Context**
   - Long-term memory
   - Document retrieval
   - Knowledge graph integration 