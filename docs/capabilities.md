# Model Capabilities in AbstractLLM

AbstractLLM provides a way to inspect and utilize the capabilities of different LLM providers through the `ModelCapability` enum and the `get_capabilities()` method.

## Available Capabilities

The following capabilities are defined in the `ModelCapability` enum:

| Capability | Description |
|------------|-------------|
| `STREAMING` | Supports streaming responses chunk by chunk |
| `MAX_TOKENS` | Maximum supported tokens for context + generation |
| `SYSTEM_PROMPT` | Supports system prompts for controlling behavior |
| `ASYNC` | Supports asynchronous generation |
| `FUNCTION_CALLING` | Supports function/tool calling |
| `VISION` | Supports image input processing |
| `FINE_TUNING` | Supports fine-tuning |
| `EMBEDDINGS` | Supports embedding generation |
| `MULTILINGUAL` | Supports multiple languages |
| `RAG` | Supports Retrieval Augmented Generation |
| `MULTI_TURN` | Supports multi-turn conversations |
| `PARALLEL_INFERENCE` | Supports parallel inference |
| `IMAGE_GENERATION` | Supports generating images |
| `AUDIO_PROCESSING` | Supports audio input/output |
| `JSON_MODE` | Supports structured JSON output |

## Checking Capabilities

You can check if a provider has a specific capability using the `get_capabilities()` method:

```python
from abstractllm import create_llm, ModelCapability

# Create an LLM instance
llm = create_llm("openai")

# Get capabilities
capabilities = llm.get_capabilities()

# Check for specific capabilities
if capabilities.get(ModelCapability.STREAMING):
    print("This provider supports streaming")
    
if capabilities.get(ModelCapability.VISION):
    print("This provider supports vision input")
    
if capabilities.get(ModelCapability.JSON_MODE):
    print("This provider supports JSON mode")
```

## Using JSON Mode

Providers that support JSON mode (like OpenAI) can be instructed to return structured JSON output by setting the `json_mode` parameter to `True`:

```python
from abstractllm import create_llm

llm = create_llm("openai")

# Generate a structured JSON response
response = llm.generate(
    "Create a user profile with name, age, and favorite foods.",
    json_mode=True
)

# Parse the response
import json
user_profile = json.loads(response)
print(f"Name: {user_profile['name']}")
print(f"Age: {user_profile['age']}")
print(f"Favorite Foods: {', '.join(user_profile['favorite_foods'])}")
```

## Provider-Specific Capability Notes

### OpenAI

- Supports streaming responses
- Supports system prompts
- Supports asynchronous generation
- Supports function calling
- Supports JSON mode
- Vision support depends on the model (e.g., GPT-4V, GPT-4o)

### Anthropic

- Supports streaming responses
- Supports system prompts
- Supports asynchronous generation
- Vision support depends on the model (Claude 3 models support vision)
- Does not support JSON mode in the same way as OpenAI

### Ollama

- Supports streaming responses
- Supports system prompts on models that support it
- Supports asynchronous generation
- Vision support depends on the model

### HuggingFace

- Supports streaming responses
- System prompt support varies by model
- Supports asynchronous generation
- Vision support depends on the model

## Capability-Aware Code

You can write capability-aware code that adapts to the provider's capabilities:

```python
from abstractllm import create_llm, ModelCapability

def process_with_llm(llm, prompt, image_path=None):
    # Check capabilities
    capabilities = llm.get_capabilities()
    
    # If image is provided but vision not supported, warn and ignore image
    if image_path and not capabilities.get(ModelCapability.VISION):
        print("Warning: Provider does not support vision. Ignoring image input.")
        return llm.generate(prompt)
    elif image_path:
        return llm.generate(prompt, image=image_path)
    
    # If JSON mode is requested and supported
    if need_json and capabilities.get(ModelCapability.JSON_MODE):
        return llm.generate(prompt, json_mode=True)
    
    # Default case
    return llm.generate(prompt)
``` 