# Architecture Detection & Model Capabilities

## Overview

AbstractLLM features an intelligent architecture detection system that automatically configures providers and models based on comprehensive JSON assets. This system enables seamless model switching while maintaining optimal performance and feature compatibility.

## How It Works

The framework uses two key JSON files to understand model capabilities:

1. **`architecture_formats.json`** - Defines **HOW** to communicate with models:
   - Message formatting patterns (chat templates)
   - Role prefixes/suffixes (system, user, assistant)
   - Tool calling formats (JSON, XML, pythonic)

2. **`model_capabilities.json`** - Defines **WHAT** models can do:
   - Context length limits (input/output tokens)
   - Tool support level (native/prompted/none)
   - Vision/audio capabilities
   - Parallel tool execution support

## Model Name Normalization

The system automatically normalizes model names for consistent detection:

```python
# Examples of model name normalization:
"mlx-community/Qwen3-30B-A3B-4bit" → "qwen3-30b-a3b"
"qwen/qwen3-next-80b" → "qwen3-next-80b-a3b"
"ollama:qwen3:30b-a3b" → "qwen3-30b-a3b"
"huggingface/microsoft/Phi-4-mini" → "phi-4-mini"
```

### Normalization Process

1. **Remove provider prefixes**: `mlx-community/`, `huggingface/`, `ollama/`, `local/`, `qwen/`
2. **Convert Ollama format**: Replace `:` with `-` (e.g., `qwen3:30b` → `qwen3-30b`)
3. **Handle special patterns**: Map `qwen3-next` → `qwen3-next-80b-a3b`
4. **Strip quantization suffixes**: Remove `-4bit`, `-8bit`, `-q4_k_m`, etc.
5. **Preserve version numbers**: Keep semantic versions like `3.5`, `2.5`

## Architecture Detection Examples

```python
from abstractllm.architectures.detection import detect_architecture, get_model_capabilities

# Automatic architecture detection
arch = detect_architecture("qwen/qwen3-next-80b")  # → "qwen"
caps = get_model_capabilities("qwen/qwen3-next-80b")

print(f"Architecture: {arch}")
print(f"Context length: {caps['context_length']:,} tokens")
print(f"Max output: {caps['max_output_tokens']:,} tokens")
print(f"Tool support: {caps['tool_support']}")
print(f"Vision support: {caps['vision_support']}")
```

## Unified Memory Management

The `/mem` command provides detailed information about model capabilities:

```bash
user> /mem

🧠 Memory System Overview
────────────────────────────────────────────────────────────
  Model: qwen/qwen3-next-80b
  Model Max: 262,144 input / 16,384 output
  Token Usage & Limits:
    • Input Context: 182 / 262,144 tokens (0.1%)
      Source: 3 messages
    • Output Limit: 16,384 tokens max
    • Commands: /mem input <n> | /mem output <n> | /mem reset

  Generation Parameters:
    • Temperature: 0.7
    • Top-P: 0.95
    • Provider: Lmstudio
    • Base URL: http://localhost:1234/v1
```

## Provider Parameter System

AbstractLLM uses a unified parameter system with automatic capability detection:

```python
from abstractllm import create_llm
from abstractllm.enums import ModelParameter

# Parameters are automatically validated against model capabilities
llm = create_llm("lmstudio",
                 model="qwen/qwen3-next-80b",
                 temperature=0.7,           # Validated parameter
                 max_tokens=16384,          # Limited by model capability
                 top_p=0.95)

# Access parameters through unified interface
temp = llm.get_parameter(ModelParameter.TEMPERATURE)
limits = llm.get_memory_limits()  # Returns actual model limits

# Memory limit management
llm.set_memory_limits(max_input_tokens=100000, max_output_tokens=8192)
```

## Supported Architectures

| Architecture | Models | Message Format | Tool Format |
|-------------|--------|----------------|-------------|
| **llama** | Llama 3.x, 4, DeepSeek | `[INST]` tags | Pythonic calls |
| **qwen** | Qwen 2.5, 3, VL | ChatML `<\|im_start\|>` | JSON format |
| **mistral** | Mistral, Mixtral | `[INST]` simple | JSON format |
| **claude** | Claude 3.x | Human:/Assistant: | XML format |
| **gpt** | GPT-3.5, 4, o1, o3 | OpenAI chat | OpenAI functions |
| **phi** | Phi-2 through Phi-4 | Instruction format | JSON format |
| **gemma** | Gemma, CodeGemma | Instruction format | JSON format |

### Architecture Details

#### Llama Family
- **Pattern Detection**: `llama`, `deepseek`, `yi`
- **Message Template**: `[INST] {user_message} [/INST]`
- **System Prompt**: `<<SYS>>\n{system_message}\n<</SYS>>\n\n`
- **Tool Format**: Pythonic function calls in text

#### Qwen Family
- **Pattern Detection**: `qwen`, `qwen2`, `qwen3`
- **Message Template**: ChatML with `<|im_start|>` and `<|im_end|>` tags
- **Tool Format**: `<|tool_call|>JSON</|tool_call|>` format
- **Special Features**: Extended context support, vision variants

#### Claude Family
- **Pattern Detection**: `claude`
- **Message Template**: `Human: {message}\n\nAssistant:`
- **Tool Format**: XML-structured tool calls
- **Provider**: Anthropic API handles formatting

#### GPT Family
- **Pattern Detection**: `gpt`, `o1`, `o3`
- **Message Template**: OpenAI chat completions format
- **Tool Format**: OpenAI functions/tools API
- **Provider**: OpenAI API handles formatting

## Model Capability Detection

The system automatically detects model capabilities:

```python
from abstractllm.architectures.detection import (
    supports_tools, supports_vision, get_context_limits, is_instruct_model
)

# Capability queries
if supports_tools("qwen/qwen3-next-80b"):
    print("Model supports tool calling")

if supports_vision("llama-3.2-vision"):
    print("Model supports image input")

limits = get_context_limits("claude-3.5-sonnet")
print(f"Context: {limits['input']:,} / {limits['output']:,}")

# Model type detection
if is_instruct_model("phi-4"):
    print("Instruction-tuned model")
```

### Capability Categories

#### Tool Support Levels
- **native**: Full API-level tool support (OpenAI functions, Anthropic tools)
- **prompted**: Tool support via careful prompt engineering
- **none**: No tool support capabilities

#### Context Management
- **context_length**: Maximum input tokens the model can process
- **max_output_tokens**: Maximum tokens the model can generate
- **Automatic limiting**: Framework enforces model limits

#### Multimodal Capabilities
- **vision_support**: Can process images (Claude 3.x, GPT-4V, Llama 3.2 Vision)
- **audio_support**: Can process audio (GPT-4o, Llama 4)
- **image_resolutions**: Supported image sizes and formats

## Provider-Specific Considerations

### Tool Support Reality Check

The architecture detection system accounts for provider limitations:

```python
# Model capability vs Provider API support
model_caps = get_model_capabilities("llama-3.1-70b")
# Returns: {"tool_support": "native"}

# But provider might override:
# - LM Studio: Always uses "prompted" (no OpenAI tools API)
# - MLX: Always uses "prompted" (local inference)
# - Ollama: May use "prompted" depending on API support
```

### Provider Override System

```python
class LMStudioProvider:
    def get_effective_tool_support(self, model_name: str) -> str:
        """LM Studio doesn't support OpenAI tools parameter."""
        return "prompted"

class MLXProvider:
    def get_effective_tool_support(self, model_name: str) -> str:
        """MLX is local inference, no API-level tools."""
        return "prompted"
```

## JSON Asset Structure

### architecture_formats.json
```json
{
  "architectures": {
    "qwen": {
      "patterns": ["qwen", "qwen2", "qwen3"],
      "message_format": "chatml",
      "system_prefix": "<|im_start|>system\n",
      "user_prefix": "<|im_start|>user\n",
      "assistant_prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n",
      "tool_format": "json"
    }
  }
}
```

### model_capabilities.json
```json
{
  "models": {
    "qwen3-next-80b-a3b": {
      "context_length": 262144,
      "max_output_tokens": 16384,
      "tool_support": "prompted",
      "structured_output": "prompted",
      "parallel_tools": true,
      "vision_support": false,
      "audio_support": false,
      "notes": "Hybrid attention, MoE with 512 experts",
      "source": "Hugging Face model card"
    }
  }
}
```

## Adding New Models

### Step 1: Add Model Capabilities
```json
// In model_capabilities.json
"your-new-model": {
  "context_length": 32768,
  "max_output_tokens": 4096,
  "tool_support": "prompted",
  "structured_output": "native",
  "parallel_tools": false,
  "vision_support": false,
  "audio_support": false,
  "source": "Official documentation"
}
```

### Step 2: Update Architecture Patterns (if needed)
```json
// In architecture_formats.json - add to existing or create new
"your_architecture": {
  "patterns": ["your-model", "model-variant"],
  "message_format": "instruction",
  "user_prefix": "[USER]",
  "assistant_prefix": "[ASSISTANT]",
  "tool_format": "json"
}
```

### Step 3: Test Detection
```python
from abstractllm.architectures.detection import get_model_capabilities
caps = get_model_capabilities("your-new-model")
print(caps)  # Verify capabilities are detected correctly
```

## Testing & Validation

### Capability Validation
```python
def test_model_detection():
    """Test cases for model capability detection."""
    test_cases = [
        ("qwen/qwen3-next-80b", "qwen", 262144, 16384),
        ("llama-3.1-70b", "llama", 128000, 8192),
        ("claude-3.5-sonnet", "claude", 200000, 8192),
    ]

    for model, expected_arch, expected_ctx, expected_out in test_cases:
        arch = detect_architecture(model)
        caps = get_model_capabilities(model)

        assert arch == expected_arch
        assert caps["context_length"] == expected_ctx
        assert caps["max_output_tokens"] == expected_out
```

### Provider Integration Testing
```python
def test_provider_model_compatibility():
    """Test provider-model combinations."""
    providers = ["openai", "anthropic", "lmstudio", "ollama", "mlx"]
    models = ["gpt-4o", "claude-3.5-sonnet", "qwen/qwen3-next-80b"]

    for provider in providers:
        for model in models:
            try:
                llm = create_llm(provider, model=model)
                caps = llm.get_memory_limits()
                print(f"{provider} + {model}: {caps}")
            except Exception as e:
                print(f"{provider} + {model}: FAILED - {e}")
```

## Performance Considerations

### Caching Strategy
- JSON files loaded once per session
- Model capability lookups cached by normalized name
- Architecture detection cached by model pattern

### Optimization Notes
- Normalization is string-based (fast)
- Pattern matching uses simple substring checks
- No regex or complex parsing in hot paths
- Capability inheritance reduces JSON size

## Troubleshooting

### Common Issues

#### Model Not Detected
```python
# Check normalization
from abstractllm.architectures.detection import _normalize_model_name
normalized = _normalize_model_name("your-model-name")
print(f"Normalized: {normalized}")

# Check if exists in capabilities
import json
with open("abstractllm/assets/model_capabilities.json") as f:
    data = json.load(f)
print(normalized in data["models"])
```

#### Wrong Capabilities Returned
```python
# Debug capability lookup
caps = get_model_capabilities("model-name")
print(f"Tool support: {caps.get('tool_support')}")
print(f"Source: {caps.get('source', 'defaults')}")

# Check if using defaults vs specific entry
```

#### Provider Incompatibility
```python
# Test provider tool support
llm = create_llm("provider", model="model")
try:
    # Test with simple tool
    response = llm.generate("test", tools=[simple_tool])
    print("Tools work")
except Exception as e:
    print(f"Tools failed: {e}")
```

## Future Enhancements

### Planned Improvements
1. **Dynamic Capability Discovery**: Runtime detection of provider API features
2. **Model Family Inheritance**: Share capabilities across model variants
3. **Community Contributions**: Crowdsourced model capability database
4. **Automatic Updates**: Sync with HuggingFace model cards

### API Evolution
1. **Provider Capability Override**: Explicit provider-level tool support detection
2. **Runtime Feature Testing**: Test API capabilities on first use
3. **Graceful Degradation**: Auto-fallback between tool support modes
4. **Better Error Messages**: Clear indication of capability mismatches

---

This system enables AbstractLLM to provide intelligent, automatic model configuration while maintaining flexibility and extensibility for new models and providers.