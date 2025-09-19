# Native Tool Support Audit

## Overview
Analysis of all models marked with "native" tool support to identify potential compatibility issues with different providers.

## Risk Assessment

### ✅ SAFE - Provider Officially Supports Tools
These models are accessed through providers that officially support OpenAI tool calling:

**OpenAI Models** (via OpenAI provider):
- `gpt-4`, `gpt-4o`, `gpt-4o-long-output`, `gpt-4o-mini`, `gpt-3.5-turbo`
- `o3`, `o3-mini`

**Anthropic Models** (via Anthropic provider):
- `claude-3.5-sonnet`, `claude-3.7-sonnet`, `claude-3.5-haiku`
- `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### ⚠️ POTENTIALLY RISKY - Check Provider Compatibility

**Llama Models** (via Ollama, MLX, HuggingFace):
- `llama-3.1-8b`, `llama-3.1-70b`, `llama-3.1-405b`
- `llama-4`, `llama4-17b-scout-16e-instruct`
- **Risk**: Ollama may not support OpenAI tools parameter
- **Risk**: MLX definitely doesn't support native tools
- **Risk**: HuggingFace may have limited tool API support

**Mistral Models** (via Ollama, HuggingFace):
- `mixtral-8x22b`, `mistral-small`, `mistral-medium`, `mistral-large`, `codestral`
- **Risk**: Depends on hosting provider's API compatibility

**Google Models** (via Ollama, MLX, HuggingFace):
- `gemma3`
- **Risk**: Likely requires prompted mode on most providers

**DeepSeek Models** (via Ollama, MLX):
- `deepseek-r1`
- **Risk**: Probably requires prompted mode

## Provider-Specific Analysis

### LM Studio Provider
- **Confirmed Issue**: Does NOT support OpenAI `tools` parameter
- **All models should use**: "prompted" tool support
- **Fix Required**: Provider-level override

### Ollama Provider
- **Likely Issue**: Uses `/api/chat` endpoint, may not support `tools` parameter
- **Needs Testing**: Llama 3.1 models, Mistral models, Gemma3
- **Expected**: Should use "prompted" mode

### MLX Provider
- **Confirmed**: No native tool API support
- **All models should use**: "prompted" tool support
- **Note**: MLX is local inference, no API-level tools

### HuggingFace Provider
- **Variable**: Depends on specific inference endpoint
- **Hosted Models**: May support tools via Inference API
- **Local Models**: Likely require prompted mode

## Testing Priority

### High Priority Tests Needed
1. **Ollama + Llama 3.1 models**: Check if tools parameter works
2. **MLX + any "native" model**: Should fail, verify graceful fallback
3. **HuggingFace + Mistral models**: Test inference API compatibility

### Medium Priority Tests
1. All Gemma3 variants across providers
2. DeepSeek models on different providers
3. Cross-provider model switching with tools

## Recommended Actions

### Immediate (This Release)
1. ✅ **Fixed LM Studio**: qwen3-next-80b-a3b now uses "prompted"
2. **Audit Ollama**: Test Llama 3.1 models with tools
3. **Verify MLX**: Ensure all "native" models fall back to prompted

### Short Term (Next Release)
1. **Provider Capability Override**: Implement per-provider tool support detection
2. **Better Error Messages**: Clearer indication when API doesn't support tools
3. **Runtime Detection**: Test tool API support on first use

### Long Term (Architecture Improvement)
1. **Separate Model vs Provider Capabilities**: Distinguish what model can do vs what API supports
2. **Feature Detection**: Automatically detect provider API capabilities
3. **Graceful Degradation**: Auto-fallback from native to prompted

## Expected Tool Support Matrix

| Provider | Supports Native Tools | Fallback Strategy |
|----------|----------------------|-------------------|
| OpenAI | ✅ Yes | N/A |
| Anthropic | ✅ Yes | N/A |
| LM Studio | ❌ No | Always use prompted |
| Ollama | ❓ Unknown | Test required |
| MLX | ❌ No | Always use prompted |
| HuggingFace | ❓ Variable | Test per model |

## Test Cases to Implement

### Provider + Model Combinations
```python
test_cases = [
    ("ollama", "llama-3.1-8b"),
    ("ollama", "mixtral-8x22b"),
    ("ollama", "gemma3"),
    ("mlx", "llama-3.1-8b"),
    ("mlx", "deepseek-r1"),
    ("huggingface", "mistral-large"),
    ("lmstudio", "llama-3.1-70b"),  # Verify our fix works broadly
]
```

### Expected Behaviors
- **Native tools should work**: OpenAI + GPT models, Anthropic + Claude models
- **Native tools should fail gracefully**: LM Studio, MLX + any model
- **Native tools need verification**: Ollama, HuggingFace + most models

## Fix Implementation Strategy

### Phase 1: Quick Fixes (Current)
- Change problematic model capabilities to "prompted"
- Document known limitations

### Phase 2: Provider Detection (Next)
```python
class LMStudioProvider:
    def get_effective_tool_support(self, model_name):
        # Override any "native" to "prompted"
        return "prompted"

class MLXProvider:
    def get_effective_tool_support(self, model_name):
        # MLX never supports API-level tools
        return "prompted"
```

### Phase 3: Runtime Detection (Future)
```python
def detect_provider_tool_support(provider):
    # Test API with dummy tool call
    # Cache result for session
    # Fall back appropriately
```

---

**Status**: In Progress
**Next Actions**: Test Ollama provider with Llama 3.1 models
**Owner**: Architecture team
**Priority**: High (affects multiple providers)