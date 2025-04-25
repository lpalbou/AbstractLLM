# Provider Interchangeability

This document outlines the design considerations and implementation details that enable seamless switching between different LLM providers in AbstractLLM.

## Core Interchangeability Principles

AbstractLLM is designed to make providers as interchangeable as possible, allowing developers to switch between providers with minimal code changes. The following principles guide this design:

1. **Consistent Interface**: All providers implement the same abstract interface (`AbstractLLMInterface`).
2. **Normalized Parameters**: Common parameters are normalized across providers.
3. **Capability Inspection**: Runtime capability checks allow code to adapt to provider-specific limitations.
4. **Standardized Response Format**: All providers return responses in a consistent format.
5. **Unified Error Handling**: Provider-specific errors are wrapped in standard exceptions.

## Parameter Harmonization

One of the key challenges in provider interchangeability is handling different parameter naming, scaling, and behavior:

### Temperature Normalization

Different providers use different scales for temperature:

```python
# Example of temperature normalization in ConfigurationManager
def _normalize_temperature(provider: str, temp_value: float) -> float:
    """Normalize temperature across providers to ensure consistent behavior."""
    if provider == "anthropic" and temp_value > 1.0:
        # Anthropic uses 0-1 scale
        return min(temp_value / 2.0, 1.0)
    elif provider == "ollama" and temp_value < 0.1:
        # Some Ollama models don't work well with very low temperatures
        return max(temp_value, 0.1)
    return temp_value
```

### Token Limit Handling

Maximum token limits vary across providers and models:

```python
# Configuration system considers model-specific token limits
max_tokens = min(
    requested_tokens,
    provider_capabilities.get(ModelCapability.MAX_TOKENS, float('inf'))
)
```

## Response Normalization

Provider-specific response formats are normalized to ensure consistency:

1. **Whitespace Handling**: Trim leading/trailing whitespace consistently.
2. **Error Messages**: Normalize error messages into standard formats.
3. **Metadata Handling**: Extract and standardize response metadata.

## Runtime Provider Switching

To switch providers during a sequential process:

```python
# Example of switching providers mid-process
from abstractllm import create_llm, ModelParameter

# Initial provider
llm1 = create_llm("openai", model="gpt-3.5-turbo")
response1 = llm1.generate("First part of a story about a robot")

# Switch to a different provider for continuation
llm2 = create_llm("anthropic", model="claude-3-5-haiku-20241022")
response2 = llm2.generate(
    f"Continue this story: {response1}\n\nNext part:", 
    system_prompt="Continue the story in the same style and tone."
)
```

## Context Preservation

When switching between providers, context can be preserved through:

1. **Explicit Context Passing**: Including previous exchanges in the prompt.
2. **System Prompt Adaptation**: Modifying system prompts to maintain consistent persona.
3. **Parameter Consistency**: Using similar generation parameters across providers.

## Provider-Specific Considerations

### OpenAI ↔ Anthropic

Both providers have similar capabilities, but there are some considerations:

- **System Prompt Usage**: OpenAI and Anthropic handle system prompts differently.
- **Function Calling**: OpenAI has more advanced function calling capabilities.
- **Vision Models**: Both support vision, but with different capabilities and limitations.

### Cloud ↔ Local Models

When switching between cloud (OpenAI/Anthropic) and local models (Ollama/HuggingFace):

- **Latency Differences**: Local models may have higher latency for first request.
- **Quality Expectations**: Adjust quality expectations when switching to smaller local models.
- **Error Handling**: Local models have different failure modes than API-based providers.

### Prompt Engineering Considerations

Different models respond differently to prompt engineering techniques:

- **Few-shot Examples**: Varying effectiveness across providers.
- **Instruction Format**: Some models are more sensitive to instruction phrasing.
- **Token Limitations**: Adjust prompt strategy based on context window size.

## Error Handling for Interchangeability

For robust provider switching, implement graceful fallbacks:

```python
def generate_with_fallback(prompt, providers=["openai", "anthropic", "ollama"]):
    """Try multiple providers in sequence if earlier ones fail."""
    for provider_name in providers:
        try:
            llm = create_llm(provider_name)
            return llm.generate(prompt)
        except Exception as e:
            print(f"Provider {provider_name} failed: {e}")
            continue
    raise RuntimeError("All providers failed")
```

## Testing Interchangeability

To ensure providers are truly interchangeable, consider:

1. **Cross-Provider Tests**: Run the same tests across all providers.
2. **A/B Comparison Tests**: Compare responses from different providers for the same prompts.
3. **Provider Switching Tests**: Test explicit provider switching scenarios.
4. **Error Recovery Tests**: Verify fallback mechanisms work as expected.

## Future Interchangeability Improvements

Planned enhancements to further improve provider interchangeability:

1. **Response Quality Normalization**: Standardize response quality expectations.
2. **Dynamic Parameter Adjustment**: Automatically adjust parameters based on provider.
3. **Context Window Management**: Tools for working within different context limitations.
4. **Prompt Templates**: Provider-optimized prompt templates for consistent results.
5. **Capability-Based Routing**: Automatically select appropriate provider based on needed capabilities. 