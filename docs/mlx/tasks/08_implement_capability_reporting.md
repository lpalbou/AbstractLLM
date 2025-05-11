# Task 8: Implement Capability Reporting

## Description
Implement the capability reporting system for the MLX provider to properly advertise what features are supported.

## Requirements
1. Implement the `get_capabilities()` method required by AbstractLLMInterface
2. Report accurate information about streaming, max tokens, system prompts, etc.
3. Report vision capabilities based on the loaded model
4. Update the capability reporting system when model changes

## Implementation Details

```python
def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
    """Return capabilities of this LLM provider."""
    capabilities = {
        ModelCapability.STREAMING: True,
        ModelCapability.MAX_TOKENS: self.config_manager.get_param(ModelParameter.MAX_TOKENS, 4096),
        ModelCapability.SYSTEM_PROMPT: True,
        ModelCapability.ASYNC: True,
        ModelCapability.FUNCTION_CALLING: False,
        ModelCapability.TOOL_USE: False,
        ModelCapability.VISION: self._is_vision_model,
    }
    
    return capabilities

def _is_vision_capable(self) -> bool:
    """Check if the current model supports vision."""
    return self._is_vision_model
```

## References
- See AbstractLLMInterface for the required method signatures
- See `abstractllm/enums.py` for ModelCapability enum
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`

## Testing
1. Test capability reporting with both vision and non-vision models
2. Verify that capabilities accurately reflect the model's abilities
3. Check that the capability system integrates with AbstractLLM's model selection
4. Verify capability updates after switching models 