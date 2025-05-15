# Knowledge Base

## MLX Provider

### Model Configurations

MLX provider now uses a centralized configuration system that handles model-specific details:

- Each model architecture (Llama, Qwen, Mistral, Phi, CodeModels) has its own configuration class
- Configurations specify:
  - EOS tokens for proper generation termination
  - BOS tokens for proper prompting 
  - Default generation parameters like repetition penalty
  - Custom system prompt formatting

This replaces the previous ad-hoc approach with a more robust system where model-specific behaviors are cleanly encapsulated.

### EOS Token Handling

The MLX provider needs special handling for EOS (End of Sequence) tokens to properly terminate text generation. Different model families use different EOS tokens:

- Qwen models: `["<|endoftext|>", "<|im_end|>", "</s>"]`
- Llama models: `["</s>", "<|endoftext|>"]`
- Mistral models: `["</s>"]`
- Phi models: `["<|endoftext|>", "</s>"]`
- Code models: `["<|endoftext|>", "</s>"]`

These tokens are added to the tokenizer using the `add_eos_token` method rather than passed as a parameter to the generate function.

### Temperature Handling

MLX-LM doesn't accept temperature directly as a parameter to the generate function. Instead, we need to create a sampler using `mlx_lm.sample_utils.make_sampler(temp=temperature)` and pass that sampler to the generate function.

### Repetition Handling

Some models, particularly Qwen models, exhibit repetitive output patterns. To address this, we apply a repetition penalty using the `make_logits_processors` function:

```python
repetition_penalty = 1.2  # Higher values penalize repetition more strongly
logits_processors = mlx_lm.sample_utils.make_logits_processors(
    repetition_penalty=repetition_penalty,
    repetition_context_size=64  # Look back at last 64 tokens
)
```

This helps prevent the model from getting stuck in repetition loops.

### System Prompts

Different model families have different formats for system prompts. The model configurations handle this automatically:

- Llama models: `<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]`
- Qwen models: `<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n`
- Code models: `# System: {system_prompt}\n\n{user_prompt}`
- Other models: Format depends on the model's chat template if available

### Chat Template Handling

Some models don't have chat templates defined. We've implemented robust error handling for system prompt formatting:

- First try to use the model's built-in chat template
- If that fails due to missing chat template, fall back to simple concatenation
- For code models, we use special formatting that adds system prompts as comments

### Model Architecture Detection

The `ModelConfigFactory` detects the correct configuration to use for each model in several ways:

1. Direct name matching (e.g., "qwen" in model name → QwenConfig)
2. Architecture hints in the name (e.g., "7b" → likely Llama-based)
3. Fallback to generic configuration for unknown architectures

This ensures that even models without specific configurations get reasonable defaults.

### Common Issues and Solutions

#### 1. Metal Function Errors

On Apple Silicon, you may encounter errors like:
```
Unable to load function sdpa_vector_float16_t_80_80
Function sdpa_vector_float16_t_80_80 was not found in the library
```

**Causes:**
- Model architecture incompatible with current MLX version
- Metal function implementation missing or incompatible with model tensor shapes

**Solutions:**
- Use models specifically created for MLX from the mlx-community organization
- Prefer 4-bit quantized models which tend to have fewer compatibility issues
- Avoid models converted from other formats without proper MLX-specific adjustments
- Try using a different MLX-compatible model from a similar architecture family

#### 2. Missing Chat Templates

Some models, particularly code models, don't have chat templates defined, causing errors when trying to format system prompts.

**Solutions:**
- Our implementation now gracefully handles these errors with a robust fallback mechanism
- For code models, system prompts are formatted as comments
- Generic concatenation is used for other models without templates

#### 3. Model Availability

Not all models have MLX-compatible versions available.

**Recommendations:**
- Use models from the `mlx-community` organization on Hugging Face
- Look for models with "MLX" or "mlx" in their name
- Check the model's "Files" tab for a `model.safetensors` file, which is the format MLX uses
- Convert models yourself using `mlx_lm.convert` if needed

## Vision Models

Vision models in MLX require special handling for the patch_size attribute, which may be missing in some models. We apply patches to set default values when needed. 