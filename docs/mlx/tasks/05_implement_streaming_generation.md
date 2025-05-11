# Task 5: Implement Streaming Text Generation

## Description
Implement streaming text generation for the MLX provider to support real-time response delivery.

## Requirements
1. Implement the `_generate_stream()` method to generate tokens one by one
2. Return a generator that yields `GenerateResponse` objects
3. Properly detect end-of-sequence conditions
4. Update the `generate()` method to use this when `stream=True`

## Implementation Details

```python
def _generate_stream(self, 
                   prompt_tokens, 
                   temperature: float, 
                   max_tokens: int, 
                   top_p: float) -> Generator[GenerateResponse, None, None]:
    """Generate a streaming response."""
    import mlx.core as mx
    from mlx_lm.utils import generate_step
    
    # Convert to MLX array if not already
    if not isinstance(prompt_tokens, mx.array):
        prompt_tokens = mx.array(prompt_tokens)
    
    # Initial state
    tokens = prompt_tokens
    finish_reason = None
    current_text = ""
    
    # Generate tokens one by one
    for _ in range(max_tokens):
        next_token, _ = generate_step(
            self._model,
            tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add token to sequence
        tokens = mx.concatenate([tokens, next_token[None]])
        
        # Convert to text
        current_text = self._tokenizer.decode(tokens.tolist()[len(prompt_tokens):])
        
        # Check for EOS token
        if hasattr(self._tokenizer, "eos_token") and self._tokenizer.eos_token in current_text:
            current_text = current_text.replace(self._tokenizer.eos_token, "")
            finish_reason = "stop"
        
        # Create response chunk
        yield GenerateResponse(
            text=current_text,
            model=self.config_manager.get_param(ModelParameter.MODEL),
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(tokens) - len(prompt_tokens),
            total_tokens=len(tokens),
            finish_reason=finish_reason
        )
        
        # Stop if we reached the end
        if finish_reason:
            break
    
    # Log the final response for streaming generation
    logger.debug(f"Streaming generation completed: {len(tokens) - len(prompt_tokens)} tokens generated")
    log_response("mlx", current_text)
```

Then, update the `generate()` method to use this for streaming:

```python
# In the generate method where streaming is handled:
if stream:
    return self._generate_stream(
        prompt_tokens, 
        temperature, 
        max_tokens, 
        top_p
    )
```

## References
- See MLX-LM documentation for `generate_step`: https://github.com/ml-explore/mlx-lm
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_usage_examples.md` for streaming examples

## Testing
1. Test streaming generation with different models
2. Verify that tokens are delivered in real-time
3. Test that the generator stops when the model outputs an EOS token
4. Compare the final streamed output with a non-streamed output 