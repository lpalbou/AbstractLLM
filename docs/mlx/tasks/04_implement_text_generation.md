# Task 4: Implement Basic Text Generation

## Description
Implement the core text generation functionality for the MLX provider, allowing it to generate responses from text prompts.

## Requirements
1. Implement the `generate()` method with support for text prompts
2. Handle system prompts correctly
3. Implement proper token counting
4. Return formatted `GenerateResponse` objects

## Implementation Details

```python
def generate(self, 
            prompt: str, 
            system_prompt: Optional[str] = None, 
            files: Optional[List[Union[str, Path]]] = None,
            stream: bool = False, 
            tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
            **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
    """Generate a response using the MLX model."""
    # Load model if not already loaded
    if not self._is_loaded:
        self.load_model()
    
    # Process system prompt if provided
    formatted_prompt = prompt
    if system_prompt:
        # Use model's chat template if available
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            # Construct messages in the expected format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            try:
                # Try to use HF's template application
                from transformers import AutoTokenizer
                formatted_prompt = AutoTokenizer.apply_chat_template(
                    messages, 
                    chat_template=self._tokenizer.chat_template,
                    tokenize=False
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}")
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            # Simple concatenation fallback
            formatted_prompt = f"{system_prompt}\n\n{prompt}"
    
    # Process files if provided
    if files and len(files) > 0:
        formatted_prompt = self._process_files(formatted_prompt, files)
    
    # Tools are not supported
    if tools:
        raise UnsupportedFeatureError(
            "tool_use",
            "MLX provider does not support tool use or function calling",
            provider="mlx"
        )
    
    # Get generation parameters
    temperature = kwargs.get("temperature", 
                           self.config_manager.get_param(ModelParameter.TEMPERATURE))
    max_tokens = kwargs.get("max_tokens", 
                          self.config_manager.get_param(ModelParameter.MAX_TOKENS))
    top_p = kwargs.get("top_p", 
                     self.config_manager.get_param(ModelParameter.TOP_P))
    
    # Validate parameters
    if temperature is not None and (temperature < 0 or temperature > 2):
        logger.warning(f"Temperature {temperature} out of recommended range [0, 2], clamping")
        temperature = max(0, min(temperature, 2))
    
    if max_tokens is not None and max_tokens <= 0:
        logger.warning(f"Invalid max_tokens {max_tokens}, using default")
        max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS)
    
    if top_p is not None and (top_p <= 0 or top_p > 1):
        logger.warning(f"Top_p {top_p} out of valid range (0, 1], clamping")
        top_p = max(0.001, min(top_p, 1.0))
    
    # Log request parameters
    model_name = self.config_manager.get_param(ModelParameter.MODEL)
    log_request("mlx", prompt, {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "has_system_prompt": system_prompt is not None,
        "stream": stream,
        "has_files": bool(files)
    })
    
    # Encode prompt
    try:
        prompt_tokens = self._tokenizer.encode(formatted_prompt)
    except Exception as e:
        logger.error(f"Failed to encode prompt: {e}")
        raise RuntimeError(f"Failed to encode prompt: {str(e)}")
    
    # Import MLX-LM generation utilities
    from mlx_lm.utils import generate
    
    # Handle streaming vs non-streaming
    if stream:
        # Streaming will be implemented in a separate task
        pass
    else:
        # Generate text (non-streaming)
        try:
            output = generate(
                self._model,
                self._tokenizer,
                prompt=prompt_tokens,
                temp=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            
            # Create response
            completion_tokens = len(self._tokenizer.encode(output)) if hasattr(self._tokenizer, "encode") else len(output.split())
            
            # Log the response
            log_response("mlx", output)
            
            return GenerateResponse(
                text=output,
                model=self.config_manager.get_param(ModelParameter.MODEL),
                prompt_tokens=len(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_tokens) + completion_tokens
            )
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise RuntimeError(f"Error generating text: {str(e)}")
```

## References
- See MLX-LM documentation for generation: https://github.com/ml-explore/mlx-lm
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_usage_examples.md` for usage patterns

## Testing
1. Test basic text generation with different models
2. Test with and without system prompts
3. Verify the token counting is accurate
4. Verify that error handling works properly 