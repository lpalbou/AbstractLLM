# Ollama Context Size Fix

## Root Cause
The Ollama provider was using a default context size of 4096 tokens, while the qwen3-30b-a3b model supports 40960 tokens. This massive reduction in context was causing issues with:
- Tool definitions not fitting in the prompt
- Model behavior degrading with longer conversations
- System prompts being truncated

## Solution
Added `num_ctx` parameter to Ollama API requests to use the model's full context capacity.

### Changes Made

1. **Added `get_context_length()` function** in `architectures/detection.py`:
```python
def get_context_length(model_name: str) -> int:
    """Get the context length (input limit) for a model."""
    caps = get_model_capabilities(model_name)
    return caps.get("context_length", 4096)
```

2. **Updated Ollama provider** to set `num_ctx` in both chat and generate endpoints:
```python
# Get context length for the model
from abstractllm.architectures.detection import get_context_length
context_length = get_context_length(model)
logger.info(f"Setting context size to {context_length} tokens for {model}")

# Base request structure
request_data = {
    "model": model,
    "stream": stream,
    "options": {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_ctx": context_length  # Set context size to model's full capacity
    }
}
```

## Benefits
- Models now use their full trained context window
- Tool definitions and system prompts fit properly
- Better performance on longer conversations
- Improved tool execution reliability

## Testing
Run `test_ollama_context.py` to verify:
1. The correct context size is being detected
2. Ollama receives the `num_ctx` parameter
3. Tool calling works better with the larger context

## Additional Fixes Applied Earlier
1. **Consistent return types** - Always return `GenerateResponse` objects
2. **Fallback tool parsing** - Try prompted parsing if native fails
3. **Enhanced logging** - Better visibility into tool processing
4. **Session bug fix** - Handle None system prompts gracefully

## Complete Fix Summary
The Ollama provider issues were caused by:
1. **Context size limitation** (main issue - now fixed)
2. **Inconsistent return types** (fixed)
3. **Tool parsing fallback** (fixed)
4. **Model behavior** (improved with larger context)

The context size fix should dramatically improve the model's ability to handle tools and longer conversations.