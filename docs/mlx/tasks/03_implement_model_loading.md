# Task 3: Implement Model Loading

## Description
Implement the model loading functionality for the MLX provider, which will load models from the Hugging Face Hub using MLX-LM's utilities.

## Requirements
1. Create the `load_model()` method in the MLX provider class
2. Implement in-memory caching of models for quick switching
3. Use the HuggingFace cache infrastructure for downloading models
4. Add basic vision model detection for future compatibility

## Implementation Details

```python
def load_model(self) -> None:
    """
    Load the MLX model and tokenizer.
    
    This method will check the cache first before loading from the source.
    """
    model_name = self.config_manager.get_param(ModelParameter.MODEL)
    cache_dir = self.config_manager.get_param("cache_dir")  # Uses HF default if None
    
    # Check in-memory cache first
    if model_name in self._model_cache:
        logger.info(f"Loading model {model_name} from in-memory cache")
        self._model, self._tokenizer, _ = self._model_cache[model_name]
        # Update last access time
        self._model_cache[model_name] = (self._model, self._tokenizer, time.time())
        self._is_loaded = True
        return
    
    # If not in memory cache, load from disk/HF
    logger.info(f"Loading model {model_name}")
    
    try:
        # Import MLX-LM utilities
        from mlx_lm.utils import load
        
        # Check if this is a vision model
        if any(x in model_name.lower() for x in ["llava", "clip", "vision"]):
            self._is_vision_model = True
            logger.info(f"Detected vision capabilities in model {model_name}")
        
        # Log loading parameters
        logger.debug(f"Loading MLX model with parameters: model={model_name}, cache_dir={cache_dir}")
        
        # Load the model using MLX-LM
        self._model, self._tokenizer = load(model_name, cache_dir=cache_dir)
        self._is_loaded = True
        
        # Add to in-memory cache
        self._update_model_cache(model_name)
        
        logger.info(f"Successfully loaded model {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Failed to load MLX model: {str(e)}")

def _update_model_cache(self, model_name: str) -> None:
    """Update the model cache with the current model."""
    self._model_cache[model_name] = (self._model, self._tokenizer, time.time())
    
    # Prune cache if needed
    if len(self._model_cache) > self._max_cached_models:
        # Find oldest model by last access time
        oldest_key = min(self._model_cache.keys(), 
                        key=lambda k: self._model_cache[k][2])
        logger.info(f"Removing {oldest_key} from model cache")
        del self._model_cache[oldest_key]
```

## References
- See MLX-LM documentation: https://github.com/ml-explore/mlx-lm
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_integration_architecture.md` for caching strategy

## Testing
1. Test loading models from the Hugging Face Hub
2. Test switching between different models to verify caching works
3. Test loading from cache vs. new loads for performance comparison 