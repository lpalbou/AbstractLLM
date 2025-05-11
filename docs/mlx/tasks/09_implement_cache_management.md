# Task 9: Implement Cache Management

## Description
Implement model cache management utilities for the MLX provider to allow users to list and clear cached models.

## Requirements
1. Implement static methods for cache management: `list_cached_models()` and `clear_model_cache()`
2. Use Hugging Face's cache infrastructure for persistent storage
3. Provide proper model information including size and last access time
4. Allow selective clearing of specific models

## Implementation Details

```python
@staticmethod
def list_cached_models(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all models cached by this implementation.
    
    Args:
        cache_dir: Custom cache directory path (uses default if None)
        
    Returns:
        List of cached model information
    """
    # For now, just leverage HF's cache scanning
    try:
        from huggingface_hub import scan_cache_dir
        
        # Use default HF cache if not specified
        if cache_dir is None:
            cache_dir = "~/.cache/huggingface/hub"
            
        if cache_dir and '~' in cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
        
        logger.info(f"Scanning cache directory: {cache_dir}")
            
        # Scan the cache
        cache_info = scan_cache_dir(cache_dir)
        
        # Filter to only include MLX models
        mlx_models = []
        for repo in cache_info.repos:
            # Look for MLX models by name or content
            if "mlx" in repo.repo_id.lower():
                mlx_models.append({
                    "name": repo.repo_id,
                    "size": repo.size_on_disk,
                    "last_used": repo.last_accessed,
                    "implementation": "mlx"
                })
        
        logger.info(f"Found {len(mlx_models)} MLX models in cache")
        return mlx_models
    except ImportError:
        logger.warning("huggingface_hub not available for cache scanning")
        return []

@staticmethod
def clear_model_cache(model_name: Optional[str] = None) -> None:
    """
    Clear model cache for this implementation.
    
    Args:
        model_name: Specific model to clear (or all if None)
    """
    # Clear the in-memory cache
    if model_name:
        # Remove specific model from cache
        if model_name in MLXProvider._model_cache:
            del MLXProvider._model_cache[model_name]
            logger.info(f"Removed {model_name} from in-memory cache")
        else:
            logger.info(f"Model {model_name} not found in in-memory cache")
    else:
        # Clear all in-memory cache
        model_count = len(MLXProvider._model_cache)
        MLXProvider._model_cache.clear()
        logger.info(f"Cleared {model_count} models from in-memory cache")
```

## References
- See huggingface_hub documentation for cache management
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_integration_architecture.md` for caching strategy

## Testing
1. Test listing cached models after loading some models
2. Test clearing specific models and verifying they're removed
3. Test clearing all models and verifying the cache is empty
4. Test cache persistence when restarting the application 