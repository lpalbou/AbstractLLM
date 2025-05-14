# Task 17: Implement Model Caching Utilities

## Description
Implement additional utility methods for managing cached models in the MLX provider with robust error handling.

## Requirements
1. Implement methods to clean up model cache
2. Add methods to list available MLX models
3. Add methods to check cache size
4. Add utility to convert PyTorch models to MLX format

## Implementation Details

Add the following methods to the MLX provider:

```python
@staticmethod
def get_cache_info(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about the model cache.
    
    This method scans the Hugging Face cache directory for MLX models
    and returns information about their size, last access time, etc.
    
    Args:
        cache_dir: Custom cache directory path (uses default if None)
        
    Returns:
        Dictionary with cache information containing:
        - models: List of model information dicts
        - total_size: Total cache size in bytes
        - total_size_human: Human-readable total size
        - model_count: Number of cached models
        - cache_dir: Path to the cache directory
        
    Raises:
        ImportError: If huggingface_hub is not available
        RuntimeError: If there's an error scanning the cache
    """
    try:
        from huggingface_hub import scan_cache_dir
        import os
        
        try:
            import humanize
            has_humanize = True
        except ImportError:
            has_humanize = False
            logger.warning("humanize package not available, sizes will be in bytes")
        
        # Use default HF cache if not specified
        if cache_dir is None:
            cache_dir = "~/.cache/huggingface/hub"
            
        if cache_dir and '~' in cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
        
        logger.info(f"Scanning cache directory for MLX models: {cache_dir}")
            
        # Scan the cache
        try:
            cache_info = scan_cache_dir(cache_dir)
        except Exception as e:
            logger.error(f"Error scanning cache directory: {e}")
            raise RuntimeError(f"Failed to scan cache directory '{cache_dir}': {str(e)}")
        
        # Filter to only include MLX models
        mlx_models = []
        total_size = 0
        
        for repo in cache_info.repos:
            # Look for MLX models by name or content
            if "mlx" in repo.repo_id.lower():
                model_info = {
                    "name": repo.repo_id,
                    "size": repo.size_on_disk,
                    "last_used": repo.last_accessed,
                }
                
                if has_humanize:
                    model_info["human_size"] = humanize.naturalsize(repo.size_on_disk)
                
                mlx_models.append(model_info)
                total_size += repo.size_on_disk
        
        result = {
            "models": mlx_models,
            "total_size": total_size,
            "model_count": len(mlx_models),
            "cache_dir": cache_dir
        }
        
        if has_humanize:
            result["total_size_human"] = humanize.naturalsize(total_size)
        
        logger.info(f"Found {len(mlx_models)} MLX models with total size: {result.get('total_size_human', f'{total_size} bytes')}")
            
        return result
    except ImportError as e:
        logger.warning(f"huggingface_hub not available for cache scanning: {e}")
        raise ImportError("huggingface_hub is required for cache management. Install with: pip install huggingface_hub")

@staticmethod
def prune_cache(max_size_gb: float = 10.0, cache_dir: Optional[str] = None) -> List[str]:
    """
    Prune the model cache to stay under the specified size.
    
    This method removes the least recently used MLX models from the
    Hugging Face cache until the total size is below the specified limit.
    
    Args:
        max_size_gb: Maximum cache size in GB
        cache_dir: Custom cache directory path (uses default if None)
        
    Returns:
        List of deleted models
        
    Raises:
        ImportError: If huggingface_hub is not available
        ValueError: If max_size_gb is invalid
        RuntimeError: If there's an error pruning the cache
    """
    if max_size_gb <= 0:
        raise ValueError("max_size_gb must be positive")
        
    try:
        from huggingface_hub import scan_cache_dir, DeleteCacheStrategy
        import os
        
        # Use default HF cache if not specified
        if cache_dir is None:
            cache_dir = "~/.cache/huggingface/hub"
            
        if cache_dir and '~' in cache_dir:
            cache_dir = os.path.expanduser(cache_dir)
        
        logger.info(f"Checking cache size against limit of {max_size_gb} GB")
            
        # Scan the cache
        try:
            cache_info = scan_cache_dir(cache_dir)
        except Exception as e:
            logger.error(f"Error scanning cache directory: {e}")
            raise RuntimeError(f"Failed to scan cache directory '{cache_dir}': {str(e)}")
        
        # Get MLX models
        mlx_repos = [
            repo for repo in cache_info.repos 
            if "mlx" in repo.repo_id.lower()
        ]
        
        # Calculate current size
        current_size = sum(repo.size_on_disk for repo in mlx_repos)
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        # If current size is under the limit, do nothing
        if current_size <= max_size_bytes:
            logger.info(f"Cache size {current_size/(1024*1024*1024):.2f} GB is under limit of {max_size_gb} GB")
            return []
        
        # Sort by last accessed (oldest first)
        mlx_repos.sort(key=lambda repo: repo.last_accessed)
        
        logger.info(f"Current cache size: {current_size/(1024*1024*1024):.2f} GB, pruning to {max_size_gb} GB")
        
        # Delete oldest repos until we're under the limit
        deleted_models = []
        
        for repo in mlx_repos:
            if current_size <= max_size_bytes:
                break
                
            try:
                # Delete repo from cache
                strategy = DeleteCacheStrategy(batch_size=1, commit_every=1)
                cache_info.delete_revisions(
                    [revision.commit_hash for revision in repo.revisions],
                    strategy=strategy
                )
                
                current_size -= repo.size_on_disk
                deleted_models.append(repo.repo_id)
                
                logger.info(f"Deleted {repo.repo_id} from cache " 
                           f"({repo.size_on_disk/(1024*1024*1024):.2f} GB)")
            except Exception as e:
                logger.warning(f"Failed to delete {repo.repo_id}: {e}")
        
        logger.info(f"Pruned cache to {current_size/(1024*1024*1024):.2f} GB " 
                   f"(limit: {max_size_gb} GB)")
        return deleted_models
    except ImportError as e:
        logger.warning(f"huggingface_hub not available for cache pruning: {e}")
        raise ImportError("huggingface_hub is required for cache management. Install with: pip install huggingface_hub")

@staticmethod
def list_available_mlx_models(limit: int = 50, only_installable: bool = True) -> List[Dict[str, Any]]:
    """
    List available MLX models from the Hugging Face Hub.
    
    Args:
        limit: Maximum number of models to return
        only_installable: Only return models that can be directly installed
        
    Returns:
        List of available MLX models with metadata
        
    Raises:
        ImportError: If huggingface_hub is not available
    """
    try:
        from huggingface_hub import HfApi
        
        # Parameter validation
        if limit <= 0:
            logger.warning("Invalid limit value, using default of 50")
            limit = 50
        
        logger.info(f"Searching for MLX models on HuggingFace Hub (limit: {limit})")
        
        # Initialize HF API
        api = HfApi()
        
        try:
            # Search for MLX models
            models = api.list_models(
                search="mlx",
                filter="mlx",
                sort="downloads",
                direction=-1,
                limit=limit
            )
            
            # Format results
            result = []
            for model in models:
                # Skip models without files if only_installable is True
                if only_installable and not hasattr(model, 'siblings') or not model.siblings:
                    continue
                    
                model_info = {
                    "id": model.id,
                    "downloads": model.downloads,
                    "tags": model.tags if hasattr(model, 'tags') else [],
                    "pipeline_tag": model.pipeline_tag if hasattr(model, 'pipeline_tag') else None,
                }
                
                if hasattr(model, 'last_modified') and model.last_modified:
                    model_info["last_modified"] = model.last_modified
                    
                result.append(model_info)
            
            logger.info(f"Found {len(result)} MLX models on HuggingFace Hub")
                
            return result
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise RuntimeError(f"Failed to list available MLX models: {str(e)}")
    except ImportError as e:
        logger.warning(f"huggingface_hub not available for model listing: {e}")
        raise ImportError("huggingface_hub is required for model listing. Install with: pip install huggingface_hub")

@staticmethod
def convert_to_mlx(model_id: str, output_dir: Optional[str] = None, quantize: bool = True) -> str:
    """
    Convert a PyTorch model to MLX format.
    
    This method runs the MLX conversion script to convert Hugging Face models
    to MLX format for efficient inference on Apple Silicon.
    
    Args:
        model_id: Hugging Face model ID
        output_dir: Output directory (defaults to ~/.cache/mlx-models)
        quantize: Whether to quantize the model (reduces size and memory usage)
        
    Returns:
        Path to the converted model
        
    Raises:
        ImportError: If mlx_lm is not available
        ValueError: If model_id is invalid
        RuntimeError: If conversion fails
    """
    if not model_id or not isinstance(model_id, str):
        raise ValueError("model_id must be a valid Hugging Face model ID")
        
    try:
        import subprocess
        import os
        import tempfile
        import sys
        
        # Check if mlx_lm is installed
        try:
            import mlx_lm
        except ImportError as e:
            raise ImportError(f"mlx_lm is required for model conversion: {e}")
        
        # Default output directory
        if output_dir is None:
            output_dir = os.path.expanduser("~/.cache/mlx-models")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output model directory
        model_name = model_id.split("/")[-1]
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Run the conversion script
        logger.info(f"Converting {model_id} to MLX format{' with quantization' if quantize else ''}")
        logger.info(f"Output directory: {model_dir}")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as log_file:
            try:
                command = [
                    sys.executable, "-m", "mlx_lm.convert", 
                    "--hf-path", model_id,
                    "--mlx-path", model_dir
                ]
                
                if quantize:
                    command.append("--quantize")
                
                logger.debug(f"Running conversion command: {' '.join(command)}")
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Capture output and write to log file
                for line in process.stdout:
                    log_file.write(line)
                    if line.strip():
                        logger.debug(f"Conversion: {line.strip()}")
                
                # Wait for the process to complete
                process.wait()
                
                # Check if conversion was successful
                if process.returncode != 0:
                    with open(log_file.name, 'r') as f:
                        log_content = f.read()
                    logger.error(f"Model conversion failed with code {process.returncode}")
                    raise RuntimeError(f"Model conversion failed with code {process.returncode}:\n{log_content}")
                
                logger.info(f"Successfully converted {model_id} to MLX format at {model_dir}")
                return model_dir
            finally:
                # Clean up log file
                try:
                    os.unlink(log_file.name)
                except:
                    pass
    except Exception as e:
        if not isinstance(e, (ImportError, ValueError, RuntimeError)):
            logger.error(f"Unexpected error converting model {model_id}: {e}")
            raise RuntimeError(f"Failed to convert model: {str(e)}")
        raise
```

## References
- See huggingface_hub documentation for cache management
- Reference MLX-LM documentation for model conversion: https://github.com/ml-explore/mlx-lm
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_integration_architecture.md` for caching strategy

## Testing
1. Test listing available MLX models with various parameters
2. Test cache information retrieval and error handling
3. Test cache pruning with different size limits and edge cases
4. Test model conversion from PyTorch to MLX with and without quantization 