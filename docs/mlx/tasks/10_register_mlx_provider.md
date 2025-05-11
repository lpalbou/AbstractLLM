# Task 10: Register MLX Provider in Factory

## Description
Register the MLX provider with the AbstractLLM factory system to make it available through the `create_llm()` function.

## Requirements
1. Create a registration function in the appropriate factory module
2. Conditionally register the provider based on platform and dependency availability
3. Ensure the provider is discoverable using "mlx" as the provider name
4. Handle import errors gracefully with helpful messages

## Implementation Details

There are two possible approaches for registering the provider. The recommended approach is to use the registry system:

### Option 1: Update registry.py (Recommended)

```python
# In abstractllm/providers/registry.py or similar file

def register_mlx_provider():
    """Register the MLX provider if available."""
    import logging
    logger = logging.getLogger("abstractllm.providers.registry")
    
    try:
        # Check platform first before importing anything
        import platform
        is_macos = platform.system().lower() == "darwin"
        is_arm = platform.processor() == "arm"
        
        # Log availability status
        if not is_macos or not is_arm:
            logger.info(
                "MLX provider not registered: requires macOS with Apple Silicon. "
                f"Current platform: {platform.system()} {platform.processor()}"
            )
            return False
        
        # Import MLX dependencies
        try:
            import mlx.core
            logger.debug("MLX package is available")
        except ImportError as e:
            logger.info(f"MLX provider not registered: mlx package not available - {e}")
            return False
            
        try:
            import mlx_lm
            logger.debug("MLX-LM package is available")
        except ImportError as e:
            logger.info(f"MLX provider not registered: mlx-lm package not available - {e}")
            return False
        
        # Register the provider
        try:
            from abstractllm.providers.registry import register_provider
            register_provider("mlx", "abstractllm.providers.mlx_provider", "MLXProvider")
            logger.info("MLX provider successfully registered for Apple Silicon")
            return True
        except Exception as e:
            logger.error(f"Failed to register MLX provider through registry system: {e}")
            return False
    except Exception as e:
        logger.warning(f"Failed to register MLX provider: {e}")
        return False
```

Then include a call to this function during AbstractLLM initialization:

```python
# In abstractllm/__init__.py or a suitable initialization module
from abstractllm.providers.registry import register_mlx_provider

# Register MLX provider
register_mlx_provider()
```

### Option 2: Update factory.py (Alternative)

```python
# In abstractllm/factory.py or similar location

def register_providers():
    """Register all available providers."""
    import logging
    logger = logging.getLogger("abstractllm.factory")
    
    # Other provider registrations...
    
    # Register MLX provider if available
    try:
        # Check if on Apple Silicon
        import platform
        is_macos = platform.system().lower() == "darwin"
        is_arm = platform.processor() == "arm"
        
        if not is_macos or not is_arm:
            logger.debug("Skipping MLX provider: requires macOS with Apple Silicon")
            return
        
        # Check if MLX is available
        try:
            import mlx.core
            import mlx_lm
            
            # If we got here, dependencies are available
            from abstractllm.providers.mlx_provider import MLXProvider
            from abstractllm.providers.registry import register_provider
            
            register_provider("mlx", MLXProvider)
            logger.info("MLX provider registered for Apple Silicon")
        except ImportError as e:
            logger.debug(f"MLX provider not registered: {e}")
    except Exception as e:
        logger.warning(f"Error while registering MLX provider: {e}")
```

Additionally, update the provider's `__init__.py` file to ensure it's imported in a platform-aware manner:

```python
# In abstractllm/providers/__init__.py

# Standard providers that should always be available
from abstractllm.providers.openai import OpenAIProvider
from abstractllm.providers.anthropic import AnthropicProvider
# ... other standard providers

# Conditionally import MLX provider only on compatible platforms
import platform
is_macos = platform.system().lower() == "darwin"
is_arm = platform.processor() == "arm"

if is_macos and is_arm:
    try:
        from abstractllm.providers.mlx_provider import MLXProvider
    except ImportError:
        # MLX is not available, log at debug level
        import logging
        logging.getLogger("abstractllm.providers").debug(
            "MLX provider not imported: dependencies not available"
        )
```

## References
- See AbstractLLM's existing provider registration mechanism
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See `docs/mlx/mlx_integration_architecture.md` for architectural guidance

## Testing
1. Test provider registration on Apple Silicon by calling `create_llm("mlx")`
2. Test error messages on non-Apple platforms are clear and helpful
3. Test error messages when dependencies are missing include installation instructions
4. Verify the provider appears when listing available providers in AbstractLLM 