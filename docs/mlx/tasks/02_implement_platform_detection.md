# Task 2: Implement Apple Silicon Detection

## Description
Add platform detection to check if the code is running on Apple Silicon hardware, and implement appropriate error handling.

## Requirements
1. Implement the `_check_apple_silicon()` method in the MLXProvider class
2. Use AbstractLLM's exception system to report errors when running on unsupported hardware
3. Call this method during initialization to prevent usage on non-Apple-Silicon devices

## Implementation Details

```python
def _check_apple_silicon(self) -> None:
    """Check if running on Apple Silicon."""
    is_macos = platform.system().lower() == "darwin"
    if not is_macos:
        logger.warning(f"MLX requires macOS, current platform: {platform.system()}")
        raise UnsupportedFeatureError(
            feature="mlx",
            message="MLX provider is only available on macOS with Apple Silicon",
            provider="mlx"
        )
    
    # Check processor architecture
    is_arm = platform.processor() == "arm"
    if not is_arm:
        logger.warning(f"MLX requires Apple Silicon, current processor: {platform.processor()}")
        raise UnsupportedFeatureError(
            feature="mlx",
            message="MLX provider requires Apple Silicon (M1/M2/M3) hardware",
            provider="mlx"
        )
    
    logger.info(f"Platform check successful: macOS with Apple Silicon detected")
```

Then, update the `__init__` method to include the platform check after checking for required dependencies:

```python
def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
    """Initialize the MLX provider."""
    super().__init__(config)
    
    # Check for MLX availability
    if not MLX_AVAILABLE:
        logger.error("MLX package not found")
        raise ImportError("MLX is required for MLXProvider. Install with: pip install mlx")
    
    if not MLXLM_AVAILABLE:
        logger.error("MLX-LM package not found")
        raise ImportError("MLX-LM is required for MLXProvider. Install with: pip install mlx-lm")
    
    # Check if running on Apple Silicon
    self._check_apple_silicon()
    
    # Rest of the initialization code...
```

## References
- See `abstractllm/exceptions.py` for `UnsupportedFeatureError`
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- See architecture document: `docs/mlx/mlx_integration_architecture.md`

## Testing
To test this implementation:
1. Run the code on different platforms (if available)
2. Mock the platform module to simulate different environments
3. Verify correct error messages are raised on unsupported platforms 