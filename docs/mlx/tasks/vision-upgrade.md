# MLX Vision Capabilities Upgrade

## Overview

Based on deep analysis of MLX-VLM and our current implementation, this task outlines necessary improvements to make vision capabilities in our MLX provider more robust, accurate, and reliable.

## Key Improvements Needed

1. **Model-Specific Configurations**
2. **Enhanced Image Processing**
3. **Memory Safety and Error Handling**
4. **Prompt Formatting Improvements**

## Implementation Steps

### 1. Add Model-Specific Configurations

Create a new configuration system for vision models:

```python
# Add at top of mlx_provider.py
MODEL_CONFIGS = {
    "llava": {
        "image_size": (336, 336),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "prompt_format": "<image>\n{prompt}"
    },
    "qwen-vl": {
        "image_size": (448, 448),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "prompt_format": "<img>{prompt}"
    },
    "kimi-vl": {
        "image_size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "prompt_format": "<|image|>{prompt}"
    },
    "default": {
        "image_size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "prompt_format": "<image>{prompt}"
    }
}
```

### 2. Improve Model Type Detection

Update the model type detection to be more robust:

```python
def _determine_model_type(self, model_name: str) -> str:
    """Determine the specific vision model type and load its configuration."""
    model_name_lower = model_name.lower()
    
    # Try to load model config from HF if available
    try:
        config = load_config(model_name, trust_remote_code=True)
        if "model_type" in config:
            model_type = config["model_type"]
            if model_type in MODEL_CONFIGS:
                return model_type
    except Exception as e:
        logger.warning(f"Could not load model config: {e}")
    
    # Fallback to name-based detection
    for model_type in MODEL_CONFIGS:
        if model_type in model_name_lower:
            return model_type
            
    logger.warning(f"Unknown model type for {model_name}, using default configuration")
    return "default"
```

### 3. Enhance Image Processing

Implement robust image processing with proper aspect ratio handling:

```python
def _process_image(self, image_input: ImageInput) -> mx.array:
    """Process image input into MLX format with robust error handling."""
    try:
        # Get image content
        image_content = image_input.get_content()
        
        # Convert to PIL Image with robust error handling
        try:
            if isinstance(image_content, Image.Image):
                image = image_content
            elif isinstance(image_content, (str, Path)):
                image = Image.open(image_content)
            else:
                from io import BytesIO
                image = Image.open(BytesIO(image_content))
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {str(e)}", provider="mlx")

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Get model-specific configuration
        config = MODEL_CONFIGS[self._model_type]
        
        # Resize with proper aspect ratio handling
        target_size = config["image_size"]
        image = self._resize_with_aspect_ratio(image, target_size)
        
        # Convert to numpy array and normalize
        try:
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Normalize using model-specific values
            mean = np.array(config["mean"], dtype=np.float32)
            std = np.array(config["std"], dtype=np.float32)
            image_array = (image_array - mean) / std
            
            # Convert to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            return mx.array(image_array)
            
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to process image array: {str(e)}", 
                provider="mlx"
            )
            
    except Exception as e:
        if isinstance(e, ImageProcessingError):
            raise
        raise ImageProcessingError(f"Image processing failed: {str(e)}", provider="mlx")

def _resize_with_aspect_ratio(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize image maintaining aspect ratio and padding if necessary."""
    target_w, target_h = target_size
    orig_w, orig_h = image.size
    
    # Calculate aspect ratios
    target_aspect = target_w / target_h
    orig_aspect = orig_w / orig_h
    
    if orig_aspect > target_aspect:
        # Image is wider than target
        new_w = target_w
        new_h = int(target_w / orig_aspect)
    else:
        # Image is taller than target
        new_h = target_h
        new_w = int(target_h * orig_aspect)
        
    # Resize maintaining aspect ratio
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create new image with padding
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_image.paste(image, (paste_x, paste_y))
    
    return new_image
```

### 4. Add Memory Safety Checks

Implement memory requirement validation:

```python
def _check_memory_requirements(self, image_size: Tuple[int, int], num_images: int) -> None:
    """Check if processing these images might exceed memory limits."""
    try:
        # Calculate approximate memory requirement for image processing
        pixels = image_size[0] * image_size[1] * 3  # RGB
        bytes_per_image = pixels * 4  # float32
        total_bytes = bytes_per_image * num_images
        
        # Get system memory info
        import psutil
        mem = psutil.virtual_memory()
        available = mem.available
        
        # Check against Metal buffer size limit
        METAL_BUFFER_LIMIT = 77309411328  # ~77GB
        
        if total_bytes > METAL_BUFFER_LIMIT:
            raise MemoryError(
                f"Processing {num_images} images of size {image_size} would require "
                f"{total_bytes/1e9:.2f}GB, exceeding Metal buffer limit of {METAL_BUFFER_LIMIT/1e9:.2f}GB"
            )
            
        # Also check against available system memory (with safety margin)
        if total_bytes * 2 > available:  # 2x for safety
            logger.warning(
                f"Processing these images may use {total_bytes/1e9:.2f}GB of memory "
                f"with only {available/1e9:.2f}GB available"
            )
            
    except Exception as e:
        logger.warning(f"Could not check memory requirements: {e}")
```

### 5. Improve Prompt Formatting

Update prompt formatting to handle multiple images and model-specific formats:

```python
def _format_prompt(self, prompt: str, num_images: int) -> str:
    """Format prompt with image tokens based on model configuration."""
    if num_images == 0:
        return prompt
        
    config = MODEL_CONFIGS[self._model_type]
    base_format = config["prompt_format"]
    
    if num_images == 1:
        return base_format.format(prompt=prompt)
        
    # For multiple images, add numbered tokens if supported
    if self._model_type in ["idefics", "qwen-vl"]:
        formatted = prompt
        for i in range(num_images):
            formatted = f"<image{i+1}>{formatted}"
        return formatted
        
    # Default: just repeat the image token
    formatted = prompt
    for _ in range(num_images):
        formatted = base_format.format(prompt=formatted)
    return formatted
```

## Testing Requirements

1. **Unit Tests**
   - Test image processing with various input types (PIL Image, file path, bytes)
   - Test aspect ratio handling
   - Test memory requirement checks
   - Test prompt formatting for different models
   - Test error handling

2. **Integration Tests**
   - Test with actual MLX vision models
   - Test with multiple images
   - Test memory limits
   - Test error cases

3. **Model-Specific Tests**
   - Test each supported model type
   - Verify correct image sizes
   - Verify prompt formatting

## Documentation Updates Needed

1. Update MLX provider documentation with:
   - Supported vision models and their configurations
   - Image processing details
   - Memory requirements and limitations
   - Error handling and troubleshooting

2. Add examples for:
   - Basic vision model usage
   - Handling multiple images
   - Error handling
   - Memory management

## Success Criteria

1. All vision models in `mlx-community` work correctly with appropriate image sizes
2. Memory errors are caught and handled gracefully
3. Image aspect ratios are preserved appropriately
4. All tests pass
5. Documentation is complete and accurate

## Dependencies

1. MLX-VLM library
2. PIL/Pillow for image processing
3. NumPy for array operations
4. psutil for memory checks

## Timeline

1. Implementation: 2-3 days
2. Testing: 1-2 days
3. Documentation: 1 day
4. Review and refinement: 1 day

Total estimated time: 5-7 days

## Notes

- This implementation favors robustness and reliability over speed
- Memory safety is a primary concern
- Error handling is comprehensive
- Model-specific configurations ensure accurate processing
- The implementation is designed to be maintainable and extensible 