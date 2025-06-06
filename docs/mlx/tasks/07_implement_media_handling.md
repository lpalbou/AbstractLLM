# Task 7: Implement Vision and Media Handling

## Status
**Implemented:** Yes
**Completion Date:** Current date

## Description
Implement comprehensive vision and media handling capabilities in the MLX provider, with support for different vision model architectures and robust image processing.

## Requirements
1. Implement model-specific configurations for different vision architectures
2. Add robust image processing with aspect ratio preservation
3. Implement memory safety checks for image processing
4. Support multiple image handling and proper prompt formatting

## Implementation Details

### 1. Model-Specific Configurations

```python
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

### 2. Image Processing Implementation

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
        
        # Check memory requirements before processing
        self._check_memory_requirements(config["image_size"], 1)
        
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
```

### 3. Aspect Ratio Preservation

```python
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

### 4. Memory Safety

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

### 5. Prompt Formatting

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

## References
- See `docs/mlx/vision-upgrade.md` for detailed implementation plan
- See `docs/mlx/deepsearch-mlx-vlm.md` for MLX-VLM insights
- See `tests/test_mlx_vision.py` for vision testing implementation

## Testing
1. Test image processing with various input types and sizes
2. Test aspect ratio preservation
3. Test memory requirement checks
4. Test prompt formatting for different models
5. Test error handling for various failure cases

## Success Criteria
1. Images are properly processed according to model requirements
2. Aspect ratios are preserved with appropriate padding
3. Memory errors are caught and handled gracefully
4. Prompts are correctly formatted for each model type
5. All vision-related tests pass 