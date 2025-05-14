# Task 17: Error Handling

## Description
Implement comprehensive error handling for the MLX provider, with special focus on vision-related errors.

## Requirements
1. Handle image processing errors
2. Handle memory-related errors
3. Handle model-specific errors
4. Provide detailed error messages
5. Implement error recovery strategies

## Implementation Details

### Error Classes

Create `abstractllm/exceptions/vision.py`:

```python
"""Vision-specific exceptions."""

from typing import Optional, Union, List
from pathlib import Path
from PIL import Image

class VisionError(Exception):
    """Base class for vision-related errors."""
    pass

class ImageProcessingError(VisionError):
    """Error during image processing."""
    
    def __init__(
        self,
        message: str,
        image: Optional[Union[str, Path, Image.Image]] = None,
        cause: Optional[Exception] = None
    ):
        self.image = image
        self.cause = cause
        super().__init__(f"{message} - Image: {image}")

class ImageLoadError(ImageProcessingError):
    """Error loading image file."""
    
    def __init__(
        self,
        path: Union[str, Path],
        cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Failed to load image file: {path}",
            image=path,
            cause=cause
        )

class ImageFormatError(ImageProcessingError):
    """Error with image format."""
    
    def __init__(
        self,
        message: str,
        image: Union[str, Path, Image.Image],
        supported_formats: Optional[List[str]] = None
    ):
        self.supported_formats = supported_formats
        details = f" Supported formats: {supported_formats}" if supported_formats else ""
        super().__init__(f"{message}{details}", image=image)

class ImageSizeError(ImageProcessingError):
    """Error with image dimensions."""
    
    def __init__(
        self,
        image: Union[str, Path, Image.Image],
        current_size: tuple,
        max_size: Optional[tuple] = None,
        min_size: Optional[tuple] = None
    ):
        self.current_size = current_size
        self.max_size = max_size
        self.min_size = min_size
        
        message = f"Invalid image size: {current_size}"
        if max_size:
            message += f" (max: {max_size})"
        if min_size:
            message += f" (min: {min_size})"
        
        super().__init__(message, image=image)

class ImageMemoryError(ImageProcessingError):
    """Error when image processing would exceed memory limits."""
    
    def __init__(
        self,
        image: Union[str, Path, Image.Image],
        required_memory: int,
        available_memory: int
    ):
        self.required_memory = required_memory
        self.available_memory = available_memory
        
        message = (
            f"Insufficient memory for image processing. "
            f"Required: {required_memory / 1024**3:.1f}GB, "
            f"Available: {available_memory / 1024**3:.1f}GB"
        )
        super().__init__(message, image=image)

class BatchProcessingError(VisionError):
    """Error processing multiple images."""
    
    def __init__(
        self,
        message: str,
        errors: Optional[List[Exception]] = None
    ):
        self.errors = errors or []
        details = "\n".join(str(e) for e in self.errors)
        super().__init__(f"{message}\nDetails:\n{details}")

class ModelVisionError(VisionError):
    """Error related to vision model capabilities."""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        model_type: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_type = model_type
        super().__init__(
            f"{message} - Model: {model_name}"
            f"{f' (type: {model_type})' if model_type else ''}"
        )

class UnsupportedVisionFeatureError(ModelVisionError):
    """Error when attempting unsupported vision features."""
    pass

class VisionModelLoadError(ModelVisionError):
    """Error loading vision model."""
    pass
```

### Error Handling Implementation

Update `abstractllm/providers/mlx_provider.py`:

```python
from abstractllm.exceptions.vision import (
    ImageProcessingError,
    ImageLoadError,
    ImageFormatError,
    ImageSizeError,
    ImageMemoryError,
    BatchProcessingError,
    ModelVisionError,
    UnsupportedVisionFeatureError,
    VisionModelLoadError
)

class MLXProvider:
    def _process_image(
        self,
        image: Union[str, Path, Image.Image],
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """Process image with comprehensive error handling."""
        try:
            # Generate cache key if not provided
            if cache_key is None:
                if isinstance(image, (str, Path)):
                    cache_key = str(image)
                else:
                    cache_key = id(image)
            
            # Check cache
            if cache_key in self._image_cache:
                return self._image_cache[cache_key]
            
            # Load image
            try:
                if isinstance(image, (str, Path)):
                    try:
                        img = Image.open(image)
                    except Exception as e:
                        raise ImageLoadError(image, cause=e)
                else:
                    img = image
            except Exception as e:
                raise ImageProcessingError(
                    "Failed to process image input",
                    image=image,
                    cause=e
                )
            
            # Verify format
            if img.format not in ['JPEG', 'PNG', 'WEBP']:
                raise ImageFormatError(
                    "Unsupported image format",
                    image=image,
                    supported_formats=['JPEG', 'PNG', 'WEBP']
                )
            
            # Check dimensions
            width, height = img.size
            max_size = (4096, 4096)  # Example maximum size
            if width > max_size[0] or height > max_size[1]:
                raise ImageSizeError(
                    image=image,
                    current_size=(width, height),
                    max_size=max_size
                )
            
            # Check memory
            try:
                self._check_memory_requirements(img)
            except MemoryError as e:
                raise ImageMemoryError(
                    image=image,
                    required_memory=e.required_memory,
                    available_memory=e.available_memory
                )
            
            # Convert to RGB
            if img.mode != 'RGB':
                try:
                    img = img.convert('RGB')
                except Exception as e:
                    raise ImageProcessingError(
                        "Failed to convert image to RGB",
                        image=image,
                        cause=e
                    )
            
            # Resize
            try:
                img = self._efficient_resize(img)
            except Exception as e:
                raise ImageProcessingError(
                    "Failed to resize image",
                    image=image,
                    cause=e
                )
            
            # Convert to array
            try:
                arr = np.asarray(img, dtype=np.float32)
            except Exception as e:
                raise ImageProcessingError(
                    "Failed to convert image to array",
                    image=image,
                    cause=e
                )
            
            # Normalize
            try:
                mean = np.array(self._get_model_config()["mean"], dtype=np.float32)
                std = np.array(self._get_model_config()["std"], dtype=np.float32)
                arr = (arr - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
            except Exception as e:
                raise ImageProcessingError(
                    "Failed to normalize image",
                    image=image,
                    cause=e
                )
            
            # Transpose
            try:
                arr = arr.transpose(2, 0, 1)  # HWC to CHW
            except Exception as e:
                raise ImageProcessingError(
                    "Failed to transpose image array",
                    image=image,
                    cause=e
                )
            
            # Cache result
            if len(self._image_cache) < self._max_cache_size:
                self._image_cache[cache_key] = arr
            
            return arr
        
        finally:
            # Cleanup
            if isinstance(image, (str, Path)) and 'img' in locals():
                img.close()
    
    def _process_image_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        max_batch_size: int = 4
    ) -> List[np.ndarray]:
        """Process multiple images with error handling."""
        processed = []
        errors = []
        
        for i in range(0, len(images), max_batch_size):
            batch = images[i:i + max_batch_size]
            
            # Process batch
            batch_results = []
            for img in batch:
                try:
                    processed_img = self._process_image(img)
                    batch_results.append(processed_img)
                except Exception as e:
                    errors.append(e)
                    if len(errors) >= 3:  # Stop after 3 errors
                        raise BatchProcessingError(
                            "Multiple errors during batch processing",
                            errors=errors
                        )
            
            processed.extend(batch_results)
            
            # Check memory
            if self._memory_tracker.should_clear_cache():
                self._clear_caches()
        
        # Report any errors
        if errors:
            raise BatchProcessingError(
                "Some images failed to process",
                errors=errors
            )
        
        return processed
    
    def _load_vision_model(self):
        """Load vision model with error handling."""
        try:
            # Verify model supports vision
            if not self._is_vision_model:
                raise UnsupportedVisionFeatureError(
                    "Model does not support vision",
                    model_name=self._config[ModelParameter.MODEL]
                )
            
            # Load model
            try:
                if self._config.get("quantize", True):
                    self._model = self._load_quantized_model()
                else:
                    self._model = self._load_full_model()
            except Exception as e:
                raise VisionModelLoadError(
                    "Failed to load vision model",
                    model_name=self._config[ModelParameter.MODEL],
                    model_type=self._model_type
                ) from e
            
            # Load processor
            try:
                self._processor = self._load_processor()
            except Exception as e:
                raise VisionModelLoadError(
                    "Failed to load vision processor",
                    model_name=self._config[ModelParameter.MODEL],
                    model_type=self._model_type
                ) from e
            
        except Exception as e:
            # Cleanup on error
            self._model = None
            self._processor = None
            self._is_loaded = False
            raise e
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        files: Optional[List[Union[str, Path, Image.Image]]] = None,
        stream: bool = False,
        **kwargs
    ):
        """Generate with comprehensive error handling."""
        try:
            # Check if vision input is supported
            if files and not self._is_vision_model:
                raise UnsupportedVisionFeatureError(
                    "Model does not support vision inputs",
                    model_name=self._config[ModelParameter.MODEL]
                )
            
            # Process images if provided
            processed_images = None
            if files:
                try:
                    processed_images = self._process_image_batch(files)
                except BatchProcessingError as e:
                    # If some images failed but others succeeded, continue
                    if not processed_images:
                        raise e
            
            # Generate response
            try:
                return super().generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    processed_images=processed_images,
                    stream=stream,
                    **kwargs
                )
            except Exception as e:
                raise ModelVisionError(
                    "Generation failed",
                    model_name=self._config[ModelParameter.MODEL],
                    model_type=self._model_type
                ) from e
            
        except Exception as e:
            # Log error details
            self._log_error(e)
            raise e
    
    def _log_error(self, error: Exception):
        """Log error details for debugging."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log error with context
        context = {
            "model": self._config[ModelParameter.MODEL],
            "model_type": self._model_type,
            "is_vision_model": self._is_vision_model,
            "memory_usage": self._memory_tracker.get_stats()
        }
        
        logger.error(
            "Error in MLX provider: %s\nContext: %s",
            str(error),
            context,
            exc_info=error
        )
```

### Error Recovery Strategies

Add `abstractllm/recovery/vision.py`:

```python
"""Vision error recovery strategies."""

from typing import Optional, Union, List, Tuple
from pathlib import Path
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class VisionRecoveryStrategy:
    """Base class for vision error recovery strategies."""
    
    @staticmethod
    def recover_from_size_error(
        image: Union[str, Path, Image.Image],
        max_size: Tuple[int, int]
    ) -> Optional[Image.Image]:
        """Attempt to recover from image size error."""
        try:
            # Load image if needed
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Calculate new size
            width, height = img.size
            aspect = width / height
            
            if width > max_size[0]:
                width = max_size[0]
                height = int(width / aspect)
            
            if height > max_size[1]:
                height = max_size[1]
                width = int(height * aspect)
            
            # Resize
            return img.resize((width, height), Image.LANCZOS)
        
        except Exception as e:
            logger.error("Failed to recover from size error: %s", e)
            return None
    
    @staticmethod
    def recover_from_memory_error(
        image: Union[str, Path, Image.Image],
        target_memory: int
    ) -> Optional[Image.Image]:
        """Attempt to recover from memory error."""
        try:
            # Load image if needed
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Calculate current memory usage
            width, height = img.size
            current_memory = width * height * 3 * 4  # RGB float32
            
            if current_memory <= 0:
                return None
            
            # Calculate scale factor
            scale = (target_memory / current_memory) ** 0.5
            
            # Resize
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            return img.resize((new_width, new_height), Image.LANCZOS)
        
        except Exception as e:
            logger.error("Failed to recover from memory error: %s", e)
            return None
    
    @staticmethod
    def recover_from_format_error(
        image: Union[str, Path, Image.Image],
        target_format: str = 'RGB'
    ) -> Optional[Image.Image]:
        """Attempt to recover from format error."""
        try:
            # Load image if needed
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Convert format
            return img.convert(target_format)
        
        except Exception as e:
            logger.error("Failed to recover from format error: %s", e)
            return None

class BatchRecoveryStrategy:
    """Recovery strategies for batch processing."""
    
    @staticmethod
    def recover_batch(
        images: List[Union[str, Path, Image.Image]],
        errors: List[Exception],
        max_retries: int = 3
    ) -> Tuple[List[np.ndarray], List[Exception]]:
        """Attempt to recover from batch processing errors."""
        recovered = []
        final_errors = []
        
        for img, error in zip(images, errors):
            for _ in range(max_retries):
                try:
                    # Attempt recovery based on error type
                    if isinstance(error, ImageSizeError):
                        recovered_img = VisionRecoveryStrategy.recover_from_size_error(
                            img,
                            error.max_size
                        )
                    elif isinstance(error, ImageMemoryError):
                        recovered_img = VisionRecoveryStrategy.recover_from_memory_error(
                            img,
                            error.available_memory // 2
                        )
                    elif isinstance(error, ImageFormatError):
                        recovered_img = VisionRecoveryStrategy.recover_from_format_error(
                            img
                        )
                    else:
                        # Can't recover from unknown error
                        raise error
                    
                    if recovered_img is not None:
                        recovered.append(recovered_img)
                        break
                
                except Exception as e:
                    error = e
            else:
                # All retries failed
                final_errors.append(error)
        
        return recovered, final_errors
```

## References
- See `docs/mlx/vision-upgrade.md` for vision implementation details
- See `docs/mlx/deepsearch-mlx-vlm.md` for MLX-VLM insights
- See Python error handling best practices

## Testing
Test error handling:
1. Test all error types
2. Test recovery strategies
3. Test batch processing errors
4. Test error logging
5. Test error context

## Success Criteria
1. All errors are properly caught and handled
2. Error messages are clear and helpful
3. Recovery strategies work effectively
4. Batch processing handles errors gracefully
5. Error logging provides useful debugging information
6. No unhandled exceptions 