# Task 16: Performance Optimization

## Description
Implement performance optimizations for the MLX provider, with special focus on vision capabilities.

## Requirements
1. Optimize image processing pipeline
2. Implement memory-efficient operations
3. Add caching mechanisms
4. Optimize model loading
5. Add performance monitoring

## Implementation Details

### Image Processing Optimizations

Update `abstractllm/providers/mlx_provider.py`:

```python
import numpy as np
from PIL import Image
import mlx.core as mx
from functools import lru_cache
import psutil
import gc
from typing import Tuple, Optional, Dict, Any

class MLXProvider:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._image_cache = {}
        self._model_cache = {}
        self._max_cache_size = 100  # Maximum number of cached items
        
        # Initialize memory tracker
        self._memory_tracker = MemoryTracker()
    
    @lru_cache(maxsize=32)
    def _get_model_config(self) -> dict:
        """Get cached model configuration."""
        return self._determine_model_config()
    
    def _clear_caches(self):
        """Clear all caches when memory is low."""
        self._image_cache.clear()
        self._model_cache.clear()
        gc.collect()
    
    def _process_image_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        max_batch_size: int = 4
    ) -> List[np.ndarray]:
        """Process multiple images efficiently in batches."""
        processed = []
        for i in range(0, len(images), max_batch_size):
            batch = images[i:i + max_batch_size]
            
            # Process batch in parallel if possible
            if hasattr(mx, 'parallel'):
                batch_processed = mx.parallel(self._process_image, batch)
            else:
                batch_processed = [self._process_image(img) for img in batch]
            
            processed.extend(batch_processed)
            
            # Clear cache if memory usage is high
            if self._memory_tracker.should_clear_cache():
                self._clear_caches()
        
        return processed
    
    def _process_image(
        self,
        image: Union[str, Path, Image.Image],
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """Process image with caching and memory optimization."""
        # Generate cache key if not provided
        if cache_key is None:
            if isinstance(image, (str, Path)):
                cache_key = str(image)
            else:
                cache_key = id(image)
        
        # Check cache
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        # Check memory before processing
        self._check_memory_requirements(image)
        
        try:
            # Load and preprocess image
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize efficiently
            img = self._efficient_resize(img)
            
            # Convert to numpy array efficiently
            arr = np.asarray(img, dtype=np.float32)
            
            # Optimize memory usage during normalization
            mean = np.array(self._get_model_config()["mean"], dtype=np.float32)
            std = np.array(self._get_model_config()["std"], dtype=np.float32)
            
            # Normalize in-place if possible
            arr -= mean.reshape(1, 1, -1)
            arr /= std.reshape(1, 1, -1)
            
            # Transpose efficiently
            arr = arr.transpose(2, 0, 1)  # HWC to CHW
            
            # Cache result if memory allows
            if len(self._image_cache) < self._max_cache_size:
                self._image_cache[cache_key] = arr
            
            return arr
        
        finally:
            # Cleanup
            if isinstance(image, (str, Path)):
                img.close()
            
            # Clear cache if needed
            if self._memory_tracker.should_clear_cache():
                self._clear_caches()
    
    def _efficient_resize(self, img: Image.Image) -> Image.Image:
        """Resize image efficiently while preserving aspect ratio."""
        target_size = self._get_model_config()["image_size"]
        
        # Calculate new size preserving aspect ratio
        width, height = img.size
        aspect = width / height
        
        if aspect > 1:
            new_width = target_size[0]
            new_height = int(target_size[1] / aspect)
        else:
            new_height = target_size[1]
            new_width = int(target_size[0] * aspect)
        
        # Resize using LANCZOS for better quality/speed trade-off
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        new_img.paste(
            img,
            ((target_size[0] - new_width) // 2,
             (target_size[1] - new_height) // 2)
        )
        
        return new_img
    
    def _check_memory_requirements(self, image: Union[str, Path, Image.Image]) -> None:
        """Check if system can handle image processing."""
        # Get image size
        if isinstance(image, (str, Path)):
            size = Image.open(image).size
        else:
            size = image.size
        
        width, height = size
        
        # Calculate memory requirements
        pixel_memory = width * height * 3 * 4  # RGB float32
        processing_memory = pixel_memory * 3  # Buffer for processing
        model_memory = self._estimate_model_memory()
        
        total_required = processing_memory + model_memory
        
        # Check system memory
        available_memory = psutil.virtual_memory().available
        if total_required > available_memory * 0.9:  # Leave 10% buffer
            raise MemoryError(
                f"Insufficient memory. Required: {total_required / 1024**3:.1f}GB, "
                f"Available: {available_memory / 1024**3:.1f}GB"
            )
    
    @lru_cache(maxsize=1)
    def _estimate_model_memory(self) -> int:
        """Estimate model memory requirements."""
        model_name = self._config[ModelParameter.MODEL].lower()
        
        # Rough estimates based on model size
        if "32b" in model_name:
            return 32 * 1024**3  # 32GB
        elif "70b" in model_name:
            return 70 * 1024**3  # 70GB
        else:
            return 16 * 1024**3  # 16GB default
    
    def _optimize_model_loading(self):
        """Optimize model loading process."""
        # Clear memory before loading
        gc.collect()
        
        # Load model efficiently
        model_path = self._config[ModelParameter.MODEL]
        if model_path in self._model_cache:
            self._model = self._model_cache[model_path]
            return
        
        # Check memory before loading
        required_memory = self._estimate_model_memory()
        available_memory = psutil.virtual_memory().available
        
        if required_memory > available_memory * 0.9:
            raise MemoryError(
                f"Insufficient memory for model loading. Required: {required_memory / 1024**3:.1f}GB, "
                f"Available: {available_memory / 1024**3:.1f}GB"
            )
        
        # Load model with quantization if enabled
        if self._config.get("quantize", True):
            self._model = self._load_quantized_model()
        else:
            self._model = self._load_full_model()
        
        # Cache model if memory allows
        if len(self._model_cache) < 2:  # Keep at most 2 models in cache
            self._model_cache[model_path] = self._model
    
    def _load_quantized_model(self):
        """Load quantized model efficiently."""
        # Implementation depends on specific quantization method
        pass
    
    def _load_full_model(self):
        """Load full precision model efficiently."""
        # Implementation depends on model architecture
        pass

class MemoryTracker:
    """Track memory usage and provide optimization suggestions."""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.peak_usage = 0
        self.last_check = time.time()
        self.check_interval = 1  # seconds
    
    def should_clear_cache(self) -> bool:
        """Check if caches should be cleared based on memory usage."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return False
        
        self.last_check = current_time
        memory = psutil.virtual_memory()
        
        # Update peak usage
        usage_percent = memory.percent
        self.peak_usage = max(self.peak_usage, usage_percent)
        
        return usage_percent > self.threshold * 100
    
    def get_stats(self) -> dict:
        """Get memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "peak_percent": self.peak_usage
        }
```

### Performance Monitoring

Add `abstractllm/monitoring/performance.py`:

```python
"""Performance monitoring utilities."""

import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    duration: float
    memory_start: Dict[str, int]
    memory_peak: Dict[str, int]
    memory_end: Dict[str, int]
    cpu_percent: float
    gpu_metrics: Optional[Dict[str, Any]] = None

class PerformanceMonitor:
    """Monitor performance metrics during operations."""
    
    def __init__(self):
        self.metrics = []
        self._monitoring = False
        self._monitor_thread = None
    
    @contextmanager
    def monitor(self, operation: str):
        """Context manager for monitoring an operation."""
        try:
            start_time = time.time()
            start_memory = self._get_memory_info()
            peak_memory = start_memory.copy()
            
            # Start background monitoring
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._background_monitor,
                args=(peak_memory,)
            )
            self._monitor_thread.start()
            
            yield
        
        finally:
            # Stop monitoring
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join()
            
            # Collect final metrics
            end_time = time.time()
            end_memory = self._get_memory_info()
            
            metrics = PerformanceMetrics(
                duration=end_time - start_time,
                memory_start=start_memory,
                memory_peak=peak_memory,
                memory_end=end_memory,
                cpu_percent=psutil.cpu_percent(),
                gpu_metrics=self._get_gpu_metrics()
            )
            
            self.metrics.append((operation, metrics))
    
    def _background_monitor(self, peak_memory: Dict[str, int]):
        """Background thread for continuous monitoring."""
        while self._monitoring:
            current = self._get_memory_info()
            for key, value in current.items():
                peak_memory[key] = max(peak_memory[key], value)
            time.sleep(0.1)
    
    def _get_memory_info(self) -> Dict[str, int]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory = process.memory_info()
        return {
            "rss": memory.rss,
            "vms": memory.vms,
            "shared": memory.shared,
            "text": memory.text,
            "lib": memory.lib,
            "data": memory.data,
            "dirty": memory.dirty
        }
    
    def _get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics if available."""
        try:
            # MLX-specific GPU metrics could be added here
            return None
        except Exception:
            return None
    
    def get_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            "operations": {},
            "summary": {
                "total_duration": 0,
                "peak_memory": 0,
                "average_cpu": 0
            }
        }
        
        cpu_values = []
        for operation, metrics in self.metrics:
            report["operations"][operation] = {
                "duration": metrics.duration,
                "memory": {
                    "start": metrics.memory_start,
                    "peak": metrics.memory_peak,
                    "end": metrics.memory_end
                },
                "cpu_percent": metrics.cpu_percent,
                "gpu_metrics": metrics.gpu_metrics
            }
            
            report["summary"]["total_duration"] += metrics.duration
            report["summary"]["peak_memory"] = max(
                report["summary"]["peak_memory"],
                metrics.memory_peak["rss"]
            )
            cpu_values.append(metrics.cpu_percent)
        
        if cpu_values:
            report["summary"]["average_cpu"] = sum(cpu_values) / len(cpu_values)
        
        return report
```

### Usage Example

```python
from abstractllm import create_llm
from abstractllm.monitoring.performance import PerformanceMonitor

# Create monitor
monitor = PerformanceMonitor()

# Create provider
llm = create_llm("mlx", model="mlx-community/Qwen2.5-VL-32B-Instruct-6bit")

# Monitor image processing
with monitor.monitor("image_processing"):
    processed_images = llm._process_image_batch([
        "image1.jpg",
        "image2.jpg"
    ])

# Monitor generation
with monitor.monitor("generation"):
    response = llm.generate(
        prompt="Compare these images.",
        files=processed_images
    )

# Get performance report
report = monitor.get_report()
print(json.dumps(report, indent=2))
```

## References
- See `docs/mlx/vision-upgrade.md` for vision implementation details
- See `docs/mlx/deepsearch-mlx-vlm.md` for MLX-VLM insights
- See MLX documentation for performance optimization guidelines

## Testing
Test optimizations:
1. Measure image processing performance
2. Verify memory usage optimization
3. Test caching effectiveness
4. Validate monitoring accuracy
5. Benchmark different scenarios

## Success Criteria
1. Reduced memory usage
2. Faster image processing
3. Efficient caching
4. Accurate monitoring
5. Stable performance under load
6. No memory leaks 