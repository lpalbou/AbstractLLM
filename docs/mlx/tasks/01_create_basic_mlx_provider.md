# Task 1: Create Basic MLX Provider Structure

## Description
Create the basic structure for the MLX provider class that follows AbstractLLM's provider pattern.

## Requirements
1. Create a new file `abstractllm/providers/mlx_provider.py`
2. Implement the basic `MLXProvider` class that extends `AbstractLLMInterface`
3. Add conditional imports for MLX dependencies (`mlx`, `mlx-lm`)
4. Set up proper logging
5. Initialize the provider with default configuration values

## Implementation Details
The initial implementation should include:

```python
"""
MLX provider for AbstractLLM.

This provider leverages Apple's MLX framework for efficient
inference on Apple Silicon devices.
"""

import os
import time
import logging
import platform
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union, Callable, Tuple, ClassVar

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import mlx_lm
    MLXLM_AVAILABLE = True
except ImportError:
    MLXLM_AVAILABLE = False

from abstractllm.interface import AbstractLLMInterface
from abstractllm.enums import ModelParameter, ModelCapability
from abstractllm.types import GenerateResponse
from abstractllm.exceptions import UnsupportedFeatureError
from abstractllm.utils.logging import log_request, log_response, log_api_key_missing

# Set up logger
logger = logging.getLogger("abstractllm.providers.mlx")

class MLXProvider(AbstractLLMInterface):
    """
    MLX implementation for AbstractLLM.
    
    This provider leverages Apple's MLX framework for efficient
    inference on Apple Silicon devices.
    """
    
    # Class-level model cache
    _model_cache: ClassVar[Dict[str, Tuple[Any, Any, float]]] = {}
    _max_cached_models = 2  # Default to 2 models in memory
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the MLX provider."""
        super().__init__(config)
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "mlx-community/qwen2.5-coder-14b-instruct-abliterated",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 4096,
            ModelParameter.TOP_P: 0.9,
            "cache_dir": None,  # Use default HuggingFace cache
            "quantize": True    # Use quantized models by default
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Initialize components
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._is_vision_model = False
        
        # Log initialization
        model = self.config_manager.get_param(ModelParameter.MODEL)
        logger.info(f"Initialized MLX provider with model: {model}")
```

## References
- See `AbstractLLMInterface`