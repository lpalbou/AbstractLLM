"""
Architecture detection and configuration system.

This module provides unified architecture detection and configuration
for different model families across all providers.
"""

from .detection import (
    detect_architecture, 
    get_tool_call_format,
    get_supported_architectures,
    normalize_model_name
)
from .capabilities import (
    get_capabilities,
    get_supported_architectures as get_supported_capability_architectures,
    detect_model_vision_capability,
    detect_model_audio_capability,
    get_model_capabilities,
    detect_model_tool_capability,
    detect_model_reasoning_capability,
    detect_model_type
)
from .configs import get_config
from .templates import get_template

__all__ = [
    # Architecture detection
    'detect_architecture',
    'get_tool_call_format', 
    'get_supported_architectures',
    'normalize_model_name',
    
    # Architecture capabilities (family-level)
    'get_capabilities',
    
    # Model capabilities (instance-level)
    'detect_model_vision_capability',
    'detect_model_audio_capability', 
    'get_model_capabilities',
    'detect_model_tool_capability',
    'detect_model_reasoning_capability', 
    'detect_model_type',
    
    # Configuration and templates
    'get_config',
    'get_template'
] 