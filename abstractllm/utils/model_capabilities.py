"""
Model capability utilities for AbstractLLM.

This module provides centralized access to model capabilities across all providers.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Cache for capabilities data
_capabilities_cache: Optional[Dict[str, List[str]]] = None

def _load_capabilities() -> Dict[str, List[str]]:
    """
    Load model capabilities from the JSON file.
    
    Returns:
        Dictionary mapping capability names to lists of model patterns
    """
    global _capabilities_cache
    
    if _capabilities_cache is not None:
        return _capabilities_cache
    
    try:
        # Get the path to the JSON file relative to this module
        json_path = Path(__file__).parent.parent / "assets" / "model_capabilities.json"
        
        with open(json_path, 'r') as f:
            _capabilities_cache = json.load(f)
        
        logger.debug(f"Loaded model capabilities from {json_path}")
        return _capabilities_cache
        
    except FileNotFoundError:
        logger.error(f"Model capabilities file not found at {json_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing model capabilities JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading model capabilities: {e}")
        return {}

def _normalize_model_name(model_name: str) -> str:
    """
    Normalize a model name for capability checking.
    
    Handles common variations like:
    - Removing size indicators (7b, 8b, 14b, etc.)
    - Removing quantization indicators (q4_k_m, 4bit, etc.)
    - Removing provider prefixes (mlx-community/, etc.)
    - Converting to lowercase
    
    Args:
        model_name: Raw model name
        
    Returns:
        Normalized model name
    """
    import re
    
    # Convert to lowercase
    normalized = model_name.lower()
    
    # Remove provider prefixes
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    
    # Remove common suffixes and patterns
    patterns_to_remove = [
        r':latest$',  # :latest
        r':.*$',      # Any tag after colon (e.g., :2b, :8b, :q4_k_m)
        r'-\d+[bm]$', # Size indicators like -7b, -8b, -14b, -32b
        r'-q\d+.*$',  # Quantization like -q4_k_m, -q8_0
        r'-\d+bit$', # Quantization like -4bit, -8bit
        r'-instruct$', # Common suffix
        r'-chat$',    # Common suffix
        r'-preview$', # Common suffix
    ]
    
    for pattern in patterns_to_remove:
        normalized = re.sub(pattern, '', normalized)
    
    return normalized

def has_capability(model_name: str, capability: str) -> bool:
    """
    Check if a model has a specific capability.
    
    Args:
        model_name: Model name (can include size, quantization, etc.)
        capability: Capability to check (e.g., 'tool_calling', 'vision')
        
    Returns:
        True if the model has the capability, False otherwise
    """
    capabilities = _load_capabilities()
    
    if capability not in capabilities:
        logger.warning(f"Unknown capability: {capability}")
        return False
    
    # Normalize the model name
    normalized_model = _normalize_model_name(model_name)
    
    # Check for exact matches and prefix matches
    for supported_pattern in capabilities[capability]:
        supported_pattern = supported_pattern.lower()
        
        # Exact match
        if normalized_model == supported_pattern:
            return True
            
        # Prefix match (model starts with pattern)
        if normalized_model.startswith(supported_pattern):
            return True
            
        # Pattern match (pattern is contained in model name)
        if supported_pattern in normalized_model:
            return True
    
    return False

def get_model_capabilities(model_name: str) -> Dict[str, bool]:
    """
    Get all capabilities for a specific model.
    
    Args:
        model_name: Model name to check
        
    Returns:
        Dictionary mapping capability names to boolean values
    """
    capabilities = _load_capabilities()
    
    result = {}
    for capability in capabilities.keys():
        result[capability] = has_capability(model_name, capability)
    
    return result

def get_models_with_capability(capability: str) -> List[str]:
    """
    Get all model patterns that have a specific capability.
    
    Args:
        capability: Capability to check
        
    Returns:
        List of model patterns that have the capability
    """
    capabilities = _load_capabilities()
    return capabilities.get(capability, [])

def add_model_capability(model_pattern: str, capability: str) -> bool:
    """
    Add a capability for a model pattern (runtime only, not persisted).
    
    Args:
        model_pattern: Model pattern to add capability for
        capability: Capability to add
        
    Returns:
        True if added successfully, False otherwise
    """
    global _capabilities_cache
    
    capabilities = _load_capabilities()
    
    if capability not in capabilities:
        capabilities[capability] = []
    
    if model_pattern not in capabilities[capability]:
        capabilities[capability].append(model_pattern)
        logger.info(f"Added {capability} capability for {model_pattern}")
        return True
    
    return False

def reload_capabilities() -> None:
    """
    Force reload of capabilities from disk.
    """
    global _capabilities_cache
    _capabilities_cache = None
    _load_capabilities()

# Legacy compatibility functions for existing providers
def supports_tool_calls(model_name: str) -> bool:
    """
    Legacy function for checking tool calling support.
    
    Args:
        model_name: Model name to check
        
    Returns:
        True if the model supports tool calling
    """
    # Import here to avoid circular import
    from abstractllm.architectures.detection import get_model_capabilities as get_arch_capabilities
    
    # Check if model has any tool support (not just "none")
    capabilities = get_arch_capabilities(model_name)
    tool_support = capabilities.get("tool_support", "none")
    return tool_support in ["native", "prompted"]

def supports_vision(model_name: str) -> bool:
    """
    Legacy function for checking vision support.
    
    Args:
        model_name: Model name to check
        
    Returns:
        True if the model supports vision
    """
    return has_capability(model_name, "vision") 