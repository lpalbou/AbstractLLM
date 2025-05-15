"""
Patch for transformers library to support MLX tensor type properly.

This module provides patches to make transformers work correctly with MLX.
"""

import logging
from typing import Optional, Any, Dict

# Set up logger
logger = logging.getLogger(__name__)

def patch_transformers_tensor_type() -> bool:
    """
    Patch the transformers library to properly handle MLX tensor type.
    
    Returns:
        bool: True if patching was successful, False otherwise
    """
    try:
        # Try to import transformers
        import transformers
        from transformers.utils.generic import TensorType
        
        # Check if the patch is needed
        needs_patch = 'mlx' not in [t.value for t in TensorType]
        
        if needs_patch:
            logger.info("Adding 'mlx' to transformers TensorType enum")
            
            # Add 'mlx' to TensorType enum - use a clever technique that works with enums
            # We're adding 'mlx' as an alias for 'np' since MLX uses numpy-like arrays
            TensorType._value2member_map_['mlx'] = TensorType.NUMPY
            
            # Verify the patch worked
            tensor_types = [t.value for t in TensorType]
            if 'mlx' in tensor_types:
                logger.info("Successfully added 'mlx' to TensorType")
                return True
            else:
                logger.warning("Failed to verify TensorType patch")
                return False
        else:
            logger.info("TensorType already supports MLX, no patching needed")
            return True
    except ImportError:
        logger.warning("Could not import transformers, skipping TensorType patch")
        return False
    except Exception as e:
        logger.error(f"Failed to patch transformers TensorType: {e}")
        return False

def apply_all_patches() -> bool:
    """
    Apply all patches for MLX integration.
    
    Returns:
        bool: True if all patches were successful, False otherwise
    """
    success = patch_transformers_tensor_type()
    return success 