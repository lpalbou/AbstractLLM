"""
Patch for transformers library to support MLX tensor type properly.

This module provides patches to make transformers work correctly with MLX.
"""

import logging
import importlib
from types import ModuleType
from typing import Any, Dict, Optional, Tuple, Callable

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
            logger.info("Transformers TensorType needs patching for MLX")
            
            # Get all current values
            current_values = {t.value: t.name for t in TensorType}
            
            # Add 'mlx' to TensorType enum - use a clever technique that works with enums
            # We're adding 'mlx' as an alias for 'np' since MLX uses numpy-like arrays
            if 'ml' not in current_values and 'mlx' not in current_values:
                # Add ML and MLX as aliases for numpy tensors since they're similar
                TensorType._value2member_map_['ml'] = TensorType.NUMPY
                TensorType._value2member_map_['mlx'] = TensorType.NUMPY
                logger.info("Added 'ml' and 'mlx' to TensorType enum as aliases for numpy")
            elif 'ml' not in current_values:
                TensorType._value2member_map_['ml'] = TensorType.NUMPY
                logger.info("Added 'ml' to TensorType enum as alias for numpy")
            elif 'mlx' not in current_values:
                TensorType._value2member_map_['mlx'] = TensorType.NUMPY
                logger.info("Added 'mlx' to TensorType enum as alias for numpy")
                
            # Verify the patch worked
            tensor_types = [t.value for t in TensorType]
            if 'mlx' in tensor_types and 'ml' in tensor_types:
                logger.info("Successfully patched TensorType to support 'ml' and 'mlx'")
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
        
def patch_image_processor():
    """
    Patch image processors to handle MLX tensor type.
    """
    try:
        from transformers import AutoImageProcessor, CLIPImageProcessor
        
        # Save original methods
        original_clip_preprocess = CLIPImageProcessor.preprocess
        
        # Create patched methods
        def patched_clip_preprocess(self, images, *args, **kwargs):
            # Intercept return_tensors for MLX
            return_tensors = kwargs.get('return_tensors', None)
            if return_tensors in ('ml', 'mlx'):
                # Use numpy instead, which will be converted to MLX array later
                kwargs['return_tensors'] = 'np'
                result = original_clip_preprocess(self, images, *args, **kwargs)
                # Log the interception
                logger.debug(f"Intercepted {return_tensors} tensor type in CLIPImageProcessor")
                return result
            return original_clip_preprocess(self, images, *args, **kwargs)
        
        # Apply patches
        CLIPImageProcessor.preprocess = patched_clip_preprocess
        logger.info("Patched CLIPImageProcessor.preprocess to handle MLX tensor types")
        
        return True
    except ImportError:
        logger.warning("Could not import transformers image processors, skipping patch")
        return False
    except Exception as e:
        logger.error(f"Failed to patch image processors: {e}")
        return False

def apply_all_patches():
    """
    Apply all patches for MLX integration.
    """
    success = True
    
    # Apply transformers patches
    if not patch_transformers_tensor_type():
        success = False
        
    # Apply image processor patches
    if not patch_image_processor():
        success = False
        
    return success 