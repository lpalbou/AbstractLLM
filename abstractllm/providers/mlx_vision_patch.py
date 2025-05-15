# MLX Vision Patch
# This file provides fixes for MLX vision models in AbstractLLM

def apply_patches():
    """Apply patches to fix MLX vision capability issues."""
    from abstractllm.providers.mlx_provider import MLXProvider
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Store the original _process_image method
    original_process_image = MLXProvider._process_image
    
    # Define our patched method
    def patched_process_image(self, image_input):
        """Patched version of _process_image that fixes patch_size issue."""
        # Fix processor if needed
        if hasattr(self, '_processor') and self._processor is not None:
            if hasattr(self._processor, 'patch_size') and self._processor.patch_size is None:
                # Standard patch size for CLIP-like models
                self._processor.patch_size = 14
                logger.info("Fixed missing patch_size in processor")
            
            # Also check image_processor if it exists
            if hasattr(self._processor, 'image_processor'):
                if hasattr(self._processor.image_processor, 'patch_size') and self._processor.image_processor.patch_size is None:
                    self._processor.image_processor.patch_size = 14
                    logger.info("Fixed missing patch_size in image_processor")
                    
            # Make sure we can handle PIL image objects directly
            if hasattr(self._processor, 'process_images'):
                original_process_images = self._processor.process_images
                
                def patched_process_images(images, **kwargs):
                    """Ensures image processing can handle PIL images directly."""
                    try:
                        return original_process_images(images, **kwargs)
                    except Exception as e:
                        logger.warning(f"Error in original process_images: {e}. Trying alternate approach.")
                        # Try to convert the image to RGB if it's not already
                        try:
                            from PIL import Image
                            if isinstance(images, list):
                                processed_images = []
                                for img in images:
                                    if hasattr(img, 'convert'):
                                        img = img.convert('RGB')
                                    processed_images.append(img)
                                return original_process_images(processed_images, **kwargs)
                            elif hasattr(images, 'convert'):
                                images = images.convert('RGB')
                                return original_process_images(images, **kwargs)
                        except Exception as inner_e:
                            logger.error(f"Failed to process image in alternate way: {inner_e}")
                            raise
                
                # Apply the patch to process_images
                if hasattr(self._processor, 'process_images'):
                    self._processor.process_images = patched_process_images
                    logger.info("Patched processor.process_images")
        
        # Call original method
        return original_process_image(self, image_input)
        
    # Apply the patch
    MLXProvider._process_image = patched_process_image
    logger.info("Successfully patched MLXProvider._process_image")
    
    # Store the original load_model method
    original_load_model = MLXProvider.load_model
    
    # Define patched load_model
    def patched_load_model(self):
        """Patched version of load_model that fixes processor issues after loading."""
        # Call original method
        original_load_model(self)
        
        # Fix processor if needed
        if hasattr(self, '_processor') and self._processor is not None:
            if hasattr(self._processor, 'patch_size') and self._processor.patch_size is None:
                self._processor.patch_size = 14
                logger.info("Fixed missing patch_size in processor after loading")
            
            if hasattr(self._processor, 'image_processor'):
                if hasattr(self._processor.image_processor, 'patch_size') and self._processor.image_processor.patch_size is None:
                    self._processor.image_processor.patch_size = 14
                    logger.info("Fixed missing patch_size in image_processor after loading")
    
    # Apply the patch
    MLXProvider.load_model = patched_load_model
    logger.info("Successfully patched MLXProvider.load_model")
    
    return True
