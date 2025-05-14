#!/usr/bin/env python3
"""
Fix for MLX Vision capabilities in AbstractLLM.

This script patches the MLXProvider to fix issues with vision models,
particularly the patch_size attribute in processors.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_mlx_provider():
    """Patch the MLXProvider to fix vision model issues."""
    try:
        from abstractllm.providers.mlx_provider import MLXProvider
        
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
            
            # Call original method
            return original_process_image(self, image_input)
            
        # Apply the patch
        MLXProvider._process_image = patched_process_image
        logger.info("Successfully patched MLXProvider._process_image")
        
        # Store the original _prepare_model_inputs method if it exists
        if hasattr(MLXProvider, '_prepare_model_inputs'):
            original_prepare_inputs = MLXProvider._prepare_model_inputs
            
            def patched_prepare_inputs(self, prompt, images=None):
                """Patched version of _prepare_model_inputs that handles processor issues."""
                # Fix processor if needed
                if hasattr(self, '_processor') and self._processor is not None:
                    if hasattr(self._processor, 'patch_size') and self._processor.patch_size is None:
                        self._processor.patch_size = 14
                        logger.info("Fixed missing patch_size in processor")
                    
                    if hasattr(self._processor, 'image_processor'):
                        if hasattr(self._processor.image_processor, 'patch_size') and self._processor.image_processor.patch_size is None:
                            self._processor.image_processor.patch_size = 14
                            logger.info("Fixed missing patch_size in image_processor")
                
                # Call original method
                return original_prepare_inputs(self, prompt, images)
                
            # Apply the patch
            MLXProvider._prepare_model_inputs = patched_prepare_inputs
            logger.info("Successfully patched MLXProvider._prepare_model_inputs")
        
        return True
    except Exception as e:
        logger.error(f"Failed to patch MLXProvider: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_monkey_patch_file():
    """Create a monkey patch file that can be imported to fix MLX vision."""
    patch_file_path = "abstractllm/providers/mlx_vision_patch.py"
    
    patch_code = """
# MLX Vision Patch
# This file provides fixes for MLX vision models in AbstractLLM

def apply_patches():
    \"\"\"Apply patches to fix MLX vision capability issues.\"\"\"
    from abstractllm.providers.mlx_provider import MLXProvider
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Store the original _process_image method
    original_process_image = MLXProvider._process_image
    
    # Define our patched method
    def patched_process_image(self, image_input):
        \"\"\"Patched version of _process_image that fixes patch_size issue.\"\"\"
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
        
        # Call original method
        return original_process_image(self, image_input)
        
    # Apply the patch
    MLXProvider._process_image = patched_process_image
    logger.info("Successfully patched MLXProvider._process_image")
    
    # Store the original load_model method
    original_load_model = MLXProvider.load_model
    
    # Define patched load_model
    def patched_load_model(self):
        \"\"\"Patched version of load_model that fixes processor issues after loading.\"\"\"
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
"""
    
    try:
        with open(patch_file_path, "w") as f:
            f.write(patch_code)
        logger.info(f"Successfully created patch file: {patch_file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create patch file: {e}")
        return False

def update_mlx_provider():
    """Update the MLXProvider file directly."""
    mlx_provider_path = "abstractllm/providers/mlx_provider.py"
    
    try:
        # Read the current file
        with open(mlx_provider_path, "r") as f:
            content = f.read()
        
        # Check if we need to update
        if "# Fix patch_size in processor if needed" in content:
            logger.info("MLXProvider already contains the patch")
            return False
        
        # Find the _process_image method
        process_image_def = "def _process_image(self, image_input)"
        if process_image_def not in content:
            logger.error("Could not find _process_image method in MLXProvider")
            return False
        
        # Find the beginning of the method body
        method_start = content.index(process_image_def)
        method_body_start = content.index(":", method_start) + 1
        
        # Insert our patch code after the method definition
        patch_code = """
        # Fix patch_size in processor if needed
        if hasattr(self, '_processor') and self._processor is not None:
            if hasattr(self._processor, 'patch_size') and self._processor.patch_size is None:
                # Standard patch size for CLIP-like models
                self._processor.patch_size = 14
                logger.debug("Fixed missing patch_size in processor")
            
            # Also check image_processor if it exists
            if hasattr(self._processor, 'image_processor'):
                if hasattr(self._processor.image_processor, 'patch_size') and self._processor.image_processor.patch_size is None:
                    self._processor.image_processor.patch_size = 14
                    logger.debug("Fixed missing patch_size in image_processor")
        """
        
        updated_content = content[:method_body_start] + patch_code + content[method_body_start:]
        
        # Also find the load_model method and patch it
        load_model_def = "def load_model(self)"
        if load_model_def in updated_content:
            # Find where the model is loaded
            model_loaded_marker = "self._is_loaded = True"
            if model_loaded_marker in updated_content:
                model_loaded_idx = updated_content.index(model_loaded_marker)
                
                patch_load_code = """
        # Fix patch_size in processor if needed
        if self._is_vision_model and hasattr(self, '_processor') and self._processor is not None:
            if hasattr(self._processor, 'patch_size') and self._processor.patch_size is None:
                # Standard patch size for CLIP-like models
                self._processor.patch_size = 14
                logger.debug("Fixed missing patch_size in processor after model load")
            
            # Also check image_processor if it exists
            if hasattr(self._processor, 'image_processor'):
                if hasattr(self._processor.image_processor, 'patch_size') and self._processor.image_processor.patch_size is None:
                    self._processor.image_processor.patch_size = 14
                    logger.debug("Fixed missing patch_size in image_processor after model load")
                    
        """
                
                # Insert before the is_loaded line
                updated_content = updated_content[:model_loaded_idx] + patch_load_code + updated_content[model_loaded_idx:]
        
        # Write the updated file
        with open(mlx_provider_path, "w") as f:
            f.write(updated_content)
            
        logger.info(f"Successfully updated {mlx_provider_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to update MLXProvider file: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_patch_file_for_mlx_repl():
    """Create a patch for mlx_repl_direct.py to fix vision issues."""
    repl_patch_path = "mlx_repl_patch.py"
    
    patch_code = '''#!/usr/bin/env python3
"""
Patch for MLX REPL to fix vision model issues.

This script applies patches to the mlx_repl_direct.py script to fix
issues with vision models, particularly the patch_size attribute in processors.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_mlx_repl():
    """Patch the MLX REPL script to fix vision model issues."""
    repl_path = "mlx_repl_direct.py"
    
    try:
        # Check if file exists
        if not os.path.exists(repl_path):
            logger.error(f"Could not find {repl_path}")
            return False
            
        # Read the current file
        with open(repl_path, "r") as f:
            content = f.read()
            
        # Check if we need to update
        if "# Fix patch_size in processor if needed" in content:
            logger.info("MLX REPL already contains the patch")
            return False
            
        # Find the do_ask method
        ask_method_def = "def do_ask(self, arg):"
        if ask_method_def not in content:
            logger.error("Could not find do_ask method in MLX REPL")
            return False
            
        # Find where the LLM is created
        llm_creation = "self.llm = create_llm("
        if llm_creation not in content:
            logger.error("Could not find LLM creation in MLX REPL")
            return False
            
        # Get the position after LLM creation
        llm_creation_idx = content.index(llm_creation)
        llm_creation_end = content.index(")", llm_creation_idx) + 1
        
        # Insert our patch code after LLM creation
        patch_code = """
                # Apply patch for vision models
                if self.images and hasattr(self.llm, "_provider"):
                    provider = self.llm._provider
                    if hasattr(provider, "_processor") and provider._processor is not None:
                        if hasattr(provider._processor, "patch_size") and provider._processor.patch_size is None:
                            provider._processor.patch_size = 14
                            print("Applied patch: Fixed missing patch_size in processor")
                        
                        if hasattr(provider._processor, "image_processor"):
                            if hasattr(provider._processor.image_processor, "patch_size") and provider._processor.image_processor.patch_size is None:
                                provider._processor.image_processor.patch_size = 14
                                print("Applied patch: Fixed missing patch_size in image_processor")
                """
                
        updated_content = content[:llm_creation_end] + patch_code + content[llm_creation_end:]
        
        # Write the updated file
        with open(repl_path, "w") as f:
            f.write(updated_content)
            
        logger.info(f"Successfully updated {repl_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to update MLX REPL file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Apply all patches."""
    print("=== MLX Vision Patches for AbstractLLM ===")
    
    # 1. Apply runtime patch
    print("\n1. Applying runtime patch to MLXProvider...")
    if patch_mlx_provider():
        print("✓ Runtime patch applied successfully")
    else:
        print("✗ Failed to apply runtime patch")
        
    # 2. Create monkey patch file
    print("\n2. Creating monkey patch file...")
    if create_monkey_patch_file():
        print("✓ Monkey patch file created successfully")
    else:
        print("✗ Failed to create monkey patch file")
        
    # 3. Update MLXProvider file directly
    print("\n3. Updating MLXProvider file directly...")
    if update_mlx_provider():
        print("✓ MLXProvider file updated successfully")
    else:
        print("✗ Failed to update MLXProvider file")
        
    # 4. Create patch for MLX REPL
    print("\n4. Creating patch for MLX REPL...")
    if create_patch_file_for_mlx_repl():
        print("✓ MLX REPL patch file created successfully")
    else:
        print("✗ Failed to create MLX REPL patch file")
        
    print("\n=== Patches Complete ===")
    print("\nTo apply the patch to your MLX REPL script, run:")
    print("    python mlx_repl_patch.py")
    print("\nTo use the monkey patch in your code:")
    print("    from abstractllm.providers.mlx_vision_patch import apply_patches")
    print("    apply_patches()")

if __name__ == "__main__":
    main()'''
    
    try:
        with open(repl_patch_path, "w") as f:
            f.write(patch_code)
        logger.info(f"Successfully created patch file: {repl_patch_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create patch file: {e}")
        return False

def main():
    """Apply all patches."""
    print("=== MLX Vision Patches for AbstractLLM ===")
    
    # 1. Apply runtime patch
    print("\n1. Applying runtime patch to MLXProvider...")
    if patch_mlx_provider():
        print("✓ Runtime patch applied successfully")
    else:
        print("✗ Failed to apply runtime patch")
        
    # 2. Create monkey patch file
    print("\n2. Creating monkey patch file...")
    if create_monkey_patch_file():
        print("✓ Monkey patch file created successfully")
    else:
        print("✗ Failed to create monkey patch file")
        
    # 3. Update MLXProvider file directly
    print("\n3. Updating MLXProvider file directly...")
    if update_mlx_provider():
        print("✓ MLXProvider file updated successfully")
    else:
        print("✗ Failed to update MLXProvider file")
        
    # 4. Create patch for MLX REPL
    print("\n4. Creating patch for MLX REPL...")
    if create_patch_file_for_mlx_repl():
        print("✓ MLX REPL patch file created successfully")
    else:
        print("✗ Failed to create MLX REPL patch file")
        
    print("\n=== Patches Complete ===")
    print("\nTo apply the patch to your MLX REPL script, run:")
    print("    python mlx_repl_patch.py")
    print("\nTo use the monkey patch in your code:")
    print("    from abstractllm.providers.mlx_vision_patch import apply_patches")
    print("    apply_patches()")

if __name__ == "__main__":
    main() 