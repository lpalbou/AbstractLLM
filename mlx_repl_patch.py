#!/usr/bin/env python3
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
    """Apply the patch to the MLX REPL."""
    print("=== MLX REPL Vision Patch ===")
    
    if patch_mlx_repl():
        print("✓ Successfully patched MLX REPL script")
    else:
        print("✗ Failed to patch MLX REPL script")
    
    print("\nNote: The patch fixes vision model issues by setting patch_size on processors")
    print("where it is missing, which causes image processing failures.")

if __name__ == "__main__":
    main()