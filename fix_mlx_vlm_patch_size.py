#!/usr/bin/env python3
"""
Patch for MLX-VLM to fix patch_size issues in LLaVA processor.

This script modifies the mlx_vlm library to better handle cases where
patch_size is None in the LLaVA processor.
"""

import os
import sys
import inspect
import logging
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_mlx_vlm_utils_path():
    """Find the path to mlx_vlm utils.py."""
    try:
        # Try to import mlx_vlm to find its path
        import mlx_vlm
        # Find the utils.py path
        mlx_vlm_path = os.path.dirname(inspect.getfile(mlx_vlm))
        utils_path = os.path.join(mlx_vlm_path, "utils.py")
        
        if os.path.exists(utils_path):
            return utils_path
        else:
            logger.error(f"utils.py not found at expected path: {utils_path}")
            return None
    except ImportError:
        logger.error("mlx_vlm is not installed. Please install it first.")
        return None
    except Exception as e:
        logger.error(f"Failed to find mlx_vlm utils.py: {e}")
        return None

def patch_prepare_inputs_function():
    """Patch the prepare_inputs function to handle None patch_size."""
    
    utils_path = find_mlx_vlm_utils_path()
    if not utils_path:
        return False
    
    logger.info(f"Found utils.py at: {utils_path}")
    
    try:
        # Create a backup of the original file
        backup_path = f"{utils_path}.backup"
        if not os.path.exists(backup_path):
            with open(utils_path, "r") as f_orig:
                content = f_orig.read()
            
            with open(backup_path, "w") as f_backup:
                f_backup.write(content)
            
            logger.info(f"Created backup at: {backup_path}")
        
        # Modify the file
        with open(utils_path, "r") as f:
            lines = f.readlines()
        
        # Find the prepare_inputs function
        in_prepare_inputs = False
        fixed = False
        
        for i, line in enumerate(lines):
            if "def prepare_inputs" in line:
                in_prepare_inputs = True
                logger.info(f"Found prepare_inputs function at line {i}")
            
            # Looking for the section where process_inputs_with_fallback is called
            if in_prepare_inputs and "process_inputs_with_fallback" in line:
                # Check if we need to insert the patch before this line
                if i > 0 and "processor.patch_size" not in lines[i-10:i]:
                    # Insert the patch before process_inputs_with_fallback is called
                    patch_code = [
                        "\n    # Patch to fix None patch_size issues\n",
                        "    if hasattr(processor, 'patch_size') and processor.patch_size is None:\n",
                        "        processor.patch_size = 14  # Default patch size for most vision models\n",
                        "    if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'patch_size') and processor.image_processor.patch_size is None:\n",
                        "        processor.image_processor.patch_size = 14\n\n"
                    ]
                    
                    lines[i:i] = patch_code
                    fixed = True
                    logger.info(f"Inserted patch before line {i}")
                    break
        
        if fixed:
            # Write the modified content back to the file
            with open(utils_path, "w") as f:
                f.writelines(lines)
            
            logger.info("Successfully patched mlx_vlm utils.py")
            return True
        else:
            logger.warning("Could not find the right location to insert the patch")
            return False
            
    except Exception as e:
        logger.error(f"Failed to patch utils.py: {e}")
        return False

def patch_process_inputs_function():
    """Add a patch to the process_inputs_with_fallback function."""
    
    utils_path = find_mlx_vlm_utils_path()
    if not utils_path:
        return False
    
    try:
        # Check if we already have a backup
        backup_path = f"{utils_path}.backup2"
        if not os.path.exists(backup_path):
            with open(utils_path, "r") as f_orig:
                content = f_orig.read()
            
            with open(backup_path, "w") as f_backup:
                f_backup.write(content)
            
            logger.info(f"Created backup at: {backup_path}")
        
        # Read the file
        with open(utils_path, "r") as f:
            lines = f.readlines()
        
        # Find the process_inputs_with_fallback function
        in_process_inputs = False
        fixed = False
        
        for i, line in enumerate(lines):
            if "def process_inputs_with_fallback" in line:
                in_process_inputs = True
                logger.info(f"Found process_inputs_with_fallback function at line {i}")
                
                # Add the patch at the start of the function
                j = i + 1  # Skip the function definition line
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("try:"):
                    j += 1
                
                if j < len(lines):
                    patch_code = [
                        "\n    # Check and fix patch_size before processing\n",
                        "    if hasattr(processor, 'patch_size') and processor.patch_size is None:\n",
                        "        processor.patch_size = 14  # Default patch size\n",
                        "    if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'patch_size'):\n",
                        "        if processor.image_processor.patch_size is None:\n",
                        "            processor.image_processor.patch_size = 14\n\n"
                    ]
                    
                    lines[j:j] = patch_code
                    fixed = True
                    logger.info(f"Inserted patch at line {j}")
                    break
        
        if fixed:
            # Write the modified content back to the file
            with open(utils_path, "w") as f:
                f.writelines(lines)
            
            logger.info("Successfully patched process_inputs_with_fallback function")
            return True
        else:
            logger.warning("Could not find the right location to insert the patch")
            return False
            
    except Exception as e:
        logger.error(f"Failed to patch process_inputs_with_fallback: {e}")
        return False

def main():
    """Main entry point."""
    print("MLX-VLM Patcher for LLaVA Patch Size Issues")
    print("===========================================")
    
    success1 = patch_prepare_inputs_function()
    success2 = patch_process_inputs_function()
    
    if success1 and success2:
        print("\nSuccessfully patched MLX-VLM. The patch_size issue should be fixed.")
        print("You'll need to restart any running Python processes for the changes to take effect.")
    elif success1 or success2:
        print("\nPartially patched MLX-VLM. Some issues may persist.")
    else:
        print("\nFailed to patch MLX-VLM. Please check the error messages above.")
    
    # Suggest reloading mlx_vlm if it's already imported
    print("\nTo apply the patch to the current Python session, run the following code:")
    print("  import importlib")
    print("  import mlx_vlm")
    print("  importlib.reload(mlx_vlm.utils)")

if __name__ == "__main__":
    main() 