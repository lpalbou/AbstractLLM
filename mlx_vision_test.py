#!/usr/bin/env python3
"""
MLX Vision Test Script

This script tests the vision capabilities of the MLX provider by loading
a vision model and generating a description of a test image.
"""

import os
import argparse
import time
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of MLX-compatible vision models to test
VISION_MODELS = {
    "phi": "mlx-community/Phi-3.5-vision-instruct-4bit",
    "qwen": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "paligemma": "mlx-community/paligemma-3b-mix-448-8bit"
    # Add more vision models here as needed
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test MLX vision capabilities")
    parser.add_argument("--model", type=str, choices=VISION_MODELS.keys(), default="paligemma",
                        help=f"Model type to test. Available: {', '.join(VISION_MODELS.keys())}")
    parser.add_argument("--image", type=str, default="test_images/sample.jpg",
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Prompt to use for image description")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation")
    return parser.parse_args()

def test_vision_capability(model_name, image_path, prompt, max_tokens, temperature):
    """Test the vision capabilities of the MLX provider."""
    try:
        # Make sure image exists
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False
        
        logger.info(f"Testing model: {model_name}")
        
        # Check for MLX availability
        try:
            import mlx_vlm
            logger.info("MLX VLM version: " + mlx_vlm.__version__)
        except ImportError:
            logger.error("MLX VLM package not found. Installing...")
            os.system("pip install mlx mlx-vlm==0.1.26 pillow")
            import mlx_vlm
            logger.info("MLX VLM package installed")
        
        # Run the MLX VLM command directly
        start_time = time.time()
        
        # Determine if we need to set trust_remote_code based on the model
        trust_remote_code = "phi" in model_name.lower()
        
        # Create a temporary script to run the MLX VLM generate command
        temp_script = """
#!/usr/bin/env python3
import sys
from mlx_vlm import load, generate
from PIL import Image
import os

# Load the model
model_path = "{model_name}"
image_path = "{image_path}"
prompt = "{prompt}"
max_tokens = {max_tokens}
temperature = {temperature}
trust_remote_code = {trust_remote_code}

try:
    print("Loading model...")
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    
    # Use the direct generate function
    print("Generating...")
    
    # Different handling for different models
    if "paligemma" in model_path.lower():
        # Paligemma has a different format
        formatted_prompt = "<image>" + prompt
        output = generate(model, processor, formatted_prompt, [image_path], max_tokens=max_tokens, temperature=temperature)
    elif "phi" in model_path.lower():
        # Phi model format - requires special handling with image tags
        # Format for Phi: Use <image_x> tags where x is 0-based index
        formatted_prompt = "<image_0> " + prompt
        
        # For debugging
        print(f"Using formatted prompt for Phi: {{formatted_prompt}}")
        
        try:
            from transformers import Phi3VisionProcessor
            
            # Create the processor directly to debug
            image_processor = Phi3VisionProcessor.from_pretrained(model_path, trust_remote_code=True)
            image = Image.open(image_path)
            
            # Process the image and text directly
            inputs = image_processor(images=image, text=formatted_prompt, return_tensors="pt")
            print("Processor inputs created successfully")
            
            # Continue with generate
            output = generate(model, processor, formatted_prompt, [image_path], max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            print(f"Error with Phi model: {{e}}")
            # Try alternative approach for Phi
            print("Trying alternative approach...")
            
            if os.path.isdir("/tmp/phi_test"):
                import shutil
                shutil.rmtree("/tmp/phi_test")
            
            os.makedirs("/tmp/phi_test", exist_ok=True)
            
            # Copy the image to a predictable location
            from shutil import copyfile
            new_image_path = "/tmp/phi_test/image_0.jpg"
            copyfile(image_path, new_image_path)
            
            # Use a system prompt that works with Phi
            system_prompt = "You are a helpful assistant that can see images and answer questions about them."
            user_message = f"<image_0>\\n{{prompt}}"
            
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
            
            config = load_config(model_path)
            messages = [
                {{"role": "system", "content": system_prompt}},
                {{"role": "user", "content": user_message}}
            ]
            
            try:
                # Try with manual message formatting
                formatted_prompt = "\\n".join([
                    f"<|role|>system\\n<|content|>{{system_prompt}}",
                    f"<|role|>user\\n<|content|><image_0>\\n{{prompt}}",
                    f"<|role|>assistant\\n<|content|>"
                ])
                output = generate(model, processor, formatted_prompt, [new_image_path], max_tokens=max_tokens, temperature=temperature)
            except Exception as inner_e:
                print(f"Error with alternative approach: {{inner_e}}")
                raise
    else:
        # Other models like Qwen2-VL
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        config = load_config(model_path)
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
        output = generate(model, processor, formatted_prompt, [image_path], max_tokens=max_tokens, temperature=temperature)
    
    print("\\nRESULT:")
    print(output)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
""".format(
            model_name=model_name,
            image_path=str(image_path),
            prompt=prompt.replace('"', '\\"'),
            max_tokens=max_tokens,
            temperature=temperature,
            trust_remote_code=str(trust_remote_code)
        )
        
        # Write the script to a temporary file
        temp_file = "temp_vision_test.py"
        with open(temp_file, "w") as f:
            f.write(temp_script)
        
        # Make it executable
        os.chmod(temp_file, 0o755)
        
        # Run the script
        logger.info(f"Running temporary script to test vision capabilities")
        result = subprocess.run([sys.executable, temp_file], capture_output=True, text=True)
        
        # Clean up the temporary file
        os.remove(temp_file)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Check if the command was successful
        if result.returncode != 0:
            logger.error(f"Script failed with error: {result.stderr}")
            return False
        
        # Extract the model's response from the output
        output = result.stdout
        
        logger.info(f"Generation completed in {elapsed_time:.2f} seconds")
        
        # Print the result
        logger.info("="*50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Prompt: {prompt}")
        logger.info("="*50)
        logger.info("RESULT:")
        logger.info(output)
        logger.info("="*50)
        
        return True
    
    except Exception as e:
        logger.error(f"Error during vision test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Get the actual model path from the model name
    model_path = VISION_MODELS[args.model]
    
    logger.info(f"Testing MLX vision capabilities with model: {model_path}")
    
    # Run the test
    success = test_vision_capability(
        model_path,
        args.image,
        args.prompt,
        args.max_tokens,
        args.temperature
    )
    
    if success:
        logger.info("Vision test completed successfully")
    else:
        logger.error("Vision test failed")
        exit(1)

if __name__ == "__main__":
    main() 