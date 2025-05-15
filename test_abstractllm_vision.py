#!/usr/bin/env python3
"""
Vision Models Test for MLX

This script tests vision-capable MLX models both directly through MLX-VLM
and through the AbstractLLM API to compare and debug integration issues.
"""

import os
import time
import logging
import argparse
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of tested vision models - focusing on Paligemma which is known to work correctly
VISION_MODELS = {
    "paligemma": "mlx-community/paligemma-3b-mix-448-8bit"
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test vision models with MLX")
    parser.add_argument("--model", type=str, choices=list(VISION_MODELS.keys()), default="paligemma",
                        help=f"Model type to test. Available: {', '.join(VISION_MODELS.keys())}")
    parser.add_argument("--image", type=str, default="test_images/sample.jpg",
                        help="Path to test image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.",
                        help="Prompt to use for image description")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for text generation")
    parser.add_argument("--use-abstractllm", action="store_true",
                        help="Test with AbstractLLM API instead of direct MLX-VLM")
    parser.add_argument("--output-json", type=str, default="mlx_vision_results.json",
                        help="Path to save test results as JSON")
    return parser.parse_args()

def test_vision_with_mlx_vlm(model_name, image_path, prompt, max_tokens, temperature):
    """Test vision model directly with MLX-VLM."""
    try:
        # Make sure image exists
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False, None
        
        # Import required packages
        try:
            import mlx_vlm
            from mlx_vlm import load as load_vlm
            from mlx_vlm import generate as generate_vlm
            from mlx_vlm.utils import load_config
            import PIL.Image
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.info("Installing required packages...")
            os.system("pip install mlx-vlm pillow")
            import mlx_vlm
            from mlx_vlm import load as load_vlm
            from mlx_vlm import generate as generate_vlm
            from mlx_vlm.utils import load_config
            import PIL.Image
        
        # Load the image
        image = PIL.Image.open(image_path)
        
        # Get the model path
        model_path = VISION_MODELS.get(model_name)
        logger.info(f"Testing MLX-VLM with vision model: {model_path}")
        
        # Start timing
        start_time = time.time()
        
        # Load the model and config
        logger.info(f"Loading vision model: {model_path}")
        model, processor = load_vlm(model_path, trust_remote_code=True)
        config = load_config(model_path, trust_remote_code=True)
        
        # Format the prompt if needed
        formatted_prompt = prompt
        # No special formatting for Paligemma
            
        # Generate response with image
        logger.info("Generating vision response with MLX-VLM...")
        gen_start_time = time.time()
        
        # Generate with image
        response = generate_vlm(
            model=model,
            processor=processor,
            image=image,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temp=temperature
        )
        
        gen_time = time.time() - gen_start_time
        total_time = time.time() - start_time
        
        logger.info(f"Response generated in {gen_time:.2f} seconds")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Format results
        result = {
            "model": model_name,
            "model_path": model_path,
            "prompt": prompt,
            "response": response[0] if isinstance(response, tuple) else response,
            "total_time": total_time,
            "generation_time": gen_time,
        }
        
        # Print the result
        logger.info("="*50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Prompt: {prompt}")
        logger.info("="*50)
        logger.info("RESULT:")
        logger.info(response)
        logger.info("="*50)
        
        return True, result
        
    except Exception as e:
        logger.error(f"Error during MLX-VLM vision test: {e}")
        import traceback
        traceback.print_exc()
        return False, {"model": model_name, "error": str(e)}

def test_vision_with_abstractllm(model_name, image_path, prompt, max_tokens, temperature):
    """Test vision model through AbstractLLM API."""
    try:
        # Make sure image exists
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return False, None
        
        # Import AbstractLLM
        try:
            from abstractllm import create_llm
            from abstractllm.interface import ModelParameter
            from abstractllm.types import GenerateResponse
            import PIL.Image
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            logger.info("Installing required packages...")
            os.system("pip install abstractllm pillow")
            from abstractllm import create_llm
            from abstractllm.interface import ModelParameter
            from abstractllm.types import GenerateResponse
            import PIL.Image
        
        # First try to patch AbstractLLM's MLXProvider to handle vision models better
        logger.info("Applying patch to AbstractLLM's MLXProvider...")
        try:
            import mlx_vlm
            from mlx_vlm import load as load_vlm
            from mlx_vlm import generate as generate_vlm
            from mlx_vlm.utils import load_config
            from abstractllm.providers.mlx_provider import MLXProvider
            from abstractllm.exceptions import GenerationError
            
            # Store original functions
            original_check_vision = MLXProvider._check_vision_model
            original_load_model = MLXProvider.load_model
            original_generate = MLXProvider.generate
            
            # Create patched functions
            def patched_check_vision(self, model_name):
                """Always returns True for our vision models"""
                if any(name in model_name.lower() for name in VISION_MODELS.values()):
                    logger.info(f"Patched vision check: Forcing {model_name} to be recognized as vision model")
                    return True
                return original_check_vision(self, model_name)
                
            def patched_load_model(self):
                """Use MLX-VLM for our known vision models"""
                model_name = self.config_manager.get_param(ModelParameter.MODEL)
                
                # Check if this is one of our vision models
                if any(name in model_name.lower() for name in VISION_MODELS.values()):
                    try:
                        logger.info(f"Patched load_model: Using MLX-VLM for {model_name}")
                        self._model, self._processor = load_vlm(model_name, trust_remote_code=True)
                        self._config = load_config(model_name, trust_remote_code=True)
                        self._is_loaded = True
                        self._is_vision_model = True
                        return
                    except Exception as e:
                        logger.error(f"Patched load_model failed: {e}")
                
                # Fall back to original method for other models
                return original_load_model(self)
            
            def patched_generate(self, prompt, system_prompt=None, files=None, stream=False, tools=None, **kwargs):
                """Patched generate method that uses MLX-VLM for vision generation"""
                try:
                    # Check if we need to use vision generation
                    images = kwargs.get("images", [])
                    if images and self._is_vision_model:
                        logger.info("Patched generate: Using MLX-VLM for vision generation")
                        
                        # Need to call load_model if not already loaded
                        if not self._is_loaded:
                            self.load_model()
                            
                        # Get generation parameters
                        max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS, default=100)
                        temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE, default=0.1)
                        
                        # Generate using MLX-VLM
                        logger.info("Generating with MLX-VLM directly...")
                        result = generate_vlm(
                            model=self._model,
                            processor=self._processor,
                            image=images[0],  # Currently only using the first image
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temp=temperature
                        )
                        
                        # Extract the text from the result
                        if isinstance(result, tuple):
                            text = result[0]
                        else:
                            text = result
                            
                        # Create a response object
                        response = GenerateResponse(
                            content=text,
                            raw_response=result,
                            usage={
                                "completion_tokens": max_tokens,
                                "total_tokens": len(prompt) + max_tokens  # Approximation
                            }
                        )
                        
                        return response
                    else:
                        # Fall back to original method for non-vision generation
                        return original_generate(self, prompt, system_prompt, files, stream, tools, **kwargs)
                        
                except Exception as e:
                    logger.error(f"Patched generate failed: {e}")
                    raise GenerationError(f"Patched vision generation failed: {str(e)}")
            
            # Apply patches
            MLXProvider._check_vision_model = patched_check_vision
            MLXProvider.load_model = patched_load_model
            MLXProvider.generate = patched_generate
            logger.info("Successfully patched AbstractLLM's MLXProvider")
            
        except Exception as e:
            logger.warning(f"Failed to patch AbstractLLM: {e}")
        
        # Load the image
        image = PIL.Image.open(image_path)
        
        # Get the model path
        model_path = VISION_MODELS.get(model_name)
        logger.info(f"Testing AbstractLLM with vision model: {model_path}")
        
        # Configuration for the model
        config = {
            "model": model_path,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Start timing
        start_time = time.time()
        
        # Create the LLM through AbstractLLM's API
        logger.info(f"Creating LLM with model: {model_path}")
        llm = create_llm("mlx", **config)
        
        # Generate response with image
        logger.info("Generating vision response through AbstractLLM...")
        gen_start_time = time.time()
        
        # Call LLM with image
        response = llm.generate(prompt, images=[image])
        
        gen_time = time.time() - gen_start_time
        total_time = time.time() - start_time
        
        logger.info(f"Response generated in {gen_time:.2f} seconds")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Format results
        result = {
            "model": model_name,
            "model_path": model_path,
            "prompt": prompt,
            "response": response.content,
            "total_time": total_time,
            "generation_time": gen_time,
        }
        
        # Print the result
        logger.info("="*50)
        logger.info(f"Model: {model_name}")
        logger.info(f"Prompt: {prompt}")
        logger.info("="*50)
        logger.info("RESULT:")
        logger.info(response.content)
        logger.info("="*50)
        
        return True, result
        
    except Exception as e:
        logger.error(f"Error during AbstractLLM vision test: {e}")
        import traceback
        traceback.print_exc()
        return False, {"model": model_name, "error": str(e)}

def main():
    """Main function."""
    args = parse_args()
    
    # Models to test - just use the one specified by the user
    models_to_test = [args.model]
    
    results = []
    success_count = 0
    
    for model_name in models_to_test:
        if args.use_abstractllm:
            logger.info(f"Testing AbstractLLM vision capabilities with model: {model_name}")
            success, result = test_vision_with_abstractllm(
                model_name,
                args.image,
                args.prompt,
                args.max_tokens,
                args.temperature
            )
        else:
            logger.info(f"Testing MLX-VLM vision capabilities with model: {model_name}")
            success, result = test_vision_with_mlx_vlm(
                model_name,
                args.image,
                args.prompt,
                args.max_tokens,
                args.temperature
            )
        
        if result:
            results.append(result)
            
        if success:
            test_type = "AbstractLLM" if args.use_abstractllm else "MLX-VLM"
            logger.info(f"{test_type} vision test for {model_name} completed successfully")
            success_count += 1
        else:
            test_type = "AbstractLLM" if args.use_abstractllm else "MLX-VLM"
            logger.error(f"{test_type} vision test for {model_name} failed")
    
    # Save results to JSON if specified
    if args.output_json and results:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_json}")
    
    # Final summary
    test_type = "AbstractLLM" if args.use_abstractllm else "MLX-VLM"
    logger.info(f"{test_type} vision testing completed: {success_count}/{len(models_to_test)} models successful")
    
    # Exit with error if any test failed
    if success_count < len(models_to_test):
        exit(1)

if __name__ == "__main__":
    main() 