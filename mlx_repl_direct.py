#!/usr/bin/env python3
"""
Interactive REPL for testing MLX provider in AbstractLLM.
This version adds direct transformers integration for Phi-3-Vision models.
"""

import os
import sys
import cmd
import logging
import platform
from pathlib import Path
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import AbstractLLM
try:
    from abstractllm import create_llm
    from abstractllm.enums import ModelParameter
except ImportError:
    print("Failed to import AbstractLLM. Make sure it's installed.")
    sys.exit(1)

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
class MLXRepl(cmd.Cmd):
    """Interactive REPL for testing MLX models."""
    
    intro = """
    ╔════════════════════════════════════════════════════════╗
    ║                   AbstractLLM MLX REPL                 ║
    ╚════════════════════════════════════════════════════════╝
    
    Type 'help' or '?' to list commands.
    """
    prompt = "mlx> "
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self.model_name = "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX"
        self.max_tokens = 100
        self.temperature = 0.7
        self.stream = False
        self.images = []
        self.direct_transformers = False
        
    def do_model(self, arg):
        """Set the model to use: model MODEL_NAME"""
        if not arg:
            print(f"Current model: {self.model_name}")
            return
        
        self.model_name = arg
        print(f"Model set to: {self.model_name}")
        # Reset LLM to force reload with new model
        self.llm = None
        
    def do_direct(self, arg):
        """Toggle using transformers directly for vision models instead of MLX-VLM"""
        if not arg:
            self.direct_transformers = not self.direct_transformers
        elif arg.lower() in ('on', 'true', 'yes', '1'):
            self.direct_transformers = True
        elif arg.lower() in ('off', 'false', 'no', '0'):
            self.direct_transformers = False
        else:
            print("Invalid argument. Use 'on'/'off' or no argument to toggle")
            return
            
        print(f"Direct transformers mode: {'ON' if self.direct_transformers else 'OFF'}")
        
    def do_temp(self, arg):
        """Set temperature: temp TEMPERATURE (0.0-1.0)"""
        if not arg:
            print(f"Current temperature: {self.temperature}")
            return
        
        try:
            temp = float(arg)
            if 0.0 <= temp <= 1.0:
                self.temperature = temp
                print(f"Temperature set to: {self.temperature}")
            else:
                print("Temperature must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid temperature value. Must be a number between 0.0 and 1.0")
            
    def do_tokens(self, arg):
        """Set maximum tokens: tokens MAX_TOKENS"""
        if not arg:
            print(f"Current max tokens: {self.max_tokens}")
            return
        
        try:
            tokens = int(arg)
            if tokens > 0:
                self.max_tokens = tokens
                print(f"Max tokens set to: {self.max_tokens}")
            else:
                print("Max tokens must be greater than 0")
        except ValueError:
            print("Invalid max tokens value. Must be a positive integer")
            
    def do_stream(self, arg):
        """Toggle streaming mode: stream [on|off]"""
        if not arg:
            self.stream = not self.stream
        elif arg.lower() in ('on', 'true', 'yes', '1'):
            self.stream = True
        elif arg.lower() in ('off', 'false', 'no', '0'):
            self.stream = False
        else:
            print("Invalid argument. Use 'on'/'off' or no argument to toggle")
            return
            
        print(f"Streaming mode: {'ON' if self.stream else 'OFF'}")
        
    def do_image(self, arg):
        """Add image for vision models: image PATH_TO_IMAGE"""
        if not arg:
            if not self.images:
                print("No images added")
            else:
                print("Current images:")
                for i, img in enumerate(self.images):
                    print(f"  {i+1}: {img}")
            return
            
        path = Path(arg)
        if not path.exists():
            print(f"Image not found: {path}")
            return
            
        # Check if we need to preprocess the image
        if PIL_AVAILABLE:
            try:
                # Check image size and potentially resize if too large
                img = Image.open(path)
                width, height = img.size
                
                # If image is very large, offer to resize
                if width > 1024 or height > 1024:
                    print(f"Image is large ({width}x{height}), which may cause issues with some models.")
                    resize = input("Resize to 512px max dimension? (y/n): ")
                    if resize.lower() in ('y', 'yes'):
                        # Resize while maintaining aspect ratio
                        if width > height:
                            new_width = 512
                            new_height = int((height / width) * 512)
                        else:
                            new_height = 512
                            new_width = int((width / height) * 512)
                            
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Create a new filename
                        resized_path = path.parent / f"{path.stem}_resized{path.suffix}"
                        img.save(resized_path)
                        print(f"Resized image saved to: {resized_path}")
                        
                        # Use the resized image instead
                        path = resized_path
            except Exception as e:
                print(f"Warning: Error checking/resizing image: {e}")
                
        self.images.append(str(path.absolute()))
        print(f"Added image: {path}")
        
    def do_clear_images(self, arg):
        """Clear all added images"""
        self.images = []
        print("All images cleared")
        
    def _ensure_llm(self):
        """Ensure LLM is created with current settings"""
        if self.llm is None:
            print(f"Creating LLM with model: {self.model_name}")
            try:
                self.llm = create_llm("mlx", **{
                    ModelParameter.MODEL: self.model_name,
                    ModelParameter.MAX_TOKENS: self.max_tokens,
                    ModelParameter.TEMPERATURE: self.temperature
                })
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
                
                print("LLM created successfully")
            except Exception as e:
                print(f"Error creating LLM: {e}")
                return False
        return True
        
    def _run_phi3_vision_direct(self, prompt: str, image_path: str) -> str:
        """Run Phi-3-Vision model directly using transformers."""
        print("\nUsing direct transformers approach for Phi-3-Vision...")
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            import mlx.core as mx
            
            # Make sure we're using the correct device
            device = "cpu"  # MLX handles this differently, but we'll use CPU for transformers
            
            # Load processor and model
            print("Loading model and processor...")
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModelForVision2Seq.from_pretrained(self.model_name)
            model.to(device)
            
            # Load and process image
            print(f"Processing image: {image_path}")
            image = Image.open(image_path)
            
            # Format prompt using the correct template for Phi-3-Vision
            formatted_prompt = f"User: {prompt}\nAssistant:"
            
            # Prepare inputs
            print("Preparing inputs...")
            inputs = processor(images=image, text=formatted_prompt, return_tensors="pt").to(device)
            
            # Generate
            print("Generating response...")
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature if self.temperature > 0 else None,
            )
            
            # Decode and return
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:", 1)[1].strip()
            
            return response
        except Exception as e:
            print(f"Error using transformers directly: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def _format_prompt_with_image_tags(self, prompt: str) -> str:
        """Format prompt with proper image tags based on model type."""
        model_name = self.model_name.lower()
        
        # Special handling for Phi-3 models
        if "phi-3" in model_name and "vision" in model_name:
            # Phi-3-Vision requires this specific format
            return f"Image: <image>\nUser: {prompt}\nAssistant:"
            
        # For LLaVA models
        elif "llava" in model_name:
            return f"<image>\n{prompt}"
            
        # For Qwen models
        elif "qwen" in model_name and ("vl" in model_name or "vision" in model_name):
            return f"<img>{prompt}"
            
        # For DeepSeek models
        elif "deepseek" in model_name and "vl" in model_name:
            return f"<img>{prompt}"
            
        # For StableLM models
        elif "stablelm" in model_name and "vision" in model_name:
            return f"<image>\nUser: {prompt}\nAssistant:"
            
        # For Llama 3 Vision models
        elif "llama" in model_name and ("vision" in model_name or "vl" in model_name):
            return f"<image>\nUser: {prompt}\nAssistant:"
            
        # For generic cases - try common format
        else:
            return f"<image>\n{prompt}"
        
    def do_list_compatible_vision(self, arg):
        """List known working vision models for MLX"""
        print("\nTested and Compatible Vision Models:")
        print("  1. mlx-community/llava-1.5-7b-4bit")
        print("  2. mlx-community/bakllava-1-4bit")
        print("  3. mlx-community/llava-phi-2-vision-4bit")
        print("  4. mlx-community/phi-3-vision-128k-instruct-4bit (with direct=on)")
        print("\nNote: Qwen2-VL models currently have compatibility issues with MLX")
        
    def do_ask(self, arg):
        """Ask the model a question: ask PROMPT"""
        if not arg:
            print("Please provide a prompt")
            return
        
        # Special handling for Phi-3-Vision with direct transformers
        is_phi3 = "phi-3" in self.model_name.lower() and "vision" in self.model_name.lower()
        if is_phi3 and self.direct_transformers and self.images:
            if len(self.images) > 1:
                print("Warning: Only using the first image with direct transformers mode")
            try:
                response_text = self._run_phi3_vision_direct(arg, self.images[0])
                print("\nResponse (direct transformers):")
                print(f"\n{response_text}")
                return
            except Exception as e:
                print(f"Direct transformers approach failed: {e}")
                print("Falling back to MLX-VLM approach...")
            
        if not self._ensure_llm():
            return
        
        # Check for PyTorch if using images and vision model
        if self.images:
            try:
                import torch
                has_torch = True
            except ImportError:
                has_torch = False
                
            if not has_torch:
                print("\nWARNING: PyTorch is not installed but required for vision models.")
                print("Install with: pip install torch")
                response = input("Continue anyway? (y/n): ")
                if response.lower() not in ('y', 'yes'):
                    return
        
        # Check for Qwen vision models which have compatibility issues
        if "qwen" in self.model_name.lower() and self.images:
            print("\nWARNING: Qwen vision models have compatibility issues with MLX.")
            print("Try using one of these compatible models instead:")
            print("  - mlx-community/llava-1.5-7b-4bit")
            print("  - mlx-community/bakllava-1-4bit")
            print("  - mlx-community/phi-3-vision-128k-instruct-4bit")
            response = input("Continue anyway? (y/n): ")
            if response.lower() not in ('y', 'yes'):
                return
            
        try:
            # Format prompt with image tags if needed
            formatted_prompt = arg
            if self.images:
                formatted_prompt = self._format_prompt_with_image_tags(arg)
                print(f"\nPrompt (with image tags): {formatted_prompt}")
            else:
                print(f"\nPrompt: {arg}")
                
            if self.images:
                print(f"Using {len(self.images)} image(s)")
                
            if self.stream:
                print("\nResponse:")
                for chunk in self.llm.generate(formatted_prompt, files=self.images, stream=True):
                    print(chunk.content, end="", flush=True)
                print("\n")
            else:
                print("\nGenerating response...")
                response = self.llm.generate(formatted_prompt, files=self.images)
                print(f"\nResponse: {response.content}")
                print(f"Usage: {response.usage}")
        except Exception as e:
            print(f"\nError during generation: {e}")
            # Print more detailed error handling suggestions
            if "torch" in str(e).lower() or "pytorch" in str(e).lower():
                print("\nThis error appears to be related to missing PyTorch.")
                print("Please install PyTorch with: pip install torch")
            
    def do_info(self, arg):
        """Show current settings"""
        print("\nCurrent Settings:")
        print(f"  Model: {self.model_name}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max Tokens: {self.max_tokens}")
        print(f"  Streaming: {'ON' if self.stream else 'OFF'}")
        print(f"  Direct transformers mode: {'ON' if self.direct_transformers else 'OFF'}")
        print(f"  Images: {len(self.images)}")
        for i, img in enumerate(self.images):
            print(f"    {i+1}: {img}")
            
    def do_models(self, arg):
        """List available MLX models from HuggingFace"""
        try:
            from huggingface_hub import HfApi
            print("\nFetching models from HuggingFace (mlx-community)...")
            api = HfApi()
            models = api.list_models(author='mlx-community')
            
            # Filter models based on optional argument
            if arg:
                filtered_models = [m for m in models if arg.lower() in m.id.lower()]
                models = filtered_models
                print(f"Filtering models with: {arg}")
            
            # Group models by type
            text_models = []
            vision_models = []
            
            vision_keywords = ['vl', 'vision', 'llava', 'clip', 'blip', 'image']
            
            for model in models:
                model_id = model.id.lower()
                if any(kw in model_id for kw in vision_keywords):
                    vision_models.append(model.id)
                else:
                    text_models.append(model.id)
            
            print(f"\nFound {len(text_models)} text models and {len(vision_models)} vision models")
            
            print("\nText Models:")
            for i, model_id in enumerate(text_models[:10]):
                print(f"  {i+1}. {model_id}")
            if len(text_models) > 10:
                print(f"  ... and {len(text_models) - 10} more")
                
            print("\nVision Models:")
            for i, model_id in enumerate(vision_models[:10]):
                print(f"  {i+1}. {model_id}")
            if len(vision_models) > 10:
                print(f"  ... and {len(vision_models) - 10} more")
                
        except ImportError:
            print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"Error listing models: {e}")
    
    def do_test_image(self, arg):
        """Test if an image can be processed correctly: test_image PATH_TO_IMAGE"""
        if not PIL_AVAILABLE:
            print("PIL not installed. Install with: pip install Pillow")
            return
            
        if not arg:
            print("Please provide an image path")
            return
            
        path = Path(arg)
        if not path.exists():
            print(f"Image not found: {path}")
            return
            
        try:
            # Open and display image information
            img = Image.open(path)
            width, height = img.size
            print(f"\nImage Information:")
            print(f"  Path: {path}")
            print(f"  Size: {width}x{height}")
            print(f"  Format: {img.format}")
            print(f"  Mode: {img.mode}")
            
            # Check if image size is appropriate
            if width > 1024 or height > 1024:
                print("  Status: ⚠️ Image is large and may cause issues with some models")
            else:
                print("  Status: ✓ Image size is acceptable")
                
            # Check if format is supported
            if img.format in ['JPEG', 'PNG', 'WebP']:
                print(f"  Format: ✓ {img.format} is well supported")
            else:
                print(f"  Format: ⚠️ {img.format} might have limited support")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            
    def do_debug_vision(self, arg):
        """Show debug information for vision models and check compatibility"""
        print("\n=== MLX Vision Model Debug Information ===")
        
        # Check if running on macOS with Apple Silicon
        print("\nSystem Requirements:")
        if sys.platform == "darwin":
            print("  ✓ Running on macOS")
            if platform.processor() == "arm":
                print("  ✓ Apple Silicon detected")
            else:
                print("  ✗ Not running on Apple Silicon (required for MLX)")
        else:
            print(f"  ✗ Not running on macOS (found {sys.platform})")
            
        # Check required packages
        print("\nRequired Packages:")
        
        # Check MLX
        try:
            import mlx
            print("  ✓ MLX installed")
        except ImportError:
            print("  ✗ MLX not installed")
            
        # Check MLX-LM
        try:
            import mlx_lm
            print("  ✓ MLX-LM installed")
        except ImportError:
            print("  ✗ MLX-LM not installed")
            
        # Check MLX-VLM
        try:
            import mlx_vlm
            print("  ✓ MLX-VLM installed")
        except ImportError:
            print("  ✗ MLX-VLM not installed")
            
        # Check PyTorch (critical for vision)
        try:
            import torch
            print(f"  ✓ PyTorch installed (version: {torch.__version__})")
        except ImportError:
            print("  ✗ PyTorch NOT installed (required for vision models)")
            
        # Check transformers (for direct mode)
        try:
            import transformers
            print(f"  ✓ Transformers installed (version: {transformers.__version__})")
        except ImportError:
            print("  ✗ Transformers NOT installed (required for direct mode)")
            
        # Check PIL
        if PIL_AVAILABLE:
            print("  ✓ PIL/Pillow installed")
        else:
            print("  ✗ PIL/Pillow not installed")
            
        # Current model information
        print("\nCurrent Model Settings:")
        print(f"  Model: {self.model_name}")
        print(f"  Direct transformers mode: {'ON' if self.direct_transformers else 'OFF'}")
        
        # Check if current model is a vision model
        is_vision = False
        vision_keywords = ['vl', 'vision', 'llava', 'clip', 'blip', 'image']
        for kw in vision_keywords:
            if kw in self.model_name.lower():
                is_vision = True
                break
                
        if is_vision:
            print("  ✓ Current model appears to be a vision model")
            # Check if it's in our known compatible list
            compatible_models = [
                "llava-1.5-7b-4bit",
                "bakllava-1-4bit", 
                "llava-phi-2-vision-4bit",
                "phi-3-vision-128k-instruct-4bit"
            ]
            is_compatible = any(cm in self.model_name.lower() for cm in compatible_models)
            if is_compatible:
                print("  ✓ Model appears to be in the compatible list")
            else:
                print("  ⚠ Model is not in the known compatible list (may still work)")
                
            # Check for problematic models
            if "qwen" in self.model_name.lower() and "vl" in self.model_name.lower():
                print("  ✗ This is a Qwen-VL model which has known compatibility issues with MLX")
                
            # Check if it's a Phi-3 model that works with direct mode
            if "phi-3" in self.model_name.lower() and "vision" in self.model_name.lower():
                print("  ℹ This is a Phi-3 Vision model which works best with direct mode")
                if self.direct_transformers:
                    print("  ✓ Direct transformers mode is ON (recommended)")
                else:
                    print("  ⚠ Direct transformers mode is OFF (recommended: ON)")
        else:
            print("  ℹ Current model does not appear to be a vision model")
            
        # Image information
        print("\nImage Information:")
        if not self.images:
            print("  No images currently added")
        else:
            print(f"  {len(self.images)} image(s) added")
            for i, img_path in enumerate(self.images):
                print(f"  Image {i+1}: {img_path}")
                if PIL_AVAILABLE:
                    try:
                        img = Image.open(img_path)
                        width, height = img.size
                        print(f"    Size: {width}x{height}")
                        print(f"    Format: {img.format}")
                        if width > 1024 or height > 1024:
                            print("    ⚠ Image is large and may cause issues")
                        else:
                            print("    ✓ Image size is acceptable")
                    except Exception as e:
                        print(f"    ✗ Error reading image: {e}")
                        
        # Prompt format
        print("\nPrompt Format:")
        if self.model_name.lower():
            if self.images:
                sample_prompt = "What's in this image?"
                formatted = self._format_prompt_with_image_tags(sample_prompt)
                print(f"  Sample prompt with image tag(s):\n    \"{formatted}\"")
            else:
                print("  No images added, so no image tags would be added to prompt")
                
        print("\n=== End of Debug Information ===")
        
    def do_exit(self, arg):
        """Exit the REPL"""
        print("\nExiting MLX REPL. Goodbye!")
        return True
        
    def do_quit(self, arg):
        """Exit the REPL"""
        return self.do_exit(arg)
        
    def do_EOF(self, arg):
        """Exit on Ctrl+D"""
        print()  # Add newline
        return self.do_exit(arg)

def main():
    """Main function to run the REPL."""
    # Check if running on macOS with Apple Silicon
    if sys.platform != "darwin":
        print("Warning: MLX requires macOS with Apple Silicon.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() not in ('y', 'yes'):
            sys.exit(1)
            
    try:
        import mlx
        print("MLX successfully imported")
    except ImportError:
        print("MLX not installed. Install with: pip install mlx mlx-lm mlx-vlm")
        sys.exit(1)
    
    # Check for PIL/Pillow for image processing    
    if not PIL_AVAILABLE:
        print("Warning: PIL/Pillow not installed. Image preprocessing features will be limited.")
        print("Install with: pip install Pillow")
    
    # Check for PyTorch (needed for vision models)
    try:
        import torch
        print(f"PyTorch found (version: {torch.__version__})")
    except ImportError:
        print("Warning: PyTorch not installed. Vision models will not work properly.")
        print("Install with: pip install torch")
    
    # Check for transformers (needed for direct mode)
    try:
        import transformers
        print(f"Transformers found (version: {transformers.__version__})")
    except ImportError:
        print("Warning: transformers not installed. Direct mode will not work.")
        print("Install with: pip install transformers")
    
    # Check for huggingface_hub
    try:
        import huggingface_hub
        print(f"HuggingFace Hub found (version: {huggingface_hub.__version__})")
    except (ImportError, AttributeError):
        print("Warning: huggingface_hub not installed. Model listing features will be limited.")
        print("Install with: pip install huggingface_hub")
    
    print("\nStarting MLX REPL. Type 'help' for available commands.")
        
    # Start the REPL
    MLXRepl().cmdloop()

if __name__ == "__main__":
    main() 