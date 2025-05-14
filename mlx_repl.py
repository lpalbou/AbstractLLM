#!/usr/bin/env python3
"""
Interactive REPL for testing MLX provider in AbstractLLM.
"""

import os
import sys
import cmd
import logging
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import platform
import traceback
import shutil

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
    from abstractllm.providers.mlx_provider import MLXProvider
    from abstractllm.providers.mlx_model_factory import MLXModelFactory
except ImportError:
    print("Failed to import AbstractLLM. Make sure it's installed.")
    sys.exit(1)

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Install with: pip install Pillow")

class MLXRepl(cmd.Cmd):
    """Interactive REPL for testing MLX models."""
    
    intro = """
    ╔════════════════════════════════════════════════════════╗
    ║                   AbstractLLM MLX REPL                 ║
    ╚════════════════════════════════════════════════════════╝
    
    Type 'help' or '?' to list commands.
    """
    prompt = "mlx> "
    
    # Known compatible vision models
    COMPATIBLE_VISION_MODELS = [
        "mlx-community/llava-1.5-7b-4bit",
        "mlx-community/bakllava-1-4bit",
        "mlx-community/llava-phi-2-vision-4bit",
        "mlx-community/phi-3-vision-128k-instruct-4bit"
    ]
    
    def __init__(self):
        super().__init__()
        self.llm = None
        self.model_name = "mlx-community/llava-1.5-7b-4bit"  # Default model
        self.images = []  # List of image paths
        self.max_tokens = 1024
        self.temperature = 0.2
        self.stream = False
        self._model_type = None  # Model type (e.g., "llava", "phi-vision", etc.)
        
        # Check MLX and PyTorch availability
        try:
            import mlx
            print("MLX successfully imported")
        except ImportError:
            print("ERROR: MLX not found. Install with: pip install mlx")
        
        try:
            import torch
            print(f"PyTorch found (version: {torch.__version__})")
        except ImportError:
            print("PyTorch not found (optional dependency for some vision models)")
        
        try:
            import huggingface_hub
            print(f"HuggingFace Hub found (version: {huggingface_hub.__version__})")
        except ImportError:
            print("HuggingFace Hub not found. Install with: pip install huggingface_hub")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    def do_exit(self, arg):
        """Exit the REPL."""
        print("Exiting...")
        return True
    
    def do_quit(self, arg):
        """Exit the REPL."""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Exit on Ctrl-D."""
        print("\nExiting...")
        return True
    
    def do_model(self, arg):
        """Set the model to use: model MODEL_NAME"""
        if not arg:
            print(f"Current model: {self.model_name}")
            if self._model_type:
                print(f"Model family: {self._model_type}")
            return
        
        # Set new model and clear existing LLM
        self.model_name = arg.strip()
        self.llm = None
        
        # Clear any caches
        MLXModelFactory.clear_cache()
        
        # Update the model type
        provider = MLXProvider({"model": self.model_name})
        self._model_type = provider._model_type if provider._is_vision_model else None
        
        print(f"Model set to: {self.model_name}")
        
        if self._model_type:
            print(f"Model family: {self._model_type}")
            
            # Get image size for this model type
            config = self._get_model_config()
            image_size = config["image_size"]
            prompt_format = config["prompt_format"]
            print(f"Image size: {image_size}")
            print(f"Prompt format: {prompt_format}")
            
            # If there are already images, ask if we should clear them
            if self.images:
                print("Model changed. Would you like to clear existing images? (y/n): ", end='')
                choice = input().lower()
                if choice == 'y':
                    self.images = []
                    print("Images cleared")
    
    def do_models(self, arg):
        """List available MLX models."""
        print("\nAvailable MLX models can be found at:")
        print("  - https://huggingface.co/mlx-community")
        print("  - https://huggingface.co/search?q=mlx")
        print("\nPopular models:")
        print("  1. mlx-community/llava-1.5-7b-4bit (vision)")
        print("  2. mlx-community/Mistral-7B-v0.1-4bit (text)")
        print("  3. mlx-community/Phi-3-mini-4bit-32k (text)")
        print("  4. mlx-community/Mixtral-8x7B-v0.1-4bit (text)")
        print("  5. mlx-community/phi-3-vision-128k-instruct-4bit (vision)")
    
    def do_list_compatible_vision(self, arg):
        """List vision models known to work well."""
        print("\nTested and Compatible Vision Models:")
        for i, model in enumerate(self.COMPATIBLE_VISION_MODELS, 1):
            print(f"  {i}. {model}")
        print("\nNote: Qwen2-VL models currently have compatibility issues with MLX")

    def do_image(self, arg):
        """Add an image to use with vision models: image PATH"""
        if not arg:
            if not self.images:
                print("No images added")
            else:
                print("Current images:")
                for i, img_path in enumerate(self.images, 1):
                    print(f"  {i}. {img_path}")
            return
        
        # Check if PIL is available for image processing
        if not PIL_AVAILABLE:
            print("Error: PIL (Pillow) is required for image processing.")
            print("Install with: pip install Pillow")
            return
        
        # Get path to image
        path = os.path.expanduser(arg.strip())
        if not os.path.exists(path):
            print(f"Error: Image not found at {path}")
            return
        
        # Check for model type to get target size
        if not self._model_type:
            # Create a temporary provider to detect model type
            provider = MLXProvider({"model": self.model_name})
            self._model_type = provider._model_type if provider._is_vision_model else None
            
            if not self._model_type:
                print("Current model does not support images. Use 'model' to set a vision model.")
                return
        
        # Get target size for current model
        config = self._get_model_config()
        target_size = config["image_size"]
        
        try:
            # Open and check original image size
            img = Image.open(path)
            orig_size = img.size
            print(f"Original image size: {orig_size[0]}x{orig_size[1]}")
            print(f"Target size for {self._model_type}: {target_size}")
            
            # Check if resizing is needed
            if orig_size[0] != target_size[0] or orig_size[1] != target_size[1]:
                print(f"Resizing image to {target_size}")
                
                # Create a properly resized copy of the image
                # This ensures the image is exactly the right size for the model
                resized_path = os.path.splitext(path)[0] + f"_{target_size[0]}x{target_size[1]}" + os.path.splitext(path)[1]
                
                # Create a new image with the right aspect ratio and padding
                new_img = Image.new("RGB", target_size, (0, 0, 0))
                
                # Resize maintaining aspect ratio
                ratio = min(target_size[0]/orig_size[0], target_size[1]/orig_size[1])
                new_size = (int(orig_size[0] * ratio), int(orig_size[1] * ratio))
                resized = img.resize(new_size, Image.LANCZOS)
                
                # Calculate position to center
                pos_x = (target_size[0] - new_size[0]) // 2
                pos_y = (target_size[1] - new_size[1]) // 2
                
                # Paste the resized image onto the black canvas
                new_img.paste(resized, (pos_x, pos_y))
                
                # Save the resized image
                new_img.save(resized_path)
                print(f"Created model-compatible image: {resized_path}")
                
                # Use the resized image
                self.images = [resized_path]
            else:
                print(f"Image already matches target size {target_size}")
                # Use the original image
                self.images = [path]
                
            print(f"Using image: {self.images[0]}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
    
    def do_clear_images(self, arg):
        """Clear all added images"""
        self.images = []
        print("Images cleared")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get the configuration for the current model type."""
        if not self._model_type:
            # Create a temporary provider to detect model type
            provider = MLXProvider({"model": self.model_name})
            self._model_type = provider._model_type if provider._is_vision_model else None
        
        # Use the MLXModelFactory to get the config
        return MLXModelFactory.get_model_config(self._model_type or "default")
    
    def do_create_test_image(self, arg):
        """Create a test image with appropriate dimensions for the current model."""
        if not PIL_AVAILABLE:
            print("Error: PIL (Pillow) is required for image processing.")
            print("Install with: pip install Pillow")
            return
        
        # Get target size for current model
        config = self._get_model_config()
        target_size = config["image_size"]
        model_type = self._model_type or "default"
        
        # Create a simple test image with text
        img = Image.new("RGB", target_size, (240, 240, 240))
        
        # Draw some shapes to make it interesting
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Add some shapes
            draw.rectangle([(20, 20), (target_size[0]-20, target_size[1]-20)], outline=(200, 0, 0), width=5)
            draw.ellipse([(50, 50), (target_size[0]-50, target_size[1]-50)], outline=(0, 0, 200), width=5)
            
            # Add diagonal lines
            draw.line([(20, 20), (target_size[0]-20, target_size[1]-20)], fill=(0, 150, 0), width=5)
            draw.line([(20, target_size[1]-20), (target_size[0]-20, 20)], fill=(0, 150, 0), width=5)
            
            # Add text
            text = f"Test Image {target_size[0]}x{target_size[1]}"
            
            # Try to get a font, use default if not available
            try:
                # Try to find a system font
                system_fonts = [
                    "/System/Library/Fonts/Helvetica.ttc",  # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                    "C:\\Windows\\Fonts\\arial.ttf"  # Windows
                ]
                
                font = None
                for font_path in system_fonts:
                    if os.path.exists(font_path):
                        try:
                            font = ImageFont.truetype(font_path, size=36)
                            break
                        except:
                            continue
                        
                if font is None:
                    # Use default font
                    font = ImageFont.load_default()
            except:
                # If error loading font, continue without text
                pass
            
            # Draw the text centered
            try:
                # Get text size if possible
                if hasattr(draw, "textbbox"):
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                else:
                    # Approximate for older PIL versions
                    text_width, text_height = font.getsize(text)
                    
                # Center the text
                position = ((target_size[0] - text_width) // 2, (target_size[1] - text_height) // 2)
                draw.text(position, text, fill=(0, 0, 0), font=font)
            except:
                # If error with text, continue without it
                pass
        except ImportError:
            # If no ImageDraw, create a simple gradient
            pixels = img.load()
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
        
        # Save the image
        filename = f"test_image_{model_type}_{target_size[0]}x{target_size[1]}.jpg"
        img.save(filename)
        print(f"Created test image: {filename}")
        
        # Use this image
        self.images = [filename]
        print(f"Now using test image: {filename}")
    
    def do_test_image(self, arg):
        """Show and test the current image."""
        if not self.images:
            print("No image set. Use 'image PATH' to set an image.")
            return
            
        image_path = self.images[0]
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return
            
        try:
            # Get image info
            img = Image.open(image_path)
            print(f"Image: {image_path}")
            print(f"Dimensions: {img.size[0]}x{img.size[1]}")
            print(f"Mode: {img.mode}")
            
            # Show image in default viewer if requested
            if arg.lower() == "show":
                print("Opening image in default viewer...")
                img.show()
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def do_info(self, arg):
        """Show information about the current setup."""
        # System info
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        
        # Model info
        print(f"Model: {self.model_name}")
        print(f"Model type: {self._model_type or 'N/A'}")
        
        # Images
        if self.images:
            print(f"Images: {len(self.images)}")
            for i, img_path in enumerate(self.images, 1):
                print(f"  {i}. {img_path}")
        else:
            print("Images: None")
        
        # Parameters
        print(f"Max tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        print(f"Streaming: {self.stream}")
        
        # Check MLX version
        try:
            import mlx
            print(f"MLX version: {mlx.__version__}")
        except (ImportError, AttributeError):
            print("MLX version: Unknown")
    
    def do_temp(self, arg):
        """Set temperature: temp 0.7"""
        try:
            temp = float(arg.strip())
            if 0 <= temp <= 1:
                self.temperature = temp
                print(f"Temperature set to {temp}")
            else:
                print("Temperature must be between 0 and 1")
        except ValueError:
            print(f"Current temperature: {self.temperature}")
    
    def do_tokens(self, arg):
        """Set max tokens: tokens 1024"""
        try:
            tokens = int(arg.strip())
            if tokens > 0:
                self.max_tokens = tokens
                print(f"Max tokens set to {tokens}")
            else:
                print("Max tokens must be positive")
        except ValueError:
            print(f"Current max tokens: {self.max_tokens}")
    
    def do_stream(self, arg):
        """Toggle streaming mode: stream [on|off]"""
        if not arg:
            self.stream = not self.stream
            print(f"Streaming mode: {'on' if self.stream else 'off'}")
        elif arg.lower() in ["on", "true", "1"]:
            self.stream = True
            print("Streaming mode: on")
        elif arg.lower() in ["off", "false", "0"]:
            self.stream = False
            print("Streaming mode: off")
    
    def _ensure_llm(self):
        """Ensure LLM is created with current settings"""
        if self.llm is None:
            print(f"Creating LLM with model: {self.model_name}")
            try:
                # Create the LLM with our MLX provider
                self.llm = create_llm("mlx", **{
                    ModelParameter.MODEL: self.model_name,
                    ModelParameter.MAX_TOKENS: self.max_tokens,
                    ModelParameter.TEMPERATURE: self.temperature
                })
                print("LLM created successfully")
            except Exception as e:
                print(f"Error creating LLM: {e}")
                traceback.print_exc()
                return False
        return True
    
    def do_debug_vision(self, arg):
        """Debug vision model capabilities."""
        provider = MLXProvider({"model": self.model_name})
        is_vision_model = provider._is_vision_model
        model_type = provider._model_type
        
        print(f"Model: {self.model_name}")
        print(f"Is vision model: {is_vision_model}")
        print(f"Model type: {model_type}")
        
        if is_vision_model:
            config = MLXModelFactory.get_model_config(model_type)
            print(f"Target image size: {config['image_size']}")
            print(f"Prompt format: {config['prompt_format']}")

            if self.images:
                image_path = self.images[0]
                print(f"Current image: {image_path}")
                
                try:
                    img = Image.open(image_path)
                    print(f"Image dimensions: {img.size}")
                    print(f"Image mode: {img.mode}")
                    
                    # Check if dimensions match target
                    if img.size != tuple(config['image_size']):
                        print("WARNING: Image dimensions don't match model's expected size")
                        print("Use 'create_test_image' to create a correctly sized test image")
                except Exception as e:
                    print(f"Error checking image: {e}")
            else:
                print("No image added. Use 'image' command to add an image.")
    
    def do_ask(self, arg):
        """Ask the model a question: ask PROMPT"""
        if not arg:
            print("Please provide a prompt. Usage: ask YOUR_PROMPT")
            return
        
        # Check if we need to create the LLM
        if not self._ensure_llm():
            return
        
        # Check if using a vision model with an image
        is_vision_model = hasattr(self.llm, "_provider") and self.llm._provider._is_vision_model

        # Warn if trying to use vision features without an image
        if is_vision_model and not self.images:
            print("\nWarning: You're using a vision model but haven't added any images.")
            print("Use the 'image' command to add an image or 'create_test_image' to create a test image.")
            
            print("Continue without images? (y/n): ", end='')
            choice = input().lower()
            if choice != 'y':
                return
        
        # Format prompt for vision models
        if self.images and is_vision_model:
            # We let the provider handle the formatting for us
            formatted_prompt = arg
            print(f"\nFormatted prompt with image tags: {formatted_prompt}")
        else:
            formatted_prompt = arg
            print(f"\nPrompt: {arg}")
            
        if self.images:
            print(f"Using image: {self.images[0]}")
            
        try:
            print("\nGenerating response...")
            if self.stream:
                print("\nResponse:")
                for chunk in self.llm.generate(formatted_prompt, files=self.images, stream=True):
                    print(chunk.content, end="", flush=True)
                print("\n")
            else:
                response = self.llm.generate(formatted_prompt, files=self.images)
                print(f"\nResponse: {response.content}")
                print(f"Usage: {response.usage}")
        except Exception as e:
            print(f"\nError during generation: {e}")
            traceback.print_exc()
            
            # Display helpful error messages
            error_str = str(e)
            if "tensor shape" in error_str.lower() or "dimensions" in error_str.lower() or "broadcast" in error_str.lower():
                print("\nThere's a mismatch between the image dimensions and what the model expects.")
                print("Try the following:")
                print("1. Use 'create_test_image' to create a model-specific test image")
                print("2. Clear your images with 'clear_images' and add only one image")
                print("3. Try a known compatible model like 'mlx-community/llava-1.5-7b-4bit'")
                
                print("\nWould you like to create a compatible test image? (y/n): ", end='')
                choice = input().lower()
                if choice == 'y':
                    self.do_create_test_image("")
            elif "image tag" in error_str.lower():
                print("\nThe model couldn't find the image tags in your prompt.")
                print("This usually happens when the prompt format doesn't match what the model expects.")
                print("Try the following:")
                print("1. Use 'create_test_image' to create a model-specific test image")
                print("2. Try a different model (list_compatible_vision shows known working models)")
                print("3. Make sure you're using the right prompt format for this model")

def main():
    """Main entry point"""
    # Check for MLX availability first
    try:
        import mlx
        import mlx_lm
        import mlx_vlm
    except ImportError:
        print("ERROR: MLX packages are not installed or are incompatible with your system.")
        print("Please install MLX and related packages:")
        print("  pip install mlx mlx-lm mlx-vlm")
        print("\nNote: MLX requires macOS with Apple Silicon (M1/M2/M3).")
        import platform
        if platform.system() != "Darwin" or "arm" not in platform.processor().lower():
            print(f"Your system ({platform.system()} {platform.processor()}) is not compatible with MLX.")
        sys.exit(1)
    
    # Check if we're on Apple Silicon
    import platform
    if platform.system() != "Darwin" or "arm" not in platform.processor().lower():
        print(f"WARNING: MLX requires macOS with Apple Silicon. Your system: {platform.system()} {platform.processor()}")
        print("The REPL may not function correctly.")
    
    # Start the REPL
    MLXRepl().cmdloop()

if __name__ == "__main__":
    main() 