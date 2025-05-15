#!/usr/bin/env python3
"""
Minimal REPL for interacting with MLX models.

Usage:
    python mlx_chat.py --model "mlx-community/llama-3-8b-instruct-4bit"
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import requests
import json
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from abstractllm import create_llm
    from abstractllm.enums import ModelParameter
    from abstractllm.providers.mlx_provider import MLXProvider
except ImportError:
    logger.error("AbstractLLM not found. Please install it or ensure it's in your PYTHONPATH.")
    sys.exit(1)

# Default models to suggest
POPULAR_TEXT_MODELS = [
    "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
    "mlx-community/llama-3-8b-instruct-4bit",
    "mlx-community/Phi-3-mini-4bit-32k",
    "mlx-community/Mistral-7B-v0.1-4bit"
]

POPULAR_VISION_MODELS = [
    "mlx-community/llava-1.5-7b-4bit",
    "mlx-community/phi-3-vision-128k-instruct-4bit",
    "mlx-community/bakllava-1-4bit"
]

def check_dependencies():
    """Check for necessary dependencies."""
    try:
        import mlx
        logger.info(f"Found MLX: {getattr(mlx, '__version__', 'unknown version')}")
    except ImportError:
        logger.error("MLX not found. Install with: pip install mlx")
        return False
    
    try:
        import mlx_lm
        logger.info(f"Found MLX-LM: {getattr(mlx_lm, '__version__', 'unknown version')}")
    except ImportError:
        logger.error("MLX-LM not found. Install with: pip install mlx-lm")
        return False
    
    try:
        import huggingface_hub
        logger.info(f"Found huggingface_hub: {getattr(huggingface_hub, '__version__', 'unknown version')}")
    except ImportError:
        logger.warning("huggingface_hub not installed. Model listing may be limited.")
    
    return True

def search_models(keyword: str) -> List[Dict[str, Any]]:
    """Search for MLX models on HuggingFace Hub."""
    try:
        # Use HuggingFace Hub API to search for models
        url = "https://huggingface.co/api/models"
        params = {
            "search": f"mlx {keyword}",
            "limit": 20,
            "full": "true"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json()
            # Filter to models that are likely MLX models
            mlx_models = [
                model for model in results 
                if "mlx" in model.get("id", "").lower() or 
                   any("mlx" in tag.lower() for tag in model.get("tags", []))
            ]
            return mlx_models
        else:
            logger.warning(f"Failed to search HuggingFace Hub: {response.status_code}")
            return []
    except Exception as e:
        logger.warning(f"Error searching for models: {e}")
        return []

def is_vision_model(model_name):
    """Check if a model is likely to be a vision model based on its name."""
    vision_indicators = [
        "vlm", "vision", "visual", "llava", "clip", "multimodal", 
        "vit", "blip", "vqa", "image", "qwen-vl", "phi-vision"
    ]
    return any(indicator in model_name.lower() for indicator in vision_indicators)

def process_image_command(command, llm):
    """Process an image command and return the image path."""
    if not command.startswith("/image "):
        return None
        
    image_path = command[7:].strip()
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
        
    # Check if the model supports images
    if not llm.get_capabilities().get("vision", False):
        print("Error: The current model does not support image inputs.")
        return None
        
    print(f"Using image: {image_path}")
    return image_path

def create_test_image(size=(336, 336)):
    """Create a test image with shapes for vision models."""
    try:
        from PIL import Image, ImageDraw
        
        # Create a new image with the specified size
        img = Image.new("RGB", size, (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Draw shapes
        draw.rectangle([(20, 20), (size[0]-20, size[1]-20)], outline=(200, 0, 0), width=5)
        draw.ellipse([(50, 50), (size[0]-50, size[1]-50)], outline=(0, 0, 200), width=5)
        draw.line([(20, 20), (size[0]-20, size[1]-20)], fill=(0, 150, 0), width=5)
        draw.line([(20, size[1]-20), (size[0]-20, 20)], fill=(0, 150, 0), width=5)
        
        # Add text
        text = f"Test Image {size[0]}x{size[1]}"
        draw.text((size[0]//2 - 80, size[1]//2 - 10), text, fill=(0, 0, 0))
        
        # Save the image
        filename = f"test_image_{size[0]}x{size[1]}.jpg"
        img.save(filename)
        print(f"Created test image: {filename}")
        return filename
    except ImportError:
        print("Error: PIL (Pillow) is required for image creation.")
        print("Install with: pip install Pillow")
        return None
    except Exception as e:
        print(f"Error creating test image: {e}")
        return None

def load_model(model_name: str, temperature: float = 0.7, max_tokens: int = 1024) -> Optional[Any]:
    """Load an MLX model."""
    try:
        print(f"Loading model: {model_name}...")
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.TEMPERATURE: temperature,
            ModelParameter.MAX_TOKENS: max_tokens
        })
        print("Model loaded successfully!")
        
        # Show capabilities
        caps = llm.get_capabilities()
        print(f"Model capabilities: Vision={caps.get('vision', False)}, Streaming={caps.get('streaming', False)}")
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def print_help():
    """Print help information."""
    print("Special commands:")
    print("  /quit               - Exit the chat")
    print("  /help               - Show this help message")
    print("  /model <model_name> - Switch to a different model")
    print("  /currentmodel       - Show the current model being used")
    print("  /models <keyword>   - List MLX models matching a keyword")
    print("  /image <path>       - Send an image (for vision models)")
    print("  /clearimage         - Clear the current image")
    print("  /testimage          - Create a test image for vision models")
    print("  /temp <value>       - Set temperature (0-1)")
    print("  /tokens <value>     - Set maximum tokens to generate")

def main():
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Chat with an MLX model")
    parser.add_argument("--model", type=str, 
                        default="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
                        help="MLX model to use")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Temperature (0-1)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--system", type=str, default="",
                        help="System prompt")
    args = parser.parse_args()
    
    # Show banner
    print("\n" + "="*60)
    print(f"MLX Chat - Model: {args.model}")
    print("="*60)
    print("Type your messages and press Enter to chat.")
    print("Type /help for available commands.")
    
    # Create LLM
    llm = load_model(args.model, args.temp, args.max_tokens)
    if not llm:
        return 1
    
    # Store current parameters
    current_model = args.model
    current_temp = args.temp
    current_max_tokens = args.max_tokens
    current_system_prompt = args.system
    current_image = None
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input == "/quit":
                print("Goodbye!")
                break
                
            elif user_input == "/help":
                print_help()
                continue
                
            elif user_input == "/currentmodel":
                print(f"Current model: {current_model}")
                print(f"Temperature: {current_temp}")
                print(f"Max tokens: {current_max_tokens}")
                continue
                
            elif user_input.startswith("/image "):
                current_image = process_image_command(user_input, llm)
                continue
                
            elif user_input == "/clearimage":
                current_image = None
                print("Image cleared")
                continue
                
            elif user_input == "/testimage":
                # Create a test image with default size
                current_image = create_test_image()
                continue
                
            elif user_input.startswith("/models "):
                keyword = user_input[8:].strip()
                print(f"\nSearching for MLX models with keyword: {keyword}")
                models = search_models(keyword)
                
                if models:
                    print(f"\nFound {len(models)} models:")
                    for i, model in enumerate(models, 1):
                        model_id = model.get("id", "")
                        downloads = model.get("downloads", 0)
                        tags = ", ".join(model.get("tags", [])[:3])  # Show first 3 tags
                        print(f"  {i}. {model_id} - Downloads: {downloads}, Tags: {tags}")
                else:
                    print("\nNo matching models found. Try these popular models:")
                    print("\nText models:")
                    for model in POPULAR_TEXT_MODELS:
                        print(f"  - {model}")
                    print("\nVision models:")
                    for model in POPULAR_VISION_MODELS:
                        print(f"  - {model}")
                continue
                
            elif user_input.startswith("/model "):
                new_model = user_input[7:].strip()
                print(f"\nSwitching to model: {new_model}")
                
                new_llm = load_model(new_model, current_temp, current_max_tokens)
                if new_llm:
                    llm = new_llm
                    current_model = new_model
                    
                    # Clear image if switching between vision/non-vision models
                    if current_image and not llm.get_capabilities().get("vision", False):
                        print("Cleared image as new model doesn't support vision")
                        current_image = None
                continue
                
            elif user_input.startswith("/temp "):
                try:
                    temp = float(user_input[6:].strip())
                    if 0 <= temp <= 1:
                        current_temp = temp
                        print(f"Temperature set to {temp}")
                        
                        # Reload model with new parameters
                        llm = load_model(current_model, current_temp, current_max_tokens)
                    else:
                        print("Temperature must be between 0 and 1")
                except ValueError:
                    print("Invalid temperature value")
                continue
                
            elif user_input.startswith("/tokens "):
                try:
                    tokens = int(user_input[8:].strip())
                    if tokens > 0:
                        current_max_tokens = tokens
                        print(f"Max tokens set to {tokens}")
                        
                        # Reload model with new parameters
                        llm = load_model(current_model, current_temp, current_max_tokens)
                    else:
                        print("Max tokens must be positive")
                except ValueError:
                    print("Invalid max tokens value")
                continue
                
            elif not user_input:
                continue
                
            # Generate response
            print("\nThinking...")
            files = [current_image] if current_image else None
            
            try:
                response = llm.generate(
                    prompt=user_input,
                    system_prompt=current_system_prompt if current_system_prompt else None,
                    files=files
                )
                
                # Print response
                print(f"\nModel: {response.content}")
                
                # Print token usage
                if hasattr(response, "usage") and response.usage:
                    print(f"\n(Tokens used: {response.usage.get('total_tokens', 'unknown')})")
            except Exception as e:
                print(f"\nError: {e}")
                
        except KeyboardInterrupt:
            print("\nInterrupted. Type /quit to exit.")
        except EOFError:
            print("\nGoodbye!")
            break
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 