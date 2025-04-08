"""
Image processing utilities for vision-enabled LLM providers.
"""

import os
import base64
import logging
from io import BytesIO
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
import mimetypes

# Configure logger
logger = logging.getLogger("abstractllm.utils.image")

def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """
    Encode an image from a file path to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded image string
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image cannot be encoded
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            logger.debug(f"Successfully encoded image: {image_path}")
            return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        raise ValueError(f"Cannot encode image: {e}")

def get_image_mime_type(image_path: Union[str, Path]) -> str:
    """
    Determine the MIME type of an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type of the image (e.g., 'image/jpeg')
    """
    # Try to guess the MIME type from the file extension
    mime_type, _ = mimetypes.guess_type(str(image_path))
    
    # Default to image/jpeg if we couldn't determine the type
    if not mime_type:
        # Use the file extension to make a best guess
        ext = Path(image_path).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_map.get(ext, 'image/jpeg')
        
    return mime_type

def format_image_for_provider(
    image_input: Union[str, Path, Dict[str, Any]], 
    provider: str
) -> Any:
    """
    Format an image for a specific provider's API format.
    
    Args:
        image_input: Image input (URL, file path, base64 string, or dict with provider-specific format)
        provider: Provider name ('openai', 'anthropic', 'ollama')
        
    Returns:
        Properly formatted image data for the specified provider
    """
    # If it's already a dict, assume it's already formatted for the provider
    if isinstance(image_input, dict):
        return image_input
        
    # Handle file paths vs URLs
    if isinstance(image_input, (str, Path)):
        path = Path(image_input) if not isinstance(image_input, Path) else image_input
        
        # Check if it's a URL
        if str(image_input).startswith(('http://', 'https://')):
            # Handle as URL
            url = str(image_input)
            
            if provider == "openai":
                return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
                
            elif provider == "anthropic":
                return {"type": "image", "source": {"type": "url", "url": url}}
                
            elif provider == "ollama":
                return {"url": url}
                
            else:
                logger.warning(f"Unknown provider {provider}, returning URL as-is")
                return url
                
        # Handle as file path
        elif path.exists():
            # Encode image to base64
            encoded_image = encode_image_to_base64(path)
            mime_type = get_image_mime_type(path)
            
            if provider == "openai":
                return {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}", 
                        "detail": "auto"
                    }
                }
                
            elif provider == "anthropic":
                return {
                    "type": "image", 
                    "source": {
                        "type": "base64", 
                        "media_type": mime_type, 
                        "data": encoded_image
                    }
                }
                
            elif provider == "ollama":
                return {"data": encoded_image}
                
            else:
                logger.warning(f"Unknown provider {provider}, returning base64 encoded data")
                return encoded_image
                
        else:
            # If it's not a URL and not a file, assume it's already a base64 string
            encoded_image = str(image_input)
            
            # Try to determine if it has a MIME type prefix
            if encoded_image.startswith('data:'):
                # Already formatted as a data URL
                if provider == "openai":
                    return {"type": "image_url", "image_url": {"url": encoded_image, "detail": "auto"}}
                    
                elif provider == "anthropic":
                    # Extract the MIME type and base64 data
                    parts = encoded_image.split(';')
                    mime_type = parts[0].split(':')[1]
                    data = parts[1].split(',')[1]
                    
                    return {
                        "type": "image", 
                        "source": {
                            "type": "base64", 
                            "media_type": mime_type, 
                            "data": data
                        }
                    }
                    
                elif provider == "ollama":
                    # Extract just the base64 data
                    data = encoded_image.split(',')[1]
                    return {"data": data}
                    
                else:
                    logger.warning(f"Unknown provider {provider}, returning as-is")
                    return encoded_image
            else:
                # Assume it's just base64 data without MIME type
                if provider == "openai":
                    # Default to JPEG if we can't determine the type
                    return {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}", 
                            "detail": "auto"
                        }
                    }
                    
                elif provider == "anthropic":
                    return {
                        "type": "image", 
                        "source": {
                            "type": "base64", 
                            "media_type": "image/jpeg", 
                            "data": encoded_image
                        }
                    }
                    
                elif provider == "ollama":
                    return {"data": encoded_image}
                    
                else:
                    logger.warning(f"Unknown provider {provider}, returning as-is")
                    return encoded_image
    
    # If we got here, the input format is not supported
    logger.error(f"Unsupported image input format: {type(image_input)}")
    raise ValueError(f"Unsupported image input format: {type(image_input)}")

def format_images_for_provider(
    images: List[Union[str, Path, Dict[str, Any]]], 
    provider: str
) -> List[Any]:
    """
    Format multiple images for a specific provider's API format.
    
    Args:
        images: List of image inputs (URLs, file paths, base64 strings, or dicts)
        provider: Provider name ('openai', 'anthropic', 'ollama')
        
    Returns:
        List of properly formatted image data for the specified provider
    """
    return [format_image_for_provider(img, provider) for img in images]

def preprocess_image_inputs(
    params: Dict[str, Any], 
    provider: str
) -> Dict[str, Any]:
    """
    Preprocess image inputs in the params dictionary for a specific provider.
    
    Args:
        params: Parameters dictionary that may contain image inputs
        provider: Provider name ('openai', 'anthropic', 'ollama')
        
    Returns:
        Updated parameters dictionary with properly formatted image data
    """
    # Make a copy of the parameters to avoid modifying the original
    processed_params = params.copy()
    
    # Handle single image parameter
    if "image" in processed_params or "IMAGE" in processed_params:
        image = processed_params.pop("image", processed_params.pop("IMAGE", None))
        if image is not None:
            formatted_image = format_image_for_provider(image, provider)
            
            # Add to the appropriate parameter based on provider
            if provider == "openai":
                # For OpenAI, images go in the 'content' field of messages
                if "messages" not in processed_params:
                    processed_params["messages"] = []
                
                # Find the user message or create one
                user_msg_idx = None
                for i, msg in enumerate(processed_params.get("messages", [])):
                    if msg.get("role") == "user":
                        user_msg_idx = i
                        break
                
                if user_msg_idx is not None:
                    # Update existing user message
                    msg = processed_params["messages"][user_msg_idx]
                    if isinstance(msg.get("content"), str):
                        # Convert string content to list format
                        text_content = msg["content"]
                        msg["content"] = [
                            {"type": "text", "text": text_content},
                            formatted_image
                        ]
                    elif isinstance(msg.get("content"), list):
                        # Add to existing content list
                        msg["content"].append(formatted_image)
                else:
                    # Create a new user message
                    processed_params["messages"].append({
                        "role": "user",
                        "content": [formatted_image]
                    })
            
            elif provider == "anthropic":
                # For Anthropic, images go in the 'content' field of messages
                if "messages" not in processed_params:
                    processed_params["messages"] = []
                
                # Find the user message or create one
                user_msg_idx = None
                for i, msg in enumerate(processed_params.get("messages", [])):
                    if msg.get("role") == "user":
                        user_msg_idx = i
                        break
                
                if user_msg_idx is not None:
                    # Update existing user message
                    msg = processed_params["messages"][user_msg_idx]
                    if isinstance(msg.get("content"), str):
                        # Convert string content to list format
                        text_content = msg["content"]
                        msg["content"] = [
                            {"type": "text", "text": text_content},
                            formatted_image
                        ]
                    elif isinstance(msg.get("content"), list):
                        # Add to existing content list
                        msg["content"].append(formatted_image)
                else:
                    # Create a new user message
                    processed_params["messages"].append({
                        "role": "user",
                        "content": [formatted_image]
                    })
            
            elif provider == "ollama":
                # For Ollama, images are added as a separate parameter
                processed_params["images"] = [formatted_image]
    
    # Handle multiple images parameter
    if "images" in processed_params or "IMAGES" in processed_params:
        images = processed_params.pop("images", processed_params.pop("IMAGES", None))
        if images is not None and isinstance(images, list):
            formatted_images = format_images_for_provider(images, provider)
            
            # Add to the appropriate parameter based on provider
            if provider == "openai":
                # For OpenAI, images go in the 'content' field of messages
                if "messages" not in processed_params:
                    processed_params["messages"] = []
                
                # Find the user message or create one
                user_msg_idx = None
                for i, msg in enumerate(processed_params.get("messages", [])):
                    if msg.get("role") == "user":
                        user_msg_idx = i
                        break
                
                if user_msg_idx is not None:
                    # Update existing user message
                    msg = processed_params["messages"][user_msg_idx]
                    if isinstance(msg.get("content"), str):
                        # Convert string content to list format
                        text_content = msg["content"]
                        content_list = [{"type": "text", "text": text_content}]
                        content_list.extend(formatted_images)
                        msg["content"] = content_list
                    elif isinstance(msg.get("content"), list):
                        # Add to existing content list
                        msg["content"].extend(formatted_images)
                else:
                    # Create a new user message with only images
                    processed_params["messages"].append({
                        "role": "user",
                        "content": formatted_images
                    })
            
            elif provider == "anthropic":
                # For Anthropic, images go in the 'content' field of messages
                if "messages" not in processed_params:
                    processed_params["messages"] = []
                
                # Find the user message or create one
                user_msg_idx = None
                for i, msg in enumerate(processed_params.get("messages", [])):
                    if msg.get("role") == "user":
                        user_msg_idx = i
                        break
                
                if user_msg_idx is not None:
                    # Update existing user message
                    msg = processed_params["messages"][user_msg_idx]
                    if isinstance(msg.get("content"), str):
                        # Convert string content to list format
                        text_content = msg["content"]
                        content_list = [{"type": "text", "text": text_content}]
                        content_list.extend(formatted_images)
                        msg["content"] = content_list
                    elif isinstance(msg.get("content"), list):
                        # Add to existing content list
                        msg["content"].extend(formatted_images)
                else:
                    # Create a new user message with only images
                    processed_params["messages"].append({
                        "role": "user",
                        "content": formatted_images
                    })
            
            elif provider == "ollama":
                # For Ollama, images are added as a separate parameter
                processed_params["images"] = formatted_images
    
    return processed_params 