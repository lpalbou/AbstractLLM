"""
Media processor for AbstractLLM.

This module provides a processor for handling media inputs for LLM providers.
"""

import logging
from typing import Any, Dict, Union, List, Optional

from abstractllm.media.interface import MediaInput
from abstractllm.media.factory import MediaFactory
from abstractllm.exceptions import ImageProcessingError, UnsupportedFeatureError

# Configure logger
logger = logging.getLogger("abstractllm.media.processor")


class MediaProcessor:
    """
    Process media inputs for LLM providers.
    
    This class provides methods for processing media inputs in requests to LLM providers.
    It handles both single and multiple media inputs and formats them according to the
    provider's requirements.
    """
    
    @classmethod
    def process_inputs(cls, params: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Process all media inputs in params for the specified provider.
        
        Args:
            params: Parameters that may include media inputs
            provider: Provider name ('openai', 'anthropic', 'ollama', 'huggingface')
            
        Returns:
            Updated parameters with media inputs formatted for the provider
            
        Raises:
            ImageProcessingError: If there's an error processing the media
            UnsupportedFeatureError: If the provider doesn't support the media type
        """
        # Make a copy of the parameters to avoid modifying the original
        processed_params = params.copy()
        
        # Handle null image values
        if "image" in processed_params and processed_params["image"] is None:
            processed_params.pop("image")
        if "images" in processed_params and processed_params["images"] is None:
            processed_params.pop("images")
        
        # Handle empty image lists
        if "images" in processed_params and not processed_params["images"]:
            processed_params.pop("images")
        
        # Process single media inputs
        # This handles image, and can be extended to other media types in the future
        processed_params = cls._process_single_media_inputs(processed_params, provider)
        
        # Process multiple media inputs
        # This handles images, and can be extended to other media types in the future
        processed_params = cls._process_multiple_media_inputs(processed_params, provider)
        
        # Ensure OpenAI and Anthropic have the correct message structure 
        # for cases without images or with null/empty images
        if provider in ["openai", "anthropic"] and "image" not in processed_params and "images" not in processed_params:
            # Format as chat message if not already formatted
            if "messages" not in processed_params and "prompt" in processed_params:
                prompt = processed_params.pop("prompt", "")
                processed_params["messages"] = [{"role": "user", "content": prompt}]
        
        return processed_params
    
    @classmethod
    def _process_single_media_inputs(cls, params: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Process single media inputs in params.
        
        Args:
            params: Parameters that may include media inputs
            provider: Provider name
            
        Returns:
            Updated parameters with media inputs formatted for the provider
        """
        processed_params = params.copy()
        
        # Handle image parameter
        if "image" in processed_params or "IMAGE" in processed_params:
            image_param = processed_params.pop("image", processed_params.pop("IMAGE", None))
            if image_param is not None:
                try:
                    # Convert to ImageInput
                    image_input = MediaFactory.from_source(image_param, media_type="image")
                    
                    # Add to the appropriate parameter based on provider
                    if provider == "openai":
                        processed_params = cls._add_image_to_openai_params(processed_params, image_input)
                    elif provider == "anthropic":
                        processed_params = cls._add_image_to_anthropic_params(processed_params, image_input)
                    elif provider == "ollama":
                        processed_params = cls._add_image_to_ollama_params(processed_params, image_input)
                    elif provider == "huggingface":
                        processed_params = cls._add_image_to_huggingface_params(processed_params, image_input)
                    else:
                        logger.warning(f"Unknown provider {provider}, image may not be processed correctly")
                        
                except Exception as e:
                    if isinstance(e, (ValueError, ImageProcessingError)):
                        raise
                    raise ImageProcessingError(
                        f"Failed to process image: {e}",
                        provider=provider,
                        original_exception=e
                    )
        
        # Add other media types here as they are implemented
        
        return processed_params
    
    @classmethod
    def _process_multiple_media_inputs(cls, params: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """
        Process multiple media inputs in params.
        
        Args:
            params: Parameters that may include media inputs
            provider: Provider name
            
        Returns:
            Updated parameters with media inputs formatted for the provider
        """
        processed_params = params.copy()
        
        # Handle images parameter
        if "images" in processed_params or "IMAGES" in processed_params:
            images_param = processed_params.pop("images", processed_params.pop("IMAGES", None))
            if images_param is not None and isinstance(images_param, list):
                try:
                    # Skip processing if the images list is empty
                    if not images_param:
                        # Don't add empty images list back to the parameters
                        return processed_params
                        
                    # Convert to list of ImageInput objects
                    image_inputs = MediaFactory.from_sources(images_param, media_type="image")
                    
                    # Add to the appropriate parameter based on provider
                    if provider == "openai":
                        processed_params = cls._add_images_to_openai_params(processed_params, image_inputs)
                    elif provider == "anthropic":
                        processed_params = cls._add_images_to_anthropic_params(processed_params, image_inputs)
                    elif provider == "ollama":
                        processed_params = cls._add_images_to_ollama_params(processed_params, image_inputs)
                    elif provider == "huggingface":
                        processed_params = cls._add_images_to_huggingface_params(processed_params, image_inputs)
                    else:
                        logger.warning(f"Unknown provider {provider}, images may not be processed correctly")
                        
                except Exception as e:
                    if isinstance(e, (ValueError, ImageProcessingError)):
                        raise
                    raise ImageProcessingError(
                        f"Failed to process images: {e}",
                        provider=provider,
                        original_exception=e
                    )
        
        # Add other media types here as they are implemented
        
        return processed_params
    
    @staticmethod
    def _add_image_to_openai_params(
        params: Dict[str, Any], 
        image_input: MediaInput
    ) -> Dict[str, Any]:
        """
        Add an image to OpenAI parameters.
        
        Args:
            params: Parameters to update
            image_input: Image input object
            
        Returns:
            Updated parameters
        """
        # Format image for OpenAI
        formatted_image = image_input.to_provider_format("openai")
        
        # For OpenAI, images go in the 'content' field of messages
        if "messages" not in params:
            params["messages"] = []
        
        # Find the user message or create one
        user_msg_idx = None
        for i, msg in enumerate(params.get("messages", [])):
            if msg.get("role") == "user":
                user_msg_idx = i
                break
        
        if user_msg_idx is not None:
            # Update existing user message
            msg = params["messages"][user_msg_idx]
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
            content_list = [formatted_image]
            
            # If prompt parameter exists, add it as a text component at the beginning
            if "prompt" in params:
                content_list.insert(0, {"type": "text", "text": params.pop("prompt")})
                
            params["messages"].append({
                "role": "user",
                "content": content_list
            })
            
        return params
    
    @staticmethod
    def _add_images_to_openai_params(
        params: Dict[str, Any], 
        image_inputs: List[MediaInput]
    ) -> Dict[str, Any]:
        """
        Add multiple images to OpenAI parameters.
        
        Args:
            params: Parameters to update
            image_inputs: List of image input objects
            
        Returns:
            Updated parameters
        """
        # Format images for OpenAI
        formatted_images = [img.to_provider_format("openai") for img in image_inputs]
        
        # For OpenAI, images go in the 'content' field of messages
        if "messages" not in params:
            params["messages"] = []
        
        # Find the user message or create one
        user_msg_idx = None
        for i, msg in enumerate(params.get("messages", [])):
            if msg.get("role") == "user":
                user_msg_idx = i
                break
        
        if user_msg_idx is not None:
            # Update existing user message
            msg = params["messages"][user_msg_idx]
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
            # Create a new user message
            content_list = formatted_images.copy()
            
            # If prompt parameter exists, add it as a text component at the beginning
            if "prompt" in params:
                content_list.insert(0, {"type": "text", "text": params.pop("prompt")})
                
            params["messages"].append({
                "role": "user",
                "content": content_list
            })
            
        return params
    
    @staticmethod
    def _add_image_to_anthropic_params(
        params: Dict[str, Any], 
        image_input: MediaInput
    ) -> Dict[str, Any]:
        """
        Add an image to Anthropic parameters.
        
        Args:
            params: Parameters to update
            image_input: Image input object
            
        Returns:
            Updated parameters
        """
        # Format image for Anthropic
        formatted_image = image_input.to_provider_format("anthropic")
        
        # For Anthropic, images go in the 'content' field of messages
        if "messages" not in params:
            params["messages"] = []
        
        # Find the user message or create one
        user_msg_idx = None
        for i, msg in enumerate(params.get("messages", [])):
            if msg.get("role") == "user":
                user_msg_idx = i
                break
        
        if user_msg_idx is not None:
            # Update existing user message
            msg = params["messages"][user_msg_idx]
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
            content_list = [formatted_image]
            
            # If prompt parameter exists, add it as a text component at the beginning
            if "prompt" in params:
                content_list.insert(0, {"type": "text", "text": params.pop("prompt")})
                
            params["messages"].append({
                "role": "user",
                "content": content_list
            })
            
        return params
    
    @staticmethod
    def _add_images_to_anthropic_params(
        params: Dict[str, Any], 
        image_inputs: List[MediaInput]
    ) -> Dict[str, Any]:
        """
        Add multiple images to Anthropic parameters.
        
        Args:
            params: Parameters to update
            image_inputs: List of image input objects
            
        Returns:
            Updated parameters
        """
        # Format images for Anthropic
        formatted_images = [img.to_provider_format("anthropic") for img in image_inputs]
        
        # For Anthropic, images go in the 'content' field of messages
        if "messages" not in params:
            params["messages"] = []
        
        # Find the user message or create one
        user_msg_idx = None
        for i, msg in enumerate(params.get("messages", [])):
            if msg.get("role") == "user":
                user_msg_idx = i
                break
        
        if user_msg_idx is not None:
            # Update existing user message
            msg = params["messages"][user_msg_idx]
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
            # Create a new user message
            content_list = formatted_images.copy()
            
            # If prompt parameter exists, add it as a text component at the beginning
            if "prompt" in params:
                content_list.insert(0, {"type": "text", "text": params.pop("prompt")})
                
            params["messages"].append({
                "role": "user",
                "content": content_list
            })
            
        return params
    
    @staticmethod
    def _add_image_to_ollama_params(
        params: Dict[str, Any], 
        image_input: MediaInput
    ) -> Dict[str, Any]:
        """
        Add an image to Ollama parameters.
        
        Args:
            params: Parameters to update
            image_input: Image input object
            
        Returns:
            Updated parameters
        """
        # Format image for Ollama
        formatted_image = image_input.to_provider_format("ollama")
        
        # For Ollama, images are added as a separate parameter
        # Use "image" key instead of "images" for single images
        params["image"] = formatted_image
        
        # Make sure the prompt is preserved
        if "prompt" in params and not params.get("messages"):
            # If we have a prompt but no messages, preserve the prompt
            # This ensures the prompt is still available for Ollama's API
            # Ollama will use the prompt parameter directly
            pass  # Keep the prompt parameter as is
        elif "prompt" in params and not any(msg.get("role") == "user" for msg in params.get("messages", [])):
            # If we have messages but no user message, add the prompt as a user message
            if "messages" not in params:
                params["messages"] = []
            params["messages"].append({
                "role": "user",
                "content": params.pop("prompt")
            })
        
        return params
    
    @staticmethod
    def _add_images_to_ollama_params(
        params: Dict[str, Any], 
        image_inputs: List[MediaInput]
    ) -> Dict[str, Any]:
        """
        Add multiple images to Ollama parameters.
        
        Args:
            params: Parameters to update
            image_inputs: List of image input objects
            
        Returns:
            Updated parameters
        """
        # Format images for Ollama
        formatted_images = [img.to_provider_format("ollama") for img in image_inputs]
        
        # For Ollama, images are added as a separate parameter
        params["images"] = formatted_images
        
        # Make sure the prompt is preserved
        if "prompt" in params and not params.get("messages"):
            # If we have a prompt but no messages, preserve the prompt
            # This ensures the prompt is still available for Ollama's API
            # Ollama will use the prompt parameter directly
            pass  # Keep the prompt parameter as is
        elif "prompt" in params and not any(msg.get("role") == "user" for msg in params.get("messages", [])):
            # If we have messages but no user message, add the prompt as a user message
            if "messages" not in params:
                params["messages"] = []
            params["messages"].append({
                "role": "user",
                "content": params.pop("prompt")
            })
        
        return params
    
    @staticmethod
    def _add_image_to_huggingface_params(
        params: Dict[str, Any], 
        image_input: MediaInput
    ) -> Dict[str, Any]:
        """
        Add an image to HuggingFace parameters.
        
        Args:
            params: Parameters to update
            image_input: Image input object
            
        Returns:
            Updated parameters
        """
        # Format image for HuggingFace
        formatted_image = image_input.to_provider_format("huggingface")
        
        # For HuggingFace, keep the image path or URL as is
        params["image"] = formatted_image
        
        # Add image_processor_format parameter if not already provided
        if "image_processor_format" not in params:
            # Default to PIL format
            params["image_processor_format"] = "pil"
        
        # For HuggingFace, we typically keep the prompt parameter as is
        # since it's used directly by the HuggingFace generate method
        # If we have messages, we could extract the prompt from there
        if "messages" in params and not "prompt" in params:
            # Try to find a user message with content
            for msg in params["messages"]:
                if msg.get("role") == "user" and "content" in msg:
                    if isinstance(msg["content"], str):
                        params["prompt"] = msg["content"]
                        break
                    elif isinstance(msg["content"], list):
                        # Try to find text content
                        for content in msg["content"]:
                            if isinstance(content, dict) and content.get("type") == "text":
                                params["prompt"] = content.get("text", "")
                                break
                        break
            
        return params
    
    @staticmethod
    def _add_images_to_huggingface_params(
        params: Dict[str, Any], 
        image_inputs: List[MediaInput]
    ) -> Dict[str, Any]:
        """
        Add multiple images to HuggingFace parameters.
        
        Args:
            params: Parameters to update
            image_inputs: List of image input objects
            
        Returns:
            Updated parameters
        """
        # Format images for HuggingFace
        formatted_images = [img.to_provider_format("huggingface") for img in image_inputs]
        
        # For HuggingFace, add as a list in 'pixel_values' parameter
        # Most HF models only support a single image at a time
        if formatted_images:
            params["image"] = formatted_images[0]
            if len(formatted_images) > 1:
                # Enhanced warning with specific model information
                logger.warning(
                    "Most HuggingFace models only support one image at a time. Using the first image only. "
                    "Multi-image support is limited to specific models like LLaVA-NeXT and IDEFICS. "
                    "For other models, you'll need to process images one at a time."
                )
            
            # Add image_processor_format parameter if not already provided
            if "image_processor_format" not in params:
                # Default to PIL format
                params["image_processor_format"] = "pil"
            
            # For HuggingFace, we typically keep the prompt parameter as is
            # since it's used directly by the HuggingFace generate method
            # If we have messages, we could extract the prompt from there
            if "messages" in params and not "prompt" in params:
                # Try to find a user message with content
                for msg in params["messages"]:
                    if msg.get("role") == "user" and "content" in msg:
                        if isinstance(msg["content"], str):
                            params["prompt"] = msg["content"]
                            break
                        elif isinstance(msg["content"], list):
                            # Try to find text content
                            for content in msg["content"]:
                                if isinstance(content, dict) and content.get("type") == "text":
                                    params["prompt"] = content.get("text", "")
                                    break
                            break
                
        return params 