"""
HuggingFace provider router for AbstractLLM.

This module serves as the main entry point for HuggingFace model interactions,
routing requests to either the Transformers or LangChain implementation based on configuration.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.exceptions import UnsupportedOperationError, ModelNotFoundError

# Configure logger
logger = logging.getLogger("abstractllm.providers.huggingface")

# Implementation types
class ImplementationType:
    TRANSFORMERS = "transformers"
    LANGCHAIN = "langchain"

class HuggingFaceProvider(AbstractLLMInterface):
    """
    Router implementation for HuggingFace models.
    
    This class serves as a router to different HuggingFace implementations:
    - Transformers: Direct implementation using PyTorch and Transformers
    - LangChain: Implementation using LangChain's abstractions
    
    The choice of implementation is controlled by the 'implementation' parameter
    in the configuration.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the HuggingFace provider router.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        default_config = {
            "implementation": ImplementationType.TRANSFORMERS,  # Default to Transformers implementation
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
        
        # Get implementation type
        impl_type = self.config_manager.get_param("implementation", ImplementationType.TRANSFORMERS)
        
        # Initialize the appropriate implementation
        if impl_type == ImplementationType.TRANSFORMERS:
            from .transformers_impl import TransformersImplementation
            self._impl = TransformersImplementation(config)
            logger.info("Using Transformers implementation for HuggingFace provider")
        elif impl_type == ImplementationType.LANGCHAIN:
            from .langchain_impl import LangChainImplementation
            self._impl = LangChainImplementation(config)
            logger.info("Using LangChain implementation for HuggingFace provider")
        else:
            raise ValueError(f"Unsupported implementation type: {impl_type}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                **kwargs) -> str:
        """Route generation request to the appropriate implementation."""
        return self._impl.generate(prompt, system_prompt, files, stream, **kwargs)
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False, 
                          **kwargs) -> str:
        """Route async generation request to the appropriate implementation."""
        return await self._impl.generate_async(prompt, system_prompt, files, stream, **kwargs)
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Get capabilities from the current implementation."""
        return self._impl.get_capabilities()
    
    @staticmethod
    def list_cached_models(cache_dir: Optional[str] = None) -> list:
        """List cached models from both implementations."""
        from .transformers_impl import TransformersImplementation
        from .langchain_impl import LangChainImplementation
        
        # Combine results from both implementations
        transformers_models = TransformersImplementation.list_cached_models(cache_dir)
        langchain_models = LangChainImplementation.list_cached_models(cache_dir)
        
        # Deduplicate and combine
        all_models = {}
        for model in transformers_models + langchain_models:
            model_id = model["name"]
            if model_id not in all_models:
                all_models[model_id] = model
            else:
                # If model exists in both, merge the information
                all_models[model_id].update({
                    "implementations": all_models[model_id].get("implementations", ["transformers"]) + ["langchain"]
                })
        
        return list(all_models.values())
    
    @staticmethod
    def clear_model_cache(model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        """Clear model cache from both implementations."""
        from .transformers_impl import TransformersImplementation
        from .langchain_impl import LangChainImplementation
        
        TransformersImplementation.clear_model_cache(model_name, cache_dir)
        LangChainImplementation.clear_model_cache(model_name, cache_dir) 