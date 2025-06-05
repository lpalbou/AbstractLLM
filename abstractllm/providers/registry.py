"""Provider registry for lazy loading providers."""

import importlib
import logging
from typing import Dict, Type, Any, Optional

# Initialize provider registry
_PROVIDER_REGISTRY = {}
logger = logging.getLogger("abstractllm.providers.registry")

def register_provider(name: str, module_path: str, class_name: str) -> None:
    """
    Register a provider without importing it.
    
    Args:
        name: The name of the provider (e.g., "openai")
        module_path: The import path to the provider module
        class_name: The name of the provider class
    """
    _PROVIDER_REGISTRY[name] = {
        "module_path": module_path,
        "class_name": class_name,
        "class": None  # Will be lazily loaded
    }
    logger.debug(f"Registered provider: {name}")

def initialize_registry() -> None:
    """
    Initialize the provider registry with built-in providers.
    This is called automatically when the registry module is imported.
    """
    # Register built-in providers that don't require additional dependencies
    register_provider("openai", "abstractllm.providers.openai", "OpenAIProvider")
    register_provider("anthropic", "abstractllm.providers.anthropic", "AnthropicProvider")
    register_provider("ollama", "abstractllm.providers.ollama", "OllamaProvider")
    register_provider("huggingface", "abstractllm.providers.huggingface", "HuggingFaceProvider")
    
    # Try to register MLX if the system supports it
    register_mlx_provider()
    
    logger.debug(f"Provider registry initialized with providers: {list(_PROVIDER_REGISTRY.keys())}")

def register_mlx_provider() -> bool:
    """
    Register the MLX provider if available.
    
    This function checks if the current platform is Apple Silicon and if
    the required MLX dependencies are installed before registering the provider.
    
    Returns:
        bool: True if the provider was registered, False otherwise
    """
    # Check platform compatibility
    from abstractllm.utils.utilities import is_apple_silicon
    
    if not is_apple_silicon():
        logger.info("MLX provider not registered: requires Apple Silicon hardware")
        return False
    
    logger.debug("Platform check passed - Apple Silicon detected")
    
    # Import MLX dependencies
    try:
        import mlx.core
        logger.debug("MLX package is available")
    except ImportError as e:
        logger.info(f"MLX provider not registered: mlx package not available - {e}")
        return False
        
    try:
        import mlx_lm
        logger.debug("MLX-LM package is available")
    except ImportError as e:
        logger.info(f"MLX provider not registered: mlx-lm package not available - {e}")
        return False
    
    # Check for vision capabilities
    try:
        import mlx_vlm
        logger.debug("MLX-VLM package is available for vision model support")
        has_vision = True
    except ImportError as e:
        logger.info(f"MLX-VLM package not available (vision models will not work): {e}")
        has_vision = False
    
    # Register the provider
    try:
        register_provider("mlx", "abstractllm.providers.mlx_provider", "MLXProvider")
        logger.info("MLX provider successfully registered for Apple Silicon")
        
        if has_vision:
            logger.info("MLX Vision support is available")
        return True
    except Exception as e:
        logger.error(f"Failed to register MLX provider through registry system: {e}")
        return False

def get_provider_class(name: str) -> Optional[Type[Any]]:
    """
    Get the provider class, lazily importing it if necessary.
    
    Args:
        name: The name of the provider
        
    Returns:
        The provider class, or None if not found
        
    Raises:
        ImportError: If the provider module cannot be imported
    """
    if name not in _PROVIDER_REGISTRY:
        logger.warning(f"Unknown provider: {name}")
        return None
        
    # If class is already loaded, return it
    if _PROVIDER_REGISTRY[name]["class"] is not None:
        return _PROVIDER_REGISTRY[name]["class"]
        
    # Otherwise, lazily import the module and get the class
    try:
        module_path = _PROVIDER_REGISTRY[name]["module_path"]
        class_name = _PROVIDER_REGISTRY[name]["class_name"]
        
        logger.debug(f"Lazily importing provider module: {module_path}")
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        
        # Cache the class
        _PROVIDER_REGISTRY[name]["class"] = provider_class
        return provider_class
    except ImportError as e:
        logger.error(f"Failed to import provider {name}: {e}")
        raise ImportError(f"Provider '{name}' requires additional dependencies: {e}")
    except AttributeError as e:
        logger.error(f"Failed to get provider class {class_name} from {module_path}: {e}")
        raise ImportError(f"Provider class not found: {e}")

def get_available_providers() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered providers.
    
    Returns:
        Dictionary of provider names to their registry entries
    """
    return _PROVIDER_REGISTRY.copy()

# Initialize the registry when the module is imported
initialize_registry() 