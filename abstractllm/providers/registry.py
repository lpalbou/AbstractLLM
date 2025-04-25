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