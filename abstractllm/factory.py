"""
Factory function for creating LLM provider instances.
"""

from typing import Dict, Any, Optional
import importlib
import logging
import platform
from abstractllm.interface import AbstractLLMInterface, ModelParameter
import os
import importlib.util
from abstractllm.providers.registry import get_provider_class, get_available_providers

# Configure logger
logger = logging.getLogger("abstractllm.factory")

# Provider mapping (kept for backward compatibility)
# Note: All of these providers should also be registered via the registry system
# in registry.py's initialize_registry function. This mapping is kept for 
# backward compatibility and fallback purposes only.
_PROVIDERS = {
    "openai": "abstractllm.providers.openai.OpenAIProvider",
    "anthropic": "abstractllm.providers.anthropic.AnthropicProvider",
    "ollama": "abstractllm.providers.ollama.OllamaProvider",
    "huggingface": "abstractllm.providers.huggingface.HuggingFaceProvider",
    "mlx": "abstractllm.providers.mlx_provider.MLXProvider",
}

# Providers that always require API keys
_REQUIRED_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY"
}

# Optional dependency mapping for providers
_PROVIDER_DEPENDENCIES = {
    "openai": ["openai"],
    "anthropic": ["anthropic"],
    "huggingface": ["torch", "transformers", "huggingface_hub"],
    "ollama": [],  # No external dependencies
    "mlx": ["mlx", "mlx_lm"]  # MLX dependencies
}

# Platform-specific providers
_PLATFORM_REQUIREMENTS = {
    "mlx": {
        "platform": "darwin",  # macOS
        "processor": "arm",    # Apple Silicon
        "error_message": "MLX provider requires macOS with Apple Silicon (M1/M2/M3)"
    }
}

def get_llm_providers() -> list[str]:
    """
    Get a list of all available LLM providers.
    """
    # Combine registry providers with hardcoded providers
    providers = set(_PROVIDERS.keys())
    providers.update(get_available_providers().keys())
    return list(providers)

def _check_dependency(module_name: str) -> bool:
    """
    Check if a Python module is installed and can be imported.
    
    Args:
        module_name: The name of the module to check
        
    Returns:
        True if the module is available, False otherwise
    """
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, AttributeError):
        return False

def _check_platform_requirements(provider: str) -> bool:
    """
    Check if the current platform meets the requirements for a provider.
    
    Args:
        provider: The provider name
        
    Returns:
        True if the platform is compatible, False otherwise
    """
    if provider not in _PLATFORM_REQUIREMENTS:
        return True
        
    requirements = _PLATFORM_REQUIREMENTS[provider]
    
    # Check platform
    if "platform" in requirements and platform.system().lower() != requirements["platform"]:
        return False
        
    # Check processor
    if "processor" in requirements and platform.processor() != requirements["processor"]:
        return False
        
    return True

def create_llm(provider: str, **config) -> AbstractLLMInterface:
    """
    Create an LLM provider instance.
    
    Args:
        provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface', 'mlx')
        **config: Provider-specific configuration
        
    Returns:
        An initialized LLM interface
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If the provider module cannot be imported
    """
    # Check if provider is in registry first, then fall back to hardcoded providers
    available_providers = get_available_providers()
    
    if provider not in _PROVIDERS and provider not in available_providers:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            f"Available providers: {', '.join(get_llm_providers())}"
        )
    
    # Check platform requirements
    if not _check_platform_requirements(provider):
        requirements = _PLATFORM_REQUIREMENTS[provider]
        error_message = requirements.get("error_message", f"Provider '{provider}' is not compatible with this platform")
        raise ValueError(error_message)
    
    # Check required dependencies before importing
    if provider in _PROVIDER_DEPENDENCIES:
        missing_deps = []
        for dep in _PROVIDER_DEPENDENCIES[provider]:
            if not _check_dependency(dep):
                missing_deps.append(dep)
        
        if missing_deps:
            deps_str = ", ".join(missing_deps)
            
            # Special handling for MLX provider
            if provider == "mlx":
                raise ImportError(
                    f"Missing required dependencies for MLX provider: {deps_str}. "
                    f"Please install them using: pip install 'abstractllm[mlx]'"
                )
            else:
                raise ImportError(
                    f"Missing required dependencies for provider '{provider}': {deps_str}. "
                    f"Please install them using: pip install abstractllm[{provider}]"
                )
    
    # Try to get provider from registry first
    provider_class = get_provider_class(provider)
    
    # If not in registry, use hardcoded provider mapping
    if provider_class is None and provider in _PROVIDERS:
        # Import the provider class
        module_path, class_name = _PROVIDERS[provider].rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import provider {provider}: {e}")
    
    # Check for required API key (only for providers that always need it)
    if provider in _REQUIRED_API_KEYS:
        api_key = config.get(ModelParameter.API_KEY) or config.get("api_key")

        if not api_key:
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            elif provider == "anthropic":                
                api_key = os.environ.get("ANTHROPIC_API_KEY")

        if api_key:
            config[ModelParameter.API_KEY] = api_key
        else:
            env_var = _REQUIRED_API_KEYS[provider]
            raise ValueError(
                f"{provider} API key not provided. Use --api-key or set {env_var} environment variable."
            )
    
    # Create provider instance with config
    return provider_class(config) 