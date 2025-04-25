"""
Factory function for creating LLM provider instances.
"""

from typing import Dict, Any, Optional
import importlib
import logging
from abstractllm.interface import AbstractLLMInterface, ModelParameter
import os
import importlib.util

# Configure logger
logger = logging.getLogger("abstractllm.factory")

# Provider mapping
_PROVIDERS = {
    "openai": "abstractllm.providers.openai.OpenAIProvider",
    "anthropic": "abstractllm.providers.anthropic.AnthropicProvider",
    "ollama": "abstractllm.providers.ollama.OllamaProvider",
    "huggingface": "abstractllm.providers.huggingface.HuggingFaceProvider",
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
    "ollama": []  # No external dependencies
}

def get_llm_providers() -> list[str]:
    """
    Get a list of all available LLM providers.
    """
    return list(_PROVIDERS.keys())

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

def create_llm(provider: str, **config) -> AbstractLLMInterface:
    """
    Create an LLM provider instance.
    
    Args:
        provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface')
        **config: Provider-specific configuration
        
    Returns:
        An initialized LLM interface
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If the provider module cannot be imported
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            f"Available providers: {', '.join(_PROVIDERS.keys())}"
        )
    
    # Check required dependencies before importing
    if provider in _PROVIDER_DEPENDENCIES:
        missing_deps = []
        for dep in _PROVIDER_DEPENDENCIES[provider]:
            if not _check_dependency(dep):
                missing_deps.append(dep)
        
        if missing_deps:
            deps_str = ", ".join(missing_deps)
            raise ImportError(
                f"Missing required dependencies for provider '{provider}': {deps_str}. "
                f"Please install them using: pip install abstractllm[{provider}]"
            )
    
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