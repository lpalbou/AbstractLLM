"""
Factory function for creating LLM provider instances.
"""

from typing import Dict, Any, Optional
import importlib
import logging
from abstractllm.interface import AbstractLLMInterface, ModelParameter
import os

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

def get_llm_providers() -> list[str]:
    """
    Get a list of all available LLM providers.
    """
    return list(_PROVIDERS.keys())

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