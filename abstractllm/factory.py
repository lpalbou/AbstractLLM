"""
Factory function for creating LLM provider instances.
"""

from typing import Dict, Any, Optional
import importlib
from abstractllm.interface import AbstractLLMInterface, ModelParameter, create_config


# Provider mapping
_PROVIDERS = {
    "openai": "abstractllm.providers.openai.OpenAIProvider",
    "anthropic": "abstractllm.providers.anthropic.AnthropicProvider",
    "ollama": "abstractllm.providers.ollama.OllamaProvider",
    "huggingface": "abstractllm.providers.huggingface.HuggingFaceProvider",
}


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
    
    # Create configuration with defaults
    provider_config = create_config(**config)
    
    # Instantiate and return the provider
    return provider_class(config=provider_config) 