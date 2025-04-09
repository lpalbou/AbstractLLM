"""
AbstractLLM: A unified interface for large language models.
"""

__version__ = "0.1.0"

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.factory import create_llm
from abstractllm.utils.config import ConfigurationManager, create_config
from abstractllm.exceptions import (
    AbstractLLMError,
    AuthenticationError,
    QuotaExceededError,
    UnsupportedProviderError,
    UnsupportedModelError,
    InvalidRequestError,
    InvalidParameterError,
    ModelLoadingError,
    ProviderConnectionError,
    ProviderAPIError,
    RequestTimeoutError,
    ContentFilterError,
    ContextWindowExceededError,
    UnsupportedFeatureError,
    ImageProcessingError,
)

__all__ = [
    # Core
    "AbstractLLMInterface",
    "create_llm",
    "ModelParameter",
    "ModelCapability",
    "ConfigurationManager",
    "create_config",  # For backward compatibility
    
    # Exceptions
    "AbstractLLMError",
    "AuthenticationError",
    "QuotaExceededError",
    "UnsupportedProviderError",
    "UnsupportedModelError",
    "InvalidRequestError",
    "InvalidParameterError",
    "ModelLoadingError",
    "ProviderConnectionError",
    "ProviderAPIError",
    "RequestTimeoutError",
    "ContentFilterError",
    "ContextWindowExceededError", 
    "UnsupportedFeatureError",
    "ImageProcessingError",
] 