"""
Exception types for AbstractLLM.

This module defines all exception types used across AbstractLLM to provide
consistent error handling regardless of the underlying provider.
"""

from typing import Optional, Dict, Any, Union


class AbstractLLMError(Exception):
    """
    Base exception class for all AbstractLLM errors.
    
    All exceptions raised by AbstractLLM should inherit from this class
    to allow for consistent error handling.
    """
    
    def __init__(self, message: str, provider: Optional[str] = None, 
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the base exception class.
        
        Args:
            message: The error message
            provider: The provider name that raised the error
            original_exception: The original exception that was caught
            details: Additional details about the error
        """
        self.provider = provider
        self.original_exception = original_exception
        self.details = details or {}
        
        # Build the full message
        full_message = message
        if provider:
            full_message = f"[{provider}] {full_message}"
            
        super().__init__(full_message)


class AuthenticationError(AbstractLLMError):
    """
    Raised when authentication with a provider fails.
    
    This typically occurs when an API key is invalid, expired, or missing.
    """
    pass


class QuotaExceededError(AbstractLLMError):
    """
    Raised when a provider's usage quota or rate limit is exceeded.
    
    This occurs when you've hit API limits, either for requests per minute
    or for your account's overall usage limits.
    """
    pass


class UnsupportedProviderError(AbstractLLMError):
    """
    Raised when attempting to use an unsupported provider.
    """
    pass


class UnsupportedModelError(AbstractLLMError):
    """
    Raised when attempting to use a model that is not supported by the provider.
    """
    pass


class InvalidRequestError(AbstractLLMError):
    """
    Raised when a request to the provider is invalid.
    
    This can happen due to malformed parameters, invalid prompt format,
    or other issues with the request.
    """
    pass


class InvalidParameterError(InvalidRequestError):
    """
    Raised when a parameter value is invalid.
    
    Args:
        parameter: The parameter name that caused the error
        value: The invalid parameter value
    """
    
    def __init__(self, parameter: str, value: Any, message: Optional[str] = None,
                 provider: Optional[str] = None, original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.parameter = parameter
        self.value = value
        
        # Build default message if not provided
        if message is None:
            message = f"Invalid value '{value}' for parameter '{parameter}'"
            
        super().__init__(message, provider, original_exception, details)


class ModelLoadingError(AbstractLLMError):
    """
    Raised when a model fails to load.
    
    This is primarily used with local models like HuggingFace and Ollama
    when there are issues loading the model files.
    """
    pass


class ProviderConnectionError(AbstractLLMError):
    """
    Raised when a connection to a provider's API cannot be established.
    
    This can happen due to network issues, API endpoint being down, etc.
    """
    pass


class ProviderAPIError(AbstractLLMError):
    """
    Raised when a provider's API returns an error.
    
    This is a catch-all for provider-specific API errors that don't
    fit into other categories.
    """
    pass


class RequestTimeoutError(AbstractLLMError):
    """
    Raised when a request to a provider times out.
    """
    pass


class ContentFilterError(AbstractLLMError):
    """
    Raised when content is filtered by the provider's safety measures.
    
    This typically occurs when the prompt or expected response violates
    the provider's content policies.
    """
    pass


class ContextWindowExceededError(InvalidRequestError):
    """
    Raised when the combined input and expected output exceeds the model's context window.
    
    Args:
        context_window: The maximum context window size
        content_length: The length of the provided content
    """
    
    def __init__(self, context_window: int, content_length: int, 
                 message: Optional[str] = None, provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.context_window = context_window
        self.content_length = content_length
        
        # Build default message if not provided
        if message is None:
            message = f"Content length ({content_length}) exceeds maximum context window ({context_window})"
            
        super().__init__(message, provider, original_exception, details)


class UnsupportedFeatureError(AbstractLLMError):
    """
    Raised when attempting to use a feature not supported by the current provider/model.
    
    Args:
        feature: The unsupported feature name
    """
    
    def __init__(self, feature: Union[str, Any], message: Optional[str] = None,
                 provider: Optional[str] = None, original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.feature = feature
        
        # Build default message if not provided
        if message is None:
            message = f"Feature '{feature}' is not supported by this provider/model"
            
        super().__init__(message, provider, original_exception, details)


class ImageProcessingError(AbstractLLMError):
    """
    Raised when there is an error processing an image for vision models.
    """
    pass


# Mapping of common provider-specific error codes to AbstractLLM exceptions
# This helps normalize error handling across providers
PROVIDER_ERROR_MAPPING = {
    "openai": {
        "authentication_error": AuthenticationError,
        "invalid_request_error": InvalidRequestError,
        "rate_limit_exceeded": QuotaExceededError,
        "quota_exceeded": QuotaExceededError,
        "context_length_exceeded": ContextWindowExceededError,
        "content_filter": ContentFilterError,
    },
    "anthropic": {
        "authentication_error": AuthenticationError,
        "invalid_request": InvalidRequestError,
        "rate_limit_error": QuotaExceededError,
        "context_window_exceeded": ContextWindowExceededError,
        "content_policy_violation": ContentFilterError,
    },
    # Add mappings for other providers as needed
}


def map_provider_error(provider: str, error_type: str, 
                       message: str, original_exception: Optional[Exception] = None,
                       details: Optional[Dict[str, Any]] = None) -> AbstractLLMError:
    """
    Map a provider-specific error to an AbstractLLM exception.
    
    Args:
        provider: The provider name
        error_type: The provider-specific error type
        message: The error message
        original_exception: The original exception
        details: Additional error details
        
    Returns:
        An appropriate AbstractLLMError subclass instance
    """
    provider_mapping = PROVIDER_ERROR_MAPPING.get(provider, {})
    error_class = provider_mapping.get(error_type, ProviderAPIError)
    
    return error_class(
        message=message,
        provider=provider,
        original_exception=original_exception,
        details=details
    ) 