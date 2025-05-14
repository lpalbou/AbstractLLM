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


class ModelNotFoundError(AbstractLLMError):
    """
    Raised when a specific model cannot be found or loaded.
    
    This is different from UnsupportedModelError in that it deals with cases where
    the model should be available but cannot be found or accessed, rather than
    models that are explicitly not supported.
    
    Args:
        model_name: Name of the model that couldn't be found
        reason: Optional reason why the model couldn't be found
        search_path: Optional path or repository where the model was searched for
    """
    
    def __init__(self, model_name: str, reason: Optional[str] = None,
                 search_path: Optional[str] = None, provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.reason = reason
        self.search_path = search_path
        
        # Build message
        message = f"Model '{model_name}' not found"
        if reason:
            message += f": {reason}"
        if search_path:
            message += f" (searched in: {search_path})"
            
        # Add search path to details if provided
        if search_path and details is None:
            details = {"search_path": search_path}
        elif search_path:
            details["search_path"] = search_path
            
        super().__init__(message, provider, original_exception, details)


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
    
    Args:
        message: Description of the error
        provider: The provider name that raised the error
        model_name: Name of the model that failed to load
        original_exception: The original exception that was caught
        details: Additional details about the error
    """
    
    def __init__(self, message: str, provider: Optional[str] = None,
                 model_name: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        
        # Add model name to details if provided
        if model_name and details is None:
            details = {"model_name": model_name}
        elif model_name:
            details["model_name"] = model_name
            
        super().__init__(message, provider, original_exception, details)


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


class GenerationError(AbstractLLMError):
    """
    Raised when there is an error during text generation.
    
    This can occur due to:
    - Model generation failures
    - Invalid generation parameters
    - Resource constraints during generation
    - Unexpected model behavior
    
    Args:
        message: Description of the error
        provider: The provider name that raised the error
        original_exception: The original exception that was caught
        details: Additional details about the error
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


class UnsupportedOperationError(AbstractLLMError):
    """
    Raised when attempting to perform an operation that is not supported.
    
    This is different from UnsupportedFeatureError in that it deals with
    specific operations rather than general features. For example, a model
    might support the vision feature but not support certain operations
    with images.
    
    Args:
        operation: The unsupported operation name
        reason: Optional reason why the operation is not supported
    """
    
    def __init__(self, operation: str, reason: Optional[str] = None,
                 provider: Optional[str] = None, original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.reason = reason
        
        # Build message
        message = f"Operation '{operation}' is not supported"
        if reason:
            message += f": {reason}"
            
        super().__init__(message, provider, original_exception, details)


class ImageProcessingError(AbstractLLMError):
    """
    Raised when there is an error processing an image for vision models.
    """
    pass


class FileProcessingError(AbstractLLMError):
    """
    Raised when there is an error processing a file input.
    
    This can occur when:
    - File cannot be read or accessed
    - File format is invalid or unsupported
    - File content cannot be processed
    - File conversion fails
    - File size exceeds limits
    
    Args:
        message: Description of the error
        provider: The provider name that raised the error
        original_exception: The original exception that was caught
        details: Additional details about the error (e.g., file path, file type)
        file_path: Path to the file that caused the error
        file_type: Type of file that was being processed
    """
    
    def __init__(self, message: str, provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None,
                 file_path: Optional[str] = None,
                 file_type: Optional[str] = None):
        super().__init__(message, provider, original_exception, details)
        self.file_path = file_path
        self.file_type = file_type
        
        # Add file info to details
        if file_path:
            self.details["file_path"] = file_path
        if file_type:
            self.details["file_type"] = file_type


class ResourceExceededError(AbstractLLMError):
    """
    Base class for errors related to resource limits being exceeded.
    """
    pass


class MemoryExceededError(ResourceExceededError):
    """
    Raised when an operation would exceed available memory.
    
    This can occur when:
    - Processing large files or images
    - Loading models that are too large for available memory
    - Batch operations that would consume too much memory
    
    Args:
        message: Description of the error
        provider: The provider name that raised the error
        original_exception: The original exception that was caught
        details: Additional details about the error
        required_memory: The amount of memory required (in bytes)
        available_memory: The amount of memory available (in bytes)
    """
    
    def __init__(self, message: str, provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None,
                 required_memory: Optional[int] = None,
                 available_memory: Optional[int] = None):
        super().__init__(message, provider, original_exception, details)
        self.required_memory = required_memory
        self.available_memory = available_memory
        
        # Add memory info to details
        if required_memory is not None:
            self.details["required_memory"] = required_memory
        if available_memory is not None:
            self.details["available_memory"] = available_memory


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