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
    Raised when content is filtered or blocked by the provider's content policy.
    """
    pass


class ContextWindowExceededError(InvalidRequestError):
    """
    Raised when the input exceeds the model's context window.
    
    Args:
        context_window: The maximum context window size
        content_length: The actual length of the content
    """
    
    def __init__(self, context_window: int, content_length: int, 
                 message: Optional[str] = None, provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.context_window = context_window
        self.content_length = content_length
        
        # Build default message if not provided
        if message is None:
            message = f"Content length ({content_length}) exceeds context window ({context_window})"
            
        super().__init__(message, provider, original_exception, details)


class UnsupportedFeatureError(AbstractLLMError):
    """
    Raised when attempting to use a feature not supported by the provider or model.
    
    Args:
        feature: The unsupported feature
        message: Optional custom error message
    """
    
    def __init__(self, feature: Union[str, Any], message: Optional[str] = None,
                 provider: Optional[str] = None, original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.feature = feature
        
        # Build default message if not provided
        if message is None:
            message = f"Feature '{feature}' is not supported"
            
        super().__init__(message, provider, original_exception, details)


class UnsupportedOperationError(AbstractLLMError):
    """
    Raised when attempting an operation that is not supported.
    
    Args:
        operation: The unsupported operation
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


class MediaProcessingError(AbstractLLMError):
    """
    Raised when there is an error processing media inputs.
    
    Args:
        message: Description of the error
        media_type: Type of media that caused the error
    """
    
    def __init__(self, message: str, 
                 media_type: Optional[str] = None,
                 provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.media_type = media_type
        
        # Add media type to message if provided
        if media_type:
            message = f"[{media_type}] {message}"
            
        super().__init__(message, provider, original_exception, details)


class FileProcessingError(AbstractLLMError):
    """
    Raised when there is an error processing files.
    
    Args:
        message: Description of the error
        file_path: Path to the file that caused the error
        file_type: Type of file that caused the error
    """
    
    def __init__(self, message: str, provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None,
                 file_path: Optional[str] = None,
                 file_type: Optional[str] = None):
        self.file_path = file_path
        self.file_type = file_type
        
        # Add file info to details
        if file_path or file_type:
            details = details or {}
            if file_path:
                details["file_path"] = file_path
            if file_type:
                details["file_type"] = file_type
            
        super().__init__(message, provider, original_exception, details)


class InvalidInputError(InvalidRequestError):
    """
    Raised when input validation fails.
    
    Args:
        message: Description of the error
        input_type: Type of input that failed validation
        validation_error: Specific validation error message
    """
    
    def __init__(self, message: str, input_type: Optional[str] = None,
                 validation_error: Optional[str] = None,
                 provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.input_type = input_type
        self.validation_error = validation_error
        
        # Add validation info to details
        if input_type or validation_error:
            details = details or {}
            if input_type:
                details["input_type"] = input_type
            if validation_error:
                details["validation_error"] = validation_error
            
        super().__init__(message, provider, original_exception, details)


# Alias for backward compatibility
ModelLoadError = ModelLoadingError


class AudioOutputError(AbstractLLMError):
    """
    Raised when there is an error with audio output processing.
    
    Args:
        message: Description of the error
        device_info: Information about the audio device
        audio_format: Information about the audio format
    """
    
    def __init__(self, message: str, device_info: Optional[Dict[str, Any]] = None,
                 audio_format: Optional[Dict[str, Any]] = None,
                 provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.device_info = device_info
        self.audio_format = audio_format
        
        # Add audio info to details
        if device_info or audio_format:
            details = details or {}
            if device_info:
                details["device_info"] = device_info
            if audio_format:
                details["audio_format"] = audio_format
            
        super().__init__(message, provider, original_exception, details)


class ResourceError(AbstractLLMError):
    """
    Raised when there are insufficient resources to perform an operation.
    
    This can occur due to:
    - Insufficient memory (RAM/VRAM)
    - Insufficient disk space
    - CPU/GPU resource constraints
    - Other system resource limitations
    
    Args:
        message: Description of the error
        resource_type: Type of resource that is insufficient
        required: Required amount of resource
        available: Available amount of resource
    """
    
    def __init__(self, message: str,
                 resource_type: Optional[str] = None,
                 required: Optional[Union[int, str]] = None,
                 available: Optional[Union[int, str]] = None,
                 provider: Optional[str] = None,
                 original_exception: Optional[Exception] = None,
                 details: Optional[Dict[str, Any]] = None):
        self.resource_type = resource_type
        self.required = required
        self.available = available
        
        # Add resource info to details
        if resource_type or required or available:
            details = details or {}
            if resource_type:
                details["resource_type"] = resource_type
            if required:
                details["required"] = required
            if available:
                details["available"] = available
            
        super().__init__(message, provider, original_exception, details)


def map_provider_error(provider: str, error_type: str, 
                       message: str, original_exception: Optional[Exception] = None,
                       details: Optional[Dict[str, Any]] = None) -> AbstractLLMError:
    """
    Map provider-specific errors to AbstractLLM exceptions.
    
    Args:
        provider: Provider name
        error_type: Provider's error type
        message: Error message
        original_exception: Original exception from provider
        details: Additional error details
        
    Returns:
        Appropriate AbstractLLM exception
    """
    # Common error mappings
    error_map = {
        "auth": AuthenticationError,
        "quota": QuotaExceededError,
        "invalid_request": InvalidRequestError,
        "timeout": RequestTimeoutError,
        "connection": ProviderConnectionError,
        "model_not_found": ModelNotFoundError,
        "content_filter": ContentFilterError,
        "context_window": ContextWindowExceededError,
        "unsupported_feature": UnsupportedFeatureError,
        "resource": ResourceError,
        "media": MediaProcessingError,
        "file": FileProcessingError,
        "input": InvalidInputError,
        "audio": AudioOutputError
    }
    
    # Get exception class
    exception_class = error_map.get(error_type, ProviderAPIError)
    
    # Create and return exception
    return exception_class(
        message=message,
        provider=provider,
        original_exception=original_exception,
        details=details
    ) 