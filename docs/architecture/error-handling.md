# Error Handling

This document explains AbstractLLM's error handling architecture, which provides a consistent way to handle errors across different providers.

## Overview

AbstractLLM implements a comprehensive error handling system with a hierarchy of exception types. This system translates provider-specific errors into a consistent set of exceptions, making it easier to handle errors across different providers.

## Exception Hierarchy

AbstractLLM defines a hierarchy of exception types:

```python
# From abstractllm/exceptions.py
class AbstractLLMError(Exception):
    """Base class for all AbstractLLM exceptions."""
    
    def __init__(self, message: str, provider: Optional[str] = None, 
                original_exception: Optional[Exception] = None):
        """
        Initialize the exception.
        
        Args:
            message: The error message
            provider: The provider that raised the error
            original_exception: The original exception that was raised
        """
        self.provider = provider
        self.original_exception = original_exception
        super().__init__(message)

# Authentication and access errors
class AuthenticationError(AbstractLLMError):
    """Raised when authentication fails."""
    pass

class QuotaExceededError(AbstractLLMError):
    """Raised when the API quota is exceeded."""
    pass

# Provider and model errors
class UnsupportedProviderError(AbstractLLMError):
    """Raised when an unsupported provider is specified."""
    pass

class UnsupportedModelError(AbstractLLMError):
    """Raised when an unsupported model is specified."""
    pass

class ModelNotFoundError(AbstractLLMError):
    """Raised when a model is not found."""
    pass

# Input errors
class InvalidRequestError(AbstractLLMError):
    """Raised when the request is invalid."""
    pass

class InvalidParameterError(InvalidRequestError):
    """Raised when a parameter is invalid."""
    pass

# Initialization errors
class ModelLoadingError(AbstractLLMError):
    """Raised when a model fails to load."""
    pass

# Connection errors
class ProviderConnectionError(AbstractLLMError):
    """Raised when connection to the provider fails."""
    pass

class ProviderAPIError(AbstractLLMError):
    """Raised when the provider API returns an error."""
    pass

# Generation errors
class GenerationError(AbstractLLMError):
    """Raised when text generation fails."""
    pass

class RequestTimeoutError(AbstractLLMError):
    """Raised when a request times out."""
    pass

class ContentFilterError(AbstractLLMError):
    """Raised when content is filtered by the provider."""
    pass

class ContextWindowExceededError(AbstractLLMError):
    """Raised when the context window is exceeded."""
    pass

# Feature support errors
class UnsupportedFeatureError(AbstractLLMError):
    """Raised when a feature is not supported by the provider or model."""
    pass

class UnsupportedOperationError(AbstractLLMError):
    """Raised when an operation is not supported by the provider or model."""
    pass

# Media errors
class ImageProcessingError(AbstractLLMError):
    """Raised when image processing fails."""
    pass

class FileProcessingError(AbstractLLMError):
    """Raised when file processing fails."""
    pass

# Tool errors
class ToolExecutionError(AbstractLLMError):
    """Raised when tool execution fails."""
    pass

class ToolValidationError(AbstractLLMError):
    """Raised when tool validation fails."""
    pass
```

## Provider Error Mapping

Each provider has its own error types, which AbstractLLM maps to its exception hierarchy:

### OpenAI Error Mapping

```python
# From abstractllm/providers/openai.py
def _handle_openai_error(self, e: Exception) -> None:
    """Map OpenAI error to AbstractLLM error."""
    # OpenAI API errors
    if isinstance(e, openai.APIConnectionError):
        raise ProviderConnectionError("Failed to connect to OpenAI API", 
                                    provider="openai", 
                                    original_exception=e)
    elif isinstance(e, openai.APITimeoutError):
        raise RequestTimeoutError("OpenAI API request timed out", 
                                provider="openai", 
                                original_exception=e)
    elif isinstance(e, openai.AuthenticationError):
        raise AuthenticationError("OpenAI API authentication failed", 
                                provider="openai", 
                                original_exception=e)
    elif isinstance(e, openai.RateLimitError):
        raise QuotaExceededError("OpenAI API rate limit exceeded", 
                                provider="openai", 
                                original_exception=e)
    elif isinstance(e, openai.BadRequestError):
        # Check for specific error types
        if "context_length_exceeded" in str(e):
            raise ContextWindowExceededError("OpenAI model context window exceeded", 
                                            provider="openai", 
                                            original_exception=e)
        else:
            raise InvalidRequestError(f"OpenAI API bad request: {str(e)}", 
                                    provider="openai", 
                                    original_exception=e)
    else:
        # Generic provider error
        raise ProviderAPIError(f"OpenAI API error: {str(e)}", 
                            provider="openai", 
                            original_exception=e)
```

### Anthropic Error Mapping

```python
# From abstractllm/providers/anthropic.py
def _handle_anthropic_error(self, e: Exception) -> None:
    """Map Anthropic error to AbstractLLM error."""
    # Anthropic API errors
    if isinstance(e, anthropic.APIConnectionError):
        raise ProviderConnectionError("Failed to connect to Anthropic API", 
                                    provider="anthropic", 
                                    original_exception=e)
    elif isinstance(e, anthropic.APITimeoutError):
        raise RequestTimeoutError("Anthropic API request timed out", 
                                provider="anthropic", 
                                original_exception=e)
    elif isinstance(e, anthropic.AuthenticationError):
        raise AuthenticationError("Anthropic API authentication failed", 
                                provider="anthropic", 
                                original_exception=e)
    elif isinstance(e, anthropic.RateLimitError):
        raise QuotaExceededError("Anthropic API rate limit exceeded", 
                                provider="anthropic", 
                                original_exception=e)
    elif isinstance(e, anthropic.BadRequestError):
        # Check for specific error types
        if "max_tokens" in str(e) or "token limit" in str(e):
            raise ContextWindowExceededError("Anthropic model context window exceeded", 
                                            provider="anthropic", 
                                            original_exception=e)
        else:
            raise InvalidRequestError(f"Anthropic API bad request: {str(e)}", 
                                    provider="anthropic", 
                                    original_exception=e)
    elif isinstance(e, anthropic.InternalServerError):
        raise ProviderAPIError("Anthropic API internal server error", 
                            provider="anthropic", 
                            original_exception=e)
    else:
        # Generic provider error
        raise ProviderAPIError(f"Anthropic API error: {str(e)}", 
                            provider="anthropic", 
                            original_exception=e)
```

## Error Handling in User Code

When using AbstractLLM, you can handle errors in a consistent way:

```python
from abstractllm import create_llm
from abstractllm.exceptions import (
    AbstractLLMError,
    AuthenticationError,
    QuotaExceededError,
    ContextWindowExceededError,
    RequestTimeoutError,
    ProviderConnectionError
)

# Create LLM
try:
    llm = create_llm("openai", model="gpt-4")
except AuthenticationError as e:
    print(f"Authentication failed: {str(e)}")
    print("Please check your API key")
except UnsupportedProviderError as e:
    print(f"Provider not supported: {str(e)}")
    print("Available providers: openai, anthropic, ollama, huggingface")
except AbstractLLMError as e:
    print(f"Error creating LLM: {str(e)}")

# Generate text
try:
    response = llm.generate("Write a very long story.")
    print(response)
except ContextWindowExceededError as e:
    print("The response was too long for the model's context window")
    print("Try a shorter prompt or reduce max_tokens")
except QuotaExceededError as e:
    print("API quota exceeded. Please try again later")
except RequestTimeoutError as e:
    print("Request timed out. Please try again")
except ProviderConnectionError as e:
    print("Failed to connect to the provider. Please check your internet connection")
except AbstractLLMError as e:
    print(f"Error generating text: {str(e)}")
```

## Error Details

AbstractLLM exceptions include useful details:

1. **Error Message**: A human-readable error message
2. **Provider**: The provider that raised the error
3. **Original Exception**: The original exception from the provider

You can access these details:

```python
try:
    response = llm.generate("Write a story.")
except AbstractLLMError as e:
    print(f"Error: {str(e)}")
    print(f"Provider: {e.provider}")
    if e.original_exception:
        print(f"Original error: {str(e.original_exception)}")
        print(f"Error type: {type(e.original_exception).__name__}")
```

## Debugging

For debugging, you can enable detailed error logging:

```python
import logging
from abstractllm.logging import setup_logging

# Set up logging with debug level
setup_logging(level=logging.DEBUG)

# Now errors will include detailed information
try:
    response = llm.generate("Write a story.")
except AbstractLLMError as e:
    # The error will also be logged with detailed information
    pass
```

## Custom Error Handling

You can implement custom error handling for specific use cases:

```python
class MyCustomErrorHandler:
    def __init__(self, fallback_provider=None):
        self.fallback_provider = fallback_provider
        
    def handle_error(self, error, llm, prompt, **kwargs):
        """Handle errors with custom logic."""
        if isinstance(error, QuotaExceededError):
            # Wait and retry
            import time
            print("Rate limit exceeded. Waiting 10 seconds...")
            time.sleep(10)
            return llm.generate(prompt, **kwargs)
            
        elif isinstance(error, ContextWindowExceededError):
            # Reduce the max tokens and try again
            print("Context window exceeded. Reducing max_tokens...")
            kwargs["max_tokens"] = kwargs.get("max_tokens", 1000) // 2
            return llm.generate(prompt, **kwargs)
            
        elif isinstance(error, ProviderConnectionError) and self.fallback_provider:
            # Try fallback provider
            print(f"Connection failed. Trying fallback provider {self.fallback_provider}...")
            fallback_llm = create_llm(self.fallback_provider)
            return fallback_llm.generate(prompt, **kwargs)
            
        else:
            # Re-raise other errors
            raise error

# Use custom error handler
handler = MyCustomErrorHandler(fallback_provider="anthropic")

try:
    response = llm.generate("Write a story.")
except AbstractLLMError as e:
    response = handler.handle_error(e, llm, "Write a story.")
```

## Resilient Applications

For building resilient applications, you can use the fallback chain feature:

```python
from abstractllm.chains import FallbackChain
from abstractllm.exceptions import AbstractLLMError

# Create providers
openai_llm = create_llm("openai", model="gpt-4")
anthropic_llm = create_llm("anthropic", model="claude-3-opus-20240229")
ollama_llm = create_llm("ollama", model="llama3")

# Create fallback chain
chain = FallbackChain(
    providers=[openai_llm, anthropic_llm, ollama_llm],
    error_types=[AbstractLLMError]  # Catch all AbstractLLM errors
)

# Generate with fallback
try:
    response = chain.generate("Write a story about space exploration.")
    print(f"Response from provider: {chain.last_successful_provider}")
    print(response)
except AbstractLLMError as e:
    print("All providers failed:", str(e))
```

## Next Steps

- [Configuration System](configuration.md): How configuration is managed
- [User Guide: Error Handling](../user-guide/error-handling.md): How to handle errors in your applications 