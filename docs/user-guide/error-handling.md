# Error Handling in AbstractLLM

AbstractLLM provides a structured approach to error handling, allowing you to gracefully manage failures when working with LLM providers. This guide explains how to handle errors effectively in your applications.

## Exception Hierarchy

AbstractLLM uses a hierarchy of exceptions to provide specific information about what went wrong:

```
AbstractLLMException (base exception)
├── ProviderException (base provider exception)
│   ├── RateLimitException
│   ├── AuthenticationException
│   ├── InvalidRequestException
│   ├── ContextLengthExceededException
│   ├── ServiceUnavailableException
│   └── ModelNotFoundException
├── ValidationException
├── ToolException
│   ├── ToolExecutionError
│   ├── ToolValidationError
│   └── ToolTimeoutError
└── ConfigurationException
```

## Basic Error Handling

### Catching Exceptions

You can catch exceptions when making LLM calls:

```python
from abstractllm import create_llm
from abstractllm.exceptions import (
    AbstractLLMException,
    RateLimitException,
    AuthenticationException,
    ServiceUnavailableException
)

provider = create_llm("openai", model="gpt-4")

try:
    response = provider.generate("Write a short poem.")
    print(response)
except RateLimitException:
    print("Rate limit exceeded. Please try again later.")
except AuthenticationException:
    print("Authentication failed. Check your API key.")
except ServiceUnavailableException:
    print("The service is currently unavailable. Please try again later.")
except AbstractLLMException as e:
    print(f"An error occurred: {str(e)}")
```

This pattern allows you to handle different types of errors differently.

### Handling Provider-Specific Errors

Provider-specific errors are mapped to AbstractLLM's general exception hierarchy:

```python
try:
    response = provider.generate("Write a short poem.")
except AbstractLLMException as e:
    # Access provider-specific details
    if hasattr(e, 'provider_error'):
        print(f"Provider error: {e.provider_error}")
    
    # Access the provider name
    if hasattr(e, 'provider_name'):
        print(f"Error from provider: {e.provider_name}")
    
    print(f"Error message: {str(e)}")
```

## Advanced Error Handling

### Retry Logic

For transient errors, you might want to implement retry logic:

```python
from abstractllm.exceptions import (
    AbstractLLMException, 
    ServiceUnavailableException,
    RateLimitException
)
import time

def generate_with_retry(provider, prompt, max_retries=3, backoff_factor=2):
    """Generate a response with retry logic for transient errors."""
    retries = 0
    while retries <= max_retries:
        try:
            return provider.generate(prompt)
        except (ServiceUnavailableException, RateLimitException) as e:
            retries += 1
            if retries > max_retries:
                raise
            
            # Exponential backoff
            wait_time = backoff_factor ** retries
            print(f"Retry {retries}/{max_retries} after {wait_time}s due to: {str(e)}")
            time.sleep(wait_time)
        except AbstractLLMException:
            # Don't retry other exceptions
            raise
```

### Fallback Providers

You can implement provider fallbacks for when one provider fails:

```python
from abstractllm import create_llm
from abstractllm.exceptions import AbstractLLMException

def generate_with_fallback(prompt, providers_config):
    """Try multiple providers in order until one succeeds."""
    last_exception = None
    
    for config in providers_config:
        try:
            provider = create_llm(**config)
            return provider.generate(prompt)
        except AbstractLLMException as e:
            print(f"Provider {config.get('provider')} failed: {str(e)}")
            last_exception = e
    
    # If we get here, all providers failed
    raise last_exception or RuntimeError("All providers failed")

# Example usage
response = generate_with_fallback(
    "Explain quantum computing simply.",
    [
        {"provider": "openai", "model": "gpt-4"},
        {"provider": "anthropic", "model": "claude-3-opus-20240229"},
        {"provider": "ollama", "model": "llama3"},
    ]
)
```

## Specific Exception Types

### Provider Exceptions

Provider exceptions are thrown when the LLM API returns an error:

1. **RateLimitException**: Thrown when you've exceeded the provider's rate limits
   ```python
   except RateLimitException as e:
       print(f"Rate limited. Try again in {e.retry_after} seconds if available.")
   ```

2. **AuthenticationException**: Thrown when authentication fails
   ```python
   except AuthenticationException:
       print("Check your API key or credentials.")
   ```

3. **InvalidRequestException**: Thrown when the request is invalid
   ```python
   except InvalidRequestException as e:
       print(f"Invalid request: {str(e)}")
   ```

4. **ContextLengthExceededException**: Thrown when the input is too long
   ```python
   except ContextLengthExceededException as e:
       print(f"Input too long for model {e.model}. Max tokens: {e.max_tokens}")
   ```

5. **ServiceUnavailableException**: Thrown when the provider is unavailable
   ```python
   except ServiceUnavailableException:
       print("Service unavailable. Try again later.")
   ```

6. **ModelNotFoundException**: Thrown when the requested model doesn't exist
   ```python
   except ModelNotFoundException as e:
       print(f"Model {e.model} not found. Available models: {', '.join(e.available_models) if e.available_models else 'unknown'}")
   ```

### Tool Exceptions

Tool exceptions are thrown when there's an issue with tool execution:

1. **ToolExecutionError**: When a tool fails during execution
   ```python
   except ToolExecutionError as e:
       print(f"Tool '{e.tool_name}' failed: {str(e)}")
   ```

2. **ToolValidationError**: When tool inputs fail validation
   ```python
   except ToolValidationError as e:
       print(f"Invalid input for tool '{e.tool_name}': {str(e)}")
   ```

3. **ToolTimeoutError**: When a tool execution times out
   ```python
   except ToolTimeoutError as e:
       print(f"Tool '{e.tool_name}' timed out after {e.timeout} seconds")
   ```

### Validation Exceptions

ValidationException is thrown when input validation fails:

```python
from abstractllm.exceptions import ValidationException

try:
    # Code that may raise ValidationException
    pass
except ValidationException as e:
    print(f"Validation error: {str(e)}")
```

### Configuration Exceptions

ConfigurationException is thrown when there's an issue with the configuration:

```python
from abstractllm.exceptions import ConfigurationException

try:
    # Code that may raise ConfigurationException
    pass
except ConfigurationException as e:
    print(f"Configuration error: {str(e)}")
```

## Error Handling with Sessions

When working with sessions, you can apply the same error handling patterns:

```python
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.exceptions import AbstractLLMException

provider = create_llm("openai", model="gpt-4")
session = Session(provider=provider)

try:
    response = session.generate("Tell me about AI.")
    print(response)
except AbstractLLMException as e:
    print(f"Error: {str(e)}")
```

## Logging Errors

AbstractLLM integrates with Python's logging system:

```python
import logging
from abstractllm import create_llm
from abstractllm.exceptions import AbstractLLMException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

provider = create_llm("openai", model="gpt-4")

try:
    response = provider.generate("Write a short poem.")
    print(response)
except AbstractLLMException as e:
    logger.error(f"LLM error: {str(e)}", exc_info=True)
```

## Error Handling with Async APIs

When using async APIs, the error handling pattern is the same but within an async context:

```python
import asyncio
from abstractllm import create_llm
from abstractllm.exceptions import AbstractLLMException

async def generate_async():
    provider = create_llm("openai", model="gpt-4")
    
    try:
        response = await provider.agenerate("Write a short poem.")
        return response
    except AbstractLLMException as e:
        print(f"Error: {str(e)}")
        return None

# Run the async function
response = asyncio.run(generate_async())
if response:
    print(response)
```

## Error Handling with Streaming

When streaming responses, you need to handle exceptions that may occur during streaming:

```python
from abstractllm import create_llm
from abstractllm.exceptions import AbstractLLMException

provider = create_llm("openai", model="gpt-4")

try:
    # Get a stream of responses
    for chunk in provider.generate_stream("Write a short poem."):
        print(chunk, end="", flush=True)
except AbstractLLMException as e:
    print(f"\nError during streaming: {str(e)}")
```

## Best Practices

1. **Always catch the base exception**: Always include a catch block for `AbstractLLMException` to ensure you don't miss any errors.

2. **Handle specific exceptions first**: Place more specific exception handlers before more general ones.

3. **Log detailed error information**: Log all relevant details about the error for debugging.

4. **Implement retries for transient errors**: Use retry logic for errors that might be temporary (rate limits, service unavailability).

5. **Provide user-friendly error messages**: Convert technical error messages into user-friendly information when building applications.

6. **Consider fallback strategies**: Have alternative providers or models ready for critical applications.

7. **Validate inputs before sending**: Catch potential issues early by validating inputs before sending them to the provider.

## Implementation Details

The exception hierarchy in AbstractLLM is designed to provide a consistent experience across different providers. Each provider adapter maps the provider-specific errors to AbstractLLM's general exception hierarchy.

```python
# Example of error mapping in a provider adapter
try:
    # Make API call to provider
    pass
except ProviderSpecificError as e:
    # Map to AbstractLLM exceptions
    if "rate limit" in str(e).lower():
        raise RateLimitException(str(e), provider_error=e)
    elif "authentication" in str(e).lower():
        raise AuthenticationException(str(e), provider_error=e)
    # ... more mappings ...
    else:
        raise ProviderException(str(e), provider_error=e)
```

This internal mapping ensures that you can use a consistent error handling approach regardless of which provider you're using.

## Conclusion

Robust error handling is essential for building reliable applications with AbstractLLM. By understanding the exception hierarchy and implementing appropriate error handling strategies, you can create resilient applications that gracefully handle failures and provide a better user experience. 