# Custom Provider Implementation Guide

This guide provides a detailed walkthrough of how to create custom provider implementations for AbstractLLM. By following these steps, you can extend AbstractLLM to support additional LLM services or specialized integrations.

## Prerequisites

Before implementing a custom provider, ensure you have:

- Familiarized yourself with the AbstractLLM architecture
- Understood the AbstractLLMInterface and its methods
- Set up a development environment with AbstractLLM installed

## Basic Implementation Steps

### 1. Create a Provider Class

Your provider class must inherit from `AbstractLLMInterface` and implement all required methods:

```python
from abstractllm.interface import AbstractLLMInterface
from abstractllm.utils.config import ConfigurationManager
from abstractllm.enums import ModelCapability
from typing import Any, Dict, List, Optional, Union, Generator, AsyncGenerator
import asyncio

class MyCustomProvider(AbstractLLMInterface):
    """Custom provider implementation for MyService."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the custom provider."""
        super().__init__(config)
        
        # Initialize your provider-specific client or API
        self.api_key = self.config_manager.get_param("api_key")
        self.api_base = self.config_manager.get_param("api_base", "https://api.myservice.com")
        self.model = self.config_manager.get_param("model", "default-model")
        
        # Initialize your provider's client
        # self.client = MyServiceClient(api_key=self.api_key, api_base=self.api_base)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the provider configuration."""
        if not self.api_key:
            # Check environment variable
            import os
            self.api_key = os.environ.get("MYSERVICE_API_KEY")
            
            if not self.api_key:
                from abstractllm.exceptions import AuthenticationError
                raise AuthenticationError("API key is required for MyService provider.")
        
        if not self.model:
            from abstractllm.exceptions import InvalidParameterError
            raise InvalidParameterError("Model name is required for MyService provider.")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using the LLM."""
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Prepare generation parameters
        params = self._get_generation_parameters()
        
        # Process files if provided
        if files:
            # Handle files according to your provider's API
            pass
        
        # Format messages
        messages = self._format_messages(prompt, system_prompt)
        
        # Process tools if provided
        if tools:
            # Handle tools according to your provider's API
            tools_formatted = self._prepare_tools(tools)
            if tools_formatted:
                params["tools"] = tools_formatted
        
        try:
            if stream:
                # Streaming implementation
                return self._generate_stream(messages, params)
            else:
                # Non-streaming implementation
                return self._generate_complete(messages, params)
        except Exception as e:
            # Handle errors
            self._handle_error(e)
    
    def _generate_complete(self, messages, params):
        """Generate a complete response (non-streaming)."""
        # Implement your API call here
        # response = self.client.complete(messages=messages, **params)
        # return response.text
        
        # Placeholder implementation
        return f"Response from {self.model} for prompt: {messages[-1]['content']}"
    
    def _generate_stream(self, messages, params):
        """Generate a streaming response."""
        # Implement your streaming API call here
        # for chunk in self.client.stream(messages=messages, **params):
        #     yield chunk.text
        
        # Placeholder implementation
        response = f"Response from {self.model} for prompt: {messages[-1]['content']}"
        for word in response.split():
            yield word + " "
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False,
                          tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response using the LLM."""
        # For simple implementation, you can wrap the synchronous method
        if stream:
            async def async_stream():
                for chunk in self.generate(prompt, system_prompt, files, True, tools, **kwargs):
                    yield chunk
            return async_stream()
        else:
            return await asyncio.to_thread(
                self.generate, prompt, system_prompt, files, False, tools, **kwargs
            )
    
    def _format_messages(self, prompt, system_prompt=None):
        """Format messages for the provider's API."""
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _get_generation_parameters(self):
        """Get generation parameters for the provider's API."""
        return {
            "model": self.model,
            "temperature": self.config_manager.get_param("temperature", 0.7),
            "max_tokens": self.config_manager.get_param("max_tokens", 1000),
            # Add other parameters specific to your provider
        }
    
    def _prepare_tools(self, tools):
        """Convert tools to the provider's format."""
        # Implement tool conversion for your provider
        # This will depend on how your provider handles tool/function calling
        return None
    
    def _handle_error(self, e):
        """Map provider-specific errors to AbstractLLM errors."""
        from abstractllm.exceptions import (
            AbstractLLMError,
            ProviderAPIError,
            AuthenticationError,
            InvalidRequestError,
            QuotaExceededError,
            ContextWindowExceededError,
            RequestTimeoutError,
        )
        
        # Map provider-specific errors to AbstractLLM errors
        # Example:
        # if isinstance(e, MyServiceAuthError):
        #     raise AuthenticationError("Authentication failed", provider="myservice", original_exception=e)
        # elif isinstance(e, MyServiceRateLimitError):
        #     raise QuotaExceededError("Rate limit exceeded", provider="myservice", original_exception=e)
        # else:
        #     raise ProviderAPIError(f"API error: {str(e)}", provider="myservice", original_exception=e)
        
        # Generic error handling
        raise ProviderAPIError(f"MyService API error: {str(e)}", provider="myservice", original_exception=e)
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM."""
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 4096,
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.TOOL_USE: False,
            ModelCapability.VISION: False,
            ModelCapability.JSON_MODE: False
        }
```

### 2. Register Your Provider

Once you've implemented your provider class, you need to register it with AbstractLLM's factory system:

```python
from abstractllm.factory import register_provider

# Register your provider
register_provider("myservice", MyCustomProvider)
```

### 3. Use Your Provider

After registering, you can use your provider like any other AbstractLLM provider:

```python
from abstractllm import create_llm

# Create an instance of your provider
llm = create_llm(
    "myservice", 
    api_key="your-api-key",
    model="your-model-name"
)

# Generate a response
response = llm.generate("Hello, world!")
print(response)
```

## Advanced Implementation Techniques

### Supporting Streaming

For proper streaming support, implement a generator function:

```python
def _generate_stream(self, messages, params):
    """Generate a streaming response."""
    # Make API call to your service with streaming enabled
    response_stream = self.client.stream(messages=messages, **params)
    
    # Yield chunks as they arrive
    for chunk in response_stream:
        if hasattr(chunk, 'text') and chunk.text:
            yield chunk.text
        elif hasattr(chunk, 'content') and chunk.content:
            yield chunk.content
        # Handle any other format your API might return
```

### Implementing Tool Calling

To support tool/function calling:

```python
def _prepare_tools(self, tools):
    """Convert tools to the provider's format."""
    if not tools:
        return None
    
    provider_tools = []
    for tool in tools:
        if callable(tool):
            # Convert function to tool definition
            from abstractllm.tools import ToolDefinition
            tool_def = ToolDefinition.from_function(tool)
            tool = tool_def
        
        if isinstance(tool, ToolDefinition):
            # Convert AbstractLLM ToolDefinition to provider format
            provider_tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema
            })
    
    return provider_tools

def _handle_tool_calls(self, response):
    """Handle tool calls from the provider response."""
    # Extract tool calls from your provider's response format
    # Convert to AbstractLLM's ToolCall format
    from abstractllm.tools import ToolCall, ToolCallRequest
    
    # Example implementation:
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        return response.text
    
    content = response.text or ""
    tool_calls = []
    
    for i, call in enumerate(response.tool_calls):
        tool_calls.append(ToolCall(
            id=f"call_{i}",
            name=call.get('name'),
            arguments=call.get('arguments', {})
        ))
    
    return ToolCallRequest(content=content, tool_calls=tool_calls)
```

### Supporting Vision Models

For vision capabilities:

```python
def _process_image(self, image):
    """Process image input for the provider."""
    from abstractllm.media import MediaFactory
    
    # Convert to provider-specific format
    if isinstance(image, str):
        # Create image input from path, URL, or base64
        image_input = MediaFactory.from_source(image, media_type="image")
    else:
        # Assume it's already an ImageInput
        image_input = image
    
    # Convert to your provider's format
    return image_input.to_provider_format("myservice")

def generate(self, prompt, system_prompt=None, files=None, stream=False, tools=None, **kwargs):
    """Generate a response, with vision support."""
    # Check for image in kwargs
    image = kwargs.pop('image', None)
    images = kwargs.pop('images', None)
    
    if image:
        # Process single image
        processed_image = self._process_image(image)
        # Add to your provider's request format
        # ...
    
    if images:
        # Process multiple images
        processed_images = [self._process_image(img) for img in images]
        # Add to your provider's request format
        # ...
    
    # Continue with normal generation
    # ...
```

## Best Practices

### Error Handling

Implement robust error handling by mapping provider-specific errors to AbstractLLM's error types:

```python
def _handle_error(self, e):
    """Map provider-specific errors to AbstractLLM errors."""
    from abstractllm.exceptions import (
        AbstractLLMError,
        ProviderAPIError,
        AuthenticationError,
        InvalidRequestError,
        QuotaExceededError,
        ContextWindowExceededError,
        RequestTimeoutError,
        ProviderConnectionError,
    )
    
    # MyService specific error types
    if isinstance(e, MyServiceAuthError):
        raise AuthenticationError(
            "Authentication failed for MyService",
            provider="myservice",
            original_exception=e
        )
    elif isinstance(e, MyServiceRateLimitError):
        raise QuotaExceededError(
            "Rate limit exceeded for MyService",
            provider="myservice",
            original_exception=e
        )
    elif isinstance(e, MyServiceTimeoutError):
        raise RequestTimeoutError(
            "Request timed out for MyService",
            provider="myservice",
            original_exception=e
        )
    elif isinstance(e, MyServiceConnectionError):
        raise ProviderConnectionError(
            "Failed to connect to MyService API",
            provider="myservice",
            original_exception=e
        )
    elif isinstance(e, MyServiceContextLimitError):
        raise ContextWindowExceededError(
            "Context window exceeded for MyService",
            provider="myservice",
            original_exception=e
        )
    elif isinstance(e, MyServiceInvalidRequestError):
        raise InvalidRequestError(
            f"Invalid request to MyService: {str(e)}",
            provider="myservice",
            original_exception=e
        )
    else:
        # Generic error handling
        raise ProviderAPIError(
            f"MyService API error: {str(e)}",
            provider="myservice",
            original_exception=e
        )
```

### Configuration Validation

Add thorough configuration validation:

```python
def _validate_config(self):
    """Validate the provider configuration."""
    from abstractllm.exceptions import AuthenticationError, InvalidParameterError
    
    # Check API key
    if not self.api_key:
        # Check environment variable
        import os
        self.api_key = os.environ.get("MYSERVICE_API_KEY")
        
        if not self.api_key:
            raise AuthenticationError(
                "API key is required for MyService provider. "
                "Provide it as 'api_key' parameter or set MYSERVICE_API_KEY environment variable."
            )
    
    # Check model
    if not self.model:
        raise InvalidParameterError("Model name is required for MyService provider.")
    
    # Check API base URL format
    if self.api_base and not self.api_base.startswith(("http://", "https://")):
        raise InvalidParameterError(
            f"Invalid API base URL for MyService: {self.api_base}. "
            "Must start with http:// or https://."
        )
```

### Package Your Provider

For reusable providers, consider packaging as a separate Python package:

```python
# File structure
myservice_abstractllm/
├── __init__.py
├── provider.py
├── setup.py
└── README.md
```

With setup.py:

```python
from setuptools import setup, find_packages

setup(
    name="myservice-abstractllm",
    version="0.1.0",
    description="MyService provider for AbstractLLM",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "abstractllm>=0.5.0",
        "myservice-python-sdk>=1.0.0",  # Your provider's Python SDK
    ],
    entry_points={
        "abstractllm.providers": [
            "myservice=myservice_abstractllm.provider:MyCustomProvider",
        ],
    },
)
```

## Real-World Example

Here's a simplified example of a custom provider for LiteLLM:

```python
from abstractllm.interface import AbstractLLMInterface
from abstractllm.exceptions import (
    ProviderAPIError, AuthenticationError, QuotaExceededError,
    InvalidRequestError, RequestTimeoutError, ContextWindowExceededError
)
from abstractllm.enums import ModelCapability
from typing import Any, Dict, List, Optional, Union, Generator, AsyncGenerator
import asyncio

class LiteLLMProvider(AbstractLLMInterface):
    """LiteLLM provider for AbstractLLM."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LiteLLM provider."""
        super().__init__(config)
        
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError("LiteLLM is required for this provider. Install with 'pip install litellm'.")
        
        # Get configuration
        self.model = self.config_manager.get_param("model")
        self.api_key = self.config_manager.get_param("api_key")
        
        # Set API key if provided
        if self.api_key:
            self.litellm.api_key = self.api_key
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the provider configuration."""
        if not self.model:
            raise InvalidRequestError("Model name is required for LiteLLM provider.")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate a response using LiteLLM."""
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Prepare messages
        messages = self._format_messages(prompt, system_prompt)
        
        # Prepare parameters
        params = self._get_generation_parameters()
        
        try:
            if stream:
                # Streaming implementation
                return self._generate_stream(messages, params)
            else:
                # Complete implementation
                response = self.litellm.completion(
                    model=self.model,
                    messages=messages,
                    **params
                )
                return response.choices[0].message.content
        except Exception as e:
            # Handle errors
            self._handle_error(e)
    
    def _generate_stream(self, messages, params):
        """Generate a streaming response."""
        stream_response = self.litellm.completion(
            model=self.model,
            messages=messages,
            stream=True,
            **params
        )
        
        for chunk in stream_response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False,
                          tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """Asynchronously generate a response using LiteLLM."""
        # Update config with any provided kwargs
        if kwargs:
            self.config_manager.update_config(kwargs)
        
        # Prepare messages
        messages = self._format_messages(prompt, system_prompt)
        
        # Prepare parameters
        params = self._get_generation_parameters()
        
        try:
            if stream:
                # Async streaming
                async def async_stream():
                    async_response = await self.litellm.acompletion(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **params
                    )
                    
                    async for chunk in async_response:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                
                return async_stream()
            else:
                # Async complete
                response = await self.litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    **params
                )
                return response.choices[0].message.content
        except Exception as e:
            # Handle errors
            self._handle_error(e)
    
    def _format_messages(self, prompt, system_prompt=None):
        """Format messages for LiteLLM."""
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _get_generation_parameters(self):
        """Get generation parameters for LiteLLM."""
        return {
            "temperature": self.config_manager.get_param("temperature", 0.7),
            "max_tokens": self.config_manager.get_param("max_tokens", 1000),
            "top_p": self.config_manager.get_param("top_p", 0.95),
            "frequency_penalty": self.config_manager.get_param("frequency_penalty", 0),
            "presence_penalty": self.config_manager.get_param("presence_penalty", 0),
        }
    
    def _handle_error(self, e):
        """Map LiteLLM errors to AbstractLLM errors."""
        error_str = str(e).lower()
        
        if "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError(
                f"LiteLLM authentication error: {str(e)}",
                provider="litellm",
                original_exception=e
            )
        elif "rate limit" in error_str or "quota" in error_str:
            raise QuotaExceededError(
                f"LiteLLM rate limit exceeded: {str(e)}",
                provider="litellm",
                original_exception=e
            )
        elif "timeout" in error_str:
            raise RequestTimeoutError(
                f"LiteLLM request timed out: {str(e)}",
                provider="litellm",
                original_exception=e
            )
        elif "context" in error_str and ("length" in error_str or "window" in error_str):
            raise ContextWindowExceededError(
                f"LiteLLM context window exceeded: {str(e)}",
                provider="litellm",
                original_exception=e
            )
        elif "invalid" in error_str or "bad request" in error_str:
            raise InvalidRequestError(
                f"LiteLLM invalid request: {str(e)}",
                provider="litellm",
                original_exception=e
            )
        else:
            raise ProviderAPIError(
                f"LiteLLM API error: {str(e)}",
                provider="litellm",
                original_exception=e
            )
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of LiteLLM provider."""
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: 4096,  # Depends on the model
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,  # Would need implementation
            ModelCapability.TOOL_USE: False,
            ModelCapability.VISION: False,  # Would need implementation
            ModelCapability.JSON_MODE: False,
        }
```

Register this provider:

```python
from abstractllm.factory import register_provider
from .provider import LiteLLMProvider

register_provider("litellm", LiteLLMProvider)
```

## Conclusion

Creating a custom provider allows you to extend AbstractLLM's capabilities to work with any LLM service or API. By following the patterns and best practices outlined in this guide, you can build robust, well-integrated provider implementations that work seamlessly with the rest of the AbstractLLM ecosystem.

## Next Steps

- [Provider-Specific Features](../providers/index.md): Learn about existing provider implementations for inspiration
- [AbstractLLM Architecture](../architecture/index.md): Understand the overall AbstractLLM architecture
- [Tool System](../architecture/tools.md): Detailed information about implementing tool support 