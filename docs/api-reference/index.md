# API Reference

This reference documentation provides detailed information about AbstractLLM's classes, methods, and functions. It's intended for developers who want to understand the library's API in depth.

## Core Components

### AbstractLLM Factory

The main entry point for creating LLM provider instances.

```python
from abstractllm import create_llm

# Signature
def create_llm(provider: str, **config) -> AbstractLLMInterface:
    """
    Create an LLM provider instance.
    
    Args:
        provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface')
        **config: Provider-specific configuration
        
    Returns:
        An initialized LLM interface
        
    Raises:
        UnsupportedProviderError: If the provider is not supported
        AuthenticationError: If API key validation fails
        ProviderInitializationError: If provider initialization fails
    """
```

### AbstractLLM Interface

The base interface that all provider implementations must follow.

```python
from abstractllm.interface import AbstractLLMInterface

class AbstractLLMInterface:
    """
    Abstract interface for LLM providers.
    
    All LLM providers must implement this interface to ensure a consistent API.
    Each provider is responsible for managing its own configuration and defaults.
    """
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                **kwargs) -> Union[str, Generator[str, None, None], Generator[Dict[str, Any], None, None]]:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt to set context
            files: Optional list of files to process (paths or URLs)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            **kwargs: Additional parameters to override configuration
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: A generator yielding response chunks
            If tools are used: May return a ToolCallRequest
            
        Raises:
            Various exceptions depending on provider implementation
        """
        
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False, 
                          tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                          **kwargs) -> Union[str, AsyncGenerator[str, None], AsyncGenerator[Dict[str, Any], None]]:
        """
        Asynchronously generate a response using the LLM.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt to set context
            files: Optional list of files to process (paths or URLs)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            **kwargs: Additional parameters to override configuration
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: An async generator yielding response chunks
            If tools are used: May return a ToolCallRequest
            
        Raises:
            Various exceptions depending on provider implementation
        """
        
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of this LLM.
        
        Returns:
            Dictionary mapping capability names to values
        """
```

## Sessions

AbstractLLM provides session management for maintaining conversation state.

```python
from abstractllm.session import Session

class Session:
    """
    A session for interacting with LLMs.
    
    Sessions maintain conversation history and can be persisted and loaded.
    """
    
    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 provider: Optional[Union[str, AbstractLLMInterface]] = None,
                 provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tools: Optional[List[Union[Dict[str, Any], Callable, "ToolDefinition"]]] = None):
        """Initialize a session."""
        
    def add_message(self, role: Union[str, MessageRole], content: str, name: Optional[str] = None,
                    tool_results: Optional[List[Dict[str, Any]]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to the session history."""
        
    def generate(self, prompt: Optional[str] = None, 
                provider: Optional[AbstractLLMInterface] = None,
                **kwargs) -> str:
        """Generate a response considering the conversation history."""
        
    def generate_with_tools(self, prompt: Optional[str] = None,
                          provider: Optional[AbstractLLMInterface] = None,
                          tool_functions: Optional[Dict[str, Callable]] = None,
                          **kwargs) -> Union[str, Dict[str, Any]]:
        """Generate a response using tools and considering the conversation history."""
        
    def get_history(self) -> List[Message]:
        """Get the conversation history."""
        
    def clear_history(self) -> None:
        """Clear the conversation history."""
        
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the session to a file."""
        
    @classmethod
    def load(cls, file_path: Union[str, Path], provider: Optional[AbstractLLMInterface] = None) -> "Session":
        """Load a session from a file."""
```

## Tool System

AbstractLLM provides a tool/function calling system.

```python
from abstractllm.tools import ToolDefinition, ToolCall, ToolCallRequest

class ToolDefinition:
    """Definition of a tool that can be called by an LLM."""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any], 
                output_schema: Optional[Dict[str, Any]] = None):
        """Initialize a tool definition."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        
    @classmethod
    def from_function(cls, func: Callable) -> "ToolDefinition":
        """Create a ToolDefinition from a Python function."""

class ToolCall:
    """A call to a tool made by the LLM."""
    
    def __init__(self, id: str, name: str, arguments: Dict[str, Any], result: Any = None):
        """Initialize a tool call."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        
    def set_result(self, result: Any) -> None:
        """Set the result of the tool call."""

class ToolCallRequest:
    """A request from the LLM to call one or more tools."""
    
    def __init__(self, content: str, tool_calls: List[ToolCall]):
        """Initialize a tool call request."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        
    def execute_tool_calls(self, tool_functions: Dict[str, Callable]) -> None:
        """Execute the tool calls with the provided functions."""
```

## Media System

AbstractLLM provides a media system for handling different types of inputs.

```python
from abstractllm.media import MediaInput, ImageInput, MediaFactory

class MediaInput(ABC):
    """
    Abstract base class for all media inputs.
    
    This class defines the interface that all media input types must implement.
    """
    
    @abstractmethod
    def to_provider_format(self, provider: str) -> Any:
        """Convert the media to a format suitable for the specified provider."""
        
    @property
    @abstractmethod
    def media_type(self) -> str:
        """Return the type of media (image, document, etc.)."""

class ImageInput(MediaInput):
    """
    A class for handling image inputs.
    
    Supports loading images from files, URLs, and base64 data.
    """
    
    def __init__(self, source: Union[str, Path, Dict[str, Any]], 
                mime_type: Optional[str] = None,
                detail_level: Optional[str] = None):
        """Initialize an image input."""
        
    def to_provider_format(self, provider: str) -> Any:
        """Convert the image to a format suitable for the specified provider."""
        
    @property
    def media_type(self) -> str:
        """Return the type of media."""
        
    @classmethod
    def from_file(cls, file_path: Union[str, Path], **kwargs) -> "ImageInput":
        """Create an ImageInput from a file."""
        
    @classmethod
    def from_url(cls, url: str, **kwargs) -> "ImageInput":
        """Create an ImageInput from a URL."""
        
    @classmethod
    def from_base64(cls, base64_data: str, mime_type: str, **kwargs) -> "ImageInput":
        """Create an ImageInput from base64 data."""

class MediaFactory:
    """
    Factory class for creating media inputs.
    
    Provides methods for creating media inputs from various sources.
    """
    
    @classmethod
    def from_source(cls, source: Union[str, Path, Dict[str, Any]], media_type: Optional[str] = None) -> MediaInput:
        """
        Create a media input object from a source.
        
        Args:
            source: File path, URL, base64 string, or provider-specific dict
            media_type: Explicit media type (optional, auto-detected if not provided)
            
        Returns:
            Appropriate MediaInput instance
        """
        
    @classmethod
    def process_files(cls, files: List[Union[str, Path, Dict[str, Any]]]) -> List[MediaInput]:
        """
        Process a list of files into MediaInput objects.
        
        Args:
            files: List of file paths, URLs, or MediaInput objects
            
        Returns:
            List of MediaInput objects
        """
```

## Configuration System

AbstractLLM provides a configuration system for managing provider parameters.

```python
from abstractllm.utils.config import ConfigurationManager, ModelParameter

class ModelParameter(str, Enum):
    """
    Parameters that can be configured for LLM providers.
    
    This enum provides a standardized set of parameter names that can be used
    across different providers. Each provider may support a subset of these parameters.
    """
    
    PROVIDER = "provider"
    MODEL = "model"
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    API_KEY = "api_key"
    API_BASE = "api_base"
    SYSTEM_PROMPT = "system_prompt"
    # ...other parameters

class ConfigurationManager:
    """
    Parameter management for AbstractLLM providers.
    Handles parameter storage, retrieval, and updates without provider-specific logic.
    """
    
    def __init__(self, initial_config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the configuration manager."""
        
    def get_param(self, param: Union[str, ModelParameter], default: Optional[T] = None) -> Optional[T]:
        """Get a parameter value from configuration, supporting both enum and string keys."""
        
    def update_config(self, updates: Dict[Union[str, ModelParameter], Any]) -> None:
        """Update configuration with new values."""
```

## Provider Registry

AbstractLLM provides a registry for managing provider implementations.

```python
from abstractllm.providers.registry import ProviderRegistry

class ProviderRegistry:
    """Registry for LLM providers."""
    
    @classmethod
    def register(cls, name: str, provider_class: Type[AbstractLLMInterface]) -> None:
        """Register a provider class with a name."""
        
    @classmethod
    def get(cls, name: str) -> Type[AbstractLLMInterface]:
        """Get a provider class by name."""
        
    @classmethod
    def create(cls, name: str, **config) -> AbstractLLMInterface:
        """Create a provider instance by name with configuration."""
        
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names."""
```

## Error Handling

AbstractLLM provides a comprehensive error hierarchy.

```python
from abstractllm.exceptions import AbstractLLMError

class AbstractLLMError(Exception):
    """Base class for all AbstractLLM exceptions."""
    
    def __init__(self, message: str, provider: Optional[str] = None, 
               original_exception: Optional[Exception] = None):
        """Initialize the exception."""

# Authentication errors
class AuthenticationError(AbstractLLMError):
    """Raised when authentication fails."""

class QuotaExceededError(AbstractLLMError):
    """Raised when the provider quota is exceeded."""

# Provider errors
class UnsupportedProviderError(AbstractLLMError):
    """Raised when an unsupported provider is requested."""

class UnsupportedModelError(AbstractLLMError):
    """Raised when an unsupported model is requested."""

class ModelNotFoundError(AbstractLLMError):
    """Raised when a model is not found."""

class InvalidRequestError(AbstractLLMError):
    """Raised when the request is invalid."""

class InvalidParameterError(InvalidRequestError):
    """Raised when a parameter is invalid."""

class ModelLoadingError(AbstractLLMError):
    """Raised when a model fails to load."""

class ProviderConnectionError(AbstractLLMError):
    """Raised when connection to the provider fails."""

class ProviderAPIError(AbstractLLMError):
    """Raised when the provider API returns an error."""

# Generation errors
class GenerationError(AbstractLLMError):
    """Raised when generation fails."""

class RequestTimeoutError(AbstractLLMError):
    """Raised when a request times out."""

class ContentFilterError(AbstractLLMError):
    """Raised when content is filtered by the provider."""

class ContextWindowExceededError(AbstractLLMError):
    """Raised when the context window is exceeded."""

# Feature errors
class UnsupportedFeatureError(AbstractLLMError):
    """Raised when a feature is not supported by the provider."""

class UnsupportedOperationError(AbstractLLMError):
    """Raised when an operation is not supported by the provider."""

# Media errors
class ImageProcessingError(AbstractLLMError):
    """Raised when image processing fails."""

class FileProcessingError(AbstractLLMError):
    """Raised when file processing fails."""
```

## Utilities

AbstractLLM provides various utility functions and classes.

```python
from abstractllm.utils.logging import enable_request_logging, enable_response_logging

def enable_request_logging(level: int = logging.INFO) -> None:
    """Enable logging of requests to the provider."""
    
def enable_response_logging(level: int = logging.INFO) -> None:
    """Enable logging of responses from the provider."""
    
def set_sensitive_values(values: List[str]) -> None:
    """Set additional values to be redacted from logs."""
    
def configure_audit_logging(filename: str, include_prompts: bool = True, 
                         include_responses: bool = True, 
                         include_metadata: bool = True) -> None:
    """Configure audit logging to a file."""
```

## Model Capabilities

AbstractLLM provides an enum for model capabilities.

```python
from abstractllm.enums import ModelCapability

class ModelCapability(str, Enum):
    """
    Capabilities that a model might support.
    
    This enum provides a standardized set of capability names that can be
    used across different providers to check what features a model supports.
    """
    
    STREAMING = "streaming"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "system_prompt"
    ASYNC = "async"
    FUNCTION_CALLING = "function_calling"
    TOOL_USE = "tool_use"
    VISION = "vision"
    JSON_MODE = "json_mode"
```

## Next Steps

For more detailed information about specific components:

- [Provider Implementations](../architecture/providers.md)
- [Media System](../architecture/media.md)
- [Tool System](../architecture/tools.md)
- [Configuration System](../architecture/configuration.md)
- [Error Handling](../architecture/error-handling.md) 