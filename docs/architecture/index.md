# AbstractLLM Architecture

This document provides an overview of AbstractLLM's architecture, explaining the key components and how they interact.

## Overview

AbstractLLM follows a modular, provider-based architecture that emphasizes extensibility, maintainability, and ease of use. The system provides a unified interface while maintaining provider-specific optimizations.

![AbstractLLM Architecture Overview](../assets/architecture-overview.svg)

## Core Components

The architecture consists of these key components:

### 1. Provider System

The provider system is the central abstraction layer that enables interaction with different LLM services through a unified interface.

```python
# From abstractllm/interface.py
class AbstractLLMInterface(ABC):
    """
    Abstract interface for LLM providers.
    
    All LLM providers must implement this interface to ensure a consistent API.
    Each provider is responsible for managing its own configuration and defaults.
    """
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """Generate a response using the LLM."""
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False, 
                          tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                          **kwargs) -> Union[GenerateResponse, AsyncGenerator[GenerateResponse, None]]:
        """Asynchronously generate a response using the LLM."""
        pass
        
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM."""
        # Default implementation returns standard capabilities
        # Provider-specific implementations override this
        pass
```

Each provider (OpenAI, Anthropic, Ollama, HuggingFace) implements this interface with provider-specific logic.

### 2. Factory System

The factory system provides a clean entry point through the `create_llm()` function, which instantiates the appropriate provider based on the provider name.

```python
# From abstractllm/factory.py
def create_llm(provider: str, **config) -> AbstractLLMInterface:
    """
    Create an LLM provider instance.
    
    Args:
        provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface')
        **config: Provider-specific configuration
        
    Returns:
        An initialized LLM interface
    """
    # Implementation details...
```

The factory handles provider validation, dependency checking, and API key validation before instantiating the provider.

### 3. Configuration Management

The configuration system provides a centralized way to manage provider settings through the `ConfigurationManager` class.

```python
# From abstractllm/utils/config.py
class ConfigurationManager:
    """
    Parameter management for AbstractLLM providers.
    Handles parameter storage, retrieval, and updates without provider-specific logic.
    """
    
    def __init__(self, initial_config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the configuration manager."""
        # Implementation details...
    
    def get_param(self, param: Union[str, ModelParameter], default: Optional[T] = None) -> Optional[T]:
        """Get a parameter value from configuration, supporting both enum and string keys."""
        # Implementation details...
    
    def update_config(self, updates: Dict[Union[str, ModelParameter], Any]) -> None:
        """Update configuration with new values."""
        # Implementation details...
```

Configuration is handled in layers:
1. Global defaults
2. Provider-specific defaults
3. User-provided creation parameters
4. Environment variables
5. Per-method parameters

### 4. Media System

The media system handles different input types, particularly for vision capabilities.

```python
# From abstractllm/media/interface.py
class MediaInput(ABC):
    """
    Abstract base class for all media inputs.
    
    This class defines the interface that all media input types must implement.
    """
    
    @abstractmethod
    def to_provider_format(self, provider: str) -> Any:
        """Convert the media to a format suitable for the specified provider."""
        pass
    
    @property
    @abstractmethod
    def media_type(self) -> str:
        """Return the type of media (image, document, etc.)."""
        pass
```

Implementations include:
- `ImageInput`: For image files, URLs, and base64 data
- `TextInput`: For text files and content
- `TabularInput`: For CSV and TSV data

### 5. Session Management

The session system maintains conversation context across interactions, particularly important for multi-turn dialogs.

```python
# From abstractllm/session.py
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
        # Implementation details...
    
    def add_message(self, role: Union[str, MessageRole], content: str, name: Optional[str] = None,
                    tool_results: Optional[List[Dict[str, Any]]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to the session history."""
        # Implementation details...
```

### 6. Tool System

The tool system provides a consistent interface for function/tool calling across providers.

```python
# From abstractllm/tools/types.py
class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by an LLM."""
    
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for inputs")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for return value")
```

## Data Flow

The data flow follows this general pattern:

1. User creates an LLM instance with `create_llm()`
2. Configuration is processed through `ConfigurationManager`
3. The provider class is instantiated with the configuration
4. User calls methods like `generate()` or `generate_async()`
5. Provider-specific parameters are extracted and formatted
6. Any media inputs are processed through the media system
7. The request is sent to the provider's API
8. Response is processed and returned to the user

![Data Flow Diagram](../assets/data-flow.svg)

## Provider Architecture

Each provider implements the AbstractLLMInterface with specific adaptations for the provider's API:

### Provider Components

1. **Initialization**: Setting up the provider with appropriate configuration and defaults
2. **Parameter Processing**: Converting AbstractLLM parameters to provider-specific parameters
3. **Request Formatting**: Preparing the request in the format expected by the provider
4. **Response Processing**: Extracting the relevant information from the provider's response
5. **Capability Reporting**: Determining which capabilities (streaming, vision, tools) are supported
6. **Tool Handling**: Processing tool definitions and tool calls

### Provider Adapter Pattern

The providers follow an adapter pattern, converting between AbstractLLM's unified interface and the provider-specific APIs:

```
┌─────────────┐     ┌───────────────────┐     ┌──────────────┐
│ AbstractLLM │ ──► │ Provider Adapter  │ ──► │ Provider API │
│  Interface  │ ◄── │ (OpenAI, etc.)    │ ◄── │              │
└─────────────┘     └───────────────────┘     └──────────────┘
```

## Error Handling

AbstractLLM provides a unified error handling system through a hierarchy of exceptions:

```
AbstractLLMError
  ├── AuthenticationError
  ├── QuotaExceededError
  ├── UnsupportedProviderError
  ├── UnsupportedModelError
  ├── ModelNotFoundError
  ├── InvalidRequestError
  │     └── InvalidParameterError
  ├── ModelLoadingError
  ├── ProviderConnectionError
  ├── ProviderAPIError
  ├── GenerationError
  ├── RequestTimeoutError
  ├── ContentFilterError
  ├── ContextWindowExceededError
  ├── UnsupportedFeatureError
  ├── UnsupportedOperationError
  ├── ImageProcessingError
  └── FileProcessingError
```

## Dependency Management

AbstractLLM uses a lazy loading approach for provider dependencies:

```python
# From abstractllm/providers/anthropic.py
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
```

This approach allows users to install only the dependencies they need for their specific use case.

## Security Considerations

AbstractLLM implements several security features, particularly for tool execution:

1. **Path Validation**: Prevents access to unauthorized directories
2. **Parameter Validation**: Validates parameters before execution
3. **Execution Timeouts**: Prevents resource exhaustion
4. **Secure Tool Wrappers**: Enforce validation, timeouts, and result sanitization
5. **Output Sanitization**: Limits output size and redacts sensitive information
6. **Configurable Security Settings**: Settings like max file size and execution time

## Extension Points

AbstractLLM is designed to be extensible in several ways:

1. **New Providers**: Implement the AbstractLLMInterface for new providers
2. **New Media Types**: Extend the MediaInput interface for new media types
3. **Custom Tools**: Define custom tools for specific use cases
4. **Custom Validation**: Add custom validation for tool parameters

## Next Steps

- [Provider Implementations](providers.md): Detailed explanation of each provider implementation
- [Media System](media.md): How the media system processes different input types
- [Tool System](tools.md): In-depth explanation of the tool system
- [Configuration System](configuration.md): How configuration is managed
- [Error Handling](error-handling.md): How errors are processed and reported 