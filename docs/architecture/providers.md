# Provider Implementations

This document explains the implementation details of each provider in AbstractLLM. It covers how each provider is structured, its unique features, and how it interacts with the broader architecture.

## Provider Architecture Overview

All providers in AbstractLLM implement the `AbstractLLMInterface` abstract base class, which defines a common interface for interacting with LLM services.

```python
# From abstractllm/interface.py
class AbstractLLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                **kwargs) -> Union[str, Generator[str, None, None], Generator[Dict[str, Any], None, None]]:
        """Generate a response using the LLM."""
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None, 
                          files: Optional[List[Union[str, Path]]] = None,
                          stream: bool = False, 
                          tools: Optional[List[Union[Dict[str, Any], callable]]] = None,
                          **kwargs) -> Union[str, AsyncGenerator[str, None], AsyncGenerator[Dict[str, Any], None]]:
        """Asynchronously generate a response using the LLM."""
        pass
        
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Return capabilities of this LLM."""
        pass
```

Each provider follows a consistent implementation pattern:

1. **Initialization**: Configure the provider with API keys, default settings, and model parameters
2. **Parameter Management**: Handle provider-specific parameters through a `ConfigurationManager`
3. **Request Formatting**: Transform AbstractLLM's unified parameters into provider-specific formats
4. **API Interaction**: Make calls to the provider's API (with error handling)
5. **Response Processing**: Extract the generated content from provider-specific response formats
6. **Capability Reporting**: Report available capabilities based on the selected model

## OpenAI Provider

### Implementation

The OpenAI provider (`OpenAIProvider`) is implemented in `abstractllm/providers/openai.py` and handles interactions with the OpenAI API.

#### Key Components

```python
class OpenAIProvider(AbstractLLMInterface):
    """Provider for OpenAI models."""
    
    def __init__(self, **config):
        """Initialize the OpenAI provider with configuration."""
        # Set up configuration manager
        self.config_manager = ConfigurationManager({
            ModelParameter.PROVIDER: "openai",
            ModelParameter.MODEL: "gpt-4-turbo",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 1000,
            # Other parameters...
        })
        
        # Update with user-provided config
        if config:
            self.config_manager.update_config(config)
            
        # Set up API client
        self._setup_client()
```

#### Message Formatting

OpenAI uses a specific message format for conversations:

```python
def _format_messages(self, prompt, system_prompt, messages):
    """Format messages for the OpenAI API."""
    if messages:
        # Use provided messages
        return messages
    
    # Build messages list
    result = []
    
    # Add system message if provided
    if system_prompt:
        result.append({"role": "system", "content": system_prompt})
    
    # Add user message
    result.append({"role": "user", "content": prompt})
    
    return result
```

#### Tool Call Handling

OpenAI has a specific format for function/tool calling:

```python
def _process_tools(self, tools):
    """Process tools into OpenAI's expected format."""
    processed_tools = []
    
    for tool in tools:
        if callable(tool):
            # Convert Python function to OpenAI tool format
            tool_def = function_to_tool_definition(tool)
        elif isinstance(tool, dict):
            # Use provided tool definition
            tool_def = tool
        else:
            tool_def = tool.to_dict()
        
        processed_tools.append({
            "type": "function",
            "function": {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "parameters": tool_def["input_schema"]
            }
        })
    
    return processed_tools
```

### Unique Features

- **Function Calling**: Comprehensive support for OpenAI's function calling format
- **Vision Support**: Special handling for GPT-4 Vision models
- **Streaming**: Token-by-token streaming support
- **Response Format Control**: Support for JSON mode and other response formats

## Anthropic Provider

### Implementation

The Anthropic provider (`AnthropicProvider`) is implemented in `abstractllm/providers/anthropic.py` and handles interactions with Anthropic's Claude models.

#### Key Components

```python
class AnthropicProvider(AbstractLLMInterface):
    """Provider for Anthropic Claude models."""
    
    def __init__(self, **config):
        """Initialize the Anthropic provider with configuration."""
        # Set up configuration manager
        self.config_manager = ConfigurationManager({
            ModelParameter.PROVIDER: "anthropic",
            ModelParameter.MODEL: "claude-3-opus-20240229",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 1000,
            # Other parameters...
        })
        
        # Update with user-provided config
        if config:
            self.config_manager.update_config(config)
            
        # Set up API client
        self._setup_client()
```

#### Message Formatting

Anthropic uses a different message format than OpenAI:

```python
def _format_messages(self, prompt, system_prompt, files):
    """Format messages for the Anthropic API."""
    messages = []
    
    # Process user message with files if present
    content = []
    
    # Add files first if any
    if files:
        for file in self._process_files(files):
            content.append(file.to_provider_format("anthropic"))
    
    # Add text prompt
    if prompt:
        content.append({
            "type": "text",
            "text": prompt
        })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": content
    })
    
    return messages, system_prompt
```

#### Tool Call Handling

Anthropic's tool use format differs from OpenAI's:

```python
def _process_tools(self, tools):
    """Process tools into Anthropic's expected format."""
    processed_tools = []
    
    for tool in tools:
        if callable(tool):
            # Convert Python function to Anthropic tool format
            tool_def = function_to_tool_definition(tool)
        elif isinstance(tool, dict):
            # Use provided tool definition
            tool_def = tool
        else:
            tool_def = tool.to_dict()
        
        processed_tools.append({
            "name": tool_def["name"],
            "description": tool_def["description"],
            "input_schema": tool_def["input_schema"]
        })
    
    return processed_tools
```

### Unique Features

- **Multimodal Content**: Native handling of text and images in the same message
- **System Prompts**: Special handling for Claude's system prompt format
- **Long Context Windows**: Support for Claude's very large context windows
- **Content Sanitization**: Special attention to Claude's content policies

## Ollama Provider

### Implementation

The Ollama provider (`OllamaProvider`) is implemented in `abstractllm/providers/ollama.py` and handles interactions with the Ollama API for local LLM deployment.

#### Key Components

```python
class OllamaProvider(AbstractLLMInterface):
    """Provider for Ollama models."""
    
    def __init__(self, **config):
        """Initialize the Ollama provider with configuration."""
        # Set up configuration manager
        self.config_manager = ConfigurationManager({
            ModelParameter.PROVIDER: "ollama",
            ModelParameter.MODEL: "llama3",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 1000,
            ModelParameter.API_BASE: "http://localhost:11434",
            # Other parameters...
        })
        
        # Update with user-provided config
        if config:
            self.config_manager.update_config(config)
```

#### API Interaction

Ollama's API is HTTP-based but differs from cloud providers:

```python
def _make_request(self, endpoint, request_data):
    """Make a request to the Ollama API."""
    api_base = self.config_manager.get_param(ModelParameter.API_BASE)
    url = f"{api_base}/{endpoint}"
    
    try:
        response = requests.post(url, json=request_data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ProviderConnectionError(
            f"Failed to connect to Ollama: {str(e)}",
            provider="ollama",
            original_exception=e
        )
```

#### Model Capabilities Detection

Ollama requires special handling to determine model capabilities:

```python
def get_capabilities(self):
    """Return capabilities of the Ollama provider."""
    # Get current model
    model = self.config_manager.get_param(ModelParameter.MODEL)
    
    # Check if model is vision-capable
    has_vision = any(vm in model for vm in VISION_CAPABLE_MODELS)
    
    # Check if model supports tool calls
    has_tool_calls = model in TOOL_CAPABLE_MODELS
    
    return {
        ModelCapability.STREAMING: True,
        ModelCapability.MAX_TOKENS: 4096,  # Varies by model
        ModelCapability.SYSTEM_PROMPT: True,
        ModelCapability.ASYNC: True,
        ModelCapability.FUNCTION_CALLING: has_tool_calls,
        ModelCapability.TOOL_USE: has_tool_calls,
        ModelCapability.VISION: has_vision
    }
```

### Unique Features

- **Local Deployment**: No external API calls required
- **Model Management**: Can pull and manage models
- **Custom Model Support**: Support for custom models deployed with Ollama
- **Reduced Latency**: Lower latency due to local deployment

## HuggingFace Provider

### Implementation

The HuggingFace provider (`HuggingFaceProvider`) is implemented in `abstractllm/providers/huggingface.py` and offers the most flexibility but also complexity due to its support for many model types.

#### Key Components

```python
class HuggingFaceProvider(AbstractLLMInterface):
    """Provider for HuggingFace models."""
    
    def __init__(self, **config):
        """Initialize the HuggingFace provider with configuration."""
        # Set up configuration manager
        self.config_manager = ConfigurationManager({
            ModelParameter.PROVIDER: "huggingface",
            ModelParameter.MODEL: "google/gemma-7b",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 1000,
            # Other parameters...
        })
        
        # Update with user-provided config
        if config:
            self.config_manager.update_config(config)
            
        # Load model (lazy loading)
        self._model = None
        self._tokenizer = None
```

#### Model Loading

HuggingFace implements lazy loading for efficiency:

```python
def _load_model(self):
    """Load the model and tokenizer if not already loaded."""
    if self._model is not None:
        return
    
    model_name = self.config_manager.get_param(ModelParameter.MODEL)
    
    try:
        # Check if model is a .gguf file (llama.cpp)
        if model_name.endswith('.gguf'):
            self._load_gguf_model(model_name)
        else:
            self._load_transformers_model(model_name)
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load model {model_name}: {str(e)}",
            provider="huggingface",
            original_exception=e
        )
```

#### Format Handling

HuggingFace's diverse model types require different prompt formatting:

```python
def _format_prompt(self, prompt, system_prompt):
    """Format prompt for the specific model architecture."""
    model_name = self.config_manager.get_param(ModelParameter.MODEL)
    
    # Handle Llama models
    if "llama" in model_name.lower():
        if system_prompt:
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    # Handle Mistral models
    elif "mistral" in model_name.lower():
        if system_prompt:
            return f"<s>[INST] {system_prompt}\n{prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    # Handle Gemma models
    elif "gemma" in model_name.lower():
        if system_prompt:
            return f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        else:
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Default case
    else:
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        else:
            return prompt
```

### Unique Features

- **Multiple Model Types**: Support for Transformers, GGUF, and other formats
- **Local Execution**: Models run locally, reducing latency and privacy concerns
- **Custom Model Support**: Support for custom fine-tuned models
- **Model Caching**: Efficient caching of models to disk

## Provider Selection and Registration

Providers are registered in the provider registry (`abstractllm/providers/registry.py`), which handles provider selection and instantiation:

```python
# From abstractllm/providers/registry.py
class ProviderRegistry:
    """Registry for LLM providers."""
    
    _providers = {}
    
    @classmethod
    def register(cls, name, provider_class):
        """Register a provider class with a name."""
        cls._providers[name] = provider_class
    
    @classmethod
    def get(cls, name):
        """Get a provider class by name."""
        if name not in cls._providers:
            raise UnsupportedProviderError(f"Provider '{name}' is not supported")
        
        return cls._providers[name]
    
    @classmethod
    def create(cls, name, **config):
        """Create a provider instance by name with configuration."""
        provider_class = cls.get(name)
        
        try:
            return provider_class(**config)
        except Exception as e:
            raise ProviderInitializationError(
                f"Failed to initialize provider '{name}': {str(e)}",
                provider=name,
                original_exception=e
            )
```

Each provider is registered when the package is imported:

```python
# Register providers
from abstractllm.providers.openai import OpenAIProvider
from abstractllm.providers.anthropic import AnthropicProvider
from abstractllm.providers.ollama import OllamaProvider
from abstractllm.providers.huggingface import HuggingFaceProvider

ProviderRegistry.register("openai", OpenAIProvider)
ProviderRegistry.register("anthropic", AnthropicProvider)
ProviderRegistry.register("ollama", OllamaProvider)
ProviderRegistry.register("huggingface", HuggingFaceProvider)
```

## Provider Dependencies

AbstractLLM implements lazy loading of provider dependencies to minimize installation requirements:

```python
# From abstractllm/providers/openai.py
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Later in the code
def _setup_client(self):
    """Set up the OpenAI client."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    # Create client
    # ...
```

This allows users to install only the dependencies they need for the providers they plan to use.

## Provider Capabilities

Each provider reports its capabilities through the `get_capabilities()` method:

```python
# Example from OpenAI provider
def get_capabilities(self):
    """Return capabilities of the OpenAI provider."""
    # Get current model
    model = self.config_manager.get_param(ModelParameter.MODEL)
    
    # Check if model is vision-capable
    has_vision = any(model.startswith(vm) for vm in VISION_CAPABLE_MODELS)
    
    return {
        ModelCapability.STREAMING: True,
        ModelCapability.MAX_TOKENS: 4096,  # This varies by model
        ModelCapability.SYSTEM_PROMPT: True,
        ModelCapability.ASYNC: True,
        ModelCapability.FUNCTION_CALLING: True,
        ModelCapability.TOOL_USE: True,
        ModelCapability.VISION: has_vision,
        ModelCapability.JSON_MODE: True
    }
```

These capabilities can be checked to ensure a provider supports the features needed for a specific application.

## Provider Error Handling

Each provider implements comprehensive error handling to normalize provider-specific errors:

```python
# Example from Anthropic provider
try:
    response = client.messages.create(**message_params)
    return response.content[0].text
except anthropic.APIError as e:
    if "rate limit" in str(e).lower():
        raise QuotaExceededError(
            f"Anthropic rate limit exceeded: {str(e)}",
            provider="anthropic",
            original_exception=e
        )
    else:
        raise ProviderAPIError(
            f"Anthropic API error: {str(e)}",
            provider="anthropic",
            original_exception=e
        )
except anthropic.APIConnectionError as e:
    raise ProviderConnectionError(
        f"Failed to connect to Anthropic API: {str(e)}",
        provider="anthropic",
        original_exception=e
    )
except Exception as e:
    raise ProviderAPIError(
        f"Anthropic API error: {str(e)}",
        provider="anthropic",
        original_exception=e
    )
```

This ensures consistent error handling regardless of the underlying provider.

## Provider-Specific Optimizations

Each provider implements optimizations specific to its API:

1. **OpenAI**: Efficient handling of tool calls, support for JSON mode
2. **Anthropic**: Multimodal content handling, long context support
3. **Ollama**: Efficient local model management, endpoint switching
4. **HuggingFace**: Model caching, architecture-specific prompt formatting

## Conclusion

AbstractLLM's provider system is designed for flexibility, extensibility, and consistent behavior across different LLM services. By implementing a common interface while respecting provider-specific details, it enables seamless switching between providers and abstracts away the complexity of each provider's unique API patterns. 