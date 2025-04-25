# Configuration System

This document explains AbstractLLM's configuration system, which provides a flexible way to manage settings and parameters across different providers.

## Overview

The configuration system in AbstractLLM is designed to handle different parameter names, defaults, and requirements across LLM providers. It provides a unified interface for setting and retrieving configuration values.

## Core Components

### ModelParameter Enum

The `ModelParameter` enum defines standard parameter names that work across all providers:

```python
# From abstractllm/enums.py
from enum import Enum, auto

class ModelParameter(str, Enum):
    """
    Parameters that can be configured for LLM providers.
    
    This enum provides a standardized set of parameter names that can be used
    across different providers. Each provider may support a subset of these parameters.
    """
    
    PROVIDER = "provider"
    MODEL = "model"
    API_KEY = "api_key"
    API_BASE = "api_base"
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    TOP_P = "top_p"
    TOP_K = "top_k"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    STOP = "stop"
    TIMEOUT = "timeout"
    SYSTEM_PROMPT = "system_prompt"
    SEED = "seed"
    METADATA = "metadata"
    STREAM = "stream"
    TOOLS = "tools"
    JSON_MODE = "json_mode"
    # ... other parameters
```

### ConfigurationManager

The `ConfigurationManager` class manages provider configuration:

```python
# From abstractllm/utils/config.py
from typing import Any, Dict, Generic, Optional, TypeVar, Union
import os

T = TypeVar('T')

class ConfigurationManager:
    """
    Parameter management for AbstractLLM providers.
    Handles parameter storage, retrieval, and updates without provider-specific logic.
    """
    
    def __init__(self, initial_config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            initial_config: Initial configuration dictionary
        """
        self._config = {}
        
        # Initialize with empty dict if none provided
        if initial_config is None:
            initial_config = {}
        
        # Update with initial config
        self.update_config(initial_config)
    
    def get_param(self, param: Union[str, ModelParameter], default: Optional[T] = None) -> Optional[T]:
        """
        Get a parameter value from configuration, supporting both enum and string keys.
        
        Args:
            param: Parameter name or enum
            default: Default value if parameter is not set
            
        Returns:
            Parameter value or default
        """
        # Convert enum to string if needed
        param_key = param.value if isinstance(param, ModelParameter) else param
        
        # Check for environment variable override
        env_var = self._get_env_var_name(param_key)
        if env_var in os.environ:
            return os.environ[env_var]
        
        # Return from config or default
        return self._config.get(param_key, default)
    
    def update_config(self, updates: Dict[Union[str, ModelParameter], Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of parameter updates
        """
        # Convert all keys to strings
        for key, value in updates.items():
            # Convert enum to string if needed
            param_key = key.value if isinstance(key, ModelParameter) else key
            self._config[param_key] = value
    
    def _get_env_var_name(self, param_key: str) -> str:
        """
        Get environment variable name for a parameter.
        
        Args:
            param_key: Parameter key
            
        Returns:
            Environment variable name
        """
        # Map parameter keys to environment variable names
        param_to_env = {
            "api_key": self.get_param("provider", "").upper() + "_API_KEY",
            "api_base": self.get_param("provider", "").upper() + "_API_BASE",
            # ... other mappings
        }
        
        return param_to_env.get(param_key, "")
```

## Configuration Layers

AbstractLLM uses a layered approach to configuration, where each layer has a different priority:

1. **Global Defaults**: Baseline values for all providers
2. **Provider-Specific Defaults**: Default values for a specific provider
3. **User-Provided Creation Parameters**: Values specified when creating the provider
4. **Environment Variables**: Values set in environment variables
5. **Per-Method Parameters**: Values provided for a specific method call

This layering allows for flexible configuration management:

```python
# Example of configuration layers
from abstractllm import create_llm

# Layer 3: User-provided creation parameters
llm = create_llm(
    "openai", 
    model="gpt-4",              # User-specified model
    temperature=0.7,            # User-specified temperature
    max_tokens=500              # User-specified max tokens
)

# Layer 5: Per-method parameters
response = llm.generate(
    prompt="Write a short story.",
    temperature=0.9,            # Override temperature for this call
    max_tokens=1000             # Override max tokens for this call
)
```

## Provider Parameter Mapping

Each provider has different parameter names and formats. The configuration system handles the mapping between AbstractLLM's standard parameters and provider-specific parameters:

```python
# From abstractllm/providers/openai.py
def _get_generation_parameters(self, **kwargs):
    """Get generation parameters for OpenAI API."""
    params = {
        "model": self.config_manager.get_param(ModelParameter.MODEL),
        "messages": self._format_messages(),
        "temperature": self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7),
        "max_tokens": self.config_manager.get_param(ModelParameter.MAX_TOKENS, 1000),
        "top_p": self.config_manager.get_param(ModelParameter.TOP_P, 1.0),
        "frequency_penalty": self.config_manager.get_param(ModelParameter.FREQUENCY_PENALTY, 0.0),
        "presence_penalty": self.config_manager.get_param(ModelParameter.PRESENCE_PENALTY, 0.0),
    }
    
    # Add stop if provided
    stop = self.config_manager.get_param(ModelParameter.STOP)
    if stop:
        params["stop"] = stop
    
    # Add stream if provided
    stream = self.config_manager.get_param(ModelParameter.STREAM, False)
    if stream:
        params["stream"] = stream
    
    # Add json mode if provided
    json_mode = self.config_manager.get_param(ModelParameter.JSON_MODE, False)
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    
    # Add tools if provided
    tools = self._prepare_tools(self.config_manager.get_param(ModelParameter.TOOLS))
    if tools:
        params["tools"] = tools
    
    # Update with any other parameters
    params.update(kwargs)
    
    return params
```

## Dynamic Parameter Resolution

The configuration system supports dynamic parameter resolution, where some parameters depend on other parameters:

```python
# Example of dynamic parameter resolution
def _get_api_base(self):
    """Get API base URL based on provider configuration."""
    # Get API base from config
    api_base = self.config_manager.get_param(ModelParameter.API_BASE)
    
    # If not set, use default based on deployment type
    if not api_base:
        deployment_type = self.config_manager.get_param("deployment_type", "cloud")
        
        if deployment_type == "cloud":
            api_base = "https://api.openai.com/v1"
        elif deployment_type == "azure":
            azure_resource = self.config_manager.get_param("azure_resource")
            azure_deployment = self.config_manager.get_param("azure_deployment", "gpt-4")
            
            if not azure_resource:
                raise ConfigurationError("Azure resource name is required for Azure deployments")
                
            api_base = f"https://{azure_resource}.openai.azure.com/deployments/{azure_deployment}"
    
    return api_base
```

## Default Configuration

AbstractLLM provides default configurations for each provider:

```python
# From abstractllm/providers/defaults.py
DEFAULT_CONFIG = {
    "openai": {
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 1000,
        ModelParameter.TOP_P: 1.0,
        ModelParameter.FREQUENCY_PENALTY: 0.0,
        ModelParameter.PRESENCE_PENALTY: 0.0,
        ModelParameter.MODEL: "gpt-4",
    },
    "anthropic": {
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 1000,
        ModelParameter.TOP_P: 1.0,
        ModelParameter.MODEL: "claude-3-opus-20240229",
    },
    "ollama": {
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.TOP_P: 0.9,
        ModelParameter.MODEL: "llama3",
        ModelParameter.API_BASE: "http://localhost:11434",
    },
    "huggingface": {
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 1000,
        ModelParameter.TOP_K: 50,
        ModelParameter.TOP_P: 0.95,
        ModelParameter.MODEL: None,  # Must be specified by user
    }
}
```

## Environment Variables

The configuration system checks for environment variables that can override configuration settings:

| Environment Variable | Description |
|---------------------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI |
| `ANTHROPIC_API_KEY` | API key for Anthropic |
| `HUGGINGFACE_API_KEY` | API key for HuggingFace |
| `OPENAI_API_BASE` | Custom API base URL for OpenAI |
| `ANTHROPIC_API_BASE` | Custom API base URL for Anthropic |
| `OLLAMA_API_BASE` | Custom API base URL for Ollama |

Example usage with environment variables:

```python
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Set API key in environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create LLM without providing API key (uses environment variable)
llm = create_llm("openai", model="gpt-4")
```

## Validation

The configuration system includes validation to ensure that required parameters are provided:

```python
# From abstractllm/providers/openai.py
def _validate_config(self):
    """Validate OpenAI configuration."""
    # Check for API key
    api_key = self.config_manager.get_param(ModelParameter.API_KEY)
    if not api_key:
        # Check environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise AuthenticationError("OpenAI API key is required. "
                                    "Provide it as 'api_key' parameter or set OPENAI_API_KEY environment variable.")
    
    # Check for model
    model = self.config_manager.get_param(ModelParameter.MODEL)
    if not model:
        raise ConfigurationError("Model name is required for OpenAI provider.")
```

## Usage Patterns

### Basic Usage

```python
from abstractllm import create_llm

# Create LLM with configuration
llm = create_llm(
    "openai",
    model="gpt-4",
    temperature=0.7,
    max_tokens=500
)

# Use LLM with request-specific configuration
response = llm.generate(
    prompt="Generate a creative idea.",
    temperature=0.9  # Override temperature for this request
)
```

### Updating Configuration

```python
from abstractllm import create_llm
from abstractllm.enums import ModelParameter

# Create LLM
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Update configuration
llm.config_manager.update_config({
    ModelParameter.TEMPERATURE: 0.5,
    ModelParameter.MAX_TOKENS: 2000,
    ModelParameter.TOP_P: 0.9
})

# Use updated configuration
response = llm.generate("Explain quantum computing.")
```

## Next Steps

- [Tool System](tools.md): How AbstractLLM implements tool calling
- [Error Handling](error-handling.md): How errors are processed and reported
- [User Guide: Configuration](../user-guide/configuration.md): How to use configuration in your applications 