# AbstractLLM Implementation Guide for LLM-Assisted Development

This guide provides step-by-step instructions for implementing the AbstractLLM package using an LLM-assisted development approach. The purpose is to make the implementation process as straightforward as possible, with clear guidance on each component.

## Implementation Overview

AbstractLLM is organized into the following key components:

1. **Abstract Interface**: Defines the core abstraction for LLM providers
2. **Provider Implementations**: Concrete implementations for each LLM provider
3. **Factory Function**: Creates provider instances based on configuration
4. **Logging Utilities**: Provides standardized logging for all providers

## Step 1: Setting Up the Project Structure

Start by creating the basic project structure:

```
abstractllm/
├── __init__.py
├── interface.py
├── factory.py
├── providers/
│   ├── __init__.py
│   ├── openai.py
│   ├── anthropic.py
│   ├── ollama.py
│   └── huggingface.py
└── utils/
    ├── __init__.py
    └── logging.py
```

## Step 2: Implementing the Abstract Interface

The core of the package is the abstract interface that all providers must implement.

File: `abstractllm/interface.py`

```python
"""
Abstract interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class AbstractLLMInterface(ABC):
    """
    Abstract interface for LLM providers.
    
    All LLM providers must implement this interface to ensure a consistent API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM provider.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self.config = config or {}
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response to the prompt using the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated response as a string
            
        Raises:
            Exception: If the generation fails
        """
        pass
        
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return capabilities of this LLM.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": False,
            "max_tokens": None,
            "supports_system_prompt": False
        }
        
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration.
        
        Args:
            config: New configuration values to merge with existing config
        """
        self.config.update(config)
        
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration as a dictionary
        """
        return self.config.copy()
```

## Step 3: Implementing the Factory Function

The factory function creates LLM provider instances based on the specified provider name.

File: `abstractllm/factory.py`

```python
"""
Factory function for creating LLM provider instances.
"""

from typing import Dict, Any
import importlib
from abstractllm.interface import AbstractLLMInterface


# Provider mapping
_PROVIDERS = {
    "openai": "abstractllm.providers.openai.OpenAIProvider",
    "anthropic": "abstractllm.providers.anthropic.AnthropicProvider",
    "ollama": "abstractllm.providers.ollama.OllamaProvider",
    "huggingface": "abstractllm.providers.huggingface.HuggingFaceProvider",
}


def create_llm(provider: str, **config) -> AbstractLLMInterface:
    """
    Create an LLM provider instance.
    
    Args:
        provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface')
        **config: Provider-specific configuration
        
    Returns:
        An initialized LLM interface
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            f"Available providers: {', '.join(_PROVIDERS.keys())}"
        )
    
    # Import the provider class
    module_path, class_name = _PROVIDERS[provider].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import provider {provider}: {e}")
    
    # Instantiate and return the provider
    return provider_class(config=config)
```

## Step 4: Creating Logging Utilities

Implement logging utilities for standardized logging across providers.

File: `abstractllm/utils/logging.py`

```python
"""
Logging utilities for AbstractLLM.
"""

import logging
from datetime import datetime
from typing import Dict, Any


# Configure logger
logger = logging.getLogger("abstractllm")


def log_request(provider: str, prompt: str, parameters: Dict[str, Any]) -> None:
    """
    Log an LLM request.
    
    Args:
        provider: Provider name
        prompt: The request prompt
        parameters: Request parameters
    """
    logger.debug(f"REQUEST [{provider}]: {datetime.now().isoformat()}")
    logger.debug(f"Parameters: {parameters}")
    logger.debug(f"Prompt: {prompt}")


def log_response(provider: str, response: str) -> None:
    """
    Log an LLM response.
    
    Args:
        provider: Provider name
        response: The response text
    """
    logger.debug(f"RESPONSE [{provider}]: {datetime.now().isoformat()}")
    logger.debug(f"Response: {response}")


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
```

## Step 5: Provider Registry Initialization

Create the provider registry file.

File: `abstractllm/providers/__init__.py`

```python
"""
Provider implementations for AbstractLLM.
"""

# This file intentionally left mostly empty
# Providers are imported in the factory module
```

## Step 6: Implementing the OpenAI Provider

Now implement the OpenAI provider.

File: `abstractllm/providers/openai.py`

```python
"""
OpenAI API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional
import os

from abstractllm.interface import AbstractLLMInterface
from abstractllm.utils.logging import log_request, log_response


class OpenAIProvider(AbstractLLMInterface):
    """
    OpenAI API implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if "api_key" not in self.config:
            self.config["api_key"] = os.environ.get("OPENAI_API_KEY")
        
        if "model" not in self.config:
            self.config["model"] = "gpt-3.5-turbo"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to override configuration
            
        Returns:
            The generated response
            
        Raises:
            Exception: If the API call fails or no API key is provided
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install it with: pip install openai"
            )
        
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Check for API key
        if not params.get("api_key"):
            raise ValueError(
                "OpenAI API key not provided. Pass it as a parameter or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Extract parameters
        api_key = params.pop("api_key")
        model = params.pop("model", "gpt-3.5-turbo")
        temperature = params.pop("temperature", 0.7)
        max_tokens = params.pop("max_tokens", None)
        system_prompt = params.pop("system_prompt", None)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Log the request
        log_request("openai", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None
        })
        
        # Initialize client and call API
        client = OpenAI(api_key=api_key)
        
        completion_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            completion_params["max_tokens"] = max_tokens
        
        response = client.chat.completions.create(**completion_params)
        
        # Extract and log the response
        result = response.choices[0].message.content
        log_response("openai", result)
        
        return result
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return capabilities of the OpenAI provider.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": True,
            "max_tokens": 4096,  # This varies by model
            "supports_system_prompt": True
        }
```

## Step 7: Implementing the Anthropic Provider

Next, implement the Anthropic provider.

File: `abstractllm/providers/anthropic.py`

```python
"""
Anthropic API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional
import os

from abstractllm.interface import AbstractLLMInterface
from abstractllm.utils.logging import log_request, log_response


class AnthropicProvider(AbstractLLMInterface):
    """
    Anthropic API implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if "api_key" not in self.config:
            self.config["api_key"] = os.environ.get("ANTHROPIC_API_KEY")
        
        if "model" not in self.config:
            self.config["model"] = "claude-3-opus-20240229"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Anthropic API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to override configuration
            
        Returns:
            The generated response
            
        Raises:
            Exception: If the API call fails or no API key is provided
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Install it with: pip install anthropic"
            )
        
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Check for API key
        if not params.get("api_key"):
            raise ValueError(
                "Anthropic API key not provided. Pass it as a parameter or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Extract parameters
        api_key = params.pop("api_key")
        model = params.pop("model", "claude-3-opus-20240229")
        temperature = params.pop("temperature", 0.7)
        max_tokens = params.pop("max_tokens", 2048)
        system_prompt = params.pop("system_prompt", None)
        
        # Log the request
        log_request("anthropic", prompt, {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None
        })
        
        # Initialize client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Prepare message
        message_params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        if system_prompt:
            message_params["system"] = system_prompt
        
        # Call API
        response = client.messages.create(**message_params)
        
        # Extract and log the response
        result = response.content[0].text
        log_response("anthropic", result)
        
        return result
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return capabilities of the Anthropic provider.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": True,
            "max_tokens": 100000,  # This varies by model
            "supports_system_prompt": True
        }
```

## Step 8: Implementing the Ollama Provider

Now implement the Ollama provider.

File: `abstractllm/providers/ollama.py`

```python
"""
Ollama API implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional
import os
import requests

from abstractllm.interface import AbstractLLMInterface
from abstractllm.utils.logging import log_request, log_response


class OllamaProvider(AbstractLLMInterface):
    """
    Ollama API implementation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if "base_url" not in self.config:
            self.config["base_url"] = os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
        
        if "model" not in self.config:
            self.config["model"] = "llama2"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Ollama API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to override configuration
            
        Returns:
            The generated response
            
        Raises:
            Exception: If the API call fails
        """
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters
        base_url = params.pop("base_url", "http://localhost:11434")
        model = params.pop("model", "llama2")
        temperature = params.pop("temperature", 0.7)
        system_prompt = params.pop("system_prompt", None)
        
        # Build the request
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "options": params  # Pass any remaining parameters as options
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Log the request
        log_request("ollama", prompt, {
            "model": model,
            "temperature": temperature,
            "base_url": base_url,
            "has_system_prompt": system_prompt is not None
        })
        
        # Make the API request
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=request_data
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API request failed: {e}")
        
        # Extract and log the response
        result = response.json().get("response", "")
        log_response("ollama", result)
        
        return result
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return capabilities of the Ollama provider.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": True,
            "max_tokens": None,  # Varies by model
            "supports_system_prompt": True
        }
```

## Step 9: Implementing the Hugging Face Provider

Next, implement the Hugging Face provider.

File: `abstractllm/providers/huggingface.py`

```python
"""
Hugging Face implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional
import os

from abstractllm.interface import AbstractLLMInterface
from abstractllm.utils.logging import log_request, log_response


class HuggingFaceProvider(AbstractLLMInterface):
    """
    Hugging Face implementation using Transformers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hugging Face provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if "model" not in self.config:
            self.config["model"] = "google/gemma-7b"
        
        self._model = None
        self._tokenizer = None
    
    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer if not already loaded.
        
        Raises:
            ImportError: If required packages are not installed
        """
        if self._model is not None and self._tokenizer is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Required packages not found. Install them with: "
                "pip install torch transformers"
            )
        
        model_name = self.config.get("model", "google/gemma-7b")
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load in 8-bit precision if specified and supported
        load_in_8bit = self.config.get("load_in_8bit", False)
        device_map = self.config.get("device_map", "auto")
        
        if load_in_8bit:
            try:
                import bitsandbytes
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map=device_map
                )
            except ImportError:
                print("Warning: bitsandbytes not installed. Falling back to default precision.")
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map
                )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Hugging Face model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters to override configuration
            
        Returns:
            The generated response
            
        Raises:
            Exception: If model loading or generation fails
        """
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters
        temperature = params.pop("temperature", 0.7)
        max_new_tokens = params.pop("max_new_tokens", 512)
        system_prompt = params.pop("system_prompt", None)
        
        # Log the request
        log_request("huggingface", prompt, {
            "model": params.get("model", "google/gemma-7b"),
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "has_system_prompt": system_prompt is not None
        })
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Prepare the input
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        import torch
        
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                **params  # Pass any remaining parameters
            }
            
            outputs = self._model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode and extract only the new content
        full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = full_output[len(full_prompt):].strip()
        
        # Log the response
        log_response("huggingface", result)
        
        return result
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return capabilities of the Hugging Face provider.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            "streaming": False,
            "max_tokens": None,  # Varies by model and hardware
            "supports_system_prompt": True
        }
```

## Step 9.5: Hugging Face Model Caching and Management

For the Hugging Face provider, implement local model caching and management:

```python
"""
Hugging Face implementation for AbstractLLM with model caching.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from abstractllm.interface import AbstractLLMInterface, ModelCapability, ModelParameter
from abstractllm.utils.logging import log_request, log_response


class HuggingFaceProvider(AbstractLLMInterface):
    """
    Hugging Face implementation using Transformers with local model caching.
    """
    
    # Default cache directory
    DEFAULT_CACHE_DIR = "~/.cache/abstractllm/models"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hugging Face provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if ModelParameter.MODEL not in self.config:
            self.config[ModelParameter.MODEL] = "google/gemma-7b"
        
        # Set up cache directory
        cache_dir = self.config.get("cache_dir", self.DEFAULT_CACHE_DIR)
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None
        self._tokenizer = None
    
    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer from cache if available.
        """
        # Implementation details...
        pass
    
    @staticmethod
    def list_cached_models(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all Hugging Face models currently cached.
        
        Args:
            cache_dir: Custom cache directory path (uses default if None)
            
        Returns:
            List of dictionaries with model information:
            - name: Model name/ID
            - size: Size on disk
            - last_used: Last access timestamp
        """
        cache_path = Path(os.path.expanduser(cache_dir or HuggingFaceProvider.DEFAULT_CACHE_DIR))
        
        if not cache_path.exists():
            return []
            
        models = []
        # Scan the cache directory for models
        # For each model found, gather metadata
        # Return the list of model information
        
        return models
        
    @staticmethod
    def clear_model_cache(model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        """
        Clear specific model or entire cache.
        
        Args:
            model_name: Specific model to remove (clears all if None)
            cache_dir: Custom cache directory path (uses default if None)
        """
        cache_path = Path(os.path.expanduser(cache_dir or HuggingFaceProvider.DEFAULT_CACHE_DIR))
        
        if not cache_path.exists():
            return
            
        if model_name:
            # Delete specific model
            pass
        else:
            # Clear entire cache
            pass
```

In the HuggingFaceProvider implementation:

1. **Cache Location**: Models are cached in `~/.cache/abstractllm/models` by default
2. **Custom Cache**: Users can specify a custom cache location
3. **Listing Models**: Include a static method to list cached models and their details
4. **Cache Management**: Add functionality to clear specific models or the entire cache

This approach ensures:
- Users don't need to download models repeatedly
- Models are stored in a standard location
- Users can inspect and manage cached models
- The API is consistent with Ollama's model listing functionality

Usage examples:

```python
# List all cached models
models = HuggingFaceProvider.list_cached_models()
for model in models:
    print(f"{model['name']} - {model['size']} - Last used: {model['last_used']}")

# Clear specific model cache
HuggingFaceProvider.clear_model_cache("google/gemma-7b")

# Clear entire cache
HuggingFaceProvider.clear_model_cache()

# Use custom cache directory
llm = create_llm("huggingface", 
                model="google/gemma-7b",
                cache_dir="/path/to/custom/cache")
```

## Step 10: Creating the Package Initialization

Finally, create the package initialization file.

File: `abstractllm/__init__.py`

```python
"""
AbstractLLM: A unified interface for large language models.
"""

__version__ = "0.1.0"

from abstractllm.interface import AbstractLLMInterface
from abstractllm.factory import create_llm

__all__ = ["AbstractLLMInterface", "create_llm"]
```

## Step 11: Creating setup.py

Create the setup.py file for packaging.

File: `setup.py`

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="abstractllm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A unified interface for large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/abstractllm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.5.0"],
        "huggingface": ["torch>=1.10.0", "transformers>=4.15.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "torch>=1.10.0",
            "transformers>=4.15.0",
        ],
    },
)
```

## Testing Implementation

After implementing all components, create simple tests to verify functionality.

File: `tests/test_openai.py`

```python
import os
import unittest
from abstractllm import create_llm

class TestOpenAIProvider(unittest.TestCase):
    def test_generate(self):
        # Skip if no API key
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OpenAI API key not set")
        
        llm = create_llm("openai")
        response = llm.generate("Say hello")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == "__main__":
    unittest.main()
```

## LLM-Assisted Implementation Best Practices

When implementing AbstractLLM with the assistance of an LLM, follow these best practices:

1. **Implement one file at a time**: Complete each file before moving to the next
2. **Test incrementally**: Test each component as you build it
3. **Focus on error handling**: Ensure robust error handling in each provider
4. **Maintain consistent style**: Keep naming, docstrings, and code style consistent
5. **Validate imports**: Check that imports are correct and dependencies are properly noted
6. **Review parameter handling**: Verify that configuration parameters are handled consistently

## Common Implementation Pitfalls

Watch out for these common issues when implementing with LLM assistance:

1. **Incomplete error handling**: Ensure all potential errors are caught and handled
2. **Inconsistent parameter names**: Keep parameter names consistent across providers
3. **Missing dependencies**: Check that all required packages are properly listed
4. **API version mismatches**: Verify that the code works with the latest API versions
5. **Tokenizer issues in HuggingFace**: Pay special attention to tokenizer handling
6. **Incomplete logging**: Ensure all important information is logged

By following this implementation guide, you should be able to successfully create the AbstractLLM package with LLM assistance.