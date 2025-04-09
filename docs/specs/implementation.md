# AbstractLLM Implementation Guide

This guide provides detailed implementation instructions for developing the AbstractLLM package.

## Core Files Implementation

### `abstractllm/__init__.py`

```python
"""
AbstractLLM: A unified interface for large language models.
"""

__version__ = "0.1.0"

from abstractllm.interface import AbstractLLMInterface
from abstractllm.factory import create_llm

__all__ = ["AbstractLLMInterface", "create_llm"]
```

### `abstractllm/interface.py`

```python
"""
Abstract interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class AbstractLLMInterface(ABC):
    """
    Abstract interface for LLM providers.
    
    All LLM providers must implement this interface.
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

### `abstractllm/factory.py`

```python
"""
Factory function for creating LLM provider instances.
"""

from typing import Dict, Any, Optional, Type
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

### `abstractllm/utils/logging.py`

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

## Provider Implementations

### `abstractllm/providers/__init__.py`

```python
"""
Provider implementations for AbstractLLM.
"""

# This file intentionally left mostly empty
# Providers are imported in the factory module
```

### `abstractllm/providers/openai.py`

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

### `abstractllm/providers/anthropic.py`

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

### `abstractllm/providers/ollama.py`

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

### `abstractllm/providers/huggingface.py`

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

## Error Handling

Each provider implementation should include proper error handling:

1. **Missing dependencies**: Check for required packages and provide helpful error messages
2. **API errors**: Handle and provide meaningful error messages
3. **Configuration issues**: Validate configuration and provide clear errors

## Testing

Create simple test files for each provider to verify functionality:

```python
# Example test for OpenAI provider
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

## Packaging

Include these files for PyPI packaging:

1. `setup.py`
2. `README.md`
3. `LICENSE`
4. `requirements.txt`

Example `setup.py`:

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