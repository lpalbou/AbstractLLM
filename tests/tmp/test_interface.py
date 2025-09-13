"""
Tests for the AbstractLLM interface.
"""

import pytest
from typing import Dict, Any, List, Union

from abstractllm import AbstractLLMInterface, ModelParameter, ModelCapability
from tests.utils import validate_response, validate_not_contains, has_capability


def test_interface_initialization(any_provider: AbstractLLMInterface) -> None:
    """
    Test that each provider initializes properly and follows the interface.
    
    Args:
        any_provider: Provider instance to test
    """
    # Should be an instance of AbstractLLMInterface
    assert isinstance(any_provider, AbstractLLMInterface)
    
    # Should have a config attribute
    assert hasattr(any_provider, "config")
    assert isinstance(any_provider.config, dict)
    
    # Should have the required methods
    assert callable(any_provider.generate)
    assert callable(any_provider.generate_async)
    assert callable(any_provider.get_capabilities)
    assert callable(any_provider.set_config)
    assert callable(any_provider.update_config)
    assert callable(any_provider.get_config)


def test_get_capabilities(any_provider: AbstractLLMInterface) -> None:
    """
    Test that get_capabilities returns a valid dictionary.
    
    Args:
        any_provider: Provider instance to test
    """
    capabilities = any_provider.get_capabilities()
    
    # Should return a dictionary
    assert isinstance(capabilities, dict)
    
    # Required capabilities should be present
    required_capabilities = [
        ModelCapability.STREAMING,
        ModelCapability.MAX_TOKENS,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.ASYNC,
    ]
    
    for capability in required_capabilities:
        assert capability in capabilities


def test_config_management(any_provider: AbstractLLMInterface) -> None:
    """
    Test configuration management methods.
    
    Args:
        any_provider: Provider instance to test
    """
    # Get initial config
    initial_config = any_provider.get_config()
    
    # Should return a copy (not affect the instance when modified)
    config_copy = any_provider.get_config()
    config_copy["test_key"] = "test_value"
    assert "test_key" not in any_provider.get_config()
    
    # Set config with kwargs
    any_provider.set_config(test_key="test_value")
    assert any_provider.get_config().get("test_key") == "test_value"
    
    # Update config with dict
    any_provider.update_config({ModelParameter.TEMPERATURE: 0.5})
    assert any_provider.get_config().get(ModelParameter.TEMPERATURE) == 0.5
    
    # Restore original config
    any_provider.update_config(initial_config)
    
    # Config should match initial state (excluding the test_key we added)
    current_config = any_provider.get_config()
    for key, value in initial_config.items():
        assert current_config.get(key) == value 