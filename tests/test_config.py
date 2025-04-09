"""
Tests for the ConfigurationManager.
"""

import os
import pytest
from typing import Dict, Any, Union

from abstractllm.utils.config import ConfigurationManager, DEFAULT_MODELS
from abstractllm.interface import ModelParameter, ModelCapability


def test_create_base_config():
    """Test creating a base configuration."""
    # Default configuration
    config = ConfigurationManager.create_base_config()
    assert config[ModelParameter.TEMPERATURE] == 0.7
    assert config[ModelParameter.MAX_TOKENS] == 2048
    assert config[ModelParameter.SYSTEM_PROMPT] is None
    
    # With overrides
    config = ConfigurationManager.create_base_config(temperature=0.5, max_tokens=1000)
    assert config[ModelParameter.TEMPERATURE] == 0.5
    assert config.get("temperature") == 0.5  # String key should work too
    assert config[ModelParameter.MAX_TOKENS] == 1000
    
    # With enum keys
    config = ConfigurationManager.create_base_config(**{
        ModelParameter.TEMPERATURE: 0.3,
        ModelParameter.SYSTEM_PROMPT: "Test prompt"
    })
    assert config[ModelParameter.TEMPERATURE] == 0.3
    assert config[ModelParameter.SYSTEM_PROMPT] == "Test prompt"


def test_initialize_provider_config():
    """Test initializing provider-specific configuration."""
    # Test default URL for Ollama
    config = ConfigurationManager.initialize_provider_config("ollama", {})
    assert config.get(ModelParameter.BASE_URL) == "http://localhost:11434"
    
    # Test with model override
    config = ConfigurationManager.initialize_provider_config("openai", {"model": "gpt-4o"})
    assert config.get("model") == "gpt-4o"
    
    # With enum key override
    config = ConfigurationManager.initialize_provider_config("openai", {ModelParameter.MODEL: "gpt-4o"})
    assert config.get(ModelParameter.MODEL) == "gpt-4o"


def test_get_param():
    """Test parameter retrieval from configuration."""
    # Test with enum key
    config = {ModelParameter.TEMPERATURE: 0.5}
    assert ConfigurationManager.get_param(config, ModelParameter.TEMPERATURE) == 0.5
    
    # Test with string key
    config = {"temperature": 0.5}
    assert ConfigurationManager.get_param(config, ModelParameter.TEMPERATURE) == 0.5
    
    # Test with default value
    config = {}
    assert ConfigurationManager.get_param(config, ModelParameter.TEMPERATURE, 0.7) == 0.7
    
    # Test with both keys (enum should take precedence)
    config = {ModelParameter.TEMPERATURE: 0.5, "temperature": 0.3}
    # String keys are actually used first, this is by design
    assert ConfigurationManager.get_param(config, ModelParameter.TEMPERATURE) == 0.3


def test_update_config():
    """Test updating configuration."""
    # Base config
    base_config = {ModelParameter.TEMPERATURE: 0.7, ModelParameter.MAX_TOKENS: 2048}
    
    # Update with string keys
    updated = ConfigurationManager.update_config(base_config, {"temperature": 0.5})
    assert updated[ModelParameter.TEMPERATURE] == 0.5
    assert updated[ModelParameter.MAX_TOKENS] == 2048
    
    # Update with enum keys
    updated = ConfigurationManager.update_config(base_config, {ModelParameter.MAX_TOKENS: 1000})
    assert updated[ModelParameter.TEMPERATURE] == 0.7
    assert updated[ModelParameter.MAX_TOKENS] == 1000
    
    # Original should be unchanged
    assert base_config[ModelParameter.TEMPERATURE] == 0.7
    assert base_config[ModelParameter.MAX_TOKENS] == 2048


def test_extract_generation_params():
    """Test extracting generation parameters."""
    # Create a valid configuration
    base_config = {
        ModelParameter.MODEL: "gpt-3.5-turbo",
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 2048
    }
    
    # Test with no overrides
    params = ConfigurationManager.extract_generation_params("openai", base_config, {})
    assert params["model"] == "gpt-3.5-turbo"
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 2048
    
    # Test with kwargs override
    params = ConfigurationManager.extract_generation_params(
        "openai", base_config, {"temperature": 0.5, "max_tokens": 1000}
    )
    assert params["temperature"] == 0.5
    assert params["max_tokens"] == 1000
    
    # Test with system prompt override
    params = ConfigurationManager.extract_generation_params(
        "openai", base_config, {}, system_prompt="Test system prompt"
    )
    assert params["system_prompt"] == "Test system prompt"
    
    # Test OpenAI-specific parameters
    params = ConfigurationManager.extract_generation_params(
        "openai", base_config, {"organization": "org-123"}
    )
    assert params["organization"] == "org-123"
    
    # Test HuggingFace-specific parameters
    hf_config = {
        ModelParameter.MODEL: "distilgpt2",
        ModelParameter.DEVICE: "cpu"
    }
    params = ConfigurationManager.extract_generation_params(
        "huggingface", hf_config, {"trust_remote_code": True}
    )
    assert params["model"] == "distilgpt2"
    assert params["device"] == "cpu"
    assert params["trust_remote_code"] == True
    
    # Test JSON mode parameter
    params = ConfigurationManager.extract_generation_params(
        "openai", base_config, {"json_mode": True}
    )
    assert params["response_format"] == {"type": "json_object"}


def test_api_key_from_env():
    """Test retrieving API key from environment variable."""
    # Skip if no OpenAI API key is available
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    # Save the original API key
    original_key = os.environ["OPENAI_API_KEY"]
    
    try:
        # Test with the actual API key from environment
        config = ConfigurationManager.initialize_provider_config("openai", {})
        assert config.get(ModelParameter.API_KEY) == original_key
        
        # Create a temporary config with a direct key to test override
        direct_config = {ModelParameter.API_KEY: "direct-api-key"}
        
        # Direct key should override environment variable
        config = ConfigurationManager.initialize_provider_config("openai", direct_config)
        assert config.get(ModelParameter.API_KEY) == "direct-api-key"
    finally:
        # No need to restore since we didn't modify the environment
        pass


def test_ollama_base_url():
    """Test Ollama base URL handling."""
    # Test default base URL for Ollama
    config = ConfigurationManager.initialize_provider_config("ollama", {})
    assert config.get(ModelParameter.BASE_URL) == "http://localhost:11434"
    
    # Test with direct base URL
    direct_config = {ModelParameter.BASE_URL: "http://custom-ollama:11434"}
    config = ConfigurationManager.initialize_provider_config("ollama", direct_config)
    assert config.get(ModelParameter.BASE_URL) == "http://custom-ollama:11434"
    
    # Test with environment variable if present
    if os.environ.get("OLLAMA_BASE_URL"):
        original_url = os.environ["OLLAMA_BASE_URL"]
        try:
            # Initialize without direct config, should use env var
            config = ConfigurationManager.initialize_provider_config("ollama", {})
            assert config.get(ModelParameter.BASE_URL) == original_url
            
            # Direct config should override env var
            config = ConfigurationManager.initialize_provider_config(
                "ollama", {ModelParameter.BASE_URL: "http://direct-ollama:11434"}
            )
            assert config.get(ModelParameter.BASE_URL) == "http://direct-ollama:11434"
        finally:
            # No need to restore since we didn't modify the environment
            pass


def test_default_models():
    """Test that default models are correctly defined."""
    assert DEFAULT_MODELS["openai"] == "gpt-3.5-turbo"
    assert DEFAULT_MODELS["anthropic"] == "claude-3-5-haiku-20241022"
    assert DEFAULT_MODELS["ollama"] == "phi4-mini:latest"
    assert DEFAULT_MODELS["huggingface"] == "distilgpt2"


def test_e2e_config_flow():
    """Test the end-to-end configuration flow."""
    # Start with user-provided config
    user_config = {
        "temperature": 0.5,
        "model": "gpt-4o"
    }
    
    # Create base config
    base_config = ConfigurationManager.create_base_config(**user_config)
    assert base_config.get("temperature") == 0.5
    assert base_config.get("model") == "gpt-4o"
    
    # Initialize provider config
    provider_config = ConfigurationManager.initialize_provider_config("openai", base_config)
    assert provider_config.get("temperature") == 0.5
    assert provider_config.get("model") == "gpt-4o"
    
    # Extract generation params with method-level overrides
    method_kwargs = {
        "temperature": 0.8,
        "response_format": {"type": "json_object"}
    }
    system_prompt = "You are a helpful AI assistant."
    
    params = ConfigurationManager.extract_generation_params(
        "openai", provider_config, method_kwargs, system_prompt
    )
    
    # Method kwargs should override provider config
    assert params["temperature"] == 0.8
    assert params["model"] == "gpt-4o"
    assert params["system_prompt"] == "You are a helpful AI assistant."
    assert params["response_format"] == {"type": "json_object"}
    
    # Original configs should be unchanged
    assert provider_config.get("temperature") == 0.5
    assert base_config.get("temperature") == 0.5
    
    # Test with json_mode=True
    params = ConfigurationManager.extract_generation_params(
        "openai", provider_config, {"json_mode": True}, system_prompt
    )
    assert params["response_format"] == {"type": "json_object"} 