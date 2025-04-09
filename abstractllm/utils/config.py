"""
Configuration management utilities for AbstractLLM.
"""

import os
from typing import Dict, Any, Optional, Union, TypeVar, Generic, List, Set
from enum import Enum
import logging

from abstractllm.interface import ModelParameter

# Configure logger
logger = logging.getLogger("abstractllm.utils.config")

# Define provider-specific default models
DEFAULT_MODELS = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-3-5-haiku-20241022",
    "ollama": "phi4-mini:latest",
    "huggingface": "distilgpt2"
}

# Environment variable mapping for API keys
ENV_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,  # No API key needed for Ollama
    "huggingface": "HUGGINGFACE_API_KEY"
}

# Provider base URLs
BASE_URLS = {
    "openai": None,  # Use default OpenAI API URL
    "anthropic": None,  # Use default Anthropic API URL
    "ollama": "http://localhost:11434",
    "huggingface": None  # No base URL for HuggingFace
}

# Generic value type for configuration
T = TypeVar('T')

class ConfigurationManager:
    """
    Centralized configuration management for AbstractLLM providers.
    """

    @staticmethod
    def create_base_config(**kwargs) -> Dict[Union[str, ModelParameter], Any]:
        """
        Create a configuration dictionary with default values.
        
        Args:
            **kwargs: Configuration parameters to override defaults
            
        Returns:
            Configuration dictionary with defaults and overrides
        """
        # Default configuration
        config = {
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 2048,
            ModelParameter.SYSTEM_PROMPT: None,
            ModelParameter.TOP_P: 1.0,
            ModelParameter.FREQUENCY_PENALTY: 0.0,
            ModelParameter.PRESENCE_PENALTY: 0.0,
            ModelParameter.STOP: None,
            ModelParameter.MODEL: None,
            ModelParameter.API_KEY: None,
            ModelParameter.BASE_URL: None,
            ModelParameter.TIMEOUT: 120,
            ModelParameter.RETRY_COUNT: 3,
            ModelParameter.SEED: None,
            ModelParameter.LOGGING_ENABLED: True,
        }
        # Update with provided values
        config.update(kwargs)
        return config

    @staticmethod
    def initialize_provider_config(
        provider_name: str,
        config: Optional[Dict[Union[str, ModelParameter], Any]] = None
    ) -> Dict[Union[str, ModelParameter], Any]:
        """
        Initialize provider-specific configuration with appropriate defaults.
        
        Args:
            provider_name: Provider name (e.g., "openai", "huggingface")
            config: Optional configuration to start with
            
        Returns:
            Provider configuration dictionary with defaults applied
        """
        # Create base config if none provided
        if config is None:
            config = {}
        
        # Start with a copy to avoid modifying the original
        result = config.copy()
        
        # Get API key from environment variables if not in config
        provider_lower = provider_name.lower()
        
        # Determine the environment variable name based on the provider
        if provider_lower in ENV_API_KEYS and ENV_API_KEYS[provider_lower] is not None:
            env_key = ENV_API_KEYS[provider_lower]
            
            # Log debug information about environment variable 
            logger.debug(f"Looking for API key in environment variable: {env_key}")
            
            # Check if API key is not already in config and available in environment
            api_key_in_env = env_key in os.environ
            if ModelParameter.API_KEY not in result and api_key_in_env:
                result[ModelParameter.API_KEY] = os.environ[env_key]
                logger.debug(f"Found API key in environment for {provider_lower}")
            elif not api_key_in_env:
                logger.debug(f"Environment variable {env_key} not found")
            else:
                logger.debug(f"API key already provided in config for {provider_lower}")
        
        # Provider-specific initialization
        if provider_lower == "openai":
            # Initialize OpenAI-specific configuration
            
            # Set default model if not specified
            if ModelParameter.MODEL not in result:
                result[ModelParameter.MODEL] = DEFAULT_MODELS["openai"]
                
            # Set organization if provided in environment
            if "OPENAI_ORG" in os.environ and ModelParameter.ORGANIZATION not in result:
                result[ModelParameter.ORGANIZATION] = os.environ["OPENAI_ORG"]
                
            # Set base URL if provided in environment
            if "OPENAI_BASE_URL" in os.environ and ModelParameter.BASE_URL not in result:
                result[ModelParameter.BASE_URL] = os.environ["OPENAI_BASE_URL"]
                
        elif provider_lower == "anthropic":
            # Initialize Anthropic-specific configuration
            
            # Set default model if not specified
            if ModelParameter.MODEL not in result:
                result[ModelParameter.MODEL] = DEFAULT_MODELS["anthropic"]
                
            # Set base URL if provided in environment
            if "ANTHROPIC_BASE_URL" in os.environ and ModelParameter.BASE_URL not in result:
                result[ModelParameter.BASE_URL] = os.environ["ANTHROPIC_BASE_URL"]
                
        elif provider_lower == "huggingface":
            # HuggingFace-specific settings
            
            # Set default model if not specified
            if ModelParameter.MODEL not in result:
                result[ModelParameter.MODEL] = DEFAULT_MODELS["huggingface"]
                
            # Set cache directory if provided in environment
            if "HUGGINGFACE_CACHE_DIR" in os.environ and ModelParameter.CACHE_DIR not in result:
                result[ModelParameter.CACHE_DIR] = os.environ["HUGGINGFACE_CACHE_DIR"]
                
        elif provider_lower == "ollama":
            # Ollama-specific settings
            
            # Set default model if not specified
            if ModelParameter.MODEL not in result:
                result[ModelParameter.MODEL] = DEFAULT_MODELS["ollama"]
                
            # Set base URL if provided in environment or use default
            if ModelParameter.BASE_URL not in result:
                result[ModelParameter.BASE_URL] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                
        return result

    @staticmethod
    def get_param(
        config: Dict[Union[str, ModelParameter], Any],
        param: Union[str, ModelParameter],
        default: Optional[T] = None
    ) -> Optional[T]:
        """
        Get a parameter value from configuration, supporting both enum and string keys.
        
        Args:
            config: Configuration dictionary
            param: Parameter to retrieve (either ModelParameter enum or string)
            default: Default value if parameter is not found
            
        Returns:
            Parameter value or default
        """
        # Handle enum parameter
        if isinstance(param, ModelParameter):
            # Enum keys have precedence over string keys
            enum_value = config.get(param)
            if enum_value is not None:
                return enum_value
            return config.get(param.value, default)
        else:
            # It's a string parameter
            return config.get(param, default)

    @staticmethod
    def update_config(
        config: Dict[Union[str, ModelParameter], Any],
        updates: Dict[Union[str, ModelParameter], Any]
    ) -> Dict[Union[str, ModelParameter], Any]:
        """
        Update configuration with new values.
        
        Args:
            config: Original configuration
            updates: Updates to apply
            
        Returns:
            Updated configuration dictionary
        """
        # Create a copy to avoid modifying the input
        config_copy = config.copy()
        config_copy.update(updates)
        return config_copy

    @staticmethod
    def extract_generation_params(
        provider: str,
        config: Dict[Union[str, ModelParameter], Any],
        kwargs: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract generation parameters from config and kwargs.
        This will combine configuration parameters with the kwargs
        provided at generation time.
        
        Args:
            provider: Provider name
            config: Configuration dictionary
            kwargs: Additional arguments for generation
            system_prompt: Optional system prompt
            
        Returns:
            Combined parameters for generation
        """
        params = {}
        
        # Copy all of configuration params
        for k, v in config.items():
            # Convert enum keys to string
            key = k.value if isinstance(k, ModelParameter) else k
            params[key] = v
            
        # Override with kwargs (which take precedence)
        for k, v in kwargs.items():
            params[k] = v
            
        # Add system prompt if provided
        if system_prompt:
            params["system_prompt"] = system_prompt
        
        # Ensure all providers have consistent parameters 
        # (even if they're named differently per API)
        if provider.lower() == "openai":
            # OpenAI-specific param naming
            # max_tokens -> max_tokens (already correct)
            # temperature -> temperature (already correct)
            # top_p -> top_p (already correct)
            
            # Add response_format if json mode is set
            json_mode = kwargs.get("json_mode", False)
            if json_mode:
                params["response_format"] = {"type": "json_object"}
                
        elif provider.lower() == "anthropic":
            # Anthropic-specific param naming
            params["max_tokens"] = ConfigurationManager.get_param(config, ModelParameter.MAX_TOKENS, 1000)
            # temperature -> temperature (already correct)
            # top_p -> top_p (already correct)
            
        elif provider.lower() == "huggingface":
            # HuggingFace-specific param naming
            # max_tokens -> max_new_tokens
            max_tokens = ConfigurationManager.get_param(config, ModelParameter.MAX_TOKENS, 1000)
            params["max_new_tokens"] = max_tokens
            
            # temperature -> temperature (already correct)
            # top_p -> top_p (already correct)
            
            # Add typical repetition penalty if set
            repetition_penalty = ConfigurationManager.get_param(config, ModelParameter.REPETITION_PENALTY, None)
            if repetition_penalty is not None:
                params["repetition_penalty"] = repetition_penalty
        
        # Ensure response_format is included when json_mode is enabled
        if kwargs.get("json_mode", False):
            if provider.lower() == "openai":
                params["response_format"] = {"type": "json_object"}
                
        return params


# Create a function alias for backward compatibility
def create_config(**kwargs) -> Dict[Union[str, ModelParameter], Any]:
    """
    Create a configuration dictionary with default values (for backward compatibility).
    
    Args:
        **kwargs: Configuration parameters to override defaults
        
    Returns:
        Configuration dictionary
    """
    return ConfigurationManager.create_base_config(**kwargs) 