"""
Abstract interface for LLM providers.
"""

from enum import Enum
from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator
from abc import ABC, abstractmethod


class ModelParameter(str, Enum):
    """Model parameters that can be configured."""
    # Basic parameters
    TEMPERATURE = "temperature"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "system_prompt"
    TOP_P = "top_p"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    STOP = "stop"
    MODEL = "model"  # Model identifier/name
    API_KEY = "api_key"  # API key for providers that need it
    BASE_URL = "base_url"  # Base URL for local/self-hosted models
    
    # Additional parameters
    TIMEOUT = "timeout"  # Request timeout in seconds
    RETRY_COUNT = "retry_count"  # Number of retries on failure
    LOGIT_BIAS = "logit_bias"  # Token biases for generation
    SEED = "seed"  # Random seed for reproducible generations
    TOP_K = "top_k"  # Top-k sampling parameter
    REPETITION_PENALTY = "repetition_penalty"  # Penalty for repeating tokens
    
    # Model loading parameters (for local models)
    DEVICE = "device"  # Device to load the model on (cpu, cuda, etc.)
    DEVICE_MAP = "device_map"  # Device mapping for model sharding
    LOAD_IN_8BIT = "load_in_8bit"  # Whether to load in 8-bit precision
    LOAD_IN_4BIT = "load_in_4bit"  # Whether to load in 4-bit precision
    CACHE_DIR = "cache_dir"  # Directory for model caching
    
    # Provider-specific parameters
    ORGANIZATION = "organization"  # Organization ID for OpenAI
    USER = "user"  # User ID for attribution/tracking
    PROXY = "proxy"  # Proxy URL for API requests
    REQUEST_TIMEOUT = "request_timeout"  # Timeout specifically for HTTP requests
    MAX_RETRIES = "max_retries"  # Maximum number of retry attempts
    
    # Security & compliance parameters
    CONTENT_FILTER = "content_filter"  # Content filtering level
    MODERATION = "moderation"  # Whether to perform moderation
    LOGGING_ENABLED = "logging_enabled"  # Whether to log requests/responses


class ModelCapability(str, Enum):
    """Capabilities that a model may support."""
    # Basic capabilities
    STREAMING = "streaming"
    MAX_TOKENS = "max_tokens"
    SYSTEM_PROMPT = "supports_system_prompt"
    ASYNC = "supports_async"
    FUNCTION_CALLING = "supports_function_calling"
    VISION = "supports_vision"
    
    # Advanced capabilities
    FINE_TUNING = "supports_fine_tuning"
    EMBEDDINGS = "supports_embeddings"
    MULTILINGUAL = "supports_multilingual"
    RAG = "supports_rag"  # Retrieval Augmented Generation
    MULTI_TURN = "supports_multi_turn"  # Multi-turn conversations
    PARALLEL_INFERENCE = "supports_parallel_inference"
    IMAGE_GENERATION = "supports_image_generation"
    AUDIO_PROCESSING = "supports_audio_processing"
    JSON_MODE = "supports_json_mode"  # Structured JSON output


def create_config(**kwargs) -> Dict[str, Any]:
    """
    Create a configuration dictionary with default values.
    
    Args:
        **kwargs: Configuration parameters to override defaults
        
    Returns:
        Configuration dictionary
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


class AbstractLLMInterface(ABC):
    """
    Abstract interface for LLM providers.
    
    All LLM providers must implement this interface to ensure a consistent API.
    This interface defines the common methods that all providers must support,
    regardless of their underlying implementation details.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the LLM provider.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self.config = config or create_config()
    
    @abstractmethod
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response to the prompt using the LLM.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response (default: False)
            **kwargs: Additional parameters to override config
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: A generator yielding response chunks
            
        Raises:
            Exception: If the generation fails
        """
        pass

    @abstractmethod
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Asynchronously generate a response to the prompt using the LLM.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response (default: False)
            **kwargs: Additional parameters to override config
            
        Returns:
            If stream=False: The complete generated response as a string
            If stream=True: An async generator yielding response chunks
            
        Raises:
            Exception: If the generation fails
        """
        pass
        
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of this LLM.
        
        Returns:
            Dictionary of capabilities including:
            - streaming: bool - Whether streaming is supported
            - max_tokens: Optional[int] - Maximum tokens supported
            - supports_system_prompt: bool - Whether system prompts are supported
            - supports_async: bool - Whether async generation is supported
            - supports_function_calling: bool - Whether function calling is supported
            - supports_vision: bool - Whether vision capabilities are supported
        """
        return {
            ModelCapability.STREAMING: False,
            ModelCapability.MAX_TOKENS: 2048,  # Default value, should be overridden by providers
            ModelCapability.SYSTEM_PROMPT: False,
            ModelCapability.ASYNC: False,
            ModelCapability.FUNCTION_CALLING: False,
            ModelCapability.VISION: False,
        }
        
    def set_config(self, **kwargs) -> None:
        """
        Update the configuration with individual parameters.
        
        Args:
            **kwargs: Configuration values to update
        """
        self.config.update(kwargs)
        
    def update_config(self, config: Dict[Union[str, ModelParameter], Any]) -> None:
        """
        Update the configuration with a dictionary of parameters.
        
        Args:
            config: Dictionary of configuration values to update
        """
        self.config.update(config)
        
    def get_config(self) -> Dict[Union[str, ModelParameter], Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration as a dictionary
        """
        return self.config.copy() 