"""
Logging utilities for AbstractLLM.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional


# Configure logger
logger = logging.getLogger("abstractllm")


def log_api_key_from_env(provider: str, env_var_name: str) -> None:
    """
    Log a warning when using an API key from an environment variable.
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        env_var_name: The environment variable name
    """
    logger.warning(
        f"Using {provider} API key from environment variable {env_var_name}. "
        "Consider passing it explicitly in configuration for better control."
    )


def log_api_key_missing(provider: str, env_var_name: str) -> None:
    """
    Log an error when an API key is missing.
    
    Args:
        provider: Provider name (e.g., 'openai', 'anthropic')
        env_var_name: The environment variable name
    """
    logger.error(
        f"{provider} API key not provided. Pass it as a parameter in config or "
        f"set the {env_var_name} environment variable."
    )


def log_request(provider: str, prompt: str, parameters: Dict[str, Any]) -> None:
    """
    Log an LLM request.
    
    Args:
        provider: Provider name
        prompt: The request prompt
        parameters: Request parameters
    """
    # Log basic request info at INFO level
    logger.info(f"LLM request to {provider} provider")
    
    # Log detailed request information at DEBUG level
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
    # Log basic response info at INFO level
    logger.info(f"LLM response received from {provider} provider")
    
    # Log detailed response at DEBUG level
    logger.debug(f"RESPONSE [{provider}]: {datetime.now().isoformat()}")
    logger.debug(f"Response: {response}")


def log_request_url(provider: str, url: str, method: str = "POST") -> None:
    """
    Log the URL being requested (for debugging).
    
    Args:
        provider: Provider name
        url: The API endpoint URL (without API key)
        method: The HTTP method used
    """
    logger.debug(f"API Request [{provider}]: {method} {url}")


def setup_logging(level: int = logging.INFO, provider_level: int = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level for root logger (default: INFO)
        provider_level: Logging level for provider loggers, if different from root (default: None)
    """
    # Set up basic configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Get root logger for the package
    package_logger = logging.getLogger("abstractllm")
    package_logger.setLevel(level)
    
    # Configure provider-specific loggers if a different level is specified
    if provider_level is not None:
        # Set level for each provider logger
        providers = ["openai", "anthropic", "ollama", "huggingface"]
        for provider in providers:
            provider_logger = logging.getLogger(f"abstractllm.providers.{provider}")
            provider_logger.setLevel(provider_level)
            
    # Log setup confirmation at DEBUG level
    package_logger.debug(f"Logging configured: root level={logging.getLevelName(level)}, " +
                       f"provider level={logging.getLevelName(provider_level if provider_level is not None else level)}") 