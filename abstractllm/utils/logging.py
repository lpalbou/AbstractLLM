"""
Logging utilities for AbstractLLM.
"""

import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Union, List, Optional
from pathlib import Path


# Configure logger
logger = logging.getLogger("abstractllm")

# Global configuration
class LogConfig:
    """Global logging configuration."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogConfig, cls).__new__(cls)
            # Default configuration
            cls._instance._log_dir = os.getenv("ABSTRACTLLM_LOG_DIR")
            cls._instance._log_level = logging.INFO
            cls._instance._provider_level = None
            cls._instance._console_output = None
            cls._instance._initialized = False
        return cls._instance
    
    @property
    def log_dir(self) -> Optional[str]:
        """Get the current log directory."""
        return self._log_dir
    
    @log_dir.setter
    def log_dir(self, value: Optional[str]) -> None:
        """Set the log directory."""
        self._log_dir = value
        if value:
            os.makedirs(value, exist_ok=True)
            logger.info(f"Log directory set to: {value}")
    
    @property
    def log_level(self) -> int:
        """Get the current log level."""
        return self._log_level
    
    @log_level.setter
    def log_level(self, value: int) -> None:
        """Set the log level."""
        self._log_level = value
        if self._initialized:
            logger.setLevel(value)
    
    @property
    def provider_level(self) -> Optional[int]:
        """Get the provider-specific log level."""
        return self._provider_level
    
    @provider_level.setter
    def provider_level(self, value: Optional[int]) -> None:
        """Set the provider-specific log level."""
        self._provider_level = value
        if self._initialized:
            logging.getLogger("abstractllm.providers").setLevel(value or self._log_level)
    
    @property
    def console_output(self) -> Optional[bool]:
        """Get the console output setting."""
        return self._console_output
    
    @console_output.setter
    def console_output(self, value: Optional[bool]) -> None:
        """Set the console output setting."""
        self._console_output = value
    
    def initialize(self) -> None:
        """Initialize logging with current configuration."""
        if not self._initialized:
            setup_logging(
                level=self._log_level,
                provider_level=self._provider_level,
                log_dir=self._log_dir,
                console_output=self._console_output
            )
            self._initialized = True

# Global configuration instance
config = LogConfig()

def configure_logging(
    log_dir: Optional[str] = None,
    log_level: Optional[int] = None,
    provider_level: Optional[int] = None,
    console_output: Optional[bool] = None
) -> None:
    """
    Configure global logging settings for AbstractLLM.
    
    This is the main function that external programs should use to configure logging.
    
    Args:
        log_dir: Directory to store log files (default: ABSTRACTLLM_LOG_DIR env var)
                If not set, no file logging occurs unless forced
        log_level: Default logging level for all loggers (default: INFO)
        provider_level: Specific level for provider loggers (default: same as log_level)
        console_output: Control console output:
            - None (default): automatic (console if no log_dir, no console if log_dir)
            - True: Force console output regardless of log_dir
            - False: Force no console output regardless of log_dir
    
    Example:
        >>> from abstractllm import configure_logging
        >>> import logging
        >>> 
        >>> # Development: Everything to console
        >>> configure_logging(log_level=logging.DEBUG)
        >>> 
        >>> # Production: Everything to files
        >>> configure_logging(
        ...     log_dir="/var/log/abstractllm",
        ...     log_level=logging.INFO
        ... )
        >>> 
        >>> # Both: Console and file output
        >>> configure_logging(
        ...     log_dir="/var/log/abstractllm",
        ...     console_output=True
        ... )
    """
    if log_dir is not None:
        config.log_dir = log_dir
    if log_level is not None:
        config.log_level = log_level
    if provider_level is not None:
        config.provider_level = provider_level
    if console_output is not None:
        config.console_output = console_output
    
    # Initialize or reinitialize logging
    config.initialize()

def truncate_base64(data: Any, max_length: int = 50) -> Any:
    """
    Truncate base64 strings for logging to avoid excessive output.
    
    Args:
        data: Data to truncate (can be a string, dict, list, or other structure)
        max_length: Maximum length of base64 strings before truncation
        
    Returns:
        Truncated data in the same structure as input
    """
    if isinstance(data, str) and len(data) > max_length:
        # For strings, check if they're likely base64 encoded (no spaces, mostly alphanumeric)
        if all(c.isalnum() or c in '+/=' for c in data) and ' ' not in data:
            # Instead of showing part of the base64 data, just show a placeholder
            return f"[base64 data, length: {len(data)} chars]"
        return data
    
    if isinstance(data, dict):
        # For dicts, truncate each value that looks like base64
        return {k: truncate_base64(v, max_length) for k, v in data.items()}
    
    if isinstance(data, list):
        # For lists, truncate each item that looks like base64
        return [truncate_base64(item, max_length) for item in data]
    
    return data


def ensure_log_directory(log_dir: Optional[str] = None) -> Optional[str]:
    """
    Ensure log directory exists and return the path.
    
    Args:
        log_dir: Directory to store log files (default: use global config)
        
    Returns:
        Path to the log directory or None if no directory is configured
    """
    directory = log_dir or config.log_dir
    if directory:
        os.makedirs(directory, exist_ok=True)
        return directory
    return None


def get_log_filename(provider: str, log_type: str, log_dir: Optional[str] = None) -> Optional[str]:
    """
    Generate a filename for a log file.
    
    Args:
        provider: Provider name
        log_type: Type of log (e.g., 'request', 'response')
        log_dir: Directory to store log files (default: use global config)
        
    Returns:
        Full path to the log file or None if no directory is configured
    """
    directory = ensure_log_directory(log_dir)
    if not directory:
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(directory, f"{provider}_{log_type}_{timestamp}.json")


def write_to_log_file(data: Dict[str, Any], filename: Optional[str]) -> None:
    """
    Write data to a log file in JSON format.
    
    Args:
        data: Data to write
        filename: Path to log file (if None, no file is written)
    """
    if not filename:
        return
        
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"Log written to: {filename}")
    except Exception as e:
        logger.warning(f"Failed to write log file: {e}")


def log_api_key_from_env(provider: str, env_var_name: str) -> None:
    """
    Log that an API key was loaded from an environment variable.
    
    Args:
        provider: Provider name
        env_var_name: Environment variable name
    """
    logger.debug(f"Using {provider} API key from environment variable {env_var_name}")


def log_api_key_missing(provider: str, env_var_name: str) -> None:
    """
    Log that an API key is missing from the environment.
    
    Args:
        provider: Provider name
        env_var_name: Environment variable name
    """
    logger.warning(f"{provider} API key not found in environment variable {env_var_name}")


def log_request(provider: str, prompt: str, parameters: Dict[str, Any], log_dir: Optional[str] = None) -> None:
    """
    Log an LLM request.
    
    Args:
        provider: Provider name
        prompt: The request prompt
        parameters: Request parameters
        log_dir: Optional override for log directory
    """
    timestamp = datetime.now().isoformat()
    
    # Create a safe copy of parameters for logging
    safe_parameters = parameters.copy()
    
    # Special handling for images parameter (in any provider)
    if "images" in safe_parameters:
        if isinstance(safe_parameters["images"], list):
            num_images = len(safe_parameters["images"])
            safe_parameters["images"] = f"[{num_images} image(s), data hidden]"
        else:
            safe_parameters["images"] = "[image data hidden]"
    
    # Check for image in parameters (in any provider)
    if "image" in safe_parameters:
        if isinstance(safe_parameters["image"], str):
            safe_parameters["image"] = "[image data hidden]"
        elif isinstance(safe_parameters["image"], dict):
            # For nested image formats like OpenAI's or Anthropic's
            if "data" in safe_parameters["image"]:
                safe_parameters["image"]["data"] = "[data hidden]"
            elif "image_url" in safe_parameters["image"]:
                if "url" in safe_parameters["image"]["image_url"] and (
                    safe_parameters["image"]["image_url"]["url"].startswith("data:")
                ):
                    safe_parameters["image"]["image_url"]["url"] = "[base64 data URL hidden]"
            elif "source" in safe_parameters["image"] and "data" in safe_parameters["image"]["source"]:
                safe_parameters["image"]["source"]["data"] = "[data hidden]"
    
    # Now apply general base64 truncation on any remaining fields
    safe_parameters = truncate_base64(safe_parameters)
    
    # Log to console if enabled
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.debug(f"REQUEST [{provider}]: {timestamp}")
        logger.debug(f"Parameters: {safe_parameters}")
        logger.debug(f"Prompt: {prompt}")
    
    # Write to file if directory is configured
    log_filename = get_log_filename(provider, "request", log_dir)
    if log_filename:
        log_data = {
            "timestamp": timestamp,
            "provider": provider,
            "prompt": prompt,
            "parameters": parameters  # Original, non-truncated parameters
        }
        write_to_log_file(log_data, log_filename)


def log_response(provider: str, response: str, log_dir: Optional[str] = None) -> None:
    """
    Log an LLM response.
    
    Args:
        provider: Provider name
        response: The response text
        log_dir: Optional override for log directory
    """
    timestamp = datetime.now().isoformat()
    
    # Log to console if enabled
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.debug(f"RESPONSE [{provider}]: {timestamp}")
        if len(response) > 10000:
            truncated_response = response[:10000] + f"... [truncated, total length: {len(response)} chars]"
            logger.debug(f"Response: {truncated_response}")
        else:
            logger.debug(f"Response: {response}")
    
    # Write to file if directory is configured
    log_filename = get_log_filename(provider, "response", log_dir)
    if log_filename:
        log_data = {
            "timestamp": timestamp,
            "provider": provider,
            "response": response  # Original, full response
        }
        write_to_log_file(log_data, log_filename)


def log_request_url(provider: str, url: str, method: str = "POST") -> None:
    """
    Log the URL for an API request.
    
    Args:
        provider: Provider name
        url: The request URL
        method: HTTP method (default: POST)
    """
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.debug(f"API Request [{provider}]: {method} {url}")


def setup_logging(
    level: int = logging.INFO,
    provider_level: Optional[int] = None,
    log_dir: Optional[str] = None,
    console_output: Optional[bool] = None
) -> None:
    """
    Set up logging configuration for AbstractLLM.
    
    Args:
        level: Default logging level for all loggers (default: INFO)
        provider_level: Specific level for provider loggers (default: same as level)
        log_dir: Directory to store log files (default: None)
        console_output: Whether to output to console (default: automatic based on log_dir)
    """
    # Use the same level for providers if not specified
    if provider_level is None:
        provider_level = level
    
    # Set up base logger
    logger.setLevel(level)
    
    # Set up provider-specific loggers
    logging.getLogger("abstractllm.providers").setLevel(provider_level)
    
    # Remove all existing handlers
    logger.handlers.clear()
    
    # Determine if we should output to console
    should_console = True if console_output is True else (
        False if console_output is False else (log_dir is None)
    )
    
    # Create console handler if needed
    if should_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to the logger
        logger.addHandler(console_handler)
    
    # Create file handler for detailed logging if we have a directory
    if log_dir:
        try:
            # Ensure log directory exists
            directory = ensure_log_directory(log_dir)
            
            # Create a file handler for detailed logs
            log_file = os.path.join(directory, f"abstractllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(min(level, logging.DEBUG))  # Always capture at least DEBUG level in files
            
            # Create formatter with more details for file logs
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # Add file handler to the logger
            logger.addHandler(file_handler)
            
            logger.info(f"Detailed logs will be written to: {log_file}")
            logger.info(f"Request and response payloads will be stored in: {directory}")
            
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}") 