"""
Logging utilities for AbstractLLM.
"""

import logging
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, Any, Union, List, Optional
from pathlib import Path

# Import color codes from formatting module
from .formatting import RED_BOLD, GREY_ITALIC, BLUE_ITALIC, RESET

# Configure logger
logger = logging.getLogger("abstractllm")

# Storage for pending requests to match with responses
_pending_requests = {}

# Immediately suppress noisy third-party loggers and warnings at import time
# This catches warnings that happen before configure_logging() is called
# Allow INFO level for progress bars, but set specific loggers higher
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)  # Allow progress, suppress deprecation warnings
logging.getLogger("huggingface_hub._snapshot_download").setLevel(logging.INFO)  # Allow progress info
logging.getLogger("transformers").setLevel(logging.ERROR)  # Still suppress transformers noise

# Suppress specific HuggingFace Hub warnings that come from MLX libraries
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    module="huggingface_hub.file_download",
    message=".*resume_download.*deprecated.*"
)

# Also suppress any HuggingFace warnings about special tokens
warnings.filterwarnings(
    "ignore",
    message=".*Special tokens have been added in the vocabulary.*"
)

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
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    console_output: Optional[bool] = None
) -> None:
    """
    Configure global logging settings for AbstractLLM with independent console and file control.
    
    This is the main function that external programs should use to configure logging.
    
    Args:
        log_dir: Directory to store log files (default: ABSTRACTLLM_LOG_DIR env var)
                If not set, no file logging occurs
        console_level: Logging level for console output (default: WARNING)
                     Set to None to disable console logging entirely
        file_level: Logging level for file output (default: DEBUG)
                   Only used if log_dir is provided
        console_output: DEPRECATED - use console_level instead
                       If provided, overrides console_level behavior for backward compatibility
    
    Recommended Usage:
        >>> # Development: Detailed console + file logging
        >>> configure_logging(
        ...     log_dir="logs",
        ...     console_level=logging.DEBUG,
        ...     file_level=logging.DEBUG
        ... )
        >>> 
        >>> # Production: Warnings to console, everything to file
        >>> configure_logging(
        ...     log_dir="/var/log/abstractllm",
        ...     console_level=logging.WARNING,  # Default
        ...     file_level=logging.DEBUG        # Default
        ... )
        >>>
        >>> # Silent mode: Only file logging
        >>> configure_logging(
        ...     log_dir="logs",
        ...     console_level=None  # Disable console
        ... )
    """
    # Set sensible defaults
    if console_level is None and console_output is None:
        console_level = logging.WARNING  # Only warnings and errors to console by default
    
    if file_level is None:
        file_level = logging.DEBUG  # Everything to file by default
    
    # Handle backward compatibility with console_output parameter
    if console_output is not None:
        if console_output is False:
            console_level = None  # Disable console logging
        elif console_output is True and console_level is None:
            console_level = logging.INFO  # Enable console with reasonable level
    
    # Update global config
    if log_dir is not None:
        config.log_dir = log_dir
    if console_level is not None:
        config.log_level = console_level  # Store console level as main level
    
    # Initialize logging with new parameters
    setup_logging(
        console_level=console_level,
        file_level=file_level,
        log_dir=log_dir or config.log_dir
    )

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


def get_log_filename(provider: str, log_type: str, log_dir: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    """
    Generate a filename for a log file.
    
    Args:
        provider: Provider name
        log_type: Type of log (e.g., 'request', 'response', 'interaction')
        log_dir: Directory to store log files (default: use global config)
        model: Model name to include in filename (optional)
        
    Returns:
        Full path to the log file or None if no directory is configured
    """
    directory = ensure_log_directory(log_dir)
    if not directory:
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Include model name in filename if provided
    if model:
        # Clean model name for filename (replace / with _)
        clean_model = model.replace("/", "_").replace(":", "_")
        return os.path.join(directory, f"{provider}_{clean_model}_{log_type}_{timestamp}.json")
    else:
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


def log_request(provider: str, prompt: str, parameters: Dict[str, Any], log_dir: Optional[str] = None, model: Optional[str] = None) -> None:
    """
    Log an LLM request.
    
    Args:
        provider: Provider name
        prompt: The request prompt
        parameters: Request parameters
        log_dir: Optional override for log directory
        model: Model name (extracted from parameters if not provided)
    """
    timestamp = datetime.now().isoformat()
    
    # Extract model from parameters if not provided
    if not model:
        model = parameters.get("model") or parameters.get("model_name")
    
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
    
    # Store request for later matching with response
    request_id = f"{provider}_{timestamp}_{id(prompt)}"
    _pending_requests[request_id] = {
        "timestamp": timestamp,
        "provider": provider,
        "prompt": prompt,
        "parameters": parameters,  # Original, non-truncated parameters
        "model": model,
        "log_dir": log_dir
    }


def log_response(provider: str, response: str, log_dir: Optional[str] = None, model: Optional[str] = None, **kwargs) -> None:
    """
    Log an LLM response and combine it with the matching request.
    
    Args:
        provider: Provider name
        response: The response text
        log_dir: Optional override for log directory  
        model: Model name (used to find matching request)
        **kwargs: Additional metadata to include in the response log
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
    
    # Find matching request (most recent one for this provider)
    matching_request = None
    request_key_to_remove = None
    
    for key, request_data in _pending_requests.items():
        if request_data["provider"] == provider:
            # If model is specified, match on model too
            if model and request_data.get("model") != model:
                continue
            matching_request = request_data
            request_key_to_remove = key
            break
    
    if matching_request:
        # Remove the request from pending
        del _pending_requests[request_key_to_remove]
        
        # Create combined interaction file
        log_filename = get_log_filename(provider, "interaction", log_dir or matching_request["log_dir"], model or matching_request.get("model"))
        if log_filename:
            # Build response data with additional metadata
            response_data = {
                "timestamp": timestamp,
                "provider": provider,
                "response": response
            }
            
            # Add any additional metadata from kwargs
            if kwargs.get("has_tool_calls"):
                response_data["has_tool_calls"] = kwargs["has_tool_calls"]
            if kwargs.get("tool_calls"):
                response_data["tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in kwargs["tool_calls"]]
            if kwargs.get("usage"):
                response_data["usage"] = kwargs["usage"]
                
            combined_data = {
                "request": {
                    "timestamp": matching_request["timestamp"],
                    "provider": matching_request["provider"],
                    "prompt": matching_request["prompt"],
                    "parameters": matching_request["parameters"]
                },
                "response": response_data
            }
            write_to_log_file(combined_data, log_filename)
    else:
        # Fallback: write response-only file if no matching request found
        log_filename = get_log_filename(provider, "response", log_dir, model)
        if log_filename:
            log_data = {
                "timestamp": timestamp,
                "provider": provider,
                "response": response
            }
            # Add any additional metadata
            for key, value in kwargs.items():
                if key not in ["log_dir", "model"]:
                    log_data[key] = value
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


def suppress_third_party_warnings() -> None:
    """
    Suppress noisy warnings from third-party libraries.
    
    This function specifically targets warnings that commonly appear when using
    MLX with HuggingFace models, such as:
    - HuggingFace Hub deprecation warnings about resume_download
    - Special token vocabulary warnings from transformers
    - Other common HuggingFace infrastructure warnings
    
    This function is called automatically during logging setup but can also
    be called independently if needed.
    """
    # Suppress HuggingFace Hub deprecation warnings
    warnings.filterwarnings(
        "ignore", 
        category=FutureWarning, 
        module="huggingface_hub.file_download",
        message=".*resume_download.*deprecated.*"
    )
    
    # Suppress special token warnings that appear during model loading
    warnings.filterwarnings(
        "ignore",
        message=".*Special tokens have been added in the vocabulary.*"
    )
    
    # Suppress other common HuggingFace warnings  
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="huggingface_hub.*"
    )
    
    # Suppress transformers warnings about deprecated features
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="transformers.*"
    )


def setup_logging(
    console_level: Optional[int] = logging.WARNING,
    file_level: Optional[int] = logging.DEBUG,
    log_dir: Optional[str] = None
) -> None:
    """
    Set up logging configuration for AbstractLLM with independent console and file control.
    
    Args:
        console_level: Logging level for console output (None to disable console)
        file_level: Logging level for file output (default: DEBUG)
        log_dir: Directory to store log files (None to disable file logging)
    """
    # Set up base logger to capture the most verbose level needed
    base_level = logging.DEBUG  # Always capture everything at the root level
    if console_level is not None and file_level is not None:
        base_level = min(console_level, file_level)
    elif console_level is not None:
        base_level = console_level
    elif file_level is not None:
        base_level = file_level
    
    logger.setLevel(base_level)
    
    # Configure the root logger to handle all messages
    root_logger = logging.getLogger()
    root_logger.setLevel(base_level)
    
    # Set up provider-specific loggers to use the same base level
    logging.getLogger("abstractllm.providers").setLevel(base_level)
    
    # Remove all existing handlers to start fresh
    root_logger.handlers.clear()
    logger.handlers.clear()
    
    # Create console handler if console logging is enabled
    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        
        # Create colored truncating formatter for console
        console_formatter = ColoredTruncatingFormatter(
            max_message_length=1000,
            fmt='%(levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Add handler to the root logger so all loggers inherit it
        root_logger.addHandler(console_handler)
        logger.info(f"Console logging enabled at {logging.getLevelName(console_level)} level")
    
    # Create file handler for detailed logging if we have a directory
    if log_dir and file_level is not None:
        try:
            # Ensure log directory exists
            directory = ensure_log_directory(log_dir)
            
            # Create a file handler for detailed logs
            log_file = os.path.join(directory, f"abstractllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            
            # Create plain formatter with full details for file logs (no colors, no truncation)
            file_formatter = PlainFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            # Add file handler to the root logger so all loggers inherit it
            root_logger.addHandler(file_handler)
            
            logger.info(f"File logging enabled at {logging.getLevelName(file_level)} level")
            logger.info(f"Detailed logs will be written to: {log_file}")
            logger.info(f"Request and response payloads will be stored in: {directory}")
            
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    # Configure specific component loggers
    # Make sure alma.steps logger follows the same levels and uses our handlers
    alma_logger = logging.getLogger("alma.steps")
    alma_logger.setLevel(base_level)
    # Don't add handlers to specific loggers - let them inherit from root
    alma_logger.propagate = True  # Ensure propagation to root logger
    
    # Also configure other common loggers used by ALMA
    for logger_name in ["alma", "abstractllm.session", "abstractllm.tools"]:
        component_logger = logging.getLogger(logger_name)
        component_logger.setLevel(base_level)
        component_logger.propagate = True
    
    # Be more targeted with third-party logger suppression
    # Allow INFO level for progress bars but suppress warnings/errors unless in debug mode
    if console_level != logging.DEBUG and file_level != logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        # More targeted HuggingFace suppression - allow progress but suppress deprecation warnings
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)  # Allow progress, suppress deprecation
        logging.getLogger("huggingface_hub._snapshot_download").setLevel(logging.INFO)  # Allow progress info
        logging.getLogger("huggingface_hub.utils._errors").setLevel(logging.WARNING)  # Allow error messages
        
        # Suppress transformers warnings but allow errors
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        # Suppress third-party warnings (HuggingFace, etc.)
        suppress_third_party_warnings()
    else:
        # In debug mode, allow more visibility but still suppress the most annoying warnings
        suppress_third_party_warnings()

def log_step(step_number: int, step_name: str, message: str, logger_name: str = "alma.steps") -> None:
    """
    Log a step in the agent's process with consistent formatting.
    
    Args:
        step_number: Sequential step number
        step_name: Name/description of the step (e.g., "USER→AGENT", "AGENT→LLM")
        message: Detailed message about what happened in this step
        logger_name: Name of the logger to use (default: "alma.steps")
    
    Example:
        >>> log_step(1, "USER→AGENT", "Received query: How to create a file?")
        >>> log_step(2, "AGENT→LLM", "Sending query to LLM with tool support enabled")
    """
    step_logger = logging.getLogger(logger_name)
    step_logger.info(f"STEP {step_number}: {step_name} - {message}")

class ColoredTruncatingFormatter(logging.Formatter):
    """
    Custom formatter that adds colors and truncates long messages for console output.
    """
    
    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: GREY_ITALIC,
        logging.INFO: BLUE_ITALIC,
        logging.WARNING: f'\033[1m\033[33m',  # Yellow bold
        logging.ERROR: RED_BOLD,
        logging.CRITICAL: RED_BOLD,
    }
    
    def __init__(self, max_message_length: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_message_length = max_message_length
    
    def format(self, record):
        # Format the record first to get the full message
        formatted = super().format(record)
        
        # Truncate message if too long (without modifying the original record)
        if len(formatted) > self.max_message_length:
            # Find a good place to truncate (try to break at word boundaries)
            half_length = self.max_message_length // 2 - 10  # Leave room for ellipsis
            
            # Get the beginning and end parts
            start_part = formatted[:half_length].rsplit(' ', 1)[0] if ' ' in formatted[:half_length] else formatted[:half_length]
            end_part = formatted[-half_length:].split(' ', 1)[-1] if ' ' in formatted[-half_length:] else formatted[-half_length:]
            
            # Create truncated message
            omitted_chars = len(formatted) - len(start_part) - len(end_part)
            formatted = f"{start_part}... [{omitted_chars} chars omitted] ...{end_part}"
        
        # Apply color based on log level
        color = self.LEVEL_COLORS.get(record.levelno, '')
        
        # Add color if we have one
        if color:
            formatted = f"{color}{formatted}{RESET}"
        
        return formatted


class PlainFormatter(logging.Formatter):
    """
    Plain formatter for file output without colors or truncation.
    """
    pass 


# Call warning suppression at module import time to catch warnings
# that happen before logging is explicitly configured
suppress_third_party_warnings() 