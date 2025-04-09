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

# Default log directory
DEFAULT_LOG_DIR = "/tmp/logs/abstractllm"


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


def ensure_log_directory(log_dir: str = DEFAULT_LOG_DIR) -> str:
    """
    Ensure log directory exists and return the path.
    
    Args:
        log_dir: Directory to store log files (default: /tmp/logs/abstractllm)
        
    Returns:
        Path to the log directory
    """
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_log_filename(provider: str, log_type: str, log_dir: str = DEFAULT_LOG_DIR) -> str:
    """
    Generate a filename for a log file.
    
    Args:
        provider: Provider name
        log_type: Type of log (e.g., 'request', 'response')
        log_dir: Directory to store log files
        
    Returns:
        Full path to the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(log_dir, f"{provider}_{log_type}_{timestamp}.json")


def write_to_log_file(data: Dict[str, Any], filename: str) -> None:
    """
    Write data to a log file in JSON format.
    
    Args:
        data: Data to write
        filename: Path to log file
    """
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


def log_request(provider: str, prompt: str, parameters: Dict[str, Any]) -> None:
    """
    Log an LLM request.
    
    Args:
        provider: Provider name
        prompt: The request prompt
        parameters: Request parameters
    """
    timestamp = datetime.now().isoformat()
    logger.debug(f"REQUEST [{provider}]: {timestamp}")
    
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
    
    # For console output, use the safe version with hidden data
    logger.debug(f"Parameters: {safe_parameters}")
    logger.debug(f"Prompt: {prompt}")
    
    # For file logging, write the complete data
    log_dir = ensure_log_directory()
    log_filename = get_log_filename(provider, "request", log_dir)
    
    log_data = {
        "timestamp": timestamp,
        "provider": provider,
        "prompt": prompt,
        "parameters": parameters  # Original, non-truncated parameters
    }
    
    write_to_log_file(log_data, log_filename)


def log_response(provider: str, response: str) -> None:
    """
    Log an LLM response.
    
    Args:
        provider: Provider name
        response: The response text
    """
    timestamp = datetime.now().isoformat()
    logger.debug(f"RESPONSE [{provider}]: {timestamp}")
    
    # Truncate very long responses for console logging
    if len(response) > 1000:
        truncated_response = response[:1000] + f"... [truncated, total length: {len(response)} chars]"
        logger.debug(f"Response: {truncated_response}")
    else:
        logger.debug(f"Response: {response}")
    
    # Write full response to log file
    log_dir = ensure_log_directory()
    log_filename = get_log_filename(provider, "response", log_dir)
    
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
    logger.debug(f"API Request [{provider}]: {method} {url}")


def setup_logging(level: int = logging.INFO, provider_level: int = None, log_dir: str = DEFAULT_LOG_DIR) -> None:
    """
    Set up logging configuration for AbstractLLM.
    
    Args:
        level: Default logging level for all loggers (default: INFO)
        provider_level: Specific level for provider loggers (default: same as level)
        log_dir: Directory to store log files (default: /tmp/logs/abstractllm)
    """
    # Use the same level for providers if not specified
    if provider_level is None:
        provider_level = level
    
    # Set up base logger
    logger.setLevel(level)
    
    # Set up provider-specific loggers
    logging.getLogger("abstractllm.providers").setLevel(provider_level)
    
    # Create console handler if needed
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to the logger
        logger.addHandler(console_handler)
    
    # Create file handler for detailed logging
    try:
        # Ensure log directory exists
        ensure_log_directory(log_dir)
        
        # Create a file handler for detailed logs
        log_file = os.path.join(log_dir, f"abstractllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(min(level, logging.DEBUG))  # Always capture at least DEBUG level in files
        
        # Create formatter with more details for file logs
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add file handler to the logger
        logger.addHandler(file_handler)
        
        logger.info(f"Detailed logs will be written to: {log_file}")
        logger.info(f"Request and response payloads will be stored in: {log_dir}")
        
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}") 