#!/usr/bin/env python3
"""
Basic agent for AbstractLLM with tool support.

This script demonstrates how to create a simple agent that uses the AbstractLLM
framework with a file reading tool, compatible with multiple providers.
"""

import os
import sys
import json
import logging
import argparse
import re
import time
import signal
import functools
import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Generator, Tuple
from pathlib import Path
from contextlib import contextmanager

from abstractllm import create_llm
from abstractllm.types import Message, GenerateResponse
from abstractllm.enums import ModelParameter, ModelCapability, MessageRole
from abstractllm.session import Session, SessionManager
from abstractllm.tools import function_to_tool_definition, ToolDefinition, ToolCall, ToolCallRequest
from abstractllm.exceptions import UnsupportedFeatureError

# Configure logging based on environment variables
LOG_LEVEL = os.environ.get("ABSTRACTLLM_LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get(
    "ABSTRACTLLM_LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE = os.environ.get("ABSTRACTLLM_LOG_FILE", None)
LOG_JSON = os.environ.get("ABSTRACTLLM_LOG_JSON", "0").lower() in ("1", "true", "yes")

def configure_logging(debug_mode=False):
    """
    Configure logging based on environment variables and debug mode.
    
    Args:
        debug_mode: Whether to enable debug mode logging
    
    Returns:
        The configured logger
    """
    # Determine log level
    level_name = "DEBUG" if debug_mode else LOG_LEVEL
    level = getattr(logging, level_name, logging.INFO)
    
    # Configure handlers
    handlers = []
    
    # Add file handler if specified
    if LOG_FILE:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console.setFormatter(console_formatter)
    handlers.append(console)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=handlers
    )
    
    # Create abstractllm logger
    logger = logging.getLogger("abstractllm")
    logger.setLevel(level)
    
    return logger

# Initialize logger
logger = logging.getLogger("abstractllm")

# Security configuration for tools
TOOL_SECURITY_CONFIG = {
    "read_file": {
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "max_execution_time": 5,  # seconds
        "allowed_directories": [os.getcwd()],  # Current working directory
        "max_lines": 10000,  # Maximum number of lines to read
        "sensitive_patterns": [
            (r"\b(?:\d{3}-\d{2}-\d{4})\b", "***-**-****"),  # SSN
            (r"\b(?:\d{4}-\d{4}-\d{4}-\d{4})\b", "****-****-****-****"),  # Credit card
        ],
    }
}

# Define timeout context manager for handling execution timeouts
@contextmanager
def timeout(seconds):
    """Context manager for timing out function execution."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function execution timed out after {seconds} seconds")
    
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def log_step(step_number: int, step_name: str, message: str, level: int = logging.INFO) -> None:
    """
    Log a step in the tool call flow.
    
    Args:
        step_number: The step number in the flow
        step_name: The name of the step (e.g., "USER→AGENT")
        message: The log message
        level: The logging level (default: INFO)
    """
    logger.log(level, f"STEP {step_number}: {step_name} - {message}")
    
    # Also log as structured event
    log_json_event("step", {
        "step_number": step_number,
        "step_name": step_name,
        "message": message
    })

def log_json_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Log a structured JSON event for machine parsing.
    
    Args:
        event_type: Type of event (e.g., "step", "tool_call", "tool_result")
        data: Event data dictionary
    """
    # Add timestamp and event type
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        **data
    }
    
    # Log as JSON
    logger.debug(f"JSON_EVENT: {json.dumps(log_data)}")

def log_llm_request(prompt: str, tools: List[Any]) -> None:
    """Log an LLM request with available tools."""
    # Extract tool names safely
    tool_names = []
    if isinstance(tools, list):
        for tool in tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            elif isinstance(tool, dict) and 'name' in tool:
                tool_names.append(tool['name'])
            else:
                tool_names.append("unknown")
    else:
        tool_names = ["N/A"]
    
    logger.debug(f"LLM Request: {prompt[:100]}... with tools: {tool_names}")
    
    # Log as structured event
    log_json_event("llm_request", {
        "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "tools": tool_names,
        "prompt_length": len(prompt)
    })

def log_tool_call_request(tool_calls: List[Dict[str, Any]]) -> None:
    """Log tool calls requested by the LLM."""
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name", "unknown")
        tool_args = tool_call.get("arguments", {})
        logger.info(f"Tool call {i+1}/{len(tool_calls)}: {tool_name}")
        logger.debug(f"Arguments: {json.dumps(tool_args)}")
        
        # Log as structured event
        log_json_event("tool_call", {
            "tool_name": tool_name,
            "tool_id": tool_call.get("id", "unknown"),
            "arguments": tool_args
        })

def log_tool_execution(tool_name: str, args: Dict[str, Any], result: Any, duration: float = None, error: Optional[str] = None) -> None:
    """Log a tool execution result with timing information."""
    if error:
        logger.error(f"Tool execution failed: {tool_name}")
        logger.error(f"Arguments: {json.dumps(args)}")
        logger.error(f"Error: {error}")
    else:
        duration_str = f" in {duration:.2f}s" if duration is not None else ""
        logger.info(f"Tool execution completed{duration_str}: {tool_name}")
        logger.debug(f"Arguments: {json.dumps(args)}")
        
        # Log result summary based on type
        if isinstance(result, str):
            preview = result[:200] + ("..." if len(result) > 200 else "")
            logger.debug(f"Result preview: {preview}")
        else:
            logger.debug(f"Result: {result}")
    
    # Log as structured event
    log_json_event("tool_result", {
        "tool_name": tool_name,
        "success": error is None,
        "error": error,
        "execution_time": duration,
        "result_length": len(result) if isinstance(result, str) else None
    })

def log_tool_results_to_llm(tool_results: List[Dict[str, Any]]) -> None:
    """Log tool results being sent back to the LLM."""
    logger.info(f"Sending {len(tool_results)} tool results back to LLM")
    
    # Log as structured event
    log_json_event("tool_results_to_llm", {
        "count": len(tool_results),
        "tool_names": [result.get("name", "unknown") for result in tool_results]
    })

def log_final_response(response: str) -> None:
    """Log the final response from the LLM."""
    preview = response[:100] + ("..." if len(response) > 100 else "")
    logger.info(f"Final response: {preview}")
    
    # Log as structured event
    log_json_event("final_response", {
        "response_preview": preview,
        "response_length": len(response)
    })

def is_safe_path(file_path: str, allowed_directories: List[str]) -> bool:
    """
    Check if a file path is within allowed directories.
    
    Args:
        file_path: The path to validate
        allowed_directories: List of allowed directory paths
        
    Returns:
        True if the path is safe, False otherwise
    """
    # Normalize and resolve the path to handle ../ and similar tricks
    abs_path = os.path.abspath(os.path.normpath(file_path))
    
    # Check if the path is within any allowed directory
    return any(
        os.path.commonpath([abs_path, os.path.abspath(allowed_dir)]) == os.path.abspath(allowed_dir)
        for allowed_dir in allowed_directories
    )

def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate tool parameters for security.
    
    Args:
        tool_name: Name of the tool
        parameters: Parameters to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if tool_name == "read_file":
        # Validate file_path
        if "file_path" not in parameters:
            return False, "Missing required parameter: file_path"
        
        file_path = parameters["file_path"]
        if not isinstance(file_path, str):
            return False, "file_path must be a string"
        
        # Check for suspicious patterns
        suspicious_patterns = ["../", "/..", "~", "$", "|", ";", "&", ">", "<"]
        if any(pattern in file_path for pattern in suspicious_patterns):
            return False, "file_path contains suspicious patterns"
        
        # Check if path is within allowed directories
        if not is_safe_path(file_path, TOOL_SECURITY_CONFIG["read_file"]["allowed_directories"]):
            return False, "Access to this file path is not allowed for security reasons"
        
        # Validate max_lines
        if "max_lines" in parameters:
            max_lines = parameters["max_lines"]
            if not (isinstance(max_lines, int) or max_lines is None):
                return False, "max_lines must be an integer or None"
            if isinstance(max_lines, int) and (max_lines <= 0 or max_lines > TOOL_SECURITY_CONFIG["read_file"]["max_lines"]):
                return False, f"max_lines must be between 1 and {TOOL_SECURITY_CONFIG['read_file']['max_lines']}"
    
    # All validations passed
    return True, None

def create_secure_tool_wrapper(func: Callable, max_execution_time: int = 5) -> Callable:
    """
    Create a wrapper that adds security measures to any tool function.
    
    Args:
        func: The tool function to wrap
        max_execution_time: Maximum execution time in seconds
        
    Returns:
        A wrapped function with security measures
    """
    @functools.wraps(func)
    def secure_wrapper(*args, **kwargs):
        # Log the tool call
        logger.debug(f"Executing secure wrapped tool: {func.__name__} with args: {args}, kwargs: {kwargs}")
        
        # Validate parameters based on tool name
        is_valid, error_message = validate_tool_parameters(func.__name__, kwargs)
        if not is_valid:
            logger.warning(f"Tool parameter validation failed: {error_message}")
            return f"Error: {error_message}"
        
        # Execute with timeout
        start_time = time.time()
        try:
            with timeout(max_execution_time):
                result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            logger.info(f"Tool executed successfully in {execution_time:.2f} seconds")
            
            # Sanitize the result
            sanitized_result = sanitize_tool_output(result, func.__name__)
            
            return sanitized_result
        except TimeoutError as e:
            logger.error(f"Tool execution timed out: {str(e)}")
            return f"Error: Tool execution timed out after {max_execution_time} seconds"
        except Exception as e:
            logger.error(f"Error during tool execution: {str(e)}")
            return f"Error during tool execution: {str(e)}"
    
    return secure_wrapper

def sanitize_tool_output(output: Any, tool_name: str) -> Any:
    """
    Sanitize tool outputs to prevent security issues.
    
    Args:
        output: The output to sanitize
        tool_name: Name of the tool that produced the output
        
    Returns:
        Sanitized output
    """
    if output is None:
        return None
    
    # Convert to string for text-based operations
    if isinstance(output, (int, float, bool)):
        output = str(output)
    
    if isinstance(output, str):
        # Limit output size for read_file
        if tool_name == "read_file":
            max_size = TOOL_SECURITY_CONFIG["read_file"]["max_file_size"]
            if len(output) > max_size:
                output = output[:max_size] + "\n... (output truncated due to size limits)"
        
        # Check for potentially sensitive patterns
        sensitive_patterns = TOOL_SECURITY_CONFIG["read_file"]["sensitive_patterns"]
        
        for pattern, replacement in sensitive_patterns:
            output = re.sub(pattern, replacement, output)
    
    return output

def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read the contents of a file with security validation.
    
    Args:
        file_path: The path of the file to read
        max_lines: Maximum number of lines to read (optional)
        
    Returns:
        The file contents as a string, or an error message
    """
    # Security validation - Allowed directories
    allowed_directories = TOOL_SECURITY_CONFIG["read_file"]["allowed_directories"]
    if not is_safe_path(file_path, allowed_directories):
        return "Error: Access to this file path is not allowed for security reasons."
    
    # Validate max_lines
    max_allowed_lines = TOOL_SECURITY_CONFIG["read_file"]["max_lines"]
    if max_lines is not None and (max_lines <= 0 or max_lines > max_allowed_lines):
        return f"Error: max_lines must be between 1 and {max_allowed_lines}."
    
    # Check if file exists
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    
    # Check file size
    max_file_size = TOOL_SECURITY_CONFIG["read_file"]["max_file_size"]
    file_size = os.path.getsize(file_path)
    if file_size > max_file_size:
        return f"Error: File size ({file_size} bytes) exceeds maximum allowed size ({max_file_size} bytes)."
    
    # Read the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_lines is not None:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
                content = ''.join(lines)
                if i >= max_lines:
                    content += f"\n... (file truncated at {max_lines} lines)"
            else:
                # Read file with size limit
                content = f.read(max_file_size)
                if len(content) >= max_file_size:
                    content += "\n... (file truncated due to size limits)"
        
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def logged_read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read file with comprehensive logging.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (optional)
        
    Returns:
        File contents or error message
    """
    logger.info(f"Executing logged_read_file tool: path={file_path}, max_lines={max_lines}")
    start_time = time.time()
    
    try:
        # Execute the function
        result = read_file(file_path, max_lines)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Determine if there was an error
        is_error = result.startswith("Error:")
        
        # Log the result
        if is_error:
            logger.error(f"logged_read_file failed in {duration:.2f}s: {result}")
            log_tool_execution(
                "logged_read_file", 
                {"file_path": file_path, "max_lines": max_lines}, 
                result,
                duration=duration,
                error=result
            )
        else:
            result_length = len(result) if result else 0
            logger.info(f"logged_read_file completed in {duration:.2f}s, result length: {result_length}")
            log_tool_execution(
                "logged_read_file", 
                {"file_path": file_path, "max_lines": max_lines}, 
                result, 
                duration=duration
            )
        
        return result
    except Exception as e:
        # Calculate duration
        duration = time.time() - start_time
        
        # Log the exception
        error_msg = f"Error in logged_read_file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Log the execution
        log_tool_execution(
            "logged_read_file", 
            {"file_path": file_path, "max_lines": max_lines}, 
            "", 
            duration=duration,
            error=str(e)
        )
        
        return f"Error: {str(e)}"

class BasicAgent:
    """
    A basic agent that uses AbstractLLM with tool support.
    """
    
    def __init__(self, provider_name: str = "anthropic", model_name: Optional[str] = None, 
                 api_key: Optional[str] = None, debug: bool = False, session_id: Optional[str] = None):
        """
        Initialize the agent with a specified provider and model.
        
        Args:
            provider_name: Name of the provider to use
            model_name: Specific model to use (optional)
            api_key: API key for the provider (optional)
            debug: Enable debug logging
            session_id: Session ID for multi-turn conversations
        """
        # Configure logging
        global logger
        logger = configure_logging(debug_mode=debug)
        logger.info(f"Initializing BasicAgent with provider: {provider_name}")
        
        # Record initialization time for profiling
        init_start_time = time.time()
        
        # Configure the LLM provider
        from abstractllm import create_llm
        
        provider_config = self._get_provider_config(provider_name, model_name, api_key)
        self.llm = create_llm(provider_name, **provider_config)
        logger.debug(f"LLM provider initialized: {provider_name}")
        
        # Wrap the read_file function with security measures
        secure_read_file = create_secure_tool_wrapper(
            logged_read_file,
            max_execution_time=TOOL_SECURITY_CONFIG["read_file"]["max_execution_time"]
        )
        
        # Create tool definition from the secured function
        file_reader_tool = function_to_tool_definition(secure_read_file)
        logger.debug(f"Tool definition: {json.dumps(file_reader_tool.to_dict(), indent=2)}")
        
        # Verify that the provider supports tool calls
        logger.info("Checking provider capabilities")
        capabilities = self.llm.get_capabilities()
        logger.debug(f"Provider capabilities: {capabilities}")
        
        # Get current model for detailed logging
        self.model_name = self.llm.config_manager.get_param("model")
        logger.info(f"Using model: {self.model_name}")
        
        can_use_tools = capabilities.get(ModelCapability.TOOL_USE, False) or capabilities.get(ModelCapability.FUNCTION_CALLING, False)
        if not can_use_tools:
            logger.warning(f"Provider may not support tool calls according to capabilities with model {self.model_name}")
            print(f"Warning: The selected provider {provider_name} with model {self.model_name} may not support tool calls")
        
        # Create a session manager for handling conversations
        self.session_manager = SessionManager()
        self.session_id = session_id or "default_session"
        
        # Create a session with a tool-focused system prompt
        system_prompt = (
            "You are a helpful AI assistant that can read files when requested. "
            "You have access to a tool called logged_read_file that can read file contents. "
            "When a user asks you to read a file, you MUST use the logged_read_file tool. "
            "DO NOT make up file contents. ONLY use the logged_read_file tool when asked to read a file. "
            "You MUST NEVER respond with 'I can help you read the file' without actually using the tool. "
            "Instead, you should immediately use the logged_read_file tool without asking for permission. "
            "The logged_read_file tool accepts the following parameters:\n"
            "- file_path: Path to the file to read (required)\n"
            "- max_lines: Maximum number of lines to read (optional)\n\n"
            "Examples of when to use the tool:\n"
            "1. USER: 'Please read test_file.txt'\n"
            "   YOU: *Use logged_read_file with file_path='test_file.txt'*\n"
            "2. USER: 'Show me the first 5 lines of test_file.txt'\n"
            "   YOU: *Use logged_read_file with file_path='test_file.txt', max_lines=5*\n"
            "3. USER: 'What's in the file test_file.txt?'\n"
            "   YOU: *Use logged_read_file with file_path='test_file.txt'*\n\n"
            "NEVER respond with statements like 'I'll help you read that file' or 'I can assist with that' "
            "without actually using the tool. Always use the tool immediately when asked to read a file."
        )
        
        # Initialize a session with the system prompt
        logger.info("Creating session with tool support")
        # Create a new session with the provider
        session = self.session_manager.create_session(
            system_prompt=system_prompt,
            provider=self.llm
        )
        
        # Store the session with our session ID
        self.session_manager.sessions[self.session_id] = session
        
        # Add the tool to the session
        session.add_tool(file_reader_tool)
        
        logger.debug(f"Session created with ID: {self.session_id}")
        
        # Define tool functions mapping with secure wrappers
        self.tool_functions = {
            "logged_read_file": secure_read_file
        }
        logger.debug(f"Tool functions registered: {list(self.tool_functions.keys())}")
        logger.info("All tools wrapped with security measures")
        
        # Store provider name for later use
        self.provider_name = provider_name
        
        # Store other configuration
        self.temperature = 0.1  # Lower temperature for more deterministic behavior
        self.max_tokens = 1000  # Default max tokens for responses
        
        # Log initialization completion time
        init_duration = time.time() - init_start_time
        logger.info(f"BasicAgent initialization completed in {init_duration:.2f}s")
        
        # Log initialization as a JSON event
        log_json_event("agent_initialization", {
            "provider": provider_name,
            "model": self.model_name,
            "session_id": self.session_id,
            "tools": list(self.tool_functions.keys()),
            "initialization_time": init_duration,
            "supports_tools": can_use_tools
        })

    def _get_provider_config(self, provider_name: str, model_name: Optional[str] = None, 
                           api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get provider-specific configuration, including API key and model.
        
        Args:
            provider_name: Name of the provider
            model_name: Specific model to use (optional)
            api_key: Provider API key (optional)
            
        Returns:
            Dictionary of provider configuration
        """
        config = {}
        
        # Set provider-specific defaults and get API key
        if provider_name == "openai":
            config["model"] = model_name or "gpt-4o"
            env_api_key = os.environ.get("OPENAI_API_KEY")
            key_name = "OPENAI_API_KEY"
        elif provider_name == "anthropic":
            config["model"] = model_name or "claude-3-7-sonnet-20250219"  # Updated to use 3.7 Sonnet by default
            env_api_key = os.environ.get("ANTHROPIC_API_KEY")
            key_name = "ANTHROPIC_API_KEY"
        elif provider_name == "ollama":
            config["model"] = model_name or "llama3"  # Adjust based on available models
            env_api_key = None  # Not typically needed for Ollama
            key_name = None
        else:
            config["model"] = model_name or "unknown"
            env_api_key = None
            key_name = f"{provider_name.upper()}_API_KEY"
            logger.warning(f"Unknown provider: {provider_name}")
        
        # Use provided API key or get from environment
        if api_key:
            config["api_key"] = api_key
        elif env_api_key:
            config["api_key"] = env_api_key
        elif key_name and key_name != "OLLAMA_API_KEY":  # Ollama doesn't need an API key
            logger.warning(f"{key_name} environment variable not set")
        
        logger.debug(f"Provider config: {config}")
        return config
    
    def run(self, query: str) -> str:
        """
        Process a user query and return the response following the LLM-first flow.
        
        Args:
            query: The user's query
            
        Returns:
            The agent's response
        """
        try:
            # STEP 1: User → Agent - Log the incoming query
            log_step(1, "USER→AGENT", f"Received query: {query[:100]}...")
            
            # Get or create a session
            session = self.session_manager.get_session(self.session_id)
            
            # Add user message to session for context
            session.add_message("user", query)
            
            # Define available tools - ALL tools the agent can use
            logger.debug(f"Available tools: {list(self.tool_functions.keys())}")
            
            # STEP 2: Agent → LLM - Send query to LLM with tools
            log_step(2, "AGENT→LLM", f"Sending query to provider: {self.provider_name}")
            log_llm_request(query, session.tools)
            
            try:
                # Start timing for LLM processing
                llm_start_time = time.time()
                
                # Use generate_with_tools to get a response with potential tool calls
                # Security is handled by wrapped tool functions
                response = session.generate_with_tools(
                    tool_functions=self.tool_functions,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Calculate LLM processing time
                llm_duration = time.time() - llm_start_time
                logger.debug(f"LLM processing time: {llm_duration:.2f}s")
                
                # Check if the response has tool calls
                has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
                
                print("RESPONSE: ", response)
                if has_tool_calls:
                    # STEP 3: LLM → Agent - Tool call request
                    tool_calls = response.tool_calls.tool_calls if hasattr(response.tool_calls, 'tool_calls') else []
                    tool_names = [tc.name for tc in tool_calls]
                    log_step(3, "LLM→AGENT", f"LLM requested {len(tool_calls)} tool(s): {', '.join(tool_names)}")
                    
                    # Log each individual tool call
                    log_tool_call_request([
                        {"name": tc.name, "id": tc.id if hasattr(tc, 'id') else 'unknown', "arguments": tc.arguments} 
                        for tc in tool_calls
                    ])
                    
                    # STEP 4: Agent → Tool execution
                    for tool_call in tool_calls:
                        log_step(4, "AGENT→TOOL", f"Executing tool: {tool_call.name}")
                    
                    # STEP 5: Tool → Agent → LLM - Tool results sent back to LLM
                    log_step(5, "TOOL→AGENT→LLM", "Tool execution completed, results sent to LLM")
                    
                    # Log the tools results being sent to LLM
                    tool_results = [{"name": tc.name, "id": tc.id if hasattr(tc, 'id') else 'unknown'} for tc in tool_calls]
                    log_tool_results_to_llm(tool_results)
                else:
                    # No tool calls - direct LLM response
                    log_step(3, "LLM→AGENT", f"LLM generated response without tool calls in {llm_duration:.2f}s")
                
                # Extract the final response content
                final_response = response.content if response.content else ""
                
                # STEP 6: LLM → Agent - Final response
                log_step(6, "LLM→AGENT", f"Received final response from LLM")
                log_final_response(final_response)
                
                # Add assistant response to session for multi-turn context
                session.add_message("assistant", final_response)
                
                # STEP 7: Agent → User - Return response to user
                log_step(7, "AGENT→USER", f"Sending response to user: {final_response[:100]}..." if len(final_response) > 100 else final_response)
                
                # Log entire execution as JSON event
                log_json_event("query_execution", {
                    "query": query[:100] + ("..." if len(query) > 100 else ""),
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "used_tools": tool_names if has_tool_calls else [],
                    "execution_time": time.time() - llm_start_time,
                    "response_length": len(final_response)
                })
                
                # Return the final response
                return final_response
                
            except UnsupportedFeatureError as e:
                logger.error(f"Tool calling not supported: {e}")
                error_msg = f"I'm sorry, tool calling is not supported with the current configuration: {str(e)}"
                log_step(6, "ERROR", error_msg, level=logging.ERROR)
                return error_msg
            except TimeoutError as e:
                logger.error(f"Tool execution timed out: {e}")
                error_msg = f"I'm sorry, the operation timed out: {str(e)}"
                log_step(6, "ERROR", error_msg, level=logging.ERROR)
                return error_msg
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            log_step(6, "ERROR", error_msg, level=logging.ERROR)
            return f"I encountered an error processing your request: {str(e)}"
        
    def run_streaming(self, query: str) -> None:
        """
        Process a user query and stream the response with tool execution.
        
        Args:
            query: The user's query
        """
        try:
            # STEP 1: User → Agent - Log the incoming query
            log_step(1, "USER→AGENT", f"Received streaming query: {query[:100]}...")
            
            # Get or create a session
            session = self.session_manager.get_session(self.session_id)
            
            # Add user message to session for context
            session.add_message("user", query)
            
            # Define available tools
            logger.debug(f"Available streaming tools: {list(self.tool_functions.keys())}")
            
            # STEP 2: Agent → LLM - Send query to LLM with tools
            log_step(2, "AGENT→LLM", f"Sending streaming query to provider: {self.provider_name}")
            log_llm_request(query, session.tools)
            
            try:
                # Track content for session history
                content_buffer = []
                
                # Track timing information
                stream_start_time = time.time()
                
                # Track tool call status
                tool_call_status = {
                    "has_tool_calls": False,
                    "tool_calls": [],
                    "current_step": 3
                }
                
                # Use the streaming version of generate_with_tools with secure tools
                logger.info("Using session.generate_with_tools_streaming")
                log_step(3, "LLM→AGENT", "Starting streaming generation")
                
                stream = session.generate_with_tools_streaming(
                    tool_functions=self.tool_functions,  # Using secure wrapped tools
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                print("\nAssistant: ", end="", flush=True)
                for chunk in stream:
                    if isinstance(chunk, str):
                        # Regular content chunk - display and save
                        print(chunk, end="", flush=True)
                        content_buffer.append(chunk)
                        
                        # Log occasional progress for long generations
                        if len(content_buffer) % 10 == 0:
                            logger.debug(f"Streaming progress: {len(''.join(content_buffer))} characters so far")
                            
                    elif isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                        print("CHUNK: ", chunk)
                        # Tool result - indicates the LLM requested a tool call
                        result = chunk.get("result", {})
                        tool_name = result.get("name", "unknown")
                        tool_args = result.get("arguments", {})
                        output = result.get("output", "")
                        error = result.get("error")
                        
                        # Update tool calls tracking
                        tool_call_status["has_tool_calls"] = True
                        tool_call_status["tool_calls"].append({
                            "name": tool_name, 
                            "arguments": tool_args,
                            "success": error is None
                        })
                        
                        # STEP 3: LLM → Agent - Tool call request
                        if tool_call_status["current_step"] == 3:
                            log_step(3, "LLM→AGENT", f"LLM requested tool call: {tool_name}")
                            tool_call_status["current_step"] = 4
                        
                        # STEP 4: Agent → Tool execution
                        log_step(4, "AGENT→TOOL", f"Executing tool: {tool_name} with args: {tool_args}")
                        
                        # STEP 5: Tool → Agent → LLM - Tool results sent back to LLM
                        log_step(5, "AGENT→LLM", f"Sending {tool_name} results to LLM")
                        
                        # Log tool execution
                        log_tool_execution(
                            tool_name, 
                            tool_args, 
                            output if not error else "", 
                            error=error
                        )
                        
                        # Visual indicator of tool execution
                        if error:
                            print(f"\n[Tool {tool_name} error: {error}]", flush=True)
                        else:
                            # Securely sanitize any output before display
                            safe_output = sanitize_tool_output(output, tool_name)
                            print(f"\n[Tool {tool_name} executed successfully]", flush=True)
                
                # Calculate total streaming time
                streaming_duration = time.time() - stream_start_time
                
                # Join content chunks for session history
                final_content = "".join(content_buffer)
                
                # Add assistant response to session for multi-turn context
                session.add_message("assistant", final_content)
                
                # STEP 6: LLM → Agent → User - Final response
                if tool_call_status["has_tool_calls"]:
                    log_step(6, "LLM→AGENT→USER", f"Completed streaming response with {len(tool_call_status['tool_calls'])} tool calls in {streaming_duration:.2f}s")
                else:
                    log_step(6, "LLM→AGENT→USER", f"Completed streaming response without tool calls in {streaming_duration:.2f}s")
                
                # Log the final content
                log_final_response(final_content)
                
                # Log entire execution as JSON event
                log_json_event("streaming_execution", {
                    "query": query[:100] + ("..." if len(query) > 100 else ""),
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "used_tools": [tc["name"] for tc in tool_call_status["tool_calls"]],
                    "execution_time": streaming_duration,
                    "response_length": len(final_content)
                })
                
                # Complete the output with a newline
                print("\n")
                
            except UnsupportedFeatureError as e:
                logger.error(f"Tool calling not supported in streaming mode: {e}")
                error_msg = f"I'm sorry, tool calling is not supported with the current configuration: {str(e)}"
                log_step(6, "ERROR", error_msg, level=logging.ERROR)
                print(f"\n{error_msg}")
            except TimeoutError as e:
                logger.error(f"Tool execution timed out in streaming mode: {e}")
                error_msg = f"I'm sorry, the operation timed out: {str(e)}"
                log_step(6, "ERROR", error_msg, level=logging.ERROR)
                print(f"\n{error_msg}")
            
        except Exception as e:
            error_msg = f"Error during streaming agent execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            log_step(6, "ERROR", error_msg, level=logging.ERROR)
            print(f"\nError: {error_msg}")
    
    def run_interactive(self):
        """
        Run the agent in interactive mode, accepting user input from the console.
        """
        logger.info("Starting interactive mode")
        print(f"Basic Agent with {self.provider_name.capitalize()} provider")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'stream' before your query to use streaming mode.")
        print("Example: 'Please read the file test_file.txt'\n")
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested to exit")
                print("\nGoodbye!")
                break
            
            # Check if user wants streaming mode
            streaming = False
            if user_input.lower().startswith("stream "):
                streaming = True
                user_input = user_input[7:]  # Remove "stream " prefix
                logger.debug("Streaming mode requested")
            
            # Process the query - uses the same session for multi-turn conversation
            try:
                if streaming:
                    self.run_streaming(user_input)
                else:
                    response = self.run(user_input)
                    print(f"\nAssistant: {response}")
            except Exception as e:
                error_msg = f"Error in interactive mode: {str(e)}"
                logger.error(error_msg, exc_info=True)
                print(f"\nError: {str(e)}")


def main():
    """Main function to parse arguments and run the agent."""
    parser = argparse.ArgumentParser(description="Basic Agent with tool support")
    parser.add_argument("--provider", default="anthropic", choices=["openai", "anthropic", "ollama"], 
                       help="Provider to use (default: anthropic)")
    parser.add_argument("--model", help="Specific model to use (optional, defaults to provider's default)")
    parser.add_argument("--api-key", help="Provider API key (optional, will use environment variable if not provided)")
    parser.add_argument("--query", help="Single query to run (if not provided, will run in interactive mode)")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for the query")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("--session-id", help="Session ID for maintaining conversation context (optional)")
    parser.add_argument("--log-file", help="Path to log file (optional, overrides ABSTRACTLLM_LOG_FILE)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Log level (optional, overrides ABSTRACTLLM_LOG_LEVEL)")
    
    args = parser.parse_args()
    
    # Override environment variables with command line arguments
    if args.log_file:
        os.environ["ABSTRACTLLM_LOG_FILE"] = args.log_file
    if args.log_level:
        os.environ["ABSTRACTLLM_LOG_LEVEL"] = args.log_level
    
    try:
        # Record execution start time
        start_time = time.time()
        
        # Initialize logger
        logger = configure_logging(debug_mode=args.debug)
        logger.info("Starting BasicAgent")
        logger.debug(f"Command line arguments: {vars(args)}")
        
        # Create the agent
        agent = BasicAgent(
            provider_name=args.provider, 
            model_name=args.model, 
            api_key=args.api_key, 
            debug=args.debug,
            session_id=args.session_id
        )
        
        # Run the agent
        if args.query:
            # Single query mode
            logger.info(f"Running single query: {args.query}")
            
            try:
                if args.stream:
                    agent.run_streaming(args.query)
                else:
                    response = agent.run(args.query)
                    print(f"Response: {response}")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                print(f"Error processing query: {str(e)}")
                return 1
        else:
            # Interactive mode
            logger.info("Starting interactive mode")
            try:
                agent.run_interactive()
            except KeyboardInterrupt:
                logger.info("Interactive mode terminated by user (KeyboardInterrupt)")
                print("\nGoodbye! Session terminated.")
                return 0
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}", exc_info=True)
                print(f"Error: {str(e)}")
                return 1
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        logger.info(f"Agent execution completed successfully in {execution_time:.2f}s")
        
        # Log execution as JSON event
        log_json_event("execution_completed", {
            "success": True,
            "execution_time": execution_time,
            "mode": "streaming" if args.stream and args.query else "single_query" if args.query else "interactive"
        })
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        
        # Log execution failure
        log_json_event("execution_failed", {
            "error": str(e),
            "execution_time": time.time() - start_time if 'start_time' in locals() else None
        })
        
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 