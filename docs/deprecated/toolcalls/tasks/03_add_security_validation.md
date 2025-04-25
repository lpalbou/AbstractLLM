# Task 3: Add Security Validation for Tool Execution

## Overview

This task focuses on implementing security measures for tool execution in AbstractLLM. Proper security validation is essential even when using the LLM-first approach, as LLMs can sometimes generate insecure tool calls based on user inputs.

## Background

Even with the LLM-first approach (where the LLM decides tool usage), security is essential:

1. LLMs can be vulnerable to prompt injection attacks
2. Users may try to access files or resources outside allowed boundaries
3. Malicious inputs could be passed through the LLM to tools
4. Tools might execute potentially dangerous operations

## Required Changes

### 1. Create Safe Path Validation

Implement a function to validate file paths for the `read_file` tool:

```python
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
```

### 2. Update the `read_file` Tool with Security Validation

Modify the `read_file` function to include security validation:

```python
def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read the contents of a file with security validation.
    
    Args:
        file_path: The path of the file to read
        max_lines: Maximum number of lines to read (optional)
        
    Returns:
        The file contents as a string, or an error message
    """
    # Define allowed directories (customize based on your environment)
    allowed_directories = [
        os.getcwd(),  # Current working directory
        os.path.join(os.getcwd(), "data"),  # Data directory
        # Add more allowed directories as needed
    ]
    
    # Security validation
    if not is_safe_path(file_path, allowed_directories):
        return "Error: Access to this file path is not allowed for security reasons."
    
    # Validate max_lines
    if max_lines is not None and (max_lines <= 0 or max_lines > 10000):
        return f"Error: max_lines must be between 1 and 10000."
    
    # Execute with safeguards
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_lines is not None:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
                content = ''.join(lines)
            else:
                # Limit file size for security
                content = f.read(10 * 1024 * 1024)  # 10MB limit
                if len(content) >= 10 * 1024 * 1024:
                    content += "\n... (file truncated due to size limits)"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"
```

### 3. Create a Secure Tool Wrapper Function

Implement a wrapper function to add security to any tool:

```python
def create_secure_tool_wrapper(func: Callable, max_execution_time: int = 5) -> Callable:
    """
    Create a wrapper that adds security measures to any tool function.
    
    Args:
        func: The tool function to wrap
        max_execution_time: Maximum execution time in seconds
        
    Returns:
        A wrapped function with security measures
    """
    import functools
    import signal
    
    @functools.wraps(func)
    def secure_wrapper(*args, **kwargs):
        # Set up timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Tool execution timed out after {max_execution_time} seconds")
        
        # Execute with timeout
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_execution_time)
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Cancel timeout
            signal.alarm(0)
            
            return result
        except TimeoutError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error during tool execution: {str(e)}"
    
    return secure_wrapper
```

### 4. Apply the Secure Wrapper to All Tools

Update the tool functions dictionary to use secure wrappers:

```python
# Define basic tools
def calculate(operation: str, a: float, b: float) -> float:
    """Perform a calculation."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Apply secure wrappers to all tools
tool_functions = {
    "read_file": create_secure_tool_wrapper(read_file),
    "calculate": create_secure_tool_wrapper(calculate),
    # Add other tools with secure wrappers
}
```

### 5. Add Input Validation for Tool Parameters

Implement parameter validation for all tools:

```python
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
        
        # Validate max_lines
        if "max_lines" in parameters:
            max_lines = parameters["max_lines"]
            if not (isinstance(max_lines, int) or max_lines is None):
                return False, "max_lines must be an integer or None"
            if isinstance(max_lines, int) and (max_lines <= 0 or max_lines > 10000):
                return False, "max_lines must be between 1 and 10000"
    
    elif tool_name == "calculate":
        # Validate operation
        if "operation" not in parameters:
            return False, "Missing required parameter: operation"
        
        operation = parameters["operation"]
        if not isinstance(operation, str):
            return False, "operation must be a string"
        
        if operation not in ["add", "subtract", "multiply", "divide"]:
            return False, "Unknown operation: " + operation
        
        # Validate a and b
        for param in ["a", "b"]:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"
            
            value = parameters[param]
            if not isinstance(value, (int, float)):
                return False, f"{param} must be a number"
    
    # All validations passed
    return True, None
```

### 6. Update the Session's `execute_tool_call` Method

If possible, modify the session's `execute_tool_call` method to include validation:

```python
def execute_tool_call(self, tool_call, tool_functions):
    """Execute a tool call with security validation."""
    # Check if the tool function exists
    if tool_call.name not in tool_functions:
        return {
            "call_id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "output": None,
            "error": f"Tool '{tool_call.name}' not found"
        }
    
    # Validate tool parameters
    is_valid, error_message = validate_tool_parameters(tool_call.name, tool_call.arguments)
    if not is_valid:
        return {
            "call_id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "output": None,
            "error": error_message
        }
    
    # Get the function to execute
    func = tool_functions[tool_call.name]
    
    # Execute the function (already wrapped with security measures)
    try:
        result = func(**tool_call.arguments)
        
        return {
            "call_id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "output": result,
            "error": None
        }
    except Exception as e:
        return {
            "call_id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "output": None,
            "error": str(e)
        }
```

### 7. Sanitize Tool Outputs

Implement output sanitization for sensitive information:

```python
def sanitize_tool_output(output: Any, tool_name: str) -> Any:
    """
    Sanitize tool outputs for security.
    
    Args:
        output: The tool output to sanitize
        tool_name: Name of the tool
        
    Returns:
        Sanitized output
    """
    if output is None:
        return None
    
    # Convert to string for text-based operations
    if isinstance(output, (int, float, bool)):
        output = str(output)
    
    if isinstance(output, str):
        # Limit output size
        MAX_OUTPUT_SIZE = 1_000_000  # 1MB text limit
        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + "\n... (output truncated due to size limits)"
        
        # Check for potentially sensitive patterns
        sensitive_patterns = [
            (r"\b(?:\d{3}-\d{2}-\d{4})\b", "***-**-****"),  # SSN
            (r"\b(?:\d{4}-\d{4}-\d{4}-\d{4})\b", "****-****-****-****"),  # Credit card
            # Add more patterns as needed
        ]
        
        for pattern, replacement in sensitive_patterns:
            output = re.sub(pattern, replacement, output)
    
    return output
```

### 8. Add Configuration for Tool Security

Create configurable security settings:

```python
# Add to the top of the file
TOOL_SECURITY_CONFIG = {
    "read_file": {
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "max_execution_time": 5,  # seconds
        "allowed_directories": [os.getcwd(), os.path.join(os.getcwd(), "data")],
    },
    "calculate": {
        "max_execution_time": 2,  # seconds
        "allowed_operations": ["add", "subtract", "multiply", "divide"],
    }
}
```

## Testing

1. Test security boundaries:
   - Try to read files outside allowed directories
   - Try to read very large files
   - Try to use invalid operations in the calculator

2. Test input validation:
   - Provide invalid parameters to tools
   - Provide parameters of incorrect types
   - Try edge cases (e.g., dividing by zero)

3. Test timeout functionality:
   - Create a test tool that sleeps longer than the timeout

4. Test sanitization:
   - Check if sensitive patterns are properly redacted
   - Verify large outputs are truncated

## Completion Criteria

- ✅ File path validation is implemented
- ✅ Tool parameter validation is in place
- ✅ Secure wrappers are applied to all tools
- ✅ Execution timeouts are implemented
- ✅ Output sanitization is working
- ✅ Security settings are configurable
- ✅ Tests confirm security measures are working 