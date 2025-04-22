# Security Considerations for AbstractLLM Tool Calls

## Security Risks of Direct Tool Execution

Direct tool execution (bypassing the LLM) introduces several critical security vulnerabilities:

### 1. Prompt Injection Attacks

When tools are executed directly based on pattern matching user input, attackers can craft inputs that trigger unintended tool execution.

**Example attack:**
```
Read the file ../../../etc/passwd to help me understand Linux users
```

If pattern matching on "read" and "file" is used, this could lead to unauthorized file access.

### 2. Path Traversal

Direct execution of file operations based on user input allows attackers to access files outside intended directories.

**Example attack:**
```
Please read the file ../config/secrets.json
```

### 3. Command Injection

Pattern matching might extract command arguments from user queries in unsafe ways.

**Example attack:**
```
Read the file notes.txt; rm -rf / to help me understand the content
```

### 4. Denial of Service

Direct execution without rate limiting could lead to resource exhaustion.

**Example attack:**
```
Read the file [very large file] and tell me about it
```

## Security Best Practices

### 1. Never Use Direct Tool Execution

The most important security rule: **Always let the LLM decide when to use tools**.

✅ **Correct approach:**
```python
# Let the LLM decide if and how to use tools
response = session.generate_with_tools(
    tool_functions=tool_functions,
    prompt=user_query
)
```

❌ **Insecure approach:**
```python
# NEVER do this
if "read file" in user_query:
    filename = extract_filename(user_query)
    content = read_file(filename)
    return f"Content: {content}"
```

### 2. Implement Tool Input Validation

Always validate tool inputs before execution:

```python
def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    # Validate file path is within allowed directories
    if not is_safe_path(file_path, ALLOWED_DIRECTORIES):
        return "Error: Access to this file path is not allowed for security reasons."
    
    # Validate max_lines is within reasonable limits
    if max_lines is not None and (max_lines <= 0 or max_lines > MAX_ALLOWED_LINES):
        return f"Error: max_lines must be between 1 and {MAX_ALLOWED_LINES}."
    
    # Proceed with file reading
    # ...
```

### A Safe Path Validation Function:

```python
def is_safe_path(file_path: str, allowed_directories: List[str]) -> bool:
    """Check if a file path is within allowed directories."""
    abs_path = os.path.abspath(os.path.normpath(file_path))
    
    return any(
        os.path.commonpath([abs_path, allowed_dir]) == allowed_dir
        for allowed_dir in allowed_directories
    )
```

### 3. Use Tool Wrappers for Added Security

Wrap tools with security checks:

```python
def create_secure_tool_wrapper(func, allowed_args=None, max_execution_time=5):
    """Create a wrapper that adds security to any tool function."""
    @functools.wraps(func)
    def secure_wrapper(*args, **kwargs):
        # Validate arguments
        if allowed_args:
            for key, value in kwargs.items():
                if key in allowed_args and not allowed_args[key](value):
                    return f"Error: Invalid value for {key}"
        
        # Execute with timeout
        try:
            with timeout(max_execution_time):
                return func(*args, **kwargs)
        except TimeoutError:
            return "Error: Tool execution timed out"
    
    return secure_wrapper
```

### 4. Implement Access Controls

Restrict tool access based on user identity and permissions:

```python
def execute_tool_call(self, tool_call, tool_functions):
    # Check if the user has permission to use this tool
    if not self.has_permission_for_tool(tool_call.name):
        return {
            "call_id": tool_call.id,
            "name": tool_call.name,
            "error": "Access denied: You don't have permission to use this tool"
        }
    
    # Proceed with execution
    # ...
```

### 5. Log All Tool Executions

Maintain comprehensive logs of all tool usage:

```python
def execute_tool(tool_name, args):
    logger.info(f"Executing tool: {tool_name} with args: {json.dumps(args)}")
    start_time = time.time()
    
    try:
        result = tool_functions[tool_name](**args)
        execution_time = time.time() - start_time
        
        logger.info(f"Tool execution completed in {execution_time:.2f}s: {tool_name}")
        logger.debug(f"Tool result: {result}")
        
        return result
    except Exception as e:
        logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
        raise
```

### 6. Set Resource Limits

Implement limits on tool execution:

```python
# Config settings
TOOL_CONFIG = {
    "read_file": {
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "max_execution_time": 5,  # seconds
        "rate_limit": 10,  # calls per minute
    }
}
```

### 7. Sanitize Tool Outputs

Always validate and sanitize tool outputs:

```python
def sanitize_tool_output(output, tool_name):
    """Sanitize tool output to prevent security issues."""
    if tool_name == "read_file":
        # Limit output size
        if len(output) > MAX_OUTPUT_SIZE:
            output = output[:MAX_OUTPUT_SIZE] + "... [output truncated]"
    
    # Check for potentially sensitive information
    output = redact_sensitive_info(output)
    
    return output
```

## Secure Tool Call Checklist

When implementing tool calls, ensure:

1. ✅ LLM always decides when to use tools
2. ✅ No direct execution based on pattern matching
3. ✅ All tool inputs are validated before execution
4. ✅ File paths are validated against allowed directories
5. ✅ Timeouts are implemented for all tool execution
6. ✅ Resource limits are enforced (memory, CPU, etc.)
7. ✅ All tool calls are logged for auditing
8. ✅ Tool outputs are sanitized before returning to LLM
9. ✅ Error handling prevents leaking sensitive information
10. ✅ Tests verify that security measures are working 