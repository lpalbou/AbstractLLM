# Using Tool Calls

AbstractLLM provides a unified interface for working with tool calls (also known as function calling) across different LLM providers. This guide explains how to implement and use tool calls in your applications.

## What Are Tool Calls?

Tool calls allow an LLM to call external functions to perform actions or retrieve information. This enables more powerful applications by combining the reasoning abilities of LLMs with the ability to interact with external systems.

## LLM-First Architecture

AbstractLLM implements tool calls using an "LLM-First Architecture" where:

1. The LLM decides when tools should be called based on reasoning
2. Tool selection is based on LLM judgment, not pattern matching
3. All user requests flow through the LLM
4. Results from tools are processed by the LLM before returning to the user

This architecture follows this flow:

```
User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User
```

## Installation

To use tool calling capabilities, install AbstractLLM with the tools dependencies:

```bash
pip install abstractllm[tools]
```

## Supported Providers

Tool calls are available across these providers:

- **OpenAI**: Full support for function/tool calling
- **Anthropic**: Tool use capability with Claude models
- **Ollama**: Partial support depending on model capabilities
- **HuggingFace**: Limited support for some models

## Basic Implementation

AbstractLLM provides two approaches to implementing tool calls: a simple approach and a more customizable approach.

### Simple Approach (Everything in One Place)

```python
from abstractllm import create_llm
from abstractllm.session import Session

# Define your tool function
def get_weather(location: str) -> dict:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        
    Returns:
        Dictionary containing weather information
    """
    # Implement actual weather lookup here
    return {
        "temperature": 72,
        "conditions": "sunny",
        "humidity": 45,
        "wind_speed": 10
    }

# Create a provider with a model that supports tool calls
provider = create_llm("openai", model="gpt-4o")

# Create a session with the tool
session = Session(
    system_prompt="You are a helpful assistant that can check the weather.",
    provider=provider,
    tools=[get_weather]  # Tool function is automatically registered
)

# Generate a response with tool support
response = session.generate_with_tools(
    prompt="What's the weather like in San Francisco?"
)

print(response.content)
```

In this approach, the tool function is:
1. Automatically converted to a tool definition based on type hints and docstrings
2. Registered with the session
3. Executed automatically when the LLM decides to call it

### Customizable Approach (Separate Definition and Execution)

```python
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.tools import function_to_tool_definition

# Define your tools
def read_file(file_path: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        The file contents as a string
    """
    # This is just the definition, not the implementation
    pass

# Initialize provider and session
provider = create_llm("anthropic", model="claude-3-opus-20240229")
session = Session(
    system_prompt="You are a helpful assistant that can read files when needed.",
    provider=provider
)

# Register tool definitions separately
session.add_tool(function_to_tool_definition(read_file))

# Create secure implementation with validation
def secure_file_read(file_path: str) -> str:
    """Enhanced implementation with security checks"""
    import os
    
    # Security checks
    allowed_directories = ["/allowed/path"]
    abs_path = os.path.abspath(os.path.normpath(file_path))
    if not any(os.path.commonpath([abs_path, allowed_dir]) == allowed_dir
              for allowed_dir in allowed_directories):
        return f"Error: Access to {file_path} is not allowed."
    
    # Read file with proper error handling
    try:
        with open(abs_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Generate with custom tool implementation
response = session.generate_with_tools(
    prompt="Read the file README.md",
    tool_functions={"read_file": secure_file_read}
)

print(response.content)
```

This approach provides:
1. Separation of tool definitions from implementations
2. Enhanced security through custom validation
3. More control over the execution flow

## Tool Definition

AbstractLLM uses JSON Schema to define tools. You can define tools in three ways:

### 1. From Python Functions

```python
def calculate_area(length: float, width: float) -> float:
    """
    Calculate the area of a rectangle.
    
    Args:
        length: The length of the rectangle in meters
        width: The width of the rectangle in meters
        
    Returns:
        The area in square meters
    """
    return length * width
```

This function is automatically converted to a tool definition using type hints and docstrings.

### 2. From Tool Definition Dictionaries

```python
tool_definition = {
    "name": "search_database",
    "description": "Search for records in a database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10
            }
        },
        "required": ["query"]
    }
}

session.add_tool(tool_definition)
```

### 3. Using Pydantic Models

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query")
    limit: int = Field(10, description="Maximum number of results to return")

def search_database(params: SearchQuery) -> list:
    """Search for records in a database"""
    # Implementation...
    pass
```

## Streaming with Tool Calls

AbstractLLM supports streaming with tool calls:

```python
# Generate with streaming and tools
for chunk in session.generate_with_tools_streaming(
    prompt="What's the weather in New York and San Francisco?",
    tool_functions={"get_weather": get_weather}
):
    if isinstance(chunk, str):
        # Content chunk
        print(chunk, end="", flush=True)
    else:
        # Tool call information
        print(f"\n[Tool Call: {chunk.tool_calls[0].name}]")
```

## Security Considerations

Tool execution can present security risks. AbstractLLM provides several security features:

### 1. Parameter Validation

Always validate parameters before execution:

```python
def read_file(file_path: str) -> str:
    """Read file contents."""
    import os
    # Validate path
    allowed_dirs = [os.path.abspath("./safe_files")]
    abs_path = os.path.abspath(file_path)
    if not any(os.path.commonpath([abs_path, d]) == d for d in allowed_dirs):
        return f"Error: Access to {file_path} is not allowed"
    
    # Rest of implementation...
```

### 2. Execution Timeouts

Set timeouts to prevent resource exhaustion:

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function execution timed out after {seconds} seconds")
    
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def long_running_tool(param: str) -> str:
    """Tool that might take a long time."""
    try:
        with timeout(5):  # 5-second timeout
            # Implementation...
            return result
    except TimeoutError as e:
        return f"Error: {str(e)}"
```

### 3. Secure Tool Wrappers

AbstractLLM provides utility functions for creating secure tool wrappers:

```python
from abstractllm.tools.validation import create_safe_tool_wrapper

# Get tool definition
tool_def = function_to_tool_definition(read_file)

# Create secure wrapper
secure_read_file = create_safe_tool_wrapper(read_file, tool_def)
```

## Advanced Usage

### Tool Call Response Processing

You can manually process tool calls for more control:

```python
# Generate a response with tools
response = session.generate_with_tools(
    prompt="What files are in the current directory?",
    tool_functions={"list_files": list_files}
)

# Check if response has tool calls
if response.has_tool_calls():
    print(f"Tool calls detected: {len(response.tool_calls.tool_calls)}")
    for tool_call in response.tool_calls.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")
```

### Custom Tool Validation

You can add custom validation for tool parameters:

```python
from abstractllm.tools.validation import validate_tool_arguments
from jsonschema.exceptions import ValidationError

def validate_path(tool_def, arguments):
    """Custom validation for file paths."""
    try:
        # Basic schema validation
        args = validate_tool_arguments(tool_def, arguments)
        
        # Custom validation
        import os
        file_path = args.get("file_path", "")
        if "../" in file_path or file_path.startswith("/"):
            raise ValidationError("Invalid file path")
        
        return args
    except ValidationError as e:
        raise ValueError(f"Validation error: {str(e)}")

# Use in session
response = session.generate_with_tools(
    prompt="Read the file config.txt",
    tool_functions={"read_file": read_file},
    validators={"read_file": validate_path}
)
```

### Multi-Step Tool Execution

For complex workflows involving multiple tool calls:

```python
def process_conversation(query):
    # Initial response
    response = session.generate_with_tools(
        prompt=query,
        tool_functions={"search": search, "get_details": get_details}
    )
    
    # Continue the conversation with tool results
    while response.has_tool_calls():
        tool_results = []
        for tool_call in response.tool_calls.tool_calls:
            if tool_call.name == "search":
                result = search(**tool_call.arguments)
            elif tool_call.name == "get_details":
                result = get_details(**tool_call.arguments)
            
            # Add result to the session
            session.add_tool_result(
                tool_call_id=tool_call.id,
                result=result,
                tool_name=tool_call.name
            )
        
        # Continue the conversation
        response = session.generate_with_tools(
            prompt=None,  # No new prompt, just continue with tool results
            tool_functions={"search": search, "get_details": get_details}
        )
    
    return response.content
```

## Provider-Specific Considerations

### OpenAI

OpenAI provides the most mature tool calling implementation:

```python
provider = create_llm("openai", model="gpt-4o")
# Support for multiple tool calls in a single response
# Parallel tool execution
```

### Anthropic

Anthropic uses a specific XML format:

```python
provider = create_llm("anthropic", model="claude-3-opus-20240229")
# Limited to one tool call at a time per response
# Tool results need special formatting
```

### Ollama

Ollama has more limited tool call support:

```python
provider = create_llm("ollama", model="llama3")
# Simplified JSON parsing
# May require multiple attempts for complex tool calls
```

### HuggingFace

HuggingFace has the most limited support:

```python
provider = create_llm("huggingface", model="google/gemma-2b")
# Requires specific models with function calling support
# Less reliable than commercial providers
```

## Common Use Cases

### Web Search

```python
def search_web(query: str, num_results: int = 5) -> list:
    """Search the web for information."""
    # Implementation...
    pass
```

### Database Interactions

```python
def query_database(sql_query: str) -> list:
    """Run a SQL query against the database."""
    # Implementation...
    pass
```

### API Calls

```python
def call_api(endpoint: str, method: str = "GET", params: dict = None) -> dict:
    """Make an API call."""
    # Implementation...
    pass
```

### File Operations

```python
def read_file(file_path: str) -> str:
    """Read a file."""
    # Implementation...
    pass

def write_file(file_path: str, content: str) -> bool:
    """Write to a file."""
    # Implementation...
    pass
```

## Limitations

1. **Provider Inconsistencies**
   - Different providers implement tool calls differently
   - Results and behavior may vary between providers
   - Error handling varies by provider

2. **Security Challenges**
   - Tool execution introduces security considerations
   - Parameter validation is critical
   - Timeouts and resource limits should be enforced

3. **Reliability Issues**
   - LLMs may format tool calls incorrectly
   - Parameter types may not match expectations
   - Error handling strategy is important

## Conclusion

AbstractLLM's tool call implementation offers a robust, secure, and flexible system for enabling LLMs to interact with external systems. By providing a unified interface across providers and focusing on security, it enables powerful applications while minimizing the risks associated with tool execution. 