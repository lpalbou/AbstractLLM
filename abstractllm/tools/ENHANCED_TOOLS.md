# Enhanced Tool System for AbstractLLM

## Overview

The enhanced tool system provides SOTA (State-of-the-Art) tool creation capabilities that match or exceed frameworks like LangChain, Instructor, and Pydantic AI. It builds upon the existing simple `@register` decorator with a powerful `@tool` decorator that includes:

- **Pydantic Validation**: Automatic input/output validation with detailed error messages
- **Rich Metadata**: Tags, examples, usage hints for better LLM understanding  
- **Docstring Parsing**: Automatic extraction of parameter descriptions from docstrings
- **Retry Logic**: Automatic retry on validation errors with LLM-friendly error messages
- **Timeout Support**: Execution timeouts to prevent hanging
- **Context Injection**: Pass session context to tools that need it
- **Deprecation Support**: Mark tools as deprecated with custom messages
- **Response Validation**: Validate tool outputs with Pydantic models

## Installation

The enhanced tools use Pydantic, which is already a core dependency:

```bash
pip install abstractllm
```

For full features including docstring parsing:

```bash
pip install abstractllm[tools]  # Includes docstring-parser
```

## Quick Start

### Basic Tool with Validation

```python
from abstractllm.tools import tool
from pydantic import Field

@tool
def add_numbers(
    a: int = Field(description="First number", ge=0),
    b: int = Field(description="Second number", ge=0)
) -> int:
    """Add two positive numbers."""
    return a + b
```

### Advanced Tool with All Features

```python
from typing import List
from abstractllm.tools import tool, ToolContext
from pydantic import Field, BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    relevance: float = Field(ge=0.0, le=1.0)

@tool(
    parse_docstring=True,      # Parse docstring for descriptions
    retry_on_error=True,        # Allow retry on validation errors
    max_retries=3,              # Maximum retry attempts
    timeout=30.0,               # Execution timeout in seconds
    tags=["search", "web"],     # Categorization tags
    when_to_use="When user needs current web information",
    examples=[                  # Example calls for LLM guidance
        {"query": "Python tutorial", "max_results": 5}
    ]
)
def search_web(
    query: str = Field(
        description="Search query",
        min_length=1,
        max_length=500
    ),
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results (1-100)"
    )
) -> List[SearchResult]:
    """
    Search the web for information.
    
    Args:
        query: The search query to execute
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with titles and URLs
    """
    # Implementation here
    return [SearchResult(title="Example", url="http://example.com", relevance=0.9)]
```

## Feature Comparison

| Feature | `@register` (Basic) | `@tool` (Enhanced) |
|---------|---------------------|-------------------|
| Function wrapping | ✅ | ✅ |
| Auto-registration | ✅ | ✅ |
| Type hints | ✅ | ✅ |
| JSON Schema | ✅ | ✅ |
| **Pydantic validation** | ❌ | ✅ |
| **Docstring parsing** | ❌ | ✅ |
| **Retry on error** | ❌ | ✅ |
| **Timeout support** | ❌ | ✅ |
| **Context injection** | ❌ | ✅ |
| **Response validation** | ❌ | ✅ |
| **Deprecation warnings** | ❌ | ✅ |
| **Usage examples** | ❌ | ✅ |
| **When-to-use hints** | ❌ | ✅ |

## Key Features

### 1. Pydantic Validation

Automatically validates inputs and provides LLM-friendly error messages:

```python
@tool
def divide(
    numerator: float = Field(description="The numerator"),
    denominator: float = Field(description="The denominator", ne=0.0)
) -> float:
    return numerator / denominator

# Invalid call: denominator is 0
# Returns: {"_error": True, "_retry": True, "_message": "denominator: Input should not be equal to 0"}
```

### 2. Docstring Parsing

Supports Google, NumPy, and Sphinx style docstrings:

```python
@tool(parse_docstring=True)
def process_data(input_file: str, output_format: str = "json") -> dict:
    """
    Process data from a file.
    
    Args:
        input_file: Path to the input file
        output_format: Output format (json, csv, xml)
        
    Returns:
        Processed data as dictionary
    """
    # Descriptions are automatically extracted
```

### 3. Context Injection

Tools can access session context:

```python
@tool(requires_context=True)
def get_user_history(
    query: str,
    context: ToolContext = None  # Automatically injected
) -> list:
    """Get user's interaction history."""
    if context.memory:
        return context.memory.search(query)
    return []
```

### 4. Timeout and Retry

Prevent hanging and handle transient failures:

```python
@tool(
    timeout=10.0,        # 10 second timeout
    retry_on_error=True,
    max_retries=3
)
def fetch_data(url: str) -> dict:
    """Fetch data from URL with timeout."""
    # Long-running operation
    return {"data": "..."}
```

### 5. Deprecation Support

Mark tools as deprecated:

```python
@tool(
    deprecated=True,
    deprecation_message="Use fetch_data_v2 instead",
    version="1.0.0"
)
def fetch_data_old(url: str) -> dict:
    """Old data fetching function."""
    # Shows deprecation warning when used
```

### 6. Response Validation

Validate tool outputs:

```python
class WeatherInfo(BaseModel):
    temperature: float = Field(ge=-100, le=100)
    humidity: int = Field(ge=0, le=100)

@tool(response_model=WeatherInfo)
def get_weather(city: str) -> WeatherInfo:
    """Get weather information."""
    return WeatherInfo(temperature=22.5, humidity=65)
```

## Migration from @register

The system is fully backward compatible:

```python
# Old way (still works)
from abstractllm.tools import register

@register
def my_tool(param: str) -> str:
    return f"Result: {param}"

# New way (with enhanced features)
from abstractllm.tools import tool
from pydantic import Field

@tool(retry_on_error=True)
def my_tool(
    param: str = Field(description="Input parameter", min_length=1)
) -> str:
    return f"Result: {param}"
```

## Tool Choice Strategies

Control how LLMs select tools:

```python
from abstractllm.tools import ToolChoice

# Let model decide
response = session.generate(prompt, tool_choice=ToolChoice.AUTO)

# No tools
response = session.generate(prompt, tool_choice=ToolChoice.NONE)

# Must use at least one tool
response = session.generate(prompt, tool_choice=ToolChoice.REQUIRED)

# Use specific tool only
response = session.generate(
    prompt,
    tool_choice=ToolChoice.SPECIFIC,
    specific_tools=["search_web"]
)
```

## Best Practices

1. **Use Descriptive Names**: Tool names should clearly indicate their function
2. **Provide Rich Descriptions**: Help LLMs understand when to use each tool
3. **Add Constraints**: Use Field validators to prevent invalid inputs
4. **Include Examples**: Show the LLM how to call your tool correctly
5. **Handle Errors Gracefully**: Return structured errors for LLM retry
6. **Set Appropriate Timeouts**: Prevent long-running operations from blocking
7. **Version Your Tools**: Use semantic versioning for tool evolution

## Advanced Patterns

### Streaming Tools

```python
@tool(stream_response=True)
async def stream_data(query: str) -> AsyncGenerator[str, None]:
    """Stream results as they arrive."""
    for chunk in fetch_chunks(query):
        yield chunk
```

### Confirmation Required

```python
@tool(requires_confirmation=True)
def delete_file(path: str) -> bool:
    """Delete a file (requires user confirmation)."""
    os.remove(path)
    return True
```

### Complex Validation

```python
from pydantic import field_validator

class EmailRequest(BaseModel):
    to: str
    subject: str
    
    @field_validator('to')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

@tool
def send_email(request: EmailRequest) -> dict:
    """Send an email."""
    # Implementation
```

## Testing Tools

Test your tools without an LLM:

```python
from abstractllm.tools import get_registry, ToolCall

# Direct call
result = my_tool(param="test")

# Through registry
registry = get_registry()
tool_call = ToolCall(name="my_tool", arguments={"param": "test"})
result = registry.execute(tool_call)

# Validate arguments
tool_def = my_tool.tool_definition
try:
    validated = tool_def.validate_arguments({"param": ""})
except ToolValidationError as e:
    print(f"Validation failed: {e.errors}")
```

## Performance Considerations

- **Validation Overhead**: Pydantic validation adds ~1-5ms per call
- **Docstring Parsing**: Cached after first use
- **Registry Lookup**: O(1) dictionary lookup
- **Timeout Threads**: Uses ThreadPoolExecutor for timeout enforcement

## Future Enhancements

Planned features for future releases:

- [ ] Async tool support with native coroutines
- [ ] Tool composition and chaining
- [ ] Conditional tool availability
- [ ] Tool usage analytics and monitoring
- [ ] Caching for expensive operations
- [ ] Rate limiting per tool
- [ ] Tool sandboxing for security
- [ ] Visual tool builder UI

## Conclusion

The enhanced tool system provides production-ready tool creation with minimal boilerplate while maintaining full backward compatibility. It combines the best practices from leading frameworks while preserving AbstractLLM's philosophy of simplicity and universality.