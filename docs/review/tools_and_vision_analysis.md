# Tool Call and Vision Capabilities in AbstractLLM

## Tool Call Capabilities

### Overview

AbstractLLM implements a unified tool call interface across supported providers (primarily OpenAI, Anthropic, and Ollama). The design follows an "LLM-First Architecture" where all tool calls are initiated by the LLM based on its reasoning, never by pattern matching on user input.

### Core Architecture

The fundamental principle of AbstractLLM's tool calling system is the "LLM-First Architecture," which follows this flow:

```
User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User
```

This architecture ensures:
1. The LLM—not the code—decides when and which tools to call
2. Tool selection is based on reasoning, not pattern matching
3. All user requests flow through the LLM
4. Tool results are processed by the LLM before returning to the user

### Implementation Approaches

AbstractLLM provides two approaches to implementing tool calls:

#### 1. Simple Approach (Everything in One Place)

```python
from abstractllm import create_llm
from abstractllm.session import Session

# Define your tool function
def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Create provider and session, passing tool directly
provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")
session = Session(
    system_prompt="You are a helpful assistant that can read files when needed.",
    provider=provider,
    tools=[read_file]  # Tool function is automatically registered
)

# Generate response with tool support
response = session.generate_with_tools(
    prompt="What is in the file README.md?"
)
```

This approach is recommended when you want the cleanest, most straightforward code and don't need custom tool handling or complex execution logic.

#### 2. Customizable Approach (Separate Definition and Execution)

```python
from abstractllm import create_llm
from abstractllm.session import Session
from abstractllm.tools import function_to_tool_definition

# Define tool functions
def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Initialize provider and session
provider = create_llm("anthropic", model="claude-3-5-haiku-20241022")
session = Session(
    system_prompt="You are a helpful assistant that can use tools when needed.",
    provider=provider
)

# Register tool definitions separately
session.add_tool(function_to_tool_definition(read_file))

# Create custom secure implementation
def secure_file_read(file_path: str) -> str:
    """Enhanced implementation with security checks"""
    import os
    # Security checks and implementation...
    pass

# Generate with custom tool implementation
response = session.generate_with_tools(
    prompt="Read the file README.md",
    tool_functions={"read_file": secure_file_read}
)
```

This approach is recommended for more control over tool execution, such as enhanced security, custom logging, or specialized error handling.

### Internal Tool Definition Model

AbstractLLM adopts the Anthropic tool definition structure as its internal standard representation for tools. This was chosen based on its relative simplicity, use of standard JSON Schema, and ease of convertibility to other provider formats.

### Security Considerations

The documentation heavily emphasizes security for tool execution:

1. **Path Validation**: All file operations are protected by path validation
   ```python
   def is_safe_path(file_path: str, allowed_directories: List[str]) -> bool:
       """Check if a file path is within allowed directories."""
       abs_path = os.path.abspath(os.path.normpath(file_path))
       return any(os.path.commonpath([abs_path, allowed_dir]) == allowed_dir
                 for allowed_dir in allowed_directories)
   ```

2. **Tool Parameter Validation**: All parameters are validated before execution
3. **Execution Timeouts**: All tool executions have timeouts to prevent resource exhaustion
4. **Secure Tool Wrappers**: Tools are wrapped with security measures
   ```python
   def create_secure_tool_wrapper(func: Callable, max_execution_time: int = 5) -> Callable:
       """Create a wrapper that adds security measures to any tool function."""
       @functools.wraps(func)
       def secure_wrapper(*args, **kwargs):
           # Validate parameters
           # Execute with timeout
           # Sanitize results
       return secure_wrapper
   ```
5. **Output Sanitization**: Tool outputs are sanitized to limit size and redact sensitive information
6. **Comprehensive Logging**: All tool executions are logged for auditing

### Anti-Patterns to Avoid

The documentation explicitly warns against these patterns:

```python
# 🚨 DANGER: Direct tool execution
if "file" in query.lower() and "read" in query.lower():
    # Extract filename and read directly

# 🚨 DANGER: Direct tool result return
tool_result = execute_tool(tool_call)
return f"The tool returned: {tool_result}"

# 🚨 DANGER: Bypassing LLM processing
if should_use_tool(query):
    # Special handling path
else:
    # Normal LLM path
```

### Implementation Status

As of version 0.5.3, tool call support is available for compatible models across providers:

- **OpenAI**: Full tool call support
- **Anthropic**: Tool call support for Claude models
- **Ollama**: Partial tool call support depending on model capabilities

Tool call dependencies are required for this functionality to work:
```bash
pip install "abstractllm[tools]"
```

## Vision Capabilities

### Overview

AbstractLLM provides vision capabilities for multimodal models, allowing the processing of images alongside text. The implementation follows a provider-agnostic approach with provider-specific optimizations.

### Supported Models

Vision capabilities are available across these providers:

- **OpenAI**: `gpt-4-vision-preview`, `gpt-4-turbo`, `gpt-4o`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-3.5-sonnet`, `claude-3.5-haiku`
- **Ollama**: `llama3.2-vision`, `deepseek-janus-pro`
- **HuggingFace**: Various vision models with provider-specific handling

### Implementation Architecture

The vision implementation consists of these key components:

1. **MediaInput Interface**: Abstract base class for all media types
2. **ImageInput Implementation**: Handles multiple input formats
3. **MediaProcessor**: Processes messages with media inputs

The data flow follows this pattern:
```
Input Source → MediaFactory → ImageInput → MediaProcessor → Provider Format → Provider API
```

### Input Format Support

The implementation handles multiple input formats:

1. **File Path**: Local filesystem path to an image
2. **URL**: HTTP/HTTPS URL to an image
3. **Base64**: Raw base64 string of image data
4. **Data URL**: base64 with MIME type prefix

Each input format is processed and converted to the appropriate provider-specific format:

- **OpenAI**: Image URL structure with detail level
- **Anthropic**: Base64 or URL with specific JSON structure
- **Ollama**: URL or base64 string
- **HuggingFace**: Path, URL, or binary content depending on model

### Usage Examples

Basic usage with a single image:
```python
from abstractllm import create_llm, ModelCapability

# Create an LLM instance with a vision-capable model
llm = create_llm("openai", model="gpt-4o")

# Check if vision is supported
capabilities = llm.get_capabilities()
if capabilities.get(ModelCapability.VISION):
    # Use vision capabilities
    image_url = "https://example.com/image.jpg"
    response = llm.generate("What's in this image?", image=image_url)
    print(response)
```

Using local image files:
```python
local_image = "/path/to/image.jpg"
response = llm.generate("Describe this image", image=local_image)
```

Multiple images:
```python
images = ["https://example.com/image1.jpg", "/path/to/image2.jpg"]
response = llm.generate("Compare these images", images=images)
```

### Implementation Details

#### Format Caching

- Provider-specific formats are cached in `_cached_formats`
- Cache key: provider name
- Cache value: formatted data structure
- No binary content caching for memory efficiency

#### MIME Type Detection

1. Constructor-provided type
2. Data URL extraction
3. File extension mapping
4. URL extension analysis
5. Default to 'image/jpeg'

#### Error Handling

- Custom `ImageProcessingError` with provider context
- Size validation (e.g., Anthropic's 100MB limit)
- Format validation
- Network error handling
- File access error handling

### Best Practices

The documentation recommends these best practices for vision capabilities:

1. **Input Handling**:
   - Early validation
   - Format preservation
   - Efficient conversion

2. **Provider Compatibility**:
   - Check format requirements
   - Verify size limitations
   - Validate capability support

3. **Error Management**:
   - Clear error messages
   - Provider context
   - Graceful fallbacks

4. **Performance**:
   - Format caching
   - Lazy loading
   - Appropriate detail levels

### Implementation Status

Vision capabilities are fully implemented for OpenAI and Anthropic with good support for Ollama. HuggingFace vision support is marked as in progress.

## Integration Between Tools and Vision

The documentation indicates that tool calls and vision capabilities can be used together in models that support both features:

```python
from abstractllm import create_llm
from abstractllm.session import Session

# Define a tool that processes images
def analyze_image(image_path: str) -> str:
    """Analyze an image and return information about it."""
    # Implementation...
    pass

# Create provider with vision-capable model
provider = create_llm("openai", model="gpt-4o")

# Create session with both vision support and tools
session = Session(
    system_prompt="You can analyze images and use tools.",
    provider=provider,
    tools=[analyze_image]
)

# Generate response with both image and tool support
response = session.generate_with_tools(
    prompt="What's in this image and can you analyze it?",
    image="/path/to/image.jpg"
)
```

However, the documentation notes that not all models support both capabilities simultaneously, and developers should check model capabilities before using these features together. 