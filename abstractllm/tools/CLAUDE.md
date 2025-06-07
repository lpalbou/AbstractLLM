# Tools Component

## Overview
The tools module provides a sophisticated system for enabling LLMs to call functions and interact with external systems. It handles tool definition, validation, conversion, and execution across multiple provider formats while maintaining type safety and security.

## Code Quality Assessment
**Rating: 9/10**

### Strengths
- Excellent type safety with Pydantic models
- Comprehensive validation using JSON Schema
- Smart docstring parsing for automatic tool creation
- Security-focused with execution sandboxing
- Provider-agnostic design with format conversion
- Extensive library of common tools
- Well-documented with clear examples

### Issues
- Complex regex parsing could be simplified
- Some functions are quite large (50+ lines)
- Architecture detection has some duplication
- Missing visual architecture documentation

## Component Mindmap
```
Tools System
├── Core Types (types.py)
│   ├── ToolDefinition (Pydantic model)
│   │   ├── name, description
│   │   ├── parameters (JSON Schema)
│   │   └── function reference
│   │
│   ├── ToolCall & ToolCallRequest
│   │   ├── Tool invocation details
│   │   └── Parameter validation
│   │
│   └── ToolResult
│       ├── Success/error status
│       └── Result data or error info
│
├── Validation (validation.py)
│   ├── JSON Schema validation
│   ├── Custom error types
│   │   ├── ToolNotFoundError
│   │   ├── InvalidToolParametersError
│   │   └── ToolExecutionError
│   └── Security checks
│
├── Conversion (conversion.py)
│   ├── Function → ToolDefinition
│   ├── Docstring parsing
│   │   ├── Google style
│   │   ├── NumPy style
│   │   └── REST style
│   ├── Type mapping
│   └── Provider format conversion
│
├── Architecture Tools (architecture_tools.py)
│   ├── Tool call detection
│   │   ├── XML format (<tool_call>)
│   │   ├── JSON format
│   │   └── Special tokens
│   ├── Response parsing
│   └── Format generation
│
├── Common Tools (common_tools.py)
│   ├── File Operations
│   │   ├── read_file, write_file
│   │   ├── list_directory
│   │   └── file_stats
│   ├── Data Processing
│   │   ├── parse_json, parse_csv
│   │   └── Data analysis
│   ├── Search & Query
│   │   ├── search_web
│   │   └── query_database
│   ├── System Info
│   │   └── get_system_info
│   └── 20+ more tools
│
└── Modular Prompts (modular_prompts.py)
    ├── Context-aware prompting
    ├── Model-specific formatting
    ├── Tool instruction injection
    └── Response format guidance
```

## Design Patterns
1. **Builder Pattern**: ToolDefinition construction from functions
2. **Strategy Pattern**: Different parsing strategies for docstrings
3. **Adapter Pattern**: Provider format conversion
4. **Validation Pattern**: Schema-based parameter validation
5. **Factory Pattern**: Tool creation from various sources

## Tool Definition Flow
```
1. Function Definition
   def my_function(param: str) -> dict:
       """Description..."""
       
2. Automatic Conversion
   tool = function_to_tool(my_function)
   
3. Validation
   - Parameter types checked
   - Schema generated
   - Security validated
   
4. Provider Formatting
   - OpenAI: {"type": "function", "function": {...}}
   - Anthropic: {"name": ..., "input_schema": {...}}
   - Others: Architecture-specific
```

## Security Features
- **Sandboxed Execution**: Tools run in controlled environment
- **Parameter Validation**: All inputs validated against schema
- **Error Isolation**: Exceptions caught and wrapped
- **No Code Injection**: String inputs sanitized
- **Capability Limits**: Tools can declare required permissions

## Common Tools Library
```python
# File Operations
read_file(filepath: str, encoding: str = "utf-8")
write_file(filepath: str, content: str, mode: str = "w")
list_directory(path: str = ".")

# Data Processing
parse_json(json_string: str)
parse_csv(csv_string: str, delimiter: str = ",")

# Search and Query
search_web(query: str, max_results: int = 5)
get_current_datetime(timezone: str = "UTC")

# System Information
get_system_info()
execute_python_code(code: str)  # Sandboxed
```

## Provider Integration
- **OpenAI/Anthropic**: Native tool/function calling
- **Ollama/MLX**: Architecture-based detection
- **HuggingFace**: Prompt engineering approach

## Dependencies
- **Required**: 
  - `pydantic`: Type definitions
  - `jsonschema`: Validation
- **Optional**:
  - `docstring-parser`: Advanced parsing

## Recommendations
1. **Simplify parsing**: Consider using AST over regex
2. **Split large functions**: Refactor 50+ line functions
3. **Add tool registry**: Central tool management
4. **Implement rate limiting**: For external API tools
5. **Add tool composition**: Chain multiple tools

## Technical Debt
- Regex complexity in docstring parsing
- Some duplication in architecture detection
- Missing async tool support
- No tool versioning system
- Limited tool dependency management

## Performance Notes
- Validation cached per tool definition
- Lazy loading of optional parsers
- Efficient JSON Schema validation
- Could benefit from compiled patterns

## Future Enhancements
1. **Visual tool builder**: GUI for creating tools
2. **Tool marketplace**: Share tools between projects
3. **Tool composition**: Build complex tools from simple ones
4. **Async tools**: Support for async function calls
5. **Tool monitoring**: Usage analytics and debugging

## Usage Example
```python
from abstractllm.tools import function_to_tool
from abstractllm import Session

# Define a tool
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather for a location.
    
    Args:
        location: City name or coordinates
        units: Temperature units (celsius/fahrenheit)
        
    Returns:
        Weather data dictionary
    """
    # Implementation...
    return {"temp": 22, "conditions": "sunny"}

# Use in session
session = Session(provider=llm, tools=[get_weather])
response = session.generate("What's the weather in Paris?")
# LLM calls get_weather("Paris") automatically
```