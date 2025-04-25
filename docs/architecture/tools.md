# Tool System

This document explains the architecture of AbstractLLM's tool calling system, which provides a unified interface for function/tool calling capabilities across different providers.

## Overview

The tool system in AbstractLLM enables LLMs to call external functions, providing a way to extend language models with custom capabilities. The system handles the conversion between AbstractLLM's unified tool format and provider-specific formats.

## Core Components

### ToolDefinition

The `ToolDefinition` class represents a tool that can be called by an LLM:

```python
# From abstractllm/tools/types.py
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union, Callable

class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by an LLM."""
    
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for inputs")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for return value")
    
    @classmethod
    def from_function(cls, func: Callable) -> "ToolDefinition":
        """Create a ToolDefinition from a Python function."""
        # Extract function metadata
        name = func.__name__
        description = func.__doc__ or f"Function {name}"
        
        # Build input schema from function signature
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Parse function signature
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                input_schema["properties"][param_name] = cls._type_to_schema(param.annotation)
                
            if param.default == inspect.Parameter.empty:
                input_schema["required"].append(param_name)
        
        # Build output schema if return type is annotated
        output_schema = None
        if sig.return_annotation != inspect.Signature.empty:
            output_schema = cls._type_to_schema(sig.return_annotation)
        
        return cls(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema
        )
```

### ToolCall

The `ToolCall` class represents a call to a tool made by the LLM:

```python
# From abstractllm/tools/types.py
class ToolCall(BaseModel):
    """A call to a tool made by the LLM."""
    
    id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")
    result: Optional[Any] = Field(None, description="Result of the tool call")
    
    def set_result(self, result: Any) -> None:
        """Set the result of the tool call."""
        self.result = result
```

### ToolCallRequest

The `ToolCallRequest` class represents a request from the LLM to call one or more tools:

```python
# From abstractllm/tools/types.py
class ToolCallRequest(BaseModel):
    """A request from the LLM to call one or more tools."""
    
    content: str = Field(..., description="Content from the LLM before tool calls")
    tool_calls: List[ToolCall] = Field(..., description="List of tool calls")
    
    def execute_tool_calls(self, tool_functions: Dict[str, Callable]) -> None:
        """Execute the tool calls with the provided functions."""
        for tool_call in self.tool_calls:
            if tool_call.name in tool_functions:
                try:
                    result = tool_functions[tool_call.name](**tool_call.arguments)
                    tool_call.set_result(result)
                except Exception as e:
                    tool_call.set_result(f"Error executing tool: {str(e)}")
            else:
                tool_call.set_result(f"Tool '{tool_call.name}' not found")
```

## Tool Conversion System

The tool conversion system handles the conversion between AbstractLLM's unified tool format and provider-specific formats:

```python
# From abstractllm/tools/conversion.py
def convert_to_openai_tools(tools: List[Union[Dict[str, Any], Callable, ToolDefinition]]) -> List[Dict[str, Any]]:
    """Convert tools to OpenAI format."""
    openai_tools = []
    
    for tool in tools:
        if isinstance(tool, dict):
            # Already in OpenAI format
            openai_tools.append(tool)
        elif isinstance(tool, ToolDefinition):
            # Convert ToolDefinition to OpenAI format
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        elif callable(tool):
            # Convert function to ToolDefinition, then to OpenAI format
            tool_def = ToolDefinition.from_function(tool)
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": tool_def.input_schema
                }
            })
    
    return openai_tools
```

Similar conversion functions exist for each provider.

## Tool Execution System

The tool execution system handles the execution of tools:

```python
# From abstractllm/tools/execution.py
def execute_tools(tool_calls: List[ToolCall], tool_functions: Dict[str, Callable]) -> List[Dict[str, Any]]:
    """Execute tool calls with the provided functions."""
    results = []
    
    for tool_call in tool_calls:
        if tool_call.name in tool_functions:
            try:
                # Execute tool with arguments
                result = tool_functions[tool_call.name](**tool_call.arguments)
                
                # Store result
                tool_call.set_result(result)
                
                # Add to results
                results.append({
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "result": result
                })
            except Exception as e:
                # Handle execution errors
                error_message = f"Error executing tool: {str(e)}"
                tool_call.set_result(error_message)
                results.append({
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "result": error_message
                })
        else:
            # Handle missing tools
            error_message = f"Tool '{tool_call.name}' not found"
            tool_call.set_result(error_message)
            results.append({
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "result": error_message
            })
    
    return results
```

## Provider-Specific Handling

Each provider has different requirements for tool calling:

### OpenAI

OpenAI uses a function calling approach:

```python
# From abstractllm/providers/openai.py
def _prepare_tools(self, tools):
    """Convert tools to OpenAI format."""
    if not tools:
        return None
    
    return convert_to_openai_tools(tools)

def _handle_tool_calls(self, response):
    """Handle tool calls from OpenAI response."""
    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.content
    
    # Extract tool calls
    content = response.choices[0].message.content or ""
    tool_calls = []
    
    for tool_call in response.choices[0].message.tool_calls:
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            arguments = {"error": "Failed to parse arguments"}
        
        tool_calls.append(ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=arguments
        ))
    
    return ToolCallRequest(content=content, tool_calls=tool_calls)
```

### Anthropic

Claude uses a different approach called "Tool Use":

```python
# From abstractllm/providers/anthropic.py
def _prepare_tools(self, tools):
    """Convert tools to Anthropic format."""
    if not tools:
        return None
    
    return convert_to_anthropic_tools(tools)

def _handle_tool_use(self, response):
    """Handle tool use from Anthropic response."""
    if "tool_use" not in response.content[0].text:
        return response.content[0].text
    
    # Extract tool use blocks
    content = response.content[0].text
    tool_use_pattern = r'<tool_use>(.*?)</tool_use>'
    tool_use_blocks = re.findall(tool_use_pattern, content, re.DOTALL)
    
    if not tool_use_blocks:
        return content
    
    # Remove tool use blocks from content
    clean_content = re.sub(tool_use_pattern, '', content, flags=re.DOTALL)
    
    # Extract tool calls
    tool_calls = []
    for i, block in enumerate(tool_use_blocks):
        try:
            tool_data = json.loads(block)
            tool_calls.append(ToolCall(
                id=f"call_{i}",
                name=tool_data.get("name"),
                arguments=tool_data.get("input", {})
            ))
        except json.JSONDecodeError:
            continue
    
    return ToolCallRequest(content=clean_content, tool_calls=tool_calls)
```

## Tool Schema Generation

The tool system automatically generates JSON Schema for Python functions:

```python
# From abstractllm/tools/schema.py
def generate_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Generate JSON Schema for a Python function."""
    # Extract function metadata
    name = func.__name__
    description = func.__doc__ or f"Function {name}"
    
    # Build input schema from function signature
    input_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    # Parse function signature
    sig = inspect.signature(func)
    for param_name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            input_schema["properties"][param_name] = type_to_schema(param.annotation)
            
        if param.default == inspect.Parameter.empty:
            input_schema["required"].append(param_name)
    
    return input_schema
```

## Usage Patterns

### Using Functions as Tools

```python
# Define a function to use as a tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    # Implementation...
    return f"The weather in {location} is sunny and 25°{unit}"

# Use the function as a tool
llm = create_llm("openai", model="gpt-4")
response = llm.generate(
    "What's the weather in Paris?",
    tools=[get_weather]
)
```

### Using ToolDefinition

```python
# Define a tool explicitly
weather_tool = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
)

# Use the tool definition
llm = create_llm("anthropic", model="claude-3-opus-20240229")
response = llm.generate(
    "What's the weather in Paris?",
    tools=[weather_tool]
)
```

### Handling Tool Calls in Sessions

```python
# Create a session with tools
session = Session(
    provider=llm,
    system_prompt="You are a helpful assistant that can use tools.",
    tools=[get_weather, get_time]
)

# Generate with tool execution
response = session.generate_with_tools(
    "What's the weather in Paris and what time is it there?"
)
```

## Security Considerations

The tool system includes several security features:

1. **Input Validation**: Tools validate their inputs against JSON Schema
2. **Timeouts**: Tool execution can be configured with timeouts
3. **Exception Handling**: Tool exceptions are caught and reported
4. **Sandboxing**: Tools can be executed in a sandbox for additional security

Example of secure tool execution:

```python
# From abstractllm/tools/security.py
def execute_tool_securely(func: Callable, arguments: Dict[str, Any], 
                        timeout: int = 5, max_output_size: int = 10000) -> Any:
    """Execute a tool securely with timeout and output limits."""
    # Validate arguments against schema
    schema = generate_schema_from_function(func)
    validate_against_schema(arguments, schema)
    
    # Execute with timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, **arguments)
        try:
            result = future.result(timeout=timeout)
        except TimeoutError:
            raise ToolExecutionError(f"Tool execution timed out after {timeout} seconds")
    
    # Limit output size
    if isinstance(result, str) and len(result) > max_output_size:
        result = result[:max_output_size] + "... [output truncated]"
    
    return result
```

## Next Steps

- [User Guide: Tool Calls](../user-guide/tools.md): How to use tools in your applications
- [Configuration System](configuration.md): How configuration is managed
- [Error Handling](error-handling.md): How errors are processed and reported 