# Phase 4: Session Implementation

This document provides detailed implementation guidance for extending the AbstractLLM session management layer to support tool calls.

## Overview

The session implementation phase focuses on enhancing the `Session` class and related components to:

1. Handle tool definitions throughout session lifecycle
2. Process tool calls in LLM responses
3. Manage tool execution and result handling
4. Maintain tool state and history in the session

## Implementation Steps

### 1. Update `abstractllm/session.py`

```python
from typing import Any, Callable, Dict, List, Optional, Union, cast

from .enums import Capability, ResponseFormat
from .providers.base import BaseProvider
from .tools import ToolCallRequest, ToolCall, ToolDefinition, function_to_tool_definition
from .types import GenerateResponse, Message, MessageRole

class Session:
    """Session class for managing conversation sessions with LLMs."""
    
    def __init__(
        self,
        provider: BaseProvider,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize a session.
        
        Args:
            provider: The LLM provider to use
            tools: A list of tool definitions available for the LLM to use
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider
        self.messages: List[Message] = []
        self.tools = tools or []
        self.config = kwargs
        
        # Validate that the provider supports tools if they're provided
        if tools and Capability.FUNCTION_CALLING not in provider.capabilities:
            raise ValueError(
                f"Provider {provider.__class__.__name__} does not support function calling"
            )
    
    def add_message(
        self, 
        role: MessageRole, 
        content: str, 
        name: Optional[str] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add a message to the session.
        
        Args:
            role: The role of the message sender
            content: The content of the message
            name: The name of the message sender
            tool_results: A list of tool results to include in the message
        """
        message: Message = {"role": role, "content": content}
        
        if name:
            message["name"] = name
            
        if tool_results:
            message["tool_results"] = tool_results
            
        self.messages.append(message)
    
    def add_tool(self, tool: Union[ToolDefinition, Callable[..., Any]]) -> None:
        """
        Add a tool to the session.
        
        Args:
            tool: The tool definition or function to add
        """
        if callable(tool):
            # Convert function to tool definition
            tool_def = function_to_tool_definition(tool)
        else:
            tool_def = tool
            
        self.tools.append(tool_def)
    
    def generate(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> GenerateResponse:
        """
        Generate a response and add it to the session.
        
        Args:
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate
            top_p: The top_p value to use
            frequency_penalty: The frequency penalty to use
            presence_penalty: The presence penalty to use
            response_format: The format of the response
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A GenerateResponse object containing the response
        """
        # Merge session config with request config
        request_config = {**self.config}
        request_config.update(kwargs)
        
        # Generate a response
        response = self.provider.generate(
            messages=self.messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            stream=stream,
            tools=self.tools,  # Include tools in the request
            **request_config
        )
        
        # Add the response to the session
        self.add_message(
            role="assistant",
            content=response.content or ""
        )
        
        return response
    
    def execute_tool_call(
        self,
        tool_call: ToolCall,
        tool_functions: Dict[str, Callable[..., Any]]
    ) -> Dict[str, Any]:
        """
        Execute a tool call using the provided functions.
        
        Args:
            tool_call: The tool call to execute
            tool_functions: A dictionary mapping tool names to their implementation functions
            
        Returns:
            A dictionary containing the tool result
            
        Raises:
            ValueError: If the tool is not found in the provided functions
        """
        import json
        
        # Get the tool function
        if tool_call.name not in tool_functions:
            raise ValueError(f"Tool function '{tool_call.name}' not found")
            
        tool_function = tool_functions[tool_call.name]
        
        # Parse the arguments
        arguments = json.loads(tool_call.arguments)
        
        # Execute the tool
        result = tool_function(**arguments)
        
        # Create the tool result
        tool_result = {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "output": str(result)  # Convert to string to ensure compatibility
        }
        
        return tool_result
    
    def execute_tool_calls(
        self,
        response: GenerateResponse,
        tool_functions: Dict[str, Callable[..., Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute all tool calls in a response and return the results.
        
        Args:
            response: The response containing tool calls
            tool_functions: A dictionary mapping tool names to their implementation functions
            
        Returns:
            A list of dictionaries containing the tool results
            
        Raises:
            ValueError: If the response does not contain tool calls
        """
        # Check if the response contains tool calls
        if not response.has_tool_calls():
            raise ValueError("Response does not contain tool calls")
            
        # Execute each tool call
        tool_results = []
        for tool_call in response.tool_calls.tool_calls:
            tool_result = self.execute_tool_call(tool_call, tool_functions)
            tool_results.append(tool_result)
            
        return tool_results
    
    def generate_with_tools(
        self,
        tool_functions: Dict[str, Callable[..., Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        **kwargs: Any
    ) -> GenerateResponse:
        """
        Generate a response, execute any tool calls, and continue the conversation.
        
        This method handles the complete flow of tool usage:
        1. Generate an initial response with tool definitions
        2. If the response contains tool calls, execute them
        3. Add the tool results to the conversation
        4. Generate a follow-up response that incorporates the tool results
        
        Args:
            tool_functions: A dictionary mapping tool names to their implementation functions
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate
            top_p: The top_p value to use
            frequency_penalty: The frequency penalty to use
            presence_penalty: The presence penalty to use
            response_format: The format of the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The final GenerateResponse after tool execution and follow-up
        """
        # Generate an initial response
        initial_response = self.generate(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            **kwargs
        )
        
        # Check if the response contains tool calls
        if not initial_response.has_tool_calls():
            return initial_response
            
        # Execute the tool calls
        tool_results = self.execute_tool_calls(initial_response, tool_functions)
        
        # Remove the last assistant message (will be replaced with the tool call message)
        self.messages.pop()
        
        # Add the assistant message with tool results
        self.add_message(
            role="assistant",
            content=initial_response.content or "",
            tool_results=tool_results
        )
        
        # Generate a follow-up response
        follow_up_response = self.generate(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            **kwargs
        )
        
        return follow_up_response
```

### 2. Add Helper Functions in `abstractllm/tools.py`

Enhance the tools module with helper functions for tool management:

```python
def register_tools(
    session: "Session", 
    tools: Dict[str, Callable[..., Any]]
) -> None:
    """
    Register a dictionary of tool functions with a session.
    
    Args:
        session: The session to register the tools with
        tools: A dictionary mapping tool names to their implementation functions
    """
    for tool_func in tools.values():
        session.add_tool(tool_func)

def execute_tools(
    response: GenerateResponse, 
    tools: Dict[str, Callable[..., Any]]
) -> List[Dict[str, Any]]:
    """
    Execute tools based on the tool calls in a response.
    
    A standalone helper function for executing tool calls outside a session.
    
    Args:
        response: The response containing tool calls
        tools: A dictionary mapping tool names to their implementation functions
        
    Returns:
        A list of tool results
        
    Raises:
        ValueError: If the response does not contain tool calls
    """
    from .session import Session
    
    # Create a temporary session to execute the tool calls
    temp_session = Session(response.provider)
    
    # Execute the tool calls
    return temp_session.execute_tool_calls(response, tools)
```

### 3. Tool Execution Flow

The general flow for tool execution using the session is:

1. Create a session with tool definitions
2. Add user messages to the session
3. Generate a response with tools
4. Check for tool calls in the response
5. If tool calls are present, execute them
6. Add the tool results back to the session
7. Generate a follow-up response with the results

### 4. Error Handling

Add robust error handling for tool execution:

```python
def execute_tool_call(
    self,
    tool_call: ToolCall,
    tool_functions: Dict[str, Callable[..., Any]]
) -> Dict[str, Any]:
    """
    Execute a tool call using the provided functions.
    
    Args:
        tool_call: The tool call to execute
        tool_functions: A dictionary mapping tool names to their implementation functions
        
    Returns:
        A dictionary containing the tool result
        
    Raises:
        ValueError: If the tool is not found in the provided functions
    """
    import json
    
    # Get the tool function
    if tool_call.name not in tool_functions:
        error_message = f"Tool function '{tool_call.name}' not found"
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "error": error_message,
            "output": f"Error: {error_message}"
        }
        
    tool_function = tool_functions[tool_call.name]
    
    try:
        # Parse the arguments
        arguments = json.loads(tool_call.arguments)
        
        # Execute the tool
        result = tool_function(**arguments)
        
        # Create the tool result
        tool_result = {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "output": str(result)  # Convert to string to ensure compatibility
        }
        
        return tool_result
    except json.JSONDecodeError as e:
        error_message = f"Failed to parse arguments: {str(e)}"
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "error": error_message,
            "output": f"Error: {error_message}"
        }
    except Exception as e:
        error_message = f"Tool execution failed: {str(e)}"
        return {
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "error": error_message,
            "output": f"Error: {error_message}"
        }
```

## Example Usage

### Basic Session with Tools

```python
from abstractllm import AbstractLLM, Session
from abstractllm.tools import function_to_tool_definition

# Define tool functions
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        A string describing the current weather
    """
    return f"The weather in {location} is 22°{unit[0].upper()}"

def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price for a symbol.
    
    Args:
        symbol: The stock symbol, e.g., "AAPL"
        
    Returns:
        A string describing the current stock price
    """
    return f"The current stock price of {symbol} is $150.75"

# Create an LLM provider
llm = AbstractLLM.create("openai", api_key="your-api-key")

# Create a session with tools
session = Session(
    provider=llm.provider,
    tools=[
        function_to_tool_definition(get_current_weather),
        function_to_tool_definition(get_stock_price)
    ]
)

# Add a user message
session.add_message(
    role="user", 
    content="What's the weather in London and the stock price for AAPL?"
)

# Define the tool functions dictionary
tool_functions = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price
}

# Generate a response with tools
response = session.generate_with_tools(
    tool_functions=tool_functions,
    model="gpt-4"
)

print(response.content)
```

### Manual Tool Processing

```python
from abstractllm import AbstractLLM, Session
import json

# Define tool functions
def get_current_weather(location: str, unit: str = "celsius") -> str:
    # Implementation...
    return f"The weather in {location} is 22°{unit[0].upper()}"

# Create an LLM provider
llm = AbstractLLM.create("openai", api_key="your-api-key")

# Create a session with tools
session = Session(
    provider=llm.provider,
    tools=[function_to_tool_definition(get_current_weather)]
)

# Add a user message
session.add_message(
    role="user", 
    content="What's the weather in London?"
)

# Generate a response
response = session.generate(model="gpt-4")

# Check if the response contains tool calls
if response.has_tool_calls():
    # Process the first tool call
    tool_call = response.tool_calls.tool_calls[0]
    print(f"Tool called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    
    # Parse the arguments
    args = json.loads(tool_call.arguments)
    
    # Execute the tool
    result = get_current_weather(**args)
    
    # Add the tool result to the session
    session.messages.pop()  # Remove the last assistant message
    session.add_message(
        role="assistant",
        content=response.content or "",
        tool_results=[{
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "output": result
        }]
    )
    
    # Generate a follow-up response
    follow_up_response = session.generate(model="gpt-4")
    print(follow_up_response.content)
else:
    print(response.content)
```

## Integration with Streaming

Tool calls during streaming require special handling:

```python
def generate_with_tools_streaming(
    self,
    tool_functions: Dict[str, Callable[..., Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[ResponseFormat] = None,
    **kwargs: Any
) -> Generator[Union[GenerateResponse, Dict[str, Any]], None, None]:
    """
    Generate a streaming response with tool support.
    
    This method yields either streaming chunks or tool executions.
    
    Args:
        tool_functions: A dictionary mapping tool names to their implementation functions
        model: The model to use
        temperature: The temperature to use
        max_tokens: The maximum number of tokens to generate
        top_p: The top_p value to use
        frequency_penalty: The frequency penalty to use
        presence_penalty: The presence penalty to use
        response_format: The format of the response
        **kwargs: Additional provider-specific parameters
        
    Yields:
        Either GenerateResponse objects for content chunks or dictionaries for tool results
    """
    # Track the accumulated response content
    accumulated_content = ""
    
    # Generate a streaming response
    for chunk in self.provider.generate(
        messages=self.messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
        stream=True,
        tools=self.tools,
        **kwargs
    ):
        # Check if this is a tool call chunk
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            # Execute each tool call
            for tool_call in chunk.tool_calls.tool_calls:
                tool_result = self.execute_tool_call(tool_call, tool_functions)
                yield {"type": "tool_result", "result": tool_result}
                
                # Add the tool result to the session
                if not hasattr(self, "_pending_tool_results"):
                    self._pending_tool_results = []
                self._pending_tool_results.append(tool_result)
        else:
            # Update accumulated content
            if chunk.content:
                accumulated_content += chunk.content
                
            # Yield the chunk
            yield chunk
    
    # After streaming is complete, add the final message to the session
    self.add_message(
        role="assistant",
        content=accumulated_content,
        tool_results=getattr(self, "_pending_tool_results", None)
    )
    
    # Clear pending tool results
    if hasattr(self, "_pending_tool_results"):
        delattr(self, "_pending_tool_results")
```

## Next Steps

After implementing tool support in the session layer, proceed to [Integration Testing](integration.md) to ensure all components work together correctly. 