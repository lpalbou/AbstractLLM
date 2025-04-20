# Phase 3: Provider Implementations

This document provides detailed implementation guidance for updating specific providers in AbstractLLM to support tool calls.

## Overview

The provider implementation phase focuses on extending individual LLM providers to support tool functionality:

1. Implement tool call support in supported providers
2. Add provider-specific conversion logic
3. Handle diverse response formats
4. Update capability reporting

## Provider Priority

Implement tool support in the following order:

1. **OpenAI** - Has robust tool/function calling support
2. **Anthropic** - Supports tool use in recent models
3. **Ollama** - Supports tools in some model configurations
4. **Other providers** - Based on their support and user demand

## OpenAI Provider Implementation

### 1. Update `abstractllm/providers/openai.py`

```python
from typing import Any, Dict, List, Optional, Union

from ..enums import Capability, ResponseFormat
from ..providers.base import BaseProvider
from ..tools import ToolCallRequest, ToolCall, ToolDefinition
from ..types import GenerateResponse, Message

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""
    
    @property
    def capabilities(self) -> List[Capability]:
        """Return a list of capabilities supported by this provider."""
        capabilities = [Capability.CHAT]
        
        # Check if the configured API key and model support vision
        if self._supports_vision():
            capabilities.append(Capability.IMAGE_UNDERSTANDING)
        
        # Add function calling capability - OpenAI supports this widely
        capabilities.append(Capability.FUNCTION_CALLING)
        
        return capabilities
    
    def _convert_tool_definitions(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """
        Convert AbstractLLM tool definitions to OpenAI function format.
        
        Args:
            tools: List of AbstractLLM tool definitions
            
        Returns:
            List of OpenAI-formatted function definitions
        """
        openai_tools = []
        
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            openai_tools.append(openai_tool)
            
        return openai_tools
    
    def _extract_tool_calls(self, response: Any) -> Optional[ToolCallRequest]:
        """
        Extract tool calls from an OpenAI response.
        
        Args:
            response: Raw OpenAI response
            
        Returns:
            ToolCallRequest object if tool calls are present, None otherwise
        """
        # Check if the response contains tool calls
        if not response.choices or not response.choices[0].message.tool_calls:
            return None
            
        # Extract tool calls from the response
        tool_calls = []
        for tool_call in response.choices[0].message.tool_calls:
            # OpenAI tool calls have an id, function name, and arguments
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                    type="function"  # OpenAI uses "function" type
                )
            )
            
        # Return a ToolCallRequest object
        return ToolCallRequest(tool_calls=tool_calls)
    
    def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> Union[GenerateResponse, Any]:
        """
        Generate a response from OpenAI.
        
        Args:
            messages: A list of messages to send to the LLM
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate
            top_p: The top_p value to use
            frequency_penalty: The frequency penalty to use
            presence_penalty: The presence penalty to use
            response_format: The format of the response
            stream: Whether to stream the response
            tools: A list of tool definitions available for the LLM to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A GenerateResponse object containing the response
        """
        import openai
        
        # Validate that the provider supports tools if they're provided
        self._validate_tool_support(tools)
        
        # Prepare request parameters
        request_params = self._prepare_request_params(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            stream=stream,
            **kwargs
        )
        
        # Add tools to the request if provided
        if tools:
            request_params["tools"] = self._convert_tool_definitions(tools)
            
        # Make the API request
        client = openai.OpenAI(api_key=self.config.get("api_key"))
        response = client.chat.completions.create(**request_params)
        
        # Process the response
        content = response.choices[0].message.content
        
        # Extract tool calls if present
        tool_calls = self._extract_tool_calls(response)
        
        # Return the processed response
        return GenerateResponse(
            content=content,
            raw_response=response,
            usage=self._extract_usage(response),
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            tool_calls=tool_calls
        )
```

### 2. Handling Tool Results in Messages

Add support for including tool results in messages:

```python
def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Prepare messages for the OpenAI API.
    
    Args:
        messages: List of AbstractLLM Message objects
        
    Returns:
        List of OpenAI-formatted message dictionaries
    """
    openai_messages = []
    
    for message in messages:
        openai_message = {
            "role": message.role,
            "content": message.content
        }
        
        # Add name if present
        if message.name:
            openai_message["name"] = message.name
        
        # Add tool results if present
        if message.tool_results:
            # OpenAI expects tool results as tool_calls with output
            # Convert from AbstractLLM format to OpenAI format
            tool_calls = []
            for result in message.tool_results:
                tool_calls.append({
                    "id": result.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": result.get("name", ""),
                        "arguments": result.get("arguments", "{}"),
                        "output": result.get("output", "")
                    }
                })
            
            # Add tool calls to the message
            openai_message["tool_calls"] = tool_calls
        
        openai_messages.append(openai_message)
    
    return openai_messages
```

## Anthropic Provider Implementation

### 1. Update `abstractllm/providers/anthropic.py`

```python
from typing import Any, Dict, List, Optional, Union
import json

from ..enums import Capability, ResponseFormat
from ..providers.base import BaseProvider
from ..tools import ToolCallRequest, ToolCall, ToolDefinition
from ..types import GenerateResponse, Message

class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""
    
    @property
    def capabilities(self) -> List[Capability]:
        """Return a list of capabilities supported by this provider."""
        capabilities = [Capability.CHAT]
        
        # Check if the configured API key and model support vision
        if self._supports_vision():
            capabilities.append(Capability.IMAGE_UNDERSTANDING)
        
        # Add function calling capability if supported
        if self._supports_function_calling():
            capabilities.append(Capability.FUNCTION_CALLING)
        
        return capabilities
    
    def _supports_function_calling(self) -> bool:
        """
        Check if the configured model supports function calling.
        
        Returns:
            True if function calling is supported, False otherwise
        """
        # Anthropic Claude 3 models support tools
        supported_models = [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku"
        ]
        
        # Get the configured model
        model = self.config.get("model", "")
        
        # Check if the model is in the supported models list
        return any(model.startswith(supported) for supported in supported_models)
    
    def _convert_tool_definitions(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """
        Convert AbstractLLM tool definitions to Anthropic tool format.
        
        Args:
            tools: List of AbstractLLM tool definitions
            
        Returns:
            List of Anthropic-formatted tool definitions
        """
        anthropic_tools = []
        
        for tool in tools:
            anthropic_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters
            }
            anthropic_tools.append(anthropic_tool)
            
        return anthropic_tools
    
    def _extract_tool_calls(self, response: Any) -> Optional[ToolCallRequest]:
        """
        Extract tool calls from an Anthropic response.
        
        Args:
            response: Raw Anthropic response
            
        Returns:
            ToolCallRequest object if tool calls are present, None otherwise
        """
        # Check if the response contains tool calls
        if not hasattr(response, "content") or not response.content:
            return None
            
        # Extract tool calls from the response content blocks
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                # Anthropic tool calls have an id, name, and input
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),  # Convert dict to JSON string
                        type="function"  # Use "function" for consistency
                    )
                )
            
        # Return None if no tool calls were found
        if not tool_calls:
            return None
            
        # Return a ToolCallRequest object
        return ToolCallRequest(tool_calls=tool_calls)
    
    def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> Union[GenerateResponse, Any]:
        """
        Generate a response from Anthropic.
        
        Args:
            messages: A list of messages to send to the LLM
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate
            top_p: The top_p value to use
            frequency_penalty: The frequency penalty to use
            presence_penalty: The presence penalty to use
            response_format: The format of the response
            stream: Whether to stream the response
            tools: A list of tool definitions available for the LLM to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A GenerateResponse object containing the response
        """
        import anthropic
        
        # Validate that the provider supports tools if they're provided
        self._validate_tool_support(tools)
        
        # Prepare request parameters
        request_params = self._prepare_request_params(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            stream=stream,
            **kwargs
        )
        
        # Add tools to the request if provided
        if tools:
            request_params["tools"] = self._convert_tool_definitions(tools)
            
        # Make the API request
        client = anthropic.Anthropic(api_key=self.config.get("api_key"))
        response = client.messages.create(**request_params)
        
        # Process the response
        content = self._extract_content(response)
        
        # Extract tool calls if present
        tool_calls = self._extract_tool_calls(response)
        
        # Return the processed response
        return GenerateResponse(
            content=content,
            raw_response=response,
            usage=self._extract_usage(response),
            model=response.model,
            finish_reason=self._extract_finish_reason(response),
            tool_calls=tool_calls
        )
```

### 2. Handling Tool Results in Messages

Add support for including tool results in messages:

```python
def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Prepare messages for the Anthropic API.
    
    Args:
        messages: List of AbstractLLM Message objects
        
    Returns:
        List of Anthropic-formatted message dictionaries
    """
    anthropic_messages = []
    
    for message in messages:
        # Base message with role and content
        anthropic_message = {
            "role": self._convert_role(message.role),
            "content": self._prepare_message_content(message)
        }
        
        # Handle tool results if present
        if message.tool_results and message.role == "assistant":
            # Anthropic expects tool results in a specific format
            content_blocks = []
            
            # Add the text content as a block
            if message.content:
                content_blocks.append({
                    "type": "text",
                    "text": message.content
                })
            
            # Add tool results as blocks
            for result in message.tool_results:
                # Add the tool use block
                content_blocks.append({
                    "type": "tool_use",
                    "id": result.get("id", ""),
                    "name": result.get("name", ""),
                    "input": json.loads(result.get("arguments", "{}"))
                })
                
                # Add the tool result block
                content_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": result.get("id", ""),
                    "content": result.get("output", "")
                })
            
            # Replace the content with content blocks
            anthropic_message["content"] = content_blocks
        
        anthropic_messages.append(anthropic_message)
    
    return anthropic_messages
```

## Ollama Provider Implementation

### 1. Update `abstractllm/providers/ollama.py`

```python
from typing import Any, Dict, List, Optional, Union
import json

from ..enums import Capability, ResponseFormat
from ..providers.base import BaseProvider
from ..tools import ToolCallRequest, ToolCall, ToolDefinition
from ..types import GenerateResponse, Message

class OllamaProvider(BaseProvider):
    """Ollama provider implementation."""
    
    @property
    def capabilities(self) -> List[Capability]:
        """Return a list of capabilities supported by this provider."""
        capabilities = [Capability.CHAT]
        
        # Check if the configured model supports vision
        if self._supports_vision():
            capabilities.append(Capability.IMAGE_UNDERSTANDING)
        
        # Add function calling capability if supported by model
        if self._supports_function_calling():
            capabilities.append(Capability.FUNCTION_CALLING)
        
        return capabilities
    
    def _supports_function_calling(self) -> bool:
        """
        Check if the configured model supports function calling.
        
        Returns:
            True if function calling is supported, False otherwise
        """
        # Models known to support tool use in Ollama
        # This may depend on the specific model and version
        supported_models = [
            "llama3",
            "llama3.1"
        ]
        
        # Get the configured model
        model = self.config.get("model", "")
        
        # Check if the model is in the supported models list
        return any(model.startswith(supported) for supported in supported_models)
    
    def _convert_tool_definitions(self, tools: List[ToolDefinition]) -> Dict[str, Any]:
        """
        Convert AbstractLLM tool definitions to Ollama tool format.
        
        Args:
            tools: List of AbstractLLM tool definitions
            
        Returns:
            Ollama-formatted tool definitions dictionary
        """
        ollama_tools = []
        
        for tool in tools:
            ollama_tool = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            ollama_tools.append(ollama_tool)
        
        # Ollama expects tools in a specific format
        return {
            "tools": ollama_tools
        }
    
    def _extract_tool_calls(self, response: Any) -> Optional[ToolCallRequest]:
        """
        Extract tool calls from an Ollama response.
        
        Args:
            response: Raw Ollama response
            
        Returns:
            ToolCallRequest object if tool calls are present, None otherwise
        """
        # Check if the response contains tool calls
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return None
            
        # Extract tool calls from the response
        tool_calls = []
        for tool_call in response.tool_calls:
            # Ollama tool calls have a name and arguments
            tool_calls.append(
                ToolCall(
                    id=str(len(tool_calls)),  # Generate an ID if not provided
                    name=tool_call.get("name", ""),
                    arguments=json.dumps(tool_call.get("parameters", {})),
                    type="function"  # Use "function" for consistency
                )
            )
            
        # Return None if no tool calls were found
        if not tool_calls:
            return None
            
        # Return a ToolCallRequest object
        return ToolCallRequest(tool_calls=tool_calls)
    
    def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> Union[GenerateResponse, Any]:
        """
        Generate a response from Ollama.
        
        Args:
            messages: A list of messages to send to the LLM
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate
            top_p: The top_p value to use
            frequency_penalty: The frequency penalty to use
            presence_penalty: The presence penalty to use
            response_format: The format of the response
            stream: Whether to stream the response
            tools: A list of tool definitions available for the LLM to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            A GenerateResponse object containing the response
        """
        import requests
        
        # Validate that the provider supports tools if they're provided
        self._validate_tool_support(tools)
        
        # Prepare request parameters
        request_params = self._prepare_request_params(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            stream=stream,
            **kwargs
        )
        
        # Add tools to the request if provided
        if tools:
            request_params.update(self._convert_tool_definitions(tools))
            
        # Make the API request
        url = f"{self.config.get('base_url', 'http://localhost:11434')}/api/chat"
        response = requests.post(url, json=request_params)
        response_data = response.json()
        
        # Process the response
        content = response_data.get("message", {}).get("content", "")
        
        # Extract tool calls if present
        tool_calls = self._extract_tool_calls(response_data)
        
        # Return the processed response
        return GenerateResponse(
            content=content,
            raw_response=response_data,
            usage=self._extract_usage(response_data),
            model=model or self.config.get("model", "unknown"),
            finish_reason=self._extract_finish_reason(response_data),
            tool_calls=tool_calls
        )
```

### 2. Handling Tool Results in Messages

Add support for including tool results in messages:

```python
def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
    """
    Prepare messages for the Ollama API.
    
    Args:
        messages: List of AbstractLLM Message objects
        
    Returns:
        List of Ollama-formatted message dictionaries
    """
    ollama_messages = []
    
    for message in messages:
        # Base message with role and content
        ollama_message = {
            "role": message.role,
            "content": message.content
        }
        
        # Handle tool results if present
        if message.tool_results and message.role == "assistant":
            # For Ollama, we need to add tool results in a specific format
            tool_results = []
            
            for result in message.tool_results:
                tool_results.append({
                    "name": result.get("name", ""),
                    "parameters": json.loads(result.get("arguments", "{}")),
                    "output": result.get("output", "")
                })
            
            # Add tool results to the message
            ollama_message["tool_results"] = tool_results
        
        ollama_messages.append(ollama_message)
    
    return ollama_messages
```

## Implementation Considerations

### Provider Compatibility

Different providers have different levels of tool support:

1. **OpenAI** - Well-established function calling support with consistent formats
2. **Anthropic** - Tool support in Claude 3 models with a different format from OpenAI
3. **Ollama** - Tool support varies by model, with evolving format standards

### Error Handling

Implement robust error handling for tool-related functionality:

1. Clear validation of tool definitions before sending to providers
2. Graceful handling of unsupported features
3. Proper error messaging when tool calls fail

### Testing

Create comprehensive tests for tool support in each provider:

1. Unit tests for tool definition conversion
2. Integration tests for tool call extraction
3. Mocked API responses for different scenarios

## Provider-Specific Challenges

### OpenAI

- **Format Evolution**: OpenAI has evolved its function calling API; ensure compatibility with both older and newer formats
- **Streaming**: Handle tool calls correctly during streaming responses

### Anthropic

- **Response Format**: Anthropic uses a different format for tool calls with content blocks
- **Tool Results**: Anthropic has specific formatting requirements for tool results

### Ollama

- **Inconsistent Support**: Tool support varies widely by model; implement careful capability detection
- **Limited Documentation**: Document limitations clearly and provide fallbacks

## Integration with Session Management

Providers need to work with the session management layer:

1. Tool calls must be properly processed and passed to the session
2. Tool results must be correctly formatted when included in follow-up messages

## Example Usage

Example of using tools with a specific provider:

```python
from abstractllm import AbstractLLM
from abstractllm.tools import function_to_tool_definition

# Define a tool function
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

# Create a tool definition
weather_tool = function_to_tool_definition(get_current_weather)

# Create an OpenAI provider with tool support
llm = AbstractLLM.create("openai", api_key="your-api-key")

# Use the tool in a request
response = llm.generate(
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[weather_tool]
)

# Check if the response contains tool calls
if response.has_tool_calls():
    # Process the first tool call
    tool_call = response.tool_calls.tool_calls[0]
    print(f"Tool called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    
    # Parse the arguments
    import json
    args = json.loads(tool_call.arguments)
    
    # Execute the tool
    result = get_current_weather(**args)
    
    # Send the result back in a follow-up request
    follow_up_response = llm.generate(
        messages=[
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant", 
                "content": response.content,
                "tool_results": [{
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": result
                }]
            }
        ]
    )
    
    print(follow_up_response.content)
else:
    print(response.content)
```

## Next Steps

After implementing tool support in the providers, proceed to [Session Implementation](sessions.md) to extend the session management layer for tool support. 