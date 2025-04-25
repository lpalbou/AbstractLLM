# Phase 2: Core Interface Adaptation

This document provides detailed implementation guidance for extending AbstractLLM's core interfaces to support tool calls.

## Overview

The core interface adaptation phase modifies the central interfaces of AbstractLLM to support tool functionality:

1. Update `AbstractLLMInterface` to accept tool definitions
2. Extend `GenerateResponse` to include tool call information
3. Add new capability flags for tool support
4. Update enums to include tool-related information

## Files to Modify

### 1. `abstractllm/interface.py`

Extend the `AbstractLLMInterface` class to support tool calls:

```python
"""Interface definition for AbstractLLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from .enums import Capability, ResponseFormat
from .tools import ToolDefinition
from .types import GenerateResponse, Message


class AbstractLLMInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @property
    @abstractmethod
    def capabilities(self) -> List[Capability]:
        """Return a list of capabilities supported by this provider."""
        pass
    
    @abstractmethod
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
        tools: Optional[List[ToolDefinition]] = None,  # New parameter
        **kwargs: Any
    ) -> Union[GenerateResponse, Any]:
        """
        Generate a response from the LLM.
        
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
        pass
```

### 2. `abstractllm/types.py`

Extend the `GenerateResponse` class to include tool call information:

```python
"""Type definitions for AbstractLLM."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .tools import ToolCallRequest  # Import from tools package


@dataclass
class GenerateResponse:
    """A response from an LLM."""
    
    content: Optional[str] = None
    raw_response: Any = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    
    # New field for tool calls
    tool_calls: Optional[ToolCallRequest] = None
    
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return (
            self.tool_calls is not None and 
            len(self.tool_calls.tool_calls) > 0
        )


@dataclass
class Message:
    """A message to send to an LLM."""
    
    role: str
    content: str
    name: Optional[str] = None
    
    # New field for tool responses
    tool_results: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        message_dict = {
            "role": self.role,
            "content": self.content,
        }
        
        if self.name is not None:
            message_dict["name"] = self.name
            
        if self.tool_results is not None:
            message_dict["tool_results"] = self.tool_results
            
        return message_dict
```

### 3. `abstractllm/enums.py`

Update the enums to include tool-related information:

```python
"""Enumerations for AbstractLLM."""

from enum import Enum, auto


class Capability(Enum):
    """Capabilities that a provider may support."""
    
    CHAT = auto()
    IMAGE_GENERATION = auto()
    IMAGE_UNDERSTANDING = auto()
    TEXT_EMBEDDING = auto()
    FILE_ATTACHMENT = auto()
    FUNCTION_CALLING = auto()  # New capability for function/tool calling
    TOOL_USE = auto()  # New capability for tool use (might be different in some providers)
    
    # You can add more capabilities as needed


class ResponseFormat(Enum):
    """Response formats that a provider may support."""
    
    TEXT = auto()
    JSON = auto()
    # You can add more response formats as needed
```

### 4. `abstractllm/providers/base.py`

If you have a base provider class, update it to support tool calls:

```python
"""Base provider implementation for AbstractLLM."""

from typing import Any, Dict, List, Optional, Union

from ..enums import Capability, ResponseFormat
from ..interface import AbstractLLMInterface
from ..tools import ToolDefinition, standardize_tool_response
from ..types import GenerateResponse, Message


class BaseProvider(AbstractLLMInterface):
    """Base class for LLM providers."""
    
    def __init__(self, **kwargs: Any):
        """Initialize the provider."""
        self._validate_config(**kwargs)
        self.config = kwargs
    
    def _validate_config(self, **kwargs: Any) -> None:
        """Validate the provider configuration."""
        # Implement validation logic in subclasses
        pass
    
    @property
    def capabilities(self) -> List[Capability]:
        """Return a list of capabilities supported by this provider."""
        # Default capabilities, override in subclasses
        return [Capability.CHAT]
    
    def supports_capability(self, capability: Capability) -> bool:
        """Check if the provider supports the given capability."""
        return capability in self.capabilities
    
    def _validate_tool_support(self, tools: Optional[List[ToolDefinition]]) -> None:
        """
        Validate that the provider supports tools if they are provided.
        
        Args:
            tools: A list of tool definitions to validate
            
        Raises:
            ValueError: If the provider does not support tools but they are provided
        """
        if tools and not self.supports_capability(Capability.FUNCTION_CALLING):
            raise ValueError(
                f"{self.__class__.__name__} does not support function/tool calling"
            )
    
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
        Generate a response from the LLM.
        
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
        # Validate tool support
        self._validate_tool_support(tools)
        
        # Implementation in subclasses
        raise NotImplementedError("Must be implemented by subclasses")
```

## Interface Extensions

### Capability Detection and Reporting

Update the capability reporting logic for existing providers to include the new tool-related capabilities:

For example, if the OpenAI provider supports function calling, update its capabilities:

```python
@property
def capabilities(self) -> List[Capability]:
    """Return a list of capabilities supported by this provider."""
    capabilities = [Capability.CHAT]
    
    # Add other capabilities based on model or configuration
    if self._supports_vision():
        capabilities.append(Capability.IMAGE_UNDERSTANDING)
    
    # Add function calling capability
    if self._supports_function_calling():
        capabilities.append(Capability.FUNCTION_CALLING)
    
    return capabilities

def _supports_function_calling(self) -> bool:
    """Check if the provider supports function calling."""
    # Logic to determine if function calling is supported
    # This could be based on the model or other configuration
    return True  # Or some condition based on the model/config
```

### Response Handling

Update response handling to process tool calls:

```python
def _process_response(self, response: Any) -> GenerateResponse:
    """
    Process the raw response from the provider.
    
    Args:
        response: The raw response from the provider
        
    Returns:
        A processed GenerateResponse object
    """
    # Extract basic information
    content = self._extract_content(response)
    usage = self._extract_usage(response)
    model = self._extract_model(response)
    finish_reason = self._extract_finish_reason(response)
    
    # Process tool calls if present
    tool_calls = None
    if self._has_tool_calls(response):
        # Use the standardize_tool_response function to convert provider-specific format
        tool_calls = standardize_tool_response(response, self.provider_name)
    
    return GenerateResponse(
        content=content,
        raw_response=response,
        usage=usage,
        model=model,
        finish_reason=finish_reason,
        tool_calls=tool_calls
    )
```

## Implementation Considerations

### Backward Compatibility

The interface changes are designed to maintain backward compatibility:

1. The `tools` parameter in `generate()` is optional
2. Existing code that doesn't use tools will continue to work without modification
3. The `has_tool_calls()` method provides a convenient way to check for tool calls

### Provider-Specific Customization

Each provider may require specific customization for tool support:

1. Some providers may use different terminology (functions vs. tools)
2. Response formats vary between providers
3. Capabilities may depend on specific models

### Error Handling

Add appropriate error handling for tool-related functionality:

1. Clear error messages when tools are provided to a provider that doesn't support them
2. Validation to ensure tool definitions meet provider requirements
3. Proper handling of malformed tool calls in responses

## Integration with Other Phases

This phase provides the foundation for:

1. **Provider Implementations**: The extended interfaces define what providers need to implement
2. **Session Lifecycle Management**: The response types enable tool call detection and processing in sessions

## Example Usage

After implementing these changes, client code can use tools as follows:

```python
from abstractllm import AbstractLLM
from abstractllm.tools import function_to_tool_definition

# Define a tool function
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and state, e.g., "San Francisco, CA"
        unit: The unit of temperature, either "celsius" or "fahrenheit"
        
    Returns:
        A string describing the current weather
    """
    # Implementation would go here
    return f"The weather in {location} is 22°{unit[0].upper()}"

# Create a tool definition
weather_tool = function_to_tool_definition(get_weather)

# Use the tool in a request
llm = AbstractLLM.create("openai")
response = llm.generate(
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[weather_tool]
)

# Check if the response contains tool calls
if response.has_tool_calls():
    # Process tool calls...
    print("Tool calls detected!")
else:
    # Normal response
    print(response.content)
```

## Next Steps

After implementing the core interface adaptations, proceed to [Provider Implementations](providers.md) to update specific provider implementations for tool support. 