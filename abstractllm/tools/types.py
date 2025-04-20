"""Type definitions for AbstractLLM tool support."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by an LLM."""
    
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema for inputs")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for return value")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate that the name follows the required pattern."""
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Tool name must only contain alphanumeric characters and underscores")
        return v
    
    @validator('input_schema')
    def validate_input_schema(cls, v):
        """Validate that the input schema is valid."""
        if not isinstance(v, dict) or v.get('type') != 'object':
            raise ValueError("Input schema must be an object schema")
        if 'properties' not in v:
            raise ValueError("Input schema must have a 'properties' field")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return self.dict(exclude_none=True)


class ToolCall(BaseModel):
    """A call to a tool by an LLM."""
    
    id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create a ToolCall from a dictionary."""
        # Handle different formats from different providers
        if "function" in data:
            # OpenAI format
            import json
            args = data["function"]["arguments"]
            # Handle arguments as string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"value": args}  # Fallback
            
            return cls(
                id=data.get("id", ""),
                name=data["function"]["name"],
                arguments=args
            )
        else:
            # Standard/Anthropic format
            return cls(
                id=data.get("id", ""),
                name=data["name"],
                arguments=data["arguments"]
            )


class ToolResult(BaseModel):
    """The result of executing a tool call."""
    
    call_id: str = Field(..., description="ID of the tool call this result responds to")
    result: Any = Field(..., description="Result of the tool execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")


class ToolCallRequest(BaseModel):
    """A request containing tool calls from an LLM."""
    
    content: Optional[str] = Field(None, description="Text content from the LLM")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tool calls")
    
    def has_tool_calls(self) -> bool:
        """Check if this request contains any tool calls."""
        return len(self.tool_calls) > 0
    
    @classmethod
    def from_provider_response(cls, response: Dict[str, Any], provider: str) -> "ToolCallRequest":
        """Create a ToolCallRequest from a provider-specific response."""
        from .conversion import standardize_tool_response
        # Delegate to the standardize_tool_response function
        return standardize_tool_response(response, provider)


class ToolCallResponse(BaseModel):
    """A response to be sent back to the LLM after executing tool calls."""
    
    tool_results: List[ToolResult] = Field(default_factory=list, description="Results of executed tools")
    
    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert to a dictionary representation."""
        return {
            "tool_results": [result.dict(exclude_none=True) for result in self.tool_results]
        } 