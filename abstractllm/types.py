"""
Type definitions for AbstractLLM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from abstractllm.enums import MessageRole

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools.types import ToolCallRequest

# Import ToolCallRequest after it's defined in tools package
# This is done conditionally to avoid circular imports
try:
    from abstractllm.tools.types import ToolCallRequest
except ImportError:
    # Fallback if tools package is not available
    if not TYPE_CHECKING:
        ToolCallRequest = Any


@dataclass
class GenerateResponse:
    """A response from an LLM."""
    
    content: Optional[str] = None
    raw_response: Any = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    
    # Field for tool calls
    tool_calls: Optional["ToolCallRequest"] = None
    
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        if self.tool_calls is None:
            return False
        
        # Handle both direct and nested tool_calls structures
        if hasattr(self.tool_calls, 'has_tool_calls'):
            # Use the has_tool_calls method if available (ToolCallRequest object)
            return self.tool_calls.has_tool_calls()
        elif hasattr(self.tool_calls, 'tool_calls'):
            # Old format compatibility: tool_calls has a tool_calls attribute
            return bool(getattr(self.tool_calls, 'tool_calls', []))
        elif isinstance(self.tool_calls, list):
            # Direct list of tool calls
            return bool(self.tool_calls)
        
        return False


@dataclass
class Message:
    """A message to send to an LLM."""
    
    role: Union[str, MessageRole]
    content: str
    name: Optional[str] = None
    
    # Field for tool responses
    tool_results: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        message_dict = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
        }
        
        if self.name is not None:
            message_dict["name"] = self.name
            
        if self.tool_results is not None:
            message_dict["tool_results"] = self.tool_results
            
        return message_dict 