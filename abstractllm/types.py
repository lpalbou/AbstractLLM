"""
Type definitions for AbstractLLM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from abstractllm.enums import MessageRole

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools.core import ToolCallResponse

# Import ToolCallResponse from new location
try:
    from abstractllm.tools.core import ToolCallResponse
except ImportError as e:
    # Fallback if tools package is not available
    if not TYPE_CHECKING:
        # Provide a placeholder to avoid failures in basic usage
        class ToolCallResponse:
            """Placeholder when tools not available."""
            def __init__(self, *args, **kwargs):
                self.content = kwargs.get("content", "")
                self.tool_calls = kwargs.get("tool_calls", [])
                
            def has_tool_calls(self) -> bool:
                """Check if has tool calls."""
                return bool(self.tool_calls)
                
        # Store the original error for introspection
        TOOL_IMPORT_ERROR = str(e)


@dataclass
class GenerateResponse:
    """A response from an LLM."""
    
    content: Optional[str] = None
    raw_response: Any = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    
    # Field for tool calls
    tool_calls: Optional["ToolCallResponse"] = None
    
    # Field for image paths used in vision models
    image_paths: Optional[List[str]] = None
    
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        if self.tool_calls is None:
            return False
        
        # Use the has_tool_calls method if available
        if hasattr(self.tool_calls, 'has_tool_calls'):
            return self.tool_calls.has_tool_calls()
        
        # Fallback for other structures
        return bool(getattr(self.tool_calls, 'tool_calls', []))


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