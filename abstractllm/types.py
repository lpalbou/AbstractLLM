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
    """A response from an LLM with optional ReAct telemetry."""
    
    content: Optional[str] = None
    raw_response: Any = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    
    # Field for tool calls - now stores readable information
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    # Field for image paths used in vision models
    image_paths: Optional[List[str]] = None
    
    # Enhanced ReAct telemetry fields
    react_cycle_id: Optional[str] = None
    tools_executed: Optional[List[Dict[str, Any]]] = None
    facts_extracted: Optional[List[str]] = None
    reasoning_trace: Optional[str] = None
    total_reasoning_time: Optional[float] = None
    
    # Complete scratchpad observability 
    scratchpad_file: Optional[str] = None
    scratchpad_manager: Optional[Any] = None
    
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return bool(self.tool_calls)
    
    def get_tools_executed(self) -> List[str]:
        """Get list of tool names that were executed."""
        if not self.tools_executed:
            return []
        return [tool.get('name', '') for tool in self.tools_executed]
    
    def get_scratchpad_trace(self) -> Optional[str]:
        """Get the ReAct cycle scratchpad trace."""
        return self.reasoning_trace
    
    def get_facts_extracted(self) -> List[str]:
        """Get facts extracted during this generation."""
        return self.facts_extracted or []
    
    def get_summary(self) -> str:
        """Get a concise summary of the response telemetry."""
        summary = []
        if self.react_cycle_id:
            summary.append(f"ReAct Cycle: {self.react_cycle_id}")
        if self.tools_executed:
            summary.append(f"Tools: {len(self.tools_executed)} executed")
        if self.facts_extracted:
            summary.append(f"Facts: {len(self.facts_extracted)} extracted")
        if self.total_reasoning_time:
            summary.append(f"Time: {self.total_reasoning_time:.2f}s")
        if self.scratchpad_file:
            summary.append(f"Scratchpad: {self.scratchpad_file}")
        
        return " | ".join(summary) if summary else "No telemetry data"
    
    def get_complete_scratchpad_file(self) -> Optional[str]:
        """Get path to the complete scratchpad file."""
        return self.scratchpad_file
    
    def subscribe_to_react_events(self, phase=None, callback=None):
        """Subscribe to real-time ReAct phase change events."""
        if self.scratchpad_manager:
            self.scratchpad_manager.subscribe_to_events(phase, callback)
        else:
            raise ValueError("No scratchpad manager available - enable memory to access events")
    
    def get_cycle_summary(self, cycle_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed summary of a ReAct cycle."""
        if self.scratchpad_manager:
            target_cycle = cycle_id or self.react_cycle_id
            if target_cycle:
                return self.scratchpad_manager.get_cycle_summary(target_cycle)
        return {}


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