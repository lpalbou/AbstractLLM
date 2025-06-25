"""
Universal tool handler for all models and providers.

This module provides a utility class for tool support that works
across all models, whether they have native tool APIs or require prompting.
It focuses on formatting and parsing without modifying request state.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable

from abstractllm.architectures import get_model_capabilities
from abstractllm.tools.core import ToolDefinition, ToolCall, ToolCallResponse, ToolResult
from abstractllm.tools.parser import detect_tool_calls, parse_tool_calls, format_tool_prompt

logger = logging.getLogger(__name__)


class UniversalToolHandler:
    """
    Utility class for tool formatting and parsing based on model capabilities.
    
    This is a pure utility class that:
    - Provides architecture-specific tool prompt formatting
    - Parses tool calls from model responses
    - Formats tool results for models
    - Reports model capabilities
    
    It does NOT modify messages or maintain state.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize handler for a specific model.
        
        Args:
            model_name: Model identifier
        """
        self.model_name = model_name
        self.capabilities = get_model_capabilities(model_name)
        
        # Determine support levels
        self.tool_support = self.capabilities.get("tool_support", "none")
        self.supports_native = self.tool_support == "native"
        self.supports_prompted = self.tool_support in ["native", "prompted"]
        
        logger.debug(f"Initialized tool handler for {model_name}: "
                    f"native={self.supports_native}, prompted={self.supports_prompted}")
    
    def format_tools_prompt(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
    ) -> str:
        """
        Format tools into a prompt based on model architecture.
        
        Args:
            tools: List of tools (ToolDefinition, callable, or dict)
            
        Returns:
            Formatted tool prompt string for the model's architecture
        """
        if not tools:
            return ""
        
        # Convert tools to ToolDefinition objects
        tool_defs = []
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                tool_defs.append(tool)
            elif callable(tool):
                tool_defs.append(ToolDefinition.from_function(tool))
            elif isinstance(tool, dict) and "name" in tool and "description" in tool:
                tool_defs.append(ToolDefinition(**tool))
            else:
                logger.warning(f"Skipping invalid tool: {tool}")
        
        if not tool_defs:
            return ""
        
        # Format based on architecture
        formatted_prompt = format_tool_prompt(tool_defs, self.model_name)
        
        # Add extra emphasis for MLX models which often misunderstand tool formats
        if "qwen" in self.model_name.lower():
            formatted_prompt += "\n\nCRITICAL: When using tools, you MUST use the exact format shown above. DO NOT use Python code blocks or any other format. Use the <|tool_call|> format EXACTLY as shown."
        
        return formatted_prompt
    
    def prepare_tools_for_native(
        self,
        tools: List[Union[ToolDefinition, Callable, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Convert tools to native API format.
        
        Args:
            tools: List of tools
            
        Returns:
            List of tool dictionaries for native API
        """
        if not tools or not self.supports_native:
            return []
        
        # Convert tools to ToolDefinition objects
        tool_defs = []
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                tool_defs.append(tool)
            elif callable(tool):
                tool_defs.append(ToolDefinition.from_function(tool))
            elif isinstance(tool, dict) and "name" in tool and "description" in tool:
                tool_defs.append(ToolDefinition(**tool))
            else:
                logger.warning(f"Skipping invalid tool: {tool}")
        
        return [t.to_dict() for t in tool_defs]
    
    def parse_response(
        self,
        response: Union[str, Dict[str, Any]],
        mode: str = "prompted"
    ) -> ToolCallResponse:
        """
        Parse model response for tool calls.
        
        Args:
            response: Model response (string or dict)
            mode: Request mode ("native" or "prompted")
            
        Returns:
            ToolCallResponse with content and tool calls
        """
        if mode == "native":
            # Handle native tool response format
            if isinstance(response, dict):
                content = response.get("content", "")
                tool_calls = []
                
                # Extract tool calls based on provider format
                if "tool_calls" in response:
                    # Direct tool calls
                    for tc in response["tool_calls"]:
                        tool_calls.append(ToolCall(
                            name=tc.get("name") or tc.get("function", {}).get("name"),
                            arguments=tc.get("arguments") or tc.get("parameters", {}),
                            id=tc.get("id")
                        ))
                elif "function_call" in response:
                    # OpenAI legacy format
                    fc = response["function_call"]
                    tool_calls.append(ToolCall(
                        name=fc["name"],
                        arguments=json.loads(fc.get("arguments", "{}"))
                    ))
                
                return ToolCallResponse(content=content, tool_calls=tool_calls)
            else:
                return ToolCallResponse(content=str(response))
        
        else:
            # Parse prompted tool calls
            content = str(response)
            tool_calls = []
            
            if detect_tool_calls(content, self.model_name):
                tool_calls = parse_tool_calls(content, self.model_name)
            
            return ToolCallResponse(content=content, tool_calls=tool_calls)
    
    def format_tool_results(
        self,
        results: List[ToolResult],
        mode: str = "prompted"
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Format tool results for the model.
        
        Args:
            results: List of tool execution results
            mode: Request mode
            
        Returns:
            Formatted results (list for native, string for prompted)
        """
        if mode == "native":
            # Native format
            return [
                {
                    "tool_call_id": r.tool_call_id,
                    "output": str(r.output) if r.success else None,
                    "error": r.error
                }
                for r in results
            ]
        else:
            # Prompted format - natural language
            parts = []
            for result in results:
                if result.success:
                    parts.append(f"Tool result: {result.output}")
                else:
                    parts.append(f"Tool error: {result.error}")
            
            return "\n\n".join(parts)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool-related capabilities."""
        return {
            "supports_tools": self.supports_prompted,
            "native_tools": self.supports_native,
            "tool_format": self.capabilities.get("tool_template", "none"),
            "max_tools": self.capabilities.get("max_tools", -1),
            "parallel_tools": self.capabilities.get("parallel_tools", False)
        }


def create_handler(model_name: str) -> UniversalToolHandler:
    """Create a tool handler for a model."""
    return UniversalToolHandler(model_name)


