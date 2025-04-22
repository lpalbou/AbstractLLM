"""
Base provider implementation for AbstractLLM.
"""

from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING
import logging
from pathlib import Path

from abstractllm.interface import AbstractLLMInterface
from abstractllm.types import GenerateResponse, Message
from abstractllm.enums import ModelParameter, ModelCapability

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools.types import ToolCallRequest, ToolDefinition
    from abstractllm.tools import function_to_tool_definition, standardize_tool_response

# Try importing from tools package, but handle if it's not available
try:
    from abstractllm.tools import (
        ToolDefinition, 
        function_to_tool_definition,
        standardize_tool_response
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    # Define placeholder for type hints if not imported during TYPE_CHECKING
    if not TYPE_CHECKING:
        class ToolDefinition:
            pass
        class ToolCallRequest:
            pass

# Configure logger
logger = logging.getLogger("abstractllm.providers.base")

class BaseProvider(AbstractLLMInterface):
    """
    Base class for LLM providers.
    
    This class implements common functionality for all providers.
    """
    
    def __init__(self, config: Optional[Dict[Any, Any]] = None):
        """Initialize the provider with configuration."""
        super().__init__(config)
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
    
    def _validate_tool_support(self, tools: Optional[List[Any]]) -> None:
        """
        Validate that the provider supports tools if they are provided.
        
        Args:
            tools: A list of tool definitions to validate
            
        Raises:
            UnsupportedFeatureError: If the provider does not support tools but they are provided
        """
        if not tools:
            return
            
        if not TOOLS_AVAILABLE:
            raise ValueError("Tool support is not available. Install the required dependencies.")
            
        capabilities = self.get_capabilities()
        supports_tools = (
            capabilities.get(ModelCapability.FUNCTION_CALLING, False) or 
            capabilities.get(ModelCapability.TOOL_USE, False)
        )
        
        if not supports_tools:
            from abstractllm.exceptions import UnsupportedFeatureError
            raise UnsupportedFeatureError(
                feature="function_calling",
                message=f"{self.__class__.__name__} does not support function/tool calling",
                provider=self.provider_name
            )
    
    def _process_tools(self, tools: List[Any]) -> List["ToolDefinition"]:
        """
        Process and validate tool definitions.
        
        Args:
            tools: A list of tool definitions or callables
            
        Returns:
            A list of validated ToolDefinition objects
        """
        if not TOOLS_AVAILABLE:
            raise ValueError("Tool support is not available. Install the required dependencies.")
            
        processed_tools = []
        
        for tool in tools:
            # If it's a callable, convert it to a tool definition
            if callable(tool):
                processed_tools.append(function_to_tool_definition(tool))
            # If it's already a ToolDefinition, use it directly
            elif isinstance(tool, ToolDefinition):
                processed_tools.append(tool)
            # If it's a dictionary, try to convert it to a ToolDefinition
            elif isinstance(tool, dict):
                from abstractllm.tools.validation import validate_tool_definition
                processed_tools.append(validate_tool_definition(tool))
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
                
        return processed_tools
    
    def _check_for_tool_calls(self, response: Any) -> bool:
        """
        Check if a provider response contains tool calls.
        
        Args:
            response: The raw response from the provider
            
        Returns:
            True if the response contains tool calls, False otherwise
        """
        # Default implementation returns False
        # Override in provider-specific implementations
        return False
    
    def _extract_tool_calls(self, response: Any) -> Optional["ToolCallRequest"]:
        """
        Extract tool calls from a provider response.
        
        Args:
            response: The raw response from the provider
            
        Returns:
            A ToolCallRequest object if tool calls are present, None otherwise
        """
        if not TOOLS_AVAILABLE or not self._check_for_tool_calls(response):
            return None
            
        try:
            return standardize_tool_response(response, self.provider_name)
        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}")
            return None
    
    def _process_response(self, 
                         response: Any, 
                         content: Optional[str] = None, 
                         usage: Optional[Dict[str, int]] = None,
                         model: Optional[str] = None,
                         finish_reason: Optional[str] = None) -> GenerateResponse:
        """
        Process a raw response from the provider.
        
        Args:
            response: The raw response from the provider
            content: Optional content to use instead of extracting from response
            usage: Optional usage statistics
            model: Optional model name
            finish_reason: Optional finish reason
            
        Returns:
            A GenerateResponse object
        """
        # Extract tool calls if present
        tool_calls = self._extract_tool_calls(response)
        
        return GenerateResponse(
            content=content,
            raw_response=response,
            usage=usage,
            model=model,
            finish_reason=finish_reason,
            tool_calls=tool_calls
        ) 