"""
Base provider implementation for AbstractLLM.
"""

from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING, Tuple
import logging
from pathlib import Path

from abstractllm.interface import AbstractLLMInterface
from abstractllm.types import GenerateResponse, Message
from abstractllm.enums import ModelParameter, ModelCapability

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools import ToolDefinition, ToolCall, ToolResult, ToolCallResponse
    from abstractllm.tools.handler import UniversalToolHandler

# Try importing from tools package, but handle if it's not available
try:
    from abstractllm.tools import (
        ToolDefinition,
        ToolCall,
        ToolResult, 
        ToolCallResponse,
        UniversalToolHandler
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    # Define placeholder for type hints if not imported during TYPE_CHECKING
    if not TYPE_CHECKING:
        class ToolDefinition:
            pass
        class ToolCall:
            pass
        class ToolResult:
            pass
        class ToolCallResponse:
            pass
        class UniversalToolHandler:
            pass

# Configure logger
logger = logging.getLogger("abstractllm.providers.base")

class BaseProvider(AbstractLLMInterface):
    """
    Base class for LLM providers.
    
    This class implements common functionality for all providers including
    tool support through the UniversalToolHandler.
    """
    
    def __init__(self, config: Optional[Dict[Any, Any]] = None):
        """Initialize the provider with configuration."""
        super().__init__(config)
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self._tool_handler: Optional[UniversalToolHandler] = None
    
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
                processed_tools.append(ToolDefinition.from_function(tool))
            # If it's already a ToolDefinition, use it directly
            elif hasattr(tool, 'name') and hasattr(tool, 'description'):  # Duck typing for ToolDefinition
                processed_tools.append(tool)
            # If it's a dictionary, convert it to a ToolDefinition
            elif isinstance(tool, dict):
                processed_tools.append(ToolDefinition(
                    name=tool.get('name', 'unknown'),
                    description=tool.get('description', ''),
                    parameters=tool.get('parameters', {})
                ))
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
    
    def _extract_tool_calls(self, response: Any) -> Optional[ToolCallResponse]:
        """
        Extract tool calls from a provider response using the universal handler.
        
        Args:
            response: The raw response from the provider
            
        Returns:
            A ToolCallResponse object if tool calls are present, None otherwise
        """
        if not TOOLS_AVAILABLE:
            return None
            
        handler = self._get_tool_handler()
        if not handler:
            return None
            
        try:
            # Determine the mode based on handler capabilities and provider
            mode = "native" if handler.supports_native and self._check_for_tool_calls(response) else "prompted"
            
            # Parse the response using the handler
            if hasattr(response, 'content') and response.content:
                parsed = handler.parse_response(response.content, mode=mode)
                if parsed.has_tool_calls():
                    return parsed
            elif isinstance(response, str):
                parsed = handler.parse_response(response, mode=mode)
                if parsed.has_tool_calls():
                    return parsed
            elif isinstance(response, dict):
                # For native responses that come as dictionaries
                parsed = handler.parse_response(response, mode="native")
                if parsed.has_tool_calls():
                    return parsed
                    
            return None
        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}")
            return None
    
    def _get_tool_handler(self) -> Optional[UniversalToolHandler]:
        """Get or create the tool handler for this provider."""
        if not TOOLS_AVAILABLE:
            return None
            
        if self._tool_handler is None:
            # Get the model from config
            model = self.get_param(ModelParameter.MODEL)
            if model:
                self._tool_handler = UniversalToolHandler(model)
        return self._tool_handler
    
    def _prepare_tool_context(self, 
                            tools: Optional[List[Any]], 
                            system_prompt: Optional[str] = None) -> Tuple[Optional[str], Optional[List[Dict]], str]:
        """
        Prepare tool context for generation.
        
        This method handles tool preparation without modifying messages.
        It returns an enhanced system prompt (for prompted mode) or 
        tool definitions (for native mode).
        
        Args:
            tools: Optional list of tools
            system_prompt: Original system prompt
            
        Returns:
            Tuple of (enhanced_system_prompt, tool_definitions, mode)
            where mode is "native", "prompted", or "none"
        """
        if not tools:
            return system_prompt, None, "none"
            
        # Validate tool support
        self._validate_tool_support(tools)
        
        # Process tools
        processed_tools = self._process_tools(tools)
        
        # Get handler
        handler = self._get_tool_handler()
        if not handler:
            raise ValueError("Tool handler not available")
        
        # Check capabilities
        if handler.supports_native:
            # Native mode - prepare tools for API
            tool_defs = handler.prepare_tools_for_native(processed_tools)
            # Format for specific provider
            formatted_tools = self._format_tools_for_provider(tool_defs)
            return system_prompt, formatted_tools, "native"
        elif handler.supports_prompted:
            # Prompted mode - enhance system prompt
            tool_prompt = handler.format_tools_prompt(processed_tools)
            
            # Combine with existing system prompt
            if system_prompt:
                enhanced = f"{system_prompt}\n\n{tool_prompt}"
            else:
                enhanced = tool_prompt
                
            return enhanced, None, "prompted"
        else:
            # No tool support
            from abstractllm.exceptions import UnsupportedFeatureError
            raise UnsupportedFeatureError(
                "tools",
                f"Model {handler.model_name} does not support tools",
                provider=self.provider_name
            )
    
    def _format_tools_for_provider(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tools for the specific provider's API.
        
        Override this method in provider implementations that need
        specific tool formatting (e.g., OpenAI needs 'type': 'function').
        
        Args:
            tools: List of tool dictionaries from handler
            
        Returns:
            List of formatted tool dictionaries
        """
        # Default implementation returns tools as-is
        return tools
    
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