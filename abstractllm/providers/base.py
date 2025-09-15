"""
Base provider implementation for AbstractLLM.
"""

from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING, Tuple, Generator
import logging
import re
import glob
from pathlib import Path
from abc import ABC, abstractmethod

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

class BaseProvider(AbstractLLMInterface, ABC):
    """
    Base class for LLM providers.
    
    This class implements common functionality for all providers including
    tool support through the UniversalToolHandler and universal @file syntax parsing.
    """
    
    def __init__(self, config: Optional[Dict[Any, Any]] = None):
        """Initialize the provider with configuration."""
        super().__init__(config)
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()
        self._tool_handler: Optional[UniversalToolHandler] = None

        # Verbatim context capture for exact LLM input tracking
        self._last_verbatim_context: Optional[str] = None
        self._last_context_timestamp: Optional[str] = None
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False, 
                tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """
        Universal generate method that handles @file parsing for all providers.
        
        This method:
        1. Parses @file references from the prompt
        2. Merges them with any explicit files parameter
        3. Delegates to the provider-specific _generate_impl method
        
        Args:
            prompt: The input prompt (may contain @file references)
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (paths or URLs)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            **kwargs: Additional parameters to override configuration
            
        Returns:
            Generated response or generator
        """
        # Parse @file references from prompt and merge with existing files
        parsed_prompt, file_refs = self._parse_file_references(prompt)
        
        # Combine parsed file references with explicitly provided files
        all_files = []
        if files:
            all_files.extend(files)
        if file_refs:
            all_files.extend(file_refs)
        
        # Log file attachment info if any files were found
        if file_refs:
            logger.info(f"Parsed {len(file_refs)} file references from prompt: {file_refs}")
        
        # Delegate to provider-specific implementation
        return self._generate_impl(
            prompt=parsed_prompt,
            system_prompt=system_prompt,
            files=all_files if all_files else None,
            stream=stream,
            tools=tools,
            **kwargs
        )
    
    @abstractmethod
    def _generate_impl(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None, 
                      files: Optional[List[Union[str, Path]]] = None,
                      stream: bool = False, 
                      tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,
                      **kwargs) -> Union[GenerateResponse, Generator[GenerateResponse, None, None]]:
        """
        Provider-specific implementation of generate.
        
        This method must be implemented by each provider to handle the actual generation.
        The prompt will already have @file references parsed and files will include
        both explicit files and those parsed from @file references.
        
        Args:
            prompt: The input prompt (with @file references already parsed)
            system_prompt: Override the system prompt in the config
            files: Optional list of files to process (includes parsed @file references)
            stream: Whether to stream the response
            tools: Optional list of tools that the model can use
            **kwargs: Additional parameters to override configuration
            
        Returns:
            Generated response or generator
        """
        pass
    
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
            parsed = None
            if hasattr(response, 'content') and response.content:
                parsed = handler.parse_response(response.content, mode=mode)
            elif isinstance(response, str):
                parsed = handler.parse_response(response, mode=mode)
            elif isinstance(response, dict):
                # For native responses that come as dictionaries
                parsed = handler.parse_response(response, mode="native")
            
            # Return tool calls if found (logging handled at session level for universal coverage)
            if parsed and parsed.has_tool_calls():
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
    
    def _log_tool_calls_found(self, tool_calls: List["ToolCall"]) -> None:
        """
        Log when tool calls are found in a response (universal for all providers).
        
        This provides clear visibility when the LLM actually decides to use tools,
        showing which tools are called and with what parameters.
        
        Args:
            tool_calls: List of tool calls found in the response
        """
        if not tool_calls:
            return
            
        # Use concise, direct format for tool call display - NO parameter truncation
        for i, tc in enumerate(tool_calls, 1):
            # Format arguments - keep ALL parameters intact
            if tc.arguments:
                args_str = str(tc.arguments)
            else:
                args_str = "{}"
            
            # Create concise, direct message
            if len(tool_calls) == 1:
                # Single tool call - more concise format
                tool_message = f"🔧 LLM called {tc.name}({args_str})"
            else:
                # Multiple tool calls - show which one
                tool_message = f"🔧 LLM called {tc.name}({args_str}) [{i}/{len(tool_calls)}]"
                
            # Log to file
            logger.info(tool_message)
            # Print to console in yellow with direct format
            print(f"\033[33m{tool_message}\033[0m")
    
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
    
    def _parse_file_references(self, text: str) -> Tuple[str, List[str]]:
        """
        Parse @file references from text and return cleaned text with list of files.
        
        This method provides universal @file syntax support across all providers.
        Following SOTA practices, it clearly indicates to the LLM that files are 
        attached and available in context to prevent tool usage.
        
        Supports:
        - @file.txt - attach single file
        - @folder/ - attach all files in folder  
        - @*.py - attach files matching pattern
        
        Args:
            text: Input text that may contain @file references
            
        Returns:
            Tuple of (cleaned_text, list_of_file_paths)
        """
        if not text or '@' not in text:
            return text, []
            
        # Pattern to match @file references
        file_pattern = r'@([^\s]+)'
        
        attached_files = []
        file_references = []
        
        def replace_file_ref(match):
            file_ref = match.group(1)
            
            # Handle different file reference types
            try:
                if file_ref.endswith('/'):
                    # Directory - get all files in directory
                    dir_path = Path(file_ref)
                    if dir_path.exists() and dir_path.is_dir():
                        files = [str(f) for f in dir_path.iterdir() if f.is_file()]
                        attached_files.extend(files)
                        # Add each file as a reference
                        for file_path in files:
                            file_references.append(f"📎 {Path(file_path).name}")
                        return ""  # Remove @reference from prompt
                    else:
                        return f"[Directory {file_ref} not found]"
                
                elif '*' in file_ref or '?' in file_ref:
                    # Glob pattern
                    files = glob.glob(file_ref)
                    if files:
                        attached_files.extend(files)
                        # Add each file as a reference
                        for file_path in files:
                            file_references.append(f"📎 {Path(file_path).name}")
                        return ""  # Remove @reference from prompt
                    else:
                        return f"[No files found matching {file_ref}]"
                
                else:
                    # Single file
                    file_path = Path(file_ref)
                    if file_path.exists() and file_path.is_file():
                        attached_files.append(str(file_path))
                        file_references.append(f"📎 {file_path.name}")
                        return ""  # Remove @reference from prompt
                    else:
                        return f"[File {file_ref} not found]"
                        
            except Exception as e:
                logger.warning(f"Error processing file reference {file_ref}: {e}")
                return f"[Error processing {file_ref}: {str(e)}]"
        
        # Replace all @file references
        cleaned_text = re.sub(file_pattern, replace_file_ref, text)
        
        # Add clear file context indication and content at the beginning if files were attached
        if file_references:
            file_list = "\n".join(file_references)
            
            # Load and include actual file content following SOTA practices
            file_content_sections = []
            for file_path in attached_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        file_name = Path(file_path).name
                        file_content_sections.append(f"""<document>
<source>{file_name}</source>
<document_content>
{content}
</document_content>
</document>""")
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    file_name = Path(file_path).name
                    file_content_sections.append(f"""<document>
<source>{file_name}</source>
<document_content>
[Error reading file: {str(e)}]
</document_content>
</document>""")
            
            file_content = "\n\n".join(file_content_sections)
            
            cleaned_text = f"""📎 ATTACHED FILES - The following files are loaded in your context:
{file_list}

<documents>
{file_content}
</documents>

IMPORTANT: The above files are already loaded in your context. Do NOT use tools to access these files. Refer to their content directly from the documents above.

{cleaned_text.strip()}"""
        
        # Log file attachment info
        if attached_files:
            logger.info(f"Parsed {len(attached_files)} file references from prompt")
            
        return cleaned_text, attached_files
    
    def _log_request_details(self, 
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           messages: Optional[List[Dict[str, Any]]] = None,
                           tools: Optional[List[Any]] = None,
                           formatted_messages: Optional[List[Dict[str, Any]]] = None,
                           request_data: Optional[Dict[str, Any]] = None,
                           endpoint: Optional[str] = None,
                           **kwargs) -> None:
        """
        Log detailed request information in a standardized way across all providers.
        
        This method ensures consistent logging across all providers, similar to
        what MLX provider does, capturing all important details for debugging.
        
        Args:
            prompt: The user prompt
            system_prompt: The system prompt (if any)
            messages: Raw messages list (if any)
            tools: Tools being used (if any)
            formatted_messages: Messages after formatting for the provider
            request_data: The actual request data being sent to the API
            endpoint: The API endpoint being called
            **kwargs: Additional parameters to log
        """
        from abstractllm.utils.logging import log_request
        
        # Get model name
        model = self.config_manager.get_param(ModelParameter.MODEL)
        
        # Build comprehensive log parameters
        log_params = {
            "model": model,
            "provider": self.provider_name,
            "temperature": kwargs.get("temperature", self.config_manager.get_param(ModelParameter.TEMPERATURE)),
            "max_tokens": kwargs.get("max_tokens", self.config_manager.get_param(ModelParameter.MAX_TOKENS)),
            "stream": kwargs.get("stream", False),
        }
        
        # Add system prompt details
        if system_prompt:
            log_params["system_prompt"] = system_prompt
            log_params["has_system_prompt"] = True
        else:
            log_params["has_system_prompt"] = False
            
        # Add enhanced system prompt if different
        enhanced_system_prompt = kwargs.get("enhanced_system_prompt")
        if enhanced_system_prompt and enhanced_system_prompt != system_prompt:
            log_params["enhanced_system_prompt"] = enhanced_system_prompt
            
        # Add tool information
        if tools:
            log_params["has_tools"] = True
            log_params["tools_count"] = len(tools)
            log_params["tools"] = []
            
            for tool in tools:
                if hasattr(tool, "name") and hasattr(tool, "description"):
                    # ToolDefinition object
                    tool_info = {
                        "name": tool.name,
                        "description": tool.description[:100] + "..." if len(tool.description) > 100 else tool.description,
                        "parameters": tool.parameters
                    }
                    log_params["tools"].append(tool_info)
                elif isinstance(tool, dict):
                    # Dictionary tool definition
                    tool_info = {
                        "name": tool.get("name", "unknown"),
                        "description": (tool.get("description", "")[:100] + "..." 
                                      if len(tool.get("description", "")) > 100 
                                      else tool.get("description", "")),
                        "parameters": tool.get("parameters", {})
                    }
                    log_params["tools"].append(tool_info)
                elif callable(tool):
                    # Function tool - convert to ToolDefinition to get parameters
                    from abstractllm.tools.core import ToolDefinition
                    try:
                        tool_def = ToolDefinition.from_function(tool)
                        tool_info = {
                            "name": tool_def.name,
                            "description": tool_def.description[:100] + "..." if len(tool_def.description) > 100 else tool_def.description,
                            "parameters": tool_def.parameters
                        }
                    except Exception as e:
                        # Fallback if conversion fails
                        tool_info = {
                            "name": getattr(tool, "__name__", str(tool)),
                            "description": getattr(tool, "__doc__", "")[:100] if getattr(tool, "__doc__", "") else ""
                        }
                    log_params["tools"].append(tool_info)
        else:
            log_params["has_tools"] = False
            log_params["tools_count"] = 0
            
        # Add messages information
        if messages:
            log_params["messages_count"] = len(messages)
            log_params["original_messages"] = messages
            
        # Add formatted messages (what actually gets sent)
        if formatted_messages:
            log_params["formatted_messages"] = formatted_messages
            log_params["formatted_messages_count"] = len(formatted_messages)
            
        # Add request data details
        if request_data:
            # Extract key information from request data
            if "messages" in request_data:
                log_params["request_messages"] = request_data["messages"]
            if "tools" in request_data:
                log_params["request_tools"] = request_data["tools"]
            if "functions" in request_data:
                log_params["request_functions"] = request_data["functions"]
                
        # Add endpoint information
        if endpoint:
            log_params["endpoint"] = endpoint
            
        # Add any additional kwargs that might be relevant
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens", "stream", "enhanced_system_prompt"]:
                log_params[key] = value
                
        # Log to both logger and file
        logger.info(f"Request to {self.provider_name}: {endpoint or 'API'}")
        logger.debug(f"Request details: model={model}, tools={log_params.get('tools_count', 0)}, messages={log_params.get('messages_count', 0)}")
        
        # Log the comprehensive request
        log_request(self.provider_name, prompt, log_params, model=model)
    
    def _log_response_details(self, 
                            response: Any,
                            content: Optional[str] = None,
                            **kwargs) -> None:
        """
        Log detailed response information in a standardized way across all providers.
        
        Args:
            response: The raw response from the provider
            content: The extracted content (if different from response)
            **kwargs: Additional response details to log
        """
        from abstractllm.utils.logging import log_response
        
        # Get model name from kwargs or config
        model = kwargs.get("model") or self.config_manager.get_param(ModelParameter.MODEL)
        
        # Extract content if not provided
        if content is None:
            if isinstance(response, str):
                content = response
            elif hasattr(response, "content"):
                content = response.content
            elif isinstance(response, dict):
                content = response.get("content", response.get("response", str(response)))
            else:
                content = str(response)
                
        # Log the response
        logger.info(f"Response from {self.provider_name}: {len(content)} chars")
        
        # Log additional details if provided
        if kwargs.get("has_tool_calls"):
            logger.info(f"Response contains tool calls: {len(kwargs.get('tool_calls', []))}")
        if kwargs.get("usage"):
            usage = kwargs["usage"]
            logger.info(f"Token usage - prompt: {usage.get('prompt_tokens', 0)}, completion: {usage.get('completion_tokens', 0)}")
            
        logger.debug(f"Response preview: {content[:200]}..." if len(content) > 200 else f"Response: {content}")
        
        # Remove model from kwargs to avoid duplicate parameter error
        log_kwargs = kwargs.copy()
        log_kwargs.pop("model", None)
        
        # Log to file with additional metadata
        log_response(self.provider_name, content, model=model, **log_kwargs)
    
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

    def _capture_verbatim_context(self, context: str) -> None:
        """
        Capture the exact verbatim context sent to the LLM.

        This method should be called by providers right before sending the context
        to the LLM to capture the exact format and content that will be processed.

        Args:
            context: The exact string/format sent to the LLM
        """
        from datetime import datetime

        self._last_verbatim_context = context
        self._last_context_timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def get_last_verbatim_context(self) -> Optional[Dict[str, str]]:
        """
        Get the last verbatim context that was sent to the LLM.

        Returns:
            Dictionary with 'context' and 'timestamp' keys, or None if no context captured
        """
        if self._last_verbatim_context is None:
            return None

        return {
            'context': self._last_verbatim_context,
            'timestamp': self._last_context_timestamp,
            'provider': self.provider_name,
            'model': self.config_manager.get_param(ModelParameter.MODEL)
        } 