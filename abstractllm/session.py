"""
Session management for AbstractLLM.

This module provides utilities for managing stateful conversations with LLMs,
including tracking conversation history and metadata across multiple requests.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Generator, TYPE_CHECKING
from datetime import datetime
import json
import os
import uuid
import logging

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.factory import create_llm
from abstractllm.exceptions import UnsupportedFeatureError
from abstractllm.enums import MessageRole

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools.types import ToolDefinition, ToolCall, ToolCallRequest, ToolResult
    from abstractllm.types import GenerateResponse

# Try importing tools package directly
try:
    from abstractllm.tools import (
        ToolDefinition,
        ToolCall,
        ToolCallRequest,
        ToolResult,
        function_to_tool_definition,
    )
    from abstractllm.types import GenerateResponse
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    if not TYPE_CHECKING:
        class ToolDefinition:
            pass
        class ToolCall:
            pass
        class ToolCallRequest:
            pass
        class ToolResult:
            pass
        class GenerateResponse:
            pass


class Message:
    """
    Represents a single message in a conversation.
    """
    
    def __init__(self, 
                 role: str, 
                 content: str, 
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tool_results: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize a message.
        
        Args:
            role: The role of the sender (e.g., "user", "assistant", "system")
            content: The message content
            timestamp: When the message was created (defaults to now)
            metadata: Additional message metadata
            tool_results: Optional list of tool execution results, each with:
                - call_id: Unique identifier for the tool call
                - name: Name of the tool that was called
                - arguments: Arguments that were passed to the tool
                - output: The output from the tool execution
                - error: Optional error message if the tool execution failed
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.tool_results = tool_results or []
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary representation.
        
        Returns:
            A dictionary representing the message
        """
        result = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
        
        if self.tool_results:
            # Ensure each tool result has all required fields
            formatted_results = []
            for tr in self.tool_results:
                formatted_result = {
                    "call_id": tr.get("call_id", ""),
                    "name": tr.get("name", "unknown_tool"),
                    "output": tr.get("output", "")
                }
                
                # Add optional fields if present
                if "arguments" in tr:
                    formatted_result["arguments"] = tr["arguments"]
                if "error" in tr:
                    formatted_result["error"] = tr["error"]
                    
                formatted_results.append(formatted_result)
                
            result["tool_results"] = formatted_results
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create a message from a dictionary representation.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            A Message instance
        """
        # Ensure tool_results are properly formatted
        tool_results = data.get("tool_results")
        if tool_results:
            # Validate and normalize each tool result
            for i, tr in enumerate(tool_results):
                if "call_id" not in tr:
                    tr["call_id"] = f"call_{i}"
                if "name" not in tr:
                    tr["name"] = "unknown_tool"
        
        message = cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            tool_results=tool_results
        )
        message.id = data.get("id", str(uuid.uuid4()))
        return message


class Session:
    """
    Manages a conversation session with one or more LLM providers.
    
    A session keeps track of conversation history and provides methods
    for continuing the conversation with the same or different providers.
    """
    
    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 provider: Optional[Union[str, AbstractLLMInterface]] = None,
                 provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tools: Optional[List[Union[Dict[str, Any], Callable, "ToolDefinition"]]] = None):
        """
        Initialize a conversation session.
        
        Args:
            system_prompt: The system prompt for the conversation
            provider: Provider name or instance to use for this session
            provider_config: Configuration for the provider
            metadata: Session metadata
            tools: Optional list of tool definitions available for the LLM to use
        """
        self.messages: List[Message] = []
        self.system_prompt = system_prompt
        self.metadata = metadata or {}
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self.tools: List["ToolDefinition"] = []
        
        # Track last assistant message index for tool results
        self._last_assistant_idx = -1
        
        # Initialize the provider if specified
        self._provider: Optional[AbstractLLMInterface] = None
        if provider is not None:
            if isinstance(provider, str):
                self._provider = create_llm(provider, **(provider_config or {}))
            else:
                self._provider = provider
        
        # Add system message if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM, system_prompt)
            
        # Add tools if provided and tools are available
        if tools and TOOLS_AVAILABLE:
            for tool in tools:
                self.add_tool(tool)
    
    def add_message(self, 
                    role: Union[str, MessageRole], 
                    content: str, 
                    name: Optional[str] = None,
                    tool_results: Optional[List[Dict[str, Any]]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to the conversation.
        
        Args:
            role: Message role ("user", "assistant", "system") - note that "tool" role
                  will be adapted based on provider support
            content: Message content
            name: Optional name for the message sender
            tool_results: Optional list of tool execution results
            metadata: Optional message metadata
            
        Returns:
            The created message
        """
        if isinstance(role, MessageRole):
            role = role.value
            
        # Trim whitespace from content to avoid API errors
        content = content.strip() if content else ""
            
        message = Message(
            role=role, 
            content=content, 
            timestamp=datetime.now(),
            metadata=metadata or {},
            tool_results=tool_results
        )
        
        self.messages.append(message)
        self.last_updated = message.timestamp
        
        # Update last assistant message index if this is an assistant message
        if role == MessageRole.ASSISTANT.value:
            self._last_assistant_idx = len(self.messages) - 1
            
        return message
    
    def get_history(self, include_system: bool = True) -> List[Message]:
        """
        Get the conversation history.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            List of messages
        """
        if include_system:
            return self.messages.copy()
        return [m for m in self.messages if m.role != MessageRole.SYSTEM.value]
    
    def get_formatted_prompt(self, new_message: Optional[str] = None) -> str:
        """
        Get a formatted prompt that includes conversation history.
        
        This method formats the conversation history and an optional new message
        into a prompt that can be sent to a provider that doesn't natively
        support chat history.
        
        Args:
            new_message: Optional new message to append
            
        Returns:
            Formatted prompt string
        """
        formatted = ""
        
        # Format each message
        for message in self.messages:
            if message.role == "system":
                continue  # System messages handled separately
                
            prefix = f"{message.role.title()}: "
            formatted += f"{prefix}{message.content}\n\n"
        
        # Add the new message if provided
        if new_message:
            formatted += f"User: {new_message}\n\nAssistant: "
        
        return formatted.strip()
    
    def get_messages_for_provider(self, provider_name: str) -> List[Dict[str, Any]]:
        """
        Get messages formatted for a specific provider's API.
        
        Args:
            provider_name: Provider name
            
        Returns:
            List of message dictionaries in the provider's expected format
        """
        formatted: List[Dict[str, Any]] = []
        
        for m in self.messages:
            # Skip system messages for Anthropic (handled separately)
            if provider_name == "anthropic" and m.role == 'system':
                continue
                
            # Add the main message with content stripped of trailing whitespace
            formatted.append({
                "role": m.role, 
                "content": m.content.strip() if isinstance(m.content, str) else m.content
            })
            
            # Process any tool results with provider-specific formatting
            if getattr(m, 'tool_results', None):
                for tr in m.tool_results:
                    tool_name = tr.get('name', 'unknown_tool')
                    output = tr.get('output', '')
                    
                    if provider_name == "openai":
                        # OpenAI uses 'function' role for tool outputs
                        formatted.append({
                            "role": "function", 
                            "name": tool_name,
                            "content": output
                        })
                    elif provider_name == "anthropic":
                        # Anthropic doesn't support 'tool' role - use 'assistant' with formatted content
                        formatted.append({
                            "role": "assistant",
                            "content": f"Tool '{tool_name}' returned the following output:\n\n{output}".strip()
                        })
                    elif provider_name in ["ollama", "huggingface"]:
                        # These providers may not have special tool formatting
                        # Add a prefixed assistant message
                        formatted.append({
                            "role": "assistant",
                            "content": f"TOOL OUTPUT [{tool_name}]: {output}".strip()
                        })
                    else:
                        # Default case: add prefixed content for clarity
                        formatted.append({
                            "role": "assistant", 
                            "content": f"TOOL OUTPUT [{tool_name}]: {output}".strip()
                        })
        
        return formatted
    
    def send(self, message: str, 
             provider: Optional[Union[str, AbstractLLMInterface]] = None,
             stream: bool = False,
             **kwargs) -> Union[str, Any]:
        """
        Send a message to the LLM and add the response to the conversation.
        
        Args:
            message: The message to send
            provider: Provider to use (overrides the session provider)
            stream: Whether to stream the response
            **kwargs: Additional parameters for the provider
            
        Returns:
            The LLM's response
        """
        # Add the user message to the conversation
        self.add_message(MessageRole.USER, message)
        
        # Determine which provider to use
        llm = self._get_provider(provider)
        
        # Get provider name for formatting
        provider_name = self._get_provider_name(llm)
        
        # Check if the provider supports chat history
        capabilities = llm.get_capabilities()
        supports_chat = capabilities.get(ModelCapability.MULTI_TURN, False)
        
        # Prepare the request based on provider capabilities
        if supports_chat:
            messages = self.get_messages_for_provider(provider_name)
            
            # Add provider-specific handling here as needed
            if provider_name == "openai":
                response = llm.generate(messages=messages, stream=stream, **kwargs)
            elif provider_name == "anthropic":
                response = llm.generate(messages=messages, stream=stream, **kwargs)
            else:
                # Default approach for other providers that support chat
                response = llm.generate(messages=messages, stream=stream, **kwargs)
        else:
            # For providers that don't support chat history, format a prompt
            formatted_prompt = self.get_formatted_prompt()
            response = llm.generate(
                formatted_prompt, 
                system_prompt=self.system_prompt,
                stream=stream, 
                **kwargs
            )
        
        # If not streaming, add the response to the conversation
        if not stream:
            self.add_message(MessageRole.ASSISTANT, response)
            
        return response
    
    def send_async(self, message: str,
                  provider: Optional[Union[str, AbstractLLMInterface]] = None,
                  stream: bool = False,
                  **kwargs) -> Any:
        """
        Send a message asynchronously and add the response to the conversation.
        
        Args:
            message: The message to send
            provider: Provider to use (overrides the session provider)
            stream: Whether to stream the response
            **kwargs: Additional parameters for the provider
            
        Returns:
            A coroutine that resolves to the LLM's response
        """
        # Add the user message
        self.add_message(MessageRole.USER, message)
        
        # Determine which provider to use
        llm = self._get_provider(provider)
        
        # Check if async is supported
        capabilities = llm.get_capabilities()
        if not capabilities.get(ModelCapability.ASYNC, False):
            raise UnsupportedFeatureError(
                "async_generation", 
                "This provider does not support async generation",
                provider=self._get_provider_name(llm)
            )
        
        # Get provider name for formatting
        provider_name = self._get_provider_name(llm)
        
        # Check if the provider supports chat history
        supports_chat = capabilities.get(ModelCapability.MULTI_TURN, False)
        
        async def _async_handler():
            # Prepare the request based on provider capabilities
            if supports_chat:
                messages = self.get_messages_for_provider(provider_name)
                
                # Add provider-specific handling here as needed
                if provider_name == "openai":
                    response = await llm.generate_async(messages=messages, stream=stream, **kwargs)
                elif provider_name == "anthropic":
                    response = await llm.generate_async(messages=messages, stream=stream, **kwargs)
                else:
                    # Default approach for other providers that support chat
                    response = await llm.generate_async(messages=messages, stream=stream, **kwargs)
            else:
                # For providers that don't support chat history, format a prompt
                formatted_prompt = self.get_formatted_prompt()
                response = await llm.generate_async(
                    formatted_prompt, 
                    system_prompt=self.system_prompt,
                    stream=stream, 
                    **kwargs
                )
            
            # If not streaming, add the response to the conversation
            if not stream:
                self.add_message(MessageRole.ASSISTANT, response)
                
            return response
        
        return _async_handler()
    
    def save(self, filepath: str) -> None:
        """
        Save the session to a file.
        
        Args:
            filepath: Path to save the session to
        """
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, 
             provider: Optional[Union[str, AbstractLLMInterface]] = None,
             provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None) -> 'Session':
        """
        Load a session from a file.
        
        Args:
            filepath: Path to load the session from
            provider: Provider to use for the loaded session
            provider_config: Configuration for the provider
            
        Returns:
            A Session instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create a new session
        session = cls(
            system_prompt=data.get("system_prompt"),
            provider=provider,
            provider_config=provider_config,
            metadata=data.get("metadata", {})
        )
        
        # Set session properties
        session.id = data.get("id", str(uuid.uuid4()))
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_updated = datetime.fromisoformat(data["last_updated"])
        
        # Clear the automatically added system message
        session.messages = []
        
        # Add messages
        for message_data in data.get("messages", []):
            message = Message.from_dict(message_data)
            session.messages.append(message)
        
        return session
    
    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        Clear the conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt
        """
        if keep_system_prompt and self.system_prompt:
            # Keep only system messages
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []
    
    def _get_provider(self, provider: Optional[Union[str, AbstractLLMInterface]] = None) -> AbstractLLMInterface:
        """
        Get the provider to use for a request.
        
        Args:
            provider: Provider override
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If no provider is available
        """
        if provider is not None:
            if isinstance(provider, str):
                return create_llm(provider)
            return provider
        
        if self._provider is not None:
            return self._provider
        
        raise ValueError(
            "No provider specified. Either initialize the session with a provider "
            "or specify one when sending a message."
        )
    
    def _get_provider_name(self, provider: AbstractLLMInterface) -> str:
        """
        Get the name of a provider.
        
        Args:
            provider: Provider instance
            
        Returns:
            Provider name
        """
        # Try to get the provider name from the class name
        class_name = provider.__class__.__name__
        if class_name.endswith("Provider"):
            return class_name[:-8].lower()
        
        # Fallback to checking class module
        module = provider.__class__.__module__
        if "openai" in module:
            return "openai"
        elif "anthropic" in module:
            return "anthropic"
        elif "ollama" in module:
            return "ollama"
        elif "huggingface" in module:
            return "huggingface"
        
        # Default
        return "unknown"

    def add_tool(self, tool: Union[Dict[str, Any], Callable, "ToolDefinition"]) -> None:
        """
        Add a tool to the session.
        
        Args:
            tool: The tool definition or function to add, can be:
                - A dictionary with tool definition
                - A callable function to be converted to a tool definition
                - A ToolDefinition object
        
        Raises:
            ValueError: If tools are not available
        """
        if not TOOLS_AVAILABLE:
            raise ValueError("Tool support is not available. Install the required dependencies.")
        
        # Convert tool to ToolDefinition
        if callable(tool):
            # Convert function to tool definition
            tool_def = function_to_tool_definition(tool)
        elif isinstance(tool, dict):
            # Convert dictionary to tool definition
            tool_def = ToolDefinition(**tool)
        else:
            # Already a ToolDefinition
            tool_def = tool
            
        self.tools.append(tool_def)
        # The provider will validate tool support when generate is called
    
    def execute_tool_call(
        self,
        tool_call: "ToolCall",
        tool_functions: Dict[str, Callable[..., Any]]
    ) -> Dict[str, Any]:
        """
        Execute a tool call using the provided functions.
        
        Args:
            tool_call: The tool call to execute
            tool_functions: Dictionary of available tool functions
            
        Returns:
            Dictionary containing the tool result or error
        """
        logger = logging.getLogger("abstractllm.session")
        logger.info(f"Session: Executing tool call: {tool_call.name} with args: {tool_call.arguments}")
        
        # Check if the tool function exists
        if tool_call.name not in tool_functions:
            error_msg = f"Tool '{tool_call.name}' not found in available tools."
            logger.error(error_msg)
            return {
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "output": None,
                "error": error_msg
            }
        
        # Get the tool function
        tool_function = tool_functions[tool_call.name]
        
        # Find the corresponding tool definition if available
        tool_def = None
        if TOOLS_AVAILABLE and hasattr(self, 'tools') and self.tools:
            for tool in self.tools:
                if isinstance(tool, dict) and tool.get('name') == tool_call.name:
                    # For dictionary tools
                    tool_def = tool
                    break
                elif hasattr(tool, 'name') and tool.name == tool_call.name:
                    # For ToolDefinition objects
                    tool_def = tool
                    break
        
        # Execute the tool and handle potential errors
        try:
            # Parse arguments as needed
            args = tool_call.arguments
            
            # Handle case where arguments are provided as a JSON string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse arguments as JSON: {str(e)}"
                    logger.error(error_msg)
                    return {
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "output": None,
                        "error": error_msg
                    }
            
            # Execute the tool function with arguments
            result = tool_function(**args)
            logger.info(f"Session: Tool execution successful: {tool_call.name}")
            
            # Validate the result against the output schema if available
            if TOOLS_AVAILABLE and tool_def and hasattr(tool_def, 'output_schema') and tool_def.output_schema:
                try:
                    # Import jsonschema for validation
                    from jsonschema import validate, ValidationError
                    try:
                        validate(instance=result, schema=tool_def.output_schema)
                    except ValidationError as e:
                        error_msg = f"Tool result validation failed: {str(e)}"
                        logger.error(error_msg)
                        return {
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "output": None,
                            "error": error_msg
                        }
                except ImportError:
                    # If jsonschema is not available, skip validation
                    pass
            
            # Return a successful result with tool name included
            return {
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "output": result,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error executing tool '{tool_call.name}': {str(e)}"
            logger.error(error_msg)
            return {
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "output": None,
                "error": error_msg
            }
    
    def execute_tool_calls(
        self,
        response: "GenerateResponse",
        tool_functions: Dict[str, Callable[..., Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute all tool calls in a response and return the results.
        
        Args:
            response: The response containing tool calls
            tool_functions: A dictionary mapping tool names to their implementation functions
            
        Returns:
            A list of dictionaries containing the tool results
            
        Raises:
            ValueError: If the response does not contain tool calls
        """
        # Check if the response contains tool calls
        if not response.has_tool_calls():
            raise ValueError("Response does not contain tool calls")
        
        # Get the tool calls from the response, handling different formats
        tool_results = []
        
        if hasattr(response.tool_calls, 'tool_calls'):
            # Standard format with nested tool_calls
            tool_calls = response.tool_calls.tool_calls
            
        elif isinstance(response.tool_calls, list):
            # Direct list of tool calls
            tool_calls = response.tool_calls
            
        else:
            # Unknown format
            raise ValueError(f"Unsupported tool_calls format: {type(response.tool_calls)}")
            
        # Execute each tool call
        for tool_call in tool_calls:
            tool_result = self.execute_tool_call(tool_call, tool_functions)
            tool_results.append(tool_result)
            
        return tool_results
        
    def add_tool_result(
        self, 
        tool_call_id: str, 
        result: Any, 
        error: Optional[str] = None,
        tool_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a tool result to the session.
        
        Args:
            tool_call_id: ID of the tool call this result responds to
            result: The result of the tool execution
            error: Optional error message if the tool execution failed
            tool_name: Optional name of the tool (for better provider formatting)
            arguments: Optional arguments used in the tool call
        """
        # Use tracked assistant message index instead of searching
        if self._last_assistant_idx < 0:
            # No assistant messages yet, can't add tool result
            raise ValueError("No assistant message found to attach tool result to")
        
        last_assistant_msg = self.messages[self._last_assistant_idx]
        
        # Initialize tool_results if not present
        if not hasattr(last_assistant_msg, "tool_results") or not last_assistant_msg.tool_results:
            last_assistant_msg.tool_results = []
        
        # Create and add the tool result with consistent format
        if TOOLS_AVAILABLE:
            # Use ToolResult for internal representation
            tool_result_obj = ToolResult(
                call_id=tool_call_id,
                result=result,
                error=error
            )
            # Convert to dict for backward compatibility
            tool_result = {
                "call_id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "output": str(result)
            }
            if error:
                tool_result["error"] = error
        else:
            # Fallback to dict representation
            tool_result = {
                "call_id": tool_call_id,
                "name": tool_name,
                "arguments": arguments,
                "output": str(result)
            }
            if error:
                tool_result["error"] = error
            
        last_assistant_msg.tool_results.append(tool_result)

    def _adjust_system_prompt_for_tool_phase(
        self, 
        original_system_prompt: str, 
        tool_call_count: int, 
        executed_tools: List[str],
        phase: str = "initial"
    ) -> str:
        """
        Adjust the system prompt based on the current tool execution phase.
        
        This helps guide the LLM through different phases of the conversation:
        - Initial phase: Encourage the LLM to use tools to gather information
        - Processing phase: Guide the LLM to process tool outputs
        - Synthesis phase: Direct the LLM to synthesize all gathered information
        
        Args:
            original_system_prompt: The original system prompt
            tool_call_count: How many tool calls have been executed
            executed_tools: List of names of tools that have been executed
            phase: Current phase ("initial", "processing", or "synthesis")
            
        Returns:
            Adjusted system prompt
        """
        # Base system prompt from the original
        base_prompt = original_system_prompt.strip()
        
        # Ensure there's proper spacing
        if not base_prompt.endswith("."):
            base_prompt += "."
            
        # Add a line break if needed
        if not base_prompt.endswith("\n"):
            base_prompt += "\n\n"
            
        # Phase-specific additions
        if phase == "initial":
            # Initial phase - encourage tool usage
            addition = (
                "When you need information to answer the user's question, use the appropriate tools provided to you. "
                "Think step by step about which tools you need and why."
            )
        elif phase == "processing" and tool_call_count == 1:
            # First tool just executed - guide tool output processing
            tool_names = ", ".join(executed_tools)
            addition = (
                f"You've received output from the following tool(s): {tool_names}. "
                "Process this information to help answer the user's question. "
                "If you need more information, you can continue using tools."
            )
        elif phase == "processing" and tool_call_count > 1:
            # Multiple tools executed - continue processing
            tool_names = ", ".join(executed_tools)
            addition = (
                f"You've now used {tool_call_count} tools: {tool_names}. "
                "Continue building your understanding based on this information. "
                "If you still need more details, you can request additional tools."
            )
        elif phase == "synthesis":
            # Final synthesis phase - create a complete answer
            tool_names = ", ".join(executed_tools)
            addition = (
                f"You now have all necessary information from tools ({tool_names}). "
                "Synthesize a complete, accurate answer to the user's original question based on all tool outputs. "
                "Make sure your response is comprehensive and addresses all aspects of the query."
            )
        else:
            # Default - minimal guidance
            addition = "Use the available information to provide the best possible answer."
            
        return base_prompt + addition

    def _track_tool_execution_metrics(
        self,
        tool_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Track metrics about tool execution success.
        
        Args:
            tool_results: List of tool execution results
            
        Returns:
            Dictionary of metrics including success rate and executed tools
        """
        metrics = {
            "total_tools": len(tool_results),
            "successful_tools": 0,
            "failed_tools": 0,
            "executed_tools": [],
            "errors": []
        }
        
        for result in tool_results:
            tool_name = result.get("name", "unknown")
            metrics["executed_tools"].append(tool_name)
            
            if result.get("error"):
                metrics["failed_tools"] += 1
                metrics["errors"].append({
                    "tool": tool_name,
                    "error": result.get("error")
                })
            else:
                metrics["successful_tools"] += 1
                
        # Calculate success rate
        if metrics["total_tools"] > 0:
            metrics["success_rate"] = metrics["successful_tools"] / metrics["total_tools"]
        else:
            metrics["success_rate"] = 0
            
        return metrics

    def generate_with_tools(
        self,
        tool_functions: Dict[str, Callable[..., Any]],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tool_calls: int = 10,
        adjust_system_prompt: bool = True,
        **kwargs
    ) -> "GenerateResponse":
        """
        Generate a response with tool execution support.
        
        This method handles the complete flow of tool usage:
        1. Generate an initial response with tool definitions
        2. If the response contains tool calls, execute them
        3. Add the tool results to the conversation (formatted according to provider requirements)
        4. Generate a follow-up response that incorporates the tool results
        
        Note: Tool results formatting varies by provider. Some providers (like OpenAI) support
        dedicated roles for tool outputs, while others (like Anthropic) require formatting
        tool outputs as assistant messages.
        
        Args:
            tool_functions: A dictionary mapping tool names to their implementation functions
            prompt: The input prompt (if None, uses the existing conversation history)
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate
            top_p: The top_p value to use
            frequency_penalty: The frequency penalty to use
            presence_penalty: The presence penalty to use
            max_tool_calls: Maximum number of tool call iterations to prevent infinite loops
            adjust_system_prompt: Whether to adjust the system prompt based on tool execution phase
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The final GenerateResponse after tool execution and follow-up
        """
        logger = logging.getLogger("abstractllm.session")
        
        # Ensure we have tools available
        if not TOOLS_AVAILABLE:
            raise ValueError("Tool support is not available. Install the required dependencies.")
        
        # Store the original prompt and system prompt for follow-up requests
        original_prompt = prompt if prompt else ""
        original_system_prompt = self.system_prompt
        
        # Track executed tools for metrics and prompt adjustment
        all_executed_tools = []
        current_system_prompt = original_system_prompt
        
        # Add user message if provided
        if prompt:
            self.add_message(MessageRole.USER, prompt)
            
        # Get the provider if needed
        provider = self._get_provider()
        
        # Get the provider name to format messages properly
        provider_name = self._get_provider_name(provider)
        
        # Get conversation history formatted for this provider
        provider_messages = self.get_messages_for_provider(provider_name)
        
        # Adjust system prompt for initial phase if needed
        if adjust_system_prompt:
            current_system_prompt = self._adjust_system_prompt_for_tool_phase(
                original_system_prompt, 
                0, 
                all_executed_tools, 
                "initial"
            )
            logger.debug(f"Session: Using initial phase system prompt")
        
        # 1) Initial generate with tools
        response = provider.generate(
            prompt=original_prompt,
            system_prompt=current_system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=self.tools,
            messages=provider_messages,
            **kwargs
        )
        logger.info("Session: Received initial response from LLM")

        # 2) Loop: execute any tool calls and regenerate until no more tools requested
        tool_call_count = 0
        while response.has_tool_calls() and tool_call_count < max_tool_calls:
            # Increment counter to prevent infinite loops
            tool_call_count += 1
            
            # Execute tool calls
            logger.info(f"Session: LLM requested tool calls (iteration {tool_call_count}/{max_tool_calls})")
            tool_results = self.execute_tool_calls(response, tool_functions)
            
            # Track tool execution metrics
            metrics = self._track_tool_execution_metrics(tool_results)
            logger.info(f"Session: Tool execution metrics - "
                       f"Success: {metrics['successful_tools']}/{metrics['total_tools']} "
                       f"({metrics['success_rate']:.0%})")
            
            # Update executed tools list for system prompt adjustment
            all_executed_tools.extend(metrics['executed_tools'])
            # Remove duplicates while preserving order
            all_executed_tools = list(dict.fromkeys(all_executed_tools))
            
            # Debug: Log tool call details
            for tool_result in tool_results:
                logger.info(f"Session: Executed tool call - ID: {tool_result.get('call_id')}, "
                            f"Name: {tool_result.get('name')}, "
                            f"Args: {tool_result.get('arguments')}")
            
            # Add the assistant message with tool results
            logger.info(f"Session: Adding assistant message with {len(tool_results)} tool results")
            tool_message = self.add_message(
                MessageRole.ASSISTANT,
                content=response.content or "",
                tool_results=tool_results,
                metadata={"tool_metrics": metrics}
            )
            
            # Adjust system prompt based on tool execution phase
            if adjust_system_prompt:
                # Determine the phase based on tool call count and whether there are more expected
                phase = "synthesis" if tool_call_count >= max_tool_calls - 1 else "processing"
                
                current_system_prompt = self._adjust_system_prompt_for_tool_phase(
                    original_system_prompt, 
                    tool_call_count, 
                    all_executed_tools, 
                    phase
                )
                logger.debug(f"Session: Adjusting to {phase} phase system prompt")
            
            # Prepare follow-up prompt
            updated_provider_messages = self.get_messages_for_provider(provider_name)
            
            # Debug: Log message structure being sent to the provider
            logger.debug(f"Session: Sending follow-up messages to LLM: {updated_provider_messages}")
            logger.info(f"Session: Generating follow-up response (iteration {tool_call_count}) with updated conversation")
            
            # Regenerate with tools still enabled, passing the original prompt
            response = provider.generate(
                prompt=original_prompt,
                system_prompt=current_system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tools=self.tools,
                messages=updated_provider_messages,
                **kwargs
            )
            logger.info(f"Session: Received follow-up response from LLM (iteration {tool_call_count})")
        
        # Log if we hit the maximum tool call limit
        if tool_call_count >= max_tool_calls and response.has_tool_calls():
            logger.warning(f"Session: Maximum tool call limit ({max_tool_calls}) reached. "
                           f"Some tool calls may not have been executed.")
        
        # Log transition to final answer mode if tools were used
        if tool_call_count > 0:
            # Final synthesis phase - create a complete answer
            if adjust_system_prompt:
                final_system_prompt = self._adjust_system_prompt_for_tool_phase(
                    original_system_prompt, 
                    tool_call_count, 
                    all_executed_tools, 
                    "synthesis"
                )
                logger.info(f"Session: Transitioned from tool calling mode to answer generation mode")
                logger.debug(f"Session: Using synthesis phase system prompt")
                
                # One final generation with synthesis prompt if needed and we didn't already synthesize
                if current_system_prompt != final_system_prompt and response.has_tool_calls():
                    updated_provider_messages = self.get_messages_for_provider(provider_name)
                    response = provider.generate(
                        prompt=original_prompt,
                        system_prompt=final_system_prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        tools=self.tools,
                        messages=updated_provider_messages,
                        **kwargs
                    )
                    logger.info(f"Session: Generated final synthesis response after tool execution")

        # 3) Final assistant response
        final_content = response.content or ""
        final_message = self.add_message(
            MessageRole.ASSISTANT, 
            final_content,
            metadata={
                "tool_execution": {
                    "tool_call_count": tool_call_count,
                    "executed_tools": all_executed_tools,
                    "phase": "synthesis" if tool_call_count > 0 else "initial"
                }
            }
        )
        
        # Reset the system prompt to the original
        if adjust_system_prompt and self.system_prompt != original_system_prompt:
            self.system_prompt = original_system_prompt
            
        return response
        
    def generate_with_tools_streaming(
        self,
        tool_functions: Dict[str, Callable[..., Any]],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tool_calls: int = 5,
        adjust_system_prompt: bool = True,
        **kwargs
    ) -> Generator[Union[str, Dict[str, Any]], None, None]:
        """
        Generate a streaming response with tool execution support.
        
        This method handles a complex streaming workflow that includes:
        1. Streaming initial response content from the LLM
        2. Detecting tool calls during streaming
        3. Executing detected tool calls
        4. Yielding both content chunks and tool execution results
        5. Buffering content for state management
        6. Managing the conversation flow with tool results
        7. Generating a follow-up response after tool execution
        
        The streaming state machine works as follows:
        - Initial state: Process incoming chunks from the LLM
        - When a tool call is detected: Execute the tool and yield a tool result
        - After all chunks are processed: Add the accumulated content as an assistant message
        - If tools were executed: Generate and stream a follow-up response
        - Final state: Add the follow-up response as an assistant message
        
        Note: Tool results formatting varies by provider. Some providers (like OpenAI) support
        dedicated roles for tool outputs, while others (like Anthropic) require formatting
        tool outputs as assistant messages.
        
        Args:
            tool_functions: A dictionary mapping tool names to their implementation functions
            prompt: The input prompt (if None, uses the existing conversation history)
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate
            top_p: The top_p value to use
            frequency_penalty: The frequency penalty to use
            presence_penalty: The presence penalty to use
            max_tool_calls: Maximum number of tool call iterations to prevent infinite loops
            adjust_system_prompt: Whether to adjust the system prompt based on tool execution phase
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Either string chunks for content or dictionaries for tool results.
            String chunks are yielded as-is for direct content display.
            Tool results are yielded as dictionaries with the following format:
            {
                "type": "tool_result",
                "tool_call": {
                    "call_id": "unique_id",
                    "name": "tool_name",
                    "arguments": {...},
                    "output": "result as string",
                    "error": "error message if any"
                }
            }
        """
        # Ensure we have tools available
        if not TOOLS_AVAILABLE:
            raise ValueError("Tool support is not available. Install the required dependencies.")
        
        # Store the original prompt and system prompt for follow-up requests
        original_prompt = prompt if prompt else ""
        original_system_prompt = self.system_prompt
        
        # Add user message if provided
        if prompt:
            self.add_message(MessageRole.USER, prompt)
            
        # Get the provider if needed
        provider = self._get_provider()
        
        # Variables to track state
        accumulated_content = ""    # Buffer for accumulating content chunks
        pending_tool_results = []   # Store tool results for later conversation state
        tool_call_count = 0         # Track tool execution count
        all_executed_tools = []     # Track all executed tools for system prompt adjustment
        current_system_prompt = original_system_prompt
        
        # Get the provider name to format messages properly
        provider_name = self._get_provider_name(provider)
        
        # Get conversation history formatted for this provider
        provider_messages = self.get_messages_for_provider(provider_name)
        
        logger = logging.getLogger("abstractllm.session")
        logger.info(f"Session: Starting streaming generation with tools")
        
        # Adjust system prompt for initial phase if needed
        if adjust_system_prompt:
            current_system_prompt = self._adjust_system_prompt_for_tool_phase(
                original_system_prompt, 
                0, 
                all_executed_tools, 
                "initial"
            )
            logger.debug(f"Session: Using initial phase system prompt for streaming")
        
        # Start streaming generation
        stream = provider.generate(
            prompt=original_prompt,  # Use original prompt consistently
            system_prompt=current_system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=self.tools,
            messages=provider_messages,
            stream=True,
            **kwargs
        )
        
        # Process the stream chunks one by one
        for chunk in stream:
            # 1) Raw text chunk (may be emitted as plain str)
            if isinstance(chunk, str):
                accumulated_content += chunk
                yield chunk
                continue

            # 2) Anthropic delta-style function/tool call event
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'tool_use'):
                # Check if we're at the tool call limit
                if tool_call_count >= max_tool_calls:
                    logger.warning(f"Session: Maximum tool call limit ({max_tool_calls}) reached. Skipping tool call.")
                    continue
                
                tool_call_count += 1
                tool_use = chunk.delta.tool_use
                call_id = getattr(tool_use, 'id', f"call_{len(pending_tool_results)}")
                name = getattr(tool_use, 'name', None) or ''
                args = getattr(tool_use, 'input', {})
                
                # Log tool call details
                logger.info(f"Session: Detected streaming tool call (iteration {tool_call_count}/{max_tool_calls}) - "
                           f"ID: {call_id}, Name: {name}, Args: {args}")
                
                # Build a ToolCall object for execution
                tool_call_obj = ToolCall(id=call_id, name=name, arguments=args)
                tool_result = self.execute_tool_call(tool_call_obj, tool_functions)
                pending_tool_results.append(tool_result)
                
                # Update executed tools list
                if name:
                    all_executed_tools.append(name)
                    # Remove duplicates while preserving order
                    all_executed_tools = list(dict.fromkeys(all_executed_tools))
                
                # Log tool execution result
                logger.info(f"Session: Executed streaming tool call - ID: {call_id}, "
                           f"Name: {name}, Result preview: {str(tool_result.get('output', ''))[:50]}...")
                
                # Yield unified tool_result dict
                yield {"type": "tool_result", "tool_call": tool_result}
                continue

            # 3) End-of-stream ToolCallRequest (Provider yields this after text)
            if isinstance(chunk, ToolCallRequest) and hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    # Check if we're at the tool call limit
                    if tool_call_count >= max_tool_calls:
                        logger.warning(f"Session: Maximum tool call limit ({max_tool_calls}) reached. Skipping tool call.")
                        continue
                    
                    tool_call_count += 1
                    
                    # Log tool call details
                    logger.info(f"Session: Detected end-of-stream tool call (iteration {tool_call_count}/{max_tool_calls}) - "
                               f"ID: {tool_call.id}, Name: {tool_call.name}, Args: {tool_call.arguments}")
                    
                    tool_result = self.execute_tool_call(tool_call, tool_functions)
                    pending_tool_results.append(tool_result)
                    
                    # Update executed tools list
                    if hasattr(tool_call, 'name') and tool_call.name:
                        all_executed_tools.append(tool_call.name)
                        # Remove duplicates while preserving order
                        all_executed_tools = list(dict.fromkeys(all_executed_tools))
                    
                    # Log tool execution result
                    logger.info(f"Session: Executed end-of-stream tool call - ID: {tool_call.id}, "
                               f"Name: {tool_call.name}, Result preview: {str(tool_result.get('output', ''))[:50]}...")
                    
                    yield {"type": "tool_result", "tool_call": tool_result}
                continue

            # 4) Standard content in structured chunk
            if hasattr(chunk, 'content') and chunk.content:
                accumulated_content += chunk.content
                yield chunk.content
        
        # After initial streaming is complete, add the final message with any tool results
        if accumulated_content:
            # Calculate metrics if tools were executed
            metrics = None
            if pending_tool_results:
                metrics = self._track_tool_execution_metrics(pending_tool_results)
                logger.info(f"Session: Tool execution metrics - "
                           f"Success: {metrics['successful_tools']}/{metrics['total_tools']} "
                           f"({metrics['success_rate']:.0%})")
            
            # Add the assistant message with accumulated content and any tool results
            logger.info(f"Session: Adding assistant message with {len(pending_tool_results)} tool results")
            self.add_message(
                MessageRole.ASSISTANT,
                content=accumulated_content,
                tool_results=pending_tool_results if pending_tool_results else None,
                metadata={"tool_metrics": metrics} if metrics else None
            )
            
        # If we executed tools, generate a follow-up response to incorporate the results
        if pending_tool_results:
            # Adjust system prompt for synthesis phase
            if adjust_system_prompt:
                phase = "synthesis" if tool_call_count >= max_tool_calls - 1 else "processing"
                current_system_prompt = self._adjust_system_prompt_for_tool_phase(
                    original_system_prompt, 
                    tool_call_count, 
                    all_executed_tools, 
                    phase
                )
                logger.debug(f"Session: Adjusting to {phase} phase system prompt for streaming follow-up")
            
            # Get updated provider messages
            updated_provider_messages = self.get_messages_for_provider(provider_name)
            
            # Debug: Log message structure being sent to the provider
            logger.debug(f"Session: Sending follow-up messages to LLM: {updated_provider_messages}")
            logger.info(f"Session: Generating follow-up streaming response with {len(pending_tool_results)} tool results")
            
            # Generate follow-up response to incorporate tool results, using original prompt
            follow_up_stream = provider.generate(
                prompt=original_prompt,  # Use original prompt consistently
                system_prompt=current_system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tools=self.tools,
                messages=updated_provider_messages,
                stream=True,
                **kwargs
            )

            # Track follow-up content for conversation state
            follow_up_content = ""

            # Yield follow-up chunks for immediate display (handle str or .content)
            for chunk in follow_up_stream:
                if isinstance(chunk, str):
                    follow_up_content += chunk
                    yield chunk
                elif hasattr(chunk, "content") and chunk.content:
                    follow_up_content += chunk.content
                    yield chunk.content

            # Add the final follow-up response to the conversation
            if follow_up_content:
                logger.info(f"Session: Adding final follow-up assistant message")
                self.add_message(
                    MessageRole.ASSISTANT, 
                    follow_up_content,
                    metadata={
                        "tool_execution": {
                            "tool_call_count": tool_call_count,
                            "executed_tools": all_executed_tools,
                            "phase": "synthesis" if tool_call_count > 0 else "initial"
                        }
                    }
                )
        
        # Reset the system prompt to the original
        if adjust_system_prompt and self.system_prompt != original_system_prompt:
            self.system_prompt = original_system_prompt


class SessionManager:
    """
    Manages multiple conversation sessions.
    """
    
    def __init__(self, sessions_dir: Optional[str] = None):
        """
        Initialize the session manager.
        
        Args:
            sessions_dir: Directory to store session files
        """
        self.sessions: Dict[str, Session] = {}
        self.sessions_dir = sessions_dir
        
        # Create the sessions directory if it doesn't exist
        if sessions_dir and not os.path.exists(sessions_dir):
            os.makedirs(sessions_dir)
    
    def create_session(self, 
                      system_prompt: Optional[str] = None,
                      provider: Optional[Union[str, AbstractLLMInterface]] = None,
                      provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Session:
        """
        Create a new session.
        
        Args:
            system_prompt: The system prompt for the conversation
            provider: Provider name or instance to use for this session
            provider_config: Configuration for the provider
            metadata: Session metadata
            
        Returns:
            The created session
        """
        session = Session(
            system_prompt=system_prompt,
            provider=provider,
            provider_config=provider_config,
            metadata=metadata
        )
        
        self.sessions[session.id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if the session was deleted, False otherwise
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            # Delete the session file if it exists
            if self.sessions_dir:
                filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
            return True
        
        return False
    
    def list_sessions(self) -> List[Tuple[str, datetime, datetime]]:
        """
        List all sessions.
        
        Returns:
            List of (session_id, created_at, last_updated) tuples
        """
        return [(s.id, s.created_at, s.last_updated) for s in self.sessions.values()]
    
    def save_all(self) -> None:
        """
        Save all sessions to disk.
        """
        if not self.sessions_dir:
            raise ValueError("No sessions directory specified")
        
        for session_id, session in self.sessions.items():
            filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
            session.save(filepath)
    
    def load_all(self, 
                provider: Optional[Union[str, AbstractLLMInterface]] = None,
                provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None) -> None:
        """
        Load all sessions from disk.
        
        Args:
            provider: Provider to use for the loaded sessions
            provider_config: Configuration for the provider
        """
        if not self.sessions_dir or not os.path.exists(self.sessions_dir):
            return
        
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.sessions_dir, filename)
                session = Session.load(
                    filepath, 
                    provider=provider,
                    provider_config=provider_config
                )
                self.sessions[session.id] = session 