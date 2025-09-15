"""
Enhanced Session management for AbstractLLM with integrated SOTA features.

This module provides comprehensive session management for LLM conversations,
integrating both core functionality and SOTA enhancements:

Core Features:
- Conversation history management with metadata
- Tool calling with enhanced telemetry
- Provider switching with state preservation  
- Session persistence and loading

SOTA Features (when available):
- Hierarchical memory system with fact extraction
- ReAct reasoning cycles with complete observability
- Retry strategies with exponential backoff
- Structured response parsing and validation
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Generator, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
import json
import os
import uuid
import logging
import time

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.exceptions import UnsupportedFeatureError
from abstractllm.enums import MessageRole
from abstractllm.utils.context_logging import log_llm_interaction

# Handle circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from abstractllm.tools import ToolDefinition, ToolCall, ToolResult, ToolCallResponse
    from abstractllm.types import GenerateResponse

# Try importing tools package directly
try:
    from abstractllm.tools import (
        ToolDefinition,
        ToolCall,
        ToolResult,
        ToolCallResponse,
        register
    )
    from abstractllm.types import GenerateResponse
    # For backwards compatibility
    ToolCallRequest = ToolCallResponse  # Alias for old code
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    # Define concrete error message with specific installation instructions
    TOOLS_ERROR_MESSAGE = (
        "Tool support is not available. Please install the required dependencies:\n"
        "1. pip install jsonschema\n"
        "2. pip install docstring-parser\n"
        "3. pip install pydantic\n\n"
        "For convenience, you can install all tool dependencies with:\n"
        "pip install abstractllm[tools]\n\n"
        f"Original error: {str(e)}"
    )
    if not TYPE_CHECKING:
        class ToolDefinition:
            pass
        class ToolCall:
            pass
        class ToolResult:
            pass
        class ToolCallResponse:
            pass
        ToolCallRequest = ToolCallResponse  # Alias
        class GenerateResponse:
            pass

# Try importing SOTA improvements (optional)
try:
    from abstractllm.memory import HierarchicalMemory, ReActCycle, MemoryComponent
    from abstractllm.retry_strategies import RetryManager, RetryConfig, with_retry
    from abstractllm.structured_response import (
        StructuredResponseHandler, 
        StructuredResponseConfig,
        ResponseFormat
    )
    from abstractllm.scratchpad_manager import (
        ScratchpadManager, 
        get_scratchpad_manager,
        ReActPhase,
        CyclePhaseEvent
    )
    SOTA_FEATURES_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.debug("SOTA features loaded successfully")
except ImportError as e:
    SOTA_FEATURES_AVAILABLE = False
    # Create placeholder classes for graceful degradation
    class HierarchicalMemory:
        def __init__(self, *args, **kwargs):
            pass
        def get_statistics(self):
            return {"error": "SOTA features not available"}
        def get_context_for_query(self, query):
            return None
        def start_react_cycle(self, query, max_iterations=10):
            return None
        def add_chat_message(self, role, content, cycle_id=None):
            return str(uuid.uuid4())
        @property
        def semantic_memory(self):
            return {}
        def save_to_disk(self):
            pass
        def visualize_links(self):
            return None
        def query_memory(self, query):
            return None
    class ReActCycle:
        def __init__(self, *args, **kwargs):
            self.cycle_id = str(uuid.uuid4())
            self.error = None
        def add_thought(self, thought, confidence=1.0):
            pass
        def add_action(self, tool_name, arguments, reasoning):
            return str(uuid.uuid4())
        def add_observation(self, action_id, content, success=True):
            pass
        def complete(self, content, success=True):
            pass
    class RetryManager:
        def __init__(self, *args, **kwargs):
            pass
        def retry_with_backoff(self, func, *args, **kwargs):
            return func(*args, **kwargs)
    class RetryConfig:
        def __init__(self, *args, **kwargs):
            pass
    class StructuredResponseHandler:
        def __init__(self, *args, **kwargs):
            pass
        def prepare_request(self, prompt, config, system_prompt=None):
            return {"prompt": prompt, "system_prompt": system_prompt}
        def parse_response(self, response, config):
            return response
        def generate_with_retry(self, generate_fn, prompt, config, **kwargs):
            return generate_fn(prompt=prompt, **kwargs)
    class StructuredResponseConfig:
        def __init__(self, *args, **kwargs):
            self.max_retries = 0
    def with_retry(func):
        return func
    def get_scratchpad_manager(*args, **kwargs):
        return ScratchpadManager()
    class ScratchpadManager:
        def __init__(self, *args, **kwargs):
            pass
        def start_cycle(self, cycle_id, prompt):
            pass
        def add_thought(self, thought, confidence=1.0, metadata=None):
            pass
        def add_action(self, tool_name, tool_args, reasoning, metadata=None):
            return str(uuid.uuid4())
        def add_observation(self, action_id, result, success=True, execution_time=0, metadata=None):
            pass
        def complete_cycle(self, final_answer, success=True):
            pass
        def get_complete_trace(self):
            return []
        def get_scratchpad_file_path(self):
            return Path("./memory/scratchpad.json")

logger = logging.getLogger(__name__)


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
    Enhanced session with comprehensive LLM conversation management and optional SOTA features.
    
    This class provides both core session functionality and optional SOTA enhancements:
    
    Core Features:
    - Conversation history management with metadata
    - Tool calling with enhanced telemetry tracking  
    - Provider switching with state preservation
    - Session persistence and loading
    - Multi-provider message formatting
    
    SOTA Features (when dependencies available):
    - Hierarchical memory system with fact extraction
    - ReAct reasoning cycles with complete observability 
    - Retry strategies with exponential backoff
    - Structured response parsing and validation
    """
    
    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 provider: Optional[Union[str, AbstractLLMInterface]] = None,
                 provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tools: Optional[List[Union[Dict[str, Any], Callable, "ToolDefinition"]]] = None,
                 # SOTA enhancement parameters (optional)
                 enable_memory: bool = True,
                 memory_config: Optional[Dict[str, Any]] = None,
                 enable_retry: bool = True,
                 retry_config: Optional[RetryConfig] = None,
                 persist_memory: Optional[Path] = None,
                 max_tool_calls: int = 25):
        """
        Initialize a comprehensive session with optional SOTA features.
        
        Args:
            system_prompt: The system prompt for the conversation
            provider: Provider name or instance to use for this session
            provider_config: Configuration for the provider
            metadata: Session metadata
            tools: Optional list of tool definitions or functions available for the LLM to use.
                  Functions will be automatically converted to tool definitions and their
                  implementations stored for use with generate_with_tools.
            enable_memory: Enable hierarchical memory system (if available)
            memory_config: Memory system configuration
            enable_retry: Enable retry strategies (if available)
            retry_config: Retry configuration
            persist_memory: Path to persist memory
            max_tool_calls: Maximum number of tool call iterations per generation (default: 25)
                  
        Raises:
            ValueError: If tools are provided but tool support is not available
        """
        # Initialize the provider first so we can check for deterministic mode
        self._provider: Optional[AbstractLLMInterface] = None
        if provider is not None:
            if isinstance(provider, str):
                from abstractllm import create_llm
                self._provider = create_llm(provider, **(provider_config or {}))
            else:
                self._provider = provider

        # Core session initialization - use deterministic values if seed is set
        self.messages: List[Message] = []
        self.system_prompt = system_prompt
        self.metadata = metadata or {}

        # Use deterministic values if seed is set
        if self._is_deterministic_mode():
            # Generate deterministic session ID based on seed
            seed = self._get_current_seed()
            import hashlib
            seed_str = str(seed) if seed is not None else "default"
            self.id = hashlib.md5(f"session_{seed_str}".encode()).hexdigest()
            # Use fixed timestamp for deterministic generation
            self.created_at = datetime.fromtimestamp(1609459200)  # 2021-01-01 00:00:00 UTC
        else:
            # Use random values for normal operation
            self.id = str(uuid.uuid4())
            self.created_at = datetime.now()

        self.last_updated = self.created_at
        self.tools: List["ToolDefinition"] = []

        # Store the original function implementations for tools
        self._tool_implementations: Dict[str, Callable[..., Any]] = {}

        # Track last assistant message index for tool results
        self._last_assistant_idx = -1

        # Store max_tool_calls configuration
        self.max_tool_calls = max_tool_calls
        
        # Add system message if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM, system_prompt)
            
        # Add tools if provided - but defer validation until add_tool is called
        if tools:
            # Only check tool availability when tools are actually provided
            if not TOOLS_AVAILABLE:
                raise ValueError(TOOLS_ERROR_MESSAGE)
                
            # Register each tool
            for tool in tools:
                self.add_tool(tool)

        # Initialize SOTA features if available
        self.enable_memory = enable_memory and SOTA_FEATURES_AVAILABLE
        self.enable_retry = enable_retry and SOTA_FEATURES_AVAILABLE
        
        if self.enable_memory:
            try:
                memory_cfg = memory_config or {}
                self.memory = HierarchicalMemory(
                    working_memory_size=memory_cfg.get('working_memory_size', 10),
                    episodic_consolidation_threshold=memory_cfg.get('consolidation_threshold', 5),
                    persist_path=persist_memory,
                    session=self
                )
                logger.debug("Hierarchical memory initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hierarchical memory: {e}")
                self.memory = None
                self.enable_memory = False
        else:
            self.memory = None
        
        if self.enable_retry:
            try:
                self.retry_manager = RetryManager(retry_config or RetryConfig())
                logger.debug("Retry manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize retry manager: {e}")
                self.retry_manager = None
                self.enable_retry = False
        else:
            self.retry_manager = None
        
        # Track current ReAct cycle (SOTA feature)
        self.current_cycle: Optional[ReActCycle] = None
        
        # Structured response handlers per provider (SOTA feature)
        self.response_handlers: Dict[str, StructuredResponseHandler] = {}
        
        # Enhanced telemetry tracking (SOTA feature)
        self.current_tool_traces: List[Dict[str, Any]] = []
        self.current_retry_attempts: int = 0
        self.facts_before_generation: int = 0
        
        # SOTA Scratchpad Manager with complete observability (if available)
        if SOTA_FEATURES_AVAILABLE and persist_memory:
            try:
                memory_folder = persist_memory.parent if persist_memory else Path("./memory")
                session_id = self.memory.session_id if self.memory else f"session_{self.id[:8]}"
                self.scratchpad = get_scratchpad_manager(session_id, memory_folder)
                logger.debug("Scratchpad manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize scratchpad manager: {e}")
                self.scratchpad = None
        else:
            self.scratchpad = None

    def _is_deterministic_mode(self) -> bool:
        """Check if the session is in deterministic mode (seed is set)."""
        if not self._provider:
            return False
        try:
            seed = self._provider.config_manager.get_param(ModelParameter.SEED)
            return seed is not None
        except:
            return False

    def _get_current_seed(self) -> Optional[int]:
        """Get the current seed value if set."""
        if not self._provider:
            return None
        try:
            return self._provider.config_manager.get_param(ModelParameter.SEED)
        except:
            return None

    @property
    def provider(self) -> Optional[AbstractLLMInterface]:
        """Get the current provider instance."""
        return self._provider
    
    @provider.setter
    def provider(self, value: Optional[Union[str, AbstractLLMInterface]]) -> None:
        """Set the provider instance."""
        if value is None:
            self._provider = None
        elif isinstance(value, str):
            from abstractllm import create_llm
            self._provider = create_llm(value)
        else:
            self._provider = value
    
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
                    elif provider_name == "mlx":
                        # MLX expects tool results in a specific format that indicates successful execution
                        formatted.append({
                            "role": "tool",
                            "name": tool_name, 
                            "content": output
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
        
        This is a wrapper around the unified generate method.
        
        Args:
            message: The message to send
            provider: Provider to use (overrides the session provider)
            stream: Whether to stream the response
            **kwargs: Additional parameters for the provider
            
        Returns:
            The LLM's response
        """
        return self.generate(
            prompt=message,
            provider=provider,
            stream=stream,
            **kwargs
        )
    
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
                    response = await llm.generate_async(messages=messages, stream=stream, tools=self.tools, **kwargs)
                elif provider_name == "anthropic":
                    response = await llm.generate_async(messages=messages, stream=stream, tools=self.tools, **kwargs)
                else:
                    # Default approach for other providers that support chat
                    response = await llm.generate_async(messages=messages, stream=stream, tools=self.tools, **kwargs)
            else:
                # For providers that don't support chat history, format a prompt
                formatted_prompt = self.get_formatted_prompt()
                response = await llm.generate_async(
                    formatted_prompt, 
                    system_prompt=self.system_prompt,
                    stream=stream,
                    tools=self.tools,  # Pass tools to provider
                    **kwargs
                )
            
            # If not streaming, add the response to the conversation
            if not stream:
                self.add_message(MessageRole.ASSISTANT, response)
                
            return response
        
        return _async_handler()
    
    def save(self, filepath: str) -> None:
        """
        Save the session to a file, including current provider state.
        
        Args:
            filepath: Path to save the session to
        """
        # Basic session data
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages]
        }
        
        # Save provider state if available
        if self.provider is not None:
            provider_name = self._get_provider_name(self.provider)
            model_name = self._get_provider_model(self.provider)
            
            # Extract provider config - be careful with different provider types
            provider_config = {}
            if hasattr(self.provider, 'config') and self.provider.config:
                # Make a copy and serialize only JSON-serializable values
                for key, value in self.provider.config.items():
                    # Convert ModelParameter enum keys to strings for JSON serialization
                    str_key = str(key) if hasattr(key, 'name') else key
                    
                    # Only include serializable values
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        provider_config[str_key] = value
                    elif hasattr(value, '__name__'):  # For enum values
                        provider_config[str_key] = str(value)
            
            data["provider_state"] = {
                "provider_name": provider_name,
                "model_name": model_name,
                "provider_config": provider_config
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the session.
        
        Returns:
            Dictionary containing session statistics including:
            - Session info (id, created_at, last_updated, duration)
            - Message statistics (total, by role, average length)
            - Tool usage statistics (if applicable)
            - Provider information
            - Token statistics (automatically computed for missing data)
        """
        from abstractllm.utils.utilities import get_session_stats
        return get_session_stats(self)
    
    @classmethod
    def load(cls, filepath: str, 
             provider: Optional[Union[str, AbstractLLMInterface]] = None,
             provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None) -> 'Session':
        """
        Load a session from a file, automatically restoring provider state if saved.
        
        Args:
            filepath: Path to load the session from
            provider: Provider to use (only if no provider was saved or as override)
            provider_config: Configuration for the provider (only if no config was saved or as override)
            
        Returns:
            A Session instance with restored provider state
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if provider state was saved
        saved_provider_state = data.get("provider_state")
        
        # Determine which provider to use
        provider_to_use = provider
        config_to_use = provider_config
        
        if saved_provider_state and provider is None:
            # Restore saved provider if no override provided
            saved_provider_name = saved_provider_state.get("provider_name")
            saved_model_name = saved_provider_state.get("model_name")
            saved_config = saved_provider_state.get("provider_config", {})
            
            if saved_provider_name and saved_model_name:
                try:                    
                    # Convert string keys back to ModelParameter enums where applicable
                    restored_config = {}
                    for key, value in saved_config.items():
                        # Try to convert back to ModelParameter enum
                        try:
                            if hasattr(ModelParameter, key):
                                restored_config[getattr(ModelParameter, key)] = value
                            else:
                                restored_config[key] = value
                        except AttributeError:
                            restored_config[key] = value
                    
                    # Add the model to config
                    restored_config[ModelParameter.MODEL] = saved_model_name
                    
                    from abstractllm import create_llm
                    provider_to_use = create_llm(saved_provider_name, **restored_config)
                    config_to_use = restored_config
                    
                    # Load the model immediately (like we do at startup)
                    print(f"🔄 Loading restored model...")
                    provider_to_use.load_model()
                    print(f"✅ Restored provider: {saved_provider_name} with model: {saved_model_name}")
                    
                except Exception as e:
                    print(f"⚠️  Could not restore saved provider {saved_provider_name}:{saved_model_name}: {e}")
                    print("📝 Session will be loaded without provider - you can set one manually")
                    provider_to_use = None
                    config_to_use = None
        elif provider_to_use and hasattr(provider_to_use, 'load_model'):
            # If we have a provider (either provided or already instantiated), make sure it's loaded
            try:
                print(f"🔄 Loading model...")
                provider_to_use.load_model()
                print(f"✅ Model loaded and ready!")
            except Exception as e:
                print(f"⚠️  Warning: Could not load model: {e}")
        
        # Create a new session
        session = cls(
            system_prompt=data.get("system_prompt"),
            provider=provider_to_use,
            provider_config=config_to_use,
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
                from abstractllm import create_llm
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
            raise ValueError(TOOLS_ERROR_MESSAGE)
        
        # Convert tool to ToolDefinition and store original implementation if callable
        if callable(tool):
            # Store the original function implementation
            func_name = tool.__name__
            self._tool_implementations[func_name] = tool
            
            # Convert function to tool definition using new method
            tool_def = ToolDefinition.from_function(tool)
        elif isinstance(tool, dict):
            # Convert dictionary to tool definition
            tool_def = ToolDefinition(**tool)
        else:
            # Already a ToolDefinition
            tool_def = tool
            
        self.tools.append(tool_def)
        # The provider will validate tool support when generate is called
    
    def execute_tool_call(self,
                         tool_call: "ToolCall",
                         tool_functions: Dict[str, Callable[..., Any]]) -> Dict[str, Any]:
        """
        Execute a tool call using the provided functions with enhanced retry and memory tracking.
        
        Args:
            tool_call: The tool call to execute
            tool_functions: Dictionary of available tool functions
            
        Returns:
            Dictionary containing the tool result or error
        """
        logger = logging.getLogger("abstractllm.session")
        logger.info(f"Session: Executing tool call: {tool_call.name} with args: {tool_call.arguments}")
        
        start_time = time.time()
        
        # Track in ReAct cycle if enabled
        if self.current_cycle:
            action_id = self.current_cycle.add_action(
                tool_name=tool_call.name,
                arguments=tool_call.arguments if hasattr(tool_call, 'arguments') else {},
                reasoning=f"Executing {tool_call.name} to gather information"
            )
        else:
            action_id = f"action_{len(self.current_tool_traces)}"
            
        # Add to scratchpad with complete observability if available
        scratchpad_action_id = None
        if self.scratchpad:
            scratchpad_action_id = self.scratchpad.add_action(
                tool_name=tool_call.name,
                tool_args=tool_call.arguments if hasattr(tool_call, 'arguments') else {},
                reasoning=f"Executing {tool_call.name} to gather information",
                metadata={"session_action_id": action_id}
            )
        
        # Execute with retry if enabled
        success = True
        error = None
        
        def _execute_tool():
            """Internal function to execute tool call."""
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
        
        if self.enable_retry:
            try:
                result = self.retry_manager.retry_with_backoff(
                    _execute_tool,
                    key=f"tool_{tool_call.name}"
                )
                # Track retry attempts
                retry_attempts = getattr(self.retry_manager, '_current_attempts', {}).get(f"tool_{tool_call.name}", 0)
                self.current_retry_attempts += retry_attempts
            except Exception as e:
                success = False
                error = str(e)
                result = {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": None,
                    "error": str(e)
                }
                if self.current_cycle and action_id:
                    self.current_cycle.add_observation(
                        action_id=action_id,
                        content=str(e),
                        success=False
                    )
        else:
            try:
                result = _execute_tool()
                if result.get("error"):
                    success = False
                    error = result["error"]
            except Exception as e:
                success = False
                error = str(e)
                result = {
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "output": None,
                    "error": str(e)
                }
        
        execution_time = time.time() - start_time
        
        # Create tool execution trace
        tool_trace = {
            "name": tool_call.name,
            "arguments": tool_call.arguments if hasattr(tool_call, 'arguments') else {},
            "result": str(result.get("output", result.get("error", result)))[:500],  # Truncate for readability
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "action_id": action_id,
            "reasoning": f"Executing {tool_call.name} to gather information",
            "error": error
        }
        
        # Add to current traces
        self.current_tool_traces.append(tool_trace)
        
        # Track observation in ReAct cycle
        if self.current_cycle and action_id:
            self.current_cycle.add_observation(
                action_id=action_id,
                content=result.get("output", result.get("error")),
                success=success
            )
            
        # Add observation to scratchpad (COMPLETE - no truncation)
        if self.scratchpad and scratchpad_action_id:
            self.scratchpad.add_observation(
                action_id=scratchpad_action_id,
                result=result.get("output", result.get("error", result)),
                success=success,
                execution_time=execution_time,
                metadata={"session_action_id": action_id}
            )
        
        return result
    
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
            
        # Log tool calls being executed using universal logging (if provider supports it)
        if hasattr(self._provider, '_log_tool_calls_found') and tool_calls:
            self._provider._log_tool_calls_found(tool_calls)
        
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
        base_prompt = (original_system_prompt or "").strip()
        
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

    def _create_tool_functions_dict(self) -> Dict[str, Callable[..., Any]]:
        """
        Create a dictionary of tool functions from registered tool definitions.
        
        This uses stored function implementations when available, or creates
        placeholder functions that raise NotImplementedError when no implementation
        is available.
        
        Returns:
            Dictionary mapping tool names to functions (actual or placeholder)
        """
        logger = logging.getLogger("abstractllm.session")
        tool_functions = {}
        
        for tool in self.tools:
            # Get the tool name from different possible formats
            if hasattr(tool, 'name'):
                tool_name = tool.name
            elif isinstance(tool, dict):
                tool_name = tool.get('name', 'unknown')
            else:
                tool_name = 'unknown'
            
            # Use the stored implementation if available
            if tool_name in self._tool_implementations:
                tool_functions[tool_name] = self._tool_implementations[tool_name]
                logger.debug(f"Session: Using stored implementation for tool '{tool_name}'")
            else:
                # Create a placeholder function with proper closure to capture the tool name
                def create_placeholder(name):
                    def placeholder_func(*args, **kwargs):
                        raise NotImplementedError(
                            f"No implementation provided for tool '{name}'. "
                            f"Please provide a tool_functions dictionary with an implementation for this tool."
                        )
                    return placeholder_func
                
                tool_functions[tool_name] = create_placeholder(tool_name)
                logger.debug(f"Session: Created placeholder function for tool '{tool_name}'")
            
        return tool_functions
    
    def _process_tool_response(self, response: Any) -> Optional[Any]:
        """
        Compatibility layer to handle both old and new tool response formats.
        
        Args:
            response: Either ToolCallResponse (new) or response with tool_calls attribute (old)
            
        Returns:
            Tool calls in a normalized format, or None if no tools
        """
        # New format: ToolCallResponse object
        if hasattr(response, 'has_tool_calls') and callable(response.has_tool_calls):
            if response.has_tool_calls():
                return response
            return None
            
        # Old format: Direct tool_calls attribute
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Wrap in ToolCallResponse for compatibility
            if TOOLS_AVAILABLE:
                return ToolCallResponse(
                    content=getattr(response, 'content', ''),
                    tool_calls=response.tool_calls
                )
            return response
            
        # No tool calls
        return None
        
    def generate_with_tools(
        self,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,  # Added for consistency
        prompt: Optional[str] = None,
        provider: Optional[Union[str, AbstractLLMInterface]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tool_calls: Optional[int] = None,
        adjust_system_prompt: bool = False,  # DISABLED - was breaking tool definitions
        system_prompt: Optional[str] = None,
        files: Optional[List[Union[str, Path]]] = None,
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
            tool_functions: A dictionary mapping tool names to their implementation functions.
                            If None, uses placeholder functions for registered tools.
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
            raise ValueError(TOOLS_ERROR_MESSAGE)
        
        # Use session's default max_tool_calls if not specified
        if max_tool_calls is None:
            max_tool_calls = self.max_tool_calls
        
        # Store the original prompt and system prompt for follow-up requests
        original_prompt = prompt if prompt else ""
        original_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        
        # Track executed tools for metrics and prompt adjustment
        all_executed_tools = []
        current_system_prompt = original_system_prompt
        
        # Add user message if provided
        if prompt:
            self.add_message(MessageRole.USER, prompt)
            
        # Get the provider if needed
        provider_instance = self._get_provider(provider)
        
        # Get the provider name to format messages properly
        provider_name = self._get_provider_name(provider_instance)
        logger.info(f"Using provider: {provider_name}")
        
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
        
        logger.info(f"Generating initial response with tools. Provider: {provider_name}")

        # Log full context before generation - EXACT verbatim what goes to LLM
        model_name = self._get_provider_model(provider_instance)
        log_llm_interaction(
            prompt=original_prompt,  # Exact prompt sent to LLM, including all metadata
            system_prompt=current_system_prompt,
            messages=provider_messages,
            memory_context=None,  # Don't separate - show everything as it actually was sent
            tools=tools if tools is not None else self.tools,
            response=None,  # Will be updated after generation
            model=model_name,
            provider=provider_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 1) Initial generate with tools
        response = provider_instance.generate(
            prompt=original_prompt,
            system_prompt=current_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=tools if tools is not None else self.tools,  # Use provided tools if available
            messages=provider_messages,
            files=files,
            **kwargs
        )

        # Log response
        response_content = response.content if hasattr(response, 'content') else str(response)
        log_llm_interaction(
            prompt=original_prompt,  # Exact prompt sent to LLM, including all metadata
            system_prompt=current_system_prompt,
            messages=provider_messages,
            memory_context=None,  # Don't separate - show everything as it actually was sent
            tools=tools if tools is not None else self.tools,
            response=response_content,
            model=model_name,
            provider=provider_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        logger.info(f"Received initial response from LLM, type: {type(response)}")
        if hasattr(response, 'has_tool_calls'):
            logger.info(f"Response has_tool_calls method, result: {response.has_tool_calls()}")
        
        # Log raw response for debugging
        if hasattr(response, 'to_dict'):
            logger.debug(f"Raw response: {response.to_dict()}")
        elif isinstance(response, dict):
            logger.debug(f"Raw response: {response}")
        else:
            logger.debug(f"Raw response (content): {response}")

        # If no tool_functions were provided but we have registered tools, create placeholder functions
        if tool_functions is None:
            if not self.tools:
                logger.warning("Session: No tool_functions provided and no tools registered in the session")
                # No tools registered, return the response as is
                return response
            
            # Create tool functions from registered tools
            tool_functions = self._create_tool_functions_dict()
            logger.info(f"Session: Using {len(tool_functions)} registered tools from session")

        # 2) Loop: execute any tool calls and regenerate until no more tools requested
        tool_call_count = 0
        
        while hasattr(response, 'has_tool_calls') and response.has_tool_calls() and tool_call_count < max_tool_calls:
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
            response = provider_instance.generate(
                prompt=original_prompt,
                system_prompt=current_system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tools=tools if tools is not None else self.tools,  # Use provided tools if available
                messages=updated_provider_messages,
                files=files,
                **kwargs
            )
            logger.info(f"Session: Received follow-up response from LLM (iteration {tool_call_count})")
            
            # Log follow-up response for debugging
            if hasattr(response, 'to_dict'):
                logger.debug(f"Raw follow-up response: {response.to_dict()}")
            elif isinstance(response, dict):
                logger.debug(f"Raw follow-up response: {response}")
            else:
                logger.debug(f"Raw follow-up response (content): {response}")
        
        # Log if we hit the maximum tool call limit
        if tool_call_count >= max_tool_calls and hasattr(response, 'has_tool_calls') and response.has_tool_calls():
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
                if current_system_prompt != final_system_prompt and hasattr(response, 'has_tool_calls') and response.has_tool_calls():
                    updated_provider_messages = self.get_messages_for_provider(provider_name)
                    response = provider_instance.generate(
                        prompt=original_prompt,
                        system_prompt=final_system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        tools=tools if tools is not None else self.tools,  # Use provided tools if available
                        messages=updated_provider_messages,
                        files=files,
                        **kwargs
                    )
                    logger.info(f"Session: Generated final synthesis response after tool execution")
                    
                    # Log final response for debugging
                    if hasattr(response, 'to_dict'):
                        logger.debug(f"Raw final response: {response.to_dict()}")
                    elif isinstance(response, dict):
                        logger.debug(f"Raw final response: {response}")
                    else:
                        logger.debug(f"Raw final response (content): {response}")

        # 3) Final assistant response
        # Handle both response types - string or object with content attribute
        if isinstance(response, str):
            final_content = response
            logger.info("Response is a string")
            response_metadata = {}
        else:
            final_content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"Response has content attribute: {hasattr(response, 'content')}")
            
            # Extract usage information and other metadata from response
            response_metadata = {}
            if hasattr(response, 'usage') and response.usage:
                response_metadata["usage"] = response.usage
            if hasattr(response, 'model') and response.model:
                response_metadata["provider"] = self._get_provider_name(provider_instance)
                response_metadata["model"] = response.model

        # Combine response metadata with tool execution metadata
        combined_metadata = {
            "tool_execution": {
                "tool_call_count": tool_call_count,
                "executed_tools": all_executed_tools,
                "phase": "synthesis" if tool_call_count > 0 else "initial"
            }
        }
        combined_metadata.update(response_metadata)

        final_message = self.add_message(
            MessageRole.ASSISTANT, 
            final_content,
            metadata=combined_metadata
        )
        
        # Reset the system prompt to the original
        if adjust_system_prompt and self.system_prompt != original_system_prompt:
            self.system_prompt = original_system_prompt
            
        return response
        
    def generate_with_tools_streaming(
        self,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,  # Added for consistency
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[Union[str, AbstractLLMInterface]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tool_calls: int = 5,
        adjust_system_prompt: bool = False,  # DISABLED - was breaking tool definitions
        system_prompt: Optional[str] = None,
        files: Optional[List[Union[str, Path]]] = None,
        **kwargs
    ) -> Generator[Union[str, Dict[str, Any]], None, None]:
        """
        Generate a streaming response with tool execution support.
        
        This method handles streaming generation with tool calls:
        1. Stream initial response with tool definitions
        2. If a tool call is detected in the stream, execute it
        3. Add tool results to the conversation
        4. Continue streaming with follow-up content that incorporates tool results
        
        Args:
            tool_functions: A dictionary mapping tool names to their implementation functions.
                            If None, uses placeholder functions for registered tools.
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
            Content chunks (strings) or tool result dictionaries
        """
        # Ensure we have tools available
        if not TOOLS_AVAILABLE:
            raise ValueError(TOOLS_ERROR_MESSAGE)
        
        # Store the original prompt and system prompt for follow-up requests
        original_prompt = prompt if prompt else ""
        original_system_prompt = self.system_prompt
        
        # Add user message if provided
        if prompt:
            self.add_message(MessageRole.USER, prompt)
            
        # Get the provider if needed
        provider_instance = self._get_provider(provider)
        
        # Variables to track state
        accumulated_content = ""    # Buffer for accumulating content chunks
        pending_tool_results = []   # Store tool results for later conversation state
        tool_call_count = 0         # Track tool execution count
        all_executed_tools = []     # Track all executed tools for system prompt adjustment
        current_system_prompt = original_system_prompt
        
        # Get the provider name to format messages properly
        provider_name = self._get_provider_name(provider_instance)
        
        # Get conversation history formatted for this provider
        provider_messages = self.get_messages_for_provider(provider_name)
        
        logger = logging.getLogger("abstractllm.session")
        logger.info(f"Session: Starting streaming generation with tools")
        
        # If no model is specified, try to get it from the provider's config
        if model is None:
            provider_model = self._get_provider_model(provider_instance)
            if provider_model:
                model = provider_model
                logger.debug(f"Session: Using model {model} from provider config")
        
        # If no tool_functions were provided but we have registered tools, create placeholder functions
        if tool_functions is None:
            if not self.tools:
                logger.warning("Session: No tool_functions provided and no tools registered in the session")
            else:
                # Create tool functions from registered tools
                tool_functions = self._create_tool_functions_dict()
                logger.info(f"Session: Using {len(tool_functions)} registered tools from session")
        
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
        stream = provider_instance.generate(
            prompt=original_prompt,  # Use original prompt consistently
            system_prompt=current_system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=tools if tools is not None else self.tools,  # Use provided tools if available
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
            follow_up_stream = provider_instance.generate(
                prompt=original_prompt,  # Use original prompt consistently
                system_prompt=current_system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tools=tools if tools is not None else self.tools,  # Use provided tools if available
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

    def _get_provider_model(self, provider: AbstractLLMInterface) -> Optional[str]:
        """
        Get the model name from a provider instance.
        
        Args:
            provider: Provider instance
            
        Returns:
            Model name or None if not found
        """
        # Try getting config through the standard interface methods
        try:
            # First try the get_config() method
            if hasattr(provider, 'get_config'):
                config = provider.get_config()
                
                # Check for ModelParameter.MODEL in config dictionary
                if ModelParameter.MODEL in config:
                    return config[ModelParameter.MODEL]
                    
                # Check for 'model' as string key in config dictionary
                if 'model' in config:
                    return config['model']
        except Exception:
            pass
            
        # Try config_manager directly
        try:
            if hasattr(provider, 'config_manager'):
                model = provider.config_manager.get_param(ModelParameter.MODEL)
                if model:
                    return model
        except Exception:
            pass
            
        # Try direct config attribute (fallback for older providers)
        try:
            if hasattr(provider, 'config') and provider.config:
                # Check for 'model' in config dictionary
                if 'model' in provider.config:
                    return provider.config['model']
                    
                # Check for ModelParameter.MODEL in config dictionary
                if ModelParameter.MODEL in provider.config:
                    return provider.config[ModelParameter.MODEL]
        except Exception:
            pass
            
        # Check for model as a direct attribute (some providers might store it this way)
        try:
            if hasattr(provider, 'model'):
                return provider.model
        except Exception:
            pass
            
        # Check for _model_id attribute (MLX provider stores it this way after loading)
        try:
            if hasattr(provider, '_model_id'):
                return provider._model_id
        except Exception:
            pass
            
        # No model found
        return None

    def generate(
        self,
        prompt: Optional[str] = None,
        provider: Optional[Union[str, AbstractLLMInterface]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
        tools: Optional[List[Union[Dict[str, Any], Callable]]] = None,  # Added for consistency
        max_tool_calls: Optional[int] = None,
        adjust_system_prompt: bool = False,  # DISABLED - was breaking tool definitions
        stream: bool = False,
        files: Optional[List[Union[str, Path]]] = None,
        # SOTA parameters
        use_memory_context: bool = True,
        create_react_cycle: bool = True,
        structured_config: Optional[StructuredResponseConfig] = None,
        **kwargs
    ) -> Union[str, "GenerateResponse", Generator[Union[str, Dict[str, Any]], None, None]]:
        """
        Enhanced unified method to generate a response with or without tool support and SOTA features.
        
        This method intelligently handles all generation use cases:
        1. Simple text generation with no tools
        2. Generation with tool support including executing tools and follow-up
        3. Streaming response with or without tools
        4. Enhanced generation with memory context, ReAct cycles, and structured responses
        
        Args:
            prompt: The input prompt or user query
            provider: Provider to use (overrides the session provider)
            system_prompt: Override the system prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling value
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            tool_functions: Optional dictionary mapping tool names to their implementations
            max_tool_calls: Maximum number of tool call iterations
            adjust_system_prompt: Whether to adjust system prompt based on tool execution phase
            stream: Whether to stream the response
            files: Optional list of files to process
            use_memory_context: Include memory context (SOTA feature)
            create_react_cycle: Create ReAct cycle for this query (SOTA feature)
            structured_config: Structured response configuration (SOTA feature)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            - String response for simple generation
            - GenerateResponse for tool-using generation
            - Generator for streaming responses
        """
        logger = logging.getLogger("abstractllm.session")
        
        # Get provider
        provider_instance = self._get_provider(provider)
        provider_name = self._get_provider_name(provider_instance)
        logger.info(f"Using provider: {provider_name}")
        
        # Use session's default max_tool_calls if not specified
        if max_tool_calls is None:
            max_tool_calls = self.max_tool_calls
        
        # SOTA Enhancement: Initialize telemetry tracking
        generation_start_time = datetime.now()
        self.current_tool_traces = []  # Reset for this generation
        self.current_retry_attempts = 0
        
        # SOTA Enhancement: Track memory state before generation
        if self.enable_memory:
            self.facts_before_generation = len(self.memory.knowledge_graph.facts) if self.memory else 0
        
        # SOTA Enhancement: Start ReAct cycle if enabled
        if self.enable_memory and create_react_cycle and prompt:
            self.current_cycle = self.memory.start_react_cycle(
                query=prompt,
                max_iterations=max_tool_calls
            )
            # Add initial thought
            self.current_cycle.add_thought(
                f"Processing query with {provider_name} provider",
                confidence=1.0
            )
            
            # Start scratchpad cycle with complete observability
            if self.scratchpad:
                self.scratchpad.start_cycle(self.current_cycle.cycle_id, prompt)
                self.scratchpad.add_thought(
                    f"Processing query with {provider_name} provider",
                    confidence=1.0,
                    metadata={"provider": provider_name, "model": getattr(provider_instance, 'model', 'unknown')}
                )
        
        # SOTA Enhancement: Add memory context if enabled
        enhanced_prompt = prompt
        if self.enable_memory and use_memory_context and prompt:
            context = self.memory.get_context_for_query(prompt)
            if context:
                enhanced_prompt = f"{context}\n\nUser: {prompt}"
                logger.debug(f"Added memory context: {len(context)} chars")
        
        # SOTA Enhancement: Prepare for structured response if configured
        if structured_config and SOTA_FEATURES_AVAILABLE:
            handler = self._get_response_handler(provider_name)
            request_params = handler.prepare_request(
                prompt=enhanced_prompt,
                config=structured_config,
                system_prompt=system_prompt
            )
            enhanced_prompt = request_params.pop("prompt")
            system_prompt = request_params.pop("system_prompt", system_prompt)
            kwargs.update(request_params)
        
        # Determine if we should use tool functionality
        use_tools = False
        
        # Handle both tools and tool_functions parameters for consistency
        if tools is not None:
            # If tools parameter is provided, register them temporarily
            if not TOOLS_AVAILABLE:
                raise ValueError(TOOLS_ERROR_MESSAGE)
            
            # Create tool_functions from provided tools if needed
            if tool_functions is None and tools:
                tool_functions = {}
                for tool in tools:
                    if callable(tool):
                        tool_functions[tool.__name__] = tool
                    elif isinstance(tool, dict) and "name" in tool:
                        # For dictionary tools, we need a callable implementation
                        # This will be handled by _create_tool_functions_dict
                        pass
                    elif hasattr(tool, "name"):
                        # For ToolDefinition objects
                        tool_functions[tool.name] = tool
            
            # Add tools to session temporarily if not already there
            for tool in tools:
                if tool not in self.tools:
                    self.add_tool(tool)  # Use add_tool to ensure proper conversion
                    
            use_tools = True
            logger.info(f"Tool support enabled with {len(tools)} provided tools")
        elif tool_functions is not None or self.tools:
            if not TOOLS_AVAILABLE:
                raise ValueError(TOOLS_ERROR_MESSAGE)
            use_tools = True
            logger.info(f"Tool support enabled with {len(self.tools)} registered tools")
        
        # If streaming is requested with tools, use specialized streaming method
        if stream and use_tools:
            return self.generate_with_tools_streaming(
                prompt=enhanced_prompt,
                tool_functions=tool_functions,
                tools=tools,  # Pass tools parameter
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tool_calls=max_tool_calls,
                adjust_system_prompt=adjust_system_prompt,
                files=files,
                **kwargs
            )
        
        # For regular streaming without tools
        if stream and not use_tools:
            # Add the user message if provided
            if enhanced_prompt:
                self.add_message(MessageRole.USER, enhanced_prompt)
                
            # Get conversation history
            system_prompt_to_use = system_prompt or self.system_prompt
            messages = self.get_messages_for_provider(provider_name)
            
            # Generate streaming response
            stream_response = provider_instance.generate(
                prompt=enhanced_prompt,
                system_prompt=system_prompt_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True,
                tools=tools if tools is not None else self.tools,  # Pass tools to provider
                files=files,
                **kwargs
            )
            
            # Create generator that adds the assistant message at the end
            accumulated_content = ""
            final_response_metadata = {}
            
            def stream_wrapper():
                nonlocal accumulated_content, final_response_metadata
                for chunk in stream_response:
                    if isinstance(chunk, str):
                        accumulated_content += chunk
                        yield chunk
                    elif hasattr(chunk, "content"):
                        content_chunk = chunk.content
                        if content_chunk:
                            accumulated_content += content_chunk
                            yield content_chunk
                        
                        # Capture metadata from the chunk (especially the last one)
                        if hasattr(chunk, 'usage') and chunk.usage:
                            final_response_metadata["usage"] = chunk.usage
                        if hasattr(chunk, 'model') and chunk.model:
                            final_response_metadata["provider"] = provider_name
                            final_response_metadata["model"] = chunk.model
                
                # Add the final message to the conversation with metadata
                if accumulated_content:
                    self.add_message(MessageRole.ASSISTANT, accumulated_content, metadata=final_response_metadata)
            
            return stream_wrapper()
        
        # Define generation function for SOTA retry support
        def _generate():
            if use_tools:
                return self.generate_with_tools(
                    prompt=enhanced_prompt,
                    tool_functions=tool_functions,
                    tools=tools,  # Pass tools parameter
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tool_calls=max_tool_calls,
                    adjust_system_prompt=adjust_system_prompt,
                    provider=provider_instance,  # Use the provider instance directly
                    system_prompt=system_prompt,
                    files=files,
                    **kwargs
                )
            else:
                # Standard generation without tools
                # Add the user message if provided
                if enhanced_prompt:
                    self.add_message(MessageRole.USER, enhanced_prompt)
                    
                # Get conversation history
                system_prompt_to_use = system_prompt or self.system_prompt
                messages = self.get_messages_for_provider(provider_name)
                
                # Generate response
                response = provider_instance.generate(
                    prompt=enhanced_prompt,
                    system_prompt=system_prompt_to_use,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    tools=tools if tools is not None else self.tools,  # Pass tools to provider
                    files=files,
                    **kwargs
                )
                
                # Extract content and metadata from the response
                if isinstance(response, str):
                    content = response
                    metadata = {}
                else:
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract usage information and other metadata from response
                    metadata = {}
                    if hasattr(response, 'usage') and response.usage:
                        metadata["usage"] = response.usage
                    if hasattr(response, 'model') and response.model:
                        metadata["provider"] = provider_name
                        metadata["model"] = response.model
                
                # Add the response to the conversation with metadata
                self.add_message(MessageRole.ASSISTANT, content, metadata=metadata)
                
                return response
        
        # SOTA Enhancement: Apply retry if enabled
        if self.enable_retry:
            try:
                response = self.retry_manager.retry_with_backoff(
                    _generate,
                    key=f"{provider_name}_generate"
                )
            except Exception as e:
                if self.current_cycle:
                    self.current_cycle.error = str(e)
                    self.current_cycle.complete("Failed to generate response", success=False)
                raise
        else:
            response = _generate()
        
        # SOTA Enhancement: Parse structured response if configured
        if structured_config and SOTA_FEATURES_AVAILABLE:
            handler = self._get_response_handler(provider_name)
            try:
                response = handler.parse_response(response, structured_config)
            except Exception as e:
                logger.error(f"Failed to parse structured response: {e}")
                if self.enable_retry and structured_config.max_retries > 0:
                    # Retry with feedback
                    response = handler.generate_with_retry(
                        generate_fn=provider_instance.generate,
                        prompt=prompt,
                        config=structured_config,
                        system_prompt=system_prompt,
                        **kwargs
                    )
        
        # SOTA Enhancement: Calculate total generation time
        total_generation_time = (datetime.now() - generation_start_time).total_seconds()
        
        # SOTA Enhancement: Update memory if enabled
        extracted_facts = []
        
        if self.enable_memory and prompt:
            # Track facts extracted before adding new messages
            facts_before = len(self.memory.knowledge_graph.facts)
            
            # Add to chat history (this triggers fact extraction)
            msg_id = self.memory.add_chat_message(
                role="user",
                content=prompt,
                cycle_id=self.current_cycle.cycle_id if self.current_cycle else None
            )
            
            # Add response
            response_content = str(response)
            resp_id = self.memory.add_chat_message(
                role="assistant",
                content=response_content,
                cycle_id=self.current_cycle.cycle_id if self.current_cycle else None
            )
            
            # Calculate facts extracted during this generation
            facts_after = len(self.memory.knowledge_graph.facts)
            new_facts_count = facts_after - self.facts_before_generation
            
            # Get the actual facts that were extracted
            if new_facts_count > 0:
                # Get the most recent facts
                all_facts = list(self.memory.knowledge_graph.facts.values())
                extracted_facts = all_facts[-new_facts_count:]
            
            # Complete ReAct cycle
            if self.current_cycle:
                self.current_cycle.complete(response_content, success=True)
                
                # Complete scratchpad cycle with final answer
                if self.scratchpad:
                    self.scratchpad.complete_cycle(
                        final_answer=response_content,
                        success=True
                    )
        
        # SOTA Enhancement: Build enhanced response with telemetry if available
        if SOTA_FEATURES_AVAILABLE and (self.enable_memory or self.current_tool_traces):
            enhanced_response = self._build_enhanced_response(
                base_response=response,
                provider_name=provider_name,
                total_time=total_generation_time,
                extracted_facts=extracted_facts
            )
        else:
            enhanced_response = response
        
        # Clear current cycle after building response
        if self.current_cycle:
            self.current_cycle = None
        
        return enhanced_response

    def get_last_interactions(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get the last N interactions as structured data.
        
        Args:
            count: Number of interactions to retrieve
            
        Returns:
            List of interaction dictionaries with message pairs and metadata,
            ordered from most recent to oldest
        """
        if count < 1:
            raise ValueError("Count must be positive")
        
        interactions = []
        messages = [msg for msg in self.messages if msg.role != MessageRole.SYSTEM]
        
        # Group messages into interactions (user message + assistant response + any tool results)
        i = 0
        while i < len(messages):
            interaction = {}
            
            # Look for user message
            if i < len(messages) and messages[i].role == MessageRole.USER:
                interaction["user"] = {
                    "content": messages[i].content,
                    "timestamp": messages[i].timestamp,
                    "metadata": messages[i].metadata
                }
                i += 1
                
                # Look for assistant response
                if i < len(messages) and messages[i].role == MessageRole.ASSISTANT:
                    interaction["assistant"] = {
                        "content": messages[i].content,
                        "timestamp": messages[i].timestamp,
                        "metadata": messages[i].metadata,
                        "tool_results": messages[i].tool_results
                    }
                    i += 1
                    
                    # Look for any tool messages that follow
                    tool_messages = []
                    while i < len(messages) and messages[i].role == MessageRole.TOOL:
                        tool_messages.append({
                            "content": messages[i].content,
                            "timestamp": messages[i].timestamp,
                            "metadata": messages[i].metadata
                        })
                        i += 1
                    
                    if tool_messages:
                        interaction["tools"] = tool_messages
                
                interactions.append(interaction)
            else:
                i += 1
        
        # Get the most recent interactions and return them in reverse chronological order (most recent first)
        recent_interactions = interactions[-count:] if len(interactions) >= count else interactions
        return list(reversed(recent_interactions))

    def get_system_prompt_info(self) -> Dict[str, Any]:
        """
        Get system prompt information as structured data.
        
        Returns:
            Dictionary containing system prompt details
        """
        return {
            "has_system_prompt": self.system_prompt is not None,
            "system_prompt": self.system_prompt,
            "character_count": len(self.system_prompt) if self.system_prompt else 0,
            "line_count": self.system_prompt.count('\n') + 1 if self.system_prompt else 0
        }

    def update_system_prompt(self, new_prompt: str) -> Dict[str, Any]:
        """
        Update the session's system prompt.
        
        Args:
            new_prompt: New system prompt text
            
        Returns:
            Dictionary with update result and details
        """
        if not new_prompt.strip():
            return {
                "success": False,
                "error": "System prompt cannot be empty",
                "old_prompt": self.system_prompt,
                "new_prompt": None
            }
        
        old_prompt = self.system_prompt
        self.system_prompt = new_prompt.strip()
        
        return {
            "success": True,
            "error": None,
            "old_prompt": old_prompt,
            "new_prompt": self.system_prompt,
            "old_length": len(old_prompt) if old_prompt else 0,
            "new_length": len(self.system_prompt)
        }

    def get_tools_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools as structured data.
        
        Returns:
            List of tool dictionaries with details
        """
        tools_list = []
        
        for tool_def in self.tools:
            if hasattr(tool_def, 'to_dict'):
                # ToolDefinition object
                tool_dict = tool_def.to_dict()
                tools_list.append({
                    "name": tool_dict.get("name", "Unknown"),
                    "description": tool_dict.get("description", "No description available"),
                    "parameters": tool_dict.get("input_schema", {}),
                    "source": "ToolDefinition"
                })
            elif callable(tool_def):
                # Function object
                tools_list.append({
                    "name": getattr(tool_def, '__name__', 'Unknown'),
                    "description": getattr(tool_def, '__doc__', 'No description available') or 'No description available',
                    "parameters": {},  # Functions don't have structured parameters in this implementation
                    "source": "Function"
                })
            elif isinstance(tool_def, dict):
                # Dictionary tool definition
                tools_list.append({
                    "name": tool_def.get("name", "Unknown"),
                    "description": tool_def.get("description", "No description available"),
                    "parameters": tool_def.get("parameters", {}),
                    "source": "Dictionary"
                })
        
        return tools_list

    def switch_provider(self, provider_name: str, model_name: str) -> Dict[str, Any]:
        """
        Switch to a new provider and model combination.
        
        Args:
            provider_name: Name of the provider (e.g., "mlx", "anthropic", "openai")
            model_name: Name of the model (e.g., "mlx-community/Qwen3-30B-A3B-4bit")
            
        Returns:
            Dictionary with switch result and details
        """
        try:
            # Store old provider info
            old_provider_name = self._get_provider_name(self.provider) if self.provider else None
            old_model_name = self._get_provider_model(self.provider) if self.provider else None
            
            # Create new provider
            from abstractllm import create_llm
            new_provider = create_llm(provider_name, model=model_name)
            
            # Set it on session (preserves conversation history and tools)
            self.provider = new_provider
            
            # Return success result
            return {
                "success": True,
                "error": None,
                "old_provider": old_provider_name,
                "old_model": old_model_name,
                "new_provider": provider_name,
                "new_model": model_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "old_provider": self._get_provider_name(self.provider) if self.provider else None,
                "old_model": self._get_provider_model(self.provider) if self.provider else None,
                "new_provider": provider_name,
                "new_model": model_name
            }

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get current provider information as structured data.
        
        Returns:
            Dictionary with current provider details
        """
        if not self.provider:
            return {
                "has_provider": False,
                "provider_name": None,
                "model_name": None,
                "capabilities": []
            }
        
        provider_name = self._get_provider_name(self.provider)
        model_name = self._get_provider_model(self.provider)
        capabilities = []
        
        # Get capabilities if available
        if hasattr(self.provider, 'get_capabilities'):
            try:
                caps = self.provider.get_capabilities()
                capabilities = [str(cap) for cap in caps.keys()] if caps else []
            except Exception:
                capabilities = []
        
        return {
            "has_provider": True,
            "provider_name": provider_name,
            "model_name": model_name,
            "capabilities": capabilities
        }

    # SOTA Enhancement Methods
    def _build_enhanced_response(self,
                                base_response: "GenerateResponse",
                                provider_name: str,
                                total_time: float,
                                extracted_facts: List[Any]) -> "GenerateResponse":
        """Build enhanced response with readable telemetry."""
        
        # Create facts list
        facts_strings = []
        for fact in extracted_facts:
            if hasattr(fact, '__str__'):
                facts_strings.append(str(fact))
            else:
                facts_strings.append(f"{fact.subject} --[{fact.predicate}]--> {fact.object}")
        
        # Get COMPLETE scratchpad trace (NO truncation) if available
        complete_scratchpad = []
        scratchpad_file_path = ""
        if self.scratchpad:
            complete_scratchpad = self.scratchpad.get_complete_trace()
            scratchpad_file_path = self.scratchpad.get_scratchpad_file_path()
        
        # Create enhanced response by copying base response
        enhanced_response = base_response
        
        # Add telemetry fields with complete observability
        enhanced_response.react_cycle_id = self.current_cycle.cycle_id if self.current_cycle else None
        enhanced_response.tool_calls = self.current_tool_traces  # Now readable list
        enhanced_response.tools_executed = self.current_tool_traces
        enhanced_response.facts_extracted = facts_strings
        enhanced_response.reasoning_trace = complete_scratchpad  # COMPLETE scratchpad
        enhanced_response.total_reasoning_time = total_time
        
        # Add scratchpad file reference for external access
        enhanced_response.scratchpad_file = str(scratchpad_file_path)
        enhanced_response.scratchpad_manager = self.scratchpad  # For direct access
        
        return enhanced_response
    
    def _get_response_handler(self, provider_name: str) -> StructuredResponseHandler:
        """Get or create structured response handler for provider."""
        if provider_name not in self.response_handlers:
            self.response_handlers[provider_name] = StructuredResponseHandler(provider_name)
        return self.response_handlers[provider_name]
    
    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """Get memory system statistics."""
        if self.memory:
            return self.memory.get_statistics()
        return None
    
    def save_memory(self):
        """Save memory to disk."""
        if self.memory:
            self.memory.save_to_disk()
            logger.info("Memory saved to disk")
    
    def visualize_memory_links(self) -> Optional[str]:
        """Get memory link visualization."""
        if self.memory:
            return self.memory.visualize_links()
        return None
    
    def query_memory(self, query: str) -> Optional[Dict[str, Any]]:
        """Query memory for relevant information."""
        if self.memory:
            return self.memory.query_memory(query)
        return None


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
        Load all sessions from disk. Provider parameters are optional since 
        Session.load() now automatically restores saved provider state.
        
        Args:
            provider: Provider to use for sessions without saved provider state
            provider_config: Configuration for the provider (fallback only)
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


def create_enhanced_session(
    provider: Optional[Union[str, AbstractLLMInterface]] = None,
    enable_memory: bool = True,
    enable_retry: bool = True,
    persist_memory: Optional[str] = None,
    **kwargs
) -> Session:
    """
    Create an enhanced session with SOTA features.
    
    Args:
        provider: Provider name or instance
        enable_memory: Enable hierarchical memory
        enable_retry: Enable retry strategies
        persist_memory: Path to persist memory
        **kwargs: Additional session parameters
        
    Returns:
        Enhanced session instance
    """
    persist_path = Path(persist_memory) if persist_memory else None
    
    return Session(
        provider=provider,
        enable_memory=enable_memory,
        enable_retry=enable_retry,
        persist_memory=persist_path,
        **kwargs
    )