"""
Session management for AbstractLLM.

This module provides utilities for managing stateful conversations with LLMs,
including tracking conversation history and metadata across multiple requests.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
import os
import uuid

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.factory import create_llm
from abstractllm.exceptions import UnsupportedFeatureError


class Message:
    """
    Represents a single message in a conversation.
    """
    
    def __init__(self, 
                 role: str, 
                 content: str, 
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a message.
        
        Args:
            role: The role of the sender (e.g., "user", "assistant", "system")
            content: The message content
            timestamp: When the message was created (defaults to now)
            metadata: Additional message metadata
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary representation.
        
        Returns:
            A dictionary representing the message
        """
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create a message from a dictionary representation.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            A Message instance
        """
        message = cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
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
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a conversation session.
        
        Args:
            system_prompt: The system prompt for the conversation
            provider: Provider name or instance to use for this session
            provider_config: Configuration for the provider
            metadata: Session metadata
        """
        self.messages: List[Message] = []
        self.system_prompt = system_prompt
        self.metadata = metadata or {}
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        
        # Initialize the provider if specified
        self._provider: Optional[AbstractLLMInterface] = None
        if provider is not None:
            if isinstance(provider, str):
                self._provider = create_llm(provider, **(provider_config or {}))
            else:
                self._provider = provider
        
        # Add system message if provided
        if system_prompt:
            self.add_message("system", system_prompt)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a message to the conversation.
        
        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            The created message
        """
        message = Message(role, content, datetime.now(), metadata)
        self.messages.append(message)
        self.last_updated = message.timestamp
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
        return [m for m in self.messages if m.role != "system"]
    
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
        if provider_name == "openai":
            return [{"role": m.role, "content": m.content} for m in self.messages]
        elif provider_name == "anthropic":
            return [{"role": m.role, "content": m.content} for m in self.messages]
        elif provider_name in ["ollama", "huggingface"]:
            # These providers typically don't support chat format directly
            # Return a simple list that can be formatted later
            return [{"role": m.role, "content": m.content} for m in self.messages]
        else:
            # Default format
            return [{"role": m.role, "content": m.content} for m in self.messages]
    
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
        self.add_message("user", message)
        
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
            self.add_message("assistant", response)
            
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
        self.add_message("user", message)
        
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
                self.add_message("assistant", response)
                
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