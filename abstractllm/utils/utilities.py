"""
Utility functions and classes for AbstractLLM.
"""

from transformers import AutoTokenizer
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import platform

logger = logging.getLogger(__name__)

class TokenCounter:
    """Simple token counter using transformers AutoTokenizer."""
    
    _tokenizers = {}  # Cache tokenizers by model name
    
    @classmethod
    def count_tokens(cls, text: str, model_name: Optional[str] = None) -> int:
        """
        Count tokens in text using the appropriate tokenizer.
        
        Args:
            text: Text to count tokens for
            model_name: Model name to use for tokenizer. If None, uses a default.
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
            
        # Use default tokenizer if no model specified
        if model_name is None:
            model_name = "Qwen/Qwen1.5-7B"
        
        # Get or create tokenizer
        if model_name not in cls._tokenizers:
            cls._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            logger.debug(f"Loaded tokenizer for {model_name}")
        
        tokenizer = cls._tokenizers[model_name]
        return len(tokenizer.encode(text))


def get_session_stats(session) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a session.
    
    Args:
        session: Session object to analyze
        
    Returns:
        Dictionary containing session statistics including:
        - Session info (id, created_at, last_updated, duration)
        - Message statistics (total, by role, average length)
        - Tool usage statistics (if applicable)
        - Provider information
        - Token statistics (automatically computed for missing data)
    """
    # Import TokenCounter for automatic token computation
    can_compute_tokens = True  # We have it locally
    
    stats = {
        "session_info": {
            "id": session.id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "duration_hours": (session.last_updated - session.created_at).total_seconds() / 3600,
            "has_system_prompt": bool(session.system_prompt),
            "metadata": session.metadata
        },
        "message_stats": {
            "total_messages": len(session.messages),
            "by_role": {},
            "total_characters": 0,
            "average_message_length": 0,
            "first_message_time": None,
            "last_message_time": None
        },
        "tool_stats": {
            "total_tool_calls": 0,
            "unique_tools_used": set(),
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "tool_success_rate": 0.0,
            "tools_available": len(session.tools),
            "tool_names": [tool.name for tool in session.tools] if hasattr(session, 'tools') and session.tools else []
        },
        "provider_info": {
            "current_provider": session._get_provider_name(session._provider) if session._provider else "None",
            "provider_capabilities": list(session._provider.get_capabilities().keys()) if session._provider else []
        }
    }
    
    # Calculate message statistics
    role_counts = {}
    total_chars = 0
    
    # Token statistics
    token_stats = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "messages_with_usage": 0,
        "average_prompt_tokens": 0,
        "average_completion_tokens": 0,
        "total_time": 0.0,
        "average_prompt_tps": 0.0,
        "average_completion_tps": 0.0,
        "average_total_tps": 0.0,
        "by_provider": {}
    }
    
    # Get current model name for token counting
    current_model = None
    if session._provider and hasattr(session._provider, 'config_manager'):
        try:
            from abstractllm.interface import ModelParameter
            current_model = session._provider.config_manager.get_param(ModelParameter.MODEL)
        except:
            current_model = None
    
    for message in session.messages:
        # Count by role
        role = message.role
        role_counts[role] = role_counts.get(role, 0) + 1
        
        # Character count
        total_chars += len(message.content)
        
        # Token usage tracking from message metadata
        if message.metadata and "usage" in message.metadata:
            usage = message.metadata["usage"]
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_message_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                time_taken = usage.get("time", 0.0)
                
                token_stats["total_prompt_tokens"] += prompt_tokens
                token_stats["total_completion_tokens"] += completion_tokens
                token_stats["total_tokens"] += total_message_tokens
                token_stats["total_time"] += time_taken
                token_stats["messages_with_usage"] += 1
                
                # Track by provider if available
                provider_name = message.metadata.get("provider", "unknown")
                if provider_name not in token_stats["by_provider"]:
                    token_stats["by_provider"][provider_name] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "total_time": 0.0,
                        "messages": 0,
                        "average_tps": 0.0
                    }
                
                provider_stats = token_stats["by_provider"][provider_name]
                provider_stats["prompt_tokens"] += prompt_tokens
                provider_stats["completion_tokens"] += completion_tokens
                provider_stats["total_tokens"] += total_message_tokens
                provider_stats["total_time"] += time_taken
                provider_stats["messages"] += 1
                
                # Calculate provider-specific TPS
                if provider_stats["total_time"] > 0:
                    provider_stats["average_tps"] = provider_stats["total_tokens"] / provider_stats["total_time"]
        
        elif can_compute_tokens and message.content:
            # Automatically compute tokens for messages without usage metadata
            try:
                message_tokens = TokenCounter.count_tokens(message.content, current_model)
                
                # For assistant messages, count as completion tokens
                # For user/system messages, count as prompt tokens
                if message.role == "assistant":
                    token_stats["total_completion_tokens"] += message_tokens
                else:
                    token_stats["total_prompt_tokens"] += message_tokens
                
                token_stats["total_tokens"] += message_tokens
                
            except Exception as e:
                logger.warning(f"Failed to compute tokens for message: {e}")
        
        # Tool statistics
        if message.tool_results:
            for tool_result in message.tool_results:
                stats["tool_stats"]["total_tool_calls"] += 1
                tool_name = tool_result.get("name", "unknown")
                stats["tool_stats"]["unique_tools_used"].add(tool_name)
                
                if tool_result.get("error"):
                    stats["tool_stats"]["failed_tool_calls"] += 1
                else:
                    stats["tool_stats"]["successful_tool_calls"] += 1
        
        # Timestamp tracking
        if stats["message_stats"]["first_message_time"] is None:
            stats["message_stats"]["first_message_time"] = message.timestamp.isoformat()
        stats["message_stats"]["last_message_time"] = message.timestamp.isoformat()
    
    # Calculate token averages
    if token_stats["messages_with_usage"] > 0:
        token_stats["average_prompt_tokens"] = token_stats["total_prompt_tokens"] / token_stats["messages_with_usage"]
        token_stats["average_completion_tokens"] = token_stats["total_completion_tokens"] / token_stats["messages_with_usage"]
        
        # Calculate TPS averages (only for messages with timing data)
        if token_stats["total_time"] > 0:
            token_stats["average_total_tps"] = token_stats["total_tokens"] / token_stats["total_time"]
            token_stats["average_prompt_tps"] = token_stats["total_prompt_tokens"] / token_stats["total_time"]
            token_stats["average_completion_tps"] = token_stats["total_completion_tokens"] / token_stats["total_time"]
    
    # Finalize message stats
    stats["message_stats"]["by_role"] = role_counts
    stats["message_stats"]["total_characters"] = total_chars
    if len(session.messages) > 0:
        stats["message_stats"]["average_message_length"] = total_chars / len(session.messages)
    
    # Add token stats to the return dictionary
    stats["token_stats"] = token_stats
    
    # Finalize tool stats
    stats["tool_stats"]["unique_tools_used"] = list(stats["tool_stats"]["unique_tools_used"])
    if stats["tool_stats"]["total_tool_calls"] > 0:
        stats["tool_stats"]["tool_success_rate"] = (
            stats["tool_stats"]["successful_tool_calls"] / stats["tool_stats"]["total_tool_calls"]
        )
    
    return stats 

def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon hardware (M1/M2/M3/M4).
    
    Returns:
        True if running on macOS with Apple Silicon, False otherwise
    """
    try:
        return (
            platform.system().lower() == "darwin" and 
            platform.machine().lower() in ["arm64", "aarch64"]
        )
    except Exception:
        return False 