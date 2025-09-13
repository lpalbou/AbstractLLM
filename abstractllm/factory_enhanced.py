"""
Enhanced factory functions with SOTA features.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
from abstractllm.factory import create_llm
from abstractllm.session_enhanced import EnhancedSession
from abstractllm.interface import AbstractLLMInterface


def create_enhanced_session(
    provider: str,
    enable_memory: bool = True,
    enable_retry: bool = True,
    persist_memory: Optional[str] = None,
    memory_config: Optional[Dict[str, Any]] = None,
    **config
) -> EnhancedSession:
    """
    Create an enhanced session with SOTA features.
    
    This is a drop-in replacement for create_session that adds:
    - Hierarchical memory with ReAct cycles
    - Retry strategies with exponential backoff
    - Structured response support
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'ollama', 'huggingface', 'mlx')
        enable_memory: Enable hierarchical memory system (default: True)
        enable_retry: Enable retry strategies (default: True)
        persist_memory: Optional path to persist memory
        memory_config: Optional memory configuration
        **config: Provider and session configuration
        
    Returns:
        EnhancedSession with SOTA features
        
    Example:
        # Simple usage - same as before but with enhancements
        session = create_enhanced_session("ollama", model="qwen3:4b")
        response = session.generate("Hello!")  # Automatically uses memory & retry
        
        # With persistence
        session = create_enhanced_session(
            "openai",
            model="gpt-4",
            persist_memory="./agent_memory"
        )
        
        # Query memory
        facts = session.query_memory("python")
        stats = session.get_memory_stats()
    """
    # Create provider instance
    provider_instance = create_llm(provider, **config)
    
    # Extract session-specific config
    system_prompt = config.pop("system_prompt", None)
    tools = config.pop("tools", None)
    metadata = config.pop("metadata", None)
    
    # Create enhanced session
    persist_path = Path(persist_memory) if persist_memory else None
    
    session = EnhancedSession(
        provider=provider_instance,
        provider_config=config,
        system_prompt=system_prompt,
        tools=tools,
        metadata=metadata,
        enable_memory=enable_memory,
        enable_retry=enable_retry,
        persist_memory=persist_path,
        memory_config=memory_config
    )
    
    return session