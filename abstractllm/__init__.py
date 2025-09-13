"""
AbstractLLM: A unified interface for interacting with various LLM providers.
"""

__version__ = "0.5.3"

from abstractllm.interface import (
    AbstractLLMInterface,
    ModelParameter,
    ModelCapability
)
from abstractllm.factory import create_llm, create_session
from abstractllm.session import (
    Session,
    SessionManager
)
from abstractllm.utils.logging import configure_logging

# Enhanced features
from abstractllm.factory_enhanced import create_enhanced_session
from abstractllm.session_enhanced import EnhancedSession
from abstractllm.memory_v2 import HierarchicalMemory
from abstractllm.retry_strategies import RetryManager, RetryConfig, with_retry
from abstractllm.structured_response import (
    StructuredResponseHandler,
    StructuredResponseConfig,
    ResponseFormat
)

__all__ = [
    "create_llm",
    "create_session",
    "create_enhanced_session",  # New enhanced factory
    "AbstractLLMInterface",
    "ModelParameter",
    "ModelCapability",
    "create_fallback_chain",
    "create_capability_chain",
    "create_load_balanced_chain",
    "Session",
    "SessionManager",
    "EnhancedSession",  # New enhanced session
    "HierarchicalMemory",  # New memory system
    "RetryManager",  # New retry strategies
    "RetryConfig",
    "with_retry",
    "StructuredResponseHandler",  # New structured response
    "StructuredResponseConfig",
    "ResponseFormat",
    "configure_logging",
] 