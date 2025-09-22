"""
AbstractLLM: A unified interface for interacting with various LLM providers.
"""

__version__ = "1.1.0"

# Force offline-first mode by default to prevent unauthorized network access
import os
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

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

# Enhanced features (now integrated into main Session class)
from abstractllm.session import create_enhanced_session
from abstractllm.memory import HierarchicalMemory
from abstractllm.retry_strategies import RetryManager, RetryConfig, with_retry
from abstractllm.structured_response import (
    StructuredResponseHandler,
    StructuredResponseConfig,
    ResponseFormat
)

__all__ = [
    "create_llm",
    "create_session",
    "create_enhanced_session",  # Enhanced factory (now uses unified Session class)
    "AbstractLLMInterface",
    "ModelParameter",
    "ModelCapability",
    "create_fallback_chain",
    "create_capability_chain",
    "create_load_balanced_chain",
    "Session",  # Now includes both core and enhanced features
    "SessionManager",
    "HierarchicalMemory",  # SOTA memory system
    "RetryManager",  # SOTA retry strategies
    "RetryConfig",
    "with_retry",
    "StructuredResponseHandler",  # SOTA structured response
    "StructuredResponseConfig",
    "ResponseFormat",
    "configure_logging",
] 