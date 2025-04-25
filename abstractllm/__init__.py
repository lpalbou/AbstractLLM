"""
AbstractLLM: A unified interface for interacting with various LLM providers.
"""

__version__ = "0.4.7"

from abstractllm.interface import (
    AbstractLLMInterface,
    ModelParameter,
    ModelCapability
)
from abstractllm.factory import create_llm
from abstractllm.session import (
    Session,
    SessionManager
)
from abstractllm.utils.logging import configure_logging

__all__ = [
    "create_llm",
    "AbstractLLMInterface",
    "ModelParameter",
    "ModelCapability",
    "create_fallback_chain",
    "create_capability_chain",
    "create_load_balanced_chain",
    "Session",
    "SessionManager",
    "configure_logging",
] 