"""
AbstractLLM: A unified interface for interacting with various LLM providers.
"""

__version__ = "0.1.0"

from abstractllm.interface import (
    AbstractLLMInterface,
    ModelParameter,
    ModelCapability
)
from abstractllm.factory import create_llm
from abstractllm.chain import (
    ProviderChain,
    create_fallback_chain,
    create_capability_chain,
    create_load_balanced_chain
)
from abstractllm.session import (
    Session,
    SessionManager
)

__all__ = [
    "create_llm",
    "AbstractLLMInterface",
    "ModelParameter",
    "ModelCapability",
    "ProviderChain",
    "create_fallback_chain",
    "create_capability_chain",
    "create_load_balanced_chain",
    "Session",
    "SessionManager",
] 