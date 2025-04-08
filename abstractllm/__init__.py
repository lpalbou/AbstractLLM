"""
AbstractLLM: A unified interface for large language models.
"""

__version__ = "0.1.0"

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability, create_config
from abstractllm.factory import create_llm

__all__ = [
    "AbstractLLMInterface", 
    "create_llm", 
    "ModelParameter", 
    "ModelCapability",
    "create_config"
] 