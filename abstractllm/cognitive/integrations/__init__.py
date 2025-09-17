"""
Integration helpers for cognitive functions with AbstractLLM

This module provides utilities to integrate cognitive functions seamlessly
with the existing AbstractLLM architecture, including memory systems and sessions.
"""

from .memory_integration import enhance_memory_with_cognitive, CognitiveMemoryAdapter
from .session_integration import create_cognitive_session, CognitiveSessionEnhancer

__all__ = [
    'enhance_memory_with_cognitive',
    'CognitiveMemoryAdapter',
    'create_cognitive_session',
    'CognitiveSessionEnhancer'
]