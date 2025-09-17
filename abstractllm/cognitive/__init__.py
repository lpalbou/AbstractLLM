"""
Cognitive Functions Module for AbstractLLM

This module provides higher-level cognitive abstractions that enhance AbstractLLM
with advanced reasoning capabilities using small, fast models like granite3.3:2b.

Main Components:
- Summarizer: Clear, concise, efficient summarization
- FactsExtractor: Semantic triplet extraction with ontological grounding
- ValueResonance: Value alignment evaluation and ethical reasoning

The cognitive functions are designed to:
1. Use optimized system prompts for specific tasks
2. Leverage fast, lightweight models for efficiency
3. Integrate seamlessly with AbstractLLM's memory and session systems
4. Provide structured, interpretable outputs
"""

from .summarizer import Summarizer, SummaryStyle, InteractionSummary
from .facts_extractor import (
    FactsExtractor,
    SemanticFact,
    FactCategory,
    OntologyType,
    CategorizedFacts
)
from .value_resonance import (
    ValueResonance,
    ValueAssessment,
    ValueScore,
    CoreValue
)

# Integration helpers
from .integrations.memory_integration import enhance_memory_with_cognitive
from .integrations.session_integration import create_cognitive_session

# Auto-enhance AbstractLLM when cognitive module is imported
try:
    from .patch_facts import auto_patch_on_import
    # This automatically patches the memory system with cognitive fact extraction
except Exception:
    pass  # Silent failure for auto-enhancement

__all__ = [
    # Core abstractions
    'Summarizer',
    'FactsExtractor',
    'ValueResonance',

    # Data classes
    'SummaryStyle',
    'InteractionSummary',
    'SemanticFact',
    'FactCategory',
    'OntologyType',
    'CategorizedFacts',
    'ValueAssessment',
    'ValueScore',
    'CoreValue',

    # Integration helpers
    'enhance_memory_with_cognitive',
    'create_cognitive_session'
]

# Version info
__version__ = "1.0.0"
__author__ = "AbstractLLM Cognitive Team"