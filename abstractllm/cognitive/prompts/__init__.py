"""
Optimized Prompts for Cognitive Functions

This module contains carefully crafted system prompts for each cognitive function,
optimized for granite3.3:2b and focused on specific tasks.
"""

from .summarizer_prompts import (
    build_summarizer_prompt,
    ABSTRACTLLM_INTERACTION_PROMPT
)
from .facts_prompts import (
    build_extraction_prompt,
    ABSTRACTLLM_FACTS_PROMPT,
    SEMANTIC_ANALYSIS_PROMPT
)
from .values_prompts import (
    build_value_evaluation_prompt,
    ABSTRACTLLM_VALUE_PROMPT,
    ETHICAL_ANALYSIS_PROMPT
)

__all__ = [
    # Summarizer prompts
    'build_summarizer_prompt',
    'ABSTRACTLLM_INTERACTION_PROMPT',

    # Facts extraction prompts
    'build_extraction_prompt',
    'ABSTRACTLLM_FACTS_PROMPT',
    'SEMANTIC_ANALYSIS_PROMPT',

    # Value evaluation prompts
    'build_value_evaluation_prompt',
    'ABSTRACTLLM_VALUE_PROMPT',
    'ETHICAL_ANALYSIS_PROMPT'
]