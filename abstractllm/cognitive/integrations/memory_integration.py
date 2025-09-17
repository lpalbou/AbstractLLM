"""
Memory Integration for Cognitive Functions

This module provides integration between cognitive functions and AbstractLLM's
memory system, particularly replacing the basic NLP fact extraction with
sophisticated semantic triplet extraction.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from ..facts_extractor import FactsExtractor, CategorizedFacts
from ..summarizer import Summarizer, SummaryStyle
from ..value_resonance import ValueResonance, ValueAssessment

logger = logging.getLogger(__name__)


class CognitiveMemoryAdapter:
    """Adapter to integrate cognitive functions with AbstractLLM memory"""

    def __init__(self, llm_provider: str = "ollama", model: str = "granite3.3:2b"):
        """
        Initialize cognitive memory adapter

        Args:
            llm_provider: LLM provider for cognitive functions
            model: Model to use for cognitive processing
        """
        self.provider = llm_provider
        self.model = model

        # Initialize cognitive functions lazily
        self._facts_extractor = None
        self._summarizer = None
        self._value_evaluator = None

    @property
    def facts_extractor(self) -> FactsExtractor:
        """Lazy initialization of facts extractor"""
        if self._facts_extractor is None:
            self._facts_extractor = FactsExtractor(
                llm_provider=self.provider,
                model=self.model
            )
        return self._facts_extractor

    @property
    def summarizer(self) -> Summarizer:
        """Lazy initialization of summarizer"""
        if self._summarizer is None:
            self._summarizer = Summarizer(
                llm_provider=self.provider,
                model=self.model
            )
        return self._summarizer

    @property
    def value_evaluator(self) -> ValueResonance:
        """Lazy initialization of value evaluator"""
        if self._value_evaluator is None:
            self._value_evaluator = ValueResonance(
                llm_provider=self.provider,
                model=self.model
            )
        return self._value_evaluator

    def extract_facts_enhanced(self, content: str, source_type: str,
                              source_id: str) -> List[Dict[str, Any]]:
        """
        Enhanced fact extraction to replace basic NLP in memory.py

        This method provides a drop-in replacement for the extract_facts method
        in AbstractLLM's memory system, using semantic triplet extraction.

        Args:
            content: Content to extract facts from
            source_type: Type of source (for memory system compatibility)
            source_id: Source identifier (for memory system compatibility)

        Returns:
            List of fact dictionaries compatible with AbstractLLM memory format
        """
        try:
            # Use semantic facts extractor
            categorized_facts = self.facts_extractor.extract_facts(
                content,
                context_type="interaction"
            )

            # Convert to AbstractLLM memory format
            enhanced_facts = []
            for i, fact in enumerate(categorized_facts.all_facts()):
                enhanced_fact = {
                    'fact_id': f"cognitive_{source_id}_{i}",
                    'subject': fact.subject,
                    'predicate': f"{fact.ontology.value}:{fact.predicate}",
                    'object': fact.object,
                    'confidence': fact.confidence,
                    'source_type': source_type,
                    'source_id': source_id,
                    'category': fact.category.value,
                    'ontology': fact.ontology.value,
                    'extraction_method': 'cognitive_semantic',
                    'timestamp': fact.timestamp,
                    'metadata': {
                        'rdf_triple': fact.to_rdf_triple(),
                        'context_snippet': fact.context,
                        'extraction_stats': {
                            'total_facts': categorized_facts.total_extracted,
                            'extraction_time': categorized_facts.extraction_time
                        }
                    }
                }
                enhanced_facts.append(enhanced_fact)

            logger.info(f"Extracted {len(enhanced_facts)} semantic facts from {source_type}:{source_id}")
            return enhanced_facts

        except Exception as e:
            logger.error(f"Cognitive fact extraction failed: {e}")
            # Fallback to empty list rather than crashing
            return []

    def enhance_interaction_context(self, interaction_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance interaction context with cognitive analysis

        Args:
            interaction_context: Original interaction context

        Returns:
            Enhanced context with cognitive insights
        """
        enhanced_context = interaction_context.copy()

        try:
            # Add semantic facts extraction
            if self.facts_extractor.is_available():
                facts_analysis = self.facts_extractor.extract_interaction_facts(interaction_context)
                enhanced_context['cognitive_facts'] = facts_analysis.to_dict()

            # Add summarization
            if self.summarizer.is_available():
                summary = self.summarizer.summarize_interaction(interaction_context)
                enhanced_context['cognitive_summary'] = summary.to_dict()

            # Add value assessment
            if self.value_evaluator.is_available():
                value_assessment = self.value_evaluator.evaluate_abstractllm_interaction(interaction_context)
                enhanced_context['value_assessment'] = value_assessment.to_dict()

            logger.debug("Enhanced interaction context with cognitive analysis")

        except Exception as e:
            logger.error(f"Failed to enhance interaction context: {e}")
            # Don't fail the original interaction, just log the error

        return enhanced_context

    def is_available(self) -> bool:
        """Check if cognitive functions are available"""
        try:
            # Test initialization of core function
            return self.facts_extractor.is_available()
        except Exception:
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all cognitive functions"""
        stats = {
            "adapter_available": self.is_available(),
            "provider": self.provider,
            "model": self.model
        }

        if self._facts_extractor:
            stats["facts_extractor"] = self._facts_extractor.get_performance_stats()

        if self._summarizer:
            stats["summarizer"] = self._summarizer.get_performance_stats()

        if self._value_evaluator:
            stats["value_evaluator"] = self._value_evaluator.get_performance_stats()

        return stats


def enhance_memory_with_cognitive(memory_instance, llm_provider: str = "ollama",
                                 model: str = "granite3.3:2b") -> None:
    """
    Enhance an existing AbstractLLM memory instance with cognitive functions

    This function modifies an existing HierarchicalMemory instance to use
    cognitive fact extraction instead of the basic regex patterns.

    Args:
        memory_instance: Instance of HierarchicalMemory to enhance
        llm_provider: LLM provider for cognitive functions
        model: Model to use for cognitive processing
    """
    # Create cognitive adapter
    adapter = CognitiveMemoryAdapter(llm_provider, model)

    # Store original extract_facts method as fallback
    original_extract_facts = memory_instance.extract_facts

    def enhanced_extract_facts(content: str, source_type, source_id: str) -> List:
        """Enhanced fact extraction with cognitive fallback"""
        try:
            if adapter.is_available():
                # Use cognitive extraction
                facts = adapter.extract_facts_enhanced(content, source_type.value, source_id)
                if facts:  # If we got results, use them
                    # Convert to AbstractLLM Fact objects
                    from abstractllm.memory import Fact, MemoryComponent

                    enhanced_facts = []
                    for fact_dict in facts:
                        fact = Fact(
                            subject=fact_dict['subject'],
                            predicate=fact_dict['predicate'],
                            object=fact_dict['object'],
                            confidence=fact_dict['confidence'],
                            source_type=source_type,
                            source_id=source_id,
                            metadata=fact_dict['metadata']
                        )
                        enhanced_facts.append(fact)

                    logger.info(f"Enhanced memory with {len(enhanced_facts)} cognitive facts")
                    return enhanced_facts
        except Exception as e:
            logger.warning(f"Cognitive fact extraction failed, using fallback: {e}")

        # Fallback to original method
        return original_extract_facts(content, source_type, source_id)

    # Replace the extract_facts method
    memory_instance.extract_facts = enhanced_extract_facts
    memory_instance._cognitive_adapter = adapter

    logger.info("Enhanced memory instance with cognitive fact extraction")


def create_cognitive_memory(*args, **kwargs):
    """
    Create a new HierarchicalMemory instance with cognitive enhancements

    This function creates a new memory instance that automatically uses
    cognitive functions for fact extraction.

    Returns:
        Enhanced HierarchicalMemory instance
    """
    from abstractllm.memory import HierarchicalMemory

    # Create standard memory instance
    memory = HierarchicalMemory(*args, **kwargs)

    # Enhance with cognitive functions
    enhance_memory_with_cognitive(memory)

    return memory


def patch_memory_system(llm_provider: str = "ollama", model: str = "granite3.3:2b"):
    """
    Globally patch the AbstractLLM memory system to use cognitive functions

    This function modifies the memory module to use cognitive fact extraction
    by default for all new HierarchicalMemory instances.

    Args:
        llm_provider: LLM provider for cognitive functions
        model: Model to use for cognitive processing
    """
    import abstractllm.memory as memory_module

    # Store original HierarchicalMemory class
    original_hierarchical_memory = memory_module.HierarchicalMemory

    class CognitiveHierarchicalMemory(original_hierarchical_memory):
        """Enhanced HierarchicalMemory with cognitive functions"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Enhance with cognitive functions
            enhance_memory_with_cognitive(self, llm_provider, model)

    # Replace the class in the module
    memory_module.HierarchicalMemory = CognitiveHierarchicalMemory

    logger.info("Patched AbstractLLM memory system with cognitive enhancements")