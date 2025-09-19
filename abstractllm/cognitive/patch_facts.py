"""
Patch AbstractLLM Memory to Use Cognitive Fact Extraction

This module provides simple patching to replace basic NLP fact extraction
with our sophisticated semantic ontological framework.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def patch_memory_facts_extraction():
    """
    Patch AbstractLLM memory system to use cognitive fact extraction

    This function replaces the basic regex-based fact extraction in the
    HierarchicalMemory class with our sophisticated semantic approach.
    """
    import abstractllm.memory as memory_module
    from .facts_extractor import FactsExtractor

    try:
        # Store the original extract_facts method
        original_memory_class = memory_module.HierarchicalMemory
        original_extract_facts = original_memory_class.extract_facts

        # Initialize cognitive fact extractor
        try:
            cognitive_extractor = FactsExtractor(
                llm_provider="ollama",
                model="granite3.3:2b"
            )
        except Exception as e:
            logger.warning(f"Could not initialize cognitive fact extractor: {e}")
            return False

        def enhanced_extract_facts(self, content: str, source_type, source_id: str) -> List:
            """Enhanced fact extraction using cognitive abstractions"""
            try:
                # Use cognitive fact extraction
                categorized_facts = cognitive_extractor.extract_facts(
                    content,
                    context_type="interaction"
                )

                # Convert to AbstractLLM Fact objects
                memory_facts = []
                for i, semantic_fact in enumerate(categorized_facts.all_facts()):
                    # Create Fact object compatible with existing memory system
                    fact = memory_module.Fact(
                        fact_id=f"cognitive_{source_id}_{i}",
                        subject=semantic_fact.subject,
                        predicate=f"{semantic_fact.ontology.value if hasattr(semantic_fact.ontology, 'value') else semantic_fact.ontology}:{semantic_fact.predicate}",
                        object=semantic_fact.object,
                        confidence=semantic_fact.confidence,
                        source_type=str(source_type),
                        source_id=str(source_id),
                        extraction_method="cognitive_ontological"
                    )
                    memory_facts.append(fact)

                logger.debug(f"Cognitive extraction: {len(memory_facts)} facts from {len(content)} chars")
                return memory_facts

            except Exception as e:
                logger.debug(f"Cognitive fact extraction failed: {e}, using fallback")
                # Fallback to original method
                return original_extract_facts(self, content, source_type, source_id)

        # Patch the method in the class
        original_memory_class.extract_facts = enhanced_extract_facts

        # Also patch any existing instances (if possible)
        for obj in memory_module.__dict__.values():
            if isinstance(obj, original_memory_class):
                obj.extract_facts = enhanced_extract_facts.__get__(obj, original_memory_class)

        logger.info("✅ AbstractLLM memory patched with cognitive fact extraction")
        return True

    except Exception as e:
        logger.error(f"Failed to patch memory fact extraction: {e}")
        return False


def auto_patch_on_import():
    """Automatically patch when this module is imported"""
    try:
        success = patch_memory_facts_extraction()
        if success:
            print("🧠 Enhanced AbstractLLM with cognitive fact extraction")
        return success
    except Exception:
        return False


# Auto-patch when module is imported
if __name__ != "__main__":
    auto_patch_on_import()