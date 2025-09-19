"""
Auto-enhancement for AbstractLLM with Cognitive Functions

This module automatically patches AbstractLLM to use cognitive functions by default,
providing seamless integration without requiring code changes.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def auto_enhance_abstractllm(llm_provider: str = "ollama", model: str = "granite3.3:2b"):
    """
    Automatically enhance AbstractLLM with cognitive functions

    This function patches the core AbstractLLM components to use cognitive
    abstractions by default, making them available system-wide.

    Args:
        llm_provider: LLM provider for cognitive functions
        model: Model to use for cognitive processing
    """
    try:
        # Patch memory system to use cognitive fact extraction
        patch_memory_system(llm_provider, model)

        # Patch session creation to include cognitive features
        patch_session_creation(llm_provider, model)

        logger.info("AbstractLLM auto-enhanced with cognitive functions")
        return True

    except Exception as e:
        logger.warning(f"Failed to auto-enhance AbstractLLM: {e}")
        return False


def patch_memory_system(llm_provider: str, model: str):
    """Patch the memory system to use cognitive fact extraction"""
    import abstractllm.memory as memory_module
    from .integrations.memory_integration import CognitiveMemoryAdapter

    try:

        # Store original HierarchicalMemory class
        original_hierarchical_memory = memory_module.HierarchicalMemory

        class EnhancedHierarchicalMemory(original_hierarchical_memory):
            """HierarchicalMemory with automatic cognitive enhancement"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # Try to enhance with cognitive functions
                try:
                    self._cognitive_adapter = CognitiveMemoryAdapter(llm_provider, model)
                    self._original_extract_facts = self.extract_facts
                    self.extract_facts = self._enhanced_extract_facts
                    logger.debug("Memory enhanced with cognitive fact extraction")
                except Exception as e:
                    logger.debug(f"Cognitive enhancement failed, using original: {e}")

            def _enhanced_extract_facts(self, content: str, source_type, source_id: str):
                """Enhanced fact extraction with cognitive fallback"""
                try:
                    # Try cognitive extraction first
                    return self._cognitive_adapter.extract_facts_enhanced(
                        content, str(source_type), source_id
                    )
                except Exception as e:
                    logger.debug(f"Cognitive fact extraction failed, using fallback: {e}")
                    # Fallback to original method
                    return self._original_extract_facts(content, source_type, source_id)

        # Replace the class in the module
        memory_module.HierarchicalMemory = EnhancedHierarchicalMemory
        logger.debug("Memory system patched with cognitive enhancement")

    except Exception as e:
        logger.warning(f"Failed to patch memory system: {e}")


def patch_session_creation(llm_provider: str, model: str):
    """Patch session creation to include cognitive features"""
    import abstractllm.factory as factory_module
    from .integrations.session_integration import enhance_existing_session

    try:

        # Store original create_session function
        original_create_session = factory_module.create_session

        def enhanced_create_session(*args, **kwargs):
            """Create session with automatic cognitive enhancement"""
            # Create standard session
            session = original_create_session(*args, **kwargs)

            # Try to enhance with cognitive features
            try:
                enhance_existing_session(
                    session,
                    cognitive_features=['summarizer', 'facts', 'values'],
                    cognitive_model=model
                )
                logger.debug("Session enhanced with cognitive features")
            except Exception as e:
                logger.debug(f"Session cognitive enhancement failed: {e}")

            return session

        # Replace the function in the module
        factory_module.create_session = enhanced_create_session
        logger.debug("Session creation patched with cognitive enhancement")

    except Exception as e:
        logger.warning(f"Failed to patch session creation: {e}")


# Auto-initialize when module is imported
try:
    auto_enhance_abstractllm()
except Exception:
    pass  # Silent failure for import-time enhancement