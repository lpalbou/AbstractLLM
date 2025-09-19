#!/usr/bin/env python3
"""
Streaming Context Management Fix for AbstractLLM

This fix addresses the rapid context accumulation during streaming by implementing
streaming-aware memory management and intelligent context reduction strategies.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class StreamingContextManager:
    """
    Manages context generation specifically for streaming scenarios.

    Key Features:
    - Prevents excessive context regeneration during streaming
    - Implements progressive context reduction as conversation grows
    - Maintains conversation coherence while respecting token limits
    - Provides streaming-specific optimizations
    """

    def __init__(self):
        self.streaming_active = False
        self.last_context_size = 0
        self.context_growth_threshold = 1.5  # Alert when context grows 50%
        self.streaming_context_cache = {}
        self.cache_timeout = timedelta(seconds=30)

    def optimize_for_streaming(self, session, provider_name: str):
        """
        Apply streaming optimizations to a session.

        Args:
            session: AbstractLLM session instance
            provider_name: Name of the provider (e.g., 'lmstudio')
        """
        self.streaming_active = True

        # Patch the session's memory context generation
        if hasattr(session, 'memory') and session.memory:
            original_get_context = session.memory.get_context_for_query
            session.memory.get_context_for_query = lambda *args, **kwargs: self._streaming_aware_context(
                original_get_context, *args, **kwargs
            )

        # Adjust memory settings for streaming
        self._adjust_memory_settings_for_streaming(session)

        logger.info(f"Applied streaming optimizations for {provider_name}")

    def _streaming_aware_context(self, original_method, query: str, max_tokens: int = 2000, **kwargs) -> str:
        """
        Streaming-aware wrapper for context generation.

        This method implements several strategies to prevent context explosion:
        1. Caching: Avoid regenerating context too frequently
        2. Progressive reduction: Reduce context detail as conversation grows
        3. Relevancy filtering: Prioritize most relevant information
        """
        # Check cache first
        cache_key = f"{query}_{max_tokens}"
        now = datetime.now()

        if (cache_key in self.streaming_context_cache and
            now - self.streaming_context_cache[cache_key]['timestamp'] < self.cache_timeout):
            logger.debug("Using cached context for streaming")
            return self.streaming_context_cache[cache_key]['context']

        # Calculate reduced max_tokens for streaming
        streaming_max_tokens = self._calculate_streaming_token_limit(max_tokens)

        # Apply streaming-specific parameters
        streaming_kwargs = kwargs.copy()
        streaming_kwargs.update({
            'max_facts': min(kwargs.get('max_facts', 5), 3),  # Reduce facts in streaming
            'include_reasoning': kwargs.get('include_reasoning', True),
            'min_confidence': max(kwargs.get('min_confidence', 0.3), 0.5),  # Higher confidence threshold
        })

        # Generate context with streaming optimizations
        context = original_method(query, streaming_max_tokens, **streaming_kwargs)

        # Monitor context growth
        context_size = len(context.split()) * 1.3  # Rough token estimate
        if self.last_context_size > 0 and context_size > self.last_context_size * self.context_growth_threshold:
            logger.warning(f"Context size grew significantly: {self.last_context_size:.0f} -> {context_size:.0f} tokens")

        self.last_context_size = context_size

        # Cache the result
        self.streaming_context_cache[cache_key] = {
            'context': context,
            'timestamp': now
        }

        # Clean old cache entries
        self._clean_context_cache()

        return context

    def _calculate_streaming_token_limit(self, base_limit: int) -> int:
        """
        Calculate appropriate token limit for streaming scenarios.

        In streaming, we need to be more conservative with context to leave
        room for the accumulating conversation.
        """
        if self.last_context_size > base_limit * 0.8:
            # If context is getting large, reduce significantly
            return max(int(base_limit * 0.4), 500)
        elif self.last_context_size > base_limit * 0.5:
            # Moderate reduction
            return max(int(base_limit * 0.6), 800)
        else:
            # Small reduction for streaming overhead
            return max(int(base_limit * 0.8), 1000)

    def _adjust_memory_settings_for_streaming(self, session):
        """Adjust memory settings to be more streaming-friendly."""
        if hasattr(session, 'memory') and session.memory:
            # Reduce working memory size to prevent accumulation
            if hasattr(session.memory, 'working_memory_size'):
                original_size = getattr(session.memory, 'working_memory_size', 10)
                session.memory.working_memory_size = min(original_size, 5)
                logger.debug(f"Reduced working memory size to {session.memory.working_memory_size} for streaming")

            # More aggressive consolidation
            if hasattr(session.memory, 'consolidation_threshold'):
                session.memory.consolidation_threshold = 3  # Consolidate more frequently

    def _clean_context_cache(self):
        """Remove expired entries from context cache."""
        now = datetime.now()
        expired_keys = [
            key for key, value in self.streaming_context_cache.items()
            if now - value['timestamp'] > self.cache_timeout
        ]
        for key in expired_keys:
            del self.streaming_context_cache[key]


class StreamingMemoryPatch:
    """
    Patch for the session's memory handling to improve streaming performance.
    """

    @staticmethod
    def patch_session_for_streaming(session):
        """
        Apply streaming-specific patches to a session.

        This method modifies the session's behavior to be more streaming-friendly:
        1. Reduces memory context injection frequency
        2. Implements smarter consolidation
        3. Prevents context explosion
        """
        if not hasattr(session, 'memory') or not session.memory:
            return

        # Store original methods
        session._original_generate = session.generate
        session._streaming_context_manager = StreamingContextManager()

        # Replace generate method with streaming-aware version
        def streaming_aware_generate(*args, **kwargs):
            """Streaming-aware wrapper for session.generate"""
            # Enable streaming optimizations
            session._streaming_context_manager.optimize_for_streaming(session, "lmstudio")

            # Check if this is a streaming request
            is_streaming = kwargs.get('stream', False)
            if is_streaming:
                # Reduce memory context injection for streaming
                kwargs['use_memory_context'] = kwargs.get('use_memory_context', False)
                logger.debug("Disabled memory context for streaming request")

            return session._original_generate(*args, **kwargs)

        session.generate = streaming_aware_generate
        logger.info("Applied streaming memory patch to session")


def apply_streaming_context_fix():
    """
    Apply streaming context fixes to the AbstractLLM system.

    This function patches the relevant components to handle streaming
    more efficiently and prevent context overflow.
    """
    try:
        # Patch the memory system
        from abstractllm.memory import HierarchicalMemory

        # Store original method
        if not hasattr(HierarchicalMemory, '_original_get_context_for_query'):
            HierarchicalMemory._original_get_context_for_query = HierarchicalMemory.get_context_for_query

        def patched_get_context_for_query(self, query: str, max_tokens: int = 2000, **kwargs):
            """Patched version with streaming awareness"""
            # Detect if we're in a streaming scenario by checking call frequency
            now = datetime.now()
            if not hasattr(self, '_last_context_call'):
                self._last_context_call = now
            elif now - self._last_context_call < timedelta(seconds=5):
                # Frequent calls indicate streaming - reduce context
                max_tokens = min(max_tokens, 1000)
                kwargs['max_facts'] = min(kwargs.get('max_facts', 5), 2)
                logger.debug(f"Detected frequent context calls - reduced max_tokens to {max_tokens}")

            self._last_context_call = now
            return self._original_get_context_for_query(query, max_tokens, **kwargs)

        HierarchicalMemory.get_context_for_query = patched_get_context_for_query
        logger.info("Applied memory context streaming patch")

    except ImportError as e:
        logger.warning(f"Could not apply memory patch: {e}")

    try:
        # Patch the session system
        from abstractllm.session import Session

        # Add streaming optimization method
        Session.optimize_for_streaming = StreamingMemoryPatch.patch_session_for_streaming
        logger.info("Added streaming optimization to Session class")

    except ImportError as e:
        logger.warning(f"Could not patch Session class: {e}")


class ContextOverflowProtection:
    """
    Protection mechanism against context overflow during streaming.

    This class monitors context usage and provides warnings/mitigations
    when context is approaching dangerous levels.
    """

    def __init__(self, max_context_tokens: int = 32768):
        self.max_context_tokens = max_context_tokens
        self.warning_threshold = int(max_context_tokens * 0.8)  # 80% warning
        self.critical_threshold = int(max_context_tokens * 0.95)  # 95% critical

    def check_context_usage(self, context: str, prompt: str, max_output: int = 8192) -> Dict[str, Any]:
        """
        Check context usage and provide recommendations.

        Args:
            context: Current memory context
            prompt: User prompt
            max_output: Maximum output tokens

        Returns:
            Dictionary with usage info and recommendations
        """
        def estimate_tokens(text: str) -> int:
            return len(str(text).split()) * 1.3

        context_tokens = estimate_tokens(context)
        prompt_tokens = estimate_tokens(prompt)
        total_input_tokens = context_tokens + prompt_tokens
        remaining_tokens = self.max_context_tokens - total_input_tokens - max_output

        status = "ok"
        recommendations = []

        if total_input_tokens > self.critical_threshold:
            status = "critical"
            recommendations.extend([
                "Context usage is critical - immediate reduction needed",
                "Consider clearing working memory",
                "Reduce memory context max_tokens parameter",
                "Enable more aggressive memory consolidation"
            ])
        elif total_input_tokens > self.warning_threshold:
            status = "warning"
            recommendations.extend([
                "Context usage is high - consider optimization",
                "Monitor memory growth",
                "Consider reducing context detail for streaming"
            ])

        return {
            "status": status,
            "context_tokens": context_tokens,
            "prompt_tokens": prompt_tokens,
            "total_input_tokens": total_input_tokens,
            "remaining_tokens": remaining_tokens,
            "max_context_tokens": self.max_context_tokens,
            "usage_percentage": (total_input_tokens / self.max_context_tokens) * 100,
            "recommendations": recommendations
        }


# Integration example for LM Studio provider
def integrate_with_lmstudio_provider():
    """
    Example of how to integrate streaming fixes with the LM Studio provider.
    """
    try:
        from abstractllm.providers.lmstudio_provider import LMStudioProvider

        # Store original generate method
        if not hasattr(LMStudioProvider, '_original_generate_impl'):
            LMStudioProvider._original_generate_impl = LMStudioProvider._generate_impl

        def streaming_optimized_generate_impl(self, *args, **kwargs):
            """Streaming-optimized version of _generate_impl"""
            # Add context overflow protection
            protection = ContextOverflowProtection(
                max_context_tokens=self.config_manager.get_param('max_input_tokens', 32768)
            )

            # If this is a streaming request, apply optimizations
            if kwargs.get('stream', False):
                logger.debug("Applying streaming optimizations for LM Studio")

                # Monitor context if messages are provided
                if kwargs.get('messages'):
                    messages_str = json.dumps(kwargs['messages'])
                    usage_info = protection.check_context_usage(
                        context=messages_str,
                        prompt=args[0] if args else "",
                        max_output=self.config_manager.get_param('max_tokens', 8192)
                    )

                    if usage_info['status'] != 'ok':
                        logger.warning(f"Context usage {usage_info['status']}: "
                                     f"{usage_info['usage_percentage']:.1f}% "
                                     f"({usage_info['total_input_tokens']}/{usage_info['max_context_tokens']} tokens)")

                        if usage_info['recommendations']:
                            logger.info("Recommendations: " + "; ".join(usage_info['recommendations']))

            return self._original_generate_impl(*args, **kwargs)

        LMStudioProvider._generate_impl = streaming_optimized_generate_impl
        logger.info("Applied streaming optimizations to LM Studio provider")

    except ImportError as e:
        logger.warning(f"Could not patch LM Studio provider: {e}")


if __name__ == "__main__":
    # Apply all streaming fixes
    apply_streaming_context_fix()
    integrate_with_lmstudio_provider()
    print("Streaming context fixes applied")