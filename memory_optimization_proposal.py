#!/usr/bin/env python3
"""
Memory Context Optimization for AbstractLLM Streaming

This proposal addresses the context overflow issue during streaming by implementing
smart memory management that adapts to streaming scenarios and prioritizes content
based on relevance and recency.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StreamingAwareMemoryManager:
    """
    Enhanced memory manager that adapts context generation for streaming scenarios.

    Key Features:
    - Streaming-aware context prioritization
    - Dynamic token allocation based on conversation phase
    - Content freshness scoring
    - Progressive context reduction strategies
    """

    def __init__(self, base_memory_system):
        self.memory = base_memory_system
        self.streaming_mode = False
        self.last_context_generation = None
        self.context_generation_frequency = timedelta(seconds=30)  # Don't regenerate too often

    def set_streaming_mode(self, enabled: bool):
        """Enable/disable streaming-aware optimizations."""
        self.streaming_mode = enabled
        if enabled:
            logger.debug("Enabled streaming-aware memory optimizations")

    def get_optimized_context_for_query(self, query: str, max_tokens: int = 2000, **kwargs) -> str:
        """
        Generate optimized context for streaming scenarios.

        Args:
            query: User query
            max_tokens: Maximum token budget
            **kwargs: Additional parameters

        Returns:
            Optimized context string
        """
        # Check if we need to regenerate context (avoid excessive regeneration during streaming)
        now = datetime.now()
        if (self.last_context_generation and
            now - self.last_context_generation < self.context_generation_frequency):
            logger.debug("Skipping context regeneration - too recent")
            return ""

        if self.streaming_mode:
            return self._get_streaming_optimized_context(query, max_tokens, **kwargs)
        else:
            return self.memory.get_context_for_query(query, max_tokens, **kwargs)

    def _get_streaming_optimized_context(self, query: str, max_tokens: int, **kwargs) -> str:
        """Generate streaming-optimized context with aggressive prioritization."""
        context_parts = []
        estimated_tokens = 0

        # Token allocation strategy for streaming
        session_info_budget = int(max_tokens * 0.05)      # 5% for session info
        recent_context_budget = int(max_tokens * 0.60)    # 60% for recent context
        reasoning_budget = int(max_tokens * 0.20)         # 20% for current reasoning
        facts_budget = int(max_tokens * 0.10)             # 10% for facts
        stats_budget = int(max_tokens * 0.05)             # 5% for stats

        def estimate_tokens(text: str) -> int:
            return len(text.split()) * 1.3

        # 1. Minimal session info (only if not deterministic)
        if not self.memory._is_deterministic_mode() and estimated_tokens < session_info_budget:
            session_id = self.memory.session_id.replace("session_", "")
            current_time = datetime.now()
            session_info = f"Session: {session_id}, {current_time.strftime('%H:%M')}"
            context_parts.append(session_info)
            estimated_tokens += estimate_tokens(session_info)

        # 2. PRIORITIZED recent context (streaming-optimized)
        if self.memory.working_memory and estimated_tokens < session_info_budget + recent_context_budget:
            context_parts.append("\\n--- Recent Context ---")

            # Get only the MOST recent item for streaming to prevent explosion
            for item in self.memory.working_memory[-1:]:  # Only last 1 item instead of 3
                if "content" in item and estimated_tokens < session_info_budget + recent_context_budget:
                    content = item["content"]
                    role = item.get('role', 'unknown')

                    # SMART TRUNCATION for streaming - keep only key information
                    if len(content) > 200:  # Truncate long content in streaming mode
                        content = content[:200] + "..."

                    # Get timestamp for context
                    message_time = ""
                    if "timestamp" in item:
                        try:
                            timestamp = datetime.fromisoformat(item["timestamp"])
                            message_time = f", {timestamp.strftime('%H:%M')}"
                        except:
                            pass

                    context_line = f"- [{role}{message_time}] {content}"
                    if estimated_tokens + estimate_tokens(context_line) <= session_info_budget + recent_context_budget:
                        context_parts.append(context_line)
                        estimated_tokens += estimate_tokens(context_line)

        # 3. Current reasoning (essential for tool-calling workflows)
        if (self.memory.current_cycle and
            estimated_tokens < session_info_budget + recent_context_budget + reasoning_budget):

            context_parts.append("\\n--- Current Reasoning ---")

            # Add only the MOST recent thought and observation (streaming-focused)
            if self.memory.current_cycle.thoughts:
                latest_thought = self.memory.current_cycle.thoughts[-1]
                thought_text = f"Thought: {latest_thought.content[:100]}"  # Truncated
                if estimated_tokens + estimate_tokens(thought_text) <= session_info_budget + recent_context_budget + reasoning_budget:
                    context_parts.append(thought_text)
                    estimated_tokens += estimate_tokens(thought_text)

            if self.memory.current_cycle.observations:
                latest_obs = self.memory.current_cycle.observations[-1]
                obs_text = f"Last Action Result: {str(latest_obs.content)[:80]}"  # Truncated
                if estimated_tokens + estimate_tokens(obs_text) <= session_info_budget + recent_context_budget + reasoning_budget:
                    context_parts.append(obs_text)
                    estimated_tokens += estimate_tokens(obs_text)

        # 4. Only TOP facts (heavily filtered for streaming)
        budget_so_far = session_info_budget + recent_context_budget + reasoning_budget
        if estimated_tokens < budget_so_far + facts_budget:
            query_results = self.memory.query_memory(query, include_links=False)
            if query_results["facts"]:
                facts_section = ["\\n--- Key Knowledge ---"]

                # Take only TOP 2 facts with highest confidence
                top_facts = sorted(query_results["facts"],
                                 key=lambda f: f.get("confidence", 0), reverse=True)[:2]

                for fact_dict in top_facts:
                    if estimated_tokens < budget_so_far + facts_budget:
                        fact = self.memory.Fact.from_dict(fact_dict)
                        fact_text = f"- {fact}"  # No confidence in streaming to save tokens
                        if estimated_tokens + estimate_tokens(fact_text) <= budget_so_far + facts_budget:
                            facts_section.append(fact_text)
                            estimated_tokens += estimate_tokens(fact_text)

                if len(facts_section) > 1:
                    context_parts.extend(facts_section)

        # 5. Minimal stats (streaming-optimized)
        budget_so_far = session_info_budget + recent_context_budget + reasoning_budget + facts_budget
        if estimated_tokens < budget_so_far + stats_budget:
            stats = self.memory.get_statistics()
            # Ultra-compact stats for streaming
            stats_text = f"\\n--- Stats ---\\nFacts: {stats['knowledge_graph']['total_facts']}, Cycles: {stats['total_react_cycles']}"
            if estimated_tokens + estimate_tokens(stats_text) <= max_tokens:
                context_parts.append(stats_text)

        final_context = "\\n".join(context_parts)
        final_tokens = estimate_tokens(final_context)

        # Update generation timestamp
        self.last_context_generation = datetime.now()

        # Log optimization results
        logger.debug(f"Streaming-optimized context: {final_tokens}/{max_tokens} tokens "
                    f"(session:{session_info_budget}, recent:{recent_context_budget}, "
                    f"reasoning:{reasoning_budget}, facts:{facts_budget})")

        if final_tokens > max_tokens:
            logger.warning(f"Streaming context ({final_tokens} tokens) still exceeds limit ({max_tokens} tokens)")

        return final_context


# Example usage for the memory system
def enhanced_get_context_for_query(memory_instance, query: str, max_tokens: int = 2000) -> str:
    """Enhanced context generation with smart allocation."""

    context_manager = SmartContextManager()

    # Get current memory state
    working_memory_size = len(memory_instance.working_memory) if memory_instance.working_memory else 0
    current_cycle_active = bool(memory_instance.current_cycle)

    # Dynamic token allocation based on conversation state
    if working_memory_size > 10:  # Heavy conversation
        # Prioritize recent context, reduce facts
        recent_budget = int(max_tokens * 0.70)
        facts_budget = int(max_tokens * 0.15)
        reasoning_budget = int(max_tokens * 0.10)
        session_budget = int(max_tokens * 0.05)
    elif current_cycle_active:  # Active reasoning
        # Prioritize reasoning context
        reasoning_budget = int(max_tokens * 0.40)
        recent_budget = int(max_tokens * 0.40)
        facts_budget = int(max_tokens * 0.15)
        session_budget = int(max_tokens * 0.05)
    else:  # Normal conversation
        # Balanced allocation
        recent_budget = int(max_tokens * 0.50)
        reasoning_budget = int(max_tokens * 0.20)
        facts_budget = int(max_tokens * 0.20)
        session_budget = int(max_tokens * 0.10)

    return context_manager.build_context(
        memory_instance, query,
        session_budget=session_budget,
        recent_budget=recent_budget,
        reasoning_budget=reasoning_budget,
        facts_budget=facts_budget
    )


class SmartContextManager:
    """Helper class for intelligent context building with token budgets."""

    def build_context(self, memory, query: str, **budget_kwargs) -> str:
        """Build context with strict token budgets per section."""
        # Implementation would go here
        pass