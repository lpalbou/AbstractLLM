#!/usr/bin/env python3
"""
Context Optimization Proposal for ReAct Streaming

This demonstrates smarter context allocation based on:
1. Current reasoning step type (Think vs Act vs Observe)
2. Query relevance scoring
3. Dynamic token allocation
4. Context freshness weighting
"""

from typing import Dict, List, Tuple
import re
from datetime import datetime, timedelta

class SmartContextManager:
    """Enhanced context manager with intelligent allocation."""

    def __init__(self):
        self.context_strategies = {
            'think': {'reasoning': 0.5, 'facts': 0.3, 'working': 0.2},
            'act': {'reasoning': 0.3, 'facts': 0.4, 'working': 0.3},
            'observe': {'reasoning': 0.4, 'facts': 0.2, 'working': 0.4}
        }

    def detect_reasoning_phase(self, query: str) -> str:
        """Detect which ReAct phase we're in based on query content."""
        # Thinking indicators
        think_patterns = [
            r'\b(?:think|consider|plan|analyze|reason|decide|strategy)\b',
            r'\b(?:approach|method|way|how)\b',
            r'\b(?:next|should|need to|ought to)\b'
        ]

        # Acting indicators
        act_patterns = [
            r'\b(?:read|write|execute|run|call|use)\b',
            r'\b(?:tool_call|function|command)\b',
            r'<\|tool_call\|>'
        ]

        # Observing indicators
        observe_patterns = [
            r'\b(?:result|output|response|found|discovered)\b',
            r'\b(?:based on|given that|now that|since)\b',
            r'\b(?:observation|evidence|data)\b'
        ]

        think_score = sum(len(re.findall(pattern, query, re.I)) for pattern in think_patterns)
        act_score = sum(len(re.findall(pattern, query, re.I)) for pattern in act_patterns)
        observe_score = sum(len(re.findall(pattern, query, re.I)) for pattern in observe_patterns)

        if act_score > think_score and act_score > observe_score:
            return 'act'
        elif observe_score > think_score:
            return 'observe'
        else:
            return 'think'  # Default to thinking phase

    def score_relevance(self, query: str, content: str) -> float:
        """Score content relevance to query using keyword overlap."""
        # Simple keyword-based relevance (could be enhanced with embeddings)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)

    def calculate_freshness_weight(self, timestamp_str: str) -> float:
        """Calculate freshness weight (more recent = higher weight)."""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age_minutes = (datetime.now() - timestamp).total_seconds() / 60

            # Exponential decay: fresh content heavily weighted
            if age_minutes < 5:      # Very fresh
                return 1.0
            elif age_minutes < 30:   # Recent
                return 0.8
            elif age_minutes < 120:  # Somewhat old
                return 0.6
            else:                    # Old
                return 0.4
        except:
            return 0.5  # Default weight

    def optimize_context_allocation(self, query: str, max_tokens: int,
                                  working_memory: List[Dict],
                                  reasoning_cycles: List[Dict],
                                  facts: List[Dict]) -> Dict[str, int]:
        """Dynamically allocate tokens based on query and reasoning phase."""

        phase = self.detect_reasoning_phase(query)
        strategy = self.context_strategies[phase]

        # Calculate base allocation
        base_allocation = {
            'reasoning': int(max_tokens * strategy['reasoning']),
            'facts': int(max_tokens * strategy['facts']),
            'working': int(max_tokens * strategy['working'])
        }

        # Adjust based on availability
        if not reasoning_cycles:
            # No reasoning to include - redistribute to facts and working
            extra = base_allocation['reasoning']
            base_allocation['facts'] += extra // 2
            base_allocation['working'] += extra // 2
            base_allocation['reasoning'] = 0

        if not facts:
            # No facts to include - redistribute to reasoning and working
            extra = base_allocation['facts']
            base_allocation['reasoning'] += extra // 2
            base_allocation['working'] += extra // 2
            base_allocation['facts'] = 0

        return base_allocation

    def select_optimal_working_memory(self, query: str, working_memory: List[Dict],
                                    token_budget: int) -> List[Dict]:
        """Select most relevant working memory items within token budget."""
        if not working_memory:
            return []

        # Score each item by relevance and freshness
        scored_items = []
        for item in working_memory:
            content = item.get('content', '')
            timestamp = item.get('timestamp', '')

            relevance = self.score_relevance(query, content)
            freshness = self.calculate_freshness_weight(timestamp)

            # Combined score: relevance is primary, freshness is secondary
            combined_score = relevance * 0.7 + freshness * 0.3

            scored_items.append((combined_score, item))

        # Sort by score and select within token budget
        scored_items.sort(key=lambda x: x[0], reverse=True)

        selected_items = []
        used_tokens = 0

        for score, item in scored_items:
            content = item.get('content', '')
            estimated_tokens = len(content.split()) * 1.3

            if used_tokens + estimated_tokens <= token_budget:
                selected_items.append(item)
                used_tokens += estimated_tokens
            else:
                break

        return selected_items

    def select_optimal_facts(self, query: str, facts: List[Dict],
                           token_budget: int) -> List[Dict]:
        """Select most relevant facts within token budget."""
        if not facts:
            return []

        # Score facts by relevance and confidence
        scored_facts = []
        for fact in facts:
            content = fact.get('content', '')
            confidence = fact.get('confidence', 0.5)

            relevance = self.score_relevance(query, content)

            # Combined score: relevance + confidence
            combined_score = relevance * 0.6 + confidence * 0.4

            scored_facts.append((combined_score, fact))

        # Sort and select within budget
        scored_facts.sort(key=lambda x: x[0], reverse=True)

        selected_facts = []
        used_tokens = 0

        for score, fact in scored_facts:
            content = fact.get('content', '')
            estimated_tokens = len(content.split()) * 1.3

            if used_tokens + estimated_tokens <= token_budget:
                selected_facts.append(fact)
                used_tokens += estimated_tokens
            else:
                break

        return selected_facts

# Example usage for the memory system
def enhanced_get_context_for_query(memory_instance, query: str, max_tokens: int = 2000) -> str:
    """Enhanced context generation with smart allocation."""

    context_manager = SmartContextManager()

    # Get current memory state
    working_memory = memory_instance.working_memory
    reasoning_cycles = memory_instance.react_cycles
    facts = list(memory_instance.knowledge_graph.facts.values())

    # Calculate optimal allocation
    allocation = context_manager.optimize_context_allocation(
        query, max_tokens, working_memory, reasoning_cycles, facts
    )

    print(f"🧠 Context Allocation for '{query[:50]}...':")
    print(f"   Phase: {context_manager.detect_reasoning_phase(query)}")
    print(f"   Tokens: Reasoning={allocation['reasoning']}, Facts={allocation['facts']}, Working={allocation['working']}")

    # Build optimized context
    context_parts = []

    # Session info (minimal)
    context_parts.append(f"Session: {memory_instance.session_id}")

    # Optimal working memory selection
    if allocation['working'] > 0:
        selected_working = context_manager.select_optimal_working_memory(
            query, working_memory, allocation['working']
        )
        if selected_working:
            context_parts.append("\\n--- Relevant Context ---")
            for item in selected_working:
                context_parts.append(f"- {item.get('content', '')}")

    # Reasoning context (if in budget and relevant)
    if allocation['reasoning'] > 0 and reasoning_cycles:
        context_parts.append("\\n--- Current Reasoning ---")
        # Add most recent reasoning cycle
        latest_cycle = reasoning_cycles[-1]
        context_parts.append(f"Cycle: {latest_cycle.get('cycle_id', 'current')}")

    # Optimal facts selection
    if allocation['facts'] > 0:
        selected_facts = context_manager.select_optimal_facts(
            query, facts, allocation['facts']
        )
        if selected_facts:
            context_parts.append("\\n--- Relevant Knowledge ---")
            for fact in selected_facts:
                context_parts.append(f"- {fact.get('content', '')}")

    return "\\n".join(context_parts)


if __name__ == "__main__":
    # Test the context manager
    cm = SmartContextManager()

    test_queries = [
        "I need to think about the best approach for this problem",
        "Please read the file and analyze its contents",
        "Based on the results from the previous tool call, I can see that..."
    ]

    for query in test_queries:
        phase = cm.detect_reasoning_phase(query)
        allocation = cm.optimize_context_allocation(query, 2000, [], [], [])
        print(f"Query: '{query}'")
        print(f"Phase: {phase}")
        print(f"Allocation: {allocation}")
        print("-" * 50)