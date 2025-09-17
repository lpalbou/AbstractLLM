"""
Summarizer - Clear, Concise, Efficient Summarization

This module provides the Summarizer abstraction for generating clear, concise summaries
using optimized system prompts and fast models like granite3.3:2b.

Key Features:
- Multiple summary styles (concise, detailed, bullet points, executive)
- Context-aware summarization for different content types
- Optimized for AbstractLLM interactions
- Performance monitoring and error handling
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union
from datetime import datetime

from .base import BaseCognitive, PromptTemplate, CognitiveConfig
from .prompts.summarizer_prompts import (
    build_summarizer_prompt,
    ABSTRACTLLM_INTERACTION_PROMPT
)


class SummaryStyle(Enum):
    """Different summarization styles available"""
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"


class ContentType(Enum):
    """Types of content that can be summarized"""
    GENERAL = "general"
    INTERACTION = "interaction"
    DOCUMENT = "document"
    CONVERSATION = "conversation"


@dataclass
class InteractionSummary:
    """Structured summary of an AbstractLLM interaction"""
    summary: str
    key_insights: List[str]
    tools_used: List[str]
    outcome: str  # "successful", "partial", "failed"
    duration: Optional[float]
    model_used: Optional[str]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            "summary": self.summary,
            "key_insights": self.key_insights,
            "tools_used": self.tools_used,
            "outcome": self.outcome,
            "duration": self.duration,
            "model_used": self.model_used,
            "timestamp": self.timestamp
        }


class Summarizer(BaseCognitive):
    """Clear, concise, efficient summarization using granite3.3:2b"""

    def __init__(self, llm_provider: str = "ollama", model: str = None, **kwargs):
        """
        Initialize Summarizer

        Args:
            llm_provider: LLM provider (default: ollama)
            model: Model to use (default: granite3.3:2b)
            **kwargs: Additional configuration
        """
        # Use default model for summarizer if not specified
        if model is None:
            model = CognitiveConfig.get_default_model("summarizer")

        # Apply default configuration
        config = CognitiveConfig.get_default_config("summarizer")
        config.update(kwargs)

        super().__init__(llm_provider, model, **config)

        # Initialize prompt templates
        self._load_prompt_templates()

    def _load_prompt_templates(self):
        """Load and prepare prompt templates"""
        self.prompt_templates = {
            style.value: PromptTemplate(
                build_summarizer_prompt(style.value),
                required_vars=["content"]
            )
            for style in SummaryStyle
        }

        # Special template for AbstractLLM interactions
        self.interaction_template = PromptTemplate(
            ABSTRACTLLM_INTERACTION_PROMPT,
            required_vars=["content"]
        )

    def _process(self, content: str, style: SummaryStyle = SummaryStyle.CONCISE,
                content_type: ContentType = ContentType.GENERAL,
                optimization: str = "quality") -> str:
        """
        Core summarization processing

        Args:
            content: Content to summarize
            style: Summary style to use
            content_type: Type of content being summarized
            optimization: "speed" or "quality"

        Returns:
            Generated summary
        """
        # Build appropriate prompt
        if content_type == ContentType.INTERACTION:
            prompt_text = self.interaction_template.format(content=content)
        else:
            base_prompt = build_summarizer_prompt(
                style.value,
                content_type.value,
                optimization
            )
            prompt_text = f"{base_prompt}\n\nCONTENT TO SUMMARIZE:\n{content}\n\nSUMMARY:"

        # Generate summary
        response = self.session.generate(prompt_text)
        return response.content.strip()

    def summarize(self, content: str, style: SummaryStyle = SummaryStyle.CONCISE,
                  content_type: ContentType = ContentType.GENERAL,
                  optimization: str = "quality") -> str:
        """
        Generate a summary of the provided content

        Args:
            content: Text content to summarize
            style: Style of summary to generate
            content_type: Type of content being summarized
            optimization: "speed" for fast generation, "quality" for better results

        Returns:
            Generated summary string

        Raises:
            CognitiveError: If summarization fails
        """
        return self.process(content, style, content_type, optimization)

    def summarize_interaction(self, interaction_context: Dict[str, Any]) -> InteractionSummary:
        """
        Create a structured summary of an AbstractLLM interaction

        Args:
            interaction_context: Context dictionary containing interaction details

        Returns:
            InteractionSummary object with structured information
        """
        # Extract relevant information from context
        content = self._extract_interaction_content(interaction_context)

        # Generate summary
        summary_text = self.summarize(
            content,
            style=SummaryStyle.CONCISE,
            content_type=ContentType.INTERACTION
        )

        # Extract structured information
        key_insights = self._extract_key_insights(interaction_context)
        tools_used = self._extract_tools_used(interaction_context)
        outcome = self._assess_interaction_outcome(interaction_context)

        return InteractionSummary(
            summary=summary_text,
            key_insights=key_insights,
            tools_used=tools_used,
            outcome=outcome,
            duration=interaction_context.get("reasoning_time"),
            model_used=interaction_context.get("model")
        )

    def _extract_interaction_content(self, context: Dict[str, Any]) -> str:
        """Extract content from interaction context for summarization"""
        parts = []

        # Add query
        if "query" in context:
            parts.append(f"USER QUERY: {context['query']}")

        # Add response content
        if "response_content" in context:
            parts.append(f"AI RESPONSE: {context['response_content']}")

        # Add tool executions
        if "tools_executed" in context and context["tools_executed"]:
            tool_info = []
            for tool in context["tools_executed"]:
                tool_name = tool.get("name", "unknown")
                tool_result = str(tool.get("result", ""))[:200]  # Limit length
                tool_info.append(f"- {tool_name}: {tool_result}")

            parts.append(f"TOOLS EXECUTED:\n" + "\n".join(tool_info))

        return "\n\n".join(parts)

    def _extract_key_insights(self, context: Dict[str, Any]) -> List[str]:
        """Extract key insights from interaction context"""
        insights = []

        # Check for successful problem solving
        if context.get("tools_executed"):
            insights.append("Used tools to gather information")

        # Check for structured thinking
        if "structured_thinking" in context:
            phases = context["structured_thinking"].get("phases", [])
            if len(phases) > 2:
                insights.append("Applied structured reasoning approach")

        # Check response quality indicators
        analysis = context.get("analysis", {})
        success_indicators = analysis.get("success_indicators", {})

        if success_indicators.get("has_definitive_answer"):
            insights.append("Provided definitive answer")

        if success_indicators.get("showed_reasoning"):
            insights.append("Demonstrated clear reasoning")

        return insights[:3]  # Limit to top 3 insights

    def _extract_tools_used(self, context: Dict[str, Any]) -> List[str]:
        """Extract list of tools used in the interaction"""
        tools = []
        if "tools_executed" in context:
            for tool in context["tools_executed"]:
                tool_name = tool.get("name", "unknown")
                if tool_name not in tools:
                    tools.append(tool_name)
        return tools

    def _assess_interaction_outcome(self, context: Dict[str, Any]) -> str:
        """Assess the overall outcome of the interaction"""
        # Check for errors
        if context.get("error") or context.get("failed", False):
            return "failed"

        # Check for partial completion
        tools_executed = context.get("tools_executed", [])
        if tools_executed:
            # Check if all tools succeeded
            failed_tools = [t for t in tools_executed if not t.get("result")]
            if failed_tools:
                return "partial"

        # Check response quality
        response_content = context.get("response_content", "")
        if len(response_content.strip()) < 50:  # Very short response
            return "partial"

        return "successful"

    def summarize_conversation(self, messages: List[Dict[str, Any]],
                             style: SummaryStyle = SummaryStyle.DETAILED) -> str:
        """
        Summarize a multi-turn conversation

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            style: Summary style to use

        Returns:
            Conversation summary
        """
        # Format conversation for summarization
        conversation_text = "\n".join([
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
            for msg in messages
        ])

        return self.summarize(
            conversation_text,
            style=style,
            content_type=ContentType.CONVERSATION
        )

    def batch_summarize(self, contents: List[str],
                       style: SummaryStyle = SummaryStyle.CONCISE) -> List[str]:
        """
        Summarize multiple pieces of content efficiently

        Args:
            contents: List of content strings to summarize
            style: Summary style to apply to all

        Returns:
            List of summaries in the same order as input
        """
        summaries = []
        for content in contents:
            summary = self.summarize(content, style=style, optimization="speed")
            summaries.append(summary)
        return summaries

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get statistics about summarization performance"""
        base_stats = self.get_performance_stats()

        # Add summarizer-specific stats
        base_stats.update({
            "model_used": self.model,
            "provider": self.provider,
            "available_styles": [style.value for style in SummaryStyle],
            "available_content_types": [ct.value for ct in ContentType]
        })

        return base_stats