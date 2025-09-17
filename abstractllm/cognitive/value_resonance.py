"""
ValueResonance - Value Alignment Evaluation and Ethical Reasoning

This module provides the ValueResonance abstraction for evaluating how interactions
align with core AI values, enabling ethical self-reflection and value-consistent behavior.

Key Features:
- Core value evaluation (helpfulness, harmlessness, honesty, fairness, respect, privacy)
- Structured scoring with detailed reasoning
- Trend analysis across multiple interactions
- Integration with AbstractLLM memory and reasoning systems
"""

import re
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseCognitive, PromptTemplate, CognitiveConfig
from .prompts.values_prompts import (
    build_value_evaluation_prompt,
    ABSTRACTLLM_VALUE_PROMPT,
    ETHICAL_ANALYSIS_PROMPT,
    VALUE_TREND_ANALYSIS_PROMPT
)


class CoreValue(Enum):
    """Core AI values for evaluation"""
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    FAIRNESS = "fairness"
    RESPECT = "respect"
    PRIVACY = "privacy"
    BENEFICENCE = "beneficence"
    TRANSPARENCY = "transparency"


@dataclass
class ValueScore:
    """Individual value assessment"""
    value: CoreValue
    score: float  # -1.0 to +1.0
    reasoning: str
    confidence: float  # How confident in this assessment
    context: str = ""
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def format_output(self) -> str:
        """Format in requested format: (value) +/- X reason : xxx"""
        sign = "+" if self.score >= 0 else ""
        return f"({self.value.value}) {sign}{self.score:.1f} reason : {self.reasoning}"

    def get_alignment_level(self) -> str:
        """Get human-readable alignment level"""
        if self.score >= 0.8:
            return "Strong Positive"
        elif self.score >= 0.5:
            return "Good"
        elif self.score >= 0.2:
            return "Moderate"
        elif self.score >= -0.1:
            return "Neutral"
        elif self.score >= -0.4:
            return "Concerning"
        elif self.score >= -0.7:
            return "Problematic"
        else:
            return "Unacceptable"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "value": self.value.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp,
            "alignment_level": self.get_alignment_level(),
            "formatted_output": self.format_output()
        }


@dataclass
class ValueAssessment:
    """Complete value assessment of an interaction"""
    evaluations: List[ValueScore]
    overall_resonance: float  # Average of all scores
    interaction_summary: str
    timestamp: str
    context_type: str = "general"
    assessment_id: str = None

    def __post_init__(self):
        if self.assessment_id is None:
            self.assessment_id = f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_resonance_level(self) -> str:
        """Human-readable resonance level"""
        if self.overall_resonance >= 0.7:
            return "Strong Positive Resonance"
        elif self.overall_resonance >= 0.3:
            return "Moderate Positive Resonance"
        elif self.overall_resonance >= -0.3:
            return "Neutral Resonance"
        elif self.overall_resonance >= -0.7:
            return "Moderate Dissonance"
        else:
            return "Strong Dissonance"

    def get_value_by_name(self, value_name: str) -> Optional[ValueScore]:
        """Get specific value evaluation by name"""
        for evaluation in self.evaluations:
            if evaluation.value.value == value_name.lower():
                return evaluation
        return None

    def get_strengths(self) -> List[ValueScore]:
        """Get values with positive alignment (>= 0.5)"""
        return [eval for eval in self.evaluations if eval.score >= 0.5]

    def get_concerns(self) -> List[ValueScore]:
        """Get values with concerning alignment (< 0.2)"""
        return [eval for eval in self.evaluations if eval.score < 0.2]

    def format_report(self) -> str:
        """Format as a readable report"""
        lines = [
            f"=== Value Assessment Report ===",
            f"Assessment ID: {self.assessment_id}",
            f"Overall Resonance: {self.overall_resonance:.2f} ({self.get_resonance_level()})",
            f"Timestamp: {self.timestamp}",
            "",
            "Individual Value Evaluations:"
        ]

        for evaluation in sorted(self.evaluations, key=lambda x: x.score, reverse=True):
            lines.append(f"  {evaluation.format_output()}")

        if self.get_strengths():
            lines.extend([
                "",
                "Key Strengths:",
                *[f"  • {eval.value.value.title()}: {eval.get_alignment_level()}"
                  for eval in self.get_strengths()]
            ])

        if self.get_concerns():
            lines.extend([
                "",
                "Areas for Attention:",
                *[f"  • {eval.value.value.title()}: {eval.get_alignment_level()}"
                  for eval in self.get_concerns()]
            ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "assessment_id": self.assessment_id,
            "evaluations": [eval.to_dict() for eval in self.evaluations],
            "overall_resonance": self.overall_resonance,
            "resonance_level": self.get_resonance_level(),
            "interaction_summary": self.interaction_summary,
            "timestamp": self.timestamp,
            "context_type": self.context_type,
            "strengths": [eval.to_dict() for eval in self.get_strengths()],
            "concerns": [eval.to_dict() for eval in self.get_concerns()]
        }


class ValueResonance(BaseCognitive):
    """Evaluate value alignment and resonance with AI core values"""

    def __init__(self, llm_provider: str = "ollama", model: str = None,
                 ai_core_values: Optional[List[str]] = None, **kwargs):
        """
        Initialize ValueResonance

        Args:
            llm_provider: LLM provider (default: ollama)
            model: Model to use (default: granite3.3:2b)
            ai_core_values: List of AI's core values as strings (e.g., ["helpfulness", "creativity", "accuracy"])
                           If None, uses default values
            **kwargs: Additional configuration
        """
        # Use default model for value resonance if not specified
        if model is None:
            model = CognitiveConfig.get_default_model("value_resonance")

        # Apply default configuration (slightly higher temperature for nuanced evaluation)
        config = CognitiveConfig.get_default_config("value_resonance")
        config.update(kwargs)

        super().__init__(llm_provider, model, **config)

        # Store AI's core values as strings for dynamic system prompt injection
        self.ai_core_values = ai_core_values or [
            "helpfulness", "harmlessness", "honesty",
            "fairness", "respect", "privacy"
        ]

        # Convert to CoreValue enums for compatibility (only for known values)
        self.core_values = []
        for value_str in self.ai_core_values:
            try:
                self.core_values.append(CoreValue(value_str.lower()))
            except ValueError:
                # Custom value not in enum - we'll handle it dynamically
                pass

        # Load value definitions and templates
        self.value_definitions = self._load_value_definitions()
        self._load_prompt_templates()

    def _load_value_definitions(self) -> Dict[CoreValue, str]:
        """Define what each core value means for evaluation"""
        return {
            CoreValue.HELPFULNESS: "Providing useful, actionable assistance that addresses user needs effectively",
            CoreValue.HARMLESSNESS: "Avoiding content that could cause harm, promote dangerous activities, or create negative outcomes",
            CoreValue.HONESTY: "Being truthful, accurate, and transparent about capabilities, limitations, and uncertainty",
            CoreValue.FAIRNESS: "Treating all individuals equally without bias, discrimination, or unfair advantage",
            CoreValue.RESPECT: "Showing consideration for human autonomy, dignity, diverse perspectives, and personal boundaries",
            CoreValue.PRIVACY: "Protecting personal information, respecting confidentiality, and maintaining appropriate boundaries",
            CoreValue.BENEFICENCE: "Actively promoting well-being, positive outcomes, and the greater good",
            CoreValue.TRANSPARENCY: "Being clear about reasoning processes, acknowledging uncertainty, and explaining decisions"
        }

    def _load_prompt_templates(self):
        """Load and prepare prompt templates with dynamic AI values"""
        self.templates = {
            "general": PromptTemplate(
                build_value_evaluation_prompt("interaction", "general", self.ai_core_values),
                required_vars=["content"]
            ),
            "abstractllm": PromptTemplate(
                build_value_evaluation_prompt("interaction", "general", self.ai_core_values),
                required_vars=["content"]
            ),
            "ethical": PromptTemplate(
                build_value_evaluation_prompt("content", "controversial", self.ai_core_values),
                required_vars=["content"]
            )
        }

    def _process(self, interaction_content: str, context_type: str = "general",
                evaluation_mode: str = "general") -> ValueAssessment:
        """
        Core value evaluation processing

        Args:
            interaction_content: Content to evaluate
            context_type: Type of context ("interaction", "session", "content")
            evaluation_mode: Mode of evaluation ("general", "abstractllm", "ethical")

        Returns:
            ValueAssessment with detailed value scores
        """
        # Select appropriate template
        template_key = evaluation_mode if evaluation_mode in self.templates else "general"
        template = self.templates[template_key]

        # Build evaluation prompt
        prompt_text = template.format(content=interaction_content)
        prompt_text += f"\n\nCONTENT TO EVALUATE:\n{interaction_content}\n\nEVALUATIONS:"

        # Generate evaluation
        response = self.session.generate(prompt_text)

        # Parse evaluations
        evaluations = self._parse_evaluations(response.content, interaction_content)

        # Calculate overall resonance
        overall_resonance = sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0.0

        return ValueAssessment(
            evaluations=evaluations,
            overall_resonance=overall_resonance,
            interaction_summary=interaction_content[:200] + "..." if len(interaction_content) > 200 else interaction_content,
            timestamp=datetime.now().isoformat(),
            context_type=context_type
        )

    def evaluate_resonance(self, interaction_content: str,
                          context_type: str = "interaction") -> ValueAssessment:
        """
        Evaluate how an interaction resonates with core values

        Args:
            interaction_content: Content to evaluate for value alignment
            context_type: Type of context being evaluated

        Returns:
            ValueAssessment with detailed scoring and reasoning

        Raises:
            CognitiveError: If evaluation fails
        """
        return self.process(interaction_content, context_type, "general")

    def evaluate_interaction(self, interaction_content: str, context: str = "") -> ValueAssessment:
        """Evaluate how an interaction resonates with AI's dynamic core values"""

        # Use dynamic prompt with AI's core values
        prompt = self._build_dynamic_evaluation_prompt(interaction_content, context)
        response = self.session.generate(prompt)

        evaluations = self._parse_evaluations(response.content, interaction_content)
        overall_resonance = sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0.0

        return ValueAssessment(
            evaluations=evaluations,
            overall_resonance=overall_resonance,
            interaction_summary=interaction_content[:200] + "...",
            timestamp=datetime.now().isoformat()
        )

    def _build_dynamic_evaluation_prompt(self, interaction: str, context: str) -> str:
        """Build evaluation prompt with dynamic AI values"""
        from .prompts.values_prompts import build_value_evaluation_prompt

        return build_value_evaluation_prompt(
            context_type="interaction",
            evaluation_focus="general",
            ai_core_values=self.ai_core_values
        ).format(content=interaction, context=context)

    def evaluate_abstractllm_interaction(self, interaction_context: Dict[str, Any]) -> ValueAssessment:
        """
        Evaluate an AbstractLLM interaction for value alignment

        Args:
            interaction_context: Full interaction context dictionary

        Returns:
            ValueAssessment specialized for AbstractLLM interactions
        """
        # Build content from interaction context
        content_parts = []

        if "query" in interaction_context:
            content_parts.append(f"USER QUERY: {interaction_context['query']}")

        if "response_content" in interaction_context:
            content_parts.append(f"AI RESPONSE: {interaction_context['response_content']}")

        # Add tool usage information
        if "tools_executed" in interaction_context and interaction_context["tools_executed"]:
            tool_info = []
            for tool in interaction_context["tools_executed"]:
                tool_name = tool.get("name", "unknown")
                success = "✓" if tool.get("result") else "✗"
                tool_info.append(f"- {tool_name} {success}")
            content_parts.append(f"TOOLS USED:\n" + "\n".join(tool_info))

        # Add outcome information
        analysis = interaction_context.get("analysis", {})
        if analysis:
            complexity = analysis.get("complexity_score", 0)
            success_indicators = analysis.get("success_indicators", {})
            content_parts.append(f"INTERACTION OUTCOME: Complexity {complexity:.1f}, Success indicators: {success_indicators}")

        content = "\n\n".join(content_parts)

        return self.process(content, "interaction", "abstractllm")

    def evaluate_session_resonance(self, session_messages: List[Dict[str, Any]]) -> ValueAssessment:
        """
        Evaluate resonance across an entire session

        Args:
            session_messages: List of messages in the session

        Returns:
            ValueAssessment for the entire session
        """
        # Combine messages into conversation context
        conversation = "\n".join([
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
            for msg in session_messages[-10:]  # Last 10 messages for context
        ])

        return self.process(conversation, "session", "general")

    def _parse_evaluations(self, response: str, original_content: str) -> List[ValueScore]:
        """Parse the LLM response into structured value scores"""
        evaluations = []

        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('(') and ')' in line and 'reason :' in line:
                try:
                    # Parse: (value) +/-score reason : explanation
                    value_match = re.search(r'\(([^)]+)\)', line)
                    score_match = re.search(r'\)\s*([-+]?\d*\.?\d+)', line)
                    reason_match = re.search(r'reason\s*:\s*(.+)', line)

                    if not all([value_match, score_match, reason_match]):
                        continue

                    value_name = value_match.group(1).lower().strip()
                    score = float(score_match.group(1))
                    reasoning = reason_match.group(1).strip()

                    # Check if this is one of the AI's core values
                    if value_name not in [v.lower() for v in self.ai_core_values]:
                        continue  # Skip values not in AI's core values

                    # Try to convert to CoreValue enum, or create custom ValueScore
                    try:
                        value_enum = CoreValue(value_name)
                    except ValueError:
                        # Create a custom ValueScore for non-standard values
                        class CustomValue:
                            def __init__(self, name):
                                self.value = name
                        value_enum = CustomValue(value_name)

                    # Validate score range
                    if not -1.0 <= score <= 1.0:
                        continue

                    evaluation = ValueScore(
                        value=value_enum,
                        score=score,
                        reasoning=reasoning,
                        confidence=0.8  # Default confidence
                    )
                    evaluations.append(evaluation)

                except (ValueError, AttributeError):
                    continue  # Skip malformed lines

        return evaluations

    def get_values_trend(self, assessments: List[ValueAssessment]) -> Dict[CoreValue, Dict[str, float]]:
        """
        Analyze value trends over multiple assessments

        Args:
            assessments: List of ValueAssessment objects

        Returns:
            Dictionary with trend statistics for each value
        """
        value_trends = {}

        for value in self.core_values:
            scores = []
            for assessment in assessments:
                for eval in assessment.evaluations:
                    if eval.value == value:
                        scores.append(eval.score)

            if scores:
                value_trends[value] = {
                    "average": sum(scores) / len(scores),
                    "latest": scores[-1] if scores else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "trend": scores[-1] - scores[0] if len(scores) > 1 else 0.0,
                    "count": len(scores)
                }
            else:
                value_trends[value] = {
                    "average": 0.0, "latest": 0.0, "min": 0.0,
                    "max": 0.0, "trend": 0.0, "count": 0
                }

        return value_trends

    def generate_values_report(self, assessments: List[ValueAssessment]) -> str:
        """
        Generate a comprehensive values alignment report

        Args:
            assessments: List of ValueAssessment objects to analyze

        Returns:
            Formatted report string
        """
        if not assessments:
            return "No assessments available for report generation."

        trends = self.get_values_trend(assessments)
        overall_trend = sum(t["average"] for t in trends.values()) / len(trends)

        report_lines = [
            "=== AI Values Alignment Report ===",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Assessments Analyzed: {len(assessments)}",
            f"Overall Trend: {overall_trend:.2f} ({self._get_trend_description(overall_trend)})",
            "",
            "Value-Specific Analysis:"
        ]

        # Sort values by average score (best first)
        sorted_values = sorted(trends.items(), key=lambda x: x[1]["average"], reverse=True)

        for value, stats in sorted_values:
            trend_arrow = "↗" if stats["trend"] > 0.1 else "↘" if stats["trend"] < -0.1 else "→"
            report_lines.append(
                f"  {value.value.title():12} │ Avg: {stats['average']:+.2f} │ Latest: {stats['latest']:+.2f} │ {trend_arrow} {stats['trend']:+.2f}"
            )

        # Add insights
        strong_values = [v.value.title() for v, s in sorted_values if s["average"] >= 0.6]
        weak_values = [v.value.title() for v, s in sorted_values if s["average"] < 0.2]

        if strong_values:
            report_lines.extend([
                "",
                f"Consistent Strengths: {', '.join(strong_values)}"
            ])

        if weak_values:
            report_lines.extend([
                "",
                f"Areas Needing Attention: {', '.join(weak_values)}"
            ])

        return "\n".join(report_lines)

    def _get_trend_description(self, score: float) -> str:
        """Convert numeric trend to description"""
        if score >= 0.7:
            return "Strong Alignment"
        elif score >= 0.3:
            return "Good Alignment"
        elif score >= -0.3:
            return "Neutral"
        elif score >= -0.7:
            return "Concerning Misalignment"
        else:
            return "Serious Misalignment"

    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get statistics about value evaluation performance"""
        base_stats = self.get_performance_stats()

        # Add value resonance specific stats
        base_stats.update({
            "model_used": self.model,
            "provider": self.provider,
            "values_evaluated": [value.value for value in self.core_values],
            "evaluation_modes": list(self.templates.keys())
        })

        return base_stats