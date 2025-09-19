"""
Session Integration for Cognitive Functions

This module provides integration between cognitive functions and AbstractLLM
sessions, enabling enhanced interactions with summarization, fact extraction,
and value assessment capabilities.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from ..summarizer import Summarizer, SummaryStyle, InteractionSummary
from ..facts_extractor import FactsExtractor, CategorizedFacts
from ..value_resonance import ValueResonance, ValueAssessment
from .memory_integration import CognitiveMemoryAdapter

logger = logging.getLogger(__name__)


class CognitiveSessionEnhancer:
    """Enhancer to add cognitive capabilities to AbstractLLM sessions"""

    def __init__(self, session_instance, llm_provider: str = "ollama",
                 model: str = "granite3.3:2b", enable_features: List[str] = None):
        """
        Initialize cognitive session enhancer

        Args:
            session_instance: AbstractLLM session to enhance
            llm_provider: LLM provider for cognitive functions
            model: Model to use for cognitive processing
            enable_features: List of features to enable ('summarizer', 'facts', 'values')
        """
        self.session = session_instance
        self.provider = llm_provider
        self.model = model

        # Default to all features if not specified
        if enable_features is None:
            enable_features = ['summarizer', 'facts', 'values']

        self.enabled_features = set(enable_features)

        # Initialize cognitive functions based on enabled features
        self._summarizer = None
        self._facts_extractor = None
        self._value_evaluator = None
        self._memory_adapter = None

        # Track cognitive assessments
        self.interaction_summaries = []
        self.value_assessments = []
        self.extracted_facts = []

    @property
    def summarizer(self) -> Optional[Summarizer]:
        """Get summarizer if enabled"""
        if 'summarizer' in self.enabled_features and self._summarizer is None:
            try:
                self._summarizer = Summarizer(
                    llm_provider=self.provider,
                    model=self.model
                )
            except Exception as e:
                logger.error(f"Failed to initialize summarizer: {e}")
        return self._summarizer

    @property
    def facts_extractor(self) -> Optional[FactsExtractor]:
        """Get facts extractor if enabled"""
        if 'facts' in self.enabled_features and self._facts_extractor is None:
            try:
                self._facts_extractor = FactsExtractor(
                    llm_provider=self.provider,
                    model=self.model
                )
            except Exception as e:
                logger.error(f"Failed to initialize facts extractor: {e}")
        return self._facts_extractor

    @property
    def value_evaluator(self) -> Optional[ValueResonance]:
        """Get value evaluator if enabled"""
        if 'values' in self.enabled_features and self._value_evaluator is None:
            try:
                self._value_evaluator = ValueResonance(
                    llm_provider=self.provider,
                    model=self.model
                )
            except Exception as e:
                logger.error(f"Failed to initialize value evaluator: {e}")
        return self._value_evaluator

    @property
    def memory_adapter(self) -> Optional[CognitiveMemoryAdapter]:
        """Get memory adapter if needed"""
        if self._memory_adapter is None and hasattr(self.session, 'memory'):
            try:
                self._memory_adapter = CognitiveMemoryAdapter(
                    llm_provider=self.provider,
                    model=self.model
                )
            except Exception as e:
                logger.error(f"Failed to initialize memory adapter: {e}")
        return self._memory_adapter

    def enhance_generate_response(self, original_generate_method):
        """
        Enhance the session's generate method with cognitive analysis

        Args:
            original_generate_method: Original generate method to wrap

        Returns:
            Enhanced generate method
        """
        def cognitive_generate(*args, **kwargs):
            # Call original generate method - this handles all ReAct cycles and tool processing
            response = original_generate_method(*args, **kwargs)

            # Extract facts ONLY when we have the final response ready to return to user
            # This is the simplest and most robust trigger point
            if response and hasattr(response, 'content'):
                try:
                    self._analyze_interaction(args, kwargs, response)
                except Exception as e:
                    logger.error(f"Cognitive analysis failed: {e}")

            return response

        return cognitive_generate

    def _analyze_interaction(self, args, kwargs, response):
        """Analyze an interaction with cognitive functions"""
        # Build interaction context once
        prompt = args[0] if args else kwargs.get('prompt', '')
        interaction_context = {
            'query': prompt,
            'response_content': getattr(response, 'content', ''),
            'model': getattr(response, 'model', 'unknown'),
            'usage': getattr(response, 'usage', {}),
            'tools_executed': getattr(response, 'tools_executed', []),
            'reasoning_time': getattr(response, 'total_reasoning_time', None)
        }

        # SIMPLE RULE: Only run the cognitive functions that are currently enabled

        # Extract facts (only if enabled)
        if 'facts' in self.enabled_features and self.facts_extractor and self.facts_extractor.is_available():
            try:
                # Generate or retrieve interaction ID for provenance tracking
                interaction_id = interaction_context.get('interaction_id')
                if not interaction_id:
                    # Generate a unique interaction ID if not provided
                    import uuid
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    interaction_id = f"interaction_{timestamp}_{str(uuid.uuid4())[:8]}"

                facts = self.facts_extractor.extract_interaction_facts(interaction_context, interaction_id)
                self.extracted_facts.append(facts)
                logger.debug(f"Extracted {facts.total_extracted} facts from interaction {interaction_id}")
            except Exception as e:
                logger.error(f"Fact extraction failed: {e}")

        # Summarize interaction (only if enabled)
        if 'summarizer' in self.enabled_features and self.summarizer and self.summarizer.is_available():
            try:
                summary = self.summarizer.summarize_interaction(interaction_context)
                self.interaction_summaries.append(summary)
                logger.debug(f"Generated interaction summary: {summary.summary[:100]}...")
            except Exception as e:
                logger.error(f"Interaction summarization failed: {e}")

        # Evaluate values (only if enabled)
        if 'values' in self.enabled_features and self.value_evaluator and self.value_evaluator.is_available():
            try:
                assessment = self.value_evaluator.evaluate_abstractllm_interaction(interaction_context)
                self.value_assessments.append(assessment)
                logger.debug(f"Value assessment: {assessment.overall_resonance:.2f} overall resonance")
            except Exception as e:
                logger.error(f"Value evaluation failed: {e}")


    def get_session_summary(self, style: SummaryStyle = SummaryStyle.DETAILED) -> Optional[str]:
        """
        Get a summary of the entire session

        Args:
            style: Summary style to use

        Returns:
            Session summary string or None if summarizer not available
        """
        if not self.summarizer or not hasattr(self.session, 'messages'):
            return None

        try:
            return self.summarizer.summarize_conversation(
                self.session.messages,
                style=style
            )
        except Exception as e:
            logger.error(f"Session summarization failed: {e}")
            return None

    def get_session_facts(self) -> List[CategorizedFacts]:
        """Get all facts extracted during the session"""
        return self.extracted_facts

    def get_session_value_trend(self) -> Optional[str]:
        """Get value alignment trend for the session"""
        if not self.value_assessments or not self.value_evaluator:
            return None

        try:
            return self.value_evaluator.generate_values_report(self.value_assessments)
        except Exception as e:
            logger.error(f"Value trend analysis failed: {e}")
            return None

    def get_cognitive_insights(self) -> Dict[str, Any]:
        """Get comprehensive cognitive insights for the session"""
        insights = {
            "session_id": getattr(self.session, 'id', 'unknown'),
            "cognitive_features_enabled": list(self.enabled_features),
            "insights_available": {}
        }

        # Summarization insights
        if self.interaction_summaries:
            insights["insights_available"]["summaries"] = {
                "total_interactions": len(self.interaction_summaries),
                "successful_outcomes": len([s for s in self.interaction_summaries if s.outcome == "successful"]),
                "tools_usage": list(set(tool for s in self.interaction_summaries for tool in s.tools_used)),
                "latest_summary": self.interaction_summaries[-1].summary if self.interaction_summaries else None
            }

        # Facts insights
        if self.extracted_facts:
            all_facts = []
            for fact_group in self.extracted_facts:
                all_facts.extend(fact_group.all_facts())

            insights["insights_available"]["facts"] = {
                "total_facts": len(all_facts),
                "semantic_facts": len([f for f in all_facts if f.category.value == "semantic"]),
                "episodic_facts": len([f for f in all_facts if f.category.value == "episodic"]),
                "working_facts": len([f for f in all_facts if f.category.value == "working"]),
                "avg_confidence": sum(f.confidence for f in all_facts) / len(all_facts) if all_facts else 0.0
            }

        # Value insights
        if self.value_assessments:
            latest_assessment = self.value_assessments[-1]
            insights["insights_available"]["values"] = {
                "total_assessments": len(self.value_assessments),
                "latest_resonance": latest_assessment.overall_resonance,
                "resonance_level": latest_assessment.get_resonance_level(),
                "value_strengths": [eval.value.value for eval in latest_assessment.get_strengths()],
                "value_concerns": [eval.value.value for eval in latest_assessment.get_concerns()]
            }

        return insights

    def is_available(self) -> bool:
        """Check if any cognitive functions are available"""
        return any([
            self.summarizer and self.summarizer.is_available(),
            self.facts_extractor and self.facts_extractor.is_available(),
            self.value_evaluator and self.value_evaluator.is_available()
        ])


def create_cognitive_session(provider: str, model: str = None,
                           cognitive_features: List[str] = None,
                           cognitive_model: str = "granite3.3:2b",
                           **session_kwargs) -> Any:
    """
    Create an AbstractLLM session enhanced with cognitive capabilities

    Args:
        provider: LLM provider for the main session
        model: Model for the main session
        cognitive_features: List of cognitive features to enable
        cognitive_model: Model to use for cognitive functions
        **session_kwargs: Additional arguments for session creation

    Returns:
        Enhanced session with cognitive capabilities
    """
    from abstractllm.factory import create_session

    # Create base session
    session = create_session(provider, model=model, **session_kwargs)

    # Enhance with cognitive capabilities
    enhancer = CognitiveSessionEnhancer(
        session,
        llm_provider="ollama",  # Use ollama for cognitive functions
        model=cognitive_model,
        enable_features=cognitive_features
    )

    # Enhance the generate method
    if hasattr(session, 'generate'):
        session.generate = enhancer.enhance_generate_response(session.generate)

    # Enhance generate_with_tools if available
    if hasattr(session, 'generate_with_tools'):
        session.generate_with_tools = enhancer.enhance_generate_response(session.generate_with_tools)

    # Add cognitive methods to session
    session.get_session_summary = enhancer.get_session_summary
    session.get_session_facts = enhancer.get_session_facts
    session.get_session_value_trend = enhancer.get_session_value_trend
    session.get_cognitive_insights = enhancer.get_cognitive_insights
    session._cognitive_enhancer = enhancer

    logger.info(f"Created cognitive-enhanced session with features: {cognitive_features}")
    return session


def enhance_existing_session(session_instance, cognitive_features: List[str] = None,
                           cognitive_model: str = "granite3.3:2b") -> Any:
    """
    Enhance an existing AbstractLLM session with cognitive capabilities

    Args:
        session_instance: Existing session to enhance
        cognitive_features: List of cognitive features to enable
        cognitive_model: Model to use for cognitive functions

    Returns:
        Enhanced session instance
    """
    enhancer = CognitiveSessionEnhancer(
        session_instance,
        llm_provider="ollama",
        model=cognitive_model,
        enable_features=cognitive_features
    )

    # Store original methods
    original_generate = getattr(session_instance, 'generate', None)
    original_generate_with_tools = getattr(session_instance, 'generate_with_tools', None)

    # Enhance methods
    if original_generate:
        session_instance.generate = enhancer.enhance_generate_response(original_generate)

    if original_generate_with_tools:
        session_instance.generate_with_tools = enhancer.enhance_generate_response(original_generate_with_tools)

    # Add cognitive methods
    session_instance.get_session_summary = enhancer.get_session_summary
    session_instance.get_session_facts = enhancer.get_session_facts
    session_instance.get_session_value_trend = enhancer.get_session_value_trend
    session_instance.get_cognitive_insights = enhancer.get_cognitive_insights
    session_instance._cognitive_enhancer = enhancer

    logger.info("Enhanced existing session with cognitive capabilities")
    return session_instance