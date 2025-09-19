"""
FactsExtractor - Semantic Triplet Extraction with Ontological Grounding

This module provides sophisticated fact extraction using the semantic models framework
with Dublin Core, Schema.org, SKOS, and CiTO ontologies for structured knowledge building.

Key Features:
- Ontology-based triplet extraction
- Semantic categorization (working, episodic, semantic)
- Knowledge graph compatibility
- Replacement for basic NLP extraction in AbstractLLM
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .base import BaseCognitive, PromptTemplate, CognitiveConfig
from .prompts.facts_prompts import (
    build_extraction_prompt,
    ABSTRACTLLM_FACTS_PROMPT,
    SEMANTIC_ANALYSIS_PROMPT
)


class OntologyType(Enum):
    """Ontology types based on adoption rates from semantic models"""
    DCTERMS = "dcterms"  # Dublin Core Terms - 60-70% adoption
    SCHEMA = "schema"    # Schema.org - 35-45% adoption
    SKOS = "skos"        # SKOS - 15-20% adoption
    CITO = "cito"        # CiTO - 15-20% adoption


class FactCategory(Enum):
    """Categorization for knowledge graph building"""
    WORKING = "working"      # Recent, temporary facts
    EPISODIC = "episodic"    # Experience-based facts
    SEMANTIC = "semantic"    # General knowledge facts


@dataclass
class SemanticFact:
    """Enhanced fact with ontological grounding, aggregation support, and provenance tracking"""
    subject: str
    predicate: str
    object: str
    category: FactCategory
    ontology: OntologyType
    confidence: float
    context: str
    extraction_method: str = "llm_guided"
    timestamp: str = None
    usage_count: int = 1  # Start at 1, not 0 - tracks how many times this fact was encountered
    importance: float = None  # Derived from confidence and usage_count
    provenance: List[str] = field(default_factory=list)  # List of interaction IDs where this fact was encountered

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.importance is None:
            self.importance = self.confidence  # Initial importance equals confidence

        # Normalize subject and object to canonical forms
        self.subject = self._normalize_canonical_form(self.subject)
        self.object = self._normalize_canonical_form(self.object)

    def to_rdf_triple(self) -> str:
        """Convert to RDF triple format"""
        prefix = self.ontology.value
        return f"ex:{self._normalize_identifier(self.subject)} {prefix}:{self.predicate} ex:{self._normalize_identifier(self.object)}"

    def _normalize_identifier(self, text: str) -> str:
        """Normalize text for use as RDF identifier"""
        # Remove articles and normalize
        normalized = re.sub(r'^(the|a|an)\s+', '', text.lower())
        # Replace spaces with hyphens
        normalized = re.sub(r'\s+', '-', normalized)
        # Remove special characters
        normalized = re.sub(r'[^\w\-]', '', normalized)
        return normalized

    def _normalize_canonical_form(self, text: str) -> str:
        """Normalize text to canonical form for consistency and deduplication"""
        # Remove common articles and normalize case
        normalized = re.sub(r'^(the|a|an)\s+', '', text.strip(), flags=re.IGNORECASE)

        # Handle common technology name variations
        tech_mappings = {
            r'\bML\b': 'machine learning',
            r'\bAI\b': 'artificial intelligence',
            r'\bNLP\b': 'natural language processing',
            r'\bLLM\b': 'large language model',
            r'\bGPT\b': 'Generative Pre-trained Transformer',
            r'\bAPI\b': 'Application Programming Interface',
            r'\bUI\b': 'user interface',
            r'\bUX\b': 'user experience'
        }

        for pattern, replacement in tech_mappings.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # Normalize whitespace and case
        normalized = ' '.join(normalized.split()).lower()

        # Handle possessive forms
        normalized = re.sub(r"'s\b", '', normalized)

        return normalized.strip()

    def create_fact_key(self) -> str:
        """Create a unique key for this fact for deduplication purposes"""
        # Use normalized subject, predicate, and object for key
        return f"{self.subject}|{self.predicate}|{self.object}|{self.ontology.value}"

    def merge_with(self, other_fact: 'SemanticFact') -> 'SemanticFact':
        """Merge this fact with another duplicate fact, combining their properties and provenance"""
        if self.create_fact_key() != other_fact.create_fact_key():
            raise ValueError("Cannot merge facts with different keys")

        # Combine usage counts
        new_usage_count = self.usage_count + other_fact.usage_count

        # Average confidences weighted by usage count
        total_confidence = (self.confidence * self.usage_count +
                          other_fact.confidence * other_fact.usage_count)
        new_confidence = total_confidence / new_usage_count

        # Calculate new importance (confidence * log(usage_count + 1))
        import math
        new_importance = new_confidence * math.log(new_usage_count + 1)

        # Use the more recent context (longer is usually more informative)
        new_context = other_fact.context if len(other_fact.context) > len(self.context) else self.context

        # Keep the earlier timestamp
        new_timestamp = min(self.timestamp, other_fact.timestamp)

        # Combine provenance lists (remove duplicates while preserving order)
        combined_provenance = list(self.provenance)
        for prov_id in other_fact.provenance:
            if prov_id not in combined_provenance:
                combined_provenance.append(prov_id)

        # Create merged fact
        return SemanticFact(
            subject=self.subject,
            predicate=self.predicate,
            object=self.object,
            category=self.category,
            ontology=self.ontology,
            confidence=new_confidence,
            context=new_context,
            extraction_method=self.extraction_method,
            timestamp=new_timestamp,
            usage_count=new_usage_count,
            importance=new_importance,
            provenance=combined_provenance
        )

    def update_importance(self):
        """Update importance based on current confidence and usage count"""
        import math
        self.importance = self.confidence * math.log(self.usage_count + 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization"""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "category": self.category.value,
            "ontology": self.ontology.value,
            "confidence": self.confidence,
            "context": self.context,
            "extraction_method": self.extraction_method,
            "timestamp": self.timestamp,
            "usage_count": self.usage_count,
            "importance": self.importance,
            "provenance": self.provenance,
            "rdf_triple": self.to_rdf_triple(),
            "fact_key": self.create_fact_key()
        }

    def __str__(self) -> str:
        """Human-readable representation"""
        return f"{self.subject} --[{self.predicate}]--> {self.object} ({self.confidence:.2f})"


@dataclass
class CategorizedFacts:
    """Facts organized by category for knowledge management"""
    working: List[SemanticFact]
    episodic: List[SemanticFact]
    semantic: List[SemanticFact]
    total_extracted: int
    extraction_time: float
    source_context: str

    def all_facts(self) -> List[SemanticFact]:
        """Get all facts in a single list"""
        return self.working + self.episodic + self.semantic

    def get_by_ontology(self, ontology: OntologyType) -> List[SemanticFact]:
        """Get facts filtered by ontology type"""
        return [fact for fact in self.all_facts() if fact.ontology == ontology]

    def get_high_confidence(self, threshold: float = 0.8) -> List[SemanticFact]:
        """Get facts above confidence threshold"""
        return [fact for fact in self.all_facts() if fact.confidence >= threshold]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "working": [fact.to_dict() for fact in self.working],
            "episodic": [fact.to_dict() for fact in self.episodic],
            "semantic": [fact.to_dict() for fact in self.semantic],
            "total_extracted": self.total_extracted,
            "extraction_time": self.extraction_time,
            "source_context": self.source_context,
            "summary": {
                "working_count": len(self.working),
                "episodic_count": len(self.episodic),
                "semantic_count": len(self.semantic),
                "avg_confidence": sum(f.confidence for f in self.all_facts()) / max(len(self.all_facts()), 1)
            }
        }


class FactsExtractor(BaseCognitive):
    """Semantic triplet extraction using ontological framework"""

    def __init__(self, llm_provider: str = "ollama", model: str = None, **kwargs):
        """
        Initialize FactsExtractor

        Args:
            llm_provider: LLM provider (default: ollama)
            model: Model to use (default: granite3.3:2b)
            **kwargs: Additional configuration
        """
        # Use default model for facts extractor if not specified
        if model is None:
            model = CognitiveConfig.get_default_model("facts_extractor")

        # Apply default configuration (low temperature for consistency)
        config = CognitiveConfig.get_default_config("facts_extractor")
        config.update(kwargs)

        super().__init__(llm_provider, model, **config)

        # Load ontology predicates and validation patterns
        self.ontology_predicates = self._load_ontology_predicates()
        self._load_prompt_templates()

        # Fact registry for deduplication and aggregation
        self.fact_registry: Dict[str, SemanticFact] = {}  # key -> fact mapping

    def _load_ontology_predicates(self) -> Dict[OntologyType, List[str]]:
        """Load predicate patterns from semantic models framework"""
        return {
            OntologyType.DCTERMS: [
                # Core Dublin Core predicates
                "creator", "title", "description", "created", "modified", "publisher",
                "isPartOf", "hasPart", "references", "isReferencedBy",
                "requires", "isRequiredBy", "replaces", "isReplacedBy",
                "subject", "language", "format", "rights", "license",

                # Extended predicates for knowledge relationships
                "enables", "allows", "causes", "leads", "results",
                "temporal", "precedes", "follows", "during",
                "influences", "affects", "impacts", "determines",
                "originates", "derives", "evolved", "developed"
            ],
            OntologyType.SCHEMA: [
                # Core Schema.org predicates
                "name", "description", "author", "about", "mentions",
                "sameAs", "oppositeOf", "member", "memberOf",
                "teaches", "learns", "knows", "worksFor",
                "startDate", "endDate", "location", "organizer",

                # Extended predicates for knowledge representation
                "enables", "allows", "uses", "implements", "supports",
                "requires", "provides", "contains", "includes",
                "specializedFor", "optimizedFor", "designedFor",
                "foundationalTo", "basedOn", "derivedFrom",
                "exhibits", "demonstrates", "achieves", "performs",
                "processes", "transforms", "generates", "produces",
                "facilitates", "improves", "enhances", "extends"
            ],
            OntologyType.SKOS: [
                # Core SKOS predicates for concept relationships
                "broader", "narrower", "related", "exactMatch", "closeMatch",
                "prefLabel", "altLabel", "definition", "note",
                "inScheme", "topConceptOf", "hasTopConcept",

                # Extended predicates for conceptual knowledge
                "similarTo", "differFrom", "contrastsWith", "complementTo",
                "subsumes", "encompassed", "instanceOf", "typeOf",
                "categoryOf", "classifiedAs", "exemplifiedBy", "represents"
            ],
            OntologyType.CITO: [
                "supports", "isSupportedBy", "disagreesWith", "isDisagreedWithBy",
                "usesDataFrom", "providesDataFor", "extends", "isExtendedBy",
                "discusses", "isDiscussedBy", "confirms", "isConfirmedBy",
                "cites", "isCitedBy", "critiques", "isCritiquedBy"
            ]
        }

    def _load_prompt_templates(self):
        """Load and prepare prompt templates"""
        self.templates = {
            "general": PromptTemplate(
                build_extraction_prompt("general", "quality", 10),
                required_vars=["content"]
            ),
            "interaction": PromptTemplate(
                ABSTRACTLLM_FACTS_PROMPT,
                required_vars=["content"]
            ),
            "semantic": PromptTemplate(
                SEMANTIC_ANALYSIS_PROMPT,
                required_vars=["content"]
            )
        }

    def _process(self, content: str, context_type: str = "general",
                max_facts: int = 10, interaction_id: str = None) -> CategorizedFacts:
        """
        Core fact extraction processing

        Args:
            content: Content to extract facts from
            context_type: Type of content ("general", "interaction", "semantic")
            max_facts: Maximum number of facts to extract

        Returns:
            CategorizedFacts object with organized results
        """
        import time
        start_time = time.time()

        # Select appropriate template
        template_key = context_type if context_type in self.templates else "general"
        template = self.templates[template_key]

        # Build prompt
        prompt_text = template.format(content=content)
        prompt_text += f"\n\nCONTENT TO ANALYZE:\n{content}\n\nEXTRACTED FACTS:"

        # Generate extraction
        response = self.session.generate(prompt_text)

        # Parse response into facts with provenance
        facts = self._parse_extracted_facts(response.content, content, interaction_id)

        # Aggregate facts (handle duplicates and merge)
        aggregated_facts = self._aggregate_facts(facts)

        # Categorize facts
        categorized = self._categorize_facts(aggregated_facts)

        extraction_time = time.time() - start_time

        return CategorizedFacts(
            working=[f for f in categorized if f.category == FactCategory.WORKING],
            episodic=[f for f in categorized if f.category == FactCategory.EPISODIC],
            semantic=[f for f in categorized if f.category == FactCategory.SEMANTIC],
            total_extracted=len(categorized),
            extraction_time=extraction_time,
            source_context=content[:200] + "..." if len(content) > 200 else content
        )

    def extract_facts(self, content: str, context_type: str = "general",
                     max_facts: int = 10, interaction_id: str = None) -> CategorizedFacts:
        """
        Extract semantic triplets from content

        Args:
            content: Text content to analyze
            context_type: Context type for optimized extraction
            max_facts: Maximum facts to extract

        Returns:
            CategorizedFacts with organized results

        Raises:
            CognitiveError: If extraction fails
        """
        return self.process(content, context_type, max_facts, interaction_id)

    def _parse_extracted_facts(self, response: str, original_content: str, interaction_id: str = None) -> List[SemanticFact]:
        """Parse LLM response into structured facts"""
        facts = []

        for line in response.strip().split('\n'):
            line = line.strip()
            if '|' in line and len(line.split('|')) >= 6:
                try:
                    parts = [p.strip() for p in line.split('|')]
                    subject, predicate, obj, ontology, category, confidence = parts[:6]

                    # Validate ontology
                    try:
                        ont_type = OntologyType(ontology.lower())
                    except ValueError:
                        continue  # Skip invalid ontologies

                    # Validate category
                    try:
                        cat_type = FactCategory(category.lower())
                    except ValueError:
                        continue  # Skip invalid categories

                    # Validate predicate against ontology
                    if not self._validate_predicate(predicate, ont_type):
                        continue  # Skip invalid predicates

                    # Parse confidence
                    try:
                        conf_value = float(confidence)
                        if not 0.0 <= conf_value <= 1.0:
                            continue
                    except ValueError:
                        continue

                    # Create provenance list with interaction ID if provided
                    provenance = [interaction_id] if interaction_id else []

                    fact = SemanticFact(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        category=cat_type,
                        ontology=ont_type,
                        confidence=conf_value,
                        context=original_content[:200] + "...",
                        provenance=provenance
                    )
                    facts.append(fact)

                except (ValueError, IndexError):
                    continue  # Skip malformed lines

        return facts

    def _validate_predicate(self, predicate: str, ontology: OntologyType) -> bool:
        """Validate that predicate belongs to the specified ontology"""
        valid_predicates = self.ontology_predicates.get(ontology, [])
        # Remove namespace prefix if present
        clean_predicate = predicate.split(':')[-1]
        return clean_predicate in valid_predicates

    def _aggregate_facts(self, new_facts: List[SemanticFact]) -> List[SemanticFact]:
        """
        Aggregate facts by checking for duplicates and merging with existing facts

        Args:
            new_facts: List of newly extracted facts

        Returns:
            List of aggregated facts (new + updated existing facts)
        """
        aggregated_facts = []

        for fact in new_facts:
            fact_key = fact.create_fact_key()

            if fact_key in self.fact_registry:
                # Duplicate found - merge with existing fact
                existing_fact = self.fact_registry[fact_key]
                merged_fact = existing_fact.merge_with(fact)
                self.fact_registry[fact_key] = merged_fact
                aggregated_facts.append(merged_fact)
            else:
                # New fact - add to registry
                self.fact_registry[fact_key] = fact
                aggregated_facts.append(fact)

        return aggregated_facts

    def _categorize_facts(self, facts: List[SemanticFact]) -> List[SemanticFact]:
        """Apply additional categorization logic and validation"""
        for fact in facts:
            # Validate and potentially adjust categorization
            fact.category = self._determine_category(fact)

        return facts

    def _determine_category(self, fact: SemanticFact) -> FactCategory:
        """Determine appropriate category for a fact"""
        # Working facts - session/temporary indicators
        working_indicators = ["current", "this", "now", "today", "session", "user", "ai"]
        if any(indicator in fact.subject.lower() or indicator in fact.object.lower()
               for indicator in working_indicators):
            return FactCategory.WORKING

        # Episodic facts - temporal/experiential
        if fact.ontology == OntologyType.CITO:  # Citations and evidence
            return FactCategory.EPISODIC

        if fact.ontology == OntologyType.DCTERMS and fact.predicate in ["created", "modified"]:
            return FactCategory.EPISODIC

        # Semantic facts - conceptual knowledge
        if fact.ontology == OntologyType.SKOS:  # Concept relationships
            return FactCategory.SEMANTIC

        if fact.ontology == OntologyType.SCHEMA and fact.predicate in ["about", "sameAs", "definition"]:
            return FactCategory.SEMANTIC

        # Default based on confidence
        if fact.confidence > 0.8:
            return FactCategory.SEMANTIC
        elif fact.confidence > 0.5:
            return FactCategory.EPISODIC
        else:
            return FactCategory.WORKING

    def replace_basic_extraction(self, content: str, source_type: str,
                               source_id: str) -> List[Dict[str, Any]]:
        """
        Replace the basic NLP extraction in memory.py

        Args:
            content: Content to extract facts from
            source_type: Type of source (for compatibility)
            source_id: Source identifier (for compatibility)

        Returns:
            List of fact dictionaries in AbstractLLM format
        """
        # Extract facts using semantic approach
        categorized_facts = self.extract_facts(content, context_type="interaction")

        # Convert to format expected by AbstractLLM memory system
        return [
            {
                'fact_id': f"semantic_{i}",
                'subject': fact.subject,
                'predicate': f"{fact.ontology.value}:{fact.predicate}",
                'object': fact.object,
                'confidence': fact.confidence,
                'source_type': source_type,
                'source_id': source_id,
                'category': fact.category.value,
                'ontology': fact.ontology.value,
                'extraction_method': 'semantic_ontological',
                'timestamp': fact.timestamp,
                'metadata': {
                    'rdf_triple': fact.to_rdf_triple(),
                    'context_snippet': fact.context
                }
            }
            for i, fact in enumerate(categorized_facts.all_facts())
        ]

    def extract_interaction_facts(self, interaction_context: Dict[str, Any], interaction_id: str = None) -> CategorizedFacts:
        """
        Extract facts specifically from AbstractLLM interaction context

        Args:
            interaction_context: Full interaction context dictionary
            interaction_id: Unique identifier for this interaction (for provenance tracking)

        Returns:
            CategorizedFacts specialized for interaction analysis
        """
        # Build content from interaction context
        content_parts = []

        if "query" in interaction_context:
            content_parts.append(f"USER QUERY: {interaction_context['query']}")

        if "response_content" in interaction_context:
            content_parts.append(f"AI RESPONSE: {interaction_context['response_content']}")

        if "tools_executed" in interaction_context:
            for tool in interaction_context["tools_executed"]:
                tool_name = tool.get("name", "unknown")
                tool_result = str(tool.get("result", ""))[:300]
                content_parts.append(f"TOOL {tool_name}: {tool_result}")

        content = "\n\n".join(content_parts)

        # Extract facts with provenance tracking
        try:
            result = self.extract_facts(content, context_type="interaction", max_facts=12, interaction_id=interaction_id)
            # Ensure we always return a valid CategorizedFacts object
            if result is None:
                # Create empty result if extraction failed
                return CategorizedFacts(
                    working=[], episodic=[], semantic=[],
                    total_extracted=0, extraction_time=0.0,
                    source_context=content[:200] + "..." if len(content) > 200 else content
                )
            return result
        except Exception as e:
            # Log error and return empty result instead of None
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Fact extraction failed: {e}")
            return CategorizedFacts(
                working=[], episodic=[], semantic=[],
                total_extracted=0, extraction_time=0.0,
                source_context=content[:200] + "..." if len(content) > 200 else content
            )

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about fact extraction performance"""
        base_stats = self.get_performance_stats()

        # Add facts extractor specific stats
        base_stats.update({
            "model_used": self.model,
            "provider": self.provider,
            "ontologies_supported": [ont.value for ont in OntologyType],
            "categories_supported": [cat.value for cat in FactCategory],
            "predicates_available": sum(len(preds) for preds in self.ontology_predicates.values())
        })

        return base_stats