"""
Modern Semantic Extractor - Entity-First Knowledge Graph Builder

This module implements a sophisticated semantic extraction system that:
1. Identifies main entities (people, events, ideas, times, places)
2. Extracts statements about those entities
3. Creates proper semantic relationships using JSON-LD

Based on the proven semantic-extractor.json pattern from promptons.
"""

import re
import json
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from .base import BaseCognitive, PromptTemplate, CognitiveConfig


class EntityType(Enum):
    """Semantic entity types based on Schema.org and other ontologies"""
    # People and Organizations
    PERSON = "schema:Person"
    ORGANIZATION = "schema:Organization"
    GROUP = "schema:Group"
    AGENT = "schema:Agent"

    # Places and Locations
    LOCATION = "schema:Location"
    PLACE = "schema:Place"

    # Events and Activities
    EVENT = "schema:Event"
    PROCESS = "schema:Process"
    ACTIVITY = "schema:Activity"

    # Ideas and Concepts
    CONCEPT = "skos:Concept"
    THEORY = "schema:Theory"
    METHOD = "schema:Method"
    HYPOTHESIS = "schema:Hypothesis"
    DEFINITION = "schema:Definition"
    IDEA = "schema:Idea"

    # Time and Temporal
    TIME_FRAME = "schema:TimeFrame"
    DATE = "schema:Date"
    PERIOD = "schema:Period"

    # Information and Documents
    TEXT = "dcterms:Text"
    PUBLICATION = "schema:Publication"
    DATASET = "schema:Dataset"

    # Systems and Technology
    SYSTEM = "schema:System"
    TECHNOLOGY = "schema:Technology"
    SOFTWARE = "schema:SoftwareApplication"

    # Insights and Findings
    INSIGHT = "schema:Insight"
    FINDING = "schema:Finding"
    DISCOVERY = "schema:Discovery"


class RelationshipType(Enum):
    """Semantic relationship types"""
    # Conceptual relationships
    BROADER = "skos:broader"
    NARROWER = "skos:narrower"
    RELATED = "skos:related"

    # Structural relationships
    IS_PART_OF = "schema:isPartOf"
    HAS_PART = "schema:hasPart"
    CONTAINS = "schema:contains"

    # Knowledge relationships
    EXPLAINS = "schema:explains"
    DESCRIBES = "schema:describes"
    DEFINES = "schema:defines"
    EXEMPLIFIES = "schema:exemplifies"

    # Causal relationships
    CAUSES = "schema:causes"
    INFLUENCES = "schema:influences"
    ENABLES = "schema:enables"
    PREVENTS = "schema:prevents"

    # Temporal relationships
    PRECEDES = "schema:precedes"
    FOLLOWS = "schema:follows"
    DURING = "schema:during"

    # Creation relationships
    CREATES = "dcterms:creator"
    DISCOVERS = "schema:discovers"
    DEVELOPS = "schema:develops"

    # Evidential relationships
    SUPPORTS = "schema:supports"
    CONTRADICTS = "schema:contradicts"
    VALIDATES = "schema:validates"

    # Functional relationships
    USES = "schema:uses"
    IMPLEMENTS = "schema:implements"
    APPLIES = "schema:applies"


@dataclass
class SemanticEntity:
    """A semantic entity with proper JSON-LD structure"""
    id: str
    type: EntityType
    name: str
    description: str
    confidence: float
    properties: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None

    def to_jsonld(self) -> Dict[str, Any]:
        """Convert to JSON-LD entity format"""
        result = {
            "@id": f"entity:{self.id}",
            "@type": self.type.value,
            "schema:name": self.name,
            "schema:description": self.description,
            "schema:confidence": self.confidence
        }

        # Add optional properties
        if self.chunk_id:
            result["chunk_id"] = self.chunk_id
        if self.doc_id:
            result["doc_id"] = self.doc_id

        # Add any additional properties
        result.update(self.properties)

        return result


@dataclass
class SemanticRelationship:
    """A semantic relationship between entities"""
    id: str
    type: RelationshipType
    about_entity: str  # Entity ID
    object_entity: str  # Entity ID
    name: str
    description: str
    confidence: float
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None

    def to_jsonld(self) -> Dict[str, Any]:
        """Convert to JSON-LD relationship format"""
        result = {
            "@id": f"relationship:{self.id}",
            "@type": self.type.value,
            "schema:name": self.name,
            "schema:about": {"@id": f"entity:{self.about_entity}"},
            "schema:object": {"@id": f"entity:{self.object_entity}"},
            "schema:description": self.description,
            "schema:confidence": self.confidence
        }

        if self.chunk_id:
            result["chunk_id"] = self.chunk_id
        if self.doc_id:
            result["doc_id"] = self.doc_id

        return result


@dataclass
class SemanticGraph:
    """Complete semantic knowledge graph"""
    entities: List[SemanticEntity]
    relationships: List[SemanticRelationship]
    context: Dict[str, str] = field(default_factory=dict)

    def to_jsonld(self) -> Dict[str, Any]:
        """Convert entire graph to JSON-LD format"""
        # Standard JSON-LD context
        default_context = {
            "schema": "https://schema.org/",
            "dcterms": "http://purl.org/dc/terms/",
            "cito": "http://purl.org/spar/cito/",
            "skos": "http://www.w3.org/2004/02/skos/core#"
        }

        # Merge with any additional context
        full_context = {**default_context, **self.context}

        # Build graph array
        graph = []

        # Add all entities
        for entity in self.entities:
            graph.append(entity.to_jsonld())

        # Add all relationships
        for relationship in self.relationships:
            graph.append(relationship.to_jsonld())

        return {
            "@context": full_context,
            "@graph": graph
        }


class ModernSemanticExtractor(BaseCognitive):
    """Modern entity-first semantic extractor"""

    def __init__(self, llm_provider: str = "ollama", model: str = None, **kwargs):
        """Initialize modern semantic extractor"""
        if model is None:
            model = CognitiveConfig.get_default_model("semantic_extractor")

        config = CognitiveConfig.get_default_config("semantic_extractor")
        config.update(kwargs)

        super().__init__(llm_provider, model, **config)

        self._load_prompt_templates()

    def _load_prompt_templates(self):
        """Load modern semantic extraction prompts"""
        self.extraction_prompt = PromptTemplate(
            self._build_extraction_prompt(),
            required_vars=["text"]
        )

    def _build_extraction_prompt(self) -> str:
        """Build the comprehensive semantic extraction prompt"""
        return """You are a specialized knowledge graph extraction system with deep expertise in semantic web technologies and knowledge representation. You excel at identifying entities and relationships in text using standard semantic vocabularies.

## TASK: Transform text into a semantic knowledge graph

### STEP 1: ENTITY IDENTIFICATION
Identify the MAIN ENTITIES in the text that represent knowledge concepts. Focus on:

**PEOPLE**: Individuals mentioned by name, role, or title
- Type: schema:Person
- Extract: Full names, titles, affiliations
- Example: "Dr. Sarah Johnson", "the researcher", "Einstein"

**ORGANIZATIONS**: Companies, institutions, groups
- Type: schema:Organization
- Extract: Company names, universities, government bodies
- Example: "MIT", "Google", "Stanford University"

**EVENTS**: Significant happenings, conferences, discoveries
- Type: schema:Event
- Extract: Named events, conferences, historical moments
- Example: "2024 I/O conference", "the discovery", "the meeting"

**IDEAS/CONCEPTS**: Theories, methods, principles, insights
- Type: skos:Concept, schema:Theory, schema:Method
- Extract: Scientific concepts, methodologies, principles
- Example: "machine learning", "quantum mechanics", "the new approach"

**TIME/DATES**: Temporal references, periods, timeframes
- Type: schema:TimeFrame, schema:Date
- Extract: Specific dates, periods, temporal contexts
- Example: "2024", "between 1990 and 2020", "the Renaissance"

**PLACES**: Locations, geographical references
- Type: schema:Location
- Extract: Cities, countries, venues, facilities
- Example: "Mountain View", "the laboratory", "Silicon Valley"

**INSIGHTS/FINDINGS**: Key discoveries, conclusions, results
- Type: schema:Insight, schema:Finding
- Extract: Research findings, conclusions, key insights
- Example: "the main finding", "this discovery", "the breakthrough"

### STEP 2: RELATIONSHIP EXTRACTION
For each pair of entities, determine meaningful semantic relationships:

**CREATION RELATIONSHIPS**:
- dcterms:creator (X created Y)
- schema:discovers (X discovered Y)
- schema:develops (X developed Y)

**KNOWLEDGE RELATIONSHIPS**:
- schema:explains (X explains Y)
- schema:describes (X describes Y)
- skos:broader (X is broader than Y)
- skos:narrower (X is more specific than Y)

**TEMPORAL RELATIONSHIPS**:
- schema:during (X happened during Y)
- schema:precedes (X came before Y)
- schema:follows (X came after Y)

**CAUSAL RELATIONSHIPS**:
- schema:causes (X causes Y)
- schema:influences (X influences Y)
- schema:enables (X enables Y)

**STRUCTURAL RELATIONSHIPS**:
- schema:isPartOf (X is part of Y)
- schema:hasPart (X contains Y)
- schema:uses (X uses Y)

### OUTPUT FORMAT: JSON-LD Knowledge Graph

Return a valid JSON structure with:

```json
{
  "entities": [
    {
      "id": "unique_entity_id",
      "type": "schema:Person|schema:Organization|schema:Event|skos:Concept|schema:TimeFrame|schema:Location|schema:Insight",
      "name": "Entity Name",
      "description": "Brief description of the entity and its role in the text",
      "confidence": 0.0-1.0,
      "properties": {}
    }
  ],
  "relationships": [
    {
      "id": "unique_rel_id",
      "type": "dcterms:creator|schema:explains|schema:during|schema:causes|etc",
      "about_entity": "id_of_subject_entity",
      "object_entity": "id_of_object_entity",
      "name": "relationship_name",
      "description": "Description of how the entities are related",
      "confidence": 0.0-1.0
    }
  ]
}
```

### QUALITY CRITERIA:
1. **Entity Relevance**: Only extract entities that are central to the text's meaning
2. **Relationship Precision**: Use the most specific relationship type available
3. **Completeness**: Capture the main knowledge structure without noise
4. **Confidence**: Rate based on how clearly the entity/relationship is stated

### TEXT TO ANALYZE:
{text}

### EXTRACTED KNOWLEDGE GRAPH:
"""

    def extract_knowledge_graph(self, text: str, doc_id: str = None,
                               chunk_id: str = None) -> SemanticGraph:
        """
        Extract a semantic knowledge graph from text

        Args:
            text: Text content to analyze
            doc_id: Optional document identifier
            chunk_id: Optional chunk identifier

        Returns:
            SemanticGraph with entities and relationships
        """
        # Generate extraction prompt
        prompt_text = self.extraction_prompt.format(text=text)

        # Get LLM response
        response = self.session.generate(prompt_text)

        # Parse the JSON response
        try:
            # Extract JSON from response (handle potential markdown formatting)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response.content, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON directly
                json_text = response.content.strip()
                if not json_text.startswith('{'):
                    # Look for the first { to the last }
                    start = json_text.find('{')
                    end = json_text.rfind('}')
                    if start != -1 and end != -1:
                        json_text = json_text[start:end+1]

            # Parse the extracted JSON
            parsed_data = json.loads(json_text)

            # Convert to semantic graph
            return self._convert_to_semantic_graph(parsed_data, doc_id, chunk_id)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Log error and return empty graph
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to parse semantic extraction: {e}")
            logger.debug(f"Raw response: {response.content}")

            return SemanticGraph(entities=[], relationships=[])

    def _convert_to_semantic_graph(self, data: Dict[str, Any],
                                 doc_id: str = None, chunk_id: str = None) -> SemanticGraph:
        """Convert parsed JSON to SemanticGraph"""
        entities = []
        relationships = []

        # Process entities
        for entity_data in data.get("entities", []):
            try:
                entity = SemanticEntity(
                    id=entity_data["id"],
                    type=EntityType(entity_data["type"]),
                    name=entity_data["name"],
                    description=entity_data["description"],
                    confidence=float(entity_data["confidence"]),
                    properties=entity_data.get("properties", {}),
                    doc_id=doc_id,
                    chunk_id=chunk_id
                )
                entities.append(entity)
            except (KeyError, ValueError) as e:
                # Skip malformed entities
                continue

        # Process relationships
        for rel_data in data.get("relationships", []):
            try:
                relationship = SemanticRelationship(
                    id=rel_data["id"],
                    type=RelationshipType(rel_data["type"]),
                    about_entity=rel_data["about_entity"],
                    object_entity=rel_data["object_entity"],
                    name=rel_data["name"],
                    description=rel_data["description"],
                    confidence=float(rel_data["confidence"]),
                    doc_id=doc_id,
                    chunk_id=chunk_id
                )
                relationships.append(relationship)
            except (KeyError, ValueError) as e:
                # Skip malformed relationships
                continue

        return SemanticGraph(entities=entities, relationships=relationships)

    def extract_from_interaction(self, interaction_context: Dict[str, Any],
                               interaction_id: str = None) -> SemanticGraph:
        """
        Extract semantic graph from AbstractLLM interaction

        Args:
            interaction_context: Interaction context dictionary
            interaction_id: Unique interaction identifier

        Returns:
            SemanticGraph with extracted knowledge
        """
        # Build content from interaction
        content_parts = []

        if "query" in interaction_context:
            content_parts.append(f"USER QUERY: {interaction_context['query']}")

        if "response_content" in interaction_context:
            content_parts.append(f"AI RESPONSE: {interaction_context['response_content']}")

        if "tools_executed" in interaction_context:
            for tool in interaction_context["tools_executed"]:
                tool_name = tool.get("name", "unknown")
                tool_result = str(tool.get("result", ""))[:500]  # Limit length
                content_parts.append(f"TOOL {tool_name}: {tool_result}")

        content = "\n\n".join(content_parts)

        # Extract using the main method
        return self.extract_knowledge_graph(
            content,
            doc_id=f"interaction_{interaction_id}" if interaction_id else None,
            chunk_id="full_interaction"
        )

    def _process(self, content: str, **kwargs) -> SemanticGraph:
        """
        Core processing method required by BaseCognitive

        Args:
            content: Text content to analyze
            **kwargs: Additional arguments (doc_id, chunk_id, etc.)

        Returns:
            SemanticGraph with extracted knowledge
        """
        doc_id = kwargs.get('doc_id')
        chunk_id = kwargs.get('chunk_id')
        return self.extract_knowledge_graph(content, doc_id, chunk_id)