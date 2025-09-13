"""
Advanced memory system for AbstractLLM with ReAct and knowledge graph support.

This module provides:
- Short-term working memory (scratchpad)
- Long-term episodic memory
- Knowledge graph extraction and storage
- ReAct reasoning traces
- Memory consolidation and retrieval
"""

from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""
    WORKING = "working"  # Short-term scratchpad
    EPISODIC = "episodic"  # Conversation history
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge


@dataclass
class Thought:
    """A single thought in the reasoning process."""
    
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    confidence: float = 1.0
    thought_type: str = "reasoning"  # reasoning, planning, reflection
    
    def __str__(self):
        return f"[{self.thought_type}] {self.content}"


@dataclass
class Action:
    """An action taken during reasoning."""
    
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str  # Why this action
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    
    def __str__(self):
        return f"{self.tool_name}({json.dumps(self.arguments)})"


@dataclass
class Observation:
    """An observation from action execution."""
    
    content: Any
    source: str  # Which tool produced this
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    success: bool = True
    
    def __str__(self):
        return f"[{self.source}] {self.content}"


@dataclass
class KnowledgeTriple:
    """A fact represented as subject-predicate-object triple."""
    
    subject: str
    predicate: str
    object: Any
    confidence: float = 1.0
    source: str = "extracted"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.subject} --[{self.predicate}]--> {self.object}"
    
    def to_dict(self):
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }


class ReActScratchpad:
    """
    Scratchpad for ReAct-style reasoning with thought-action-observation cycles.
    """
    
    def __init__(self, max_iterations: int = 10):
        self.thoughts: List[Thought] = []
        self.actions: List[Action] = []
        self.observations: List[Observation] = []
        self.current_iteration = 0
        self.max_iterations = max_iterations
        self.start_time = datetime.now()
        
    def add_thought(self, content: str, thought_type: str = "reasoning", confidence: float = 1.0):
        """Add a reasoning thought."""
        thought = Thought(
            content=content,
            iteration=self.current_iteration,
            thought_type=thought_type,
            confidence=confidence
        )
        self.thoughts.append(thought)
        logger.debug(f"Thought {self.current_iteration}: {content}")
        
    def add_action(self, tool_name: str, arguments: Dict[str, Any], reasoning: str):
        """Record an action to be taken."""
        action = Action(
            tool_name=tool_name,
            arguments=arguments,
            reasoning=reasoning,
            iteration=self.current_iteration
        )
        self.actions.append(action)
        logger.debug(f"Action {self.current_iteration}: {tool_name}")
        
    def add_observation(self, content: Any, source: str, success: bool = True):
        """Record an observation from action execution."""
        observation = Observation(
            content=content,
            source=source,
            iteration=self.current_iteration,
            success=success
        )
        self.observations.append(observation)
        logger.debug(f"Observation {self.current_iteration}: {source} - {str(content)[:100]}")
        
    def next_iteration(self):
        """Move to next reasoning iteration."""
        self.current_iteration += 1
        if self.current_iteration >= self.max_iterations:
            logger.warning(f"Reached maximum iterations ({self.max_iterations})")
            
    def get_trace(self) -> str:
        """Get formatted reasoning trace."""
        trace = []
        
        # Group by iteration
        for i in range(self.current_iteration + 1):
            # Add thoughts for this iteration
            iteration_thoughts = [t for t in self.thoughts if t.iteration == i]
            for thought in iteration_thoughts:
                trace.append(f"Thought {i}: {thought.content}")
            
            # Add actions
            iteration_actions = [a for a in self.actions if a.iteration == i]
            for action in iteration_actions:
                trace.append(f"Action {i}: {action.tool_name}({json.dumps(action.arguments)})")
                trace.append(f"  Reasoning: {action.reasoning}")
            
            # Add observations
            iteration_obs = [o for o in self.observations if o.iteration == i]
            for obs in iteration_obs:
                status = "✓" if obs.success else "✗"
                trace.append(f"Observation {i} {status}: {str(obs.content)[:200]}")
        
        return "\n".join(trace)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "iterations": self.current_iteration,
            "thoughts": len(self.thoughts),
            "actions": len(self.actions),
            "observations": len(self.observations),
            "duration_seconds": duration,
            "success_rate": sum(1 for o in self.observations if o.success) / max(len(self.observations), 1)
        }


class KnowledgeGraph:
    """
    Simple knowledge graph for storing and querying facts.
    """
    
    def __init__(self):
        self.triples: List[KnowledgeTriple] = []
        self.subject_index: Dict[str, List[KnowledgeTriple]] = defaultdict(list)
        self.predicate_index: Dict[str, List[KnowledgeTriple]] = defaultdict(list)
        self.object_index: Dict[str, List[KnowledgeTriple]] = defaultdict(list)
        
    def add_triple(self, subject: str, predicate: str, object: Any, 
                  confidence: float = 1.0, source: str = "extracted"):
        """Add a fact triple to the graph."""
        triple = KnowledgeTriple(
            subject=self._normalize(subject),
            predicate=self._normalize(predicate),
            object=object,
            confidence=confidence,
            source=source
        )
        
        self.triples.append(triple)
        self.subject_index[triple.subject].append(triple)
        self.predicate_index[triple.predicate].append(triple)
        
        # Index objects if they're strings
        if isinstance(object, str):
            self.object_index[self._normalize(object)].append(triple)
            
        logger.debug(f"Added triple: {triple}")
        
    def query_subject(self, subject: str) -> List[KnowledgeTriple]:
        """Find all facts about a subject."""
        return self.subject_index.get(self._normalize(subject), [])
    
    def query_predicate(self, predicate: str) -> List[KnowledgeTriple]:
        """Find all facts with a predicate."""
        return self.predicate_index.get(self._normalize(predicate), [])
    
    def query_object(self, object: str) -> List[KnowledgeTriple]:
        """Find all facts with an object."""
        return self.object_index.get(self._normalize(object), [])
    
    def query(self, subject: Optional[str] = None, 
             predicate: Optional[str] = None,
             object: Optional[str] = None) -> List[KnowledgeTriple]:
        """Query with any combination of subject, predicate, object."""
        results = self.triples
        
        if subject:
            results = [t for t in results if t.subject == self._normalize(subject)]
        if predicate:
            results = [t for t in results if t.predicate == self._normalize(predicate)]
        if object:
            results = [t for t in results if t.object == object or 
                      (isinstance(t.object, str) and t.object == self._normalize(object))]
            
        return results
    
    def get_related(self, entity: str, max_depth: int = 2) -> Set[str]:
        """Find all entities related to the given entity up to max_depth."""
        entity = self._normalize(entity)
        visited = set()
        to_visit = {entity}
        
        for _ in range(max_depth):
            new_entities = set()
            for current in to_visit:
                if current in visited:
                    continue
                    
                visited.add(current)
                
                # Find direct connections
                for triple in self.query_subject(current):
                    if isinstance(triple.object, str):
                        new_entities.add(self._normalize(triple.object))
                        
                for triple in self.query_object(current):
                    new_entities.add(triple.subject)
                    
            to_visit = new_entities - visited
            
        return visited - {entity}
    
    def _normalize(self, text: str) -> str:
        """Normalize text for indexing."""
        return text.lower().strip()
    
    def to_json(self) -> List[Dict[str, Any]]:
        """Export graph as JSON."""
        return [t.to_dict() for t in self.triples]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_triples": len(self.triples),
            "unique_subjects": len(self.subject_index),
            "unique_predicates": len(self.predicate_index),
            "unique_objects": len(self.object_index),
            "average_confidence": sum(t.confidence for t in self.triples) / max(len(self.triples), 1)
        }


class ConversationMemory:
    """
    Comprehensive memory system for conversations.
    """
    
    def __init__(self, 
                 max_working_memory: int = 10,
                 consolidation_threshold: int = 5):
        """
        Initialize memory system.
        
        Args:
            max_working_memory: Maximum items in working memory
            consolidation_threshold: When to consolidate to long-term memory
        """
        self.working_memory: List[Dict[str, Any]] = []
        self.episodic_memory: List[Dict[str, Any]] = []
        self.knowledge_graph = KnowledgeGraph()
        self.scratchpad = ReActScratchpad()
        
        self.max_working_memory = max_working_memory
        self.consolidation_threshold = consolidation_threshold
        self.session_start = datetime.now()
        
    def add_to_working_memory(self, item: Dict[str, Any]):
        """Add item to working memory."""
        self.working_memory.append({
            **item,
            "timestamp": datetime.now().isoformat()
        })
        
        # Consolidate if needed
        if len(self.working_memory) > self.max_working_memory:
            self._consolidate_memory()
            
    def _consolidate_memory(self):
        """Move old items from working to episodic memory."""
        # Move oldest items to episodic
        to_move = self.working_memory[:self.consolidation_threshold]
        self.episodic_memory.extend(to_move)
        self.working_memory = self.working_memory[self.consolidation_threshold:]
        
        # Extract facts from consolidated items
        for item in to_move:
            self._extract_facts(item)
            
        logger.info(f"Consolidated {len(to_move)} items to long-term memory")
        
    def _extract_facts(self, item: Dict[str, Any]):
        """Extract facts from a memory item."""
        content = item.get("content", "")
        
        # Simple fact extraction patterns
        patterns = [
            r"(\w+) is (\w+)",  # X is Y
            r"(\w+) has (\w+)",  # X has Y
            r"(\w+) can (\w+)",  # X can Y
            r"(\w+) needs (\w+)",  # X needs Y
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    self.knowledge_graph.add_triple(
                        subject=match[0],
                        predicate="is" if "is" in pattern else pattern.split()[1],
                        object=match[1],
                        source="pattern_extraction"
                    )
    
    def retrieve_relevant(self, query: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query."""
        relevant = []
        
        # Search working memory first (most recent)
        for item in reversed(self.working_memory):
            if query.lower() in str(item).lower():
                relevant.append(item)
                if len(relevant) >= max_items:
                    return relevant
        
        # Search episodic memory
        for item in reversed(self.episodic_memory):
            if query.lower() in str(item).lower():
                relevant.append(item)
                if len(relevant) >= max_items:
                    return relevant
                    
        return relevant
    
    def get_context(self, include_knowledge: bool = True) -> str:
        """Get current memory context for prompting."""
        context = []
        
        # Add recent working memory
        if self.working_memory:
            context.append("Recent context:")
            for item in self.working_memory[-3:]:
                context.append(f"- {item.get('content', '')[:100]}")
        
        # Add ReAct trace if active
        if self.scratchpad.thoughts:
            context.append("\nReasoning trace:")
            context.append(self.scratchpad.get_trace()[-500:])  # Last 500 chars
        
        # Add relevant knowledge
        if include_knowledge and self.knowledge_graph.triples:
            context.append("\nKnown facts:")
            for triple in self.knowledge_graph.triples[-5:]:
                context.append(f"- {triple}")
        
        return "\n".join(context)
    
    def save_state(self) -> Dict[str, Any]:
        """Save memory state for persistence."""
        return {
            "working_memory": self.working_memory,
            "episodic_memory": self.episodic_memory,
            "knowledge_graph": self.knowledge_graph.to_json(),
            "session_start": self.session_start.isoformat(),
            "scratchpad_summary": self.scratchpad.get_summary()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load memory state from saved data."""
        self.working_memory = state.get("working_memory", [])
        self.episodic_memory = state.get("episodic_memory", [])
        
        # Rebuild knowledge graph
        for triple_dict in state.get("knowledge_graph", []):
            self.knowledge_graph.add_triple(
                subject=triple_dict["subject"],
                predicate=triple_dict["predicate"],
                object=triple_dict["object"],
                confidence=triple_dict.get("confidence", 1.0),
                source=triple_dict.get("source", "loaded")
            )
        
        if "session_start" in state:
            self.session_start = datetime.fromisoformat(state["session_start"])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "knowledge_graph": self.knowledge_graph.get_statistics(),
            "scratchpad": self.scratchpad.get_summary(),
            "session_duration": (datetime.now() - self.session_start).total_seconds()
        }