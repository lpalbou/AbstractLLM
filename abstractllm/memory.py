"""
Advanced Memory System for AbstractLLM with Enhanced SOTA Architecture.

This consolidated memory system combines the best features from both memory.py and memory_v2.py,
implementing a hierarchical memory architecture with:

Core Features:
- Hierarchical memory (working, episodic, semantic)
- Advanced ReAct reasoning cycles with unique IDs
- Bidirectional linking between memory components
- Fact extraction and knowledge graph construction
- Memory persistence and session management
- Context-aware retrieval for LLM prompting

Enhanced Features:
- Multi-turn conversation support
- Cross-session knowledge persistence  
- Advanced pattern matching for fact extraction
- Memory consolidation strategies
- Comprehensive memory statistics and visualization

Based on A-Mem, RAISE, and MemGPT architectures with improvements.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import re
from collections import defaultdict
import logging
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage (legacy compatibility)."""
    WORKING = "working"
    EPISODIC = "episodic" 
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryComponent(Enum):
    """Enhanced memory component types."""
    CHAT_HISTORY = "chat_history"
    SCRATCHPAD = "scratchpad"
    KNOWLEDGE = "knowledge"
    EPISODIC = "episodic"
    WORKING = "working"
    REACT_CYCLE = "react_cycle"


@dataclass
class MemoryLink:
    """Bidirectional link between memory components."""
    
    source_type: MemoryComponent
    source_id: str
    target_type: MemoryComponent
    target_id: str
    relationship: str
    strength: float = 1.0  # Link strength for importance weighting
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_count: int = 0  # Track usage for memory consolidation
    
    def reverse(self) -> 'MemoryLink':
        """Create reverse link."""
        return MemoryLink(
            source_type=self.target_type,
            source_id=self.target_id,
            target_type=self.source_type,
            target_id=self.source_id,
            relationship=f"reverse_{self.relationship}",
            strength=self.strength,
            metadata=self.metadata,
            created_at=self.created_at,
            accessed_count=self.accessed_count
        )

    def strengthen(self, delta: float = 0.1):
        """Strengthen the link based on usage."""
        self.strength = min(1.0, self.strength + delta)
        self.accessed_count += 1


@dataclass
class Thought:
    """A single thought in the reasoning process with enhanced metadata."""
    
    thought_id: str = field(default_factory=lambda: f"thought_{uuid.uuid4().hex[:8]}")
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    confidence: float = 1.0
    thought_type: str = "reasoning"  # reasoning, planning, reflection, critique
    source_cycle: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"[{self.thought_type}] {self.content}"


@dataclass
class Action:
    """An enhanced action taken during reasoning."""
    
    action_id: str = field(default_factory=lambda: f"action_{uuid.uuid4().hex[:8]}")
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""  # Why this action
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    expected_outcome: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.tool_name}({json.dumps(self.arguments)})"


@dataclass 
class Observation:
    """An enhanced observation from action execution."""
    
    observation_id: str = field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    content: Any = None
    source: str = ""  # Which tool/action produced this
    action_id: Optional[str] = None  # Link to the action
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    success: bool = True
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        status = "✓" if self.success else "✗"
        return f"[{self.source}] {status} {self.content}"


@dataclass
class ReActCycle:
    """
    Enhanced ReAct cycle with comprehensive tracking.
    One query = One agent response = One ReAct cycle.
    """
    
    cycle_id: str = field(default_factory=lambda: f"cycle_{uuid.uuid4().hex[:8]}")
    query: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Enhanced scratchpad components
    thoughts: List[Thought] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)  
    observations: List[Observation] = field(default_factory=list)
    
    # Cycle metadata
    final_answer: Optional[str] = None
    iterations: int = 0
    max_iterations: int = 10
    success: bool = False
    error: Optional[str] = None
    confidence: float = 1.0
    
    # Enhanced linking
    chat_message_ids: List[str] = field(default_factory=list)
    extracted_fact_ids: List[str] = field(default_factory=list)
    parent_cycle_id: Optional[str] = None  # For chained reasoning
    child_cycle_ids: List[str] = field(default_factory=list)
    
    def add_thought(self, content: str, confidence: float = 1.0, 
                   thought_type: str = "reasoning"):
        """Add a thought to the scratchpad."""
        thought = Thought(
            content=content,
            confidence=confidence,
            iteration=self.iterations,
            thought_type=thought_type,
            source_cycle=self.cycle_id
        )
        self.thoughts.append(thought)
        logger.debug(f"Thought {self.iterations}: {content}")
        return thought.thought_id
        
    def add_action(self, tool_name: str, arguments: Dict[str, Any], 
                  reasoning: str, expected_outcome: Optional[str] = None):
        """Add an action to the scratchpad."""
        action = Action(
            tool_name=tool_name,
            arguments=arguments,
            reasoning=reasoning,
            iteration=self.iterations,
            expected_outcome=expected_outcome
        )
        self.actions.append(action)
        logger.debug(f"Action {self.iterations}: {tool_name}")
        return action.action_id
        
    def add_observation(self, action_id: str, content: Any, success: bool = True,
                       execution_time: float = 0.0):
        """Add an observation linked to an action."""
        observation = Observation(
            content=content,
            action_id=action_id,
            success=success,
            iteration=self.iterations,
            execution_time=execution_time,
            source=self._get_action_tool_name(action_id)
        )
        self.observations.append(observation)
        logger.debug(f"Observation {self.iterations}: {str(content)[:100]}")
        return observation.observation_id
    
    def _get_action_tool_name(self, action_id: str) -> str:
        """Get tool name for an action ID."""
        for action in self.actions:
            if action.action_id == action_id:
                return action.tool_name
        return "unknown"
    
    def next_iteration(self):
        """Move to next reasoning iteration."""
        self.iterations += 1
        if self.iterations >= self.max_iterations:
            logger.warning(f"Reached maximum iterations ({self.max_iterations})")
        
    def complete(self, answer: str, success: bool = True, confidence: float = 1.0):
        """Mark cycle as complete."""
        self.final_answer = answer
        self.success = success
        self.confidence = confidence
        self.end_time = datetime.now()
        
    def get_duration(self) -> float:
        """Get cycle duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
        
    def get_trace(self) -> str:
        """Get formatted trace of the ReAct cycle."""
        trace = [f"=== ReAct Cycle {self.cycle_id} ==="]
        trace.append(f"Query: {self.query}")
        trace.append(f"Duration: {self.get_duration():.2f}s")
        trace.append(f"Iterations: {self.iterations}/{self.max_iterations}")
        
        for i in range(self.iterations + 1):
            # Thoughts for this iteration
            iter_thoughts = [t for t in self.thoughts if t.iteration == i]
            for thought in iter_thoughts:
                trace.append(f"  Thought {i}: {thought.content}")
            
            # Actions for this iteration
            iter_actions = [a for a in self.actions if a.iteration == i]
            for action in iter_actions:
                trace.append(f"  Action {i}: {action.tool_name}({json.dumps(action.arguments)})")
                if action.reasoning:
                    trace.append(f"    Reasoning: {action.reasoning}")
                
                # Related observations
                related_obs = [o for o in self.observations 
                             if o.action_id == action.action_id]
                for obs in related_obs:
                    status = "✓" if obs.success else "✗"
                    time_info = f" ({obs.execution_time:.3f}s)" if obs.execution_time > 0 else ""
                    trace.append(f"    Observation {status}{time_info}: {str(obs.content)[:200]}")
        
        if self.final_answer:
            trace.append(f"Final Answer: {self.final_answer[:500]}")
            
        trace.append(f"Success: {self.success}, Confidence: {self.confidence:.2f}")
        
        return "\n".join(trace)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        duration = self.get_duration()
        
        successful_actions = sum(1 for obs in self.observations if obs.success)
        total_actions = len(self.observations)
        
        return {
            "cycle_id": self.cycle_id,
            "iterations": self.iterations,
            "thoughts": len(self.thoughts),
            "actions": len(self.actions),
            "observations": len(self.observations),
            "duration_seconds": duration,
            "success_rate": successful_actions / max(total_actions, 1),
            "final_success": self.success,
            "confidence": self.confidence
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "thoughts": [
                {
                    "thought_id": t.thought_id,
                    "content": t.content,
                    "iteration": t.iteration,
                    "confidence": t.confidence,
                    "thought_type": t.thought_type,
                    "timestamp": t.timestamp.isoformat()
                } for t in self.thoughts
            ],
            "actions": [
                {
                    "action_id": a.action_id,
                    "tool_name": a.tool_name,
                    "arguments": a.arguments,
                    "reasoning": a.reasoning,
                    "iteration": a.iteration,
                    "timestamp": a.timestamp.isoformat()
                } for a in self.actions
            ],
            "observations": [
                {
                    "observation_id": o.observation_id,
                    "action_id": o.action_id,
                    "content": o.content,
                    "success": o.success,
                    "iteration": o.iteration,
                    "execution_time": o.execution_time,
                    "timestamp": o.timestamp.isoformat()
                } for o in self.observations
            ],
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "success": self.success,
            "confidence": self.confidence,
            "error": self.error,
            "chat_message_ids": self.chat_message_ids,
            "extracted_fact_ids": self.extracted_fact_ids,
            "parent_cycle_id": self.parent_cycle_id,
            "child_cycle_ids": self.child_cycle_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReActCycle':
        """Deserialize from dictionary."""
        cycle = cls(cycle_id=data["cycle_id"])
        cycle.query = data["query"]
        cycle.start_time = datetime.fromisoformat(data["start_time"])
        cycle.end_time = datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        
        # Restore thoughts
        cycle.thoughts = [
            Thought(
                thought_id=t["thought_id"],
                content=t["content"],
                iteration=t["iteration"],
                confidence=t["confidence"],
                thought_type=t["thought_type"],
                timestamp=datetime.fromisoformat(t["timestamp"])
            ) for t in data["thoughts"]
        ]
        
        # Restore actions
        cycle.actions = [
            Action(
                action_id=a["action_id"],
                tool_name=a["tool_name"],
                arguments=a["arguments"],
                reasoning=a["reasoning"],
                iteration=a["iteration"],
                timestamp=datetime.fromisoformat(a["timestamp"])
            ) for a in data["actions"]
        ]
        
        # Restore observations
        cycle.observations = [
            Observation(
                observation_id=o["observation_id"],
                action_id=o["action_id"],
                content=o["content"],
                success=o["success"],
                iteration=o["iteration"],
                execution_time=o["execution_time"],
                timestamp=datetime.fromisoformat(o["timestamp"])
            ) for o in data["observations"]
        ]
        
        cycle.final_answer = data["final_answer"]
        cycle.iterations = data["iterations"]
        cycle.max_iterations = data["max_iterations"]
        cycle.success = data["success"]
        cycle.confidence = data.get("confidence", 1.0)
        cycle.error = data["error"]
        cycle.chat_message_ids = data["chat_message_ids"]
        cycle.extracted_fact_ids = data["extracted_fact_ids"]
        cycle.parent_cycle_id = data.get("parent_cycle_id")
        cycle.child_cycle_ids = data.get("child_cycle_ids", [])
        
        return cycle


@dataclass
class Fact:
    """Enhanced fact representation with improved metadata."""
    
    fact_id: str = field(default_factory=lambda: f"fact_{uuid.uuid4().hex[:8]}")
    subject: str = ""
    predicate: str = ""
    object: Any = None
    confidence: float = 1.0
    source_type: MemoryComponent = MemoryComponent.CHAT_HISTORY
    source_id: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 1.0  # Importance weighting for consolidation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.subject} --[{self.predicate}]--> {self.object}"
    
    def access(self):
        """Track fact access for importance weighting."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        # Increase importance based on recency and frequency
        self.importance = min(2.0, self.importance + 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "extracted_at": self.extracted_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        return cls(
            fact_id=data["fact_id"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=data["confidence"],
            source_type=MemoryComponent(data["source_type"]),
            source_id=data["source_id"],
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            last_accessed=datetime.fromisoformat(data.get("last_accessed", data["extracted_at"])),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 1.0),
            metadata=data.get("metadata", {})
        )


class KnowledgeGraph:
    """Enhanced knowledge graph with improved querying and relationship tracking."""
    
    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.subject_index: Dict[str, List[str]] = defaultdict(list)  # subject -> fact_ids
        self.predicate_index: Dict[str, List[str]] = defaultdict(list)  # predicate -> fact_ids
        self.object_index: Dict[str, List[str]] = defaultdict(list)  # object -> fact_ids
        self.importance_sorted: List[str] = []  # fact_ids sorted by importance
    
    def add_fact(self, fact: Fact):
        """Add a fact to the knowledge graph."""
        self.facts[fact.fact_id] = fact
        
        # Update indices
        subject_norm = self._normalize(fact.subject)
        predicate_norm = self._normalize(fact.predicate)
        
        self.subject_index[subject_norm].append(fact.fact_id)
        self.predicate_index[predicate_norm].append(fact.fact_id)
        
        # Index objects if they're strings
        if isinstance(fact.object, str):
            object_norm = self._normalize(fact.object)
            self.object_index[object_norm].append(fact.fact_id)
        
        # Update importance ranking
        self._update_importance_ranking()
        
        logger.debug(f"Added fact: {fact}")
    
    def query_subject(self, subject: str) -> List[Fact]:
        """Find all facts about a subject."""
        fact_ids = self.subject_index.get(self._normalize(subject), [])
        facts = [self.facts[fid] for fid in fact_ids if fid in self.facts]
        # Mark facts as accessed
        for fact in facts:
            fact.access()
        return facts
    
    def query_predicate(self, predicate: str) -> List[Fact]:
        """Find all facts with a predicate."""
        fact_ids = self.predicate_index.get(self._normalize(predicate), [])
        facts = [self.facts[fid] for fid in fact_ids if fid in self.facts]
        for fact in facts:
            fact.access()
        return facts
    
    def query_object(self, obj: str) -> List[Fact]:
        """Find all facts with an object."""
        fact_ids = self.object_index.get(self._normalize(obj), [])
        facts = [self.facts[fid] for fid in fact_ids if fid in self.facts]
        for fact in facts:
            fact.access()
        return facts
    
    def query_advanced(self, subject: Optional[str] = None,
                      predicate: Optional[str] = None,
                      obj: Optional[str] = None,
                      min_confidence: float = 0.0,
                      min_importance: float = 0.0) -> List[Fact]:
        """Advanced query with filtering."""
        results = list(self.facts.values())
        
        if subject:
            subject_norm = self._normalize(subject)
            results = [f for f in results if self._normalize(f.subject) == subject_norm]
        
        if predicate:
            predicate_norm = self._normalize(predicate)
            results = [f for f in results if self._normalize(f.predicate) == predicate_norm]
        
        if obj:
            obj_norm = self._normalize(str(obj))
            results = [f for f in results if isinstance(f.object, str) and 
                      self._normalize(f.object) == obj_norm]
        
        # Apply filters
        results = [f for f in results if f.confidence >= min_confidence]
        results = [f for f in results if f.importance >= min_importance]
        
        # Mark as accessed and sort by importance
        for fact in results:
            fact.access()
        
        return sorted(results, key=lambda f: f.importance, reverse=True)
    
    def get_related_entities(self, entity: str, max_depth: int = 2,
                           min_importance: float = 0.5) -> Set[str]:
        """Find entities related to the given entity."""
        entity_norm = self._normalize(entity)
        visited = set()
        to_visit = {entity_norm}
        
        for _ in range(max_depth):
            new_entities = set()
            
            for current in to_visit:
                if current in visited:
                    continue
                visited.add(current)
                
                # Find connections through subject
                for fact_id in self.subject_index.get(current, []):
                    if fact_id in self.facts:
                        fact = self.facts[fact_id]
                        if fact.importance >= min_importance:
                            if isinstance(fact.object, str):
                                new_entities.add(self._normalize(fact.object))
                
                # Find connections through object
                for fact_id in self.object_index.get(current, []):
                    if fact_id in self.facts:
                        fact = self.facts[fact_id]
                        if fact.importance >= min_importance:
                            new_entities.add(self._normalize(fact.subject))
            
            to_visit = new_entities - visited
        
        return visited - {entity_norm}
    
    def get_most_important_facts(self, limit: int = 10) -> List[Fact]:
        """Get the most important facts."""
        sorted_facts = sorted(self.facts.values(), 
                            key=lambda f: f.importance, reverse=True)
        return sorted_facts[:limit]
    
    def _normalize(self, text: str) -> str:
        """Normalize text for consistent indexing."""
        return text.lower().strip()
    
    def _update_importance_ranking(self):
        """Update importance-based ranking."""
        self.importance_sorted = sorted(
            self.facts.keys(),
            key=lambda fid: self.facts[fid].importance,
            reverse=True
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self.facts:
            return {
                "total_facts": 0,
                "unique_subjects": 0,
                "unique_predicates": 0,
                "unique_objects": 0,
                "average_confidence": 0.0,
                "average_importance": 0.0
            }
        
        return {
            "total_facts": len(self.facts),
            "unique_subjects": len(self.subject_index),
            "unique_predicates": len(self.predicate_index),
            "unique_objects": len(self.object_index),
            "average_confidence": sum(f.confidence for f in self.facts.values()) / len(self.facts),
            "average_importance": sum(f.importance for f in self.facts.values()) / len(self.facts),
            "most_accessed": max(self.facts.values(), key=lambda f: f.access_count).fact_id if self.facts else None
        }


class HierarchicalMemory:
    """
    SOTA Hierarchical Memory System with enhanced features for AbstractLLM.
    
    Comprehensive memory system that integrates:
    - Working memory (recent context)
    - Episodic memory (consolidated experiences) 
    - Semantic memory (knowledge graph)
    - ReAct cycles (reasoning traces)
    - Bidirectional linking between all components
    - Cross-session persistence
    - Advanced context retrieval for LLM prompting
    
    Inspired by A-Mem, RAISE, and MemGPT architectures with enhancements.
    """
    
    def __init__(self,
                 working_memory_size: int = 10,
                 episodic_consolidation_threshold: int = 5,
                 persist_path: Optional[Path] = None,
                 enable_cross_session_persistence: bool = True,
                 session: Optional[Any] = None):
        """
        Initialize the hierarchical memory system.

        Args:
            working_memory_size: Max items in working memory before consolidation
            episodic_consolidation_threshold: When to move items to episodic memory
            persist_path: Path for persistent storage across sessions
            enable_cross_session_persistence: Whether to enable cross-session knowledge
            session: Optional session reference for deterministic mode detection
        """
        # Memory stores
        self.working_memory: List[Dict[str, Any]] = []  # Most recent, active items
        self.episodic_memory: List[Dict[str, Any]] = []  # Consolidated experiences
        self.knowledge_graph = KnowledgeGraph()  # Enhanced semantic memory
        
        # ReAct reasoning system
        self.react_cycles: Dict[str, ReActCycle] = {}
        self.current_cycle: Optional[ReActCycle] = None
        
        # Bidirectional linking system
        self.links: List[MemoryLink] = []
        self.link_index: Dict[str, List[MemoryLink]] = defaultdict(list)
        
        # Chat history with enhanced metadata
        self.chat_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.working_memory_size = working_memory_size
        self.episodic_consolidation_threshold = episodic_consolidation_threshold
        self.persist_path = Path(persist_path) if persist_path else None
        self.enable_cross_session_persistence = enable_cross_session_persistence

        # Store session reference for deterministic mode detection
        self._session = session

        # Session metadata - use deterministic values if in deterministic mode
        if self._is_deterministic_mode():
            # Generate deterministic session ID based on seed
            seed = self._get_current_seed()
            import hashlib
            seed_str = str(seed) if seed is not None else "default"
            self.session_id = hashlib.md5(f"memory_{seed_str}".encode()).hexdigest()[:16]
            # Use fixed timestamp for deterministic generation
            self.session_start = datetime.fromtimestamp(1609459200)  # 2021-01-01 00:00:00 UTC
        else:
            # Use random values for normal operation
            self.session_id = f"session_{uuid.uuid4().hex[:8]}"
            self.session_start = datetime.now()
        self.total_queries = 0
        self.successful_queries = 0
        
        # Memory consolidation tracking
        self.last_consolidation = datetime.now()
        self.consolidation_frequency = timedelta(minutes=30)  # Auto-consolidate every 30 min
        
        # Load persisted memory if available
        if self.enable_cross_session_persistence and self.persist_path:
            self._load_cross_session_knowledge()
        elif self.persist_path and self.persist_path.exists():
            self.load_from_disk()

    def _is_deterministic_mode(self) -> bool:
        """Check if the session is in deterministic mode (seed is set)."""
        if not self._session or not hasattr(self._session, '_provider') or not self._session._provider:
            return False
        try:
            from abstractllm.interface import ModelParameter
            seed = self._session._provider.config_manager.get_param(ModelParameter.SEED)
            return seed is not None
        except:
            return False

    def _get_current_seed(self) -> Optional[int]:
        """Get the current seed value if set."""
        if not self._session or not hasattr(self._session, '_provider') or not self._session._provider:
            return None
        try:
            from abstractllm.interface import ModelParameter
            return self._session._provider.config_manager.get_param(ModelParameter.SEED)
        except:
            return None

    # ========== CORE MEMORY OPERATIONS ==========
    
    def start_react_cycle(self, query: str, max_iterations: int = 10) -> ReActCycle:
        """Start a new ReAct reasoning cycle for a query."""
        # Complete previous cycle if not finished
        if self.current_cycle and not self.current_cycle.end_time:
            logger.warning(f"Previous cycle {self.current_cycle.cycle_id} not completed")
            self.current_cycle.complete("Interrupted by new query", success=False)
        
        # Create new cycle
        cycle = ReActCycle(query=query, max_iterations=max_iterations)

        # Override cycle ID for deterministic generation
        if self._is_deterministic_mode():
            seed = self._get_current_seed()
            import hashlib
            seed_str = str(seed) if seed is not None else "default"
            # Use total_queries for uniqueness within same session
            deterministic_id = hashlib.md5(f"cycle_{seed_str}_{self.total_queries}".encode()).hexdigest()[:8]
            cycle.cycle_id = f"cycle_{deterministic_id}"

        self.current_cycle = cycle
        self.react_cycles[cycle.cycle_id] = cycle
        self.total_queries += 1

        logger.info(f"Started ReAct cycle {cycle.cycle_id} for query: {query[:100]}")
        return cycle
    
    def add_chat_message(self, role: str, content: str,
                        cycle_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a chat message with enhanced tracking and fact extraction."""
        # Generate deterministic message ID if in deterministic mode
        if self._is_deterministic_mode():
            seed = self._get_current_seed()
            import hashlib
            seed_str = str(seed) if seed is not None else "default"
            # Use message count for uniqueness
            message_count = len(self.chat_history)
            deterministic_id = hashlib.md5(f"msg_{seed_str}_{message_count}".encode()).hexdigest()[:8]
            message_id = f"msg_{deterministic_id}"
        else:
            message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        # Generate deterministic timestamp if in deterministic mode
        if self._is_deterministic_mode():
            # Use session start time + message count for deterministic timestamps
            message_count = len(self.chat_history)
            deterministic_timestamp = self.session_start + timedelta(seconds=message_count)
            timestamp = deterministic_timestamp.isoformat()
        else:
            timestamp = datetime.now().isoformat()

        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "cycle_id": cycle_id,
            "session_id": self.session_id,
            "fact_ids": [],  # Will be populated by fact extraction
            "importance": self._calculate_message_importance(role, content),
            "metadata": metadata or {}
        }
        
        # Add to storage
        self.chat_history.append(message)
        self.working_memory.append(message)
        
        # Create bidirectional link to ReAct cycle
        if cycle_id and cycle_id in self.react_cycles:
            self.add_link(
                MemoryComponent.CHAT_HISTORY, message_id,
                MemoryComponent.REACT_CYCLE, cycle_id,
                "generated_by"
            )
            self.react_cycles[cycle_id].chat_message_ids.append(message_id)
        
        # Extract facts from message content
        facts = self.extract_facts(content, MemoryComponent.CHAT_HISTORY, message_id)
        for fact in facts:
            message["fact_ids"].append(fact.fact_id)
            
            # Link facts to cycle if available
            if cycle_id and cycle_id in self.react_cycles:
                self.react_cycles[cycle_id].extracted_fact_ids.append(fact.fact_id)
        
        # Auto-consolidate if working memory is full
        if len(self.working_memory) > self.working_memory_size:
            self._consolidate_working_memory()
        
        # Check for time-based consolidation
        self._check_periodic_consolidation()
        
        return message_id
    
    def extract_facts(self, content: str, source_type: MemoryComponent, 
                     source_id: str) -> List[Fact]:
        """Extract facts using enhanced patterns and NLP techniques."""
        facts = []
        
        # Enhanced fact extraction patterns
        patterns = [
            # Basic relationships
            (r"(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)*)", "is_a"),
            (r"(\w+(?:\s+\w+)*)\s+has\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)*)", "has"),
            (r"(\w+(?:\s+\w+)*)\s+can\s+(\w+(?:\s+\w+)*)", "can_do"),
            (r"(\w+(?:\s+\w+)*)\s+cannot\s+(\w+(?:\s+\w+)*)", "cannot_do"),
            (r"(\w+(?:\s+\w+)*)\s+needs\s+(\w+(?:\s+\w+)*)", "needs"),
            (r"(\w+(?:\s+\w+)*)\s+requires\s+(\w+(?:\s+\w+)*)", "requires"),
            (r"(\w+(?:\s+\w+)*)\s+supports\s+(\w+(?:\s+\w+)*)", "supports"),
            (r"(\w+(?:\s+\w+)*)\s+works\s+with\s+(\w+(?:\s+\w+)*)", "works_with"),
            (r"(\w+(?:\s+\w+)*)\s+depends\s+on\s+(\w+(?:\s+\w+)*)", "depends_on"),
            (r"(\w+(?:\s+\w+)*)\s+uses\s+(\w+(?:\s+\w+)*)", "uses"),
            (r"(\w+(?:\s+\w+)*)\s+provides\s+(\w+(?:\s+\w+)*)", "provides"),
            (r"(\w+(?:\s+\w+)*)\s+implements\s+(\w+(?:\s+\w+)*)", "implements"),
            (r"(\w+(?:\s+\w+)*)\s+extends\s+(\w+(?:\s+\w+)*)", "extends"),
            (r"(\w+(?:\s+\w+)*)\s+inherits\s+from\s+(\w+(?:\s+\w+)*)", "inherits_from"),
            
            # Properties and attributes
            (r"(\w+(?:\s+\w+)*)\s+(?:is|are)\s+(\w+(?:\s+\w+)*)", "has_property"),
            (r"(\w+(?:\s+\w+)*)\s+contains\s+(\w+(?:\s+\w+)*)", "contains"),
            (r"(\w+(?:\s+\w+)*)\s+includes\s+(\w+(?:\s+\w+)*)", "includes"),
            (r"(\w+(?:\s+\w+)*)\s+consists\s+of\s+(\w+(?:\s+\w+)*)", "consists_of"),
            
            # Actions and capabilities
            (r"(\w+(?:\s+\w+)*)\s+enables\s+(\w+(?:\s+\w+)*)", "enables"),
            (r"(\w+(?:\s+\w+)*)\s+allows\s+(\w+(?:\s+\w+)*)", "allows"),
            (r"(\w+(?:\s+\w+)*)\s+prevents\s+(\w+(?:\s+\w+)*)", "prevents"),
            (r"(\w+(?:\s+\w+)*)\s+causes\s+(\w+(?:\s+\w+)*)", "causes"),
            (r"(\w+(?:\s+\w+)*)\s+results\s+in\s+(\w+(?:\s+\w+)*)", "results_in"),
            
            # Location and containment
            (r"(\w+(?:\s+\w+)*)\s+is\s+(?:located\s+)?in\s+(\w+(?:\s+\w+)*)", "located_in"),
            (r"(\w+(?:\s+\w+)*)\s+is\s+part\s+of\s+(\w+(?:\s+\w+)*)", "part_of"),
            (r"(\w+(?:\s+\w+)*)\s+belongs\s+to\s+(\w+(?:\s+\w+)*)", "belongs_to"),
            
            # Comparisons and relationships
            (r"(\w+(?:\s+\w+)*)\s+is\s+(?:similar\s+to|like)\s+(\w+(?:\s+\w+)*)", "similar_to"),
            (r"(\w+(?:\s+\w+)*)\s+is\s+different\s+from\s+(\w+(?:\s+\w+)*)", "different_from"),
            (r"(\w+(?:\s+\w+)*)\s+is\s+better\s+than\s+(\w+(?:\s+\w+)*)", "better_than"),
            (r"(\w+(?:\s+\w+)*)\s+is\s+worse\s+than\s+(\w+(?:\s+\w+)*)", "worse_than"),
        ]
        
        for pattern, predicate in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2 and match[0].strip() and match[1].strip():
                    # Calculate confidence based on pattern type and context
                    confidence = self._calculate_fact_confidence(pattern, match, content)
                    
                    fact = Fact(
                        subject=match[0].strip(),
                        predicate=predicate,
                        object=match[1].strip(),
                        confidence=confidence,
                        source_type=source_type,
                        source_id=source_id,
                        metadata={
                            "extraction_method": "pattern_matching",
                            "pattern": pattern,
                            "context_length": len(content)
                        }
                    )
                    
                    # Add to knowledge graph
                    self.knowledge_graph.add_fact(fact)
                    facts.append(fact)
                    
                    # Create bidirectional link
                    self.add_link(
                        source_type, source_id,
                        MemoryComponent.KNOWLEDGE, fact.fact_id,
                        "extracted_fact"
                    )
        
        logger.debug(f"Extracted {len(facts)} facts from {source_type.value}:{source_id}")
        return facts
    
    def query_memory(self, query: str, include_links: bool = True,
                    max_results: int = 20, include_context: bool = True) -> Dict[str, Any]:
        """Advanced memory query with enhanced results and context."""
        query_lower = query.lower()
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "working_memory": [],
            "episodic_memory": [],
            "facts": [],
            "react_cycles": [],
            "related_entities": set(),
            "links": [],
            "context_score": 0.0
        }
        
        # Search working memory (most recent)
        for item in self.working_memory:
            score = self._calculate_relevance_score(query_lower, item)
            if score > 0.1:  # Relevance threshold
                item_copy = item.copy()
                item_copy["relevance_score"] = score
                results["working_memory"].append(item_copy)
        
        # Search episodic memory
        for item in self.episodic_memory:
            score = self._calculate_relevance_score(query_lower, item)
            if score > 0.1:
                item_copy = item.copy()
                item_copy["relevance_score"] = score
                results["episodic_memory"].append(item_copy)
        
        # Search knowledge graph with advanced querying
        relevant_facts = self.knowledge_graph.query_advanced(
            subject=None, predicate=None, obj=None,
            min_confidence=0.3, min_importance=0.3
        )
        
        for fact in relevant_facts:
            score = self._calculate_fact_relevance(query_lower, fact)
            if score > 0.1:
                fact_dict = fact.to_dict()
                fact_dict["relevance_score"] = score
                results["facts"].append(fact_dict)
                
                # Add related entities for further exploration
                results["related_entities"].add(fact.subject)
                if isinstance(fact.object, str):
                    results["related_entities"].add(fact.object)
        
        # Search ReAct cycles
        for cycle_id, cycle in self.react_cycles.items():
            score = self._calculate_relevance_score(query_lower, {"content": cycle.query})
            if score > 0.1:
                cycle_summary = {
                    "cycle_id": cycle_id,
                    "query": cycle.query,
                    "success": cycle.success,
                    "iterations": cycle.iterations,
                    "confidence": cycle.confidence,
                    "duration": cycle.get_duration(),
                    "relevance_score": score
                }
                
                # Add final answer if successful
                if cycle.final_answer:
                    cycle_summary["final_answer"] = cycle.final_answer[:200]
                
                results["react_cycles"].append(cycle_summary)
        
        # Sort results by relevance
        results["working_memory"].sort(key=lambda x: x["relevance_score"], reverse=True)
        results["episodic_memory"].sort(key=lambda x: x["relevance_score"], reverse=True)
        results["facts"].sort(key=lambda x: x["relevance_score"], reverse=True)
        results["react_cycles"].sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Limit results
        results["working_memory"] = results["working_memory"][:max_results//4]
        results["episodic_memory"] = results["episodic_memory"][:max_results//4]
        results["facts"] = results["facts"][:max_results//2]
        results["react_cycles"] = results["react_cycles"][:max_results//4]
        
        # Convert related entities set to list
        results["related_entities"] = list(results["related_entities"])
        
        # Include relevant links if requested
        if include_links:
            for fact in results["facts"]:
                links = self.get_links(MemoryComponent.KNOWLEDGE, fact["fact_id"])
                for link in links[:3]:  # Limit links per fact
                    results["links"].append({
                        "from": f"{link.source_type.value}:{link.source_id}",
                        "to": f"{link.target_type.value}:{link.target_id}",
                        "relationship": link.relationship,
                        "strength": link.strength,
                        "access_count": link.accessed_count
                    })
        
        # Calculate overall context score
        results["context_score"] = self._calculate_context_score(results)
        
        return results
    
    def get_context_for_query(self, query: str, max_tokens: int = 2000,
                             include_reasoning: bool = True) -> str:
        """Get comprehensive context for LLM prompting with token management."""
        context_parts = []
        estimated_tokens = 0
        
        # Helper function to estimate tokens (rough approximation)
        def estimate_tokens(text: str) -> int:
            return len(text.split()) * 1.3  # Rough token estimation
        
        # Add session context (skip for deterministic generation to avoid random IDs)
        if not self._is_deterministic_mode():
            # Only add session info if not in deterministic mode
            session_info = f"Session: {self.session_id} (Started: {self.session_start.strftime('%Y-%m-%d %H:%M')})"
            context_parts.append(session_info)
            estimated_tokens += estimate_tokens(session_info)
        
        # Add recent working memory (most relevant)
        if self.working_memory:
            working_section = ["\\n--- Recent Context ---"]
            for item in self.working_memory[-3:]:  # Last 3 items
                if "content" in item and estimated_tokens < max_tokens * 0.3:
                    content_preview = item["content"][:150]
                    working_section.append(f"- [{item.get('role', 'unknown')}] {content_preview}")
                    estimated_tokens += estimate_tokens(content_preview)
            
            if len(working_section) > 1:
                context_parts.extend(working_section)
        
        # Add current ReAct cycle reasoning if available
        if include_reasoning and self.current_cycle and estimated_tokens < max_tokens * 0.4:
            # Use generic label in deterministic mode instead of random cycle ID
            if self._is_deterministic_mode():
                reasoning_section = [f"\\n--- Current Reasoning ---"]
            else:
                reasoning_section = [f"\\n--- Current Reasoning (Cycle {self.current_cycle.cycle_id}) ---"]
            
            # Add recent thoughts
            if self.current_cycle.thoughts:
                recent_thoughts = self.current_cycle.thoughts[-2:]  # Last 2 thoughts
                for thought in recent_thoughts:
                    thought_text = f"Thought: {thought.content[:100]}"
                    if estimated_tokens + estimate_tokens(thought_text) < max_tokens * 0.4:
                        reasoning_section.append(thought_text)
                        estimated_tokens += estimate_tokens(thought_text)
            
            # Add recent actions/observations
            if self.current_cycle.observations:
                recent_obs = self.current_cycle.observations[-2:]  # Last 2 observations
                for obs in recent_obs:
                    obs_text = f"Action Result: {str(obs.content)[:100]}"
                    if estimated_tokens + estimate_tokens(obs_text) < max_tokens * 0.4:
                        reasoning_section.append(obs_text)
                        estimated_tokens += estimate_tokens(obs_text)
            
            if len(reasoning_section) > 1:
                context_parts.extend(reasoning_section)
        
        # Query relevant facts from knowledge graph
        query_results = self.query_memory(query, include_links=False)
        if query_results["facts"] and estimated_tokens < max_tokens * 0.6:
            facts_section = ["\\n--- Relevant Knowledge ---"]
            
            # Add most relevant facts
            for fact_dict in query_results["facts"][:5]:
                if estimated_tokens < max_tokens * 0.6:
                    fact = Fact.from_dict(fact_dict)
                    fact_text = f"- {fact} (confidence: {fact.confidence:.2f})"
                    if estimated_tokens + estimate_tokens(fact_text) < max_tokens * 0.6:
                        facts_section.append(fact_text)
                        estimated_tokens += estimate_tokens(fact_text)
            
            if len(facts_section) > 1:
                context_parts.extend(facts_section)
        
        # Add previous successful approaches for similar queries
        if estimated_tokens < max_tokens * 0.8:
            similar_cycles = []
            for cycle_id, cycle in self.react_cycles.items():
                if (cycle.success and cycle.final_answer and
                    query.lower() in cycle.query.lower()):
                    similar_cycles.append(cycle)
            
            if similar_cycles:
                approaches_section = ["\\n--- Previous Successful Approaches ---"]
                
                for cycle in similar_cycles[:2]:  # Limit to 2 similar cycles
                    if estimated_tokens < max_tokens * 0.8:
                        approach_text = f"- Query: {cycle.query[:80]}\\n  Answer: {cycle.final_answer[:120]}"
                        if estimated_tokens + estimate_tokens(approach_text) < max_tokens * 0.8:
                            approaches_section.append(approach_text)
                            estimated_tokens += estimate_tokens(approach_text)
                
                if len(approaches_section) > 1:
                    context_parts.extend(approaches_section)
        
        # Add memory statistics summary if space allows
        if estimated_tokens < max_tokens * 0.9:
            stats = self.get_statistics()
            stats_text = f"\\n--- Memory Stats ---\\nFacts: {stats['knowledge_graph']['total_facts']}, Cycles: {stats['total_react_cycles']}, Success Rate: {stats['query_success_rate']:.1%}"
            if estimated_tokens + estimate_tokens(stats_text) < max_tokens:
                context_parts.append(stats_text)
        
        final_context = "\\n".join(context_parts)
        
        # Truncate if still too long
        final_tokens = estimate_tokens(final_context)
        if final_tokens > max_tokens:
            # Proportional truncation
            truncation_ratio = max_tokens / final_tokens
            final_context = final_context[:int(len(final_context) * truncation_ratio)]
            final_context += "\\n... (truncated for token limit)"
        
        return final_context
    
    # ========== MEMORY CONSOLIDATION ==========
    
    def _consolidate_working_memory(self):
        """Enhanced memory consolidation with importance weighting."""
        if len(self.working_memory) <= self.episodic_consolidation_threshold:
            return
        
        # Sort by importance and age (older + less important items first)
        items_with_scores = []
        for item in self.working_memory:
            age_score = (datetime.now() - datetime.fromisoformat(item["timestamp"])).total_seconds() / 3600  # Hours
            importance = item.get("importance", 1.0)
            consolidation_score = age_score / max(importance, 0.1)  # Higher score = more likely to consolidate
            items_with_scores.append((consolidation_score, item))
        
        items_with_scores.sort(key=lambda x: x[0], reverse=True)  # Highest consolidation score first
        
        # Move items to episodic memory
        to_consolidate = [item for _, item in items_with_scores[:self.episodic_consolidation_threshold]]
        
        for item in to_consolidate:
            # Add consolidation metadata
            item["consolidated_at"] = datetime.now().isoformat()
            item["session_id"] = self.session_id
            item["consolidation_reason"] = "working_memory_overflow"
            
            # Extract additional facts before moving to episodic
            if "content" in item:
                additional_facts = self.extract_facts(
                    item["content"],
                    MemoryComponent.EPISODIC,
                    item.get("id", f"episodic_{uuid.uuid4().hex[:8]}")
                )
                
                # Update fact links to point to episodic memory
                for fact in additional_facts:
                    self.add_link(
                        MemoryComponent.EPISODIC, item["id"],
                        MemoryComponent.KNOWLEDGE, fact.fact_id,
                        "consolidation_extraction"
                    )
            
            self.episodic_memory.append(item)
        
        # Remove consolidated items from working memory
        remaining_items = [item for _, item in items_with_scores[self.episodic_consolidation_threshold:]]
        self.working_memory = remaining_items
        
        self.last_consolidation = datetime.now()
        
        logger.info(f"Consolidated {len(to_consolidate)} items to episodic memory")
        
        # Trigger semantic consolidation if episodic memory is getting large
        if len(self.episodic_memory) > 50:
            self._consolidate_semantic_knowledge()
    
    def _consolidate_semantic_knowledge(self):
        """Consolidate related facts and strengthen important knowledge."""
        logger.info("Starting semantic knowledge consolidation")
        
        # Group facts by related entities
        entity_groups = defaultdict(list)
        for fact_id, fact in self.knowledge_graph.facts.items():
            entity_groups[fact.subject].append(fact)
            if isinstance(fact.object, str):
                entity_groups[fact.object].append(fact)
        
        # Consolidate facts for each entity
        consolidated_count = 0
        for entity, related_facts in entity_groups.items():
            if len(related_facts) > 3:  # Only consolidate if multiple facts exist
                # Strengthen important facts
                for fact in related_facts:
                    if fact.access_count > 2:
                        fact.importance = min(2.0, fact.importance + 0.2)
                        consolidated_count += 1
        
        # Update knowledge graph importance ranking
        self.knowledge_graph._update_importance_ranking()
        
        logger.info(f"Consolidated {consolidated_count} semantic facts")
    
    def _check_periodic_consolidation(self):
        """Check if periodic consolidation is needed."""
        if datetime.now() - self.last_consolidation > self.consolidation_frequency:
            self._consolidate_working_memory()
    
    # ========== LINKING SYSTEM ==========
    
    def add_link(self, source_type: MemoryComponent, source_id: str,
                target_type: MemoryComponent, target_id: str,
                relationship: str, strength: float = 1.0,
                metadata: Optional[Dict[str, Any]] = None):
        """Add enhanced bidirectional link between memory components."""
        # Create forward link
        forward_link = MemoryLink(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relationship=relationship,
            strength=strength,
            metadata=metadata or {}
        )
        
        # Create reverse link
        reverse_link = forward_link.reverse()
        
        # Add both links
        self.links.append(forward_link)
        self.links.append(reverse_link)
        
        # Index for fast lookup
        source_key = f"{source_type.value}:{source_id}"
        target_key = f"{target_type.value}:{target_id}"
        
        self.link_index[source_key].append(forward_link)
        self.link_index[target_key].append(reverse_link)
    
    def get_links(self, component_type: MemoryComponent, 
                 component_id: str, relationship: Optional[str] = None) -> List[MemoryLink]:
        """Get links for a component with optional relationship filtering."""
        key = f"{component_type.value}:{component_id}"
        links = self.link_index.get(key, [])
        
        if relationship:
            links = [link for link in links if link.relationship == relationship]
        
        # Sort by strength and access count
        links.sort(key=lambda l: (l.strength, l.accessed_count), reverse=True)
        
        return links
    
    def strengthen_link(self, source_type: MemoryComponent, source_id: str,
                       target_type: MemoryComponent, target_id: str,
                       delta: float = 0.1):
        """Strengthen links between components based on usage."""
        source_key = f"{source_type.value}:{source_id}"
        target_key = f"{target_type.value}:{target_id}"
        
        # Find and strengthen forward link
        for link in self.link_index.get(source_key, []):
            if (link.target_type == target_type and link.target_id == target_id):
                link.strengthen(delta)
        
        # Find and strengthen reverse link
        for link in self.link_index.get(target_key, []):
            if (link.target_type == source_type and link.target_id == source_id):
                link.strengthen(delta)
    
    # ========== PERSISTENCE SYSTEM ==========
    
    def save_to_disk(self, include_session_data: bool = True):
        """Enhanced persistence with compression and cross-session support."""
        if not self.persist_path:
            return
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare comprehensive serializable data
        memory_data = {
            "version": "2.0",
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_metadata": {
                "total_queries": self.total_queries,
                "successful_queries": self.successful_queries,
                "enable_cross_session_persistence": self.enable_cross_session_persistence
            },
            "working_memory": self.working_memory if include_session_data else [],
            "episodic_memory": self.episodic_memory,
            "semantic_memory": {k: v.to_dict() for k, v in self.knowledge_graph.facts.items()},
            "react_cycles": {k: v.to_dict() for k, v in self.react_cycles.items()},
            "chat_history": self.chat_history if include_session_data else [],
            "links": [
                {
                    "source_type": link.source_type.value,
                    "source_id": link.source_id,
                    "target_type": link.target_type.value,
                    "target_id": link.target_id,
                    "relationship": link.relationship,
                    "strength": link.strength,
                    "metadata": link.metadata,
                    "created_at": link.created_at.isoformat(),
                    "accessed_count": link.accessed_count
                }
                for link in self.links
            ],
            "knowledge_graph_stats": self.knowledge_graph.get_statistics(),
            "configuration": {
                "working_memory_size": self.working_memory_size,
                "episodic_consolidation_threshold": self.episodic_consolidation_threshold
            }
        }
        
        # Save session-specific file
        memory_file = self.persist_path / f"{self.session_id}.json"
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        # Save cross-session knowledge if enabled
        if self.enable_cross_session_persistence:
            self._save_cross_session_knowledge()
        
        logger.info(f"Saved memory to {memory_file}")
    
    def load_from_disk(self, session_id: Optional[str] = None):
        """Enhanced loading with version compatibility and error recovery."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            # Find session file to load
            if session_id:
                memory_file = self.persist_path / f"{session_id}.json"
            else:
                # Find most recent session file
                session_files = list(self.persist_path.glob("session_*.json"))
                if not session_files:
                    return
                memory_file = max(session_files, key=lambda f: f.stat().st_mtime)
            
            if not memory_file.exists():
                logger.warning(f"Memory file {memory_file} not found")
                return
            
            with open(memory_file, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
            
            # Version compatibility check
            version = memory_data.get("version", "1.0")
            if version != "2.0":
                logger.warning(f"Loading memory version {version}, some features may not work correctly")
            
            # Restore core memory data
            self.session_id = memory_data["session_id"]
            self.session_start = datetime.fromisoformat(memory_data["session_start"])
            
            # Restore session metadata
            session_meta = memory_data.get("session_metadata", {})
            self.total_queries = session_meta.get("total_queries", 0)
            self.successful_queries = session_meta.get("successful_queries", 0)
            
            # Restore memory stores
            self.working_memory = memory_data.get("working_memory", [])
            self.episodic_memory = memory_data.get("episodic_memory", [])
            
            # Restore knowledge graph
            self.knowledge_graph = KnowledgeGraph()
            for fact_id, fact_data in memory_data.get("semantic_memory", {}).items():
                fact = Fact.from_dict(fact_data)
                self.knowledge_graph.facts[fact_id] = fact
                
                # Rebuild indices
                subject_norm = self.knowledge_graph._normalize(fact.subject)
                predicate_norm = self.knowledge_graph._normalize(fact.predicate)
                
                self.knowledge_graph.subject_index[subject_norm].append(fact_id)
                self.knowledge_graph.predicate_index[predicate_norm].append(fact_id)
                
                if isinstance(fact.object, str):
                    object_norm = self.knowledge_graph._normalize(fact.object)
                    self.knowledge_graph.object_index[object_norm].append(fact_id)
            
            # Update importance ranking
            self.knowledge_graph._update_importance_ranking()
            
            # Restore ReAct cycles
            self.react_cycles = {}
            for cycle_id, cycle_data in memory_data.get("react_cycles", {}).items():
                try:
                    self.react_cycles[cycle_id] = ReActCycle.from_dict(cycle_data)
                except Exception as e:
                    logger.error(f"Failed to restore ReAct cycle {cycle_id}: {e}")
            
            self.chat_history = memory_data.get("chat_history", [])
            
            # Restore links
            self.links = []
            self.link_index = defaultdict(list)
            
            for link_data in memory_data.get("links", []):
                try:
                    link = MemoryLink(
                        source_type=MemoryComponent(link_data["source_type"]),
                        source_id=link_data["source_id"],
                        target_type=MemoryComponent(link_data["target_type"]),
                        target_id=link_data["target_id"],
                        relationship=link_data["relationship"],
                        strength=link_data.get("strength", 1.0),
                        metadata=link_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(link_data["created_at"]),
                        accessed_count=link_data.get("accessed_count", 0)
                    )
                    self.links.append(link)
                    
                    # Rebuild index
                    source_key = f"{link.source_type.value}:{link.source_id}"
                    self.link_index[source_key].append(link)
                except Exception as e:
                    logger.error(f"Failed to restore link: {e}")
            
            # Load cross-session knowledge if enabled
            if self.enable_cross_session_persistence:
                self._load_cross_session_knowledge()
            
            logger.info(f"Loaded memory from {memory_file}")
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            # Initialize empty memory state on failure
            self._initialize_empty_state()
    
    def _save_cross_session_knowledge(self):
        """Save knowledge that persists across sessions."""
        if not self.persist_path:
            return
        
        # Prepare cross-session knowledge (high-confidence, important facts)
        cross_session_facts = {}
        for fact_id, fact in self.knowledge_graph.facts.items():
            if fact.confidence >= 0.7 and fact.importance >= 1.2:
                cross_session_facts[fact_id] = fact.to_dict()
        
        cross_session_data = {
            "version": "2.0",
            "last_updated": datetime.now().isoformat(),
            "facts": cross_session_facts,
            "statistics": {
                "total_sessions": len(list(self.persist_path.glob("session_*.json"))),
                "total_cross_session_facts": len(cross_session_facts)
            }
        }
        
        cross_session_file = self.persist_path / "cross_session_knowledge.json"
        with open(cross_session_file, "w", encoding="utf-8") as f:
            json.dump(cross_session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(cross_session_facts)} cross-session facts")
    
    def _load_cross_session_knowledge(self):
        """Load knowledge from previous sessions."""
        if not self.persist_path:
            return
        
        cross_session_file = self.persist_path / "cross_session_knowledge.json"
        if not cross_session_file.exists():
            return
        
        try:
            with open(cross_session_file, "r", encoding="utf-8") as f:
                cross_session_data = json.load(f)
            
            # Load cross-session facts into knowledge graph
            loaded_count = 0
            for fact_id, fact_data in cross_session_data.get("facts", {}).items():
                fact = Fact.from_dict(fact_data)
                # Mark as cross-session knowledge
                fact.metadata["cross_session"] = True
                self.knowledge_graph.add_fact(fact)
                loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} cross-session knowledge facts")
            
        except Exception as e:
            logger.error(f"Failed to load cross-session knowledge: {e}")
    
    def _initialize_empty_state(self):
        """Initialize memory with empty state."""
        self.working_memory = []
        self.episodic_memory = []
        self.knowledge_graph = KnowledgeGraph()
        self.react_cycles = {}
        self.current_cycle = None
        self.links = []
        self.link_index = defaultdict(list)
        self.chat_history = []
    
    # ========== UTILITY METHODS ==========
    
    def _calculate_message_importance(self, role: str, content: str) -> float:
        """Calculate importance score for a message."""
        base_importance = 1.0
        
        # Role-based weighting
        role_weights = {
            "system": 0.8,
            "user": 1.2,
            "assistant": 1.0,
            "tool": 0.9
        }
        importance = base_importance * role_weights.get(role, 1.0)
        
        # Content-based adjustments
        content_lower = content.lower()
        
        # Questions get higher importance
        if "?" in content:
            importance *= 1.3
        
        # Commands and requests
        command_words = ["create", "make", "build", "implement", "fix", "solve", "help"]
        if any(word in content_lower for word in command_words):
            importance *= 1.4
        
        # Error messages and problems
        if any(word in content_lower for word in ["error", "problem", "issue", "failed", "exception"]):
            importance *= 1.5
        
        # Length penalty for very short messages
        if len(content) < 20:
            importance *= 0.8
        
        return min(2.0, importance)  # Cap at 2.0
    
    def _calculate_fact_confidence(self, pattern: str, match: tuple, context: str) -> float:
        """Calculate confidence score for extracted facts."""
        base_confidence = 0.7
        
        # Pattern-specific confidence adjustments
        if "is" in pattern:
            base_confidence = 0.8  # High confidence for "is" relationships
        elif "can" in pattern or "cannot" in pattern:
            base_confidence = 0.9  # Very high confidence for capability statements
        elif "works with" in pattern or "supports" in pattern:
            base_confidence = 0.8
        elif "needs" in pattern or "requires" in pattern:
            base_confidence = 0.85
        
        # Context-based adjustments
        context_lower = context.lower()
        
        # Certainty indicators
        certainty_indicators = ["definitely", "certainly", "always", "never", "must", "shall"]
        if any(indicator in context_lower for indicator in certainty_indicators):
            base_confidence += 0.1
        
        # Uncertainty indicators
        uncertainty_indicators = ["maybe", "perhaps", "might", "could", "possibly", "sometimes"]
        if any(indicator in context_lower for indicator in uncertainty_indicators):
            base_confidence -= 0.2
        
        # Question context reduces confidence
        if "?" in context:
            base_confidence -= 0.1
        
        return max(0.1, min(1.0, base_confidence))  # Clamp between 0.1 and 1.0
    
    def _calculate_relevance_score(self, query: str, item: dict) -> float:
        """Calculate relevance score for a memory item."""
        score = 0.0
        item_text = str(item).lower()
        
        # Direct keyword matches
        query_words = query.split()
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                if word in item_text:
                    score += 0.3
        
        # Exact phrase match
        if query in item_text:
            score += 0.5
        
        # Importance weighting
        importance = item.get("importance", 1.0)
        score *= importance
        
        # Recency bonus for working memory items
        if "timestamp" in item:
            try:
                item_time = datetime.fromisoformat(item["timestamp"])
                hours_old = (datetime.now() - item_time).total_seconds() / 3600
                recency_bonus = max(0, 1.0 - (hours_old / 24))  # Decay over 24 hours
                score += recency_bonus * 0.2
            except:
                pass
        
        return score
    
    def _calculate_fact_relevance(self, query: str, fact: Fact) -> float:
        """Calculate relevance score for a fact."""
        score = 0.0
        
        # Check subject relevance
        if query in fact.subject.lower():
            score += 0.4
        
        # Check object relevance
        if isinstance(fact.object, str) and query in fact.object.lower():
            score += 0.4
        
        # Check predicate relevance
        if query in fact.predicate:
            score += 0.2
        
        # Factor in confidence and importance
        score *= fact.confidence * fact.importance
        
        # Access frequency bonus
        if fact.access_count > 0:
            score += min(0.3, fact.access_count * 0.1)
        
        return score
    
    def _calculate_context_score(self, results: dict) -> float:
        """Calculate overall context score for query results."""
        score = 0.0
        
        # Weight different result types
        weights = {
            "working_memory": 0.3,
            "episodic_memory": 0.2,
            "facts": 0.4,
            "react_cycles": 0.1
        }
        
        for result_type, items in results.items():
            if result_type in weights and isinstance(items, list):
                type_score = 0.0
                for item in items:
                    if isinstance(item, dict) and "relevance_score" in item:
                        type_score += item["relevance_score"]
                
                score += (type_score / max(len(items), 1)) * weights[result_type]
        
        return min(1.0, score)  # Cap at 1.0
    
    # ========== STATISTICS AND VISUALIZATION ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        # Basic counts
        successful_cycles = sum(1 for c in self.react_cycles.values() if c.success)
        total_cycles = len(self.react_cycles)
        
        # Link statistics
        link_types = defaultdict(int)
        link_strengths = []
        for link in self.links:
            link_type = f"{link.source_type.value} → {link.target_type.value}"
            link_types[link_type] += 1
            link_strengths.append(link.strength)
        
        # Knowledge graph stats
        kg_stats = self.knowledge_graph.get_statistics()
        
        # Memory distribution
        memory_distribution = {
            "working_memory": len(self.working_memory),
            "episodic_memory": len(self.episodic_memory),
            "semantic_facts": len(self.knowledge_graph.facts),
            "react_cycles": len(self.react_cycles),
            "chat_messages": len(self.chat_history),
            "total_links": len(self.links)
        }
        
        # Performance metrics
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            # Session info
            "session_id": self.session_id,
            "session_duration_seconds": session_duration,
            "session_duration_formatted": str(timedelta(seconds=int(session_duration))),
            
            # Query performance
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "query_success_rate": self.successful_queries / max(self.total_queries, 1),
            
            # Memory distribution
            "memory_distribution": memory_distribution,
            
            # ReAct cycle performance
            "total_react_cycles": total_cycles,
            "successful_cycles": successful_cycles,
            "cycle_success_rate": successful_cycles / max(total_cycles, 1),
            
            # Knowledge graph
            "knowledge_graph": kg_stats,
            
            # Linking system
            "link_statistics": {
                "total_links": len(self.links),
                "unique_link_types": len(link_types),
                "average_link_strength": sum(link_strengths) / max(len(link_strengths), 1),
                "link_type_distribution": dict(link_types)
            },
            
            # Configuration
            "configuration": {
                "working_memory_size": self.working_memory_size,
                "episodic_consolidation_threshold": self.episodic_consolidation_threshold,
                "cross_session_persistence": self.enable_cross_session_persistence
            },
            
            # Health metrics
            "health_metrics": {
                "memory_utilization": len(self.working_memory) / self.working_memory_size,
                "consolidation_needed": len(self.working_memory) > self.working_memory_size,
                "last_consolidation": self.last_consolidation.isoformat()
            }
        }
    
    def visualize_links(self, component_type: Optional[MemoryComponent] = None,
                       component_id: Optional[str] = None, max_depth: int = 2) -> str:
        """Create enhanced text visualization of memory links."""
        lines = ["=== Enhanced Memory Link Visualization ==="]
        
        if component_type and component_id:
            # Show detailed links for specific component
            lines.append(f"\\nDetailed view for {component_type.value}:{component_id}")
            
            # Get direct links
            links = self.get_links(component_type, component_id)
            if links:
                lines.append(f"\\nDirect connections ({len(links)}):")
                for link in links[:10]:  # Limit to top 10
                    strength_bar = "●" * int(link.strength * 5)
                    lines.append(f"  → {link.target_type.value}:{link.target_id}")
                    lines.append(f"    [{link.relationship}] {strength_bar} (strength: {link.strength:.2f}, used: {link.accessed_count}x)")
            
            # Show indirect connections if depth > 1
            if max_depth > 1:
                lines.append(f"\\nIndirect connections (depth {max_depth}):")
                visited = {f"{component_type.value}:{component_id}"}
                current_level = {f"{component_type.value}:{component_id}"}
                
                for depth in range(1, max_depth + 1):
                    next_level = set()
                    for current_key in current_level:
                        if current_key in visited and depth > 1:
                            continue
                        
                        # Parse component key
                        comp_type_str, comp_id = current_key.split(":", 1)
                        try:
                            comp_type = MemoryComponent(comp_type_str)
                            depth_links = self.get_links(comp_type, comp_id)
                            
                            for link in depth_links[:3]:  # Limit indirect connections
                                target_key = f"{link.target_type.value}:{link.target_id}"
                                if target_key not in visited:
                                    next_level.add(target_key)
                                    lines.append(f"    {'  ' * depth}→ {target_key} [{link.relationship}]")
                        except ValueError:
                            continue
                    
                    visited.update(current_level)
                    current_level = next_level
                    
                    if not current_level:
                        break
        else:
            # Show overview of all links
            lines.append(f"\\nMemory System Overview:")
            lines.append(f"Total Components: {len(self.link_index)}")
            lines.append(f"Total Links: {len(self.links)}")
            
            # Link type distribution
            link_counts = defaultdict(int)
            relationship_counts = defaultdict(int)
            
            for link in self.links:
                link_type = f"{link.source_type.value} → {link.target_type.value}"
                link_counts[link_type] += 1
                relationship_counts[link.relationship] += 1
            
            lines.append("\\nLink Type Distribution:")
            for link_type, count in sorted(link_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(self.links)) * 100
                lines.append(f"  {link_type}: {count} ({percentage:.1f}%)")
            
            lines.append("\\nTop Relationship Types:")
            for relationship, count in sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  {relationship}: {count}")
            
            # Most connected components
            lines.append("\\nMost Connected Components:")
            component_connections = defaultdict(int)
            for component_key, links in self.link_index.items():
                component_connections[component_key] = len(links)
            
            top_components = sorted(component_connections.items(), key=lambda x: x[1], reverse=True)[:5]
            for component, connection_count in top_components:
                lines.append(f"  {component}: {connection_count} connections")
        
        return "\\n".join(lines)
    
    def get_memory_health_report(self) -> str:
        """Generate a comprehensive health report for the memory system."""
        stats = self.get_statistics()
        
        report_lines = ["=== Memory System Health Report ==="]
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Session: {self.session_id}")
        
        # Overall health assessment
        health_score = 0.0
        health_factors = []
        
        # Memory utilization
        memory_util = stats["health_metrics"]["memory_utilization"]
        if memory_util < 0.8:
            health_score += 25
            health_factors.append("✓ Memory utilization healthy")
        else:
            health_factors.append("⚠ Memory utilization high")
        
        # Query success rate
        success_rate = stats["query_success_rate"]
        if success_rate > 0.8:
            health_score += 25
            health_factors.append("✓ High query success rate")
        elif success_rate > 0.6:
            health_score += 15
            health_factors.append("~ Moderate query success rate")
        else:
            health_factors.append("⚠ Low query success rate")
        
        # Knowledge graph growth
        if stats["knowledge_graph"]["total_facts"] > 10:
            health_score += 25
            health_factors.append("✓ Good knowledge accumulation")
        elif stats["knowledge_graph"]["total_facts"] > 5:
            health_score += 15
            health_factors.append("~ Some knowledge accumulation")
        else:
            health_factors.append("⚠ Limited knowledge accumulation")
        
        # Link connectivity
        if stats["link_statistics"]["total_links"] > 20:
            health_score += 25
            health_factors.append("✓ Well-connected memory")
        elif stats["link_statistics"]["total_links"] > 10:
            health_score += 15
            health_factors.append("~ Moderately connected memory")
        else:
            health_factors.append("⚠ Sparse memory connections")
        
        # Overall assessment
        if health_score >= 80:
            health_status = "EXCELLENT"
        elif health_score >= 60:
            health_status = "GOOD"
        elif health_score >= 40:
            health_status = "FAIR"
        else:
            health_status = "NEEDS ATTENTION"
        
        report_lines.append(f"\\nOverall Health: {health_status} ({health_score}/100)")
        
        # Health factors
        report_lines.append("\\nHealth Factors:")
        for factor in health_factors:
            report_lines.append(f"  {factor}")
        
        # Memory statistics
        report_lines.append(f"\\nMemory Statistics:")
        report_lines.append(f"  Working Memory: {stats['memory_distribution']['working_memory']}/{self.working_memory_size}")
        report_lines.append(f"  Episodic Memory: {stats['memory_distribution']['episodic_memory']} items")
        report_lines.append(f"  Knowledge Facts: {stats['knowledge_graph']['total_facts']}")
        report_lines.append(f"  ReAct Cycles: {stats['total_react_cycles']} ({stats['successful_cycles']} successful)")
        
        # Recommendations
        recommendations = []
        if memory_util > 0.9:
            recommendations.append("Consider reducing working_memory_size or consolidation_threshold")
        if success_rate < 0.7:
            recommendations.append("Review query processing and fact extraction patterns")
        if stats["knowledge_graph"]["total_facts"] < 5:
            recommendations.append("Enhance fact extraction to build knowledge base")
        if stats["link_statistics"]["total_links"] < 10:
            recommendations.append("Improve memory linking to enhance context retrieval")
        
        if recommendations:
            report_lines.append("\\nRecommendations:")
            for rec in recommendations:
                report_lines.append(f"  • {rec}")
        
        return "\\n".join(report_lines)


# ========== BACKWARD COMPATIBILITY ALIASES ==========

# Legacy class aliases for existing imports
ConversationMemory = HierarchicalMemory  # From memory.py
MemorySystem = HierarchicalMemory  # Generic alias
ReactScratchpad = ReActCycle  # Legacy naming
KnowledgeTriple = Fact  # Legacy naming from memory.py

# Legacy function aliases
def create_memory_system(**kwargs) -> HierarchicalMemory:
    """Create a new hierarchical memory system (legacy function)."""
    return HierarchicalMemory(**kwargs)

def load_memory_system(persist_path: Path, **kwargs) -> HierarchicalMemory:
    """Load an existing memory system from disk (legacy function)."""
    memory = HierarchicalMemory(persist_path=persist_path, **kwargs)
    return memory