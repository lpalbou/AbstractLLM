"""
Advanced Memory System v2 for AbstractLLM with SOTA Architecture.

Implements:
- Hierarchical memory (working, episodic, semantic)
- Scratchpad per ReAct cycle with unique IDs
- Bidirectional linking between all memory components
- Fact extraction and knowledge graph
- Proper serialization/deserialization
- Memory persistence and retrieval

Based on A-Mem, RAISE, and MemGPT architectures.
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


class MemoryComponent(Enum):
    """Types of memory components."""
    CHAT_HISTORY = "chat_history"
    SCRATCHPAD = "scratchpad"
    KNOWLEDGE = "knowledge"
    EPISODIC = "episodic"
    WORKING = "working"


@dataclass
class MemoryLink:
    """Bidirectional link between memory components."""
    
    source_type: MemoryComponent
    source_id: str
    target_type: MemoryComponent
    target_id: str
    relationship: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def reverse(self) -> 'MemoryLink':
        """Create reverse link."""
        return MemoryLink(
            source_type=self.target_type,
            source_id=self.target_id,
            target_type=self.source_type,
            target_id=self.source_id,
            relationship=f"reverse_{self.relationship}",
            metadata=self.metadata,
            created_at=self.created_at
        )


@dataclass
class ReActCycle:
    """
    A single ReAct cycle with its own scratchpad.
    One query = One agent response = One ReAct cycle.
    """
    
    cycle_id: str = field(default_factory=lambda: f"cycle_{uuid.uuid4().hex[:8]}")
    query: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Scratchpad for this specific cycle
    thoughts: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Result and metadata
    final_answer: Optional[str] = None
    iterations: int = 0
    max_iterations: int = 10
    success: bool = False
    error: Optional[str] = None
    
    # Links to other memory components
    chat_message_ids: List[str] = field(default_factory=list)
    extracted_fact_ids: List[str] = field(default_factory=list)
    
    def add_thought(self, content: str, confidence: float = 1.0):
        """Add a thought to the scratchpad."""
        self.thoughts.append({
            "id": f"thought_{uuid.uuid4().hex[:8]}",
            "content": content,
            "confidence": confidence,
            "iteration": self.iterations,
            "timestamp": datetime.now().isoformat()
        })
        
    def add_action(self, tool_name: str, arguments: Dict[str, Any], reasoning: str):
        """Add an action to the scratchpad."""
        action_id = f"action_{uuid.uuid4().hex[:8]}"
        self.actions.append({
            "id": action_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "reasoning": reasoning,
            "iteration": self.iterations,
            "timestamp": datetime.now().isoformat()
        })
        return action_id
        
    def add_observation(self, action_id: str, content: Any, success: bool = True):
        """Add an observation linked to an action."""
        self.observations.append({
            "id": f"obs_{uuid.uuid4().hex[:8]}",
            "action_id": action_id,
            "content": content,
            "success": success,
            "iteration": self.iterations,
            "timestamp": datetime.now().isoformat()
        })
        
    def complete(self, answer: str, success: bool = True):
        """Mark cycle as complete."""
        self.final_answer = answer
        self.success = success
        self.end_time = datetime.now()
        
    def get_trace(self) -> str:
        """Get formatted trace of the ReAct cycle."""
        trace = [f"=== ReAct Cycle {self.cycle_id} ==="]
        trace.append(f"Query: {self.query}")
        trace.append(f"Iterations: {self.iterations}/{self.max_iterations}")
        
        for i in range(self.iterations + 1):
            # Thoughts for this iteration
            iter_thoughts = [t for t in self.thoughts if t["iteration"] == i]
            for thought in iter_thoughts:
                trace.append(f"  Thought {i}: {thought['content']}")
            
            # Actions for this iteration
            iter_actions = [a for a in self.actions if a["iteration"] == i]
            for action in iter_actions:
                trace.append(f"  Action {i}: {action['tool_name']}({json.dumps(action['arguments'])})")
                
                # Related observations
                related_obs = [o for o in self.observations 
                             if o["action_id"] == action["id"]]
                for obs in related_obs:
                    status = "✓" if obs["success"] else "✗"
                    trace.append(f"  Observation {i} {status}: {str(obs['content'])[:200]}")
        
        if self.final_answer:
            trace.append(f"Final Answer: {self.final_answer[:500]}")
        
        return "\n".join(trace)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "query": self.query,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations,
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "success": self.success,
            "error": self.error,
            "chat_message_ids": self.chat_message_ids,
            "extracted_fact_ids": self.extracted_fact_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReActCycle':
        """Deserialize from dictionary."""
        cycle = cls(cycle_id=data["cycle_id"])
        cycle.query = data["query"]
        cycle.start_time = datetime.fromisoformat(data["start_time"])
        cycle.end_time = datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        cycle.thoughts = data["thoughts"]
        cycle.actions = data["actions"]
        cycle.observations = data["observations"]
        cycle.final_answer = data["final_answer"]
        cycle.iterations = data["iterations"]
        cycle.max_iterations = data["max_iterations"]
        cycle.success = data["success"]
        cycle.error = data["error"]
        cycle.chat_message_ids = data["chat_message_ids"]
        cycle.extracted_fact_ids = data["extracted_fact_ids"]
        return cycle


@dataclass
class Fact:
    """A single fact extracted from conversation."""
    
    fact_id: str = field(default_factory=lambda: f"fact_{uuid.uuid4().hex[:8]}")
    subject: str = ""
    predicate: str = ""
    object: Any = None
    confidence: float = 1.0
    source_type: MemoryComponent = MemoryComponent.CHAT_HISTORY
    source_id: str = ""
    extracted_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.subject} --[{self.predicate}]--> {self.object}"
    
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
            metadata=data.get("metadata", {})
        )


class HierarchicalMemory:
    """
    SOTA Hierarchical Memory System with bidirectional linking.
    Inspired by A-Mem, RAISE, and MemGPT architectures.
    """
    
    def __init__(self, 
                 working_memory_size: int = 10,
                 episodic_consolidation_threshold: int = 5,
                 persist_path: Optional[Path] = None):
        """
        Initialize hierarchical memory system.
        
        Args:
            working_memory_size: Max items in working memory
            episodic_consolidation_threshold: When to consolidate to episodic
            persist_path: Optional path for persistent storage
        """
        # Memory stores
        self.working_memory: List[Dict[str, Any]] = []  # Most recent, active
        self.episodic_memory: List[Dict[str, Any]] = []  # Consolidated experiences
        self.semantic_memory: Dict[str, Fact] = {}  # Facts and knowledge
        
        # ReAct cycles
        self.react_cycles: Dict[str, ReActCycle] = {}
        self.current_cycle: Optional[ReActCycle] = None
        
        # Bidirectional links
        self.links: List[MemoryLink] = []
        self.link_index: Dict[str, List[MemoryLink]] = defaultdict(list)
        
        # Chat history with links
        self.chat_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.working_memory_size = working_memory_size
        self.episodic_consolidation_threshold = episodic_consolidation_threshold
        self.persist_path = Path(persist_path) if persist_path else None
        
        # Session metadata
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.session_start = datetime.now()
        
        # Load persisted memory if available
        if self.persist_path and self.persist_path.exists():
            self.load_from_disk()
    
    def start_react_cycle(self, query: str, max_iterations: int = 10) -> ReActCycle:
        """Start a new ReAct cycle for a query."""
        if self.current_cycle and not self.current_cycle.end_time:
            logger.warning(f"Previous cycle {self.current_cycle.cycle_id} not completed")
            self.current_cycle.complete("Interrupted", success=False)
        
        cycle = ReActCycle(query=query, max_iterations=max_iterations)
        self.current_cycle = cycle
        self.react_cycles[cycle.cycle_id] = cycle
        
        logger.info(f"Started ReAct cycle {cycle.cycle_id} for query: {query[:100]}")
        return cycle
    
    def add_chat_message(self, role: str, content: str, 
                        cycle_id: Optional[str] = None) -> str:
        """Add a chat message and link to ReAct cycle if provided."""
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        message = {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "cycle_id": cycle_id,
            "fact_ids": []  # Will be populated by fact extraction
        }
        
        self.chat_history.append(message)
        self.working_memory.append(message)
        
        # Create bidirectional link to ReAct cycle
        if cycle_id and cycle_id in self.react_cycles:
            self.add_link(
                MemoryComponent.CHAT_HISTORY, message_id,
                MemoryComponent.SCRATCHPAD, cycle_id,
                "generated_by"
            )
            self.react_cycles[cycle_id].chat_message_ids.append(message_id)
        
        # Extract facts from message
        facts = self.extract_facts(content, MemoryComponent.CHAT_HISTORY, message_id)
        for fact in facts:
            message["fact_ids"].append(fact.fact_id)
        
        # Consolidate if needed
        if len(self.working_memory) > self.working_memory_size:
            self._consolidate_working_memory()
        
        return message_id
    
    def extract_facts(self, content: str, source_type: MemoryComponent, 
                     source_id: str) -> List[Fact]:
        """Extract facts from content using patterns and NLP."""
        facts = []
        
        # Pattern-based extraction
        patterns = [
            (r"(\w+)\s+is\s+(\w+)", "is"),
            (r"(\w+)\s+has\s+(\w+)", "has"),
            (r"(\w+)\s+can\s+(\w+)", "can"),
            (r"(\w+)\s+needs\s+(\w+)", "needs"),
            (r"(\w+)\s+works\s+with\s+(\w+)", "works_with"),
            (r"(\w+)\s+supports\s+(\w+)", "supports"),
        ]
        
        for pattern, predicate in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    fact = Fact(
                        subject=match[0].lower(),
                        predicate=predicate,
                        object=match[1].lower(),
                        source_type=source_type,
                        source_id=source_id,
                        confidence=0.7  # Pattern-based extraction has lower confidence
                    )
                    
                    # Add to semantic memory
                    self.semantic_memory[fact.fact_id] = fact
                    facts.append(fact)
                    
                    # Create bidirectional link
                    self.add_link(
                        source_type, source_id,
                        MemoryComponent.KNOWLEDGE, fact.fact_id,
                        "extracted_fact"
                    )
        
        logger.debug(f"Extracted {len(facts)} facts from {source_type.value}:{source_id}")
        return facts
    
    def add_link(self, source_type: MemoryComponent, source_id: str,
                target_type: MemoryComponent, target_id: str,
                relationship: str):
        """Add bidirectional link between memory components."""
        # Forward link
        forward_link = MemoryLink(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            relationship=relationship
        )
        
        # Reverse link
        reverse_link = forward_link.reverse()
        
        # Add both links
        self.links.append(forward_link)
        self.links.append(reverse_link)
        
        # Index for fast lookup
        self.link_index[f"{source_type.value}:{source_id}"].append(forward_link)
        self.link_index[f"{target_type.value}:{target_id}"].append(reverse_link)
    
    def get_links(self, component_type: MemoryComponent, 
                 component_id: str) -> List[MemoryLink]:
        """Get all links for a component."""
        key = f"{component_type.value}:{component_id}"
        return self.link_index.get(key, [])
    
    def get_related_facts(self, entity: str, max_depth: int = 2) -> List[Fact]:
        """Get facts related to an entity using graph traversal."""
        related_facts = []
        visited = set()
        to_visit = {entity.lower()}
        
        for depth in range(max_depth):
            new_entities = set()
            
            for current_entity in to_visit:
                if current_entity in visited:
                    continue
                visited.add(current_entity)
                
                # Find facts where entity is subject or object
                for fact_id, fact in self.semantic_memory.items():
                    if fact.subject == current_entity:
                        related_facts.append(fact)
                        new_entities.add(fact.object)
                    elif str(fact.object).lower() == current_entity:
                        related_facts.append(fact)
                        new_entities.add(fact.subject)
            
            to_visit = new_entities - visited
        
        return related_facts
    
    def query_memory(self, query: str, include_links: bool = True) -> Dict[str, Any]:
        """Query memory for relevant information."""
        query_lower = query.lower()
        results = {
            "working_memory": [],
            "episodic_memory": [],
            "facts": [],
            "react_cycles": [],
            "links": []
        }
        
        # Search working memory
        for item in self.working_memory:
            if query_lower in str(item).lower():
                results["working_memory"].append(item)
        
        # Search episodic memory
        for item in self.episodic_memory:
            if query_lower in str(item).lower():
                results["episodic_memory"].append(item)
        
        # Search facts
        for fact_id, fact in self.semantic_memory.items():
            if (query_lower in fact.subject.lower() or 
                query_lower in str(fact.object).lower()):
                results["facts"].append(fact.to_dict())
        
        # Search ReAct cycles
        for cycle_id, cycle in self.react_cycles.items():
            if query_lower in cycle.query.lower():
                results["react_cycles"].append({
                    "cycle_id": cycle_id,
                    "query": cycle.query,
                    "success": cycle.success,
                    "iterations": cycle.iterations
                })
        
        # Include relevant links if requested
        if include_links:
            # Get links for found items
            for fact in results["facts"]:
                links = self.get_links(MemoryComponent.KNOWLEDGE, fact["fact_id"])
                for link in links:
                    results["links"].append({
                        "from": f"{link.source_type.value}:{link.source_id}",
                        "to": f"{link.target_type.value}:{link.target_id}",
                        "relationship": link.relationship
                    })
        
        return results
    
    def get_context_for_query(self, query: str, max_items: int = 10) -> str:
        """Get relevant context for a query."""
        context_parts = []
        
        # Add recent working memory
        if self.working_memory:
            context_parts.append("Recent context:")
            for item in self.working_memory[-3:]:
                if "content" in item:
                    context_parts.append(f"- {item['content'][:100]}")
        
        # Add current ReAct cycle trace
        if self.current_cycle:
            context_parts.append(f"\nCurrent reasoning (cycle {self.current_cycle.cycle_id}):")
            trace = self.current_cycle.get_trace()
            context_parts.append(trace[-500:])  # Last 500 chars
        
        # Add relevant facts
        query_results = self.query_memory(query, include_links=False)
        if query_results["facts"]:
            context_parts.append("\nRelevant facts:")
            for fact_dict in query_results["facts"][:5]:
                fact = Fact.from_dict(fact_dict)
                context_parts.append(f"- {fact}")
        
        # Add previous successful ReAct cycles for similar queries
        similar_cycles = []
        for cycle_id, cycle in self.react_cycles.items():
            if cycle.success and query.lower() in cycle.query.lower():
                similar_cycles.append(cycle)
        
        if similar_cycles:
            context_parts.append("\nPrevious successful approaches:")
            for cycle in similar_cycles[:2]:
                context_parts.append(f"- Query: {cycle.query[:100]}")
                context_parts.append(f"  Answer: {cycle.final_answer[:100]}")
        
        return "\n".join(context_parts)
    
    def _consolidate_working_memory(self):
        """Move items from working to episodic memory."""
        to_consolidate = self.working_memory[:self.episodic_consolidation_threshold]
        
        for item in to_consolidate:
            # Add consolidation metadata
            item["consolidated_at"] = datetime.now().isoformat()
            item["session_id"] = self.session_id
            
            # Extract additional facts before moving to episodic
            if "content" in item:
                self.extract_facts(
                    item["content"],
                    MemoryComponent.EPISODIC,
                    item.get("id", str(uuid.uuid4()))
                )
            
            self.episodic_memory.append(item)
        
        # Remove from working memory
        self.working_memory = self.working_memory[self.episodic_consolidation_threshold:]
        
        logger.info(f"Consolidated {len(to_consolidate)} items to episodic memory")
    
    def save_to_disk(self):
        """Persist memory to disk."""
        if not self.persist_path:
            return
        
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        memory_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "working_memory": self.working_memory,
            "episodic_memory": self.episodic_memory,
            "semantic_memory": {k: v.to_dict() for k, v in self.semantic_memory.items()},
            "react_cycles": {k: v.to_dict() for k, v in self.react_cycles.items()},
            "chat_history": self.chat_history,
            "links": [
                {
                    "source_type": link.source_type.value,
                    "source_id": link.source_id,
                    "target_type": link.target_type.value,
                    "target_id": link.target_id,
                    "relationship": link.relationship,
                    "metadata": link.metadata,
                    "created_at": link.created_at.isoformat()
                }
                for link in self.links
            ]
        }
        
        # Save as JSON
        memory_file = self.persist_path / f"{self.session_id}.json"
        with open(memory_file, "w") as f:
            json.dump(memory_data, f, indent=2)
        
        logger.info(f"Saved memory to {memory_file}")
    
    def load_from_disk(self):
        """Load memory from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        # Find most recent session file
        session_files = list(self.persist_path.glob("session_*.json"))
        if not session_files:
            return
        
        latest_file = max(session_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, "r") as f:
            memory_data = json.load(f)
        
        # Restore memory components
        self.session_id = memory_data["session_id"]
        self.session_start = datetime.fromisoformat(memory_data["session_start"])
        self.working_memory = memory_data["working_memory"]
        self.episodic_memory = memory_data["episodic_memory"]
        
        # Restore semantic memory
        self.semantic_memory = {
            k: Fact.from_dict(v) 
            for k, v in memory_data["semantic_memory"].items()
        }
        
        # Restore ReAct cycles
        self.react_cycles = {
            k: ReActCycle.from_dict(v)
            for k, v in memory_data["react_cycles"].items()
        }
        
        self.chat_history = memory_data["chat_history"]
        
        # Restore links
        for link_data in memory_data["links"]:
            link = MemoryLink(
                source_type=MemoryComponent(link_data["source_type"]),
                source_id=link_data["source_id"],
                target_type=MemoryComponent(link_data["target_type"]),
                target_id=link_data["target_id"],
                relationship=link_data["relationship"],
                metadata=link_data["metadata"],
                created_at=datetime.fromisoformat(link_data["created_at"])
            )
            self.links.append(link)
            
            # Rebuild index
            key = f"{link.source_type.value}:{link.source_id}"
            self.link_index[key].append(link)
        
        logger.info(f"Loaded memory from {latest_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        successful_cycles = sum(1 for c in self.react_cycles.values() if c.success)
        total_cycles = len(self.react_cycles)
        
        return {
            "session_id": self.session_id,
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "total_facts": len(self.semantic_memory),
            "total_react_cycles": total_cycles,
            "successful_cycles": successful_cycles,
            "success_rate": successful_cycles / max(total_cycles, 1),
            "total_links": len(self.links),
            "chat_messages": len(self.chat_history)
        }
    
    def visualize_links(self, component_type: Optional[MemoryComponent] = None,
                       component_id: Optional[str] = None) -> str:
        """Create a text visualization of memory links."""
        lines = ["=== Memory Link Visualization ==="]
        
        if component_type and component_id:
            # Show links for specific component
            links = self.get_links(component_type, component_id)
            lines.append(f"\nLinks for {component_type.value}:{component_id}:")
            for link in links:
                lines.append(f"  → {link.target_type.value}:{link.target_id} ({link.relationship})")
        else:
            # Show overview of all links
            link_counts = defaultdict(int)
            for link in self.links:
                key = f"{link.source_type.value} → {link.target_type.value}"
                link_counts[key] += 1
            
            lines.append("\nLink distribution:")
            for link_type, count in sorted(link_counts.items()):
                lines.append(f"  {link_type}: {count}")
        
        return "\n".join(lines)