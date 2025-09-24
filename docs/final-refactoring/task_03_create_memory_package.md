# Task 03: Create Memory Package Structure (Priority 2)

**Duration**: 3 hours
**Risk**: Medium
**Dependencies**: Task 02 completed

## Objectives
- Create AbstractMemory package with **two-tier strategy** for different agent types
- Implement **ScratchpadMemory** for simple task agents (summarizers, extractors, ReAct)
- Implement **TemporalMemory** with full three-tier architecture for autonomous agents
- Provide clean, efficient memory selection based on agent purpose
- Avoid over-engineering: use BasicSession when sufficient

## SOTA Research Foundation
Based on 2024 research, **memory should match agent purpose**:

### Simple Agents (Task-Specific)
- **Use Cases**: Summarizers, extractors, ReAct agents, single-task tools
- **Memory Need**: Temporary scratchpad, no persistence required
- **Solution**: ScratchpadMemory or BasicSession from AbstractLLM Core
- **Example**: A summarizer just needs input text + working space

### Complex Agents (Autonomous)
- **Use Cases**: Personal assistants, learning agents, multi-session agents
- **Memory Need**: Persistence, user profiles, knowledge accumulation
- **Solution**: Full TemporalMemory with three tiers (Core → Working → Episodic)
- **Example**: A personal assistant that remembers user preferences across sessions

## Steps

### 1. Create Package Structure (30 min)

```bash
# Navigate to new package location
cd /Users/albou/projects
mkdir -p abstractmemory
cd abstractmemory

# Create package structure
mkdir -p abstractmemory/{core,components,graph,cognitive,storage}
mkdir -p tests docs examples

# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="abstractmemory",
    version="1.0.0",
    author="AbstractLLM Team",
    description="Temporal knowledge graph memory system for LLM agents",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "abstractllm>=2.0.0",
        "networkx>=3.0",        # For graph operations
        "lancedb>=0.3.0",       # For vector storage
        "sentence-transformers>=2.0.0",  # For embeddings
        "pydantic>=2.0.0",      # For data validation
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy"],
    }
)
EOF

# Create __init__.py files
touch abstractmemory/__init__.py
touch abstractmemory/core/__init__.py
touch abstractmemory/components/__init__.py
touch abstractmemory/graph/__init__.py
touch abstractmemory/cognitive/__init__.py
touch abstractmemory/storage/__init__.py
```

### 2. Implement Core Interfaces (30 min)

Create `abstractmemory/core/interfaces.py`:
```python
"""
Core memory interfaces based on SOTA research.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class MemoryItem:
    """Base class for memory items"""
    content: Any
    event_time: datetime      # When it happened
    ingestion_time: datetime  # When we learned it
    confidence: float = 1.0
    metadata: Dict[str, Any] = None


class IMemoryComponent(ABC):
    """Interface for memory components"""

    @abstractmethod
    def add(self, item: MemoryItem) -> str:
        """Add item to memory, return ID"""
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Retrieve relevant items"""
        pass

    @abstractmethod
    def consolidate(self) -> int:
        """Consolidate memory, return items consolidated"""
        pass


class IRetriever(ABC):
    """Interface for retrieval strategies"""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[Tuple[float, Any]]:
        """Search and return (score, item) tuples"""
        pass


class IStorage(ABC):
    """Interface for storage backends"""

    @abstractmethod
    def save(self, key: str, value: Any) -> None:
        """Save value with key"""
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """Load value by key"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
```

### 3. Implement Simple Memory for Task Agents (30 min)

Create `abstractmemory/simple.py`:
```python
"""
Simple, efficient memory for task-specific agents.
No over-engineering - just what's needed for the job.
"""

from typing import List, Optional, Dict, Any
from collections import deque
from datetime import datetime


class ScratchpadMemory:
    """
    Lightweight memory for ReAct agents and single-task tools.

    Use this for:
    - ReAct agent thought-action-observation cycles
    - Summarizer working memory
    - Extractor temporary context
    - Any agent that doesn't need persistence

    Example:
        # For a ReAct agent
        scratchpad = ScratchpadMemory(max_entries=20)
        scratchpad.add_thought("Need to search for Python tutorials")
        scratchpad.add_action("search", {"query": "Python basics"})
        scratchpad.add_observation("Found 10 relevant tutorials")

        # Get full context for next iteration
        context = scratchpad.get_context()
    """

    def __init__(self, max_entries: int = 100):
        """Initialize scratchpad with bounded size"""
        self.entries: deque = deque(maxlen=max_entries)
        self.thoughts: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[str] = []

    def add(self, content: str, entry_type: str = "note"):
        """Add any entry to scratchpad"""
        entry = {
            "type": entry_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.entries.append(entry)

    def add_thought(self, thought: str):
        """Add a thought (for ReAct pattern)"""
        self.thoughts.append(thought)
        self.add(thought, "thought")

    def add_action(self, action: str, params: Optional[Dict] = None):
        """Add an action (for ReAct pattern)"""
        action_entry = {"action": action, "params": params or {}}
        self.actions.append(action_entry)
        self.add(f"Action: {action} with {params}", "action")

    def add_observation(self, observation: str):
        """Add an observation (for ReAct pattern)"""
        self.observations.append(observation)
        self.add(observation, "observation")

    def get_context(self, last_n: Optional[int] = None) -> str:
        """Get scratchpad context as string"""
        entries_to_use = list(self.entries)
        if last_n:
            entries_to_use = entries_to_use[-last_n:]

        context_lines = []
        for entry in entries_to_use:
            if entry["type"] == "thought":
                context_lines.append(f"Thought: {entry['content']}")
            elif entry["type"] == "action":
                context_lines.append(f"Action: {entry['content']}")
            elif entry["type"] == "observation":
                context_lines.append(f"Observation: {entry['content']}")
            else:
                context_lines.append(entry['content'])

        return "\n".join(context_lines)

    def get_react_history(self) -> Dict[str, List]:
        """Get structured ReAct history"""
        return {
            "thoughts": self.thoughts,
            "actions": self.actions,
            "observations": self.observations
        }

    def clear(self):
        """Clear the scratchpad"""
        self.entries.clear()
        self.thoughts.clear()
        self.actions.clear()
        self.observations.clear()

    def __len__(self) -> int:
        return len(self.entries)

    def __str__(self) -> str:
        return f"ScratchpadMemory({len(self.entries)} entries)"


class BufferMemory:
    """
    Simple conversation buffer (wrapper around BasicSession).

    Use this when BasicSession from AbstractLLM Core is sufficient.
    This is just a thin adapter for compatibility.

    Example:
        # For a simple chatbot
        memory = BufferMemory(max_messages=50)
        memory.add_message("user", "What's the weather?")
        memory.add_message("assistant", "I don't have weather data")
        context = memory.get_context()
    """

    def __init__(self, max_messages: int = 100):
        """Initialize buffer with size limit"""
        self.messages: deque = deque(maxlen=max_messages)

    def add_message(self, role: str, content: str):
        """Add a message to the buffer"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages for LLM context"""
        return [{"role": m["role"], "content": m["content"]}
                for m in self.messages]

    def get_context(self, last_n: Optional[int] = None) -> str:
        """Get conversation as formatted string"""
        messages = list(self.messages)
        if last_n:
            messages = messages[-last_n:]

        lines = []
        for msg in messages:
            lines.append(f"{msg['role']}: {msg['content']}")

        return "\n".join(lines)

    def clear(self):
        """Clear the buffer"""
        self.messages.clear()
```

### 4. Implement Temporal Anchoring for Complex Memory (45 min)

Create `abstractmemory/core/temporal.py`:
```python
"""
Bi-temporal data model based on Zep/Graphiti research.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TemporalSpan:
    """Represents a time span with validity"""
    start: datetime
    end: Optional[datetime] = None
    valid: bool = True


@dataclass
class RelationalContext:
    """Who is involved in this memory"""
    user_id: str                     # Primary user/speaker
    agent_id: Optional[str] = None   # Which agent persona
    relationship: Optional[str] = None  # "owner", "colleague", "stranger"
    session_id: Optional[str] = None   # Conversation session

@dataclass
class GroundingAnchor:
    """Multi-dimensional grounding for experiential memory"""
    # Temporal grounding (when)
    event_time: datetime        # When it happened
    ingestion_time: datetime    # When we learned about it
    validity_span: TemporalSpan # When it was/is valid

    # Relational grounding (who)
    relational: RelationalContext  # Who is involved

    # Additional grounding
    confidence: float = 1.0
    source: Optional[str] = None
    location: Optional[str] = None  # Where (optional)


class TemporalIndex:
    """Index for efficient temporal queries"""

    def __init__(self):
        self._by_event_time = []      # Sorted by event time
        self._by_ingestion_time = []  # Sorted by ingestion time
        self._anchors = {}             # ID -> TemporalAnchor

    def add_anchor(self, anchor_id: str, anchor: TemporalAnchor):
        """Add temporal anchor to index"""
        self._anchors[anchor_id] = anchor

        # Insert into sorted lists
        self._insert_sorted(self._by_event_time,
                          (anchor.event_time, anchor_id))
        self._insert_sorted(self._by_ingestion_time,
                          (anchor.ingestion_time, anchor_id))

    def query_at_time(self, point_in_time: datetime,
                     use_event_time: bool = True) -> List[str]:
        """Get valid anchor IDs at specific time"""
        valid_ids = []

        for anchor_id, anchor in self._anchors.items():
            # Check if anchor was known at this time
            if anchor.ingestion_time > point_in_time:
                continue

            # Check if anchor was valid at this time
            if use_event_time:
                if anchor.event_time <= point_in_time:
                    if anchor.validity_span.valid:
                        if (anchor.validity_span.end is None or
                            anchor.validity_span.end > point_in_time):
                            valid_ids.append(anchor_id)

        return valid_ids

    def _insert_sorted(self, lst: list, item: tuple):
        """Insert item into sorted list"""
        import bisect
        bisect.insort(lst, item)

    def get_evolution(self, start: datetime, end: datetime) -> List[Tuple[datetime, str]]:
        """Get evolution of knowledge between times"""
        changes = []

        for anchor_id, anchor in self._anchors.items():
            # Include if ingested during period
            if start <= anchor.ingestion_time <= end:
                changes.append((anchor.ingestion_time, f"Added: {anchor_id}"))

            # Include if invalidated during period
            if anchor.validity_span.end:
                if start <= anchor.validity_span.end <= end:
                    changes.append((anchor.validity_span.end, f"Invalidated: {anchor_id}"))

        return sorted(changes)
```

### 5. Implement Memory Components for Complex Agents (60 min)

**For Autonomous Agents Only - SOTA Three-Tier Architecture**: Core → Working → Episodic

Create `abstractmemory/components/core.py`:
```python
"""
Core memory - always-accessible foundational facts (MemGPT/Letta pattern).
"""

from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from abstractmemory.core.interfaces import IMemoryComponent, MemoryItem


@dataclass
class CoreMemoryBlock:
    """A block of core memory (always in context)"""
    block_id: str
    label: str           # "persona" or "user_info"
    content: str         # Max ~200 tokens
    last_updated: datetime
    edit_count: int = 0

    def update(self, new_content: str, agent_reasoning: str):
        """Agent can self-edit this block"""
        self.content = new_content
        self.last_updated = datetime.now()
        self.edit_count += 1
        # In production: log agent_reasoning for transparency


class CoreMemory(IMemoryComponent):
    """
    Always-accessible core memory for fundamental facts.
    Based on MemGPT/Letta research - stores agent persona + user information.
    """

    def __init__(self, max_blocks: int = 10, max_tokens_per_block: int = 200):
        self.blocks: Dict[str, CoreMemoryBlock] = {}
        self.max_blocks = max_blocks
        self.max_tokens_per_block = max_tokens_per_block

        # Initialize default blocks (MemGPT pattern)
        self.blocks["persona"] = CoreMemoryBlock(
            block_id="persona",
            label="persona",
            content="I am an AI assistant with persistent memory capabilities.",
            last_updated=datetime.now()
        )
        self.blocks["user_info"] = CoreMemoryBlock(
            block_id="user_info",
            label="user_info",
            content="User information will be learned over time.",
            last_updated=datetime.now()
        )

    def get_context(self) -> str:
        """Get all core memory as context string (always included in prompts)"""
        context_parts = []
        for block in self.blocks.values():
            context_parts.append(f"[{block.label}] {block.content}")
        return "\n".join(context_parts)

    def update_block(self, block_id: str, content: str, reasoning: str = "") -> bool:
        """Agent updates a core memory block (self-editing capability)"""
        if block_id in self.blocks:
            if len(content) <= self.max_tokens_per_block * 4:  # Rough token estimate
                self.blocks[block_id].update(content, reasoning)
                return True
        return False

    def add_block(self, label: str, content: str) -> Optional[str]:
        """Add new core memory block if space available"""
        if len(self.blocks) < self.max_blocks:
            block_id = f"core_{len(self.blocks)}"
            self.blocks[block_id] = CoreMemoryBlock(
                block_id=block_id,
                label=label,
                content=content,
                last_updated=datetime.now()
            )
            return block_id
        return None

    # IMemoryComponent interface implementation
    def add(self, item: MemoryItem) -> str:
        """Add important fact to core memory"""
        # Convert MemoryItem to core memory block
        content = str(item.content)
        if "user" in content.lower():
            return self.update_block("user_info", content) and "user_info" or ""
        elif "persona" in content.lower() or "agent" in content.lower():
            return self.update_block("persona", content) and "persona" or ""
        else:
            return self.add_block("general", content) or ""

    def retrieve(self, query: str, limit: int = 10) -> list[MemoryItem]:
        """Retrieve core memory blocks matching query"""
        results = []
        query_lower = query.lower()

        for block in self.blocks.values():
            if query_lower in block.content.lower() or query_lower in block.label.lower():
                results.append(MemoryItem(
                    content={"label": block.label, "content": block.content},
                    event_time=block.last_updated,
                    ingestion_time=block.last_updated,
                    confidence=1.0,  # Core memory is always high confidence
                    metadata={"block_id": block.block_id, "edit_count": block.edit_count}
                ))

        return results[:limit]

    def consolidate(self) -> int:
        """Core memory doesn't consolidate - it's manually curated"""
        return 0
```

Create `abstractmemory/components/working.py`:
```python
"""
Working memory with sliding window.
"""

from collections import deque
from typing import List, Optional
from datetime import datetime

from abstractmemory.core.interfaces import IMemoryComponent, MemoryItem


class WorkingMemory(IMemoryComponent):
    """Short-term working memory with fixed capacity"""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.items = deque(maxlen=capacity)

    def add(self, item: MemoryItem) -> str:
        """Add item to working memory"""
        item_id = f"wm_{datetime.now().timestamp()}"
        self.items.append((item_id, item))

        # Auto-consolidate if at capacity
        if len(self.items) >= self.capacity:
            self.consolidate()

        return item_id

    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Retrieve recent items matching query"""
        results = []
        query_lower = query.lower()

        for item_id, item in self.items:
            if query_lower in str(item.content).lower():
                results.append(item)
                if len(results) >= limit:
                    break

        return results

    def consolidate(self) -> int:
        """Move old items to episodic memory"""
        # In real implementation, would move to episodic
        to_consolidate = len(self.items) // 2
        for _ in range(to_consolidate):
            self.items.popleft()
        return to_consolidate

    def get_context_window(self) -> List[MemoryItem]:
        """Get current context window"""
        return [item for _, item in self.items]
```

Create `abstractmemory/components/semantic.py`:
```python
"""
Semantic memory for facts, concepts, and learned knowledge.
Separate from Core (identity) and Episodic (events).
"""

from typing import List, Dict, Set
from datetime import datetime
from collections import defaultdict

from abstractmemory.core.interfaces import IMemoryComponent, MemoryItem


class SemanticMemory(IMemoryComponent):
    """
    Long-term storage of facts and concepts learned over time.
    Only stores validated, recurring knowledge.
    """

    def __init__(self, validation_threshold: int = 3):
        """
        Args:
            validation_threshold: How many times a fact must be observed to be stored
        """
        self.facts: Dict[str, Dict] = {}  # Validated facts
        self.concepts: Dict[str, Set[str]] = {}  # Concept relationships
        self.pending_facts: defaultdict = defaultdict(int)  # Counting occurrences
        self.validation_threshold = validation_threshold

    def add(self, item: MemoryItem) -> str:
        """Add potential fact - only stored after validation"""
        fact_key = str(item.content).lower()

        # Count occurrence
        self.pending_facts[fact_key] += 1

        # Promote to validated facts if threshold met
        if self.pending_facts[fact_key] >= self.validation_threshold:
            fact_id = f"fact_{len(self.facts)}_{datetime.now().timestamp()}"
            self.facts[fact_id] = {
                'content': item.content,
                'confidence': min(1.0, self.pending_facts[fact_key] / 10),  # Confidence grows with repetition
                'first_seen': item.event_time,
                'validated_at': datetime.now(),
                'occurrence_count': self.pending_facts[fact_key]
            }
            # Clear from pending
            del self.pending_facts[fact_key]
            return fact_id

        return ""  # Not yet validated

    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Retrieve validated facts matching query"""
        results = []
        query_lower = query.lower()

        for fact_id, fact in self.facts.items():
            if query_lower in str(fact['content']).lower():
                results.append(MemoryItem(
                    content=fact['content'],
                    event_time=fact['first_seen'],
                    ingestion_time=fact['validated_at'],
                    confidence=fact['confidence'],
                    metadata={'occurrence_count': fact['occurrence_count']}
                ))
                if len(results) >= limit:
                    break

        # Sort by confidence
        return sorted(results, key=lambda x: x.confidence, reverse=True)[:limit]

    def consolidate(self) -> int:
        """Link related facts into concepts"""
        consolidated = 0
        # Group facts by common terms
        for fact_id, fact in self.facts.items():
            words = str(fact['content']).lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    if word not in self.concepts:
                        self.concepts[word] = set()
                    self.concepts[word].add(fact_id)
                    consolidated += 1
        return consolidated

    def get_concept_network(self, concept: str) -> Dict[str, Set[str]]:
        """Get related facts for a concept"""
        if concept.lower() in self.concepts:
            fact_ids = self.concepts[concept.lower()]
            return {
                'concept': concept,
                'facts': [self.facts[fid]['content'] for fid in fact_ids if fid in self.facts]
            }
        return {'concept': concept, 'facts': []}
```

Create `abstractmemory/components/episodic.py`:
```python
"""
Episodic memory for experiences and events.
"""

from typing import List, Dict
from datetime import datetime

from abstractmemory.core.interfaces import IMemoryComponent, MemoryItem
from abstractmemory.core.temporal import TemporalAnchor, TemporalSpan


class EpisodicMemory(IMemoryComponent):
    """Long-term episodic memory with temporal organization"""

    def __init__(self):
        self.episodes = {}  # ID -> Episode
        self.temporal_index = {}  # For temporal queries

    def add(self, item: MemoryItem) -> str:
        """Add episode to memory"""
        episode_id = f"ep_{len(self.episodes)}_{datetime.now().timestamp()}"

        # Create temporal anchor
        anchor = TemporalAnchor(
            event_time=item.event_time,
            ingestion_time=item.ingestion_time,
            validity_span=TemporalSpan(start=item.event_time),
            confidence=item.confidence
        )

        self.episodes[episode_id] = {
            'item': item,
            'anchor': anchor,
            'related': []  # Links to related episodes
        }

        # Update temporal index
        self.temporal_index[episode_id] = anchor

        return episode_id

    def retrieve(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Retrieve episodes matching query"""
        # Simple implementation - would use embeddings in production
        results = []
        query_lower = query.lower()

        for episode in self.episodes.values():
            if query_lower in str(episode['item'].content).lower():
                results.append(episode['item'])
                if len(results) >= limit:
                    break

        return results

    def consolidate(self) -> int:
        """Consolidate similar episodes"""
        # Would implement clustering/summarization
        return 0

    def get_episodes_between(self, start: datetime, end: datetime) -> List[MemoryItem]:
        """Get episodes between times"""
        results = []
        for episode in self.episodes.values():
            if start <= episode['anchor'].event_time <= end:
                results.append(episode['item'])
        return sorted(results, key=lambda x: x.event_time)
```

### 5. Implement Temporal Knowledge Graph (45 min)

Create `abstractmemory/graph/knowledge_graph.py`:
```python
"""
Temporal knowledge graph implementation.
"""

import networkx as nx
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from abstractmemory.core.temporal import TemporalAnchor, TemporalSpan


class TemporalKnowledgeGraph:
    """
    Knowledge graph with bi-temporal modeling.
    Based on Zep/Graphiti architecture.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._node_counter = 0
        self._edge_counter = 0
        self.ontology = {}  # Auto-built ontology

    def add_entity(self, value: str, entity_type: str = 'entity') -> str:
        """Add or get entity node"""
        # Check for existing entity (deduplication)
        for node_id, data in self.graph.nodes(data=True):
            if data.get('value') == value:
                # Update access time
                self.graph.nodes[node_id]['last_accessed'] = datetime.now()
                return node_id

        # Create new entity
        node_id = f"entity_{self._node_counter}"
        self._node_counter += 1

        self.graph.add_node(
            node_id,
            value=value,
            type=entity_type,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance=1.0
        )

        # Update ontology
        if entity_type not in self.ontology:
            self.ontology[entity_type] = []
        self.ontology[entity_type].append(node_id)

        return node_id

    def add_fact(self, subject: str, predicate: str, object: str,
                event_time: datetime, confidence: float = 1.0,
                source: Optional[str] = None) -> str:
        """Add temporally anchored fact"""

        # Get or create nodes
        subj_id = self.add_entity(subject)
        obj_id = self.add_entity(object)

        # Create temporal anchor
        anchor = TemporalAnchor(
            event_time=event_time,
            ingestion_time=datetime.now(),
            validity_span=TemporalSpan(start=event_time),
            confidence=confidence,
            source=source
        )

        # Check for contradictions
        self._handle_contradictions(subj_id, predicate, obj_id, anchor)

        # Add edge with temporal data
        edge_id = f"edge_{self._edge_counter}"
        self._edge_counter += 1

        self.graph.add_edge(
            subj_id, obj_id,
            key=edge_id,
            predicate=predicate,
            anchor=anchor,
            confidence=confidence,
            valid=True
        )

        return edge_id

    def _handle_contradictions(self, subj_id: str, predicate: str,
                              obj_id: str, new_anchor: TemporalAnchor):
        """Handle temporal contradictions"""
        # Check existing edges for contradictions
        for _, _, key, data in self.graph.edges(subj_id, keys=True, data=True):
            if data.get('predicate') == predicate and data.get('valid'):
                old_anchor = data.get('anchor')
                if old_anchor:
                    # Check for temporal overlap
                    if self._has_temporal_overlap(old_anchor, new_anchor):
                        # Invalidate older fact (new info takes precedence)
                        if old_anchor.ingestion_time < new_anchor.ingestion_time:
                            data['valid'] = False
                            old_anchor.validity_span.end = new_anchor.event_time
                            old_anchor.validity_span.valid = False

    def _has_temporal_overlap(self, anchor1: TemporalAnchor,
                             anchor2: TemporalAnchor) -> bool:
        """Check if two anchors have temporal overlap"""
        span1 = anchor1.validity_span
        span2 = anchor2.validity_span

        # If either span has no end, check if starts overlap
        if span1.end is None or span2.end is None:
            return True  # Conservative: assume overlap

        # Check for actual overlap
        return not (span1.end < span2.start or span2.end < span1.start)

    def query_at_time(self, query: str, point_in_time: datetime) -> List[Dict[str, Any]]:
        """Query knowledge state at specific time"""
        results = []

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            anchor = data.get('anchor')
            if not anchor:
                continue

            # Check if fact was known and valid at this time
            if (anchor.ingestion_time <= point_in_time and
                anchor.event_time <= point_in_time and
                data.get('valid', False)):

                # Check if still valid at query time
                if (anchor.validity_span.end is None or
                    anchor.validity_span.end > point_in_time):

                    # Check if matches query
                    if query.lower() in data.get('predicate', '').lower():
                        results.append({
                            'subject': self.graph.nodes[u]['value'],
                            'predicate': data['predicate'],
                            'object': self.graph.nodes[v]['value'],
                            'confidence': data.get('confidence', 1.0),
                            'event_time': anchor.event_time,
                            'source': anchor.source
                        })

        return results

    def get_entity_evolution(self, entity: str, start: datetime,
                            end: datetime) -> List[Dict[str, Any]]:
        """Track how entity's relationships evolved over time"""
        # Find entity node
        entity_id = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('value') == entity:
                entity_id = node_id
                break

        if not entity_id:
            return []

        evolution = []

        # Check all edges involving this entity
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if u == entity_id or v == entity_id:
                anchor = data.get('anchor')
                if anchor and start <= anchor.event_time <= end:
                    evolution.append({
                        'time': anchor.event_time,
                        'type': 'fact_added' if data.get('valid') else 'fact_invalidated',
                        'subject': self.graph.nodes[u]['value'],
                        'predicate': data['predicate'],
                        'object': self.graph.nodes[v]['value']
                    })

        return sorted(evolution, key=lambda x: x['time'])
```

### 7. Create Memory Factory and Main Classes (30 min)

Create `abstractmemory/__init__.py`:
```python
"""
AbstractMemory - Two-tier memory strategy for different agent types.

Simple agents use ScratchpadMemory or BufferMemory.
Complex agents use full TemporalMemory.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
import uuid

from .simple import ScratchpadMemory, BufferMemory
from .core.interfaces import MemoryItem
from .components.core import CoreMemory
from .components.working import WorkingMemory
from .components.semantic import SemanticMemory
from .components.episodic import EpisodicMemory
from .graph.knowledge_graph import TemporalKnowledgeGraph


def create_memory(
    memory_type: Literal["scratchpad", "buffer", "grounded"] = "scratchpad",
    **kwargs
) -> Union[ScratchpadMemory, BufferMemory, 'GroundedMemory']:
    """
    Factory function to create appropriate memory for agent type.

    Args:
        memory_type: Type of memory to create
            - "scratchpad": For ReAct agents and task tools
            - "buffer": For simple chatbots
            - "grounded": For autonomous agents (multi-dimensional memory)

    Examples:
        # For a ReAct agent
        memory = create_memory("scratchpad", max_entries=50)

        # For a simple chatbot
        memory = create_memory("buffer", max_messages=100)

        # For an autonomous assistant with user tracking
        memory = create_memory("grounded", working_capacity=10, enable_kg=True)
        memory.set_current_user("alice", relationship="owner")
    """
    if memory_type == "scratchpad":
        return ScratchpadMemory(**kwargs)
    elif memory_type == "buffer":
        return BufferMemory(**kwargs)
    elif memory_type == "grounded":
        return GroundedMemory(**kwargs)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")


class GroundedMemory:
    """
    Multi-dimensionally grounded memory for autonomous agents.
    Grounds memory in WHO (relational), WHEN (temporal), and WHERE (spatial).

    Memory Architecture:
    - Core: Agent identity and persona (rarely changes)
    - Semantic: Validated facts and concepts (requires recurrence)
    - Working: Current context (transient)
    - Episodic: Event archive (long-term)
    """

    def __init__(self,
                 working_capacity: int = 10,
                 enable_kg: bool = True,
                 storage_backend: Optional[str] = None,
                 default_user_id: str = "default",
                 semantic_threshold: int = 3):
        """Initialize grounded memory system"""

        # Initialize memory components (Four-tier architecture)
        self.core = CoreMemory()  # Agent identity (rarely updated)
        self.semantic = SemanticMemory(validation_threshold=semantic_threshold)  # Validated facts
        self.working = WorkingMemory(capacity=working_capacity)  # Transient context
        self.episodic = EpisodicMemory()  # Event archive

        # Initialize knowledge graph if enabled
        self.kg = TemporalKnowledgeGraph() if enable_kg else None

        # Relational tracking
        self.current_user = default_user_id
        self.user_profiles: Dict[str, Dict] = {}  # User-specific profiles
        self.user_memories: Dict[str, List] = {}  # User-specific memory indices

        # Learning tracking
        self.failure_patterns: Dict[str, int] = {}  # Track repeated failures
        self.success_patterns: Dict[str, int] = {}  # Track successful patterns

        # Core memory update tracking
        self.core_update_candidates: Dict[str, int] = {}  # Track potential core updates
        self.core_update_threshold = 5  # Require 5 occurrences before core update

        # Storage backend
        self.storage = self._init_storage(storage_backend)

    def set_current_user(self, user_id: str, relationship: Optional[str] = None):
        """Set the current user for relational context"""
        self.current_user = user_id

        # Initialize user profile if new
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "first_seen": datetime.now(),
                "relationship": relationship or "unknown",
                "interaction_count": 0,
                "preferences": {},
                "facts": []
            }
            self.user_memories[user_id] = []

    def add_interaction(self, user_input: str, agent_response: str,
                       user_id: Optional[str] = None):
        """Add user-agent interaction with relational grounding"""
        now = datetime.now()
        user_id = user_id or self.current_user

        # Create relational context
        relational = RelationalContext(
            user_id=user_id,
            agent_id="main",
            relationship=self.user_profiles.get(user_id, {}).get("relationship"),
            session_id=str(uuid.uuid4())[:8]
        )

        # Add to working memory with relational context
        user_item = MemoryItem(
            content={
                'role': 'user',
                'text': user_input,
                'user_id': user_id  # Track who said it
            },
            event_time=now,
            ingestion_time=now,
            metadata={'relational': relational.__dict__}
        )
        item_id = self.working.add(user_item)

        # Track in user-specific memory index
        if user_id in self.user_memories:
            self.user_memories[user_id].append(item_id)

        # Update user profile
        if user_id in self.user_profiles:
            self.user_profiles[user_id]["interaction_count"] += 1

        # Add to episodic memory with full context
        episode = MemoryItem(
            content={
                'interaction': {
                    'user': user_input,
                    'agent': agent_response,
                    'user_id': user_id
                }
            },
            event_time=now,
            ingestion_time=now,
            metadata={'relational': relational.__dict__}
        )
        self.episodic.add(episode)

        # Extract facts if KG enabled
        if self.kg:
            self._extract_facts_to_kg(agent_response, now)

    def _extract_facts_to_kg(self, text: str, event_time: datetime):
        """Extract facts from text and add to KG"""
        # Simplified extraction - would use NLP/LLM in production
        # Look for patterns like "X is Y" or "X has Y"
        import re

        patterns = [
            r'(\w+)\s+is\s+(\w+)',
            r'(\w+)\s+has\s+(\w+)',
            r'(\w+)\s+can\s+(\w+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    self.kg.add_fact(
                        subject=match[0],
                        predicate='is' if 'is' in pattern else 'has' if 'has' in pattern else 'can',
                        object=match[1],
                        event_time=event_time
                    )

    def get_full_context(self, query: str, max_items: int = 5,
                        user_id: Optional[str] = None) -> str:
        """Get user-specific context through relational lens"""
        user_id = user_id or self.current_user
        context_parts = []

        # Include user profile if known
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            context_parts.append(f"=== User Profile: {user_id} ===")
            context_parts.append(f"Relationship: {profile['relationship']}")
            context_parts.append(f"Known for: {profile['interaction_count']} interactions")
            if profile.get('facts'):
                context_parts.append(f"Known facts: {', '.join(profile['facts'][:3])}")

        # Always include core memory (agent identity)
        core_context = self.core.get_context()
        if core_context:
            context_parts.append("\n=== Core Memory (Identity) ===")
            context_parts.append(core_context)

        # Include relevant semantic memory (validated facts)
        semantic_facts = self.semantic.retrieve(query, limit=max_items//2)
        if semantic_facts:
            context_parts.append("\n=== Learned Facts ===")
            for fact in semantic_facts:
                context_parts.append(f"- {fact.content} (confidence: {fact.confidence:.2f})")

        # Check for learned failures/successes relevant to query
        for pattern, count in self.failure_patterns.items():
            if query.lower() in pattern.lower() and count >= 2:
                context_parts.append(f"\n⚠️ Warning: Previous failures with similar action ({count} times)")
                break

        # Get from working memory (recent context)
        working_items = self.working.retrieve(query, limit=max_items)
        if working_items:
            context_parts.append("\n=== Recent Context ===")
            for item in working_items:
                if isinstance(item.content, dict):
                    context_parts.append(f"- {item.content.get('text', str(item.content))}")

        # Get from episodic memory (retrieved as needed)
        episodes = self.episodic.retrieve(query, limit=max_items)
        if episodes:
            context_parts.append("\n=== Relevant Episodes ===")
            for episode in episodes:
                context_parts.append(f"- {str(episode.content)[:100]}...")

        # Get from knowledge graph
        if self.kg:
            facts = self.kg.query_at_time(query, datetime.now())
            if facts:
                context_parts.append("\n=== Known Facts ===")
                for fact in facts[:max_items]:
                    context_parts.append(
                        f"- {fact['subject']} {fact['predicate']} {fact['object']}"
                    )

        return "\n\n".join(context_parts) if context_parts else "No relevant context found."

    def retrieve_context(self, query: str, max_items: int = 5) -> str:
        """Backward compatibility wrapper"""
        return self.get_full_context(query, max_items)

    def _init_storage(self, backend: Optional[str]):
        """Initialize storage backend"""
        if backend == 'lancedb':
            from .storage.lancedb import LanceDBStorage
            return LanceDBStorage()
        elif backend == 'file':
            from .storage.file_storage import FileStorage
            return FileStorage()
        return None

    def save(self, path: str):
        """Save memory to disk"""
        if self.storage:
            # Save each component (three-tier architecture)
            self.storage.save(f"{path}/core", self.core)
            self.storage.save(f"{path}/working", self.working)
            self.storage.save(f"{path}/episodic", self.episodic)
            if self.kg:
                self.storage.save(f"{path}/kg", self.kg)

    def load(self, path: str):
        """Load memory from disk"""
        if self.storage and self.storage.exists(path):
            # Load components (three-tier architecture)
            if self.storage.exists(f"{path}/core"):
                self.core = self.storage.load(f"{path}/core")
            self.working = self.storage.load(f"{path}/working")
            self.episodic = self.storage.load(f"{path}/episodic")
            if self.storage.exists(f"{path}/kg"):
                self.kg = self.storage.load(f"{path}/kg")

    def learn_about_user(self, fact: str, user_id: Optional[str] = None):
        """Learn and remember a fact about a specific user"""
        user_id = user_id or self.current_user

        if user_id in self.user_profiles:
            # Add to user's facts
            if 'facts' not in self.user_profiles[user_id]:
                self.user_profiles[user_id]['facts'] = []

            # Avoid duplicates
            if fact not in self.user_profiles[user_id]['facts']:
                self.user_profiles[user_id]['facts'].append(fact)

                # Track for potential core memory update (requires recurrence)
                core_key = f"user:{user_id}:{fact}"
                self.core_update_candidates[core_key] = self.core_update_candidates.get(core_key, 0) + 1

                # Only update core memory after threshold met
                if self.core_update_candidates[core_key] >= self.core_update_threshold:
                    if user_id == self.current_user:
                        current_info = self.core.blocks.get("user_info").content
                        updated_info = f"{current_info}\n- {fact}"
                        self.core.update_block("user_info", updated_info,
                                             f"Validated through recurrence: {fact}")
                        del self.core_update_candidates[core_key]

    def track_failure(self, action: str, context: str):
        """Track a failed action to learn from mistakes"""
        failure_key = f"{action}:{context}"
        self.failure_patterns[failure_key] = self.failure_patterns.get(failure_key, 0) + 1

        # After repeated failures, add to semantic memory as a learned constraint
        if self.failure_patterns[failure_key] >= 3:
            fact = f"Action '{action}' tends to fail in context: {context}"
            self.semantic.add(MemoryItem(
                content=fact,
                event_time=datetime.now(),
                ingestion_time=datetime.now(),
                confidence=0.9,
                metadata={'type': 'learned_constraint', 'failure_count': self.failure_patterns[failure_key]}
            ))

    def track_success(self, action: str, context: str):
        """Track a successful action to reinforce patterns"""
        success_key = f"{action}:{context}"
        self.success_patterns[success_key] = self.success_patterns.get(success_key, 0) + 1

        # After repeated successes, add to semantic memory as a learned strategy
        if self.success_patterns[success_key] >= 3:
            fact = f"Action '{action}' works well in context: {context}"
            self.semantic.add(MemoryItem(
                content=fact,
                event_time=datetime.now(),
                ingestion_time=datetime.now(),
                confidence=0.9,
                metadata={'type': 'learned_strategy', 'success_count': self.success_patterns[success_key]}
            ))

    def consolidate_memories(self):
        """Consolidate working memory to semantic/episodic based on importance"""
        # Get items from working memory
        working_items = self.working.get_context_window()

        for item in working_items:
            # Extract potential facts for semantic memory
            if isinstance(item.content, dict):
                content_text = item.content.get('text', '')
                # Simple heuristic: statements with "is", "are", "means" are potential facts
                if any(word in content_text.lower() for word in ['is', 'are', 'means', 'equals']):
                    self.semantic.add(item)

            # Important items go to episodic memory
            if item.confidence > 0.7 or (item.metadata and item.metadata.get('important')):
                self.episodic.add(item)

        # Consolidate semantic memory concepts
        self.semantic.consolidate()

    def get_user_context(self, user_id: str) -> Optional[Dict]:
        """Get everything we know about a specific user"""
        return self.user_profiles.get(user_id)

    def update_core_memory(self, block_id: str, content: str, reasoning: str = "") -> bool:
        """Agent can update core memory blocks (self-editing capability)"""
        return self.core.update_block(block_id, content, reasoning)

    def get_core_memory_context(self) -> str:
        """Get core memory context for always-accessible facts"""
        return self.core.get_context()


# Export main classes and factory
__all__ = [
    'create_memory',  # Factory function
    'ScratchpadMemory',  # Simple memory for task agents
    'BufferMemory',  # Simple buffer for chatbots
    'GroundedMemory',  # Multi-dimensional memory for autonomous agents
    'MemoryItem',  # Data structure
    'CoreMemory',  # Core memory component (identity)
    'SemanticMemory',  # Semantic memory component (validated facts)
    'WorkingMemory',  # Working memory component (transient)
    'EpisodicMemory',  # Episodic memory component (events)
    'TemporalKnowledgeGraph',  # Knowledge graph
    'RelationalContext'  # For tracking who
]
```

## Validation

### Test the package
```bash
cd /Users/albou/projects/abstractmemory

# Install in development mode
pip install -e .

# Test two-tier memory strategy
python << 'EOF'
from abstractmemory import create_memory
from datetime import datetime

print("=== Testing Two-Tier Memory Strategy ===")

# Test 1: Simple ScratchpadMemory for ReAct Agent
print("\n1. Testing ScratchpadMemory (for ReAct agents):")
scratchpad = create_memory("scratchpad", max_entries=10)

# Simulate ReAct cycle
scratchpad.add_thought("User wants to know about Python memory management")
scratchpad.add_action("search", {"query": "Python garbage collection"})
scratchpad.add_observation("Found information about reference counting and gc module")
scratchpad.add_thought("Should explain both reference counting and cyclic GC")

print(f"Scratchpad has {len(scratchpad)} entries")
print("Recent context:")
print(scratchpad.get_context(last_n=3))

# Test 2: BufferMemory for Simple Chatbot
print("\n2. Testing BufferMemory (for simple chatbots):")
buffer = create_memory("buffer", max_messages=50)

buffer.add_message("user", "What is Python?")
buffer.add_message("assistant", "Python is a high-level programming language")
buffer.add_message("user", "What makes it popular?")
buffer.add_message("assistant", "Python is popular for its simplicity and versatility")

print(f"Buffer has {len(buffer.messages)} messages")
print("Conversation context:")
print(buffer.get_context(last_n=2))

# Test 3: GroundedMemory for Autonomous Agent with User Tracking
print("\n3. Testing GroundedMemory (for autonomous agents):")
grounded = create_memory("grounded", working_capacity=5)

# Set up user context - this is crucial for personalization
print("\n3a. Testing Relational Grounding (WHO):")
grounded.set_current_user("alice", relationship="owner")

# First interaction with Alice
grounded.add_interaction(
    "My name is Alice and I love Python",
    "Nice to meet you, Alice! Python is a great language.",
    user_id="alice"
)
grounded.learn_about_user("loves Python programming", user_id="alice")

# Switch to different user
grounded.set_current_user("bob", relationship="colleague")
grounded.add_interaction(
    "I prefer Java over Python",
    "Java is indeed powerful for enterprise applications.",
    user_id="bob"
)
grounded.learn_about_user("prefers Java", user_id="bob")

# Get user-specific context for Alice
print("\nContext when talking to Alice:")
alice_context = grounded.get_full_context("programming language", user_id="alice")
print(alice_context)

# Get user-specific context for Bob
print("\nContext when talking to Bob:")
bob_context = grounded.get_full_context("programming language", user_id="bob")
print(bob_context)

# Show how the agent adapts based on who it's talking to
print("\n3b. User Profiles:")
print(f"Alice profile: {grounded.get_user_context('alice')}")
print(f"Bob profile: {grounded.get_user_context('bob')}")

# Test 4: Knowledge Graph Integration
print("\n4. Testing Knowledge Graph:")
if memory.kg:
    facts = memory.kg.query_at_time("is", datetime.now())
    print(f"Extracted {len(facts)} facts:")
    for fact in facts:
        print(f"  - {fact['subject']} {fact['predicate']} {fact['object']}")

# Test 4: Learning from Failures
print("\n4. Testing Learning from Failures:")
# Track some failures
grounded.track_failure("search", "no internet connection")
grounded.track_failure("search", "no internet connection")
grounded.track_failure("search", "no internet connection")
# After 3 failures, it should be in semantic memory

# Track some successes
grounded.track_success("calculate", "math problem")
grounded.track_success("calculate", "math problem")
grounded.track_success("calculate", "math problem")

# Check if learned
learned_facts = grounded.semantic.retrieve("search", limit=5)
print(f"Learned {len(learned_facts)} facts about failures/successes")

# Test 5: Memory Consolidation
print("\n5. Testing Memory Consolidation:")
grounded.consolidate_memories()
print(f"✓ Semantic facts after consolidation: {len(grounded.semantic.facts)}")

# Test 6: Memory Hierarchy Validation
print("\n6. Validating Four-Tier Architecture:")
print(f"✓ Core Memory: {len(grounded.core.blocks)} blocks (identity)")
print(f"✓ Semantic Memory: {len(grounded.semantic.facts)} validated facts")
print(f"✓ Working Memory: {len(grounded.working.items)} transient items")
print(f"✓ Episodic Memory: {len(grounded.episodic.episodes)} events")
if grounded.kg:
    print(f"✓ Knowledge Graph: {grounded.kg.graph.number_of_nodes()} entities")

print("\n=== Grounded Memory Test Complete ===")
EOF
```

## Success Criteria

### Two-Tier Memory Strategy Validation
- [ ] **Simple Memory Types** implemented and tested:
  - [ ] ScratchpadMemory for ReAct agents (thought-action-observation)
  - [ ] BufferMemory for simple chatbots (conversation history)
  - [ ] Clean, efficient, no over-engineering
- [ ] **Complex Memory (TemporalMemory)** for autonomous agents:
  - [ ] Core Memory with self-editing capability
  - [ ] Working Memory with sliding window
  - [ ] Episodic Memory with temporal anchoring
  - [ ] Knowledge Graph with bi-temporal model
- [ ] **Memory Factory** (`create_memory`) correctly instantiates based on agent type

### Performance Validation
- [ ] Simple memory operations < 1ms
- [ ] Complex memory retrieval < 100ms for 10k facts
- [ ] Memory selection appropriate to agent purpose
- [ ] No unnecessary overhead for simple agents

### Integration Tests
- [ ] ReAct agent using ScratchpadMemory efficiently
- [ ] Simple chatbot using BufferMemory or BasicSession
- [ ] Autonomous agent using full TemporalMemory
- [ ] Memory type selection based on use case works correctly

## Next Task

Proceed to Task 04: Create Agent Package Structure