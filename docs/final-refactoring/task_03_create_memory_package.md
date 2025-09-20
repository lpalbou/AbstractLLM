# Task 03: Create Memory Package Structure (Priority 2)

**Duration**: 3 hours
**Risk**: Medium
**Dependencies**: Task 02 completed

## Objectives
- Create AbstractMemory package structure
- Implement temporal knowledge graph
- Integrate cognitive enhancements
- Set up storage backends

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

### 3. Implement Temporal Anchoring (45 min)

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
class TemporalAnchor:
    """Bi-temporal anchor for facts and events"""
    event_time: datetime        # When it happened
    ingestion_time: datetime    # When we learned about it
    validity_span: TemporalSpan # When it was/is valid
    confidence: float = 1.0
    source: Optional[str] = None


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

### 4. Implement Memory Components (45 min)

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

### 6. Create Main Memory Class (30 min)

Create `abstractmemory/__init__.py`:
```python
"""
AbstractMemory - Temporal knowledge graph memory for LLM agents.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from .core.interfaces import MemoryItem
from .components.working import WorkingMemory
from .components.episodic import EpisodicMemory
from .graph.knowledge_graph import TemporalKnowledgeGraph


class TemporalMemory:
    """
    Main memory system combining all components.
    """

    def __init__(self,
                 working_capacity: int = 10,
                 enable_kg: bool = True,
                 storage_backend: Optional[str] = None):
        """Initialize temporal memory system"""

        # Initialize components
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory()

        # Initialize knowledge graph if enabled
        self.kg = TemporalKnowledgeGraph() if enable_kg else None

        # Storage backend
        self.storage = self._init_storage(storage_backend)

    def add_interaction(self, user_input: str, agent_response: str):
        """Add user-agent interaction to memory"""
        now = datetime.now()

        # Add to working memory
        user_item = MemoryItem(
            content={'role': 'user', 'text': user_input},
            event_time=now,
            ingestion_time=now
        )
        self.working.add(user_item)

        # Add to episodic memory
        episode = MemoryItem(
            content={'interaction': {'user': user_input, 'agent': agent_response}},
            event_time=now,
            ingestion_time=now
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

    def retrieve_context(self, query: str, max_items: int = 5) -> str:
        """Retrieve relevant context for query"""
        context_parts = []

        # Get from working memory
        working_items = self.working.retrieve(query, limit=max_items)
        if working_items:
            context_parts.append("Recent context:")
            for item in working_items:
                if isinstance(item.content, dict):
                    context_parts.append(f"- {item.content.get('text', str(item.content))}")

        # Get from episodic memory
        episodes = self.episodic.retrieve(query, limit=max_items)
        if episodes:
            context_parts.append("\nRelevant episodes:")
            for episode in episodes:
                context_parts.append(f"- {str(episode.content)[:100]}...")

        # Get from knowledge graph
        if self.kg:
            facts = self.kg.query_at_time(query, datetime.now())
            if facts:
                context_parts.append("\nKnown facts:")
                for fact in facts[:max_items]:
                    context_parts.append(
                        f"- {fact['subject']} {fact['predicate']} {fact['object']}"
                    )

        return "\n".join(context_parts) if context_parts else "No relevant context found."

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
            # Save each component
            self.storage.save(f"{path}/working", self.working)
            self.storage.save(f"{path}/episodic", self.episodic)
            if self.kg:
                self.storage.save(f"{path}/kg", self.kg)

    def load(self, path: str):
        """Load memory from disk"""
        if self.storage and self.storage.exists(path):
            # Load components
            self.working = self.storage.load(f"{path}/working")
            self.episodic = self.storage.load(f"{path}/episodic")
            if self.storage.exists(f"{path}/kg"):
                self.kg = self.storage.load(f"{path}/kg")


# Export main classes
__all__ = ['TemporalMemory', 'MemoryItem', 'TemporalKnowledgeGraph']
```

## Validation

### Test the package
```bash
cd /Users/albou/projects/abstractmemory

# Install in development mode
pip install -e .

# Test basic functionality
python << 'EOF'
from abstractmemory import TemporalMemory
from datetime import datetime

# Create memory
memory = TemporalMemory(working_capacity=5)

# Add interaction
memory.add_interaction(
    "What is Python?",
    "Python is a programming language. Python has dynamic typing."
)

# Retrieve context
context = memory.retrieve_context("Python")
print("Retrieved context:")
print(context)

# Check knowledge graph
if memory.kg:
    facts = memory.kg.query_at_time("is", datetime.now())
    print(f"\nExtracted {len(facts)} facts")
    for fact in facts:
        print(f"  - {fact}")
EOF
```

## Success Criteria

- [ ] Package structure created
- [ ] Temporal anchoring implemented
- [ ] Working memory functional
- [ ] Episodic memory functional
- [ ] Knowledge graph with bi-temporal model
- [ ] Basic fact extraction working
- [ ] Context retrieval working

## Next Task

Proceed to Task 04: Create Agent Package Structure