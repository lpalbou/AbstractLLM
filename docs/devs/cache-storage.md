# LanceDB Observability System Implementation Plan
## For AbstractLLM with RAG Capabilities

**Goal**: Build a unified observability system using LanceDB that combines SQL power with vector search for RAG, enabling time-based queries and user tracking.

---

## 🎯 Core Requirements

1. **Time-based Search**: Query conversations by exact timeframe (e.g., "Sept 15, 2025 10:00-12:15")
2. **User Management**: Track user_id with reference to users table
3. **RAG Search**: Semantic search over verbatim LLM contexts using embeddings
4. **SQL Power**: Filter, aggregate, and analyze with SQL syntax
5. **Complete Observability**: Store and search all verbatim contexts, facts, and ReAct cycles

---

## 🏗️ Architecture Overview

```
~/.abstractllm/
├── lancedb/                     # LanceDB data directory
│   ├── registry.lance/          # Global registry table
│   ├── users.lance/             # Users table
│   ├── sessions.lance/          # Sessions table
│   ├── interactions.lance/      # Interactions with embeddings
│   └── react_cycles.lance/      # ReAct scratchpad data
└── embeddings/
    └── model_cache/             # Cached embedding model
```

---

## 📊 Database Schema

### 1. Users Table
```python
{
    "user_id": str,           # Primary key (UUID)
    "username": str,          # Unique username
    "created_at": datetime,
    "metadata": dict          # Extensible user metadata
}
```

### 2. Sessions Table
```python
{
    "session_id": str,        # Primary key (UUID)
    "user_id": str,           # Foreign key to users
    "created_at": datetime,
    "last_active": datetime,
    "provider": str,          # ollama, openai, etc.
    "model": str,             # Model identifier
    "temperature": float,
    "max_tokens": int,
    "seed": int,
    "system_prompt": str,
    "metadata": dict
}
```

### 3. Interactions Table (with embeddings)
```python
{
    "interaction_id": str,    # Primary key
    "session_id": str,        # Foreign key to sessions
    "user_id": str,           # Foreign key to users
    "timestamp": datetime,    # Exact timestamp for time queries
    "query": str,             # User query
    "response": str,          # LLM response
    "context_verbatim": str,  # Full verbatim context
    "context_embedding": Vector(1536),  # Embedding for RAG
    "facts_extracted": list,
    "token_usage": dict,
    "duration_ms": int,
    "metadata": dict
}
```

### 4. ReAct Cycles Table
```python
{
    "react_id": str,          # Primary key
    "interaction_id": str,    # Foreign key to interactions
    "timestamp": datetime,
    "scratchpad": str,        # Full scratchpad JSON
    "scratchpad_embedding": Vector(1536),  # For searching reasoning
    "steps": list,
    "success": bool,
    "metadata": dict
}
```

---

## 🚀 Implementation Steps

### Phase 1: Core Setup (30 mins)

#### Step 1.1: Install Dependencies
```bash
pip install lancedb sentence-transformers numpy pandas pyarrow
```

#### Step 1.2: Create Database Manager
Create `abstractllm/storage/lancedb_store.py`:

```python
import lancedb
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import uuid

class ObservabilityStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.home() / ".abstractllm"
        self.db_path = self.base_dir / "lancedb"
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self):
        """Initialize all tables if they don't exist"""
        # Users table
        if "users" not in self.db.table_names():
            self.db.create_table("users", data=[{
                "user_id": "system",
                "username": "system",
                "created_at": datetime.now(),
                "metadata": {}
            }])

        # Sessions table
        if "sessions" not in self.db.table_names():
            self.db.create_table("sessions", data=[{
                "session_id": str(uuid.uuid4()),
                "user_id": "system",
                "created_at": datetime.now(),
                "last_active": datetime.now(),
                "provider": "system",
                "model": "none",
                "temperature": 0.0,
                "max_tokens": 0,
                "seed": 0,
                "system_prompt": "",
                "metadata": {}
            }])
```

### Phase 2: Embedding Integration (20 mins)

#### Step 2.1: Create Embedding Manager
Create `abstractllm/storage/embeddings.py`:

```python
from sentence_transformers import SentenceTransformer
from functools import lru_cache

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Use small, fast model for low resource usage
        self.model = SentenceTransformer(model_name)

    @lru_cache(maxsize=1000)
    def embed_text(self, text: str) -> list:
        """Cache embeddings for repeated text"""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[list]:
        """Batch embedding for efficiency"""
        return self.model.encode(texts).tolist()
```

### Phase 3: Core Operations (30 mins)

#### Step 3.1: Time-based Search
Add to `lancedb_store.py`:

```python
def search_by_timeframe(self,
                        start_time: datetime,
                        end_time: datetime,
                        user_id: Optional[str] = None) -> pd.DataFrame:
    """Search interactions within timeframe"""
    table = self.db.open_table("interactions")

    # Build SQL-like query
    query = f"timestamp >= '{start_time}' AND timestamp <= '{end_time}'"
    if user_id:
        query += f" AND user_id = '{user_id}'"

    return table.search().where(query).to_pandas()
```

#### Step 3.2: RAG Search
```python
def semantic_search(self,
                    query: str,
                    limit: int = 10,
                    filters: Optional[Dict] = None) -> List[Dict]:
    """Search by semantic similarity with SQL filters"""
    query_embedding = self.embedder.embed_text(query)
    table = self.db.open_table("interactions")

    # Start with vector search
    search = table.search(query_embedding).limit(limit)

    # Apply SQL filters if provided
    if filters:
        if "user_id" in filters:
            search = search.where(f"user_id = '{filters['user_id']}'")
        if "start_time" in filters:
            search = search.where(f"timestamp >= '{filters['start_time']}'")
        if "end_time" in filters:
            search = search.where(f"timestamp <= '{filters['end_time']}'")

    return search.to_list()
```

### Phase 4: Session Integration (20 mins)

#### Step 4.1: Hook into AbstractLLM Session
Update `abstractllm/session.py` (add at line ~230):

```python
# Import at top
from abstractllm.storage.lancedb_store import ObservabilityStore
from abstractllm.storage.embeddings import EmbeddingManager

# In __init__
self.lance_store = ObservabilityStore()
self.embedder = EmbeddingManager()

# In _save_interaction (around line 850)
def _save_interaction(self, query, response, cycle_data):
    # Existing code...

    # Add to LanceDB
    interaction_data = {
        "interaction_id": self.current_cycle.cycle_id,
        "session_id": self.id,
        "user_id": getattr(self, 'user_id', 'default'),
        "timestamp": datetime.now(),
        "query": query,
        "response": str(response.content) if hasattr(response, 'content') else str(response),
        "context_verbatim": self._capture_verbatim_context(),
        "context_embedding": self.embedder.embed_text(self._capture_verbatim_context()),
        "facts_extracted": cycle_data.get('facts_extracted', []),
        "token_usage": response.usage if hasattr(response, 'usage') else {},
        "duration_ms": int(cycle_data.get('reasoning_time', 0) * 1000),
        "metadata": {}
    }

    self.lance_store.add_interaction(interaction_data)
```

### Phase 5: Command Interface (15 mins)

#### Step 5.1: Add Commands
Update `abstractllm/utils/commands.py` (add new commands):

```python
def cmd_search(self, *args):
    """Search interactions using natural language
    Usage: /search <query> [--user <id>] [--from <date>] [--to <date>]"""

    if not args:
        return "Usage: /search <query> [--user <id>] [--from <date>] [--to <date>]"

    # Parse arguments
    query = []
    filters = {}
    i = 0
    while i < len(args):
        if args[i] == '--user' and i + 1 < len(args):
            filters['user_id'] = args[i + 1]
            i += 2
        elif args[i] == '--from' and i + 1 < len(args):
            filters['start_time'] = datetime.fromisoformat(args[i + 1])
            i += 2
        elif args[i] == '--to' and i + 1 < len(args):
            filters['end_time'] = datetime.fromisoformat(args[i + 1])
            i += 2
        else:
            query.append(args[i])
            i += 1

    query_text = ' '.join(query)
    results = self.session.lance_store.semantic_search(query_text, filters=filters)

    # Display results
    for r in results[:5]:
        print(f"\n📅 {r['timestamp']} | User: {r['user_id']}")
        print(f"Q: {r['query'][:100]}...")
        print(f"Similarity: {r['_distance']:.3f}")

def cmd_timeframe(self, *args):
    """Query by exact timeframe
    Usage: /timeframe 2025-09-15T10:00 2025-09-15T12:15 [user_id]"""

    if len(args) < 2:
        return "Usage: /timeframe <start> <end> [user_id]"

    start = datetime.fromisoformat(args[0])
    end = datetime.fromisoformat(args[1])
    user_id = args[2] if len(args) > 2 else None

    results = self.session.lance_store.search_by_timeframe(start, end, user_id)

    print(f"\nFound {len(results)} interactions:")
    for _, r in results.iterrows():
        print(f"  • {r['timestamp']} | {r['interaction_id'][:8]} | {r['query'][:50]}...")
```

### Phase 6: Migration (10 mins)

#### Step 6.1: Create Migration Script
Create `scripts/migrate_to_lancedb.py`:

```python
#!/usr/bin/env python
"""Migrate existing SQLite data to LanceDB"""

import sqlite3
import json
from pathlib import Path
from abstractllm.storage.lancedb_store import ObservabilityStore
from abstractllm.storage.embeddings import EmbeddingManager

def migrate():
    print("Starting migration to LanceDB...")

    # Initialize stores
    lance = ObservabilityStore()
    embedder = EmbeddingManager()

    # Find SQLite databases
    sessions_dir = Path.home() / ".abstractllm" / "sessions"

    migrated = 0
    for session_dir in sessions_dir.iterdir():
        db_path = session_dir / "observability.db"
        if db_path.exists():
            print(f"Migrating {session_dir.name}...")

            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT * FROM interactions")

            for row in cursor:
                # Convert and add embedding
                interaction = {
                    "interaction_id": row[0],
                    "context_verbatim": row[1],
                    "context_embedding": embedder.embed_text(row[1] if row[1] else ""),
                    # ... map other fields
                }
                lance.add_interaction(interaction)
                migrated += 1

            conn.close()

    print(f"✅ Migrated {migrated} interactions to LanceDB")

if __name__ == "__main__":
    migrate()
```

---

## 🎮 Usage Examples

### Time-based Query
```bash
# Find all interactions between specific times
/timeframe 2025-09-15T10:00:00 2025-09-15T12:15:00 alice

# Results:
Found 23 interactions:
  • 2025-09-15 10:03:21 | ab9d1848 | How do I implement caching...
  • 2025-09-15 10:15:47 | cd3e2959 | The cache isn't working...
```

### Semantic Search with Filters
```bash
# Search for debugging discussions from a specific user
/search debugging cache problems --user alice --from 2025-09-01

# Results:
📅 2025-09-15 10:15:47 | User: alice
Q: The cache isn't working properly when...
Similarity: 0.892
```

### RAG-powered Analysis
```bash
# Find similar past conversations
/similar "how to optimize database queries"

# Automatically finds semantically similar interactions
# regardless of exact wording
```

---

## 📈 Performance Optimizations

1. **Batch Operations**: Always batch embeddings and inserts
2. **Caching**: Use LRU cache for frequently accessed embeddings
3. **Indexes**: LanceDB automatically creates vector indexes
4. **Compression**: Store large text fields compressed
5. **Partitioning**: Partition by date for faster time-range queries

---

## 🔒 Security Considerations

1. **User Isolation**: Always filter by user_id in multi-user scenarios
2. **Embedding Privacy**: Embeddings can leak information - store securely
3. **Access Control**: Implement user permission checks
4. **Data Retention**: Implement automatic cleanup of old data

---

## 📊 Monitoring & Maintenance

```python
# Add to lancedb_store.py
def get_stats(self) -> Dict:
    """Get storage statistics"""
    stats = {}
    for table_name in self.db.table_names():
        table = self.db.open_table(table_name)
        stats[table_name] = {
            "count": len(table),
            "size_mb": table.to_pandas().memory_usage(deep=True).sum() / 1024 / 1024
        }
    return stats

def cleanup_old_data(self, days: int = 30):
    """Remove data older than N days"""
    cutoff = datetime.now() - timedelta(days=days)

    table = self.db.open_table("interactions")
    # LanceDB doesn't support DELETE yet, so recreate table
    df = table.to_pandas()
    df_filtered = df[df['timestamp'] > cutoff]

    self.db.drop_table("interactions")
    self.db.create_table("interactions", data=df_filtered)

    return len(df) - len(df_filtered)  # Number deleted
```

---

## ✅ Validation Checklist

After implementation, verify:

- [ ] Can query by exact timeframe (e.g., Sept 15, 10:00-12:15)
- [ ] User filtering works correctly
- [ ] Semantic search returns relevant results
- [ ] Embeddings are cached efficiently
- [ ] Migration from SQLite successful
- [ ] Commands work in CLI (`/search`, `/timeframe`)
- [ ] Performance acceptable (< 100ms for most queries)

---

## 🚦 Quick Start for AI Implementation

1. **Start Here**: Create `abstractllm/storage/lancedb_store.py` with Phase 1 code
2. **Test Connection**: Run `python -c "from abstractllm.storage.lancedb_store import ObservabilityStore; store = ObservabilityStore(); print('✅ Connected')"`
3. **Add Embeddings**: Create `abstractllm/storage/embeddings.py` from Phase 2
4. **Integrate**: Add 5 lines to `session.py` as shown in Phase 4
5. **Test Commands**: Add commands from Phase 5 and test with `/search test`
6. **Migrate**: Run migration script if you have existing data

---

## 📝 Notes

- LanceDB is embedded (like SQLite) - no server needed
- Supports both SQL and vector operations in one query
- Automatically handles indexing for both types
- Can scale to billions of records
- Python-native with zero external dependencies for core functionality

This implementation provides complete observability with RAG capabilities while remaining simple and maintainable.