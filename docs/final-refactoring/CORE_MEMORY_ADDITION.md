# Critical Addition: Core Memory Component

## Problem Identified
Our current memory design follows a **two-tier system** (Working + Episodic) but SOTA research clearly shows **three-tier systems** are standard:

1. **Core Memory** ← MISSING from our design
2. Working Memory ✅
3. Archival/Episodic Memory ✅

## SOTA Evidence for Core Memory

### MemGPT/Letta (UC Berkeley, 2024)
- **Core Memory**: "Stores personal, always-relevant data"
- **Split into**: Agent persona + User information
- **Self-editable**: Agent can update its own core memory
- **Always accessible**: Never leaves context window

### A-MEM (2024)
- **Structured attributes**: Context, keywords, tags
- **Dynamic linking**: Creates interconnected knowledge networks
- **Memory evolution**: Updates existing memories with new insights

### Mem0 (Production System, 2024)
- **26% higher accuracy** than OpenAI's memory system
- **Dynamic extraction**: Automatically identifies important information
- **Multi-session relationships**: Maintains state across conversations

## Core Memory Design Specification

### Core Memory Properties
```python
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
        # Log the reasoning for transparency

class CoreMemory(IMemoryComponent):
    """Always-accessible core memory for fundamental facts"""

    def __init__(self, max_blocks: int = 10, max_tokens_per_block: int = 200):
        self.blocks: Dict[str, CoreMemoryBlock] = {}
        self.max_blocks = max_blocks
        self.max_tokens_per_block = max_tokens_per_block

        # Initialize default blocks
        self.blocks["persona"] = CoreMemoryBlock(
            block_id="persona",
            label="persona",
            content="I am an AI assistant with memory capabilities.",
            last_updated=datetime.now()
        )
        self.blocks["user_info"] = CoreMemoryBlock(
            block_id="user_info",
            label="user_info",
            content="User information will be learned over time.",
            last_updated=datetime.now()
        )

    def get_context(self) -> str:
        """Get all core memory as context string"""
        context_parts = []
        for block in self.blocks.values():
            context_parts.append(f"[{block.label}] {block.content}")
        return "\n".join(context_parts)

    def update_block(self, block_id: str, content: str, reasoning: str) -> bool:
        """Agent updates a core memory block"""
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
```

### Integration with Existing Architecture

**Add to components/ folder:**
```python
# abstractmemory/components/core.py
```

**Update main TemporalMemory class:**
```python
class TemporalMemory:
    def __init__(self, working_capacity: int = 10, ...):
        # Existing components
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory()
        self.kg = TemporalKnowledgeGraph() if enable_kg else None

        # NEW: Core memory component
        self.core = CoreMemory()  # Always accessible

    def get_full_context(self, query: str) -> str:
        """Get complete context including core memory"""
        context_parts = []

        # Always include core memory first
        core_context = self.core.get_context()
        if core_context:
            context_parts.append("=== Core Memory ===")
            context_parts.append(core_context)

        # Add other context as before
        working_context = self.retrieve_context(query)
        if working_context:
            context_parts.append("\n=== Recent Context ===")
            context_parts.append(working_context)

        return "\n\n".join(context_parts)
```

## Memory Flow with Core Memory

```
User Interaction
       ↓
1. Core Memory (always included in prompt)
       ↓
2. Working Memory (recent context)
       ↓
3. Episodic/Archival (retrieved as needed)
       ↓
   LLM Response
       ↓
4. Core Memory Update (if agent decides to update facts about user/self)
```

## Why This Wasn't a Design Choice

Looking at the original `task_03_create_memory_package.md`, it appears **Core Memory was overlooked**, not intentionally excluded:

1. **No discussion** of why core memory was excluded
2. **Direct jump** from interfaces to working memory
3. **Missing SOTA comparison** with MemGPT/Letta standards
4. **No mention** of agent persona or user information storage

This appears to be a **design gap** rather than a conscious decision.

## Implementation Priority

**HIGH PRIORITY** - Add before implementing the memory package:

1. **Add Core Memory component** to the design
2. **Update TemporalMemory** to include core memory
3. **Test core memory persistence** and self-editing
4. **Validate SOTA compliance** with MemGPT patterns

## Validation Against SOTA

With Core Memory added, our architecture becomes:

| Component | Our Design | MemGPT | A-MEM | Mem0 | Status |
|-----------|------------|---------|-------|------|---------|
| Core Memory | ✅ CoreMemory | ✅ Core | ✅ Structured | ✅ Consolidated | ALIGNED |
| Working Memory | ✅ WorkingMemory | ✅ Chat History | ✅ Recent | ✅ Session | ALIGNED |
| Long-term Memory | ✅ EpisodicMemory | ✅ Archival | ✅ Historical | ✅ Multi-session | ALIGNED |
| Temporal Model | ✅ Bi-temporal | ❌ Basic | ✅ Evolution | ✅ Dynamic | SUPERIOR |

## Conclusion

Adding Core Memory makes our design **SOTA-compliant** and provides the always-accessible foundational memory that agents need for consistent identity and user relationship management.

---

*Priority: CRITICAL - Add before memory package implementation*
*Evidence: MemGPT, A-MEM, Mem0 all include core memory as essential component*
*Status: Design gap identified and solution provided*