# AbstractLLM Final Architecture - Definitive Version

## Executive Summary

After reviewing all analysis documents, the final architecture consists of three packages:
1. **AbstractLLM**: Core LLM abstraction with tools, media, and basic session
2. **AbstractMemory**: Temporal knowledge graph with cognitive enhancements
3. **AbstractAgent**: Single agent orchestration with ReAct and CLI

## Package 1: AbstractLLM (Core LLM Platform)

### Purpose
Unified interface to all LLM providers with essential infrastructure for tools and media handling.

### Size
~8,000 LOC (reduced from current ~15,000)

### Contains

```
abstractllm/
├── core/
│   ├── interface.py         # AbstractLLMInterface (ABC)
│   ├── factory.py           # create_llm() entry point
│   ├── types.py             # GenerateResponse, Message
│   ├── enums.py             # ModelParameter, ModelCapability
│   └── exceptions.py        # Error hierarchy
│
├── providers/               # ALL 6 providers
│   ├── base.py             # BaseProvider with event support
│   ├── openai.py           # OpenAI implementation
│   ├── anthropic.py        # Anthropic implementation
│   ├── ollama.py           # Ollama implementation
│   ├── huggingface.py      # HuggingFace implementation
│   ├── mlx_provider.py     # Apple Silicon optimization
│   └── lmstudio_provider.py # LM Studio local server
│
├── architectures/           # Model detection & capabilities
│   ├── detection.py        # Architecture detection
│   ├── capabilities.py     # Model capabilities
│   └── formats.py          # Message formatting
│
├── tools/                   # ESSENTIAL - each provider different
│   ├── core.py             # ToolDefinition, ToolCall, ToolResult
│   ├── handler.py          # UniversalToolHandler
│   ├── parser.py           # Architecture-aware parsing
│   └── registry.py         # Tool registration
│
├── media/                   # ESSENTIAL - provider-specific
│   ├── processor.py        # MediaProcessor facade
│   ├── image.py            # Image handling (base64, URLs, paths)
│   ├── text.py             # Text/document processing
│   └── tabular.py          # CSV/TSV support
│
├── session.py              # BasicSession - 500 lines MAX
│                           # Core: add_message, get_messages, generate_with_tools
│                           # Executes single tool calls, emits events for observability
│
├── events/                 # Extensibility without coupling
│   ├── bus.py             # EventBus implementation
│   └── types.py           # Event type definitions (includes tool execution events)
│
└── utils/
    ├── logging.py         # Telemetry with verbatim capture
    ├── config.py          # Configuration management
    └── tokenizer.py       # Token counting

```

### Why Tools & Media MUST Be in Core

**Evidence from investigation:**
- OpenAI: `tools` parameter with `{"type": "function", "function": {...}}`
- Anthropic: XML format `<tool_call>...</tool_call>` in content
- Ollama: Architecture-specific (`<|tool_call|>` for Qwen, `<function_call>` for Llama)
- MLX/HuggingFace: Prompted mode with format detection

**Conclusion**: This complexity MUST be abstracted at the core level.

## Package 2: AbstractMemory (Temporal Knowledge Graph)

### Purpose
Sophisticated memory system with temporal anchoring and cognitive enhancements.

### Size
~6,000 LOC

### Contains

```
abstractmemory/
├── core/
│   ├── interfaces.py       # Memory interfaces (IMemory, IRetriever)
│   ├── temporal.py         # Bi-temporal data model
│   └── retrieval.py        # Hybrid retrieval strategies
│
├── components/
│   ├── base.py            # BaseMemoryComponent
│   ├── working.py         # WorkingMemory (10-item sliding window)
│   ├── episodic.py        # EpisodicMemory (events with timestamps)
│   └── semantic.py        # SemanticMemory (facts and relations)
│
├── graph/
│   ├── knowledge_graph.py # TemporalKnowledgeGraph main class
│   ├── nodes.py           # Entity, Fact, Event, Concept nodes
│   ├── edges.py           # Temporal, Causal, Semantic edges
│   └── ontology.py        # Auto-built ontology management
│
├── cognitive/              # Cognitive enhancements
│   ├── extractor.py       # Semantic triple extraction
│   ├── summarizer.py      # Event summarization
│   ├── values.py          # Value alignment tracking
│   └── integration.py     # Integration with memory components
│
└── storage/
    ├── interfaces.py      # IStorage interface
    ├── serialization.py   # Customizable serialization
    ├── file_storage.py    # File-based persistence
    └── lancedb.py         # LanceDB (SQL filtering + embeddings)
```

### Temporal KG Architecture (Based on SOTA)

```python
class TemporalKnowledgeGraph:
    """Based on Zep/Graphiti research"""

    def add_fact(self, subject, predicate, object,
                 event_time,      # When it happened
                 ingestion_time): # When we learned it
        # Bi-temporal anchoring
        pass

    def query_at_time(self, query, point_in_time):
        # Reconstruct knowledge state at any point
        pass

    def get_evolution(self, entity, start_time, end_time):
        # Track how knowledge evolved
        pass
```

### Why Cognitive Goes with Memory
**Evidence**: Cognitive modules are memory adapters, not standalone features
- `CognitiveMemoryAdapter` enhances fact extraction
- Summarizer reduces episodic memory
- Value tracking aligns with semantic memory

## Package 3: AbstractAgent (Single Agent Orchestration)

### Purpose
Orchestrate LLM + Memory for autonomous agent behavior.

### Size
~7,000 LOC

### Contains

```
abstractagent/
├── agent.py                # Main Agent class (replaces complex Session)
│
├── orchestration/          # Single agent coordination
│   ├── coordinator.py      # Coordinates LLM, memory, tools
│   ├── context_manager.py  # Context window management
│   ├── tool_executor.py    # Advanced tool execution
│   └── state_machine.py    # Agent state management
│
├── reasoning/              # Reasoning patterns
│   ├── react.py           # ReAct implementation
│   ├── scratchpad.py      # Reasoning trace storage
│   ├── plan_execute.py    # Plan-and-execute pattern
│   └── chain_of_thought.py # CoT reasoning
│
├── workflows/              # Single-agent workflows
│   ├── pipelines.py       # Sequential processing
│   ├── branches.py        # Conditional logic
│   ├── loops.py           # Iterative refinement
│   └── patterns.py        # Workflow patterns
│
├── strategies/            # Response strategies
│   ├── retry.py          # Retry with exponential backoff
│   ├── structured.py     # Structured output validation
│   ├── validation.py     # Response validation
│   └── fallback.py       # Fallback strategies
│
├── tools/                 # Advanced agent-specific tools
│   ├── code_intelligence.py  # Code analysis tools
│   ├── web_tools.py          # Web search, scraping
│   ├── data_tools.py         # Data processing
│   └── catalog.py            # Tool discovery
│
└── cli/                   # CLI for agent interaction
    ├── alma.py           # Main CLI entry point
    ├── commands/         # Command processors
    │   ├── memory.py    # Memory commands
    │   ├── tools.py     # Tool commands
    │   └── session.py   # Session commands
    └── display.py        # Terminal UI components
```

### Why ReAct Goes Here (Not in Core or Memory)

**SOTA Evidence**:
- LangChain: ReAct is an agent behavior, implements in agent layer
- LlamaIndex: ReActAgent separate from memory components
- Research papers: ReAct is orchestration pattern

**Separation of Concerns**:
- **AbstractLLM Core**: Executes single tool calls, emits events for observability
- **AbstractAgent**: Orchestrates multiple LLM calls in ReAct cycles, handles complex retry/fallback logic
- **AbstractMemory**: Stores persistent facts and experiences

**Technical Reason**: ReAct generates reasoning traces (temporary), Memory stores facts (persistent)

### Agent vs Multi-Agent

**This Package**: Single agent orchestration
**Future Package**: AbstractSwarm for multi-agent (if needed)

```python
# AbstractAgent - single agent
agent = Agent(llm_config={...}, memory_config={...})
response = agent.chat("Hello")

# Future AbstractSwarm - multi-agent
swarm = Swarm(agents=[agent1, agent2, agent3])
result = swarm.collaborate("Complex task")
```

## Session Handling & Chat History

### Current Monolithic Session (4,097 lines)
- Core conversation: 500 lines
- Memory integration: 800 lines
- Tool orchestration: 1,200 lines
- ReAct reasoning: 600 lines
- Everything else: 997 lines

### New Architecture

**In AbstractLLM**: `BasicSession` (500 lines max)
```python
class BasicSession:
    """Simple conversation tracking only"""

    def __init__(self, provider):
        self.provider = provider
        self.messages = []

    def add_message(self, role, content):
        self.messages.append(Message(role, content))

    def generate(self, prompt, **kwargs):
        response = self.provider.generate(prompt, messages=self.messages)
        self.add_message('user', prompt)
        self.add_message('assistant', response.content)
        return response
```

**In AbstractAgent**: `Agent` (replaces complex Session)
```python
class Agent:
    """Full agent capabilities"""

    def __init__(self, llm_config, memory_config=None):
        self.llm = create_llm(**llm_config)
        self.session = BasicSession(self.llm)  # Uses simple session
        self.memory = TemporalMemory(**memory_config)
        self.orchestrator = Coordinator(self)

    def chat(self, prompt):
        # Full orchestration with memory, tools, reasoning
        pass
```

## Critical Design Decisions Summary

| Decision | Choice | Justification |
|----------|--------|--------------|
| Tools in core? | YES | Each provider completely different format |
| Media in core? | YES | Provider-specific handling (base64, URLs, paths) |
| Memory separate? | YES | Complex temporal KG deserves own package |
| ReAct placement? | Agent | It's orchestration, not memory storage |
| Cognitive placement? | Memory | Enhances memory components |
| Multi-agent? | NO | Future AbstractSwarm if needed |
| CLI placement? | Agent | For agent development/testing |

## Migration Path

### Compatibility Layer
```python
# abstractllm/__init__.py (monolithic package)
import warnings

try:
    # Try new packages
    from abstractagent import Agent as Session
    from abstractmemory import TemporalMemory as HierarchicalMemory
    warnings.warn("Using compatibility layer - please migrate", DeprecationWarning)
except ImportError:
    # Fall back to monolithic
    from .session import Session
    from .memory import HierarchicalMemory
```

### Usage Examples

**Simple LLM calls (no changes)**:
```python
from abstractllm import create_llm
llm = create_llm('openai')
response = llm.generate("Hello")
```

**Conversations (minor change)**:
```python
# Old
from abstractllm import Session
session = Session(provider='openai')

# New
from abstractllm import create_llm, BasicSession
llm = create_llm('openai')
session = BasicSession(llm)
```

**Full agent (clear separation)**:
```python
# Old
from abstractllm import Session
session = Session(provider='openai', enable_memory=True)

# New
from abstractagent import Agent
agent = Agent(
    llm_config={'provider': 'openai'},
    memory_config={'temporal': True}
)
```

## Success Metrics

- [ ] session.py < 500 lines (from 4,097)
- [ ] No file > 1,000 lines
- [ ] Zero circular dependencies
- [ ] Import time < 200ms for core
- [ ] All existing tests pass
- [ ] Clean dependency graph