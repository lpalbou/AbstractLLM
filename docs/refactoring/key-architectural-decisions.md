# Key Architectural Decisions

*Based on deep code investigation and testing*

## Core Principle: AbstractLLM Unifies Provider Complexity

After thorough investigation, the fundamental role of AbstractLLM is to **abstract away the complexity** of different LLM providers behind a unified interface. This includes tools, media, and communication patterns - all of which vary dramatically between providers.

## Critical Decisions & Justifications

### 1. Tools MUST Be in Core AbstractLLM

**Evidence from Code Investigation**:
```python
# OpenAI: Native API with function objects
tools=[{"type": "function", "function": {...}}]

# Anthropic: XML format in messages
content="<tool_call>...</tool_call>"

# Ollama/Qwen: Special tokens
content="<|tool_call|>{...}</|tool_call|>"

# Ollama/Llama: XML-like format
content="<function_call>{...}</function_call>"
```

**Decision**: Tools stay in AbstractLLM core because:
- Each provider has completely different tool formats
- Architecture detection determines format selection
- BaseProvider already has `UniversalToolHandler`
- This complexity MUST be abstracted for users

### 2. Media Handling Is Essential Infrastructure

**Evidence from Code Investigation**:
```python
# OpenAI: content array with image_url objects
"content": [{"type": "image_url", "image_url": {"url": "..."}}]

# Anthropic: content array with base64 source
"content": [{"type": "image", "source": {"type": "base64", "data": "..."}}]

# Ollama: separate images parameter
{"images": ["base64string"], "prompt": "..."}

# HuggingFace: file paths or PIL images
{"image": "/path/to/image.jpg", "prompt": "..."}
```

**Decision**: Media stays in AbstractLLM core because:
- Provider-specific formatting is complex
- Resolution requirements vary by model
- All providers use media processing
- Users shouldn't worry about these differences

### 3. Architecture Detection Is Fundamental

**Evidence from Investigation**:
- 80+ models across 7 architecture families
- Determines message formatting (templates, prefixes)
- Determines tool formats (JSON, XML, Python-style)
- Determines capabilities (context, vision, tools)

**Decision**: Architecture system is core AbstractLLM:
- It's the "HOW" of communication
- Required by all providers
- Central to format selection
- Enables model portability

### 4. Event System for Extensibility

**Design Based on Investigation**:
```python
class EventBus:
    def emit(self, event_type: str, **data):
        # Core events in AbstractLLM
        'llm.request.start'      # Before processing
        'llm.request.verbatim'   # Exact payload (security/debug)
        'llm.response.complete'  # After response
        'llm.tool.executed'      # Tool execution
        'llm.media.processed'    # Media handling
```

**Decision**: Event system in core enables:
- Telemetry without coupling
- Plugin architecture
- Cross-cutting concerns
- Verbatim capture for security

### 5. Three-Package Architecture

**Final Architecture Decision**:

#### AbstractLLM (Core Platform)
```python
# What: Unified LLM interface with essential infrastructure
# Why: Tools, media, and architecture are provider-specific complexity

abstractllm/
├── providers/       # Different APIs, formats, capabilities
├── tools/          # Different tool formats per provider
├── media/          # Different media handling per provider
├── architectures/  # HOW to communicate with models
├── events/         # Extensibility without coupling
└── session.py      # Simple conversation tracking
```

#### AbstractMemory (Memory System)
```python
# What: Sophisticated memory with cognitive enhancements
# Why: Complex enough to deserve its own package

abstractmemory/
├── memory.py       # Hierarchical memory
├── knowledge/      # Knowledge graphs
├── react/          # Reasoning cycles
└── cognitive/      # Fact extraction, summarization
```

#### AbstractAgent (Agent Framework)
```python
# What: Orchestrates LLM + Memory for intelligence
# Why: Agents need both LLM and Memory

abstractagent/
├── agent.py        # Combines LLM + Memory
├── strategies/     # Retry, structured responses
├── tools/          # Advanced agent-specific tools
└── cli/           # Development & testing
```

### 6. Session Simplification

**Current Problem**: session.py has 109 methods, 4,097 lines
**Decision**: Split into:
- **Core Session** (~800 lines): Just conversation tracking
- **Agent behaviors**: Move to AbstractAgent

**Justification**:
- Session should just track messages
- Memory is optional enhancement
- Agent adds intelligence

### 7. Telemetry in Core

**Current State**:
- `_capture_verbatim_context()` captures exact requests
- Critical for debugging provider issues
- Needed for security auditing

**Decision**: Keep verbatim capture in core + add events:
- Verbatim capture is essential for debugging
- Events enable extensibility
- Both serve different purposes

## What Changed from Initial Analysis

### Corrected Misconceptions
1. **Media is optional** → Media is ESSENTIAL infrastructure
2. **Tools are agent features** → Tools are PROVIDER abstraction
3. **Cognitive is standalone** → Cognitive ENHANCES memory
4. **CLI is separate** → CLI belongs with agent development

### Validated Insights
1. Session is doing too much (confirmed: 4,097 lines)
2. Memory deserves its own package (confirmed: complex enough)
3. Three packages is right balance (confirmed: not too many)
4. SOTA alignment matters (confirmed: LangChain/LlamaIndex patterns)

## Implementation Priorities

### Week 1: Emergency Surgery
- Split session.py (CRITICAL - 4,097 → 800 lines)
- Extract memory components
- Extract agent features

### Week 2: Package Structure
- Create three package directories
- Establish clean dependencies
- Add event system

### Week 3: Testing & Validation
- Unit tests per package
- Integration tests
- Performance benchmarks

### Week 4: Migration Support
- Compatibility layer
- Migration tools
- Documentation

## Success Criteria

### Technical Success
- No file > 1,000 lines
- No circular imports
- Import time < 200ms for core
- Clean dependency graph

### User Success
- Simple cases unchanged
- Clear migration path
- Better mental model
- Progressive complexity

## Final Recommendation

**Proceed with three-package architecture**:
1. **AbstractLLM**: Provider abstraction + essential infrastructure
2. **AbstractMemory**: Sophisticated memory system
3. **AbstractAgent**: Agent orchestration

This architecture:
- Respects actual code dependencies
- Unifies provider complexity in core
- Enables clean separation of concerns
- Aligns with SOTA patterns
- Provides sustainable growth path

The investigation confirms this is not just desirable but **necessary** to prevent AbstractLLM from becoming unmaintainable. The current trajectory leads to crisis within 6 months without intervention.