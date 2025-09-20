# AbstractLLM Detailed Refactoring Plan

*Based on deep code investigation and actual testing*

## Executive Summary

After thorough investigation of the codebase, I confirm that the three-package architecture (AbstractLLM + AbstractMemory + AbstractAgent) is the correct approach. The core insight is that AbstractLLM must unify the complexity of different providers' tool and media handling, while maintaining clean separation from memory and agent behaviors.

## Core Findings from Investigation

### 1. Tools Are Core Infrastructure
**Evidence**: Each provider handles tools completely differently:
- **OpenAI**: Native API with `tools` parameter, function objects
- **Anthropic**: XML format in messages, tool_use blocks
- **Ollama**: Architecture-based (Qwen uses `<|tool_call|>`, Llama uses `<function_call>`)
- **MLX/HuggingFace**: Prompted mode with architecture-specific formats

**Conclusion**: Tools MUST be in AbstractLLM to abstract this complexity behind a unified interface.

### 2. Media Is Essential Infrastructure
**Evidence**: Media handling is provider-specific:
- **OpenAI**: `content` array with `image_url` objects
- **Anthropic**: `content` array with base64 `source` objects
- **Ollama**: Separate `images` parameter with base64 strings
- **HuggingFace**: File paths or PIL images

**Conclusion**: Media handling MUST be in AbstractLLM core.

### 3. Architecture Detection Is Fundamental
**Evidence**: The architecture system determines:
- How to format messages (templates, prefixes/suffixes)
- What tool format to use (JSON, XML, Python-style)
- Model capabilities (context, vision, tools)

**Conclusion**: Architecture detection is core AbstractLLM functionality.

### 4. Telemetry Needs Enhancement
**Current State**:
- `_capture_verbatim_context()` captures exact LLM requests
- `log_request()`/`log_response()` track interactions
- `_pending_requests` dict has memory leak potential

**Need**: Event system for extensibility while keeping verbatim capture in core.

## Proposed Event System

### Core Events in AbstractLLM
```python
# abstractllm/events.py
from typing import Any, Dict, Callable, List
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class LLMEvent:
    """Base event for all LLM interactions."""
    event_id: str
    event_type: str
    timestamp: datetime
    provider: str
    model: str
    data: Dict[str, Any]

class EventBus:
    """Central event system for AbstractLLM."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._middleware: List[Callable] = []

    def on(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def emit(self, event_type: str, **data) -> LLMEvent:
        """Emit an event."""
        event = LLMEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            provider=data.get('provider', 'unknown'),
            model=data.get('model', 'unknown'),
            data=data
        )

        # Apply middleware
        for middleware in self._middleware:
            event = middleware(event)
            if event is None:
                return None

        # Call handlers
        for handler in self._handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                # Log but don't crash
                import logging
                logging.error(f"Event handler error: {e}")

        return event

# Global event bus
event_bus = EventBus()
```

### Event Integration in Providers
```python
# In BaseProvider
class BaseProvider:
    def generate(self, prompt, **kwargs):
        # Emit pre-request event
        event_bus.emit(
            'llm.request.start',
            provider=self.provider_name,
            model=self.get_param(ModelParameter.MODEL),
            prompt=prompt,
            kwargs=kwargs
        )

        # Capture verbatim for security/debugging
        self._capture_verbatim_context(request_payload)

        # Emit verbatim event
        event_bus.emit(
            'llm.request.verbatim',
            provider=self.provider_name,
            model=self.get_param(ModelParameter.MODEL),
            verbatim_context=request_payload,
            endpoint=api_endpoint
        )

        # Make actual API call
        response = self._generate_impl(prompt, **kwargs)

        # Emit response event
        event_bus.emit(
            'llm.response.complete',
            provider=self.provider_name,
            model=self.get_param(ModelParameter.MODEL),
            response=response,
            duration=elapsed_time
        )

        return response
```

## Phased Refactoring Plan

### Phase 1: Emergency Session Surgery (Week 1)
**Goal**: Reduce session.py from 4,097 to ~800 lines

#### Day 1-2: Extract Memory Components
```python
# Move to abstractllm/_memory/core.py (future abstractmemory)
class HierarchicalMemory:
    # All memory logic (1,959 lines from memory.py)

# Move to abstractllm/_memory/react.py
class ReActCycle:
    # ReAct reasoning logic

# Keep in session.py
class Session:
    def __init__(self, provider, memory=None):
        self.provider = provider
        self.messages = []
        self.memory = memory  # Optional, injected
```

#### Day 3-4: Extract Agent Features
```python
# Move to abstractllm/_agent/strategies.py
class RetryStrategy:
    # Retry logic

class StructuredResponseHandler:
    # Structured response logic

# Move to abstractllm/_agent/workflows.py
class WorkflowManager:
    # Complex multi-step workflows
```

#### Day 5: Clean Up Session
```python
# Final session.py (~800 lines)
class Session:
    """Simple conversation management."""

    def __init__(self, provider):
        self.provider = provider
        self.messages = []
        self.id = str(uuid.uuid4())

    def generate(self, prompt, **kwargs):
        # Simple delegation to provider
        response = self.provider.generate(prompt, **kwargs)
        self.add_message('user', prompt)
        self.add_message('assistant', response.content)
        return response

    def add_message(self, role, content):
        self.messages.append(Message(role, content))

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []

    def save(self, path):
        # Simple persistence

    def load(self, path):
        # Simple loading
```

### Phase 2: Create Package Structure (Week 2)

#### Day 1: AbstractLLM Core Structure
```bash
abstractllm/
├── __init__.py          # Minimal exports
├── interface.py         # AbstractLLMInterface
├── factory.py          # create_llm()
├── session.py          # Simplified Session
├── events.py           # Event system
├── providers/
│   ├── base.py        # Enhanced with events
│   ├── openai.py
│   ├── anthropic.py
│   ├── ollama.py
│   ├── mlx.py
│   └── huggingface.py
├── architectures/      # HOW to communicate
│   ├── detection.py
│   └── enums.py
├── tools/              # Tool infrastructure
│   ├── core.py
│   ├── handler.py
│   ├── parser.py
│   └── registry.py
├── media/              # Media handling
│   ├── processor.py
│   ├── image.py
│   └── text.py
└── utils/
    ├── logging.py      # Enhanced with events
    └── config.py
```

#### Day 2: AbstractMemory Structure
```bash
abstractmemory/
├── __init__.py
├── memory.py           # HierarchicalMemory
├── components/
│   ├── working.py
│   ├── episodic.py
│   └── semantic.py
├── knowledge/
│   ├── graph.py
│   └── facts.py
├── react/
│   ├── cycles.py
│   └── scratchpad.py
├── cognitive/          # Enhancements
│   ├── facts_extractor.py
│   ├── summarizer.py
│   └── values.py
└── storage/
    ├── base.py
    └── lancedb.py
```

#### Day 3: AbstractAgent Structure
```bash
abstractagent/
├── __init__.py
├── agent.py            # Main Agent class
├── orchestration/
│   ├── coordinator.py
│   └── workflows.py
├── strategies/
│   ├── retry.py
│   ├── structured.py
│   └── reasoning.py
├── tools/              # Advanced tools
│   ├── code.py
│   ├── web.py
│   └── catalog.py
└── cli/
    ├── alma.py
    └── commands/
```

#### Day 4-5: Wire Dependencies
```python
# abstractagent/agent.py
from abstractllm import create_llm, Session, event_bus
from abstractmemory import HierarchicalMemory

class Agent:
    def __init__(self, llm_config, memory_config=None):
        # Create LLM
        self.llm = create_llm(**llm_config)

        # Create session for conversation tracking
        self.session = Session(self.llm)

        # Optional memory
        if memory_config:
            self.memory = HierarchicalMemory(**memory_config)
        else:
            self.memory = None

        # Register event handlers
        self._register_events()

    def _register_events(self):
        event_bus.on('llm.request.start', self._on_request_start)
        event_bus.on('llm.response.complete', self._on_response_complete)

    def chat(self, prompt):
        # Get memory context if available
        context = None
        if self.memory:
            context = self.memory.get_context_for_query(prompt)

        # Generate with context
        if context:
            enhanced_prompt = f"Context: {context}\n\nUser: {prompt}"
            response = self.session.generate(enhanced_prompt)
        else:
            response = self.session.generate(prompt)

        # Update memory if available
        if self.memory:
            self.memory.add_interaction(prompt, response.content)

        return response
```

### Phase 3: Testing & Validation (Week 3)

#### Day 1-2: Unit Tests for Each Package
```python
# tests/test_abstractllm.py
def test_provider_creation():
    llm = create_llm('ollama', model='qwen3:4b')
    assert llm is not None

def test_tool_abstraction():
    # Test that tools work across providers

def test_media_handling():
    # Test media processing

# tests/test_abstractmemory.py
def test_memory_isolation():
    memory = HierarchicalMemory()
    # Test memory without LLM

# tests/test_abstractagent.py
def test_agent_orchestration():
    agent = Agent(llm_config={...})
    # Test agent behaviors
```

#### Day 3-4: Integration Tests
```python
def test_agent_with_memory():
    # Full stack test
    agent = Agent(
        llm_config={'provider': 'ollama', 'model': 'qwen3:4b'},
        memory_config={'working_memory_size': 10}
    )

    response1 = agent.chat("My name is Alice")
    response2 = agent.chat("What's my name?")
    assert "Alice" in response2.content
```

#### Day 5: Performance Benchmarks
```python
def benchmark_import_times():
    # Measure import performance
    start = time.time()
    import abstractllm
    core_time = time.time() - start

    start = time.time()
    import abstractmemory
    memory_time = time.time() - start

    assert core_time < 0.2  # 200ms max
    assert memory_time < 0.3  # 300ms max
```

### Phase 4: Migration Support (Week 4)

#### Day 1-2: Compatibility Layer
```python
# abstractllm/__init__.py (compatibility mode)
import warnings

# Try to import from new packages
try:
    from abstractmemory import HierarchicalMemory as _Memory
    from abstractagent import Agent as _Agent

    # Provide compatibility aliases
    class Session(_Agent):
        """Compatibility wrapper for old Session API."""
        def __init__(self, provider=None, enable_memory=True, **kwargs):
            warnings.warn(
                "Session with memory is deprecated. Use Agent from abstractagent.",
                DeprecationWarning
            )
            # Convert old API to new
            llm_config = {'provider': provider} if isinstance(provider, str) else {}
            memory_config = {} if enable_memory else None
            super().__init__(llm_config, memory_config)

    # Export for compatibility
    HierarchicalMemory = _Memory

except ImportError:
    # New packages not available, use monolithic
    from .session import Session
    from .memory import HierarchicalMemory
```

#### Day 3-4: Migration Tools
```python
# tools/migrate.py
import ast
import os

class CodeMigrator(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == 'abstractllm':
            # Map old imports to new
            replacements = {
                'Session': 'from abstractagent import Agent',
                'HierarchicalMemory': 'from abstractmemory import HierarchicalMemory',
            }
            # Transform imports
        return node

def migrate_file(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())

    migrator = CodeMigrator()
    new_tree = migrator.visit(tree)

    # Write back
    with open(filepath, 'w') as f:
        f.write(ast.unparse(new_tree))
```

#### Day 5: Documentation
```markdown
# Migration Guide

## For Simple LLM Usage
No changes needed:
```python
from abstractllm import create_llm
llm = create_llm('openai')
response = llm.generate("Hello")
```

## For Conversations
Minor change:
```python
# Old
from abstractllm import Session
session = Session(provider='openai')

# New
from abstractllm import create_llm, Session
llm = create_llm('openai')
session = Session(llm)
```

## For Agents with Memory
New cleaner API:
```python
# Old
from abstractllm import Session
session = Session(provider='openai', enable_memory=True)

# New
from abstractagent import Agent
from abstractmemory import HierarchicalMemory

agent = Agent(
    llm_config={'provider': 'openai'},
    memory_config={'working_memory_size': 10}
)
```
```

## Success Metrics

### Code Quality Metrics
- [ ] session.py < 800 lines (from 4,097)
- [ ] No circular imports
- [ ] All tests pass
- [ ] Import time < 200ms for core

### Architecture Metrics
- [ ] Clean dependency graph
- [ ] Event system working
- [ ] Telemetry capturing verbatim
- [ ] Memory properly isolated

### User Experience Metrics
- [ ] Migration guide complete
- [ ] Compatibility layer working
- [ ] No breaking changes for simple use cases
- [ ] Clear upgrade path

## Risk Mitigation

### Risk: Complex Migration
**Mitigation**:
- Compatibility layer for 6 months
- Automated migration tools
- Extensive examples

### Risk: Performance Regression
**Mitigation**:
- Benchmark suite before/after
- Lazy imports where possible
- Profile critical paths

### Risk: Feature Parity
**Mitigation**:
- Complete test coverage
- Feature flag for experimental
- Gradual rollout

## Conclusion

This refactoring plan addresses the critical issues while maintaining AbstractLLM's strengths:

1. **Tools and Media stay in core** - They're essential for provider abstraction
2. **Event system enables extensibility** - While keeping verbatim capture
3. **Clean separation of concerns** - LLM, Memory, Agent as distinct layers
4. **Incremental migration** - Users adopt at their own pace

The plan is actionable, testable, and achievable in 4 weeks with clear daily objectives.