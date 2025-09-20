# AbstractLLM Refactoring Implementation Guide

*Actionable steps with exact commands and code*

## Pre-Refactoring Checklist

### 1. Backup Everything
```bash
# Create complete backup
cd /Users/albou/projects
cp -r abstractllm abstractllm_backup_$(date +%Y%m%d)
cd abstractllm
git checkout -b refactoring_backup
git add -A && git commit -m "Pre-refactoring backup"
```

### 2. Create Migration Scripts
```python
# tools/split_session.py
import ast
import os

class SessionSplitter(ast.NodeVisitor):
    def __init__(self):
        self.memory_methods = []
        self.agent_methods = []
        self.core_methods = []

    def visit_FunctionDef(self, node):
        if 'memory' in node.name or 'Memory' in node.name:
            self.memory_methods.append(node)
        elif 'tool' in node.name or 'react' in node.name:
            self.agent_methods.append(node)
        else:
            self.core_methods.append(node)

# Run: python tools/split_session.py
```

## Week 1: Emergency Surgery

### Day 1: Extract Memory Components

```bash
# Create internal structure
cd /Users/albou/projects/abstractllm/abstractllm
mkdir -p _internal/{memory,agent,session}

# Extract memory from session.py
python << 'EOF'
import ast

# Read session.py
with open('session.py', 'r') as f:
    tree = ast.parse(f.read())

# Extract memory-related code
memory_code = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        if 'Memory' in node.name:
            memory_code.append(ast.unparse(node))

# Write to new file
with open('_internal/memory/core.py', 'w') as f:
    f.write('\n'.join(memory_code))
EOF
```

### Day 2: Extract Agent Components

```python
# _internal/agent/orchestrator.py
"""Agent orchestration extracted from session.py"""

class AgentOrchestrator:
    """Coordinates LLM, memory, and tools"""

    def __init__(self, llm, memory=None):
        self.llm = llm
        self.memory = memory
        self.tools = {}

    def execute_with_reasoning(self, prompt):
        # ReAct cycle implementation
        thought = self.think(prompt)
        action = self.act(thought)
        observation = self.observe(action)
        return self.respond(observation)
```

### Day 3: Simplify Session

```python
# _internal/session/basic.py
"""Simplified session for basic conversation tracking"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class BasicSession:
    """Minimal session for conversation management"""

    def __init__(self, provider):
        self.provider = provider
        self.messages = []
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()

    def add_message(self, role: str, content: str):
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })

    def generate(self, prompt: str, **kwargs):
        # Add user message
        self.add_message('user', prompt)

        # Call provider with history
        response = self.provider.generate(
            prompt=prompt,
            messages=self.messages,
            **kwargs
        )

        # Add assistant response
        self.add_message('assistant', response.content)
        return response

    def clear(self):
        self.messages = []
```

### Day 4: Create Compatibility Bridge

```python
# abstractllm/session_compat.py
"""Compatibility layer for existing code"""

from ._internal.session.basic import BasicSession
from ._internal.memory.core import HierarchicalMemory
from ._internal.agent.orchestrator import AgentOrchestrator

class Session:
    """Compatible wrapper mimicking old Session API"""

    def __init__(self, provider=None, enable_memory=True, **kwargs):
        # Initialize components
        self.basic_session = BasicSession(provider)
        self.memory = HierarchicalMemory() if enable_memory else None
        self.orchestrator = AgentOrchestrator(provider, self.memory)

        # Delegate attributes
        self.messages = self.basic_session.messages
        self.id = self.basic_session.id

    def generate(self, prompt, **kwargs):
        if self.memory:
            # Use orchestrator for memory-aware generation
            return self.orchestrator.execute_with_reasoning(prompt)
        else:
            # Use basic session
            return self.basic_session.generate(prompt, **kwargs)

    # Delegate other methods...
    def __getattr__(self, name):
        # Try basic session first
        if hasattr(self.basic_session, name):
            return getattr(self.basic_session, name)
        # Then orchestrator
        if hasattr(self.orchestrator, name):
            return getattr(self.orchestrator, name)
        raise AttributeError(f"Session has no attribute '{name}'")
```

### Day 5: Test Compatibility

```bash
# Run existing tests with compatibility layer
cd /Users/albou/projects/abstractllm
pytest tests/ -v

# Fix any broken tests
python << 'EOF'
import os
import re

# Update imports in tests
for root, dirs, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            # Update imports to use compatibility layer
            content = re.sub(
                r'from abstractllm import Session',
                'from abstractllm.session_compat import Session',
                content
            )
            with open(path, 'w') as f:
                f.write(content)
EOF
```

## Week 2: Package Creation

### Day 1: Create AbstractLLM Core

```bash
# Create new package structure
mkdir -p /Users/albou/projects/abstractllm-core
cd /Users/albou/projects/abstractllm-core

# Initialize package
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="abstractllm",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "requests",
        "pydantic>=2.0.0"
    ],
    python_requires=">=3.8",
)
EOF

# Copy core components
cp -r ../abstractllm/abstractllm/{interface.py,factory.py,types.py,enums.py} abstractllm/
cp -r ../abstractllm/abstractllm/providers abstractllm/
cp -r ../abstractllm/abstractllm/architectures abstractllm/
cp -r ../abstractllm/abstractllm/tools abstractllm/
cp -r ../abstractllm/abstractllm/media abstractllm/

# Create simplified session
cp ../abstractllm/abstractllm/_internal/session/basic.py abstractllm/session.py

# Add event system
cat > abstractllm/events.py << 'EOF'
from typing import Dict, List, Callable
import uuid
from datetime import datetime

class EventBus:
    def __init__(self):
        self._handlers = {}

    def on(self, event_type: str, handler: Callable):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def emit(self, event_type: str, **data):
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': datetime.now(),
            **data
        }
        for handler in self._handlers.get(event_type, []):
            handler(event)

event_bus = EventBus()
EOF
```

### Day 2: Create AbstractMemory

```bash
mkdir -p /Users/albou/projects/abstractmemory
cd /Users/albou/projects/abstractmemory

# Package structure
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="abstractmemory",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "abstractllm>=2.0.0",
        "lancedb",
        "sentence-transformers",
        "networkx"  # For graph operations
    ],
)
EOF

# Create temporal knowledge graph
mkdir -p abstractmemory/{core,components,graph,cognitive,storage}

cat > abstractmemory/graph/knowledge_graph.py << 'EOF'
import networkx as nx
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class TemporalKnowledgeGraph:
    """Temporal knowledge graph with bi-temporal modeling"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self._node_counter = 0

    def add_fact(self, subject: str, predicate: str, object: str,
                 event_time: datetime, confidence: float = 1.0):
        """Add temporally anchored fact"""
        # Create or get nodes
        subj_id = self._get_or_create_node(subject, 'entity')
        obj_id = self._get_or_create_node(object, 'entity')

        # Add temporal edge
        self.graph.add_edge(
            subj_id, obj_id,
            predicate=predicate,
            event_time=event_time,
            ingestion_time=datetime.now(),
            confidence=confidence,
            valid=True
        )

    def _get_or_create_node(self, value: str, node_type: str) -> str:
        """Get existing node or create new one"""
        # Check for existing node
        for node_id, data in self.graph.nodes(data=True):
            if data.get('value') == value:
                return node_id

        # Create new node
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1
        self.graph.add_node(
            node_id,
            value=value,
            type=node_type,
            created_at=datetime.now()
        )
        return node_id

    def query_at_time(self, query: str, point_in_time: datetime):
        """Query knowledge state at specific time"""
        valid_edges = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if (data['event_time'] <= point_in_time and
                data['valid'] and
                query.lower() in data['predicate'].lower()):
                valid_edges.append((u, v, data))
        return valid_edges
EOF
```

### Day 3: Create AbstractAgent

```bash
mkdir -p /Users/albou/projects/abstractagent
cd /Users/albou/projects/abstractagent

cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="abstractagent",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "abstractllm>=2.0.0",
        "abstractmemory>=1.0.0",
        "pydantic>=2.0.0",
        "rich"  # For CLI
    ],
    entry_points={
        'console_scripts': [
            'alma=abstractagent.cli.alma:main',
        ],
    },
)
EOF

# Create main agent
cat > abstractagent/agent.py << 'EOF'
from abstractllm import create_llm, BasicSession
from abstractmemory import TemporalMemory
from typing import Optional, Dict, Any

class Agent:
    """Autonomous agent with LLM + Memory"""

    def __init__(self,
                 llm_config: Dict[str, Any],
                 memory_config: Optional[Dict[str, Any]] = None):
        # Initialize LLM
        self.llm = create_llm(**llm_config)

        # Initialize session for conversation tracking
        self.session = BasicSession(self.llm)

        # Initialize memory if configured
        self.memory = None
        if memory_config:
            self.memory = TemporalMemory(**memory_config)

        # Initialize reasoning
        from .reasoning.react import ReActOrchestrator
        self.reasoner = ReActOrchestrator(self)

    def chat(self, prompt: str, use_reasoning: bool = False):
        """Main interaction method"""
        # Get memory context if available
        context = None
        if self.memory:
            context = self.memory.retrieve_context(prompt)

        # Use reasoning if requested
        if use_reasoning:
            response = self.reasoner.execute(prompt, context)
        else:
            # Direct generation with context
            if context:
                enhanced_prompt = f"Context: {context}\n\nUser: {prompt}"
                response = self.session.generate(enhanced_prompt)
            else:
                response = self.session.generate(prompt)

        # Store in memory if available
        if self.memory:
            self.memory.add_interaction(prompt, response)

        return response
EOF

# Create ReAct reasoning
mkdir -p abstractagent/reasoning
cat > abstractagent/reasoning/react.py << 'EOF'
from typing import Optional, Any
import json

class ReActOrchestrator:
    """ReAct reasoning implementation"""

    def __init__(self, agent):
        self.agent = agent
        self.max_iterations = 5

    def execute(self, prompt: str, context: Optional[str] = None):
        """Execute ReAct cycle"""
        iteration = 0
        thoughts = []
        actions = []
        observations = []

        while iteration < self.max_iterations:
            # Think
            thought = self._think(prompt, context, thoughts, actions, observations)
            thoughts.append(thought)

            # Check if we have final answer
            if "Final Answer:" in thought:
                return thought.split("Final Answer:")[-1].strip()

            # Act
            action = self._act(thought)
            actions.append(action)

            # Observe
            observation = self._observe(action)
            observations.append(observation)

            iteration += 1

        # Synthesize final answer
        return self._synthesize(thoughts, actions, observations)

    def _think(self, prompt, context, thoughts, actions, observations):
        """Generate thought"""
        react_prompt = f"""
        Task: {prompt}
        Context: {context or 'None'}
        Previous Thoughts: {thoughts[-3:] if thoughts else 'None'}
        Previous Actions: {actions[-3:] if actions else 'None'}
        Previous Observations: {observations[-3:] if observations else 'None'}

        Think step by step about what to do next.
        If you have enough information, provide "Final Answer: <answer>".
        Otherwise, describe what action to take next.

        Thought:"""

        response = self.agent.llm.generate(react_prompt)
        return response.content
EOF
```

### Day 4: Integration Testing

```python
# test_integration.py
"""Test the new three-package architecture"""

def test_basic_llm():
    """Test AbstractLLM alone"""
    from abstractllm import create_llm, BasicSession

    llm = create_llm('ollama', model='qwen3:4b')
    session = BasicSession(llm)
    response = session.generate("Hello")
    assert response is not None

def test_memory_alone():
    """Test AbstractMemory independently"""
    from abstractmemory import TemporalMemory
    from datetime import datetime

    memory = TemporalMemory()
    memory.add_fact(
        subject="Python",
        predicate="is_a",
        object="programming language",
        event_time=datetime.now()
    )

    facts = memory.query("Python")
    assert len(facts) > 0

def test_full_agent():
    """Test complete stack"""
    from abstractagent import Agent

    agent = Agent(
        llm_config={'provider': 'ollama', 'model': 'qwen3:4b'},
        memory_config={'temporal': True}
    )

    response = agent.chat("What is Python?")
    assert response is not None

    # Test with reasoning
    response = agent.chat("Explain step by step: What is 2+2?", use_reasoning=True)
    assert "Think" in str(response) or "Final Answer" in str(response)

if __name__ == "__main__":
    test_basic_llm()
    print("✓ AbstractLLM works")

    test_memory_alone()
    print("✓ AbstractMemory works")

    test_full_agent()
    print("✓ AbstractAgent works")

    print("\n✅ All integration tests passed!")
```

### Day 5: Migration Tools

```python
# migrate.py
"""Automated migration tool"""

import ast
import os
import re

def migrate_imports(file_path):
    """Update imports in a Python file"""
    with open(file_path, 'r') as f:
        content = f.read()

    # Map old imports to new
    replacements = [
        # Old monolithic imports
        (r'from abstractllm import Session',
         'from abstractagent import Agent'),
        (r'from abstractllm import HierarchicalMemory',
         'from abstractmemory import TemporalMemory'),
        (r'from abstractllm\.session import Session',
         'from abstractagent import Agent'),
        (r'from abstractllm\.memory import (.+)',
         r'from abstractmemory import \1'),
    ]

    for old, new in replacements:
        content = re.sub(old, new, content)

    # Update class instantiation
    content = re.sub(
        r'Session\(provider=([^,]+), enable_memory=True',
        r'Agent(llm_config={"provider": \1}, memory_config={}',
        content
    )

    return content

def migrate_project(project_dir):
    """Migrate entire project"""
    for root, dirs, files in os.walk(project_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Migrating {file_path}")

                try:
                    new_content = migrate_imports(file_path)
                    with open(file_path + '.new', 'w') as f:
                        f.write(new_content)
                    print(f"  ✓ Created {file_path}.new")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        migrate_project(sys.argv[1])
    else:
        print("Usage: python migrate.py <project_directory>")
```

## Week 3: Deployment

### Publishing Packages

```bash
# Build and test packages
cd /Users/albou/projects/abstractllm-core
python setup.py sdist bdist_wheel
twine check dist/*

cd /Users/albou/projects/abstractmemory
python setup.py sdist bdist_wheel
twine check dist/*

cd /Users/albou/projects/abstractagent
python setup.py sdist bdist_wheel
twine check dist/*

# Test installation
pip install ./abstractllm-core/dist/abstractllm-2.0.0.tar.gz
pip install ./abstractmemory/dist/abstractmemory-1.0.0.tar.gz
pip install ./abstractagent/dist/abstractagent-1.0.0.tar.gz

# Verify
python -c "
from abstractllm import create_llm
from abstractmemory import TemporalMemory
from abstractagent import Agent
print('✅ All packages imported successfully')
"
```

### Documentation

```markdown
# docs/migration.md

## Quick Start

### Before (Monolithic)
```python
from abstractllm import Session
session = Session(provider='openai', enable_memory=True)
response = session.generate("Hello")
```

### After (Modular)
```python
from abstractagent import Agent
agent = Agent(
    llm_config={'provider': 'openai'},
    memory_config={'temporal': True}
)
response = agent.chat("Hello")
```

## API Mapping

| Old API | New API |
|---------|---------|
| `Session(provider='x')` | `Agent(llm_config={'provider': 'x'})` |
| `session.generate()` | `agent.chat()` |
| `session.memory` | `agent.memory` |
| `session.tools` | `agent.tools` |
```

## Validation Checklist

### Code Quality
- [ ] session.py < 500 lines
- [ ] No circular imports: `python -m py_compile abstractllm/**/*.py`
- [ ] All tests pass: `pytest tests/`
- [ ] Type checking: `mypy abstractllm/`

### Performance
```python
# test_performance.py
import time
import sys

# Test import times
start = time.time()
import abstractllm
core_time = time.time() - start

start = time.time()
import abstractmemory
memory_time = time.time() - start

start = time.time()
import abstractagent
agent_time = time.time() - start

print(f"AbstractLLM: {core_time*1000:.1f}ms")
print(f"AbstractMemory: {memory_time*1000:.1f}ms")
print(f"AbstractAgent: {agent_time*1000:.1f}ms")

assert core_time < 0.2, "Core import too slow"
assert memory_time < 0.3, "Memory import too slow"
assert agent_time < 0.5, "Agent import too slow"
```

## Post-Refactoring

### Monitor & Iterate
```bash
# Set up monitoring
cat > monitor.py << 'EOF'
import psutil
import time

def monitor_performance():
    process = psutil.Process()

    # Memory usage
    mem = process.memory_info().rss / 1024 / 1024
    print(f"Memory: {mem:.1f} MB")

    # CPU usage
    cpu = process.cpu_percent(interval=1)
    print(f"CPU: {cpu:.1f}%")

if __name__ == "__main__":
    from abstractagent import Agent
    agent = Agent(
        llm_config={'provider': 'ollama', 'model': 'qwen3:4b'}
    )

    monitor_performance()

    # Test operations
    for i in range(10):
        agent.chat(f"Test {i}")
        monitor_performance()
EOF

python monitor.py
```

## Final Notes

This implementation guide provides:
1. **Exact commands** to run at each step
2. **Complete code** for new components
3. **Migration tools** for existing code
4. **Validation tests** to ensure success

Execute these steps sequentially, testing after each phase to ensure the refactoring maintains functionality while achieving the architectural improvements.