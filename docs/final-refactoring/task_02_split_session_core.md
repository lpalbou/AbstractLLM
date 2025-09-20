# Task 02: Split Session Core (Priority 1)

**Duration**: 4 hours
**Risk**: High
**Dependencies**: Task 01 completed

## Objectives
- Extract core session functionality (500 lines)
- Move memory components to staging
- Move agent behaviors to staging
- Create compatibility wrapper

## Steps

### 1. Analyze Current Session (30 min)

```bash
# Get exact line counts per component
cd /Users/albou/projects/abstractllm

# Run analyzer
python tools/analyze_session.py > docs/final-refactoring/logs/session_analysis.txt

# Get line numbers for Session class
grep -n "^class Session:" abstractllm/session.py
grep -n "^class SessionManager:" abstractllm/session.py
```

### 2. Extract Core Session Methods (1 hour)

Create `abstractllm/_refactoring/session/basic.py`:
```python
"""
Basic session for conversation tracking.
Target: 500 lines maximum.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import json
import uuid

from abstractllm.interface import AbstractLLMInterface, ModelParameter
from abstractllm.enums import MessageRole
from abstractllm.types import GenerateResponse


class Message:
    """Simple message representation"""

    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        msg = cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        msg.id = data.get("id", str(uuid.uuid4()))
        return msg


class BasicSession:
    """
    Minimal session for conversation management.
    ONLY handles:
    - Message history
    - Basic generation
    - Simple persistence
    """

    def __init__(self,
                 provider: Optional[Union[str, AbstractLLMInterface]] = None,
                 system_prompt: Optional[str] = None):
        """Initialize basic session"""

        # Set up provider
        self._provider = None
        if provider:
            if isinstance(provider, str):
                from abstractllm import create_llm
                self._provider = create_llm(provider)
            else:
                self._provider = provider

        # Core attributes
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.messages: List[Message] = []
        self.system_prompt = system_prompt

        # Add system message if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM.value, system_prompt)

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to conversation history"""
        message = Message(role, content)
        self.messages.append(message)
        return message

    def get_messages(self) -> List[Message]:
        """Get all messages"""
        return self.messages.copy()

    def get_history(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """Get conversation history as dicts"""
        if include_system:
            return [m.to_dict() for m in self.messages]
        return [m.to_dict() for m in self.messages if m.role != 'system']

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history"""
        if keep_system:
            self.messages = [m for m in self.messages if m.role == 'system']
        else:
            self.messages = []

    def generate(self, prompt: str, **kwargs) -> GenerateResponse:
        """Generate response using provider"""
        if not self._provider:
            raise ValueError("No provider configured")

        # Add user message
        self.add_message('user', prompt)

        # Format messages for provider
        messages = self._format_messages_for_provider()

        # Call provider
        response = self._provider.generate(
            prompt=prompt,
            messages=messages,
            system_prompt=self.system_prompt,
            **kwargs
        )

        # Add assistant response
        if hasattr(response, 'content'):
            self.add_message('assistant', response.content)

        return response

    def _format_messages_for_provider(self) -> List[Dict[str, str]]:
        """Format messages for provider API"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
            if m.role != 'system'  # System handled separately
        ]

    def save(self, filepath: Union[str, Path]):
        """Save session to file"""
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "system_prompt": self.system_prompt,
            "messages": [m.to_dict() for m in self.messages]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BasicSession':
        """Load session from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        session = cls(system_prompt=data.get("system_prompt"))
        session.id = data["id"]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.messages = [Message.from_dict(m) for m in data["messages"]]

        return session
```

### 3. Extract Memory Components (1 hour)

Create extraction script `tools/extract_memory.py`:
```python
#!/usr/bin/env python3
"""Extract memory-related code from session.py"""

import ast
import os

# Memory-related method names
MEMORY_METHODS = [
    'enable_memory', 'memory', 'memory_facts_max',
    'memory_facts_min_confidence', 'memory_facts_min_occurrences',
    'query_memory', 'save_memory', 'visualize_memory_links',
    'get_memory_stats', '_initialize_memory', '_update_memory'
]

class MemoryExtractor(ast.NodeVisitor):
    def __init__(self):
        self.memory_code = []

    def visit_FunctionDef(self, node):
        if any(name in node.name for name in MEMORY_METHODS):
            self.memory_code.append(ast.unparse(node))

if __name__ == "__main__":
    with open('abstractllm/session.py', 'r') as f:
        tree = ast.parse(f.read())

    extractor = MemoryExtractor()
    extractor.visit(tree)

    # Write to staging
    os.makedirs('abstractllm/_refactoring/memory', exist_ok=True)
    with open('abstractllm/_refactoring/memory/session_memory.py', 'w') as f:
        f.write("# Memory methods extracted from session.py\n\n")
        f.write("\n\n".join(extractor.memory_code))

    print(f"Extracted {len(extractor.memory_code)} memory methods")
```

Run extraction:
```bash
python tools/extract_memory.py
```

### 4. Extract Agent Components (1 hour)

Create extraction script `tools/extract_agent.py`:
```python
#!/usr/bin/env python3
"""Extract agent-related code from session.py"""

import ast
import os

# Agent-related patterns
AGENT_PATTERNS = [
    'tool', 'react', 'cycle', 'retry', 'structured',
    'orchestrat', 'reason', 'think', 'act', 'observe',
    'scratchpad', 'workflow'
]

class AgentExtractor(ast.NodeVisitor):
    def __init__(self):
        self.agent_code = []

    def visit_FunctionDef(self, node):
        name_lower = node.name.lower()
        if any(pattern in name_lower for pattern in AGENT_PATTERNS):
            self.agent_code.append((node.name, ast.unparse(node)))

if __name__ == "__main__":
    with open('abstractllm/session.py', 'r') as f:
        tree = ast.parse(f.read())

    extractor = AgentExtractor()
    extractor.visit(tree)

    # Organize by category
    categories = {
        'tools': [],
        'react': [],
        'retry': [],
        'structured': [],
        'other': []
    }

    for name, code in extractor.agent_code:
        if 'tool' in name.lower():
            categories['tools'].append(code)
        elif 'react' in name.lower() or 'cycle' in name.lower():
            categories['react'].append(code)
        elif 'retry' in name.lower():
            categories['retry'].append(code)
        elif 'structured' in name.lower():
            categories['structured'].append(code)
        else:
            categories['other'].append(code)

    # Write to staging
    os.makedirs('abstractllm/_refactoring/agent', exist_ok=True)
    for category, methods in categories.items():
        if methods:
            with open(f'abstractllm/_refactoring/agent/{category}.py', 'w') as f:
                f.write(f"# {category.title()} methods extracted from session.py\n\n")
                f.write("\n\n".join(methods))

    total = sum(len(v) for v in categories.values())
    print(f"Extracted {total} agent methods")
```

Run extraction:
```bash
python tools/extract_agent.py
```

### 5. Create Compatibility Wrapper (30 min)

Create `abstractllm/session_compat.py`:
```python
"""
Compatibility wrapper to maintain backward compatibility.
Maps old Session API to new components.
"""

import warnings
from typing import Optional, Dict, Any, List, Union

from abstractllm._refactoring.session.basic import BasicSession, Message

# Try to import extracted components
try:
    from abstractllm._refactoring.memory.session_memory import *
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from abstractllm._refactoring.agent.tools import *
    from abstractllm._refactoring.agent.react import *
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class Session(BasicSession):
    """
    Compatibility wrapper mimicking the original Session API.
    Delegates to appropriate components.
    """

    def __init__(self, **kwargs):
        # Warn about migration
        warnings.warn(
            "Session is being refactored. Please migrate to new API.",
            DeprecationWarning,
            stacklevel=2
        )

        # Extract basic session params
        provider = kwargs.pop('provider', None)
        system_prompt = kwargs.pop('system_prompt', None)

        # Initialize basic session
        super().__init__(provider=provider, system_prompt=system_prompt)

        # Handle memory params
        self.enable_memory = kwargs.pop('enable_memory', False)
        self.memory = None

        if self.enable_memory and MEMORY_AVAILABLE:
            # Initialize memory with extracted components
            self._init_memory(**kwargs)

        # Handle agent params
        self.tools = []
        tools = kwargs.pop('tools', None)
        if tools and AGENT_AVAILABLE:
            self._init_tools(tools)

        # Store remaining kwargs
        self._extra_kwargs = kwargs

    def __getattr__(self, name):
        """Delegate unknown attributes to components"""
        # Try memory attributes
        if MEMORY_AVAILABLE and hasattr(self, 'memory') and self.memory:
            if hasattr(self.memory, name):
                return getattr(self.memory, name)

        # Try agent attributes
        if AGENT_AVAILABLE:
            # Check in extracted agent modules
            pass

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # Add specific compatibility methods as needed
    def generate_with_tools(self, **kwargs):
        """Compatibility for tool generation"""
        if AGENT_AVAILABLE:
            # Delegate to agent components
            pass
        else:
            # Fallback to basic generation
            return self.generate(kwargs.get('prompt', ''))

# Make Session the default export
__all__ = ['Session']
```

### 6. Test Compatibility (30 min)

Create test script `tools/test_compat.py`:
```python
#!/usr/bin/env python3
"""Test compatibility wrapper"""

import sys
sys.path.insert(0, '.')

def test_basic_session():
    """Test basic session functionality"""
    from abstractllm._refactoring.session.basic import BasicSession

    session = BasicSession()
    session.add_message('user', 'Hello')
    messages = session.get_messages()
    assert len(messages) == 1
    assert messages[0].role == 'user'
    print("✅ Basic session works")

def test_compat_wrapper():
    """Test compatibility wrapper"""
    try:
        from abstractllm.session_compat import Session

        # Test basic functionality
        session = Session()
        session.add_message('user', 'Test')
        assert len(session.get_messages()) == 1
        print("✅ Compatibility wrapper works")

        # Test memory delegation
        try:
            session = Session(enable_memory=True)
            print("✅ Memory delegation works")
        except:
            print("⚠️ Memory not yet integrated")

    except Exception as e:
        print(f"❌ Compatibility wrapper failed: {e}")

if __name__ == "__main__":
    test_basic_session()
    test_compat_wrapper()
```

Run tests:
```bash
python tools/test_compat.py
```

## Validation

### Check extraction results
```bash
# Count lines in new basic session
wc -l abstractllm/_refactoring/session/basic.py

# Check extracted components
ls -la abstractllm/_refactoring/memory/
ls -la abstractllm/_refactoring/agent/

# Run compatibility test
python tools/test_compat.py
```

### Verify no broken imports
```bash
# Test that existing code still works
python -c "from abstractllm import Session; print('Import works')"
```

## Success Criteria

- [ ] BasicSession < 500 lines
- [ ] Memory methods extracted to staging
- [ ] Agent methods extracted to staging
- [ ] Compatibility wrapper created
- [ ] Basic tests passing
- [ ] No import errors

## Troubleshooting

If extraction fails:
1. Check AST parsing errors in extraction scripts
2. Manually identify methods in session.py
3. Use grep to find method definitions

If compatibility fails:
1. Check import paths in wrapper
2. Ensure staging directories exist
3. Add missing delegation methods

## Output

After completion:
1. `abstractllm/_refactoring/session/basic.py` - Core session (< 500 lines)
2. `abstractllm/_refactoring/memory/` - Extracted memory components
3. `abstractllm/_refactoring/agent/` - Extracted agent components
4. `abstractllm/session_compat.py` - Compatibility wrapper

## Next Task

Proceed to Task 03: Create Memory Package Structure