# AbstractLLM Core Addendum: BasicSession Enhancements

**Purpose**: Minor enhancements to BasicSession for better integration with AbstractMemory and AbstractAgent

## Context

After implementing the two-tier memory strategy, we identified that BasicSession from AbstractLLM Core is already sufficient for many simple agents. However, a few minor enhancements would improve integration with the memory package.

## Current State

BasicSession (151 lines) already provides:
- Message history tracking
- Basic persistence (save/load)
- Simple conversation management
- Session ID and timestamps

## Recommended Enhancements

### 1. Add Message Window Limit (Optional Enhancement)

**File**: `/Users/albou/projects/abstractllm_core/abstractllm/core/session.py`

Add optional message limit to prevent unbounded growth:

```python
class BasicSession:
    """
    Minimal session for conversation management.
    """

    def __init__(self,
                 provider: Optional[AbstractLLMInterface] = None,
                 system_prompt: Optional[str] = None,
                 max_messages: Optional[int] = None):  # NEW
        """Initialize basic session"""

        self.provider = provider
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.messages: List[Message] = []
        self.system_prompt = system_prompt
        self.max_messages = max_messages  # NEW

        # Add system message if provided
        if system_prompt:
            self.add_message(MessageRole.SYSTEM.value, system_prompt)

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to conversation history"""
        message = Message(role=role, content=content)
        self.messages.append(message)

        # Trim if exceeds max_messages (keep system message)
        if self.max_messages and len(self.messages) > self.max_messages:
            system_msgs = [m for m in self.messages if m.role == 'system']
            other_msgs = [m for m in self.messages if m.role != 'system']
            # Keep system messages + last N messages
            keep_count = self.max_messages - len(system_msgs)
            self.messages = system_msgs + other_msgs[-keep_count:]

        return message
```

### 2. Add Context Window Helper

Add method to get recent context efficiently:

```python
    def get_recent_context(self, n: int = 5) -> str:
        """Get last n message pairs as formatted context"""
        # Get non-system messages
        non_system = [m for m in self.messages if m.role != 'system']
        recent = non_system[-n*2:] if n else non_system  # n exchanges = n*2 messages

        lines = []
        for msg in recent:
            lines.append(f"{msg.role}: {msg.content}")

        return "\n".join(lines)
```

### 3. Add Compatibility Method

For compatibility with AbstractMemory BufferMemory:

```python
    def as_buffer_format(self) -> List[Dict[str, str]]:
        """Export messages in BufferMemory format for compatibility"""
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if hasattr(m, 'timestamp') else None
            }
            for m in self.messages
        ]
```

## Integration Examples

### Using BasicSession as Simple Memory

```python
from abstractllm import BasicSession
from abstractagent import Agent

# For simple task agents, BasicSession is sufficient
agent = Agent(
    llm_config={"provider": "openai", "model": "gpt-3.5-turbo"},
    memory_type="session",  # Just use BasicSession
    purpose="task"
)

# BasicSession handles the conversation history
response = agent.chat("Translate 'hello' to French")
# Session automatically tracks: user message -> assistant response
```

### Upgrading from BasicSession to BufferMemory

```python
from abstractllm import BasicSession
from abstractmemory import BufferMemory

# Start with BasicSession
session = BasicSession(max_messages=100)
session.add_message("user", "Hello")
session.add_message("assistant", "Hi there!")

# If needed, upgrade to BufferMemory
buffer = BufferMemory(max_messages=100)
for msg in session.as_buffer_format():
    buffer.add_message(msg["role"], msg["content"])

# Now you have BufferMemory with the conversation
```

## Why These Changes are Minimal

1. **No Breaking Changes**: All enhancements are optional parameters
2. **Backward Compatible**: Existing code continues to work
3. **Clean and Simple**: No over-engineering, just practical additions
4. **Performance**: No overhead for users who don't need these features

## Implementation Priority

**LOW PRIORITY** - BasicSession works well as-is for most use cases.

These enhancements are nice-to-have improvements that can be added when convenient:

1. **max_messages parameter**: Prevents unbounded growth (5 minutes)
2. **get_recent_context()**: Convenience method (5 minutes)
3. **as_buffer_format()**: Compatibility helper (5 minutes)

Total: ~15 minutes of work if needed

## Summary

BasicSession from AbstractLLM Core is already well-designed and sufficient for simple agents. The recommended enhancements are minor quality-of-life improvements that maintain the clean, simple design while improving integration with the memory package.

**Key Principle**: Don't over-engineer. BasicSession's simplicity is its strength. These minimal enhancements preserve that simplicity while adding practical value.