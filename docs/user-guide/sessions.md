# Working with Sessions

Sessions in AbstractLLM provide a way to manage conversation history and maintain context across multiple interactions with LLM providers. This guide explains how to use sessions effectively in your applications.

## Basic Session Usage

### Creating a Session

```python
from abstractllm import create_llm
from abstractllm.session import Session

# Create a provider
provider = create_llm("openai", model="gpt-4")

# Create a session with that provider
session = Session(
    system_prompt="You are a helpful AI assistant.",
    provider=provider
)
```

### Adding Messages

You can add messages to a session:

```python
from abstractllm.enums import MessageRole

# Add a user message
session.add_message(MessageRole.USER, "Hello, who are you?")

# Add an assistant message
session.add_message(MessageRole.ASSISTANT, "I'm an AI assistant built with AbstractLLM. How can I help you today?")
```

### Generating Responses

Sessions provide a simpler way to generate responses that maintain context:

```python
# Get a response that considers the conversation history
response = session.generate("What's the weather like today?")
print(response)

# Continue the conversation
response = session.generate("And what about tomorrow?")
print(response)
```

Behind the scenes, the session formats the full conversation history for the provider's API.

## Session Management

### Saving and Loading Sessions

Sessions can be saved to and loaded from files:

```python
# Save a session
session.save("my_conversation.json")

# Load a session
from abstractllm.session import Session
loaded_session = Session.load("my_conversation.json", provider=provider)
```

This is useful for:
- Persisting conversations between application restarts
- Analyzing conversations later
- Sharing conversations between different parts of your application

### Managing Multiple Sessions

For applications that need to manage multiple sessions, the `SessionManager` class is available:

```python
from abstractllm.session import SessionManager

# Create a session manager
manager = SessionManager(sessions_dir="./sessions")

# Create a new session
session_id = "user123"
session = manager.create_session(
    system_prompt="You are a helpful assistant.",
    provider=provider
)

# Get an existing session
session = manager.get_session(session_id)

# List all sessions
sessions = manager.list_sessions()
for session_id, created_at, last_modified_at in sessions:
    print(f"Session {session_id}: Created {created_at}, Last modified {last_modified_at}")

# Delete a session
manager.delete_session(session_id)
```

## Advanced Session Features

### Provider Switching

Sessions allow you to switch providers while maintaining conversation history:

```python
# Create a session with OpenAI
openai_provider = create_llm("openai", model="gpt-4")
session = Session(
    system_prompt="You are a helpful assistant.",
    provider=openai_provider
)

# Get a response from OpenAI
response = session.generate("Tell me about quantum computing.")
print(f"OpenAI response: {response}")

# Switch to Anthropic for the next response
anthropic_provider = create_llm("anthropic", model="claude-3-opus-20240229")
response = session.generate(
    "Explain it in simpler terms.",
    provider=anthropic_provider
)
print(f"Anthropic response: {response}")
```

This feature allows you to use different providers for different parts of a conversation based on their strengths.

### Using Sessions with Tools

Sessions are particularly powerful when working with tool calls:

```python
from abstractllm.session import Session

# Define a tool
def get_current_time() -> str:
    """Get the current time."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

# Create a session with tools
session = Session(
    system_prompt="You are a helpful assistant that can provide the current time.",
    provider=provider,
    tools=[get_current_time]
)

# Generate a response using tools
response = session.generate_with_tools("What time is it right now?")
print(response.content)
```

The session handles:
1. Registering the tools
2. Executing the tools when the LLM calls them
3. Providing the results back to the LLM
4. Maintaining the conversation history, including tool calls and results

### Customizing Message Formatting

Sessions allow you to customize how messages are formatted for different providers:

```python
# Get provider-specific formatted messages
openai_messages = session.get_messages_for_provider("openai")
anthropic_messages = session.get_messages_for_provider("anthropic")
```

This is useful when you need to access the raw formatted messages for a specific provider.

## Session Configuration

Sessions can be configured with various parameters:

```python
session = Session(
    system_prompt="You are a helpful assistant.",
    provider=provider,
    provider_config={
        "temperature": 0.7,
        "max_tokens": 500
    },
    metadata={
        "user_id": "123",
        "session_name": "Technical Support Chat"
    }
)
```

Parameters:
- `system_prompt`: Sets the system prompt for the conversation
- `provider`: The LLM provider to use
- `provider_config`: Provider-specific configuration
- `metadata`: Custom metadata for the session
- `tools`: Tools available to the session

## Implementation Details

The `Session` class in AbstractLLM is implemented in `abstractllm/session.py` and provides:

```python
# From abstractllm/session.py
class Session:
    """
    A session for interacting with LLMs.
    
    Sessions maintain conversation history and can be persisted and loaded.
    """
    
    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 provider: Optional[Union[str, AbstractLLMInterface]] = None,
                 provider_config: Optional[Dict[Union[str, ModelParameter], Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 tools: Optional[List[Union[Dict[str, Any], Callable, "ToolDefinition"]]] = None):
        """Initialize a session."""
        # Implementation details...
```

Key methods include:
- `add_message()`: Add a message to the session history
- `get_history()`: Get the conversation history
- `generate()`: Generate a response considering the conversation history
- `generate_with_tools()`: Generate a response using tools
- `save()`/`load()`: Save and load sessions
- `clear_history()`: Clear the conversation history

## Best Practices

### Handling Long Conversations

As conversations grow longer, they may exceed the context window of the model. Consider:

1. **Summarization**: Periodically summarize the conversation to reduce token usage
   ```python
   # Create a summary of the current conversation
   summary_prompt = "Summarize our conversation so far in a concise way."
   summary = session.generate(summary_prompt)
   
   # Create a new session with the summary
   new_session = Session(
       system_prompt=f"You are a helpful assistant. Previous conversation summary: {summary}",
       provider=provider
   )
   ```

2. **Selective History**: Only include relevant messages for certain queries
   ```python
   # Get full history
   history = session.get_history()
   
   # Select relevant messages
   relevant_messages = [msg for msg in history if "topic" in msg.content]
   
   # Create targeted prompt
   targeted_prompt = "Based on our discussion about the topic, answer this question..."
   ```

### Provider-Specific Considerations

Different providers handle session context differently:

1. **OpenAI**: Supports system messages and long context windows
2. **Anthropic**: Has specific formatting for system prompts and multi-turn conversations
3. **Ollama**: May have more limited context windows depending on the model
4. **HuggingFace**: Context size varies significantly by model

## Conclusion

Sessions in AbstractLLM provide a powerful way to manage conversation state and context across multiple interactions. By leveraging sessions, you can build more contextually aware applications that maintain coherent conversations over time. 