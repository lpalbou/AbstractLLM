# Tool Architecture Across Providers

**Date**: September 15, 2025
**Author**: AbstractLLM Analysis
**Status**: Complete Implementation Analysis

## Executive Summary

This report provides a comprehensive analysis of how tool definitions are sent to Large Language Models across all providers in AbstractLLM. The system uses an intelligent dual-strategy approach: **native API tools** for providers that support them, and **system prompt injection** for those that don't.

## Universal Architecture

### BaseProvider Strategy

All providers inherit from `BaseProvider` which implements the `_prepare_tool_context()` method. This method automatically determines the optimal tool delivery strategy:

```python
def _prepare_tool_context(self, tools, system_prompt):
    if handler.supports_native:
        return system_prompt, formatted_tools, "native"
    elif handler.supports_prompted:
        enhanced = f"{system_prompt}\n\n{tool_prompt}"
        return enhanced, None, "prompted"
    else:
        raise UnsupportedFeatureError("tools", ...)
```

### Decision Matrix

| Provider | Primary Strategy | Fallback | Determination Method |
|----------|-----------------|----------|---------------------|
| **OpenAI** | Native API | None | Always native |
| **Anthropic** | Native API | None | Always native |
| **Ollama** | Model-dependent | Prompted | `model_capabilities.json` |
| **MLX** | Prompted | None | Always prompted |

## Provider-Specific Implementation

### 1. OpenAI Provider

**Strategy**: ✅ **Native Tools Parameter**

**Implementation**:
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are an intelligent AI assistant..."},
    {"role": "user", "content": "User query"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
          "type": "object",
          "properties": {
            "file_path": {"type": "string"},
            "should_read_entire_file": {"type": "boolean"}
          },
          "required": ["file_path"]
        }
      }
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2048
}
```

**Key Points**:
- Tools sent as dedicated `tools` parameter
- System prompt remains unchanged
- OpenAI handles tool parsing natively
- Supports parallel tool calls
- Tools never appear in message content

### 2. Anthropic Provider

**Strategy**: ✅ **Native Tools Parameter**

**Implementation**:
```json
{
  "model": "claude-3-5-haiku-latest",
  "system": "You are an intelligent AI assistant with memory and reasoning capabilities.",
  "messages": [
    {"role": "user", "content": "User query"}
  ],
  "tools": [
    {
      "name": "read_file",
      "description": "Read the contents of a file with optional line range.",
      "input_schema": {
        "type": "object",
        "properties": {
          "file_path": {"type": "string"},
          "should_read_entire_file": {"type": "boolean"}
        },
        "required": ["file_path"]
      }
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.7
}
```

**Key Points**:
- Tools sent as dedicated `tools` parameter
- Uses Anthropic's `input_schema` format (not `parameters`)
- System prompt sent separately as `system` parameter
- Native tool call parsing
- Tools never appear in message content

### 3. Ollama Provider

**Strategy**: ⚡ **Hybrid Approach** (Model-Dependent)

#### Option A: Native Mode (Supported Models)
```json
{
  "model": "llama3.1:8b",
  "messages": [
    {"role": "system", "content": "You are an intelligent AI assistant..."},
    {"role": "user", "content": "User query"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {...}
      }
    }
  ]
}
```

#### Option B: Prompted Mode (Unsupported Models)
```json
{
  "model": "qwen3-coder:30b",
  "messages": [
    {
      "role": "system",
      "content": "You are an intelligent AI assistant with memory and reasoning capabilities.\n\nYou are a helpful AI assistant with tool access.\n\nAvailable tools:\n[\n  {\n    \"name\": \"read_file\",\n    \"description\": \"Read the contents of a file with optional line range.\",\n    \"parameters\": {\n      \"type\": \"object\",\n      \"properties\": {\n        \"file_path\": {\"type\": \"string\"},\n        \"should_read_entire_file\": {\"type\": \"boolean\"}\n      },\n      \"required\": [\"file_path\"]\n    }\n  }\n]\n\nEXAMPLES:\nread_file - Read file contents\nExample 1: <|tool_call|>{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"example.txt\"}}</|tool_call|>\n\nCRITICAL: When using tools, you MUST use the exact format shown above."
    },
    {"role": "user", "content": "User query"}
  ]
}
```

**Key Points**:
- **Decision Logic**: Checks `model_capabilities.json` for `tool_support` value
- **Native Models**: Models with `"tool_support": "native"` use dedicated tools parameter
- **Prompted Models**: Models with `"tool_support": "prompted"` get tools in system prompt
- **Automatic Detection**: No manual configuration required

### 4. MLX Provider

**Strategy**: ✅ **System Prompt Injection** (Always Prompted)

**Implementation**:
```json
{
  "model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
  "prompt": "<|system|>\nYou are an intelligent AI assistant with memory and reasoning capabilities.\n\nYou are a helpful AI assistant with tool access.\n\nAvailable tools:\n[\n  {\n    \"name\": \"read_file\",\n    \"description\": \"Read the contents of a file with optional line range.\",\n    \"parameters\": {...}\n  }\n]\n\nEXAMPLES:\nread_file - Read file contents\nExample 1: <|tool_call|>{\"name\": \"read_file\", \"arguments\": {\"file_path\": \"example.txt\"}}</|tool_call|>\n\nCRITICAL: When using tools, you MUST use the exact format shown above.\n<|user|>\nUser query<|assistant|>\n",
  "max_tokens": 2048,
  "temperature": 0.7
}
```

**Key Points**:
- **Always prompted mode**: Local models lack native tool APIs
- **Chat template formatting**: Uses model-specific chat template
- **Architecture-aware**: Different formats for different model architectures (Qwen, Llama, etc.)
- **Single prompt**: Everything combined into one formatted prompt

## Model-Specific Analysis

### Ollama qwen3-coder:30b

**Answer to Your Question**:

✅ **PROMPTED MODE**: According to `model_capabilities.json`:
```json
"qwen3-coder-30b": {
  "context_length": 32768,
  "max_output_tokens": 8192,
  "tool_support": "prompted",
  "structured_output": "prompted",
  "parallel_tools": true,
  "max_tools": -1,
  "vision_support": false,
  "audio_support": false,
  "notes": "Code-focused model with prompted tool support",
  "source": "Alibaba official docs"
}
```

✅ **YES, APPEARS IN /context**: Since it uses prompted mode, the tool definitions ARE injected into the system prompt and will be visible in the `/context` command as part of the system message content.

**Example /context output for qwen3-coder:30b**:
```
╔══════════════ EXACT VERBATIM LLM INPUT ══════════════╗
║ Timestamp: 2025/09/15 14:30:45
║ Model: qwen3-coder:30b
║ Provider: ollama
╚═══════════════════════════════════════════════════════╝

ENDPOINT: http://localhost:11434/api/chat

REQUEST PAYLOAD:
{
  "model": "qwen3-coder:30b",
  "messages": [
    {
      "role": "system",
      "content": "You are an intelligent AI assistant...\n\nYou are a helpful AI assistant with tool access.\n\nAvailable tools:\n[...full tool definitions...]\n\nEXAMPLES:\n..."
    },
    {
      "role": "user",
      "content": "User query"
    }
  ]
}
```

## Tool Definition Persistence & Visibility

### Critical Behavior: Tools Sent on Every Call

**⚠️ IMPORTANT**: For providers using prompted mode (Ollama prompted models, MLX), tool definitions are **ALWAYS sent with every single call**, regardless of chat history length.

#### Technical Implementation Details

**BaseProvider._prepare_tool_context() Execution**:
```python
def generate(self, prompt, tools=None, **kwargs):
    # This happens on EVERY call when tools are provided
    if tools:
        enhanced_system_prompt, tool_defs, mode = self._prepare_tool_context(tools, system_prompt)
    # Tool definitions are ALWAYS regenerated and sent
```

**Ollama Prompted Mode Behavior**:
```python
# Lines 525-582 in ollama.py - EVERY generate() call
if tools:
    enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, system_prompt)

# For conversation history (provided_messages)
if not has_system_in_provided and system_prompt:
    messages.insert(0, {"role": "system", "content": system_prompt})  # enhanced_system_prompt with tools

# Or replace existing system message
if role == 'system' and system_prompt and i == 0:
    messages.append({"role": "system", "content": system_prompt})  # enhanced_system_prompt with tools
```

**MLX Provider Behavior**:
```python
# Lines 1047 & 1291 in mlx_provider.py - EVERY generate() call
if tools:
    enhanced_system_prompt, tool_defs, mode = self._prepare_tool_context(tools, system_prompt)

# For conversation history preservation
if msg["role"] == "system":
    if not system_added:
        # Replace first system prompt with enhanced version containing tools
        formatted_messages.append({"role": "system", "content": enhanced_system_prompt})
```

### Tool Definition Visibility in /context

| Provider | Native Tools | Tools in /context | Persistence Behavior |
|----------|-------------|------------------|---------------------|
| **OpenAI** | ✅ Separate parameter | ❌ Not in messages | ✅ Sent every call |
| **Anthropic** | ✅ Separate parameter | ❌ Not in messages | ✅ Sent every call |
| **Ollama (native)** | ✅ Separate parameter | ❌ Not in messages | ✅ Sent every call |
| **Ollama (prompted)** | ❌ In system prompt | ✅ **Visible in system message** | ✅ **Sent every call** |
| **MLX** | ❌ In system prompt | ✅ **Visible in formatted prompt** | ✅ **Sent every call** |

### Conversation History Impact

**Question**: "Do tool definitions persist through long conversations?"

**Answer**: ✅ **YES, ALWAYS** - Tools are regenerated and sent on every call.

**Key Behaviors**:

1. **New Conversations**: Tools included in initial system prompt
2. **With Chat History**: Tools are ALWAYS injected into system message, either:
   - **Replacing** existing system message with enhanced version
   - **Adding** new system message if none exists
3. **Long Conversations**: No degradation - tools sent on call #1 and call #1000
4. **Message Array**: Tool definitions become part of the message array sent to LLM

**Memory Usage**: For prompted mode, tool definitions are included in EVERY API call, contributing to context length but ensuring consistent tool availability.

## Implementation Benefits

### Unified Developer Experience
- **Single API**: Developers use the same tool interface regardless of provider
- **Automatic Optimization**: System chooses the best approach for each model
- **Transparent Fallbacks**: Seamless degradation from native to prompted mode

### Provider Optimization
- **Native APIs**: Maximum efficiency and reliability when available
- **Prompted Fallback**: Universal compatibility with any model
- **Model-Specific Formatting**: Optimized examples for each architecture

### Debugging Transparency
- **Full Visibility**: `/context` command shows exactly what each model receives
- **Provider Differentiation**: Clear understanding of how each provider handles tools
- **Verbatim Capture**: Exact API payloads for debugging
- **Tool Persistence Verification**: Can confirm tools are sent on every call via `/context`

## Debugging Tool Behavior

### Using /context Command for Tool Analysis

**For Native Mode Providers (OpenAI, Anthropic)**:
```bash
alma> /context
# Tools appear in API logs but NOT in message content
# Look for: "tools": [...] in request logs
# System message shows original prompt without tool definitions
```

**For Prompted Mode Providers (Ollama prompted, MLX)**:
```bash
alma> /context
# Tools appear in FULL within system message content
# Look for: "Available tools:" in system message
# Complete tool definitions, parameters, and examples visible
```

### Tool Troubleshooting Guide

**Problem**: "Tools not working with model X"
1. Check tool mode: Look for `[TOOL SETUP] Using PROMPTED mode` or `[TOOL SETUP] Using NATIVE mode` in logs
2. Verify capability: Check `model_capabilities.json` for `tool_support` value
3. Inspect context: Use `/context` to see if tools are actually sent to model

**Problem**: "Tools disappear in long conversations"
- **This should NEVER happen** - tools are sent on every call
- Use `/context` after several exchanges to verify tools are still present
- If tools missing, it indicates a bug in the provider implementation

**Problem**: "Model can't see tool definitions"
- For prompted mode: Check `/context` - tools should be in system message
- For native mode: Tools won't be in `/context` but should be in API logs
- Verify model actually supports the claimed tool mode

### Expected Context Patterns

**Ollama qwen3-coder:30b Example**:
```
━━━ SYSTEM PROMPT ━━━
You are an intelligent AI assistant with memory and reasoning capabilities.

You are a helpful AI assistant with tool access.

Available tools:
[
  {
    "name": "read_file",
    "description": "Read the contents of a file with optional line range.",
    "parameters": {
      "type": "object",
      "properties": {
        "file_path": {"type": "string"},
        "should_read_entire_file": {"type": "boolean"}
      },
      "required": ["file_path"]
    }
  }
]

EXAMPLES:
read_file - Read file contents
Example 1: <|tool_call|>{"name": "read_file", "arguments": {"file_path": "example.txt"}}</|tool_call|>

CRITICAL: When using tools, you MUST use the exact format shown above.
```

**OpenAI Example**:
```
━━━ SYSTEM PROMPT ━━━
You are an intelligent AI assistant with memory and reasoning capabilities.

━━━ AVAILABLE TOOLS ━━━
  • read_file: Read the contents of a file with optional line range.

# Note: Full tool definitions sent separately, not in system prompt
```

## Future Considerations

### Potential Enhancements
1. **Dynamic Model Detection**: Runtime capability detection for new models
2. **Custom Tool Formats**: Provider-specific tool instruction templates
3. **Performance Optimization**: Caching of tool preparation results
4. **Hybrid Approaches**: Combining native tools with enhanced prompting

### Architecture Evolution
- **Plugin System**: Allow custom tool handlers for new providers
- **Format Registry**: Centralized tool format definitions
- **Capability Updates**: Automated model capability discovery

## Conclusion

AbstractLLM's tool architecture provides **universal tool support** while optimizing for each provider's strengths. The hybrid approach ensures maximum compatibility (prompted mode) while leveraging advanced capabilities (native mode) when available.

### Key Findings Summary

1. **Universal Persistence**: Tools are sent on **every single call** regardless of chat history length
2. **Provider Optimization**: Native APIs used when available, prompted mode as reliable fallback
3. **Complete Transparency**: `/context` command reveals exactly what each model receives
4. **Conversation Stability**: Tool availability never degrades during long conversations
5. **Debugging Support**: Clear patterns for troubleshooting tool-related issues

### Specific Answers to Key Questions

**Q: For Ollama qwen3-coder:30b, is it native or prompted mode?**
- ✅ **PROMPTED MODE** (confirmed in `model_capabilities.json`)

**Q: Will tools appear in /context for prompted mode?**
- ✅ **YES, FULLY VISIBLE** as part of enhanced system message content

**Q: Do tools persist through long conversations?**
- ✅ **YES, ALWAYS** - tools regenerated and sent on every call

**Q: Independent of chat history length?**
- ✅ **YES, COMPLETELY INDEPENDENT** - same behavior on call #1 and call #1000

### Technical Guarantees

For **prompted mode providers** (Ollama prompted models, MLX):
- Tool definitions are **always included** in system message
- **Full tool specifications** sent with every API call
- **Complete visibility** via `/context` command
- **Zero degradation** regardless of conversation length
- **Consistent availability** across all interactions

This architecture ensures that tools remain **consistently available** and **fully transparent** throughout the entire lifecycle of any conversation, providing developers with reliable tool-based AI interactions across all supported providers.