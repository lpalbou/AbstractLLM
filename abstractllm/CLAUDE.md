# AbstractLLM Core

## Overview
AbstractLLM is a unified interface framework for Large Language Models, providing seamless interoperability between OpenAI, Anthropic, Ollama, HuggingFace, and MLX providers. It abstracts provider differences while maintaining access to unique capabilities through a consistent, extensible architecture.

## Code Quality Assessment
**Overall Rating: 9/10** ⬆️

### Strengths
- Clean, modular architecture with clear separation of concerns
- Excellent use of design patterns (Factory, Strategy, Registry, Adapter)
- Comprehensive error handling with meaningful exceptions
- Strong typing with Pydantic and type hints throughout
- Provider-agnostic design enables true interchangeability
- Well-documented with extensive docstrings
- Extensible plugin-like architecture
- **NEW**: Simplified universal tool system

### Issues
- Some components have grown large (MLX provider 1400+ lines)
- Missing model_capabilities.json path issue
- Some complex functions need refactoring
- Limited async support in some areas

## System Architecture Mindmap
```
AbstractLLM Framework
├── Core Layer
│   ├── Interface (interface.py)
│   │   └── AbstractLLMInterface (ABC)
│   │       ├── generate() / generate_async()
│   │       ├── Configuration management
│   │       └── Capability reporting
│   │
│   ├── Factory (factory.py)
│   │   ├── create_llm() entry point
│   │   ├── Provider validation
│   │   ├── Dependency checking
│   │   └── API key management
│   │
│   ├── Session (session.py)
│   │   ├── Conversation history
│   │   ├── Tool execution
│   │   ├── Provider switching
│   │   └── Save/load state
│   │
│   └── Type System
│       ├── types.py (Response, Message)
│       ├── enums.py (Parameters, Capabilities)
│       └── exceptions.py (Error hierarchy)
│
├── Provider Layer (providers/)
│   ├── Base Infrastructure
│   │   ├── BaseProvider (common functionality)
│   │   └── Registry (dynamic loading)
│   │
│   └── Implementations
│       ├── OpenAI (GPT-3.5/4, vision, tools)
│       ├── Anthropic (Claude 3/3.5, vision, tools)
│       ├── Ollama (local models, streaming)
│       ├── HuggingFace (transformers, GGUF)
│       └── MLX (Apple Silicon, vision)
│
├── Intelligence Layer
│   ├── Architectures (architectures/)
│   │   ├── Detection (pattern matching)
│   │   ├── Capabilities (per model)
│   │   ├── Templates (chat formatting)
│   │   └── Enums (tool formats, types)
│   │
│   └── Model Data (assets/)
│       ├── architecture_formats.json
│       └── model_capabilities.json
│
├── Extension Layer
│   ├── Tools (tools/) ⚡ REWRITTEN
│   │   ├── Core types (ToolDefinition, ToolCall)
│   │   ├── Universal handler (all models)
│   │   ├── Architecture-based parser
│   │   └── Tool registry & execution
│   │
│   └── Media (media/)
│       ├── Image processing
│       ├── Text handling
│       └── Tabular data
│
└── Support Layer (utils/)
    ├── Configuration management
    ├── Rich formatting
    ├── Logging system
    ├── Model capabilities
    └── Token counting
```

## Component Quality Summary

| Component | Rating | Status | Key Updates |
|-----------|--------|--------|-------------|
| **Core** | 9/10 | Excellent | Minor refactoring needed |
| **Providers** | 8/10 | Good | MLX provider too large |
| **Architectures** | 9/10 | Excellent | Clean separation HOW/WHAT |
| **Tools** | 10/10 ⬆️ | Perfect | Complete rewrite, minimal & clean |
| **Media** | 9/10 | Excellent | Missing async |
| **Utils** | 8/10 | Good | Wrong asset path |
| **Assets** | 8.5/10 | Very Good | Well-structured JSONs |

## Recent Tool System Improvements

### Before (6 files, complex):
- types.py, validation.py, conversion.py
- modular_prompts.py, architecture_tools.py
- universal_tools.py
- Circular imports, code duplication

### After (4 files, simple):
- **core.py**: Clean type definitions
- **handler.py**: Universal tool handler
- **parser.py**: Architecture-aware parsing
- **registry.py**: Tool management
- No circular imports, minimal API

### New Tool Usage
```python
from abstractllm.tools import create_handler, register

@register
def search(query: str) -> str:
    return f"Results for: {query}"

handler = create_handler("gpt-4")
request = handler.prepare_request(tools=[search])
```

## Key Design Patterns
1. **Abstract Factory**: Provider creation through unified factory
2. **Strategy**: Providers implement common interface differently
3. **Registry**: Dynamic provider discovery and loading
4. **Adapter**: Provider-specific API adaptation
5. **Session**: Stateful conversation management
6. **Template Method**: Base provider defines algorithm
7. **Facade**: Media processor simplifies complex operations

## Integration Flow
```python
# 1. Create provider
llm = create_llm("openai", model="gpt-4")

# 2. Direct use
response = llm.generate("Hello")

# 3. Session use with tools
from abstractllm.tools import register

@register
def get_time() -> str:
    return "2:30 PM"

session = Session(provider=llm, tools=[get_time])
response = session.generate("What time is it?")

# 4. Provider switching
session.set_provider(create_llm("anthropic"))
response = session.generate("Continue...")
```

## Critical Issues to Address
1. **Fix model_capabilities.json path** in utils/model_capabilities.py
2. **Split MLX provider** into multiple modules
3. **Add cleanup** for logging._pending_requests memory leak
4. **Refactor complex functions** (get_session_stats, etc.)

## Recommendations
1. **Immediate Actions**:
   - Fix capability file path issue
   - Add memory cleanup in logging
   
2. **Short-term Improvements**:
   - Split MLX provider into 3-4 modules
   - Add provider health checks
   
3. **Long-term Enhancements**:
   - Plugin system for custom providers
   - Visual tool builder interface
   - Performance monitoring dashboard
   - Cost tracking across providers

## Security Considerations
- API keys managed securely via environment variables
- No credential logging or exposure
- Input validation throughout
- Sandboxed tool execution
- Timeout protections

## Performance Notes
- Lazy provider loading minimizes startup time
- Efficient streaming reduces memory usage
- Caching in multiple layers (templates, tokens, media)
- Could benefit from async improvements

## Maintenance Guidelines
1. **Adding Providers**: Implement AbstractLLMInterface, register in registry
2. **Adding Tools**: Use @register decorator, auto-converts to ToolDefinition
3. **Adding Architectures**: Update detection patterns in architectures/
4. **Testing**: Use real examples, never mock critical paths

## Conclusion
AbstractLLM demonstrates mature software engineering with a well-architected, extensible design. The recent tool system rewrite exemplifies the commitment to simplicity and clean code. With the recommended fixes, this framework provides an excellent foundation for unified LLM interaction across multiple providers.

## Quick Reference
- **Entry Point**: `create_llm(provider, **config)`
- **Main Interface**: `AbstractLLMInterface`
- **Stateful Usage**: `Session` class
- **Tool System**: `@register` decorator + `create_handler()`
- **Provider Count**: 5 (OpenAI, Anthropic, Ollama, HuggingFace, MLX)
- **Architecture Count**: 10+ detected architectures
- **Tool Support**: Universal (native or prompted)
- **Vision Support**: Yes (provider and model-dependent)

## Task Completion Summary
✅ Investigated architecture detection and model capabilities
✅ Analyzed existing tool implementation (found 6 files with duplication)
✅ Designed minimal tool system (4 clean files)
✅ Implemented new system with universal support
✅ Deleted redundant files
✅ Updated documentation

The tool system now provides clean, universal support for all models through a minimal set of well-designed components.

## Session Update (2025-01-06)

### Issue: Tool Support Rejection for Qwen3 Model
**Problem**: MLX provider was rejecting tool calls for `mlx-community/Qwen3-30B-A3B-4bit` even though Qwen3 models have native tool support.

**Root Cause**: Model name normalization mismatch
- The model name `mlx-community/Qwen3-30B-A3B-4bit` was normalized to `qwen3-30b-a3b` (removing the `-4bit` suffix)
- But the model_capabilities.json had the key as `qwen3-30B-A3B-4bit` (with the suffix)
- This caused the capability lookup to fail, defaulting to "no tool support"

**Fix Applied**:
1. Fixed import errors in mlx_provider.py from the previous refactoring
2. Changed the model key in model_capabilities.json from `qwen3-30B-A3B-4bit` to `qwen3-30b-a3b` to match the normalized name

**Lessons Learned**:
- Model capability keys in JSON should match the normalized form
- The normalization strips quantization suffixes like `-4bit`, `-8bit`, etc.
- ALL models can support tools through prompting - the framework should never completely reject tools

**Principle Violated**: The framework incorrectly assumed some models can't use tools at all. In reality, ANY model can support tools through careful prompting, even if they don't have native tool APIs.

### Tool Format Issue for Qwen3 in MLX
**Problem**: Qwen3 model wasn't outputting tool calls in the correct format - it was using plain function syntax instead of `<|tool_call|>` format.

**Root Causes**:
1. Model was marked as having "native" tool support, but MLX provider doesn't support native tool APIs
2. alma-minimal.py was manually describing tools in the system prompt, conflicting with framework's tool formatting

**Fixes Applied**:
1. Changed `qwen3-30b-a3b` in model_capabilities.json from "native" to "prompted" tool support
2. Removed manual tool descriptions from alma-minimal.py system prompt
3. Now the framework properly adds Qwen-specific tool formatting instructions

**Key Learning**: Provider-specific capabilities matter - a model might support native tools through its official API but need prompted tools when running through MLX or other local providers.

### Tool Parsing Bug for Qwen Format
**Problem**: Tool calls were being detected but not parsed correctly - the regex pattern couldn't handle nested JSON in the `<|tool_call|>` format.

**Root Cause**: The regex `r'<\|tool_call\|>\s*(\{.*?\})'` was using non-greedy matching that stopped at the first `}`, breaking on nested JSON like `{"arguments": {"recursive": true}}`.

**Fix Applied**: Updated the pattern to match content between `<|tool_call|>` tags properly, handling both with and without closing tags.

**Final Result**: Complete tool execution flow now works:
1. Model emits: `<|tool_call|>{"name": "list_files", "arguments": {"recursive": true}}</|tool_call|>`
2. Parser extracts the tool call
3. Session executes the tool
4. Results are returned to the model
5. Model presents formatted results to the user

## Session Update (2025-01-06) - Part 2

### Ollama Provider Tool Support
**Problem**: Ollama was rejecting tools for the same Qwen model that worked in MLX.

**Root Causes**:
1. Legacy function `supports_tool_calls` was checking for "tool_calling" capability, but JSON uses "tool_support"
2. Ollama model names use `:` format (e.g., `qwen3:30b-a3b-q4_K_M`) which wasn't normalized properly

**Fixes Applied**:
1. Updated `supports_tool_calls` to use the correct architecture detection function
2. Enhanced model name normalization to convert Ollama's `:` format to standard `-` format
3. Now both `mlx-community/Qwen3-30B-A3B-4bit` and `qwen3:30b-a3b-q4_K_M` normalize to `qwen3-30b-a3b`

### Robust Tool Parsing
**Problem**: Models sometimes forget closing tags but still produce valid JSON tool calls.

**Example**:
```
<|tool_call|>
{"name": "read_file", "arguments": {"file_path": "..."}}
<|tool_call|>
{"name": "read_file", "arguments": {"file_path": "..."}}
```
(Missing `</|tool_call|>` closing tags)

**Fix Applied**: Enhanced all parser functions to:
1. First try to find properly closed tags
2. Fallback to finding opening tags followed by valid JSON
3. Use duplicate detection to avoid parsing the same call twice
4. Made detection more lenient - only requires opening tags

**Result**: Tool parsing now gracefully handles edge cases where models forget closing tags, ensuring tool calls are still executed correctly across all providers.

### Duplicate Tool Call Parsing
**Problem**: Parser was deduplicating identical tool calls, preventing models from intentionally calling the same tool multiple times.

**Example**:
```
<|tool_call|>
{"name": "read_file", "arguments": {"file_path": "...", "should_read_entire_file": true}}
<|tool_call|>
{"name": "read_file", "arguments": {"file_path": "...", "should_read_entire_file": true}}
```
Only one call was being parsed instead of two.

**Fix Applied**: Removed deduplication logic from all parser functions. Now uses position-based overlap detection to avoid parsing the same text twice while allowing multiple identical tool calls.

**Note on Tool Call Limits**: The session has a `max_tool_calls` parameter (default 10, but alma-minimal.py sets it to 25) to prevent infinite loops. If a model repeatedly calls tools, it may hit this limit and stop executing further calls.

### Ollama Provider System Prompt Issue
**Problem**: User reported that Ollama was not receiving the system prompt, while MLX was.

**Investigation**: 
1. Fixed undefined `messages` variable in use_chat_endpoint check
2. Added request interception to verify actual API calls
3. Tested with explicit BANANAS system prompt requirement

**Findings**:
- The system prompt IS being sent correctly to Ollama:
  - Without tools: Uses `/api/generate` with `"system"` field
  - With tools: Uses `/api/chat` with system message in messages array
- The confusion arose from the logging system:
  - `log_request` only logs metadata (`has_system_prompt: true`)
  - The actual system prompt content is not included in the interaction logs
  - This makes it appear that no system prompt was sent, but it actually is
- Models DO follow the system prompt correctly when sent through Ollama

**Resolution**: The real issue was that Ollama provider didn't support the `messages` parameter that Session uses for conversation history in ReAct loops.

### Ollama Messages Parameter Support (Fixed)
**Problem**: Session's ReAct loop passes conversation history via `messages` parameter, but Ollama ignored it.

**Root Cause**: 
- Session passes `messages` with tool results to maintain conversation context
- Ollama's generate() method ignored the `messages` parameter completely
- Each iteration only saw the original prompt with no tool results
- Model kept trying the same tools repeatedly until hitting the 25 iteration limit

**Fix Applied**:
1. Extract `messages` from kwargs in both sync and async generate methods
2. Use chat endpoint when messages are provided
3. Update `_prepare_request_for_chat` to accept and use provided messages
4. Ensure enhanced system prompt (with tool instructions) is preserved when using messages

**Result**: Ollama now maintains conversation context across tool iterations, enabling proper ReAct loop execution.