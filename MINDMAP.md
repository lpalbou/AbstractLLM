# AbstractLLM Library Architecture Mindmap

## Overview
AbstractLLM is a unified interface library for interacting with multiple Large Language Model providers. It abstracts the differences between providers while maintaining access to provider-specific features.

## Core Architecture

### 1. Factory Pattern (`factory.py`)
```
create_llm(provider, **config) -> AbstractLLMInterface
├── Provider validation & dependency checking
├── Platform requirements validation (e.g., MLX on Apple Silicon)
├── API key management (env vars, explicit)
├── Registry-based provider resolution
└── Fallback to hardcoded provider mapping
```

### 2. Abstract Interface (`interface.py`)
```
AbstractLLMInterface (ABC)
├── generate(prompt, system_prompt, files, stream, tools, **kwargs)
├── generate_async(...)
├── get_capabilities() -> Dict[ModelCapability, Any]
├── Configuration management via ConfigurationManager
└── Provider-specific parameter handling
```

### 3. Provider Registry System (`providers/registry.py`)
```
Dynamic Provider Loading
├── register_provider(name, module_path, class_name)
├── get_provider_class(name) -> lazy import
├── Platform-specific registration (MLX checks)
└── Dependency validation before registration
```

### 4. Base Provider (`providers/base.py`)
```
BaseProvider(AbstractLLMInterface)
├── Tool validation & processing
├── Tool call extraction from responses
├── Response processing & standardization
└── Common functionality across providers
```

## Provider Implementations

### Supported Providers
- **OpenAI** (`openai.py`) - GPT models, vision support, tool calling
- **Anthropic** (`anthropic.py`) - Claude models, tool calling
- **Ollama** (`ollama.py`) - Local models, self-hosted
- **HuggingFace** (`huggingface.py`) - Open source models
- **MLX** (`mlx_provider.py` + `mlx_model_configs.py`) - Apple Silicon optimized models

### MLX Provider Architecture

The MLX provider uses a **two-file architecture** that's justified by its complexity:

```
MLX Provider Structure
├── mlx_provider.py - Core provider logic
│   ├── Model loading (text & vision models) ✅ COMPLETE
│   ├── Generation (sync/async, streaming) ✅ COMPLETE
│   ├── Apple Silicon platform checks ✅ COMPLETE
│   ├── Vision model detection & processing ✅ COMPLETE
│   └── Tool support (basic via prompt engineering) ✅ COMPLETE
└── mlx_model_configs.py - Model-specific configurations ✅ COMPLETE
    ├── ModelConfigFactory - Maps model names to configs ✅ COMPLETE
    ├── Base MLXModelConfig class ✅ COMPLETE
    ├── Model families: Llama, Qwen, Mistral, Phi, etc. ✅ COMPLETE
    ├── Vision models: PaliGemma, Qwen2VL, etc. ✅ COMPLETE
    └── Format handlers: chat templates, tokens, generation params ✅ COMPLETE
```

**✅ MLX Provider Status: COMPLETE AND THOROUGHLY TESTED**

The MLX provider now works with the **same simple pattern** as other providers and has been verified with comprehensive testing:

**Core Functionality:**
```python
# Simple usage - works exactly like Ollama!
provider = create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
response = provider.generate("Hello!")
```

**✅ Verified Capabilities (all tested and working):**
- ✅ **Basic Generation**: Standard text generation
- ✅ **Streaming Generation**: Real-time token streaming  
- ✅ **Async Generation**: Asynchronous generation support
- ✅ **System Prompts**: System prompt handling (**FIXED**: No longer ignored)
- ✅ **Tool Calling**: Basic tool support via prompt engineering (**FIXED**: No infinite loop)
- ✅ **Vision Models**: Vision model detection and capabilities
- ✅ **File Handling**: File input processing

**🐛 BUGS FIXED:**
- ✅ **Infinite Loop in Tool Calling**: System prompts were being ignored, causing Session to loop indefinitely waiting for tool calls that never happened. **FIXED** by properly passing and formatting system prompts in `_generate_text()` and `_generate_text_stream()` methods.
- ✅ **System Prompt Ignored**: The `_generate_text` methods weren't using the system_prompt parameter. **FIXED** by formatting system and user prompts together using model-specific chat templates.

**⚠️ Known Limitations:**
- **Tool calling is "basic"**: Uses prompt engineering, not structured API like OpenAI/Anthropic
- **Response formatting**: May include chat template artifacts in responses
- **Tool execution**: Model shows "thinking" process rather than clean tool execution

### MLX-Specific Features
```
Capabilities
├── Text-only models (via mlx-lm)
├── Vision models (via mlx-vlm) 
├── Streaming generation (stream_generate)
├── Model-specific optimizations
├── Apple Silicon platform validation
├── Automatic model config detection
└── Basic tool support (prompt engineering)
```

### Provider Features
```
Each Provider Implements:
├── Model-specific configuration & defaults
├── Provider-specific API format conversion
├── Error handling & exception mapping
├── Capability reporting (streaming, vision, tools)
├── Tool calling format conversion
└── Response standardization
```

## Session Management (`session.py`)

### Session Class
```
Session
├── Conversation history management
├── Provider switching mid-conversation
├── Tool function management & execution
├── Automatic tool calling workflows
├── Save/load conversation state
└── Metadata tracking
```

### Key Features
- **Provider Interchangeability**: Switch providers while maintaining context
- **Tool Execution**: Automatic tool calling with function registration
- **Streaming Support**: Real-time response generation
- **Persistence**: Save/restore conversation sessions

## Tool System (`tools/`)

### Architecture
```
Tool System
├── Function-to-tool conversion (docstring parsing)
├── Tool definition validation (JSON schema)
├── Provider-specific format conversion
├── Automatic tool execution in sessions
└── Tool call result processing
```

### Workflow
1. Functions with type hints → Tool definitions
2. Tool definitions → Provider-specific format
3. LLM response → Tool call extraction
4. Tool execution → Results back to LLM

## Type System

### Core Types (`types.py`)
```
GenerateResponse
├── content: Optional[str]
├── raw_response: Any
├── usage: Optional[Dict[str, int]]
├── tool_calls: Optional[ToolCallRequest]
└── has_tool_calls() -> bool

Message
├── role: Union[str, MessageRole]
├── content: str
├── tool_results: Optional[List[Dict]]
└── to_dict() / from_dict()
```

### Enums (`enums.py`)
```
ModelParameter - Configuration parameters
├── Basic: temperature, max_tokens, model, api_key
├── Advanced: top_p, frequency_penalty, seed
├── Vision: image, images, image_detail
└── Tools: tools, tool_choice

ModelCapability - Provider capabilities
├── Basic: streaming, async, system_prompt
├── Advanced: function_calling, vision, tool_use
└── Specialized: fine_tuning, embeddings, json_mode

MessageRole - Conversation roles
├── system, user, assistant, tool
```

## Configuration Management (`utils/config.py`)

### Features
- Parameter validation and type checking
- Default value management per provider
- Runtime configuration updates
- Provider-specific parameter mapping

## Media Support (`media/`)

### Vision Capabilities
- Image processing for vision models
- Multi-modal content handling
- File type detection and processing
- URL and local file support

## Key Design Principles

### 1. Provider Abstraction
- Consistent API across all providers
- Provider-specific features accessible via kwargs
- Graceful degradation for unsupported features

### 2. Lazy Loading
- Providers loaded on-demand
- Dependency checking before import
- Platform-specific provider registration

### 3. Extensibility
- Registry system for custom providers
- Tool system for function calling
- Configuration system for customization

### 4. Error Handling
- Standardized exceptions across providers
- Clear error messages with resolution hints
- Graceful fallbacks where possible

### 5. Developer Experience
- Type-safe parameters via enums
- Comprehensive logging and debugging
- Clear documentation and examples
- Optional dependencies with helpful error messages

## Usage Patterns

### Basic Usage
```python
# Simple generation
llm = create_llm("openai", api_key="...")
response = llm.generate("Hello!")

# Provider switching
openai_llm = create_llm("openai")
claude_llm = create_llm("anthropic")
```

### Session-based Usage
```python
# Stateful conversation
session = Session(system_prompt="...", provider=llm)
session.add_message("user", "Hello")
response = session.generate("How are you?")
```

### Tool Calling
```python
# Function registration
def get_weather(location: str) -> str:
    return f"Weather in {location}"

session = Session(tools=[get_weather])
response = session.generate_with_tools("What's the weather in Paris?")
```

## Dependencies & Installation

### Core Dependencies
- `pydantic` - Configuration and validation
- `typing-extensions` - Enhanced typing

### Provider Dependencies
- `openai` - OpenAI API
- `anthropic` - Anthropic API  
- `torch`, `transformers` - HuggingFace
- `mlx`, `mlx-lm` - Apple Silicon MLX
- `requests`, `aiohttp` - Ollama

### Optional Dependencies
- `docstring-parser`, `jsonschema` - Tool support
- `pillow` - Image processing

This architecture provides a clean abstraction layer while maintaining flexibility and extensibility for different LLM providers and use cases. 