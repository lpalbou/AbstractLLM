# Architectures Component

## Overview
The architectures module provides intelligent detection and configuration for different LLM architectures. It cleanly separates HOW to communicate with models (message formatting, templates) from WHAT models can do (capabilities like tools, vision, context limits). This separation enables AbstractLLM to properly format messages for any model while understanding its specific capabilities.

## Code Quality Assessment
**Rating: 9/10**

### Strengths
- Excellent separation of concerns (architecture vs capabilities)
- Clean, functional design with clear interfaces
- Comprehensive model detection with fallback patterns
- Well-documented functions with type hints
- Efficient caching of loaded JSON data
- Convenience functions for common queries
- Robust normalization for model name variations

### Issues
- Minor duplication in pattern matching logic
- Could benefit from more unit tests
- Some hardcoded special cases (o1, o3 models)

## Component Mindmap
```
Architectures Module
├── Core Detection (detection.py)
│   ├── Architecture Detection
│   │   ├── detect_architecture() → architecture name
│   │   ├── Pattern matching against known architectures
│   │   └── Special case handling (OpenAI o-series)
│   │
│   ├── Model Type Detection
│   │   ├── detect_model_type() → base/instruct
│   │   ├── Identifies instruction-tuned vs base models
│   │   └── Smart defaults for ambiguous cases
│   │
│   ├── Capability Lookup
│   │   ├── get_model_capabilities() → full capability dict
│   │   ├── Merges defaults with model-specific data
│   │   ├── Auto-detects embeddings support
│   │   └── Includes architecture template info
│   │
│   ├── Message Formatting
│   │   ├── format_messages() → formatted prompt
│   │   ├── Architecture-specific templates
│   │   └── Role-based prefix/suffix application
│   │
│   └── Convenience Functions
│       ├── supports_tools() → boolean
│       ├── supports_vision() → boolean
│       ├── supports_audio() → boolean
│       ├── supports_embeddings() → boolean
│       ├── get_context_limits() → {input, output}
│       └── is_instruct_model() → boolean
│
├── Data Loading
│   ├── Cached JSON loading
│   ├── architecture_formats.json
│   └── model_capabilities.json
│
└── Integration Points
    ├── Used by all providers
    ├── Session management
    └── Tool availability
```

## Key Features

### Architecture Detection
```python
# Detects HOW to format messages
arch = detect_architecture("llama-3.2-instruct")  # → "llama"
format = get_architecture_format(arch)
# Returns template with prefixes/suffixes for roles
```

### Capability Detection
```python
# Detects WHAT the model can do
caps = get_model_capabilities("gpt-4o")
# Returns: context_length, tool_support, vision_support, etc.
```

### Model Type Detection
```python
# Distinguishes base vs instruction-tuned
model_type = detect_model_type("llama-3-base")  # → "base"
is_instruct = is_instruct_model("gpt-4")  # → True
```

### Convenience Access
```python
# Simple boolean checks
if supports_tools("claude-3.5-sonnet"):
    # Enable tool calling
    
if supports_vision("llama-3.2-vision"):
    # Enable image input
    
limits = get_context_limits("gpt-4")
# → {"input": 128000, "output": 4096}
```

## Architecture vs Capabilities

This module maintains a critical distinction:

1. **Architecture** (HOW): Message formatting patterns
   - Chat templates (user/assistant prefixes)
   - System message handling
   - Tool format (JSON, XML, pythonic)
   - Defined in `architecture_formats.json`

2. **Capabilities** (WHAT): Model-specific features
   - Context lengths (input/output)
   - Tool support level (native/prompted/none)
   - Vision/audio support
   - Parallel tool execution
   - Defined in `model_capabilities.json`

## Normalization Logic

The module handles various model naming conventions:
- Removes provider prefixes (mlx-community/, huggingface/)
- Strips quantization suffixes (-4bit, -q4_k_m)
- Preserves version numbers (3.5, 2.5)
- Case-insensitive matching

## Integration Points
- **Providers**: Query capabilities before generation
- **Session**: Determines available tools
- **Factory**: Validates model selection
- **Media**: Checks vision support
- **Tools**: Verifies tool calling capability

## Special Cases
1. **OpenAI o-series**: Maps to GPT architecture
2. **Embedding models**: Auto-detected by name patterns
3. **Vision models**: Detected by name indicators (vl, vision)
4. **Base models**: Explicit indicators override defaults

## Performance Considerations
- JSON data cached after first load
- Efficient string matching with early returns
- Minimal regex usage for performance
- Lazy loading of architecture data

## Security Notes
- No code execution or evaluation
- Safe JSON parsing with error handling
- Input validation on all public functions
- No external network calls

## Testing Guidelines
1. Test with real model names from each provider
2. Verify fallback behavior for unknown models
3. Check edge cases (base models, embeddings)
4. Validate capability inheritance

## Future Enhancements
1. **Dynamic capability discovery**: Query provider APIs
2. **Model family inheritance**: Share capabilities
3. **Version-aware detection**: Handle model versions
4. **Custom architecture registration**: Plugin system
5. **Performance profiling**: Optimize pattern matching

## Maintenance Notes
- Update JSON files when new models release
- Test capability accuracy with actual models
- Keep architecture patterns minimal and specific
- Document sources for capability data

## Common Patterns

### Adding a New Architecture
```python
# In architecture_formats.json:
"new_arch": {
    "patterns": ["newmodel", "custom"],
    "message_format": "custom_format",
    "user_prefix": "[USER]",
    "assistant_prefix": "[ASSISTANT]",
    "tool_format": "json"
}
```

### Adding a New Model
```python
# In model_capabilities.json:
"newmodel-7b": {
    "context_length": 32768,
    "max_output_tokens": 4096,
    "tool_support": "prompted",
    "structured_output": "native",
    "vision_support": false,
    "source": "Official docs"
}
```

## Architecture Families

### Llama Family
- Pattern: Instruction-based with [INST] tags
- Tools: Pythonic format (function calls)
- Variants: Llama 3.x, 4, DeepSeek, Yi

### Qwen Family  
- Pattern: ChatML format with <|im_start|> tags
- Tools: JSON format
- Variants: Qwen 2.5, 3, VL models

### Claude Family
- Pattern: Human:/Assistant: format
- Tools: XML format
- All models support vision

### GPT Family
- Pattern: OpenAI chat format (API-handled)
- Tools: OpenAI functions/tools format
- Includes o1, o3 reasoning models

### Mistral Family
- Pattern: Simple [INST] tags
- Tools: JSON format
- Includes Mixtral MoE variants

## Summary
The architectures module provides the intelligence layer for AbstractLLM, enabling it to communicate effectively with any LLM by understanding both how to format messages and what capabilities each model possesses. Its clean design and comprehensive coverage make it a critical component of the framework's model-agnostic architecture.