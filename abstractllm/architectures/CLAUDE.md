# Architectures Component

## Overview
The architectures module is the brain of AbstractLLM's model detection and capability system. It provides unified architecture detection, capability discovery, and chat template management across all supported providers (HuggingFace, Ollama, MLX, OpenAI, Anthropic).

## Code Quality Assessment
**Rating: 8.5/10**

### Strengths
- Excellent separation of concerns between detection, capabilities, and templates
- Multi-layered fallback strategies for robustness
- Provider-agnostic design enables true interchangeability
- Well-structured data models using dataclasses
- Performance-conscious with caching mechanisms

### Issues
- Some code duplication between `detection.py` and `unified_detection.py`
- Relationship between hardcoded configs and `model_capabilities.json` unclear
- Limited inline documentation for complex pattern matching logic
- Missing comprehensive test coverage for edge cases

## Component Mindmap
```
Architectures System
├── Architecture Detection (detection.py)
│   ├── normalize_model_name() - strips prefixes
│   ├── detect_architecture() - pattern matching
│   └── get_tool_call_format() - maps to formats
│
├── Capability System (capabilities.py)
│   ├── Architecture Level (what families CAN do)
│   │   ├── ToolCallFormat enum
│   │   ├── ArchitectureCapabilities dataclass
│   │   └── ARCHITECTURE_CAPABILITIES database
│   │
│   └── Model Level (what instances DO)
│       ├── detect_model_type() - base vs instruct
│       ├── detect_model_tool_capability()
│       ├── detect_model_vision_capability()
│       ├── detect_model_audio_capability()
│       └── detect_model_reasoning_capability()
│
├── Template Management (templates.py)
│   ├── TemplateManager class (with caching)
│   ├── Template Sources (priority order)
│   │   ├── 1. HuggingFace tokenizer (direct)
│   │   ├── 2. Harvested templates (fallback)
│   │   └── 3. Static templates (last resort)
│   └── Architecture-specific formatters
│
├── Unified Detection (unified_detection.py)
│   ├── UnifiedModelConfig dataclass
│   ├── UnifiedArchitectureDetector class
│   ├── MLX integration parameters
│   └── Vision model variant handling
│
└── Configuration (configs/)
    └── See configs/CLAUDE.md for details
```

## Supported Architectures
- **granite** - IBM's Granite models (special_tokens format)
- **qwen** - Alibaba's Qwen family (im_start_end format)
- **llama** - Meta's LLaMA models (inst_format)
- **mistral** - Mistral AI models (inst_format)
- **phi** - Microsoft's Phi models (basic format)
- **gemma** - Google's Gemma models (basic format)
- **deepseek** - DeepSeek models (im_start_end format)
- **yi** - 01.AI's Yi models (basic format)
- **claude** - Anthropic's Claude (XML format)
- **gpt** - OpenAI's GPT models (function calling)

## Tool Call Formats
```python
class ToolCallFormat(Enum):
    NONE = "none"                    # No tool support
    SPECIAL_TOKEN = "special_token"  # Uses special tokens
    XML_WRAPPED = "xml_wrapped"      # Claude-style XML
    FUNCTION_CALL = "function_call"  # OpenAI-style
    JSON_STRUCTURED = "json"         # Direct JSON
    GRANITE_STYLE = "granite"        # Granite-specific
```

## Integration Flow
```
1. Model Name Input
   ↓
2. Architecture Detection
   ├── Normalize name
   ├── Pattern match architecture
   └── Identify model type
   ↓
3. Capability Discovery
   ├── Get architecture capabilities
   ├── Check model-specific overrides
   └── Merge with model_capabilities.json
   ↓
4. Template Resolution
   ├── Try HF tokenizer
   ├── Fallback to harvested
   └── Use static if needed
   ↓
5. Unified Configuration
   └── Combine all settings for provider
```

## Dependencies
- External: None (pure Python)
- Internal: 
  - `assets/model_capabilities.json` for model-specific data
  - Imported by all providers for detection

## Recommendations
1. **Consolidate detection logic**: Merge `detection.py` and `unified_detection.py`
2. **Externalize patterns**: Move regex patterns to configuration
3. **Add architecture tests**: Create test suite for each architecture
4. **Document patterns**: Add examples of model names for each pattern
5. **Version templates**: Track template versions for compatibility

## Technical Debt
- Duplication between detection modules (~30% overlap)
- Hardcoded template strings should be externalized
- No clear migration path for deprecated architectures
- Missing telemetry for detection failures

## Security Considerations
- Template injection risks if user-provided templates allowed
- Pattern matching could be exploited with crafted model names
- No validation of template outputs

## Performance Notes
- Template caching prevents repeated tokenizer loading
- Pattern matching is efficient with early returns
- Could benefit from compiled regex patterns

## Future Enhancements
1. Plugin system for custom architectures
2. Template validation framework
3. Architecture capability inheritance
4. Dynamic capability discovery from model cards