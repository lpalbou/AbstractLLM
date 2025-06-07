# Utils Component

## Overview
The utils module provides essential supporting functionality for AbstractLLM, including configuration management, output formatting, logging, model capability detection, and general utilities. It forms the foundation layer that all other components rely on.

## Code Quality Assessment
**Rating: 8/10**

### Strengths
- Clean, well-documented utility functions
- Excellent formatting system with rich terminal output
- Sophisticated logging with separate console/file control
- Smart configuration management with parameter precedence
- Comprehensive session analytics
- Good error handling throughout

### Issues
- model_capabilities.py has wrong path (looks for missing assets/model_capabilities.json)
- Memory leak potential in logging._pending_requests
- Some functions too complex (get_session_stats)
- Hardcoded values in token counter
- Provider-specific logic in generic config module

## Component Mindmap
```
Utils Module
├── Configuration (config.py)
│   ├── ConfigurationManager class
│   │   ├── Parameter storage (enum/string keys)
│   │   ├── Default management
│   │   ├── Update/merge operations
│   │   └── Provider param extraction
│   └── Integration with ModelParameter enum
│
├── Formatting (formatting.py) ⭐
│   ├── Response Parsing
│   │   ├── parse_response_content()
│   │   ├── Extract <think>/<answer> tags
│   │   └── Clean output formatting
│   ├── Display Functions
│   │   ├── format_message() - Role-based colors
│   │   ├── format_response() - Stream/complete
│   │   ├── format_tools() - Tool definitions
│   │   └── format_session_state() - Full state
│   └── ANSI Color Support
│
├── Logging (logging.py)
│   ├── LogConfig singleton
│   │   ├── Independent console/file levels
│   │   ├── Suppress third-party warnings
│   │   └── Global enable/disable
│   ├── Custom Formatters
│   │   ├── ColoredTruncatingFormatter
│   │   └── PlainFormatter
│   ├── Request/Response Tracking
│   │   └── Match interactions with IDs
│   └── Sensitive Data Handling
│
├── Model Capabilities (model_capabilities.py) ⚠️
│   ├── Capability Detection
│   │   ├── has_capability(model, capability)
│   │   ├── get_model_capabilities(model)
│   │   └── get_models_with_capability(capability)
│   ├── Model Normalization
│   │   └── Strip size/quant/provider prefixes
│   ├── Caching System
│   └── ❌ BROKEN: Wrong JSON path
│
└── Utilities (utilities.py)
    ├── TokenCounter class
    │   ├── Cached tokenizer loading
    │   ├── Multiple encoding support
    │   └── Fallback estimation
    ├── Session Statistics
    │   ├── Message counts by role
    │   ├── Token usage tracking
    │   ├── Tool call analysis
    │   └── Timing information
    └── Platform Detection
        └── is_apple_silicon()
```

## Key Features

### Configuration Management
```python
# Clean parameter handling
config = ConfigurationManager()
config.set(ModelParameter.TEMPERATURE, 0.7)
config.update({"max_tokens": 1000})
params = config.get_provider_params("openai")
```

### Rich Formatting
```python
# Beautiful terminal output
formatted = format_response(response, provider="openai")
# Shows: role colors, token counts, tool calls, content
```

### Flexible Logging
```python
# Separate console/file control
setup_logging(console_level="INFO", file_level="DEBUG")
log_request(provider, prompt, params)
log_response(provider, response, duration)
```

### Model Intelligence
```python
# Capability checking (when fixed)
if has_capability("llama3.2", "tool_calling"):
    # Enable tools
```

## Dependencies
- **Required**: Standard library only
- **Optional**: 
  - `tiktoken`: OpenAI tokenization
  - `transformers`: HF tokenization

## Critical Issues
1. **model_capabilities.json path wrong**: Update line 31 to correct path
2. **Memory leak in logging**: _pending_requests grows unbounded
3. **Hardcoded tokenizer model**: Should be configurable

## Recommendations
1. **Fix capabilities path**: Point to correct model_capabilities.json location
2. **Add request cleanup**: Expire old pending requests in logging
3. **Extract provider logic**: Move provider-specific code from config.py
4. **Simplify complex functions**: Break down get_session_stats
5. **Add validation**: Parameter value validation in ConfigurationManager

## Technical Debt
- Wrong asset path in model_capabilities.py
- Complex session stats function (100+ lines)
- No cleanup for logging request tracking
- Hardcoded model in TokenCounter
- Missing parameter validation

## Performance Notes
- Tokenizer caching prevents repeated downloads
- Capability data cached after first load
- Efficient string formatting with f-strings
- Could benefit from lazy imports

## Security Considerations
- Sensitive data masked in logs (API keys, base64)
- No execution of user input
- Safe file path handling
- No credential storage

## Future Enhancements
1. **Dynamic capabilities**: Runtime capability discovery
2. **Structured logging**: JSON output option
3. **Performance metrics**: Detailed timing breakdowns
4. **Configuration schemas**: Validate parameters
5. **Plugin system**: Extensible formatters/loggers

## Integration Points
- Used by ALL components for configuration
- Logging used throughout for debugging
- Formatting essential for CLI experience
- Capabilities inform provider behavior
- Token counting for usage tracking