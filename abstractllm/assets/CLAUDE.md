# Assets Component

## Overview
The assets folder contains critical configuration data for AbstractLLM's architecture detection and model capability system. It houses two essential JSON files that enable the framework to understand how to communicate with different model architectures and what capabilities specific models possess.

## Code Quality Assessment
**Rating: 8.5/10**

### Strengths
- Clean separation between architecture formats (HOW) and model capabilities (WHAT)
- Comprehensive coverage of major model families and architectures
- Well-structured JSON with clear field definitions
- Easy to update and extend for new models
- Version-control friendly format
- Detailed capability specifications from official sources

### Issues
- No schema validation for the JSON structures
- Some redundancy between the two JSON files
- Missing automated validation against actual model behavior
- Could benefit from a contribution guide

## Component Mindmap
```
Assets/
├── architecture_formats.json
│   ├── Architecture Definitions (HOW to communicate)
│   │   ├── llama (Llama family message format)
│   │   ├── qwen (ChatML-based format)
│   │   ├── mistral (Instruction format)
│   │   ├── phi (Microsoft's format)
│   │   ├── claude (Human/Assistant format)
│   │   ├── gpt (OpenAI chat format)
│   │   └── generic (Fallback format)
│   │
│   └── Each Architecture Contains:
│       ├── patterns: Model name patterns to match
│       ├── message_format: Format type identifier
│       ├── prefixes/suffixes: Role-specific tags
│       └── tool_format: How tools are formatted
│
└── model_capabilities.json
    ├── Model-Specific Capabilities (WHAT models can do)
    │   ├── OpenAI Models (gpt-4*, o1, o3)
    │   ├── Anthropic Models (claude-3*)
    │   ├── Meta Models (llama-3*, llama-4)
    │   ├── Alibaba Models (qwen2*, qwen3*)
    │   ├── Microsoft Models (phi-*)
    │   ├── Mistral Models (mistral-*, mixtral-*)
    │   └── Google Models (gemma*, paligemma)
    │
    └── Capability Fields:
        ├── context_length: Input token limit
        ├── max_output_tokens: Output token limit
        ├── tool_support: native/prompted/none
        ├── structured_output: native/prompted/none
        ├── parallel_tools: Boolean
        ├── max_tools: Integer (-1 for unlimited)
        ├── vision_support: Boolean
        ├── image_resolutions: Array of supported sizes
        ├── audio_support: Boolean
        └── source: Data source reference
```

## Data Structures

### architecture_formats.json
```json
{
  "architectures": {
    "architecture_name": {
      "patterns": ["model_pattern1", "model_pattern2"],
      "message_format": "format_type",
      "system_prefix": "<tag>",
      "user_prefix": "<tag>",
      "assistant_prefix": "<tag>",
      "tool_format": "json|xml|pythonic"
    }
  }
}
```

### model_capabilities.json
```json
{
  "models": {
    "model_name": {
      "context_length": 128000,
      "max_output_tokens": 4096,
      "tool_support": "native|prompted|none",
      "structured_output": "native|prompted|none",
      "parallel_tools": true,
      "max_tools": -1,
      "vision_support": false,
      "audio_support": false,
      "source": "Official docs"
    }
  },
  "default_capabilities": { ... }
}
```

## Integration Points
- **Primary Consumer**: `architectures/detection.py`
- **Architecture Detection**: Maps model names to communication formats
- **Capability Lookup**: Provides detailed model specifications
- **Provider Usage**: All providers use this for capability queries
- **Session Management**: Informs tool availability and context limits

## Model Coverage Analysis

### By Architecture
- **llama**: Llama 3.x, 4, DeepSeek, Yi models
- **qwen**: Qwen 2.5, 3, and vision variants
- **mistral**: Mistral 7B through Large, Mixtral MoE
- **phi**: Phi-2 through Phi-4, including vision
- **claude**: All Claude 3 variants
- **gpt**: GPT-3.5, 4, o1, o3 families
- **gemma**: Gemma, CodeGemma, PaliGemma

### By Capabilities
- **Native Tools**: GPT-4*, Claude-3*, Llama-3.1+, Qwen3, Mistral Large
- **Vision Support**: GPT-4o, Claude-3*, Llama-3.2-vision, Qwen-VL, PaliGemma
- **Audio Support**: GPT-4o, Llama-4 (multimodal)
- **Extended Context**: Claude (200K), GPT-4 (128K), Llama (128K)
- **Large Output**: GPT-4o-long (64K), Claude-3.7 (128K), Mistral Large (128K)

## Recommendations
1. **Add JSON Schema**: Create schema validation for both JSON files
2. **Version the data**: Add version field for tracking changes
3. **Unify patterns**: Consider merging pattern definitions
4. **Automate validation**: Script to verify capabilities against providers
5. **Add embeddings models**: Expand coverage for RAG use cases
6. **Document sources**: Link to official documentation for each model

## Technical Debt
- Some pattern duplication between the two JSON files
- Manual maintenance without validation tooling
- No capability inheritance for model families
- Missing dedicated embeddings model section
- Could benefit from automated updates from model cards

## Maintenance Guidelines

### Adding a New Architecture
1. Add entry to `architecture_formats.json` with:
   - Unique patterns that identify the architecture
   - Message format specification (prefixes/suffixes)
   - Tool format type (json/xml/pythonic)
2. Test with example models using that architecture

### Adding a New Model
1. Add entry to `model_capabilities.json` with:
   - Exact model name as key
   - All capability fields (use defaults if unknown)
   - Source citation for the information
2. Verify capabilities with actual testing when possible
3. Consider if architecture detection needs updating

## Future Enhancements
1. **Schema validation**: JSON Schema for both files
2. **Capability evolution**: Track capability versions over time
3. **Embeddings section**: Dedicated configuration for embedding models
4. **Auto-discovery**: Parse HuggingFace model cards automatically
5. **Provider overrides**: Allow providers to override capabilities
6. **Performance data**: Add latency/throughput benchmarks
7. **Cost information**: Include pricing data for commercial models