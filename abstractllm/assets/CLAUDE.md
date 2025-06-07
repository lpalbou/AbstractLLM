# Assets Component

## Overview
The assets folder contains static data files used by AbstractLLM, currently housing the `model_capabilities.json` file which serves as a curated database of model-specific capabilities across different providers.

## Code Quality Assessment
**Rating: 7/10**

### Strengths
- Clean JSON structure with clear categorization
- Comprehensive coverage of popular models
- Easy to update and maintain
- Version-control friendly format

### Issues
- No schema validation for the JSON structure
- No documentation on how to add new models
- Mixing of model families and specific versions (inconsistent granularity)
- No automated validation against actual model capabilities

## Component Mindmap
```
model_capabilities.json
├── Capability Categories
│   ├── tool_calling (38 models)
│   │   ├── Open source: llama, mistral, qwen, phi4, granite
│   │   └── Proprietary: claude-3*, gpt-4*
│   │
│   ├── structured_output (17 models)
│   │   └── Subset of tool-calling models
│   │
│   ├── reasoning (8 models)
│   │   ├── deepseek-r1
│   │   ├── qwq, cogito
│   │   └── claude-3-opus, claude-3-*-sonnet
│   │
│   ├── vision (14 models)
│   │   ├── Specialized: llama3.2-vision, qwen2-vl, paligemma
│   │   └── Multimodal: claude-3*, molmo, cogvlm
│   │
│   └── audio (2 models)
│       ├── whisper
│       └── gemini-audio
│
└── Usage Pattern
    └── Loaded by utils/model_capabilities.py
        └── Used by providers for capability detection
```

## Data Structure
```json
{
  "capability_name": [
    "model_pattern_1",
    "model_pattern_2",
    ...
  ]
}
```

## Integration Points
- **Primary Consumer**: `utils/model_capabilities.py`
- **Usage**: Supplements architecture-based capability detection
- **Providers**: All providers query this for model-specific overrides

## Model Coverage Analysis
- **Most Capable**: Claude-3 family (tool_calling, structured_output, reasoning, vision)
- **Tool Specialists**: llama3.x, mistral, qwen, granite families
- **Vision Specialists**: qwen2-vl, llava, moondream, paligemma
- **Reasoning Specialists**: deepseek-r1, qwq, cogito
- **Audio**: Limited to whisper and gemini-audio

## Recommendations
1. **Add JSON Schema**: Create schema validation for the structure
2. **Version the data**: Add version field for tracking changes
3. **Document patterns**: Explain model naming conventions
4. **Automate validation**: Script to verify capabilities against providers
5. **Add metadata**: Include last updated date, sources

## Technical Debt
- No clear distinction between model families and specific versions
- Manual maintenance without validation tooling
- No capability inheritance system
- Missing many newer models

## Maintenance Guidelines
When adding new models:
1. Use consistent naming (family vs specific version)
2. Verify capability with actual testing
3. Add to most specific category that applies
4. Document the source of capability information

## Future Enhancements
1. Move to a more structured format (YAML with schemas)
2. Add capability versions (e.g., tool_calling_v1, tool_calling_v2)
3. Include capability parameters (e.g., max tools, vision resolution)
4. Automated capability discovery from model cards
5. Provider-specific capability overrides