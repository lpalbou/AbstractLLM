# LM Studio Native Tools Bug

## Bug ID
`BUG-2025-09-17-001`

## Priority
**HIGH** - Prevents usage of models with "native" tool support in LM Studio

## Summary
LM Studio provider fails with 400 Bad Request when using models configured with "native" tool support, but works correctly with "prompted" tool support.

## Environment
- **Date Discovered**: 2025-09-17 17:24
- **Provider**: LM Studio (`lmstudio_provider.py`)
- **Session ID**: f29e1f63
- **AbstractLLM Version**: Current dev branch
- **LM Studio URL**: http://localhost:1234/v1

## Affected Models
- **Failing**: `qwen/qwen3-next-80b` (configured as "native" tool support initially)
- **Working**: `qwen3-coder:30b` (configured as "prompted" tool support)

## Bug Description

### What Happens
1. User creates LM Studio provider with model that has "native" tool support
2. Simple request like "who are you?" triggers tool preparation logic
3. Provider sends OpenAI `tools` parameter to LM Studio API
4. LM Studio returns 400 Bad Request error
5. Retry mechanism fails multiple times
6. Session falls back to simplified mode

### Root Cause Analysis

#### The Problem
```python
# In lmstudio_provider.py:510-511
if tool_mode == "native" and tool_defs:
    formatted_tools = self._format_tools_for_provider(tool_defs)

# Later in _prepare_openai_request:430-431
if tools:
    request_data["tools"] = tools
```

**Issue**: LM Studio's OpenAI-compatible API does not support the `tools` parameter, but AbstractLLM treats "native" tool support as requiring API-level tool definitions.

#### Why It Fails vs Works
- **"prompted" tool support**: Tools are converted to system prompt instructions ✅
- **"native" tool support**: Tools are sent as `tools` parameter in API request ❌

## Error Traces

### User Experience
```
user> who are you ?
ThinkingERROR - abstractllm.providers.lmstudio.LMStudioProvider - LM Studio API error: 400 Client Error: Bad Request for url: http://localhost:1234/v1/chat/completions
WARNING - abstractllm.retry_strategies - Attempt 1 failed with unknown. Retrying in 1.10s...
Thinking..ERROR - abstractllm.providers.lmstudio.LMStudioProvider - LM Studio API error: 400 Client Error: Bad Request for url: http://localhost:1234/v1/chat/completions
ERROR - abstractllm.retry_strategies - Not retrying after unknown error: [lmstudio] LM Studio API error: 400 Client Error: Bad Request for url: http://localhost:1234/v1/chat/completions
Note: Using simplified mode due to session compatibility
ERROR - abstractllm.providers.lmstudio.LMStudioProvider - LM Studio API error: 400 Client Error: Bad Request for url: http://localhost:1234/v1/chat/completions

❌ Error
──────────────────────────────────────────────────
  [lmstudio] LM Studio API error: 400 Client Error: Bad Request for url: http://localhost:1234/v1/chat/completions
```

### Failing Request (Native Tools)
```json
{
  "model": "qwen/qwen3-next-80b",
  "messages": [...],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 16384,
  "top_p": 0.95,
  "tools": [...]  // ← LM Studio rejects this parameter
}
```

### Working Request (Prompted Tools)
```json
{
  "model": "qwen/qwen3-next-80b",
  "messages": [
    {
      "role": "system",
      "content": "You are an intelligent AI assistant with memory and tool capabilities... Use the <|tool_call|> format EXACTLY as shown."
    },
    ...
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 16384,
  "top_p": 0.95
  // No "tools" parameter
}
```

## Impact Assessment

### Severity: HIGH
- **User Experience**: Complete failure for certain model configurations
- **Provider Coverage**: Affects LM Studio provider specifically
- **Model Scope**: Any model configured with "native" tool support fails
- **Workaround Complexity**: Requires manual model capability JSON editing

### Business Impact
- Prevents usage of advanced models like `qwen3-next-80b` with LM Studio
- Confusing error messages for users
- Undermines trust in "native" tool support claims
- Forces users to manually edit configuration files

## Technical Analysis

### Architecture Issue
The problem reveals a fundamental misunderstanding in the tool support classification:

1. **"native" ≠ API-level tools**: Just because a model supports tools natively doesn't mean the hosting provider's API supports OpenAI tool format
2. **Provider vs Model capabilities**: Model capabilities should be separate from provider API capabilities
3. **LM Studio limitations**: LM Studio provides OpenAI-compatible API but with limited feature parity

### Code Locations
- **Bug Location**: `abstractllm/providers/lmstudio_provider.py:510-511`
- **Request Building**: `abstractllm/providers/lmstudio_provider.py:430-431`
- **Model Config**: `abstractllm/assets/model_capabilities.json` (qwen3-next-80b-a3b entry)

## Reproduction Steps

### Reliable Reproduction
1. Set model to "native" tool support in `model_capabilities.json`:
   ```json
   "qwen3-next-80b-a3b": {
     "tool_support": "native",  // This causes the bug
     ...
   }
   ```
2. Create LM Studio provider: `create_llm("lmstudio", model="qwen/qwen3-next-80b")`
3. Send any request that triggers tool preparation
4. Observe 400 Bad Request error

### Verification Fix Works
1. Change model to "prompted" tool support:
   ```json
   "qwen3-next-80b-a3b": {
     "tool_support": "prompted",  // This works
     ...
   }
   ```
2. Same provider creation and request
3. Observe successful response with tool instructions in system prompt

## Root Cause Deep Dive

### The Design Flaw
AbstractLLM's architecture assumes that "native" tool support means the provider's API supports OpenAI-style tool calling. This is incorrect for LM Studio:

- **Model Reality**: qwen3-next-80b model can handle tools natively when prompted correctly
- **API Reality**: LM Studio's API doesn't support the OpenAI `tools` parameter
- **Framework Assumption**: "native" = send tools via API ❌

### Correct Understanding
```
Model Tool Capability ≠ Provider API Tool Support

qwen3-next-80b: CAN handle tools natively via prompting
LM Studio API: CANNOT handle OpenAI tools parameter
Result: Use "prompted" mode even for "native" models
```

## Solutions Analysis

### Option 1: Quick Fix (IMPLEMENTED)
Change `qwen3-next-80b-a3b` tool support from "native" to "prompted" in model_capabilities.json.

**Pros**: Immediate fix, works correctly
**Cons**: Doesn't solve architectural issue

### Option 2: Provider-Aware Tool Support (RECOMMENDED)
Implement provider-specific tool capability override:

```python
class LMStudioProvider:
    def get_effective_tool_support(self, model_name: str) -> str:
        """Override model's native tool support - LM Studio always uses prompted mode."""
        return "prompted"  # LM Studio doesn't support OpenAI tools parameter
```

**Pros**: Architecturally correct, future-proof
**Cons**: Requires provider interface changes

### Option 3: API Feature Detection
Detect LM Studio API capabilities and fall back appropriately.

**Pros**: Most robust
**Cons**: Complex implementation, requires API introspection

## Immediate Actions Taken

### 1. Emergency Fix Applied ✅
```json
// In model_capabilities.json
"qwen3-next-80b-a3b": {
  "tool_support": "prompted",  // Changed from "native"
  ...
}
```

### 2. Verification ✅
- Confirmed `qwen/qwen3-next-80b` now works with LM Studio
- Verified prompted tool support functions correctly
- Tool calls properly formatted as `<|tool_call|>JSON</|tool_call|>`

## Broader Implications

### Other Affected Providers?
This bug pattern could affect other OpenAI-compatible providers:
- **Ollama**: Uses `/api/chat` endpoint - may have similar limitations
- **HuggingFace Inference**: May not support full OpenAI tool format
- **Local providers**: Generally limited OpenAI feature parity

### Model Capability Accuracy
Need to audit ALL models marked as "native" tool support:
- Verify if hosting providers actually support OpenAI tools parameter
- Consider provider-specific capability overrides
- Document provider limitations clearly

## Testing Requirements

### Regression Testing
- [ ] Test all LM Studio-compatible models with tools
- [ ] Verify other providers aren't affected by similar issues
- [ ] Test model switching between providers with different tool support

### Integration Testing
- [ ] alma-simple.py with LM Studio + tools
- [ ] Session management with tool calls
- [ ] Memory integration with tool results

## Prevention Measures

### Code Reviews
- Require explicit testing of "native" tool support claims
- Verify provider API compatibility before marking tools as "native"
- Document provider-specific limitations

### Documentation Updates
- Clear distinction between model capabilities and provider API support
- Provider-specific tool support matrices
- Troubleshooting guide for tool-related errors

## Related Issues
- Provider capability detection improvements needed
- Model vs provider capability separation
- Better error messages for unsupported API features

## Resolution Status
- **Status**: PARTIALLY RESOLVED
- **Quick Fix**: Applied (prompted tool support)
- **Architectural Fix**: PENDING
- **Priority**: Consider architectural improvements for future releases

---

**Reporter**: User (2025-09-17)
**Investigator**: Claude Code
**Last Updated**: 2025-09-17 17:30
**Next Review**: When implementing provider capability overrides