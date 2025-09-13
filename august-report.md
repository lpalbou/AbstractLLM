### AbstractLLM – Comprehensive Code Review and Fix Plan (August)

This report is self-contained. It explains what the project is, what is broken, why it breaks, and exactly how to fix it with minimal, maintainable edits. It also includes simulation recipes and validation steps that do not rely on the project’s test suite or external APIs.

The guidance below is based purely on the current code state (not on test suite expectations nor external docs).

---

### Overview: What this project is

- A unified interface for interacting with multiple LLM providers (OpenAI, Anthropic, Ollama, HuggingFace, MLX) via `create_llm` / `create_session`.
- Provides stateful sessions with:
  - Conversation history, system prompt, metadata
  - Save/load of sessions, stats
  - Unified generation entry point: streaming, tools, files
- Tool use (function calling) that is architecture-aware:
  - Formats tool prompts for models without native tool APIs
  - Parses tool calls from provider outputs for various model families
  - Executes Python tool implementations and feeds results back to the model
- `@file` attachment syntax: injects file content into prompt in a standard way
- Structured logging for requests/responses
- Minimal CLI agent (`alma-minimal.py`)
- Media scaffolding; MLX has a vision path (MLX-VLM) with image preprocessing

---

### What works (verified by local simulation, without tests)

- **Session tool loop**: `Session.generate(..., tools=[callables])` correctly
  - Triggers a tool call (from provider output)
  - Executes the mapped Python tool function
  - Appends an assistant message with `tool_results`
  - Generates a final assistant response
- **Provider-specific formatting of tool results** in `Session.get_messages_for_provider(...)`:
  - OpenAI: uses `function` role (function name + output)
  - Anthropic: assistant message that embeds the tool output text
  - MLX: uses `tool` role
  - Ollama / HuggingFace: assistant message prefixed with `TOOL OUTPUT [name]:`
- **Architecture-aware tool parsing**:
  - Qwen: `<|tool_call|>...` special token JSON format detected and parsed
  - Llama: `<function_call>...</function_call>` JSON format detected and parsed
  - Gemma family: ```tool_code``` and `tool_call:` code-like patterns are parsed
- **Tool definition from functions**: `ToolDefinition.from_function` produces clean JSON schema

Reproducible validation steps are provided at the end of this report.

---

### High-priority bugs and minimal fixes

Each fix includes: file path, what is wrong, why, and the smallest safe edit (with before/after snippets). Apply them in order.

#### 1) Session.add_tool_result uses wrong ToolResult constructor args

- Symptom: `TypeError` when `Session.add_tool_result` constructs a `ToolResult` (only occurs when the tools package is installed).
- Cause: The `ToolResult` dataclass (in `abstractllm.tools.core`) has fields `tool_call_id`, `output`, `error` but the code passes `call_id` and `result`.
- File: `abstractllm/session.py`
- Minimal edit: Rename constructor arguments to match dataclass field names.

Before:
```python
# abstractllm/session.py (inside add_tool_result)
tool_result_obj = ToolResult(
    call_id=tool_call_id,
    result=result,
    error=error
)
```

After:
```python
# abstractllm/session.py (inside add_tool_result)
tool_result_obj = ToolResult(
    tool_call_id=tool_call_id,
    output=result,
    error=error
)
```

Why safe: This is a naming fix only. All downstream behavior remains unchanged. It unlocks correct operation when the tools package is present.

Validation: Use the “Session tool loop” simulation at the end; the assistant message should have one tool result attached and no exceptions.

---

#### 2) MLX provider: PIL type annotation can cause import-time errors when vision deps are absent

- Symptom: Import-time `NameError` in `mlx_provider` if `PIL` or `mlx_vlm` isn’t installed. The return annotation `Image.Image` may be evaluated in some environments at import time.
- File: `abstractllm/providers/mlx_provider.py`
- Minimal edit: Relax the return type to a non-evaluated type (`Any`). The function already imports PIL internally and returns a PIL Image when available.

Before:
```python
def _process_image(self, image_input: Union[str, Path, Any]) -> Image.Image:
```

After:
```python
from typing import Any

def _process_image(self, image_input: Union[str, Path, Any]) -> Any:
```

Why safe: No behavior change; eliminates hard dependency on PIL at import time.

---

#### 3) MLX provider: force “prompted tools” mode instead of “native”

- Symptom: Some capability metadata or model names may imply “native” tool support, but MLX (local inference) can’t actually send native tool schemas. The robust approach is prompting.
- File: `abstractllm/providers/mlx_provider.py`, method `_get_tool_handler`
- Minimal edit: After creating `UniversalToolHandler(model)`, force prompted-only behavior.

Patch in `_get_tool_handler` right after `self._tool_handler = UniversalToolHandler(model)`:
```python
# Force prompted mode for MLX local inference
self._tool_handler.supports_native = False
self._tool_handler.supports_prompted = True
```

Why safe: Increases tool-call reliability with local MLX models; matches actual capability.

---

#### 4) MLX provider: noisy console printing of the entire prompt during streaming

- Symptom: `_generate_text_stream` prints a huge prompt block to console (the “EXACT PROMPT SENT TO MLX MODEL (STREAMING)” section), leaking the prompt and cluttering logs.
- File: `abstractllm/providers/mlx_provider.py`, in `_generate_text_stream`
- Minimal edit: Remove those `print` statements or guard them with an environment flag:

Example guard:
```python
import os
if os.getenv("ABSTRACTLLM_DEBUG_PROMPTS") == "1":
    print("\n" + "="*80)
    print("🔥 EXACT PROMPT SENT TO MLX MODEL (STREAMING):")
    print("="*80)
    print(formatted_prompt)
    print("="*80)
    print(f"📊 Length: {len(formatted_prompt)} characters")
    print("="*80 + "\n")
```

Why safe: Avoids leaking prompts by default and reduces console noise.

---

#### 5) Factory: brittle platform check for MLX provider

- Symptom: Current logic uses `platform.processor()` equality to gate MLX, which is unreliable on macOS AS variants. This may block MLX incorrectly.
- File: `abstractllm/factory.py`, within `_check_platform_requirements`
- Minimal edit: Use the project’s robust helper instead of `platform.processor()` equality.

Replace MLX check with:
```python
from abstractllm.utils.utilities import is_apple_silicon

if provider == "mlx" and not is_apple_silicon():
    return False
```

Why safe: Consistent detection logic already available in the repository; reduces false negatives.

---

#### 6) CLI model switching uses wrong provider name

- Symptom: In `alma-minimal.py`, when the user runs `/model <name>`, code calls `session.switch_provider(session.provider.__class__.__name__.lower(), new_model)`, which yields incorrect registry names (e.g., `mlxprovider` instead of `mlx`).
- File: `alma-minimal.py`
- Minimal edits:
  1) Use the session’s provider info for the correct registry name:
     ```python
provider_name = session.get_provider_info().get("provider_name")
result = session.switch_provider(provider_name, new_model)
     ```
  2) Update the CLI help strings to reflect actual default arguments (current defaults are `--provider ollama`, `--model qwen3:4b`).

Why safe: Correctly resolves provider names; improves UX.

---

#### 7) OpenAI provider: async method uses undefined variables; wrong imports in streaming helpers

- Symptoms:
  - In `generate_async`, references to `enhanced_system_prompt`, `tool_mode`, `formatted_tools` before defining them.
  - In streaming finalization blocks (both sync and async), imports `ToolCallRequest` and `ToolCall` from `.types` (wrong module). The correct module is `abstractllm.tools.core`.
- File: `abstractllm/providers/openai.py`
- Minimal edits:
  1) In `generate_async(...)`, mirror the sync setup before logging and API call:
     ```python
# Prepare tools exactly like sync path
enhanced_system_prompt = system_prompt or self.config_manager.get_param(ModelParameter.SYSTEM_PROMPT)
formatted_tools = None
tool_mode = "none"

if tools:
    enhanced_system_prompt, tool_defs, tool_mode = self._prepare_tool_context(tools, enhanced_system_prompt)
    if tool_mode == "native" and tool_defs:
        formatted_tools = tool_defs  # Already formatted by _prepare_tool_context for OpenAI
     ```
     Use `enhanced_system_prompt`, `tool_mode`, and `formatted_tools` in the logging call and API params consistently.

  2) Replace wrong imports in streaming consolidation blocks:
     - From:
       ```python
from .types import ToolCallRequest, ToolCall
       ```
     - To:
       ```python
from abstractllm.tools.core import ToolCallRequest, ToolCall
       ```

  3) Keep sync/async behavior parity: when no tools found for non-streaming calls, return a string; otherwise return `GenerateResponse` with `tool_calls`.

Why safe: Fixes NameErrors/ImportErrors; aligns async with known-good sync behavior; minimal surface change.

---

### Optional low-risk improvements (deferred)

- Root `abstractllm/__init__.py`: prune `__all__` entries for APIs that don’t exist (e.g., chains) to avoid confusion. Optionally export media convenience imports:
  ```python
from abstractllm.media import ImageInput, MediaFactory, MediaProcessor
  ```
- Document in `Session.generate()` that passing function callables in `tools=[...]` is preferred so the session has implementations for execution (passing only tool definitions creates placeholders by design).

---

### Validation (no unit tests, no external APIs)

Run the following scripts to verify functionality after applying the changes. These do not require API keys or network calls.

1) Tool parsing and prompt formatting sanity check

```python
from abstractllm.tools.core import ToolDefinition
from abstractllm.tools.parser import format_tool_prompt, detect_tool_calls, parse_tool_calls

def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

tool_def = ToolDefinition.from_function(add)
print('ToolDefinition:', tool_def.to_dict())

sample_qwen = '<|tool_call|>{"name": "add", "arguments": {"a": 2, "b": 3}}'
print('DETECT qwen:', detect_tool_calls(sample_qwen, 'qwen3:4b'))
print('CALLS qwen:', [(c.name, c.arguments) for c in parse_tool_calls(sample_qwen, 'qwen3:4b')])

sample_llama = '<function_call>{"name": "add", "arguments": {"a": 4, "b": 5}}</function_call>'
print('DETECT llama:', detect_tool_calls(sample_llama, 'llama-3.1-8b'))
print('CALLS llama:', [(c.name, c.arguments) for c in parse_tool_calls(sample_llama, 'llama-3.1-8b')])
```

Expected output highlights:
- ToolDefinition prints with `parameters` object and `required` fields `['a', 'b']`.
- Qwen: `DETECT qwen: True`, calls include `('add', {'a': 2, 'b': 3})`.
- Llama: `DETECT llama: True`, calls include `('add', {'a': 4, 'b': 5})`.

2) Session tool loop with a fake provider (end-to-end, no APIs)

```python
from typing import Optional
from abstractllm.providers.base import BaseProvider
from abstractllm.interface import ModelParameter, ModelCapability
from abstractllm.types import GenerateResponse
from abstractllm.tools.core import ToolCall, ToolCallResponse
from abstractllm.session import Session

# A simple tool

def add(a: int, b: int) -> int:
    return a + b

# Minimal fake provider: first call emits a tool call, second call returns final answer
class FakeProvider(BaseProvider):
    def __init__(self, config=None):
        super().__init__(config or {})
        if not self.config_manager.get_param(ModelParameter.MODEL):
            self.config_manager.update_config({ModelParameter.MODEL: 'fake-model-1'})
        self._did_tool_phase = False

    def _generate_impl(self, prompt: str, system_prompt: Optional[str] = None, files=None, stream: bool=False, tools=None, messages=None, **kwargs):
        if tools and not self._did_tool_phase:
            self._did_tool_phase = True
            tcall = ToolCall(name='add', arguments={'a': 2, 'b': 3})
            return GenerateResponse(content='Calling tool add', tool_calls=ToolCallResponse(content='Calling tool add', tool_calls=[tcall]), model='fake-model-1')
        return GenerateResponse(content='The sum is 5.', model='fake-model-1')

    async def generate_async(self, *args, **kwargs):
        raise NotImplementedError

    def get_capabilities(self):
        return {
            ModelCapability.STREAMING: False,
            ModelCapability.MAX_TOKENS: 2048,
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: False,
            ModelCapability.FUNCTION_CALLING: True,
            ModelCapability.TOOL_USE: True,
            ModelCapability.MULTI_TURN: True,
        }

sess = Session(system_prompt='You are helpful.', provider=FakeProvider())
resp = sess.generate(prompt='please add', tools=[add], max_tool_calls=3)
print('Final content:', resp.content)
print('History roles:', [m.role for m in sess.get_history()])
assistant_msgs = [m for m in sess.get_history() if m.role == 'assistant']
print('Assistant tool_results sizes:', [len(m.tool_results or []) for m in assistant_msgs])
```

Expected output highlights:
- `Final content: The sum is 5.`
- `History roles:` includes `system`, `user`, `assistant` (tool phase), `assistant` (final)
- `Assistant tool_results sizes:` shows `[1, 0]` (first assistant carries one tool result, final carries none)

3) Provider-specific formatting of tool outputs

```python
# Continue from the previous session
for prov in ['openai','anthropic','mlx','ollama','huggingface']:
    fm = sess.get_messages_for_provider(prov)
    print(prov, 'last messages:', fm[-4:])
```

Expected output highlights:
- For `openai`, the last messages include a dict `{ 'role': 'function', 'name': 'add', 'content': 3 }`.
- For `anthropic`, an assistant message like `"Tool 'add' returned the following output:\n\n3"` appears before the final.
- For `mlx`, the `tool` role with `{ 'name': 'add', 'content': 3 }`.
- For `ollama` / `huggingface`, an assistant message `"TOOL OUTPUT [add]: 3"` appears before the final.

---

### Notes and priorities

- Apply the 7 high-priority fixes first; they are small, local, and materially improve correctness and developer experience.
- The Validation section suffices to confirm correctness without relying on the current test suite or external APIs.
- After these, consider pruning/export cleanup in the root package and adding small docstrings or comments for nuanced behavior (MLX tool prompting).

---

### Summary of edits by file

- `abstractllm/session.py`
  - Fix `ToolResult` constructor arg names in `add_tool_result`
- `abstractllm/providers/mlx_provider.py`
  - Change `_process_image` return annotation to `Any`
  - Force prompted tools in `_get_tool_handler` (set `supports_native=False`, `supports_prompted=True`)
  - Remove/guard prompt printing in `_generate_text_stream`
- `abstractllm/factory.py`
  - Replace brittle MLX platform check with `is_apple_silicon()`
- `alma-minimal.py`
  - Correct provider name for `session.switch_provider`
  - Update CLI help defaults to match actual defaults (`--provider ollama`, `--model qwen3:4b`)
- `abstractllm/providers/openai.py`
  - In `generate_async`, define `enhanced_system_prompt`, `tool_mode`, `formatted_tools` as in sync path; use them in logging and API params
  - Replace wrong `from .types` imports in streaming consolidation with `from abstractllm.tools.core import ToolCallRequest, ToolCall`

This plan yields immediate stability improvements and preserves the modular, clean architecture.
