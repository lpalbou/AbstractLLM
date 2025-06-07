# Qwen Model Tool Prompt Formatting

Based on the codebase analysis, here's how tool prompts are formatted for Qwen models in the AbstractLLM library:

## Architecture Detection

Qwen models are detected by the following patterns in `abstractllm/architectures/detection.py`:
- Model names containing "qwen" or "qwq"
- Architecture family: `"qwen"`

## Message Format

Qwen uses the `im_start_end` message format with these prefixes/suffixes:
- System: `<|im_start|>system\n` ... `<|im_end|>\n`
- User: `<|im_start|>user\n` ... `<|im_end|>\n`
- Assistant: `<|im_start|>assistant\n` ... `<|im_end|>\n`

## Tool Format

According to `abstractllm/tools/parser.py`, Qwen uses the `SPECIAL_TOKEN` format for tools:

### Tool Prompt Structure

The `_format_qwen_style` function formats tools as:

```python
You are a helpful AI assistant with tool access.

Available tools:
[JSON list of tool definitions with name, description, and parameters]

To use a tool:
<|tool_call|>
{"name": "tool_name", "arguments": {"param": "value"}}
```

### Tool Call Detection

The parser looks for tool calls in Qwen responses using:
1. Opening tag: `<|tool_call|>`
2. Optional closing tag: `</|tool_call|>` (but models may forget this)
3. JSON content between tags with structure: `{"name": "...", "arguments": {...}}`

### Parsing Strategy

The `_parse_special_token` function handles Qwen tool calls with robust fallback:
1. First tries to find properly closed tags: `<|tool_call|>...</|tool_call|>`
2. Falls back to finding opening tag followed by valid JSON (no closing tag required)
3. Avoids duplicate parsing from overlapping patterns

## Example Tool Definition

When tools are provided to a Qwen model, they're formatted as:

```json
[
  {
    "name": "search_web",
    "description": "Search the web for information",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query"
        }
      },
      "required": ["query"]
    }
  }
]
```

## Example Qwen Tool Response

A Qwen model would respond with tool calls like:

```
I'll search for that information for you.

<|tool_call|>
{"name": "search_web", "arguments": {"query": "current weather in Paris"}}
</|tool_call|>
```

Or without closing tag (also valid):

```
I'll search for that information for you.

<|tool_call|>
{"name": "search_web", "arguments": {"query": "current weather in Paris"}}
```

## Integration Notes

1. Qwen models use "prompted" tool support (not native API-based)
2. Tool prompts are injected into the system message
3. The `UniversalToolHandler` automatically detects Qwen architecture and applies the correct formatting
4. Tool results are formatted back as natural language for the model to process

## Configuration in Assets

From `abstractllm/assets/architecture_formats.json`:
- Tool format: `"json"`
- Uses special token `<|tool_call|>` for tool invocation
- Follows ChatML-style message formatting