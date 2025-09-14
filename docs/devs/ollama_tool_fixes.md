# Ollama Provider Tool Support Fixes

## Issues Fixed

### 1. Tool Call Parsing Fallback
**Problem**: When `tool_mode` was "native" but the model output tool calls in the text (prompted format), they were not being parsed.

**Fix**: Added fallback logic to ALWAYS try prompted parsing if no native tool calls are found:
```python
# If no tool calls found (or not in native mode), try prompted parsing
if not tool_response or not tool_response.has_tool_calls():
    prompted_response = handler.parse_response(content, mode="prompted")
    if prompted_response and prompted_response.has_tool_calls():
        tool_response = prompted_response
```

### 2. Streaming Support for Prompted Tools
**Problem**: Streaming responses only checked for native tool calls, not prompted ones.

**Fix**: Added prompted tool parsing at the end of streaming when `done=true`:
```python
# If no native tool calls, check for prompted tool calls in content
if not tool_response and current_content and tool_mode in ["native", "prompted"]:
    handler = self._get_tool_handler()
    if handler:
        prompted_response = handler.parse_response(current_content, mode="prompted")
```

### 3. Enhanced Logging
**Added logging to debug tool support issues**:
- Log tool context preparation results
- Log enhanced system prompt content when in prompted mode
- Log actual messages being sent to the API
- Log when fallback to prompted parsing occurs

### 4. Messages Parameter Support
The `messages` parameter was already being handled correctly for conversation history. The `_prepare_request_for_chat` method properly:
- Preserves provided messages
- Replaces/adds system message with enhanced prompt when tools are present
- Maintains conversation context across tool iterations

## How It Works Now

1. **Tool Preparation**: When tools are provided, `_prepare_tool_context` creates an enhanced system prompt with tool definitions
2. **Request Building**: The enhanced system prompt is passed to `_prepare_request_for_chat`
3. **Response Parsing**: 
   - First tries native tool parsing if in native mode
   - Always falls back to prompted parsing if no tool calls found
   - Works for both streaming and non-streaming responses
4. **Session Integration**: Messages parameter preserves conversation history for ReAct loops

## Testing

Run the test script to verify:
```bash
python test_ollama_tools.py
```

This should now properly:
- Send tool definitions in the system prompt
- Parse tool calls in `<|tool_call|>` format
- Execute tools and maintain conversation context
- Work with Session's ReAct loop