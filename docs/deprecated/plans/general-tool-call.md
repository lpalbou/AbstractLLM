# General Tool Call Implementation Plan for AbstractLLM

## 1. Overview

This document outlines the high-level strategy and plan for implementing unified tool call (function calling) support across the relevant providers (OpenAI, Anthropic, Ollama) within the AbstractLLM project. The goal is to provide a consistent interface and experience for developers using AbstractLLM, regardless of the underlying LLM provider.

## 2. Objectives

- Implement robust tool call support for OpenAI, Anthropic, and Ollama providers.
- Define a unified internal representation for tool definitions within AbstractLLM.
- Ensure a consistent developer experience through the AbstractLLM interface (`generate`, `generate_async`) and `Session` class.
- Maintain provider-specific optimizations and capabilities where appropriate.
- Leverage existing AbstractLLM components (e.g., `MediaFactory`, `ConfigManager`, `Error Handling`).
- Avoid adding direct dependencies on provider-specific client libraries where our abstraction provides equivalent or better control (e.g., Ollama).
- Provide seamless integration with streaming responses, including when tools are invoked.

## 3. Chosen Internal Tool Definition Model

AbstractLLM will adopt the **Anthropic tool definition structure** as its internal standard representation for tools. This decision is based on its relative simplicity, use of standard JSON Schema, and ease of convertibility to other provider formats (OpenAI, common Ollama usage).

**Reference:** See `docs/plans/general-tool-call-model.md` for a detailed description of this internal format.

## 4. Core Implementation Strategy

The implementation will follow these core principles:

1.  **Interface Extension:** Modify the `AbstractLLMInterface` (`generate` and `generate_async` methods) and the `Session` class (`send` method) to accept a `tools` parameter. This parameter will expect a list of tool definitions conforming to the internal standard format (Anthropic-like).
2.  **Internal Standard:** All internal logic within AbstractLLM (e.g., `Session` state management) will work with the chosen internal tool definition format.
3.  **Provider Adapters:** Each provider (`OpenAIProvider`, `AnthropicProvider`, `OllamaProvider`) will implement logic to:
    *   Receive tool definitions in the internal standard format.
    *   Convert these definitions into the specific format required by their respective APIs.
    *   Make the API call including the formatted tool definitions.
    *   Parse the API response to identify tool call requests from the model.
    *   Convert the provider-specific tool call response back into a standardized AbstractLLM format to be returned to the user/Session.
4.  **Standardized Return Value:** When a model requests a tool call, the `generate`/`generate_async` methods will return a standardized structure (e.g., a dictionary containing `content` and `tool_calls`), allowing the calling code (like the `Session` class) to handle the execution flow consistently.
5.  **Session Handling:** The `Session` class will be updated to manage the tool call lifecycle:
    *   Pass `tools` definitions to the provider.
    *   Receive and store tool call requests from the provider.
    *   Provide methods (e.g., `add_tool_result`) for the developer to supply the execution results.
    *   Send tool results back to the provider for the model to generate a final response.
6.  **Streaming Support:** Implement a robust approach to handling streaming when tool calls are involved:
    *   Detect tool calls during streaming.
    *   Provide callbacks for tool execution during streaming.
    *   Support resuming the stream after tool execution.

## 5. Implementation Phases (High-Level)

1.  **Define Internal Model & Helpers:**
    *   Finalize and document the internal tool model (`general-tool-call-model.md`).
    *   Implement helper utilities (e.g., `function_to_tool`) to convert Python functions into the internal format.
2.  **Update Core Interfaces:**
    *   Modify `AbstractLLMInterface`.
    *   Modify `ModelCapability` enum.
    *   Modify `Session` class interface for sending tools and receiving/adding results.
3.  **Implement Provider Adapters:**
    *   **OpenAI:** Implement conversion logic and API interaction for OpenAI function calling.
    *   **Anthropic:** Implement conversion logic and API interaction for Anthropic tool use.
    *   **Ollama:** Implement conversion logic and API interaction for Ollama tool use (following `ollama-tool-call.md` plan).
    *   Update `get_capabilities` for each provider.
4.  **Implement Session Logic:**
    *   Implement the internal logic within the `Session` class to manage the tool call conversation flow.
5.  **Testing & Documentation:**
    *   Write comprehensive unit and integration tests for all components.
    *   Update project documentation (README, usage guides) with examples.

## 6. Provider-Specific Plans

Detailed implementation steps for each provider will be documented separately:

-   `docs/plans/ollama-tool-call.md` (To be updated)
-   *(New plans to be created for OpenAI and Anthropic)*

## 7. Streaming with Tool Calls

Streaming presents unique challenges when combined with tool calls. We will address this with the following approach:

### 7.1 Streaming Implementation Strategy

1. **Streaming Detection Pattern:** During streaming, we need to detect when a provider is sending a tool call rather than regular content. Each provider has different patterns:
   - OpenAI: Tool calls appear in the "delta" field with a "tool_calls" key
   - Anthropic: Tool calls appear as content blocks with type "tool_use"
   - Ollama: Tool calls may appear in various formats depending on the model

2. **Buffered Streaming:** When streaming with tools enabled, implement a buffering system that can:
   - Collect streamed tokens until a complete tool call is detected
   - Parse and validate the complete tool call
   - Emit a special event or callback indicating a tool call is needed
   - Pause the stream while the tool is executed
   - Resume streaming after tool execution

3. **Callback Architecture:** Extend the existing callback system to support tool calls during streaming:
   ```python
   # Example callback signature
   def on_tool_call(tool_call: ToolCall) -> Optional[Any]:
       """Called when a tool call is detected during streaming.
       
       Args:
           tool_call: The detected tool call
           
       Returns:
           Optional result from executing the tool
       """
   ```

4. **Stream State Management:** Create a state machine for stream processing that can handle:
   - Regular content streaming
   - Tool call detection
   - Tool execution
   - Resumption after tool execution

### 7.2 Provider-Specific Streaming Considerations

Different providers handle streaming with tools differently:

1. **OpenAI:**
   - OpenAI streams tool calls piece by piece
   - Need to buffer and assemble complete tool call before processing

2. **Anthropic:**
   - Anthropic may deliver tool calls as discrete blocks
   - Need to detect when a complete tool_use block has been received

3. **Ollama:**
   - Ollama's behavior may vary by model
   - May need to implement model-specific detection heuristics

## 8. Security Considerations

When implementing tool calling, security is a critical concern:

1. **Input Validation:** Always validate inputs before executing tools
2. **Sandboxing:** Consider running tools in a restricted environment
3. **Rate Limiting:** Implement rate limiting for tool executions to prevent abuse
4. **Permission System:** Consider adding a permission system for tools

## 9. Conclusion

This plan provides a clear path towards integrating unified tool call support into AbstractLLM, enhancing its capabilities while maintaining architectural consistency and developer-friendliness. The addition of dedicated strategies for streaming and security considerations ensures a robust implementation. 