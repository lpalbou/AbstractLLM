# AbstractLLM Unified Tool Call Implementation Plan

This document outlines the 5 key phases for implementing the unified tool call functionality across AbstractLLM, based on the strategy defined in `docs/plans/general-tool-call.md` and the internal model specified in `docs/plans/general-tool-call-model.md`.

---

## Phase 1: Establish Tool Definition Foundation

*   **Task Description:** Define the standardized internal format for tool calls returned by providers and implement the helper utility for creating tool definitions from Python functions.
*   **Prompt/Goal:** Create the foundational code structures and utilities for handling tool definitions consistently within AbstractLLM, ensuring the internal model is clearly defined and easily usable.

---

## Phase 2: Adapt Core Interfaces

*   **Task Description:** Modify the core `AbstractLLMInterface`, `ModelCapability` enum, and `Session` class to accept tool definitions and handle the standardized tool call response structure. This involves updating method signatures and defining how tool results are passed back.
*   **Prompt/Goal:** Update the central AbstractLLM interfaces and session management to be fully aware of the tool calling mechanism, integrating the `tools` parameter and the concept of tool results into the core workflow.

---

## Phase 3: Implement Provider Adapters

*   **Task Description:** For each priority provider (OpenAI, Anthropic, Ollama), implement the necessary logic within their respective `Provider` classes. This includes: (1) Translating internal tool definitions to the provider-specific format, (2) Updating API call logic (e.g., using correct endpoints like Ollama's `/api/chat`), (3) Parsing tool call requests from the provider's response, (4) Converting the response back to the standardized AbstractLLM tool call format, and (5) Updating the `get_capabilities` method.
*   **Prompt/Goal:** Enable each provider to handle tool calls by bridging the gap between the AbstractLLM internal standard and the provider's specific API requirements for tool definition, request formatting, and response parsing.

---

## Phase 4: Implement Session Tool Call Lifecycle

*   **Task Description:** Implement the logic within the `Session` class to manage the complete tool call conversation flow. This includes receiving the standardized tool call request structure from the provider, storing it, allowing the user to add tool results via `add_tool_result` (including `tool_call_id`), formatting the `tool` role message correctly, and sending the updated conversation history back to the provider.
*   **Prompt/Goal:** Empower the `Session` class to orchestrate the multi-turn interaction required for tool calls, maintaining conversation state and correctly sequencing user prompts, tool call requests, tool results, and final LLM responses.

---

## Phase 5: Comprehensive Testing & Documentation

*   **Task Description:** Develop thorough unit tests for helper utilities and provider conversion logic. Create integration tests for each provider using actual API calls (or mocks) involving tool calls. Test the `Session` class's handling of the tool call lifecycle. Update all relevant documentation (README, Usage Guides, Provider specifics) with clear explanations and examples of how to use the new tool calling feature.
*   **Prompt/Goal:** Ensure the implemented tool call functionality is reliable, robust, and easy for developers to understand and use by verifying its correctness through tests and providing comprehensive documentation and examples. 