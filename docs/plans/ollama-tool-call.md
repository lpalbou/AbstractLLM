# Ollama Tool Call Implementation Plan (Updated)

## Overview

This document outlines the specific plan for implementing tool call support in AbstractLLM's **Ollama provider**. This implementation aligns with the **general tool call strategy** defined in `docs/plans/general-tool-call.md` and utilizes the **internal tool definition model** specified in `docs/plans/general-tool-call-model.md` (based on the Anthropic format).

**Key Principle:** AbstractLLM will interact directly with the Ollama API endpoint for tool calls. It will **not** add a dependency on the `ollama-python` client library. However, the structure and patterns used by `ollama-python` (found in `/Users/alboul/projects/abstractllm/investigate/ollama-python`) will serve as a **reference** for formatting API requests correctly.

## Objectives

1.  Enable tool call/function calling support for compatible models run via Ollama.
2.  Adhere to the AbstractLLM internal standard for tool definitions (Anthropic-like format).
3.  Correctly format requests to the Ollama API's `/api/chat` endpoint (when tools are used) based on observed patterns.
4.  Parse tool call requests from Ollama API responses.
5.  Integrate seamlessly with AbstractLLM's `Session` management and error handling.

## Implementation Phases (Refined for Ollama)

### Phase 1: Interface Alignment (Covered by General Plan)

-   Ensure `AbstractLLMInterface` and `Session` accept the `tools` parameter (expecting the internal format).
-   Ensure `ModelCapability.FUNCTION_CALLING` is defined.

### Phase 2: Ollama Provider Implementation (2-3 days)

1.  **Update `OllamaProvider.get_capabilities`**
    -   Modify the logic to dynamically report `ModelCapability.FUNCTION_CALLING: True` for models known or configured to support tool calls via Ollama (e.g., specific versions of Llama 3, Mistral, Gemma, etc.). This might require maintaining a list or using configuration flags.

```python
# Example snippet (conceptual)
def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
    # ... existing capability logic ...
    model = self.config_manager.get_param(ModelParameter.MODEL)
    
    # Models known/configured to support function calling via Ollama
    # TODO: Refine this list or make it configurable
    TOOL_CALLING_MODELS = ["llama3", "llama3.1", "mistral", "gemma2"] 
    
    capabilities[ModelCapability.FUNCTION_CALLING] = any(
        model.lower().startswith(m.lower()) for m in TOOL_CALLING_MODELS
    )
    
    # ... existing vision capability logic ...
    return capabilities
```

2.  **Implement `_process_tools` Method**
    -   Create a private method `_process_tools` within `OllamaProvider`.
    -   This method takes the list of tools (in the internal Anthropic-like format) received by `generate`/`generate_async`.
    -   It **converts** each tool definition into the format expected by the Ollama API (which typically mirrors the OpenAI format).

```python
# Example snippet (conceptual)
import copy # Ensure copy is imported if not already

def _process_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts internal tool format (Anthropic-like) to Ollama API format (OpenAI-like)."""
    processed_tools = []
    for internal_tool in tools:
        # Validate internal_tool format (optional but recommended)
        if not all(k in internal_tool for k in ['name', 'description', 'input_schema']):
            logger.warning(f"Skipping invalid tool definition: {internal_tool}")
            continue            
        # Convert to OpenAI/Ollama format
        ollama_tool = {
            "type": "function",
            "function": {
                "name": internal_tool['name'],
                "description": internal_tool['description'],
                # Directly use the input_schema as parameters, assuming it's valid JSON schema
                "parameters": internal_tool['input_schema'] 
            }
        }
        processed_tools.append(ollama_tool)
    return processed_tools
```

3.  **Update `_prepare_request_data` Method**
    -   Modify the existing `_prepare_request_data` (or create if needed for chat endpoint). 
    -   **Crucially:** When `tools` are provided:
        *   The request **must** use the `/api/chat` endpoint, not `/api/generate`.
        *   The payload **must** use the `messages` format instead of `prompt`.
        *   The processed (Ollama-formatted) tools from `_process_tools` must be included in the payload under the `tools` key.
    -   Ensure correct handling of `system_prompt` and `files` (especially images) within the `messages` structure when using the chat endpoint.

```python
# Example snippet (conceptual)
def _prepare_request_data(
    self, 
    model: str, 
    prompt: str, 
    system_prompt: Optional[str], 
    processed_files: List[Any], 
    processed_tools: Optional[List[Dict[str, Any]]],
    temperature: float, 
    max_tokens: int, 
    stream: bool
) -> Dict[str, Any]:
    """Prepare request data for Ollama API."""
    # Base request structure
    request_data = {
        "model": model,
        "stream": stream,
        "options": {
            "temperature": temperature
        }
    }
    
    if processed_tools:
        # USING /api/chat format
        request_data["messages"] = []
        
        # Add system prompt if exists
        if system_prompt:
            request_data["messages"].append({"role": "system", "content": system_prompt})
            
        # Add user prompt and files
        user_message_content = prompt
        images = []
        if processed_files: # Assuming processed_files contains ImageInput objects
            for media_input in processed_files:
                if isinstance(media_input, ImageInput):
                    # Convert to base64 or format expected by Ollama
                    images.append(media_input.to_provider_format("ollama")) 
                # else: Handle other file types if needed for chat endpoint
        
        user_message = {"role": "user", "content": user_message_content}
        if images:
            user_message["images"] = images
        request_data["messages"].append(user_message)
        
        # Add conversation history if applicable (requires Session integration)
        # for message in session_history:
        #     request_data["messages"].append(message) 
            
    else:
        # USING /api/generate format (existing logic)
        request_data = {
            "model": model,
            "prompt": prompt, 
            "stream": stream,
            "options": {"temperature": temperature, ... },
            # ... handle files/images for generate endpoint ...
            # ... handle system prompt for generate endpoint ...
        }
        
    return request_data
```

4.  **Update `_make_api_call` / `_make_async_api_call` Methods**
    -   Modify the API calling logic to conditionally use the `/api/chat` endpoint when `tools` are present in the `request_data`. Reference: `ollama-python` likely uses `/api/chat` when tools are passed.
    -   Ensure the response parsing logic can handle the structure returned by `/api/chat`, especially the `message` object containing `content` and potentially `tool_calls`.

5.  **Implement Response Processing for Tool Calls**
    -   Update `_process_response` / `_process_async_response`.
    -   If the response data (from `/api/chat`) contains `message.tool_calls`:
        *   Extract the `tool_calls` list.
        *   Extract the assistant's textual `content` (if any) from `message.content`.
        *   Return a **standardized AbstractLLM dictionary** containing both `content` and `tool_calls`. This standardized format will be consumed by the `Session` class.

```python
# Example snippet (conceptual)
def _process_response(self, data: Dict[str, Any], stream: bool = False) -> Union[str, Dict[str, Any]]:
    # ... (handle streaming separately if needed) ... 
    
    if "message" in data and isinstance(data["message"], dict):
        message_content = data["message"].get("content", "")
        tool_calls = data["message"].get("tool_calls")
        
        if tool_calls:
            # Return standardized tool call structure
            return {
                "content": message_content,
                "tool_calls": tool_calls 
            }
        else:
            # Standard chat response
            return message_content
    elif "response" in data: 
        # Response from /api/generate
        return data["response"]
    else:
        logger.warning(f"Unexpected Ollama response format: {data}")
        return str(data) # Fallback
```

### Phase 3: Async Implementation (1-2 days)

-   Implement the asynchronous versions (`_process_async_response`, update `generate_async`, `_make_async_api_call`) following the patterns established in the synchronous implementation, ensuring correct handling of the `/api/chat` endpoint and `tool_calls` in async contexts.

### Phase 4: Session Integration (Covered by General Plan)

-   The `Session` class will interact with `OllamaProvider` via the updated `generate` method, passing tools in the internal format and receiving the standardized tool call response structure.
-   The `Session` handles adding tool results using `add_tool_result` (which adds a `tool` role message) and sending subsequent requests, potentially reusing the `_prepare_request_data` logic to include the history.

### Phase 5 & 6: Testing, Documentation, Examples (Covered by General Plan)

-   Focus Ollama-specific tests on:
    *   Correct conversion of internal tool format to Ollama API format.
    *   Correct usage of `/api/chat` vs `/api/generate` endpoints.
    *   Parsing of `tool_calls` from Ollama responses.
    *   End-to-end tests using a local Ollama instance with a tool-capable model.

## Timeline Adjustment

-   The core Ollama provider work remains estimated at 2-3 days for the synchronous part and 1-2 days for async, assuming the general interface/session work is done concurrently or beforehand.

## Conclusion

This updated plan details how the `OllamaProvider` will implement tool call support by adhering to AbstractLLM's internal standards while correctly interacting with the Ollama API, using the `ollama-python` client patterns as a reference guide for API formatting rather than a direct dependency. 