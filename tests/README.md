# AbstractLLM Tests

This directory contains the test suite for AbstractLLM, with a particular focus on verifying the proper implementation of the LLM-first tool call flow.

## Test Structure

The test suite is organized in layers from unit tests to system tests:

### Unit Tests

- `test_direct_execution.py`: Verifies that direct tool execution (bypassing the LLM) is not happening
- `test_tool_error_handling.py`: Tests error handling during tool execution
- `test_tool_security.py`: Tests security features for tool execution
- `test_streaming_tools.py`: Tests streaming tool execution

### Integration Tests

- `integration/test_llm_first_flow.py`: Tests the complete LLM-first flow with mock sessions
- `integration/test_streaming_tools.py`: Tests streaming tool calls with mocks and real providers
- `integration/test_session_tools.py`: Tests the Session implementation of tool call handling
- `integration/test_edge_cases.py`: Tests edge cases in tool call handling

### System Tests

- `system/test_tool_system.py`: End-to-end system tests using real LLM providers

## Running the Tests

To run all tests:

```bash
pytest -xvs tests/
```

To run a specific test file:

```bash
pytest -xvs tests/test_direct_execution.py
```

To run tests that don't require API keys:

```bash
pytest -xvs tests/ -k "not test_real_openai_tool_call"
```

## Test Coverage

These tests verify that:

1. **No Direct Tool Execution**: Tool calls are always initiated by the LLM, not by pattern matching in the agent code
2. **Proper Flow**: The correct LLM-first flow is followed for all queries
3. **Error Handling**: Errors during tool execution are properly handled and communicated back to the user
4. **Security**: Security measures like path validation and execution timeouts are enforced
5. **Streaming**: Tool execution works properly in streaming mode
6. **System Integration**: The complete system works end-to-end with real LLM providers

## API Keys for Testing

Some tests require API keys for real LLM providers. To run these tests, set the following environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
```

Tests that require API keys are marked with `@pytest.mark.skipif` and will be skipped if the keys are not available. 