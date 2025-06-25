# Tool Tests for AbstractLLM

This directory contains tests for the AbstractLLM tool calling implementation.

## Test Files

- `test_foundation.py`: Tests the core tool functionality:
  - ToolDefinition, ToolCall, and ToolResult dataclasses
  - function_to_tool_definition utility with various function signatures
  - Standardization utilities for tool responses
  - Validation utilities

- `test_interfaces.py`: Tests AbstractLLM interface extensions:
  - AbstractLLMInterface tool methods
  - BaseProvider tool handling
  - GenerateResponse tool functionality
  - Error handling for tool-related operations

## Provider Tests

Provider-specific tests are in the `tests/providers/test_tool_calls.py` file, which includes:
- Tests for tool definition conversion to provider-specific formats
- Tests for extracting tool calls from provider responses
- Tests for handling provider-specific error conditions
- Both synchronous and asynchronous implementation tests

## Running Tests

Tests can be run with pytest:

```bash
# Run all tool tests
pytest tests/tools/

# Run all provider tool tests
pytest tests/providers/test_tool_calls.py

# Run tests for a specific provider
pytest tests/providers/test_tool_calls.py::TestOpenAIToolCalls
```

Provider-specific tests require API keys to be set as environment variables:
- `OPENAI_API_KEY` for OpenAI tests
- `ANTHROPIC_API_KEY` for Anthropic tests
- `OLLAMA_HOST` for Ollama tests (defaults to http://localhost:11434)

## Live API Tests

Some tests that make actual API calls are skipped by default. To run these tests:

```bash
# Run all tests including those marked as skip
pytest tests/providers/test_tool_calls.py::TestLiveApiCalls -v --runxfail
```

## Contributing

When adding new tool functionality, please add corresponding tests in the appropriate files. 