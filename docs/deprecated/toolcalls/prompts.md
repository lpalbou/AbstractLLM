Based on your request, I'll create specific prompts for each task document that will help a less-capable LLM execute the tasks successfully. Each prompt will reference the necessary documentation and code files.

## Prompt for Task 1: Remove Direct Tool Execution

```
Your task is to remove direct tool execution from the BasicAgent implementation in the AbstractLLM framework.

1. First, read docs/toolcalls/tasks/01_remove_direct_execution.md to understand the detailed task requirements.

2. Review docs/toolcalls/architecture.md to understand the fundamental "LLM-First Architecture" principle - tool calls must ONLY be initiated by the LLM, never by pattern matching in the agent code.

3. Review docs/toolcalls/best_practices.md for implementation guidelines, especially the "DON'T" section that highlights pattern matching anti-patterns.

4. Analyze basic_agent.py and look for these problematic patterns:
   - If statements that check for keywords like "file", "read", or other tool names in user queries
   - Direct calls to tool functions like read_file() based on user input
   - Response construction that bypasses LLM processing after tool execution

5. Remove all direct tool execution code from basic_agent.py, specifically:
   - Remove the extract_tool_call_from_text method
   - Modify the run method to exclusively use session.generate_with_tools() for ALL queries
   - Ensure all responses come from the LLM, not from direct tool execution

6. Consult docs/toolcalls/troubleshooting.md for examples of problematic code patterns and their fixes.

7. Verify your changes against the code_review_checklist.md to ensure you haven't missed any patterns.

The correct tool call flow must follow: User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User
```

## Prompt for Task 2: Implement LLM-First Flow

```
Your task is to implement the proper LLM-first tool call flow in the BasicAgent implementation in AbstractLLM.

1. First, read docs/toolcalls/tasks/02_implement_llm_first_flow.md to understand the detailed task requirements.

2. Review docs/toolcalls/architecture.md to understand the correct tool call flow:
   User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User

3. Study docs/toolcalls/best_practices.md for implementation guidelines and examples of the correct pattern.

4. Examine docs/toolcalls/index.md for a reference implementation approach.

5. Modify the BasicAgent.run method in basic_agent.py to:
   - Create or get a session from SessionManager
   - Define available tool functions with proper names matching ToolDefinitions
   - Call session.generate_with_tools() to get an initial response
   - Check if the response contains tool calls using response.has_tool_calls()
   - If tool calls exist, execute them using execute_tool_call
   - Send results back to the LLM for processing
   - Return the final response from the LLM

6. Also update the run_streaming method to handle tool calls during streaming.

7. Ensure you maintain multi-turn conversation context by properly adding messages to the session.

8. Add error handling as specified in the task document.

9. Verify your implementation by checking it against the examples in docs/toolcalls/troubleshooting.md.

The key principle is that ONLY the LLM should decide when to use tools, based on its reasoning about the user's query.
```

## Prompt for Task 3: Add Security Validation

```
Your task is to add security validation to tool execution in the AbstractLLM framework.

1. First, read docs/toolcalls/tasks/03_add_security_validation.md to understand the detailed task requirements.

2. Review docs/toolcalls/security.md to understand the security risks in tool execution and best practices.

3. Implement the following security features in basic_agent.py:

   a. Create a safe path validation function:
   ```python
   def is_safe_path(file_path: str, allowed_dirs: List[str]) -> bool:
       """Validate if a file path is safe to access."""
       # Implementation details in task document
   ```

   b. Update the read_file tool to use path validation:
   ```python
   def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
       """Read a file safely with validation."""
       # Add path validation before reading
   ```

   c. Create a secure tool wrapper function:
   ```python
   def create_secure_tool_wrapper(func: Callable, max_execution_time: int = 10) -> Callable:
       """Create a wrapper that adds timeout and validation to a tool function."""
       # Implementation details in task document
   ```

   d. Add input validation for tool parameters.

4. Ensure all tools are wrapped with these security measures in the BasicAgent.run method.

5. Add timeout protection to prevent long-running tools from consuming excessive resources.

6. Add proper error handling for security violations.

7. Test your implementation with path traversal attempts and timeout scenarios.

8. Verify your changes against the docs/toolcalls/code_review_checklist.md to ensure all security checks are implemented.

Security is critical because tool execution provides access to system resources. Every tool call must be properly validated before execution.
```

## Prompt for Task 4: Improve Logging

```
Your task is to improve logging for the tool call flow in AbstractLLM.

1. First, read docs/toolcalls/tasks/04_improve_logging.md to understand the detailed task requirements.

2. Review docs/toolcalls/architecture.md to understand the complete tool call flow that needs to be logged.

3. Understand the importance of logging from docs/toolcalls/troubleshooting.md, which uses logs to diagnose issues.

4. Implement the following logging improvements in basic_agent.py:

   a. Add the standardized log_step function:
   ```python
   def log_step(step_number: int, step_name: str, message: str, level: int = logging.INFO) -> None:
       """Log a step in the tool call flow."""
       # Implementation details in task document
   ```

   b. Update the run method with detailed logging for each step of the tool call process.

   c. Create tool-specific logging wrappers:
   ```python
   def logged_read_file(file_path: str, max_lines: Optional[int] = None) -> str:
       """Read file with logging."""
       # Implementation with timing and result logging
   ```

   d. Implement JSON logging for machine parsing:
   ```python
   def log_json_event(event_type: str, data: Dict[str, Any]) -> None:
       """Log a structured JSON event."""
       # Implementation details in task document
   ```

   e. Add streaming-specific logging in the run_streaming method.

   f. Implement logging configuration options.

5. Ensure logs capture the full flow from user query to tool execution to final response.

6. Verify your implementation includes error logging and timing information.

7. Test the logging with different query types and verify all steps are properly logged.

Proper logging is essential for debugging tool call issues, security auditing, and understanding user-agent interactions. Your implementation should provide clear visibility into each step of the process.
```

## Prompt for Task 5: Add Tests

```
Your task is to add comprehensive tests for tool call flow in AbstractLLM.

1. First, read docs/toolcalls/tasks/05_add_tests.md to understand the detailed test requirements.

2. Review docs/toolcalls/architecture.md to understand what the correct behavior is that you'll be testing.

3. Study docs/toolcalls/best_practices.md to understand the implementation patterns that should be tested.

4. Create the following test files:

   a. tests/test_direct_execution.py:
   - Test for direct tool execution detection
   - Test for proper LLM-first flow
   - Test for absence of conditional paths based on query content

   b. tests/integration/test_llm_first_flow.py:
   - Test complete LLM-first flow with mock sessions
   - Test streaming tool calls
   - Test with real OpenAI API (when available)

   c. tests/test_tool_error_handling.py:
   - Test tool not found errors
   - Test tool execution errors

   d. tests/test_tool_security.py:
   - Test path validation
   - Test file security boundaries
   - Test timeout functionality

   e. tests/test_streaming_tools.py:
   - Test streaming tool execution
   - Test streaming error handling

   f. tests/system/test_tool_system.py:
   - End-to-end system tests

5. Use pytest fixtures and mocks as detailed in the task document.

6. Skip tests that require API keys when not available.

7. Organize tests from unit level to system level.

8. Include tests for error conditions and security boundaries.

The tests should verify that direct tool execution doesn't happen, the LLM-first flow works correctly, error handling is robust, and security measures are effective. Pay special attention to tests that would catch if someone reintroduces direct tool execution patterns.
```

Each of these prompts provides a structured approach for the LLM to execute the specific task, referencing the appropriate documentation files and providing code snippets where helpful. The prompts break down complex tasks into manageable steps, which should help a less-capable LLM successfully implement the required changes.
