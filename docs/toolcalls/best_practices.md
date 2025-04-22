# Tool Call Best Practices in AbstractLLM

## Core Principles

1. **LLM as Decision Maker**: The LLM—not your code—decides when and which tools to call
2. **Complete Information Flow**: Always ensure tool results go back to the LLM
3. **Separation of Concerns**: Tools handle actions, LLM handles reasoning
4. **Security First**: Never execute tools directly from user input
5. **Context Preservation**: Maintain conversation history across tool calls

## Implementation Guidelines

### DO:

- ✅ **Use the proper tool call flow:**
  ```
  User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User
  ```

- ✅ **Implement tool calls using the official AbstractLLM patterns:**
  ```python
  # Correct implementation
  response = session.generate_with_tools(
      tool_functions=tool_functions,
      prompt=user_query
  )
  ```

- ✅ **Let the LLM determine when to use tools:**
  ```python
  # Provide tools to the LLM
  tools = [read_file, get_weather, calculate]
  # Let the LLM decide which one (if any) to use
  response = provider.generate(prompt=user_query, tools=tools)
  ```

- ✅ **Return tool results back to the LLM:**
  ```python
  # After tool execution
  tool_results = execute_tool_calls(response.tool_calls)
  # Send results back to LLM for processing
  final_response = provider.generate(
      prompt=user_query,
      messages=[*previous_messages, tool_results]
  )
  ```

- ✅ **Include comprehensive logging:**
  ```python
  logger.debug("LLM requested tool: %s with args: %s", tool_name, args)
  logger.debug("Tool execution result: %s", result)
  ```

### DON'T:

- ❌ **Never pattern match on user input to execute tools:**
  ```python
  # WRONG - Direct tool execution
  if "read" in query and "file" in query:
      filename = extract_filename(query)
      content = read_file(filename)
      return f"Here's the content: {content}"
  ```

- ❌ **Never bypass the LLM for tool calls:**
  ```python
  # WRONG - Bypassing LLM decision making
  def determine_tool(query):
      if "weather" in query:
          return weather_tool
      elif "file" in query:
          return file_tool
  ```

- ❌ **Never return tool results directly to the user:**
  ```python
  # WRONG - Skipping LLM processing of results
  result = execute_tool(tool_call)
  return f"Tool result: {result}"
  ```

- ❌ **Never hard-code tool selection logic:**
  ```python
  # WRONG - Hard-coded tool selection
  if query.startswith("read"):
      # Direct file reading
  elif query.startswith("calculate"):
      # Direct calculation
  ```

## Testing Tool Call Implementation

Always test your implementation with these scenarios:

1. **Explicit tool requests**: "Read the file test.txt"
2. **Implicit tool needs**: "What does the test.txt file contain?"
3. **Ambiguous requests**: "I need information from a file"
4. **Multi-tool scenarios**: "Read file.txt and summarize its contents"
5. **Non-tool queries**: "Hello, how are you today?"

Your implementation should handle all these scenarios properly, with the LLM determining when tools are needed.

## Security Considerations

1. **Validate tool inputs**: Always validate parameters before execution
2. **Implement access controls**: Restrict file access to safe directories
3. **Rate limit tool calls**: Prevent excessive resource usage
4. **Log all tool executions**: Maintain audit trails of all tool usage
5. **Sanitize tool outputs**: Ensure tool outputs don't contain harmful content

## Debugging Tool Call Issues

When tool calls aren't working as expected:

1. Enable DEBUG level logging
2. Check if the LLM is generating proper tool call formats
3. Verify tool execution is happening after LLM request (not before)
4. Confirm tool results are being sent back to the LLM
5. Review the final response to ensure it incorporates tool results 