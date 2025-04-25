# Tool Call Troubleshooting Guide

## Common Tool Call Issues

### Issue 1: Direct Tool Execution

**Symptoms:**
- Log messages like: `"Direct tool execution was used instead of LLM-requested tool call"`
- Tool results are returned directly to users without LLM interpretation
- Tool calls happen immediately after user query, before LLM processing

**Diagnosis:**
1. Look for pattern matching on user queries in the agent implementation
2. Check for code that directly executes tools based on keywords in user input
3. Look for response construction that bypasses the LLM

**Fix:**
1. Remove all direct tool execution code
2. Ensure queries always go to the LLM first
3. Use `session.generate_with_tools()` consistently
4. Make sure tool results are sent back to the LLM for processing

**Example Problematic Code:**
```python
def run(self, query: str) -> str:
    # Problematic pattern matching
    if "file" in query.lower() and "read" in query.lower():
        # Extract filename
        file_path = extract_filename(query)
        
        # Direct tool execution
        try:
            result = read_file(file_path)
            return f"I've read the file for you:\n\n{result[:200]}..."
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    # Normal LLM flow
    response = session.generate_with_tools(...)
    return response.content
```

**Fixed Code:**
```python
def run(self, query: str) -> str:
    # Create or get session
    session = self.session_manager.get_session(self.session_id)
    
    # Add user message
    session.add_message("user", query)
    
    # Define tool functions
    tool_functions = {
        "read_file": read_file,
        # Other tools...
    }
    
    # Let the LLM decide if/which tools to use
    response = session.generate_with_tools(
        tool_functions=tool_functions,
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_tokens
    )
    
    # Get the LLM's final response
    final_response = response.content if response.content else ""
    
    # Add assistant response to session
    session.add_message("assistant", final_response)
    
    # Return the LLM's response
    return final_response
```

### Issue 2: Tool Results Not Returned to LLM

**Symptoms:**
- LLM responses don't incorporate tool results
- Multiple tool calls for the same information
- Responses like "I'll need to check that file" even after the file was read

**Diagnosis:**
1. Check if tool results are being passed back to the LLM
2. Verify that the conversation history includes tool results
3. Check if the provider supports tool response formats

**Fix:**
1. Ensure tool results are formatted correctly
2. Verify they're included in the next LLM request
3. Check provider-specific format requirements

### Issue 3: Tool Calls Not Being Detected

**Symptoms:**
- LLM tries to call tools but no tool execution happens
- Log shows no tool call detection
- Responses contain text like "I'll use the read_file tool" but no actual tool call occurs

**Diagnosis:**
1. Check if tool definitions are correctly passed to the LLM
2. Verify that the provider's tool call format is correctly parsed
3. Check if the `_extract_tool_calls` method is working properly

**Fix:**
1. Ensure tools are properly defined and passed to the provider
2. Verify provider-specific tool formats
3. Test tool call extraction logic independently

## Tool Call Logging Checklist

When troubleshooting, enable these log points and verify the expected flow:

1. ✅ Initial user query received
2. ✅ Query sent to LLM with available tools
3. ✅ LLM response received with tool call request
4. ✅ Tool call extracted and executed
5. ✅ Tool execution results obtained
6. ✅ Results sent back to LLM for processing
7. ✅ Final LLM response incorporates tool results
8. ✅ Response returned to user

## Provider-Specific Considerations

### OpenAI
- Tool calls appear in the `tool_calls` field of the message
- Each tool call has an ID that must be included in the result

### Anthropic
- Tool calls appear in the `tool_use` sections of the response
- Anthropic uses a slightly different format for tool definitions

### Ollama
- May require custom handling of tool call formats
- Check for provider-specific adaptations in the code

## Preventing Tool Call Issues

1. Add test cases specifically for tool call sequences
2. Implement logging for each step of the tool call process
3. Add validation to reject direct tool execution patterns
4. Create integration tests with different query patterns
5. Use static analysis to find pattern matching on user queries 