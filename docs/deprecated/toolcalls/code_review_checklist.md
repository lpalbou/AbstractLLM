# Tool Call Code Review Checklist

## Purpose

This checklist helps identify and fix issues related to direct tool execution in AbstractLLM implementations. Use it during code reviews and when troubleshooting tool-related problems.

## Anti-Patterns to Look For

### 1. Pattern Matching on User Queries

❌ **Red Flag**: Code that examines user queries for keywords and executes tools directly.

**What to look for:**
```python
# Problematic examples
if "file" in query.lower() and "read" in query.lower():
    # Direct tool execution
    
if query.startswith("read file"):
    # Direct tool execution
    
if re.match(r'read.*file', query, re.IGNORECASE):
    # Direct tool execution
```

**Solution:**
Remove pattern matching and use proper LLM-first tool execution.

### 2. Direct Tool Execution Shortcuts

❌ **Red Flag**: Code paths that bypass LLM processing for certain query types.

**What to look for:**
```python
def run(self, query):
    # Bypass checks
    if self._should_use_tool_directly(query):
        return self._execute_tool_directly(query)
    
    # Elsewhere in the code
    def _should_use_tool_directly(self, query):
        return "file" in query or "weather" in query
```

**Solution:**
Remove direct execution paths and ensure all queries go through the LLM.

### 3. Response Building Without LLM Processing

❌ **Red Flag**: Building responses that directly include tool output without LLM interpretation.

**What to look for:**
```python
result = read_file(filename)
return f"Here's the content of {filename}:\n{result}"

# Or
if tool_executed:
    return f"Tool result: {tool_result}"
else:
    return llm_response
```

**Solution:**
Always send tool results back to the LLM for interpretation and response generation.

### 4. Conditional Response Paths

❌ **Red Flag**: Different handling of the response flow based on detected keywords.

**What to look for:**
```python
if any(keyword in query for keyword in ["file", "read", "open"]):
    # Special handling path
else:
    # Normal LLM path
```

**Solution:**
Use a single, consistent path through the LLM for all queries.

### 5. Missing Tool Result Processing

❌ **Red Flag**: Missing code to send tool results back to the LLM.

**What to look for:**
```python
# Tool execution without sending results back to LLM
tool_result = execute_tool(tool_call)
# No follow-up LLM call with the result
```

**Solution:**
Ensure tool results are always sent back to the LLM for processing.

## LLM-First Implementation Patterns

✅ **Best Practice Examples**:

### Proper Tool Execution Flow

```python
def run(self, query: str) -> str:
    # Single path for all queries
    session = self.session_manager.get_session(self.session_id)
    session.add_message("user", query)
    
    # Define available tools
    tool_functions = {
        "read_file": read_file,
        "search_data": search_data,
        # Other tools...
    }
    
    # Let the LLM decide which tools to use (if any)
    response = session.generate_with_tools(
        tool_functions=tool_functions,
        model=self.model_name,
        temperature=self.temperature
    )
    
    # Return the LLM's response
    return response.content if response.content else ""
```

### Proper Tool Definition

```python
# Define tools once, use consistently
tools = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["file_path"]
        }
    }
]
```

## Issues to Fix by Priority

1. **Critical**: Direct execution of tools based on user query pattern matching
2. **Critical**: Bypassing LLM for tool selection decisions
3. **High**: Returning tool results directly to users without LLM processing
4. **High**: Multiple/inconsistent paths for handling tool-related queries
5. **Medium**: Insufficient logging of the tool call process
6. **Medium**: Lack of validation on tool inputs before execution

## Checklist for Reviewing Tool Call Implementations

- [ ] No pattern matching on user queries to determine tool use
- [ ] No conditional branching based on query content
- [ ] All user queries pass through the LLM
- [ ] The LLM makes all decisions about tool usage
- [ ] Tool results are always sent back to the LLM
- [ ] Final responses come from the LLM, not direct tool output
- [ ] Proper error handling for tool execution failures
- [ ] Consistent logging across the tool call process
- [ ] Tool validation to prevent security issues
- [ ] Tests covering various tool call scenarios 