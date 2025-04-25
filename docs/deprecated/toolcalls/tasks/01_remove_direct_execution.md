# Task 1: Remove Direct Tool Execution

## Overview

Direct tool execution based on pattern matching user queries is a critical issue in the current implementation of BasicAgent. This task involves identifying and removing all instances of direct tool execution to ensure that all tool calls are initiated by the LLM, not by pattern matching in the agent code.

## Background

The current implementation of BasicAgent contains code that directly executes tools based on pattern matching user queries. This approach:

1. Bypasses the LLM's reasoning capabilities
2. Introduces security vulnerabilities
3. Prevents the LLM from incorporating tool results into its response
4. Creates inconsistent user experiences

As shown in the logs:
```
STEP 6: AGENT→USER - Direct tool execution was used instead of LLM-requested tool call
```

## Required Changes

### 1. Locate Direct Tool Execution Code

Search for pattern matching code in `basic_agent.py` that looks like:

```python
if "file" in query.lower() and "read" in query.lower():
    # Direct tool execution
```

or

```python
def _should_use_tool_directly(self, query):
    # Pattern matching logic
```

or any similar code that bypasses the LLM and directly executes tools based on user input patterns.

### 2. Remove Direct Execution Paths

Delete all code paths that directly execute tools based on pattern matching. This includes:

- Pattern matching conditionals (`if "file" in query.lower():`)
- Methods that determine tool use from the query alone
- Direct tool result return to the user
- Any shortcuts that bypass the standard LLM flow

### 3. Implement Proper LLM-First Flow

Replace all direct tool execution with the proper LLM-first flow:

```python
def run(self, query: str) -> str:
    """Run the agent on a query."""
    # Log the incoming query
    log_step(1, "USER→AGENT", f"Received query: {query[:100]}...")
    
    # Create or get session
    session = self.session_manager.get_session(self.session_id)
    
    # Add user message to session
    session.add_message("user", query)
    
    # Define available tools
    tool_functions = {
        "read_file": read_file,
        # Other tools...
    }
    
    # Log the LLM request
    log_step(2, "AGENT→LLM", f"Sending query to provider: {self.provider_name}")
    log_llm_request(query, tool_functions)
    
    # Generate response with tools - THIS SHOULD BE THE ONLY PATH
    response = session.generate_with_tools(
        tool_functions=tool_functions,
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_tokens
    )
    
    # Extract content from the response
    final_response = response.content if response.content else ""
    
    # Add assistant response to session
    session.add_message("assistant", final_response)
    
    # Log the final response
    log_step(5, "LLM→AGENT", f"Received final response from LLM")
    log_step(6, "AGENT→USER", f"Sending response to user: {final_response[:100]}...")
    
    # Return the final response
    return final_response
```

### 4. Update Streaming Implementation

If there's a separate streaming implementation, update it to use the same LLM-first approach:

```python
def run_streaming(self, query: str) -> None:
    """Run the agent on a query with streaming output."""
    # Similar to run but with streaming
    # ...
    
    # Use generate_with_tools_streaming
    for chunk in session.generate_with_tools_streaming(
        tool_functions=tool_functions,
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_tokens
    ):
        if isinstance(chunk, str):
            # Content chunk
            print(chunk, end="", flush=True)
        else:
            # Tool result
            print(f"\n[Executing tool: {chunk['name']}]\n", flush=True)
    
    print("\n")
```

### 5. Review and Update Logging

Update logging to clearly distinguish between the standardized flow steps:

1. User query → Agent
2. Agent → LLM (with tools)
3. LLM → Agent (with or without tool calls)
4. If tool calls present: Tool execution
5. Tool results → LLM
6. Final LLM response → Agent
7. Agent → User

## Testing

1. Test the updated implementation with queries that explicitly mention tools:
   - "Read the file test.txt"
   - "Calculate 2 + 2"

2. Test with queries that implicitly require tools:
   - "What does the test.txt file contain?"
   - "What's the result of 2 + 2?"

3. Test with queries that shouldn't use tools:
   - "Hello, how are you?"
   - "What's the weather like today?" (if no weather tool available)

4. Verify logs show the correct flow for each scenario

## Completion Criteria

- ✅ All direct tool execution code has been removed
- ✅ All queries go through the LLM for processing
- ✅ The LLM decides when to use tools
- ✅ Tool results are sent back to the LLM for interpretation
- ✅ The agent returns the LLM's final response to the user
- ✅ Logs show the proper tool call flow
- ✅ Tests pass with various query types 