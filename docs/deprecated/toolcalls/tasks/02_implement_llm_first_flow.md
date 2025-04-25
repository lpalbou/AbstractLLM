# Task 2: Implement LLM-First Tool Call Flow

## Overview

This task focuses on implementing the proper LLM-first tool call flow in AbstractLLM's BasicAgent implementation, ensuring that all tool usage decisions are made by the LLM, not the agent code.

## Background

After removing direct tool execution (Task 1), we need to ensure the proper LLM-first flow is consistently implemented for all queries. The correct flow follows this sequence:

```
User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User
```

In this flow:
1. The user sends a query to the agent
2. The agent forwards the query to the LLM with available tools
3. The LLM decides if and which tools to use
4. If tools are needed, the agent executes them and returns results to the LLM
5. The LLM generates a final response incorporating tool results
6. The agent returns the final response to the user

## Required Changes

### 1. Update the `run` Method

Implement the proper LLM-first flow in the main `run` method:

```python
def run(self, query: str) -> str:
    """Run the agent on a query."""
    # Log the incoming query
    log_step(1, "USER→AGENT", f"Received query: {query[:100]}...")
    
    # Create or get session
    session = self.session_manager.get_session(self.session_id)
    
    # Add user message to session
    session.add_message("user", query)
    
    # Define tool functions - ALL available tools should be provided
    tool_functions = {
        "read_file": read_file,
        "calculate": calculate,
        "search_data": search_data,
        # Include ALL tools the agent can use
    }
    
    # Log the LLM request
    log_step(2, "AGENT→LLM", f"Sending query to provider: {self.provider_name}")
    log_llm_request(query, tool_functions)
    
    try:
        # Generate response with tools
        response = session.generate_with_tools(
            tool_functions=tool_functions,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Extract content from the response
        final_response = response.content if response.content else ""
        
        # Log the final response
        log_step(5, "LLM→AGENT", f"Received final response from LLM")
        log_step(6, "AGENT→USER", f"Sending response to user: {final_response[:100]}...")
        
        # Add assistant response to session
        session.add_message("assistant", final_response)
        
        # Return the final response
        return final_response
        
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}"
        logger.error(error_msg)
        return error_msg
```

### 2. Update Streaming Method

Implement the proper flow in the streaming method:

```python
def run_streaming(self, query: str) -> None:
    """Run the agent on a query with streaming output."""
    # Log the incoming query
    log_step(1, "USER→AGENT", f"Received query: {query[:100]}...")
    
    # Create or get session
    session = self.session_manager.get_session(self.session_id)
    
    # Add user message to session
    session.add_message("user", query)
    
    # Define tool functions - ALL available tools should be provided
    tool_functions = {
        "read_file": read_file,
        "calculate": calculate,
        "search_data": search_data,
        # Include ALL tools the agent can use
    }
    
    # Log the LLM request
    log_step(2, "AGENT→LLM", f"Sending query to provider: {self.provider_name}")
    log_llm_request(query, tool_functions)
    
    try:
        # Track content for session history
        content_buffer = []
        
        # Process streaming response with tools
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ):
            if isinstance(chunk, str):
                # Content chunk - display and save
                print(chunk, end="", flush=True)
                content_buffer.append(chunk)
            else:
                # Tool result notification
                tool_name = chunk.get("name", "unknown")
                print(f"\n[Executing tool: {tool_name}]\n", flush=True)
        
        # Join content chunks for session history
        final_content = "".join(content_buffer)
        
        # Add assistant response to session
        session.add_message("assistant", final_content)
        
        # Complete the output with a newline
        print("\n")
        
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}"
        logger.error(error_msg)
        print(f"\nError: {error_msg}")
```

### 3. Implement Proper Error Handling

Add better error handling for tool call issues:

```python
try:
    response = session.generate_with_tools(
        tool_functions=tool_functions,
        model=self.model_name,
        temperature=self.temperature,
        max_tokens=self.max_tokens
    )
except UnsupportedFeatureError as e:
    logger.error(f"Tool calling not supported: {e}")
    return f"I'm sorry, tool calling is not supported with the current configuration: {str(e)}"
except Exception as e:
    logger.error(f"Error during tool execution: {e}")
    return f"I encountered an error while processing your request: {str(e)}"
```

### 4. Add Tool Call Logging

Implement detailed logging for tool calls:

```python
# Before calling generate_with_tools
logger.debug(f"Available tools: {list(tool_functions.keys())}")

# Inside the tool function wrappers
def logged_read_file(file_path, max_lines=None):
    logger.info(f"Executing read_file: path={file_path}, max_lines={max_lines}")
    try:
        result = read_file(file_path, max_lines)
        logger.info(f"read_file complete: result length={len(result)}")
        return result
    except Exception as e:
        logger.error(f"read_file error: {str(e)}")
        raise

# Update tool_functions to use logged versions
tool_functions = {
    "read_file": logged_read_file,
    # Other logged tools...
}
```

### 5. Handle Multi-Turn Conversations

Ensure multi-turn conversations maintain context:

```python
def run_interactive(self):
    """Run the agent in interactive mode."""
    print("Starting interactive session. Type 'exit' to quit.\n")
    
    # Create a new session for the conversation
    session = self.session_manager.get_session(self.session_id)
    
    while True:
        # Get user input
        user_input = input("> ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Ending session. Goodbye!")
            break
            
        # Process the query through the LLM-first flow
        response = self.run(user_input)
        
        # Display the response (already added to session in run)
        print(f"\n{response}\n")
```

## Integration with Session API

Ensure the agent uses the Session API correctly to maintain conversation history and pass tool results back to the LLM:

1. Create or get a session
2. Add user messages to the session
3. Use session.generate_with_tools() for processing
4. Add assistant responses back to the session
5. Leverage the session history for multi-turn conversations

## Testing

1. Test explicit tool requests:
   - "Read the file test.txt"
   - "What is 2 + 2?"

2. Test implicit tool requests:
   - "What's in the test.txt file?"
   - "Tell me about the contents of test.txt"

3. Test multi-turn conversations:
   - "Hello"
   - "Can you read a file for me?"
   - "The file is test.txt"

4. Test error conditions:
   - "Read a non-existent file"
   - "Calculate 1/0"

5. Verify proper logging for each scenario

## Completion Criteria

- ✅ The run method correctly follows the LLM-first flow
- ✅ The streaming implementation follows the same flow
- ✅ All available tools are provided to the LLM
- ✅ The LLM decides when to use tools
- ✅ Tool results are properly passed back to the LLM
- ✅ Final responses incorporate tool results
- ✅ Multi-turn conversations maintain context
- ✅ Proper error handling is implemented
- ✅ Logging correctly tracks the flow 