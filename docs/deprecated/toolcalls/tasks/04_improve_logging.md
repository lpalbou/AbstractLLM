# Task 4: Improve Logging for Tool Call Flow

## Overview

This task focuses on implementing comprehensive logging for the tool call flow in AbstractLLM. Proper logging is essential for debugging, security auditing, and understanding the interactions between the user, agent, LLM, and tools.

## Background

The current logging in AbstractLLM may not provide sufficient visibility into the tool call process. We need to implement detailed logging that covers each step of the correct flow:

```
User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User
```

The log entry `"Direct tool execution was used instead of LLM-requested tool call"` indicates that proper logging can help identify incorrect flow patterns.

## Required Changes

### 1. Define Standardized Log Points

Create standardized log points that clearly identify each step in the tool call flow:

```python
# At the top of the file
import logging
logger = logging.getLogger("abstractllm.tool_calls")

# Log step function
def log_step(step_number: int, step_name: str, message: str, level: int = logging.INFO) -> None:
    """
    Log a step in the tool call flow.
    
    Args:
        step_number: The step number in the flow
        step_name: The name of the step (e.g., "USER→AGENT")
        message: The log message
        level: The logging level (default: INFO)
    """
    logger.log(level, f"STEP {step_number}: {step_name} - {message}")
```

### 2. Implement Detailed Logging in the `run` Method

Update the `run` method with detailed logging:

```python
def run(self, query: str) -> str:
    """Run the agent on a query."""
    # Log the incoming query
    log_step(1, "USER→AGENT", f"Received query: {query[:100]}...")
    
    # Create or get session
    session = self.session_manager.get_session(self.session_id)
    session.add_message("user", query)
    
    # Define tool functions
    tool_functions = {
        "read_file": read_file,
        # Other tools...
    }
    logger.debug(f"Available tools: {list(tool_functions.keys())}")
    
    # Log the LLM request
    log_step(2, "AGENT→LLM", f"Sending query to provider: {self.provider_name}")
    
    try:
        # Generate response with tools
        log_start_time = time.time()
        response = session.generate_with_tools(
            tool_functions=tool_functions,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        log_duration = time.time() - log_start_time
        
        # Check for tool calls
        has_tool_calls = response.has_tool_calls()
        
        # Log LLM response
        if has_tool_calls:
            tool_calls = response.tool_calls.tool_calls
            tool_names = [tc.name for tc in tool_calls]
            log_step(3, "LLM→AGENT", f"LLM requested {len(tool_calls)} tool(s): {', '.join(tool_names)}")
            
            # Log each tool call
            for tool_call in tool_calls:
                log_step(4, "AGENT→TOOL", f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")
            
            # Log tool call completion
            log_step(5, "TOOL→AGENT", f"Tool execution completed")
            
            # Log sending results back to LLM
            log_step(6, "AGENT→LLM", f"Sending tool results to LLM")
        else:
            log_step(3, "LLM→AGENT", f"LLM response received (no tool calls) in {log_duration:.2f}s")
        
        # Extract content from the response
        final_response = response.content if response.content else ""
        
        # Log the final response
        log_step(7 if has_tool_calls else 4, "LLM→AGENT", f"Received final response from LLM")
        log_step(8 if has_tool_calls else 5, "AGENT→USER", f"Sending response to user: {final_response[:100]}...")
        
        # Add assistant response to session
        session.add_message("assistant", final_response)
        
        # Return the final response
        return final_response
        
    except Exception as e:
        error_msg = f"Error during agent execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg
```

### 3. Create Tool-Specific Logging Wrappers

Implement logging wrappers for tools:

```python
def logged_read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """Read file with logging."""
    logger.info(f"Executing read_file tool: path={file_path}, max_lines={max_lines}")
    start_time = time.time()
    
    try:
        result = read_file(file_path, max_lines)
        duration = time.time() - start_time
        result_length = len(result) if isinstance(result, str) else "N/A"
        logger.info(f"read_file completed in {duration:.2f}s, result length: {result_length}")
        return result
    except Exception as e:
        logger.error(f"Error in read_file: {str(e)}", exc_info=True)
        raise

def logged_calculate(operation: str, a: float, b: float) -> float:
    """Calculate with logging."""
    logger.info(f"Executing calculate tool: operation={operation}, a={a}, b={b}")
    start_time = time.time()
    
    try:
        result = calculate(operation, a, b)
        duration = time.time() - start_time
        logger.info(f"calculate completed in {duration:.2f}s, result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in calculate: {str(e)}", exc_info=True)
        raise

# Update tool_functions
tool_functions = {
    "read_file": logged_read_file,
    "calculate": logged_calculate,
    # Other logged tools...
}
```

### 4. Add Detailed Logging for Tool Results

Update tool result logging to include detailed information:

```python
def log_tool_result(tool_name: str, arguments: Dict[str, Any], result: Any, error: Optional[str] = None) -> None:
    """
    Log a tool execution result.
    
    Args:
        tool_name: Name of the tool
        arguments: Tool arguments
        result: Tool result
        error: Error message (if any)
    """
    if error:
        logger.error(f"Tool execution failed: {tool_name}")
        logger.error(f"Arguments: {json.dumps(arguments)}")
        logger.error(f"Error: {error}")
    else:
        logger.info(f"Tool execution succeeded: {tool_name}")
        logger.debug(f"Arguments: {json.dumps(arguments)}")
        
        # Log result summary based on type
        if isinstance(result, str):
            logger.debug(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
        else:
            logger.debug(f"Result: {result}")
```

### 5. Implement JSON Logging for Machine Parsing

Add structured JSON logging for tool calls to enable machine parsing:

```python
def log_json_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Log a structured JSON event.
    
    Args:
        event_type: Type of event
        data: Event data
    """
    # Add timestamp and event type
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": event_type,
        **data
    }
    
    # Log as JSON
    logger.info(f"JSON_EVENT: {json.dumps(log_data)}")
```

Usage examples:

```python
# Log tool call
log_json_event("tool_call", {
    "tool_name": tool_call.name,
    "tool_id": tool_call.id,
    "arguments": tool_call.arguments
})

# Log tool result
log_json_event("tool_result", {
    "tool_name": tool_call.name,
    "tool_id": tool_call.id,
    "success": error is None,
    "error": error,
    "execution_time": duration
})
```

### 6. Add Streaming-Specific Logging

Implement logging for streaming responses:

```python
def run_streaming(self, query: str) -> None:
    """Run the agent on a query with streaming output."""
    # Log the incoming query
    log_step(1, "USER→AGENT", f"Received query (streaming): {query[:100]}...")
    
    # Create or get session
    session = self.session_manager.get_session(self.session_id)
    session.add_message("user", query)
    
    # Define tool functions
    tool_functions = {
        "read_file": logged_read_file,
        "calculate": logged_calculate,
        # Other tools...
    }
    
    # Log the LLM request
    log_step(2, "AGENT→LLM", f"Sending query to provider (streaming): {self.provider_name}")
    
    try:
        # Track content for session history
        content_buffer = []
        
        # Process streaming response with tools
        log_step(3, "LLM→AGENT", f"Starting streaming response")
        
        # Track tool call status for logging
        tool_call_status = {
            "has_tool_calls": False,
            "current_step": 3
        }
        
        for chunk in session.generate_with_tools_streaming(
            tool_functions=tool_functions,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ):
            if isinstance(chunk, str):
                # Content chunk
                content_buffer.append(chunk)
                print(chunk, end="", flush=True)
            else:
                # Tool call detected
                if not tool_call_status["has_tool_calls"]:
                    tool_call_status["has_tool_calls"] = True
                    tool_call_status["current_step"] = 4
                    log_step(4, "LLM→AGENT", f"Tool call detected during streaming")
                
                # Tool result
                tool_name = chunk.get("name", "unknown")
                args = chunk.get("arguments", {})
                log_step(5, "AGENT→TOOL", f"Executing tool during streaming: {tool_name}")
                print(f"\n[Executing tool: {tool_name}]\n", flush=True)
                
                # Log tool completion
                log_step(6, "TOOL→AGENT", f"Tool execution completed during streaming")
        
        # Join content chunks for session history
        final_content = "".join(content_buffer)
        
        # Log completion
        next_step = tool_call_status["current_step"] + 1
        log_step(next_step, "LLM→AGENT", f"Streaming completed, final response length: {len(final_content)}")
        log_step(next_step + 1, "AGENT→USER", f"Sending final streaming response to user")
        
        # Add assistant response to session
        session.add_message("assistant", final_content)
        
        # Complete the output with a newline
        print("\n")
        
    except Exception as e:
        error_msg = f"Error during streaming: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"\nError: {error_msg}")
```

### 7. Add Configuration for Logging

Add logging configuration options:

```python
# At the top of the file
import os

# Configure logging
LOG_LEVEL = os.environ.get("ABSTRACTLLM_LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get(
    "ABSTRACTLLM_LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE = os.environ.get("ABSTRACTLLM_LOG_FILE", None)

def configure_logging():
    """Configure logging based on environment variables."""
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        filename=LOG_FILE
    )
    
    # Create abstractllm logger
    logger = logging.getLogger("abstractllm")
    logger.setLevel(level)
    
    # Create console handler if not logging to file
    if not LOG_FILE:
        console = logging.StreamHandler()
        console.setLevel(level)
        formatter = logging.Formatter(LOG_FORMAT)
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    return logger

# Call at application startup
logger = configure_logging()
```

### 8. Add Logging in the Session Class

If possible, add logging to the Session class methods:

```python
# In Session.execute_tool_call
def execute_tool_call(self, tool_call, tool_functions):
    """Execute a tool call with logging."""
    logger.info(f"Executing tool call: {tool_call.name}")
    logger.debug(f"Tool call arguments: {tool_call.arguments}")
    
    # ... existing implementation ...
    
    # Log result
    if error:
        logger.error(f"Tool execution failed: {error}")
    else:
        logger.info(f"Tool execution succeeded: {tool_call.name}")
        if isinstance(result, str):
            logger.debug(f"Result preview: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
```

## Testing

1. Test logging in standard scenarios:
   - Queries that don't require tools
   - Queries that require a single tool
   - Queries that require multiple tools
   - Error scenarios

2. Test streaming logging:
   - Verify that streaming log points are correctly ordered
   - Check that tool executions during streaming are properly logged

3. Test JSON logging:
   - Verify that JSON events are properly formatted
   - Ensure they can be parsed by log analysis tools

4. Test log configuration:
   - Test different log levels
   - Test logging to file
   - Test console logging

## Completion Criteria

- ✅ Each step in the tool call flow has a clear log entry
- ✅ Tool executions are logged with timing information
- ✅ Errors are comprehensively logged
- ✅ Streaming has appropriate log points
- ✅ JSON logging is available for machine parsing
- ✅ Log configuration is externally configurable
- ✅ Session class has appropriate logging
- ✅ Tests verify correct logging behavior 