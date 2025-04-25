# AbstractLLM Tool Call Documentation

## Introduction

This documentation provides comprehensive guidance on implementing and using tool calls in AbstractLLM. The most critical principle is that **tool calls must always be initiated by the LLM, never directly by the agent code**. Direct tool execution (based on pattern matching user input) introduces serious security vulnerabilities and undermines the LLM's ability to correctly interpret user intent.

## Core Documents

- [Architecture](architecture.md) - The proper architecture and flow for tool calls in AbstractLLM
- [Best Practices](best_practices.md) - Do's and don'ts for implementing tool calls
- [Security](security.md) - Security considerations and vulnerabilities to avoid
- [Troubleshooting](troubleshooting.md) - Common issues and how to fix them
- [Code Review Checklist](code_review_checklist.md) - Use this during code reviews to catch issues

## Tasks for Implementation

Specific implementation tasks are available in the [tasks directory](tasks/):

1. [01_remove_direct_execution.md](tasks/01_remove_direct_execution.md) - Remove direct tool execution from BasicAgent
2. [02_implement_llm_first_flow.md](tasks/02_implement_llm_first_flow.md) - Implement proper LLM-first flow
3. [03_add_security_validation.md](tasks/03_add_security_validation.md) - Add security validation for tool inputs
4. [04_improve_logging.md](tasks/04_improve_logging.md) - Improve logging for tool call flows
5. [05_add_tests.md](tasks/05_add_tests.md) - Add tests to verify correct tool call handling
6. [06_documentation_updates.md](tasks/06_documentation_updates.md) - Update user-facing documentation

## Quick Reference

### Correct Tool Call Flow

```
User → Agent → LLM → Tool Call Request → Agent → Tool Execution → LLM → Final Response → User
```

### Common Issues to Watch For

1. Pattern matching on user queries to determine tool use
2. Returning tool results directly to users without LLM interpretation
3. Multiple code paths that handle tool-related queries differently
4. Security vulnerabilities in tool execution
5. Missing validation on tool inputs
6. Insufficient logging of the tool call process

### Best Implementation Approach

```python
def run(self, query: str) -> str:
    # Create or get session
    session = self.session_manager.get_session(self.session_id)
    
    # Add user message
    session.add_message("user", query)
    
    # Define available tools
    tool_functions = {
        "read_file": read_file,
        # Other tools...
    }
    
    # Let the LLM decide if/which tools to use
    response = session.generate_with_tools(
        tool_functions=tool_functions,
        model=self.model_name,
        temperature=self.temperature
    )
    
    # Get the LLM's final response
    final_response = response.content if response.content else ""
    
    # Add assistant response to session
    session.add_message("assistant", final_response)
    
    # Return the LLM's response
    return final_response
```

## Key Warning Signs

If you see any of these patterns, they likely indicate improper tool call implementation:

```python
# 🚨 DANGER: Direct tool execution
if "file" in query.lower() and "read" in query.lower():
    # Extract filename and read directly

# 🚨 DANGER: Direct tool result return
tool_result = execute_tool(tool_call)
return f"The tool returned: {tool_result}"

# 🚨 DANGER: Bypassing LLM processing
if should_use_tool(query):
    # Special handling path
else:
    # Normal LLM path
``` 