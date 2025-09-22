# Architecture Decision: Tool Execution in AbstractLLM Core

## Executive Summary

**Decision**: AbstractLLM Core should execute single tool calls while emitting events for observability and optional interception.

## Context

During the refactoring analysis, a key question arose: Should AbstractLLM Core:
1. **Execute** tool calls directly, or
2. **Emit events** for external systems to handle execution?

## Analysis

### SOTA Research (2025)
- **LangChain/LangGraph**: Core executes tools but provides hooks for customization
- **LlamaIndex**: Tools executed at core level with event-driven observability
- **Industry Pattern**: Separation between tool *detection/parsing* (LLM layer) and tool *orchestration* (Agent layer)

### Architectural Considerations

**LLM Core Responsibilities:**
- Unified tool format translation across providers
- Tool call detection and parsing from responses
- Single tool execution with proper error handling
- Event emission for observability

**Agent Layer Responsibilities:**
- Complex orchestration (retry, fallback, validation)
- ReAct reasoning cycles with multiple tool calls
- Parallel tool execution across multiple LLM calls
- Workflow state management

## Decision

**Chosen Approach**: Execution with Interception Hooks (Proposal 3)

```python
class AbstractLLMInterface:
    def generate(self, prompt, tools=None, **kwargs):
        response = self.provider.generate(prompt, tools, **kwargs)
        if tool_calls := self.parse_tool_calls(response):
            # Emit pre-execution event with prevention capability
            event = self.emit(EventType.BEFORE_TOOL_EXECUTION, {
                "tool_calls": tool_calls,
                "can_prevent": True
            })

            if not event.prevented:
                # Execute tools synchronously
                results = self.tool_handler.execute_all(tool_calls)

                # Emit completion event
                self.emit(EventType.AFTER_TOOL_EXECUTION, {
                    "tool_calls": tool_calls,
                    "results": results
                })

                return self._merge_tool_results(response, results)
        return response
```

## Justification

### 1. **Simplicity for Common Cases**
99% of use cases just need tools to work without additional setup:
```python
llm = create_llm('openai')
response = llm.generate("What's the weather?", tools=[weather_tool])
# Tool executes automatically, result included in response
```

### 2. **Aligns with SOTA Patterns**
Mirrors successful patterns from Express middleware, Django signals, and React lifecycle methods.

### 3. **Sequential Execution is Correct at Core Level**
- Single tool call must complete before LLM processes result
- Generation must wait for tool completion anyway
- Parallelization belongs at Agent layer for independent tasks

### 4. **Maintains Separation of Concerns**
- **Core**: Handles *mechanism* (how to call a tool)
- **Agent**: Handles *policy* (when/whether to call, retry logic)

### 5. **Event System for Advanced Use Cases**
```python
# Advanced orchestration via event interception
llm.on(EventType.BEFORE_TOOL_EXECUTION, lambda e:
    e.prevent() if not validate_tool_call(e.data) else None)
```

## Impact on Refactoring

### Documentation Updates Required

1. **BasicSession scope** (01-architecture-final.md:55-56):
   ```markdown
   ├── session.py              # BasicSession - 500 lines MAX
   │                           # ONLY: add_message, get_messages, generate_with_tools
   ```

2. **Tool execution flow** clarification in diagrams.md

3. **Event system specification** for tool lifecycle

4. **Clear boundaries** between Core and Agent responsibilities

### Code Implementation

- BasicSession gains tool execution capability
- Tool events added to event system
- Agent layer focuses on orchestration patterns
- Maintains backward compatibility

## Alternatives Considered

1. **Pure Event-Emission**: Too complex for simple cases
2. **Dual-Mode Architecture**: Multiple code paths increase maintenance burden

## Success Metrics

- [ ] Simple tool calls work without event handlers
- [ ] Complex orchestration possible via events
- [ ] Zero breaking changes for basic usage
- [ ] Clear separation between core execution and agent orchestration
- [ ] Event system provides full observability

---

**Status**: Approved
**Date**: 2025-01-XX
**Review**: Required before implementation