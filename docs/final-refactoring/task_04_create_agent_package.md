# Task 04: Create Agent Package Structure (Priority 2)

**Duration**: 3 hours
**Risk**: Medium
**Dependencies**: Tasks 02-03 completed

## Objectives
- Create AbstractAgent package structure
- Implement main Agent class with smart memory selection
- Set up ReAct reasoning with learning from failures
- Integrate with AbstractLLM and AbstractMemory
- Leverage BasicSession from AbstractLLM(core) for conversation tracking

## Key Design Principles
- **Leverage AbstractLLM(core)**: Use BasicSession for all conversation tracking
- **Memory as a Lens**: Different users get different responses
- **Learn from Failures**: Track and learn from failed actions
- **No Over-engineering**: Simple agents use simple memory

## Steps

### 1. Create Package Structure (30 min)

```bash
# Navigate to new package location
cd /Users/albou/projects
mkdir -p abstractagent
cd abstractagent

# Create package structure
mkdir -p abstractagent/{orchestration,reasoning,workflows,strategies,tools,cli}
mkdir -p abstractagent/cli/commands
mkdir -p tests docs examples

# Create setup.py
cat > setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="abstractagent",
    version="1.0.0",
    author="AbstractLLM Team",
    description="Single agent orchestration framework for LLM agents",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "abstractllm>=2.0.0",
        "abstractmemory>=1.0.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",        # For CLI display
        "prompt-toolkit>=3.0",  # For enhanced input
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy"],
    },
    entry_points={
        'console_scripts': [
            'alma=abstractagent.cli.alma:main',
        ],
    },
)
EOF

# Create __init__.py files
touch abstractagent/__init__.py
touch abstractagent/orchestration/__init__.py
touch abstractagent/reasoning/__init__.py
touch abstractagent/workflows/__init__.py
touch abstractagent/strategies/__init__.py
touch abstractagent/tools/__init__.py
touch abstractagent/cli/__init__.py
```

### 2. Implement Main Agent Class with Smart Memory Selection (45 min)

Create `abstractagent/agent.py`:
```python
"""
Main Agent class - orchestrates LLM + Memory for autonomous behavior.
Smart memory selection based on agent purpose.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
import logging

from abstractllm import create_llm, BasicSession
from abstractllm.types import GenerateResponse
from abstractmemory import create_memory, ScratchpadMemory, BufferMemory, GroundedMemory

from .orchestration.coordinator import Coordinator
from .reasoning.react import ReActOrchestrator
from .strategies.retry import RetryStrategy
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """
    Intelligent agent with appropriate memory for its purpose.

    Examples:
        # Simple task agent with scratchpad
        agent = Agent(
            llm_config={"provider": "openai", "model": "gpt-4"},
            memory_type="scratchpad",
            purpose="task"
        )

        # Autonomous assistant with full memory
        agent = Agent(
            llm_config={"provider": "openai", "model": "gpt-4"},
            memory_type="temporal",
            purpose="assistant"
        )
    """

    def __init__(self,
                 llm_config: Dict[str, Any],
                 memory_type: Literal["none", "session", "scratchpad", "buffer", "grounded"] = "session",
                 memory_config: Optional[Dict[str, Any]] = None,
                 purpose: Literal["task", "chat", "assistant"] = "task",
                 tools: Optional[List[Any]] = None,
                 enable_reasoning: bool = True,
                 enable_retry: bool = True,
                 default_user: str = "default"):
        """
        Initialize agent with appropriate memory for its purpose.

        Args:
            llm_config: Configuration for LLM provider
            memory_type: Type of memory to use:
                - "none": No memory beyond basic session
                - "session": Use BasicSession from AbstractLLM (default)
                - "scratchpad": For ReAct and task agents
                - "buffer": For simple chatbots
                - "grounded": For autonomous agents with user tracking
            memory_config: Additional config for memory
            purpose: Agent purpose (helps select memory if not specified)
            tools: List of tools available to agent
            enable_reasoning: Enable ReAct reasoning
            enable_retry: Enable retry strategies
            default_user: Default user ID for grounded memory
        """

        # Initialize LLM
        self.llm = create_llm(**llm_config)

        # Initialize basic session (always present for conversation tracking)
        self.session = BasicSession(self.llm)

        # Auto-select memory based on purpose if not specified
        if memory_type == "session" and purpose != "task":
            if purpose == "chat":
                memory_type = "buffer"
            elif purpose == "assistant":
                memory_type = "grounded"  # Use grounded for autonomous agents

        # Initialize appropriate memory
        self.memory = None
        self.current_user = default_user

        if memory_type != "none" and memory_type != "session":
            config = memory_config or {}

            # Set sensible defaults based on memory type
            if memory_type == "scratchpad":
                config.setdefault("max_entries", 100)
            elif memory_type == "buffer":
                config.setdefault("max_messages", 100)
            elif memory_type == "grounded":
                config.setdefault("working_capacity", 10)
                config.setdefault("enable_kg", True)
                config.setdefault("default_user_id", default_user)

            self.memory = create_memory(memory_type, **config)
            logger.info(f"Initialized {memory_type} memory for {purpose} agent")

        # Initialize coordinator
        self.coordinator = Coordinator(self)

        # Initialize reasoning if enabled
        self.reasoner = None
        if enable_reasoning:
            self.reasoner = ReActOrchestrator(self)

        # Initialize retry strategy if enabled
        self.retry_strategy = None
        if enable_retry:
            self.retry_strategy = RetryStrategy()

        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

        # Tracking
        self.interaction_count = 0
        self.total_tokens = 0

    def set_user(self, user_id: str, relationship: Optional[str] = None):
        """Set the current user for personalized interactions"""
        self.current_user = user_id

        # Update grounded memory if available
        if isinstance(self.memory, GroundedMemory):
            self.memory.set_current_user(user_id, relationship)
            logger.info(f"Set current user to {user_id} ({relationship or 'unknown'})")

    def chat(self, prompt: str,
            use_reasoning: bool = False,
            use_tools: bool = False,
            max_iterations: int = 5,
            user_id: Optional[str] = None) -> str:
        """
        Main interaction method with user-aware memory.

        Args:
            prompt: User input
            use_reasoning: Use ReAct reasoning
            use_tools: Enable tool usage
            max_iterations: Max reasoning iterations
            user_id: Optional user ID for personalized interaction

        Returns:
            Agent's response
        """
        self.interaction_count += 1

        # Set user if provided
        if user_id:
            self.set_user(user_id)

        # Get appropriate context based on memory type
        context = None
        if self.memory:
            if isinstance(self.memory, ScratchpadMemory):
                # For ReAct agents, get recent cycle history
                context = self.memory.get_context(last_n=10)
            elif isinstance(self.memory, BufferMemory):
                # For chatbots, get conversation history
                context = self.memory.get_context(last_n=5)
            elif isinstance(self.memory, GroundedMemory):
                # For autonomous agents, get user-specific context
                context = self.memory.get_full_context(prompt[:50], user_id=user_id or self.current_user)
            elif hasattr(self.memory, 'get_full_context'):
                # Fallback for other memory types
                context = self.memory.get_full_context(prompt[:50])

        # Determine execution path
        if use_reasoning and self.reasoner:
            # Use ReAct reasoning (typically with ScratchpadMemory)
            response = self.reasoner.execute(
                prompt=prompt,
                context=context,
                tools=self.tool_registry if use_tools else None,
                max_iterations=max_iterations
            )
        elif use_tools and self.tool_registry.has_tools():
            # Use tools without reasoning
            response = self.coordinator.execute_with_tools(
                prompt=prompt,
                context=context,
                tools=self.tool_registry
            )
        else:
            # Direct generation
            response = self.coordinator.execute_direct(
                prompt=prompt,
                context=context
            )

        # Update memory based on type
        if self.memory:
            if isinstance(self.memory, ScratchpadMemory):
                # Track the exchange for task agents
                self.memory.add(f"User: {prompt[:100]}", "exchange")
                self.memory.add(f"Assistant: {response[:100]}", "exchange")
            elif isinstance(self.memory, BufferMemory):
                # Track full conversation for chatbots
                self.memory.add_message('user', prompt)
                self.memory.add_message('assistant', response)
            elif isinstance(self.memory, GroundedMemory):
                # Full memory update with user tracking
                self.memory.add_interaction(prompt, response, user_id=user_id or self.current_user)
            elif hasattr(self.memory, 'add_interaction'):
                # Fallback for other memory types
                self.memory.add_interaction(prompt, response)

        # Update session history (leveraging BasicSession from AbstractLLM Core)
        self.session.add_message('user', prompt)
        self.session.add_message('assistant', response)

        # Trigger memory consolidation for grounded memory
        if isinstance(self.memory, GroundedMemory):
            # Consolidate every 10 interactions
            if self.interaction_count % 10 == 0:
                self.memory.consolidate_memories()

        return response

    def think(self, prompt: str) -> str:
        """
        Generate a thought without acting.
        Used by reasoning components.
        """
        think_prompt = f"Think step by step about: {prompt}"
        response = self.llm.generate(think_prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def act(self, thought: str, available_tools: Optional[List] = None) -> Dict[str, Any]:
        """
        Decide on action based on thought.
        Used by reasoning components.
        """
        if not available_tools:
            return {'action': 'respond', 'content': thought}

        # Parse thought for tool calls
        if 'need to' in thought.lower() or 'should' in thought.lower():
            # Simple heuristic - would use better parsing
            for tool in available_tools:
                if tool.name.lower() in thought.lower():
                    return {
                        'action': 'tool',
                        'tool': tool.name,
                        'reasoning': thought
                    }

        return {'action': 'respond', 'content': thought}

    def observe(self, action_result: Any, action_context: Optional[str] = None) -> str:
        """
        Process action result into observation and track outcomes.
        Used by reasoning components.
        """
        if isinstance(action_result, dict):
            if action_result.get('error'):
                # Track failure for learning
                if isinstance(self.memory, GroundedMemory) and action_context:
                    self.memory.track_failure(action_context, action_result['error'])
                return f"Error: {action_result['error']}"
            if action_result.get('output'):
                # Track success for reinforcement
                if isinstance(self.memory, GroundedMemory) and action_context:
                    self.memory.track_success(action_context, str(action_result['output'])[:50])
                return f"Result: {action_result['output']}"

        return f"Observation: {action_result}"

    def reset(self):
        """Reset agent state"""
        self.session.clear_history()
        if self.memory:
            # Reset working memory only
            self.memory.working = WorkingMemory()
        self.interaction_count = 0

    def save_state(self, path: str):
        """Save agent state"""
        state = {
            'interaction_count': self.interaction_count,
            'total_tokens': self.total_tokens,
            'session_id': self.session.id
        }

        # Save session
        self.session.save(f"{path}/session.json")

        # Save memory if available
        if self.memory:
            self.memory.save(f"{path}/memory")

        # Save state
        import json
        with open(f"{path}/agent_state.json", 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load agent state"""
        import json
        from abstractllm import BasicSession

        # Load state
        with open(f"{path}/agent_state.json", 'r') as f:
            state = json.load(f)

        self.interaction_count = state['interaction_count']
        self.total_tokens = state['total_tokens']

        # Load session
        self.session = BasicSession.load(f"{path}/session.json")

        # Load memory if available
        if self.memory:
            self.memory.load(f"{path}/memory")
```

### 3. Implement Coordinator (30 min)

Create `abstractagent/orchestration/coordinator.py`:
```python
"""
Coordinator for single agent orchestration.
Note: This is NOT multi-agent coordination.
"""

from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinates LLM, memory, and tools for a single agent.
    """

    def __init__(self, agent):
        self.agent = agent

    def execute_direct(self, prompt: str, context: Optional[str] = None) -> str:
        """Execute direct generation without tools or reasoning"""

        # Build enhanced prompt with context
        if context:
            enhanced_prompt = f"""Context from memory:
{context}

User: {prompt}"""
        else:
            enhanced_prompt = prompt

        # Generate response
        response = self.agent.llm.generate(
            prompt=enhanced_prompt,
            messages=self.agent.session.get_messages()
        )

        return response.content if hasattr(response, 'content') else str(response)

    def execute_with_tools(self, prompt: str,
                           context: Optional[str] = None,
                           tools: Any = None) -> str:
        """Execute generation with tool support"""

        # Build context-enhanced prompt
        enhanced_prompt = prompt
        if context:
            enhanced_prompt = f"Context: {context}\n\n{prompt}"

        # Generate with tools
        response = self.agent.llm.generate(
            prompt=enhanced_prompt,
            messages=self.agent.session.get_messages(),
            tools=tools.get_definitions() if tools else []
        )

        # Execute tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                result = tools.execute(
                    tool_call.name,
                    tool_call.arguments
                )
                tool_results.append(result)

            # Generate final response with tool results
            follow_up = f"Tool results:\n{tool_results}\n\nNow respond to: {prompt}"
            final_response = self.agent.llm.generate(
                prompt=follow_up,
                messages=self.agent.session.get_messages()
            )
            return final_response.content if hasattr(final_response, 'content') else str(final_response)

        return response.content if hasattr(response, 'content') else str(response)
```

### 4. Implement ReAct Orchestrator (45 min)

Create `abstractagent/reasoning/react.py`:
```python
"""
ReAct reasoning implementation.
Based on SOTA research: Think -> Act -> Observe -> Repeat
"""

from typing import Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class ReActOrchestrator:
    """
    Implements ReAct reasoning cycles with scratchpad memory.

    Example usage:
        # Agent with scratchpad for ReAct
        agent = Agent(
            llm_config={"provider": "openai"},
            memory_type="scratchpad",
            purpose="task"
        )
        result = agent.chat(
            "Find the latest Python version",
            use_reasoning=True,
            use_tools=True
        )
    """

    def __init__(self, agent):
        self.agent = agent
        self.max_iterations = 5

    def execute(self, prompt: str,
                context: Optional[str] = None,
                tools: Optional[Any] = None,
                max_iterations: int = 5) -> str:
        """
        Execute ReAct reasoning cycle with scratchpad tracking.

        Pattern:
        1. Think about the problem
        2. Act (use tool or respond)
        3. Observe the result
        4. Repeat until solution found
        """
        self.max_iterations = max_iterations

        # Use scratchpad if available for tracking
        scratchpad = None
        if self.agent.memory and isinstance(self.agent.memory, ScratchpadMemory):
            scratchpad = self.agent.memory
            # Clear scratchpad for new reasoning cycle
            scratchpad.clear()

        current_prompt = prompt
        if context:
            current_prompt = f"Context: {context}\n\n{prompt}"

        for iteration in range(max_iterations):
            # THINK: Reason about the problem
            thought = self.agent.think(current_prompt)
            if scratchpad:
                scratchpad.add_thought(thought)
            logger.debug(f"Iteration {iteration} - Thought: {thought[:100]}...")

            # ACT: Decide on action
            action = self.agent.act(thought, tools.get_tools() if tools else None)
            if scratchpad and action.get('action') == 'tool':
                scratchpad.add_action(action['tool'], action.get('arguments', {}))

            if action['action'] == 'tool' and tools:
                # Execute tool
                tool_result = tools.execute(
                    action['tool'],
                    action.get('arguments', {})
                )

                # OBSERVE: Process tool result with context for learning
                observation = self.agent.observe(
                    tool_result,
                    action_context=f"{action['tool']} in iteration {iteration}"
                )
                if scratchpad:
                    scratchpad.add_observation(observation)

                # Update prompt for next iteration
                current_prompt = f"{prompt}\nObservation: {observation}"

            elif action['action'] == 'respond':
                # Final answer reached
                return action['content']

            # Safety check for max iterations
            if iteration == max_iterations - 1:
                # Force a response on last iteration
                summary = self._summarize_reasoning(thoughts, actions, observations)
                return f"After {max_iterations} iterations of reasoning:\n{summary}"

        return "Unable to complete reasoning within iteration limit."

    def _summarize_reasoning(self, thoughts: List[str],
                            actions: List[dict],
                            observations: List[str]) -> str:
        """Summarize the reasoning process"""
        summary = []

        for i, (thought, action) in enumerate(zip(thoughts, actions)):
            summary.append(f"Step {i+1}: {thought[:100]}...")
            if action['action'] == 'tool':
                summary.append(f"  Used tool: {action['tool']}")
                if i < len(observations):
                    summary.append(f"  Observed: {observations[i][:100]}...")

        return "\n".join(summary)
```

### 5. Implement Tool Registry (30 min)

Create `abstractagent/tools/registry.py`:
```python
"""
Tool registry for agent-specific advanced tools.
Note: Basic tools are in AbstractLLM. These are agent-level tools.
"""

from typing import Dict, List, Any, Callable
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for agent-specific tools.
    Extends beyond basic AbstractLLM tools.
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.definitions: List[Dict[str, Any]] = []

    def register(self, tool: Any):
        """Register a tool"""

        if callable(tool):
            # Function-based tool
            name = tool.__name__
            self.tools[name] = tool

            # Create definition from function
            definition = {
                'name': name,
                'description': tool.__doc__ or 'No description',
                'parameters': self._extract_parameters(tool)
            }
            self.definitions.append(definition)

        elif isinstance(tool, dict):
            # Definition-based tool
            name = tool['name']
            self.definitions.append(tool)
            # Note: Implementation should be provided separately

        logger.info(f"Registered tool: {name}")

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name"""

        if name not in self.tools:
            return {'error': f'Tool {name} not found'}

        try:
            result = self.tools[name](**arguments)
            return {'output': result}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {'error': str(e)}

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions"""
        return self.definitions

    def get_tools(self) -> List[Any]:
        """Get tool objects for reasoning"""
        return [{'name': name, 'function': func}
                for name, func in self.tools.items()]

    def has_tools(self) -> bool:
        """Check if any tools are registered"""
        return len(self.tools) > 0

    def _extract_parameters(self, func: Callable) -> Dict[str, Any]:
        """Extract parameters from function signature"""
        import inspect

        sig = inspect.signature(func)
        params = {}

        for name, param in sig.parameters.items():
            if name == 'self':
                continue

            param_info = {'type': 'string'}  # Default type

            if param.annotation != param.empty:
                # Try to infer type from annotation
                if param.annotation == int:
                    param_info['type'] = 'integer'
                elif param.annotation == float:
                    param_info['type'] = 'number'
                elif param.annotation == bool:
                    param_info['type'] = 'boolean'

            if param.default != param.empty:
                param_info['default'] = param.default

            params[name] = param_info

        return params
```

### 6. Implement CLI (30 min)

Create `abstractagent/cli/alma.py`:
```python
"""
ALMA CLI - The intelligent agent interface.
This replaces the monolithic CLI in abstractllm.
"""

import argparse
import sys
from pathlib import Path

from abstractagent import Agent
from abstractllm import create_llm


def create_agent_from_args(args) -> Agent:
    """Create agent from CLI arguments"""

    # LLM configuration
    llm_config = {
        'provider': args.provider,
        'model': args.model
    }

    # Memory configuration
    memory_config = None
    if args.memory:
        memory_config = {
            'persist_path': Path(args.memory),
            'temporal': True
        }

    # Create agent
    agent = Agent(
        llm_config=llm_config,
        memory_config=memory_config,
        enable_reasoning=not args.no_reasoning,
        enable_retry=not args.no_retry
    )

    # Load tools if specified
    if args.tools:
        from abstractagent.tools import load_tool_suite
        tools = load_tool_suite(args.tools)
        for tool in tools:
            agent.tool_registry.register(tool)

    return agent


def interactive_mode(agent: Agent):
    """Run interactive chat"""

    print("ALMA - Intelligent Agent")
    print("Type 'exit' to quit, 'help' for commands")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nUser: ")

            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'help':
                print_help()
                continue

            # Process input
            response = agent.chat(
                prompt=user_input,
                use_reasoning=True,
                use_tools=True
            )

            print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def print_help():
    """Print help message"""
    print("""
Commands:
  exit     - Quit the application
  help     - Show this message
  /memory  - Show memory statistics
  /tools   - List available tools
  /reasoning on|off - Toggle reasoning
    """)


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="ALMA - Intelligent Agent powered by AbstractLLM"
    )

    parser.add_argument(
        '--provider',
        default='ollama',
        help='LLM provider to use'
    )

    parser.add_argument(
        '--model',
        default='llama2',
        help='Model to use'
    )

    parser.add_argument(
        '--memory',
        help='Path to persist memory'
    )

    parser.add_argument(
        '--tools',
        help='Tool suite to load'
    )

    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help='Disable ReAct reasoning'
    )

    parser.add_argument(
        '--no-retry',
        action='store_true',
        help='Disable retry strategies'
    )

    parser.add_argument(
        '--prompt',
        help='Single prompt to execute'
    )

    args = parser.parse_args()

    # Create agent
    agent = create_agent_from_args(args)

    # Execute or run interactive
    if args.prompt:
        response = agent.chat(args.prompt)
        print(response)
    else:
        interactive_mode(agent)


if __name__ == '__main__':
    main()
```

## Usage Examples

### Example 1: Simple Task Agent with Scratchpad
```python
from abstractagent import Agent

# Create a task agent with scratchpad memory
agent = Agent(
    llm_config={
        "provider": "openai",
        "model": "gpt-4"
    },
    memory_type="scratchpad",  # Lightweight memory for ReAct
    purpose="task",
    enable_reasoning=True
)

# Execute a task with reasoning
result = agent.chat(
    "Find information about Python async programming",
    use_reasoning=True,
    use_tools=True
)

# The scratchpad automatically tracks thought-action-observation cycles
# No complex memory overhead for this simple task
```

### Example 2: Chatbot with Buffer Memory
```python
from abstractagent import Agent

# Create a chatbot with simple buffer memory
chatbot = Agent(
    llm_config={
        "provider": "openai",
        "model": "gpt-3.5-turbo"
    },
    memory_type="buffer",  # Simple conversation history
    memory_config={"max_messages": 50},
    purpose="chat"
)

# Have a conversation
response1 = chatbot.chat("What's the weather like?")
response2 = chatbot.chat("What did I just ask about?")
# Buffer memory maintains conversation context efficiently
```

### Example 3: Autonomous Assistant with Grounded Memory (User-Aware)
```python
from abstractagent import Agent

# Create an autonomous assistant with multi-dimensional grounded memory
assistant = Agent(
    llm_config={
        "provider": "openai",
        "model": "gpt-4"
    },
    memory_type="grounded",  # Multi-dimensional memory with WHO, WHEN, WHERE
    memory_config={
        "working_capacity": 10,
        "enable_kg": True  # Knowledge graph for relationships
    },
    purpose="assistant"
)

# Interaction with Alice
assistant.set_user("alice", relationship="owner")
response = assistant.chat(
    "My name is Alice and I love Python. I work at TechCorp",
    user_id="alice"
)
# Memory learns: Alice loves Python, works at TechCorp

# Later interaction with Bob
assistant.set_user("bob", relationship="colleague")
response = assistant.chat(
    "I prefer Java and work at StartupInc",
    user_id="bob"
)
# Memory learns: Bob prefers Java, works at StartupInc

# When Alice returns
response = assistant.chat(
    "What programming language should I use for my project?",
    user_id="alice"
)
# Agent knows Alice loves Python and responds accordingly

# When Bob asks the same question
response = assistant.chat(
    "What programming language should I use for my project?",
    user_id="bob"
)
# Agent knows Bob prefers Java and responds differently

# The agent provides personalized responses based on WHO is asking
```

### Example 4: Tool-only Agent (No Extra Memory)
```python
from abstractagent import Agent

# Create a tool agent that just uses BasicSession
tool_agent = Agent(
    llm_config={
        "provider": "openai",
        "model": "gpt-3.5-turbo"
    },
    memory_type="session",  # Just BasicSession from AbstractLLM
    purpose="task",
    tools=[calculator_tool, search_tool]
)

# Execute tool calls without memory overhead
result = tool_agent.chat(
    "Calculate 15% of 2500",
    use_tools=True
)
# No unnecessary memory for this simple calculation
```

## Validation

### Test agent functionality
```bash
cd /Users/albou/projects/abstractagent

# Test basic agent
python -c "
from abstractagent import Agent

agent = Agent(
    llm_config={'provider': 'ollama', 'model': 'llama2'},
    memory_config={'temporal': True}
)

response = agent.chat('Hello')
print(f'Response: {response}')
"

# Test with reasoning
python -c "
from abstractagent import Agent

agent = Agent(
    llm_config={'provider': 'ollama', 'model': 'llama2'},
    enable_reasoning=True
)

response = agent.chat(
    'What is 2+2?',
    use_reasoning=True
)
print(f'With reasoning: {response}')
"
```

### Test CLI
```bash
# Install in development mode
pip install -e .

# Test CLI
alma --help

# Interactive mode
alma --provider ollama --model llama2

# Single prompt
alma --prompt "Hello" --provider openai
```

## Success Criteria

- [ ] Agent class < 300 lines (core orchestration)
- [ ] Coordinator handles LLM + memory integration
- [ ] ReAct reasoning works independently
- [ ] Tool registry extensible
- [ ] CLI provides good UX
- [ ] All components properly separated

## Next Task

Proceed to Task 05: Testing and Integration

User: {prompt}