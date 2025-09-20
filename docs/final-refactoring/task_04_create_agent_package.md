# Task 04: Create Agent Package Structure (Priority 2)

**Duration**: 3 hours
**Risk**: Medium
**Dependencies**: Tasks 02-03 completed

## Objectives
- Create AbstractAgent package structure
- Implement main Agent class
- Set up ReAct reasoning
- Integrate with AbstractLLM and AbstractMemory

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

### 2. Implement Main Agent Class (45 min)

Create `abstractagent/agent.py`:
```python
"""
Main Agent class - orchestrates LLM + Memory for autonomous behavior.
This replaces the complex Session class.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from abstractllm import create_llm, BasicSession
from abstractllm.types import GenerateResponse
from abstractmemory import TemporalMemory

from .orchestration.coordinator import Coordinator
from .reasoning.react import ReActOrchestrator
from .strategies.retry import RetryStrategy
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """
    Autonomous agent with LLM + Memory + Reasoning.
    Replaces the monolithic Session class with clean separation.
    """

    def __init__(self,
                 llm_config: Dict[str, Any],
                 memory_config: Optional[Dict[str, Any]] = None,
                 tools: Optional[List[Any]] = None,
                 enable_reasoning: bool = True,
                 enable_retry: bool = True):
        """
        Initialize agent with components.

        Args:
            llm_config: Configuration for LLM provider
            memory_config: Configuration for memory system
            tools: List of tools available to agent
            enable_reasoning: Enable ReAct reasoning
            enable_retry: Enable retry strategies
        """

        # Initialize LLM
        self.llm = create_llm(**llm_config)

        # Initialize basic session for conversation tracking
        self.session = BasicSession(self.llm)

        # Initialize memory if configured
        self.memory = None
        if memory_config:
            self.memory = TemporalMemory(**memory_config)

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

    def chat(self, prompt: str,
            use_reasoning: bool = False,
            use_tools: bool = False,
            max_iterations: int = 5) -> str:
        """
        Main interaction method.

        Args:
            prompt: User input
            use_reasoning: Use ReAct reasoning
            use_tools: Enable tool usage
            max_iterations: Max reasoning iterations

        Returns:
            Agent's response
        """
        self.interaction_count += 1

        # Get memory context if available
        context = None
        if self.memory:
            context = self.memory.retrieve_context(prompt)

        # Determine execution path
        if use_reasoning and self.reasoner:
            # Use ReAct reasoning
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

        # Update memory if available
        if self.memory:
            self.memory.add_interaction(prompt, response)

        # Update session history
        self.session.add_message('user', prompt)
        self.session.add_message('assistant', response)

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

    def observe(self, action_result: Any) -> str:
        """
        Process action result into observation.
        Used by reasoning components.
        """
        if isinstance(action_result, dict):
            if action_result.get('error'):
                return f"Error: {action_result['error']}"
            if action_result.get('output'):
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
    Implements ReAct reasoning cycles.
    This is NOT memory - it's orchestration.
    """

    def __init__(self, agent):
        self.agent = agent
        self.max_iterations = 5

    def execute(self, prompt: str,
                context: Optional[str] = None,
                tools: Optional[Any] = None,
                max_iterations: int = 5) -> str:
        """
        Execute ReAct reasoning cycle.

        Pattern:
        1. Think about the problem
        2. Act (use tool or respond)
        3. Observe the result
        4. Repeat until solution found
        """
        self.max_iterations = max_iterations

        # Initialize cycle tracking
        thoughts = []
        actions = []
        observations = []

        current_prompt = prompt
        if context:
            current_prompt = f"Context: {context}\n\n{prompt}"

        for iteration in range(max_iterations):
            # THINK: Reason about the problem
            thought = self.agent.think(current_prompt)
            thoughts.append(thought)
            logger.debug(f"Iteration {iteration} - Thought: {thought[:100]}...")

            # ACT: Decide on action
            action = self.agent.act(thought, tools.get_tools() if tools else None)
            actions.append(action)

            if action['action'] == 'tool' and tools:
                # Execute tool
                tool_result = tools.execute(
                    action['tool'],
                    action.get('arguments', {})
                )

                # OBSERVE: Process tool result
                observation = self.agent.observe(tool_result)
                observations.append(observation)

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