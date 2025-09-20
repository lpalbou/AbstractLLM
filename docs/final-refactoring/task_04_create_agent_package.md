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

User: {prompt}