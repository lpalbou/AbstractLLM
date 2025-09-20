# Task 05: Testing and Validation (Priority 1)

**Duration**: 4 hours
**Risk**: High
**Dependencies**: Tasks 01-04 completed

## Objectives
- Create comprehensive test suites for each package
- Validate separation of concerns
- Test integration between packages
- Performance benchmarking
- Ensure backward compatibility

## Steps

### 1. Test Infrastructure Setup (30 min)

```bash
# Create test structure for all three packages
cd /Users/albou/projects

# AbstractLLM tests
cd abstractllm
mkdir -p tests/{unit,integration,performance}
touch tests/__init__.py
touch tests/conftest.py

# Create pytest configuration
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow tests
EOF

# AbstractMemory tests
cd ../abstractmemory
mkdir -p tests/{unit,integration,performance}
cp ../abstractllm/pytest.ini .

# AbstractAgent tests
cd ../abstractagent
mkdir -p tests/{unit,integration,performance}
cp ../abstractllm/pytest.ini .
```

### 2. Unit Tests for AbstractLLM Core (60 min)

Create `abstractllm/tests/unit/test_basic_session.py`:
```python
"""
Unit tests for BasicSession - the core conversation manager.
"""

import pytest
from datetime import datetime
from abstractllm._refactoring.session.basic import BasicSession, Message


class TestBasicSession:
    """Test BasicSession functionality"""

    def test_session_creation(self):
        """Test session can be created"""
        session = BasicSession()
        assert session.id is not None
        assert isinstance(session.created_at, datetime)
        assert len(session.messages) == 0

    def test_add_message(self):
        """Test adding messages"""
        session = BasicSession()

        # Add user message
        msg = session.add_message('user', 'Hello')
        assert msg.role == 'user'
        assert msg.content == 'Hello'
        assert len(session.messages) == 1

    def test_system_prompt(self):
        """Test system prompt handling"""
        session = BasicSession(system_prompt="You are helpful")

        # Should have one system message
        assert len(session.messages) == 1
        assert session.messages[0].role == 'system'

    def test_get_history(self):
        """Test conversation history retrieval"""
        session = BasicSession(system_prompt="System")
        session.add_message('user', 'Hello')
        session.add_message('assistant', 'Hi')

        # With system
        history = session.get_history(include_system=True)
        assert len(history) == 3

        # Without system
        history = session.get_history(include_system=False)
        assert len(history) == 2

    def test_clear_history(self):
        """Test clearing conversation"""
        session = BasicSession(system_prompt="System")
        session.add_message('user', 'Hello')

        # Clear keeping system
        session.clear_history(keep_system=True)
        assert len(session.messages) == 1
        assert session.messages[0].role == 'system'

        # Clear completely
        session.clear_history(keep_system=False)
        assert len(session.messages) == 0

    def test_persistence(self, tmp_path):
        """Test save and load"""
        # Create and save
        session = BasicSession(system_prompt="Test")
        session.add_message('user', 'Hello')
        session.add_message('assistant', 'Hi')

        save_path = tmp_path / "session.json"
        session.save(save_path)

        # Load and verify
        loaded = BasicSession.load(save_path)
        assert loaded.id == session.id
        assert len(loaded.messages) == 3
        assert loaded.messages[1].content == 'Hello'
```

Create `abstractllm/tests/unit/test_provider_abstraction.py`:
```python
"""
Test provider abstraction - each provider handles tools differently.
"""

import pytest
from abstractllm.tools import ToolDefinition


class TestProviderToolAbstraction:
    """Test tool handling across providers"""

    def test_openai_tool_format(self):
        """Test OpenAI uses function format"""
        from abstractllm.providers.openai import OpenAIProvider

        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"query": {"type": "string"}}
        )

        # OpenAI should convert to function format
        provider = OpenAIProvider()
        formatted = provider._format_tools_for_api([tool])

        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "search"

    def test_anthropic_tool_format(self):
        """Test Anthropic uses XML format"""
        from abstractllm.providers.anthropic import AnthropicProvider

        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"query": {"type": "string"}}
        )

        # Anthropic uses XML in content
        provider = AnthropicProvider()
        formatted = provider._format_tools_for_prompt([tool])

        assert "<tool_call>" in formatted
        assert "search" in formatted

    def test_ollama_architecture_detection(self):
        """Test Ollama detects architecture for format"""
        from abstractllm.providers.ollama import OllamaProvider
        from abstractllm.architectures import detect_architecture

        # Qwen model
        arch = detect_architecture("qwen:7b")
        assert arch == "qwen"

        # Llama model
        arch = detect_architecture("llama3:8b")
        assert arch == "llama"

        # Format differs by architecture
        provider = OllamaProvider(model="qwen:7b")
        assert provider._get_tool_format() == "<|tool_call|>"

        provider = OllamaProvider(model="llama3:8b")
        assert provider._get_tool_format() == "<function_call>"


class TestMediaAbstraction:
    """Test media handling across providers"""

    def test_image_handling_differences(self):
        """Test providers handle images differently"""
        from abstractllm.media.processor import MediaProcessor

        processor = MediaProcessor()

        # OpenAI wants base64
        openai_format = processor.process_for_provider(
            "image.jpg",
            provider="openai"
        )
        assert openai_format.startswith("data:image")

        # Anthropic wants different format
        anthropic_format = processor.process_for_provider(
            "image.jpg",
            provider="anthropic"
        )
        assert anthropic_format != openai_format
```

### 3. Unit Tests for AbstractMemory (60 min)

Create `abstractmemory/tests/unit/test_temporal_memory.py`:
```python
"""
Test temporal knowledge graph implementation.
"""

import pytest
from datetime import datetime, timedelta
from abstractmemory.core import TemporalMemory
from abstractmemory.graph import TemporalKnowledgeGraph


class TestTemporalMemory:
    """Test temporal memory system"""

    def test_bi_temporal_anchoring(self):
        """Test bi-temporal data model"""
        memory = TemporalMemory()

        # Add fact with both timestamps
        event_time = datetime(2024, 1, 1, 10, 0)
        ingestion_time = datetime.now()

        memory.add_fact(
            subject="Alice",
            predicate="visited",
            object="Paris",
            event_time=event_time,
            ingestion_time=ingestion_time
        )

        # Can query by event time
        facts_at_event = memory.query_at_time(event_time)
        assert len(facts_at_event) == 1

        # Can query by ingestion time
        facts_at_ingestion = memory.as_of(ingestion_time)
        assert len(facts_at_ingestion) == 1

    def test_working_memory_window(self):
        """Test 10-item sliding window"""
        from abstractmemory.components import WorkingMemory

        memory = WorkingMemory(size=10)

        # Add 15 items
        for i in range(15):
            memory.add(f"Item {i}")

        # Should only have last 10
        assert len(memory.items) == 10
        assert memory.items[0] == "Item 5"
        assert memory.items[-1] == "Item 14"

    def test_episodic_memory(self):
        """Test episodic event storage"""
        from abstractmemory.components import EpisodicMemory

        memory = EpisodicMemory()

        # Add episode
        episode_id = memory.add_episode(
            event="User asked about weather",
            context={"location": "NYC"},
            timestamp=datetime.now()
        )

        # Retrieve by time range
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)
        episodes = memory.get_episodes_in_range(start, end)

        assert len(episodes) == 1
        assert episodes[0]["id"] == episode_id

    def test_semantic_memory(self):
        """Test semantic fact storage"""
        from abstractmemory.components import SemanticMemory

        memory = SemanticMemory()

        # Add facts
        memory.add_fact("Python", "is_a", "Programming Language")
        memory.add_fact("Python", "created_by", "Guido van Rossum")

        # Query by subject
        facts = memory.get_facts_about("Python")
        assert len(facts) == 2

        # Query by predicate
        is_a_facts = memory.get_facts_with_predicate("is_a")
        assert len(is_a_facts) == 1


class TestKnowledgeGraph:
    """Test knowledge graph functionality"""

    def test_graph_traversal(self):
        """Test graph traversal for retrieval"""
        graph = TemporalKnowledgeGraph()

        # Build graph
        graph.add_entity("Alice", type="Person")
        graph.add_entity("Bob", type="Person")
        graph.add_entity("Paris", type="City")

        graph.add_relation("Alice", "knows", "Bob")
        graph.add_relation("Alice", "visited", "Paris")
        graph.add_relation("Bob", "lives_in", "Paris")

        # Traverse from Alice
        neighbors = graph.get_neighbors("Alice", depth=1)
        assert "Bob" in neighbors
        assert "Paris" in neighbors

        # Two-hop traversal
        extended = graph.get_neighbors("Alice", depth=2)
        assert len(extended) > len(neighbors)

    def test_ontology_building(self):
        """Test automatic ontology construction"""
        from abstractmemory.graph import OntologyBuilder

        builder = OntologyBuilder()

        # Add facts
        builder.observe("Python", "is_a", "Language")
        builder.observe("Java", "is_a", "Language")
        builder.observe("Language", "is_a", "Tool")

        # Should build hierarchy
        ontology = builder.get_ontology()
        assert ontology.is_subclass("Python", "Tool")
        assert ontology.is_subclass("Java", "Language")
```

### 4. Unit Tests for AbstractAgent (45 min)

Create `abstractagent/tests/unit/test_agent.py`:
```python
"""
Test Agent orchestration.
"""

import pytest
from abstractagent import Agent


class TestAgent:
    """Test agent functionality"""

    def test_agent_creation(self):
        """Test agent can be created"""
        agent = Agent(
            llm_config={'provider': 'mock', 'model': 'test'},
            memory_config={'temporal': True}
        )

        assert agent.llm is not None
        assert agent.memory is not None
        assert agent.coordinator is not None

    def test_simple_chat(self):
        """Test basic chat without tools/reasoning"""
        agent = Agent(
            llm_config={'provider': 'mock', 'model': 'test'}
        )

        response = agent.chat("Hello")
        assert response is not None
        assert agent.interaction_count == 1

    def test_think_act_observe(self):
        """Test TAO methods for ReAct"""
        agent = Agent(
            llm_config={'provider': 'mock', 'model': 'test'}
        )

        # Think
        thought = agent.think("What is 2+2?")
        assert "think" in thought.lower() or "4" in thought

        # Act
        action = agent.act(thought)
        assert action['action'] in ['respond', 'tool']

        # Observe
        observation = agent.observe({'output': '4'})
        assert "4" in observation


class TestReActReasoning:
    """Test ReAct orchestration"""

    def test_react_cycle(self):
        """Test full ReAct cycle"""
        from abstractagent.reasoning import ReActOrchestrator

        # Mock agent
        class MockAgent:
            def think(self, prompt):
                return "I need to calculate 2+2"

            def act(self, thought, tools):
                return {'action': 'tool', 'tool': 'calculator', 'arguments': {'expr': '2+2'}}

            def observe(self, result):
                return "The answer is 4"

        agent = MockAgent()
        orchestrator = ReActOrchestrator(agent)

        # Mock tools
        class MockTools:
            def get_tools(self):
                return [{'name': 'calculator'}]

            def execute(self, name, args):
                return {'output': '4'}

        tools = MockTools()

        # Execute cycle
        result = orchestrator.execute(
            prompt="What is 2+2?",
            tools=tools,
            max_iterations=3
        )

        # Should complete in one iteration
        assert result is not None

    def test_max_iterations_safety(self):
        """Test max iterations prevents infinite loops"""
        from abstractagent.reasoning import ReActOrchestrator

        # Mock agent that never completes
        class LoopingAgent:
            def __init__(self):
                self.iteration = 0

            def think(self, prompt):
                self.iteration += 1
                return f"Thinking iteration {self.iteration}"

            def act(self, thought, tools):
                return {'action': 'tool', 'tool': 'search'}

            def observe(self, result):
                return "Need more information"

        agent = LoopingAgent()
        orchestrator = ReActOrchestrator(agent)

        # Should stop after max iterations
        result = orchestrator.execute(
            prompt="Impossible task",
            max_iterations=3
        )

        assert "3 iterations" in result
        assert agent.iteration == 3
```

### 5. Integration Tests (60 min)

Create `abstractllm/tests/integration/test_three_package_integration.py`:
```python
"""
Test integration between the three packages.
"""

import pytest
from abstractllm import create_llm, BasicSession
from abstractmemory import TemporalMemory
from abstractagent import Agent


class TestPackageIntegration:
    """Test packages work together"""

    def test_agent_uses_llm_and_memory(self):
        """Test agent integrates both packages"""
        # Create agent with both
        agent = Agent(
            llm_config={
                'provider': 'ollama',
                'model': 'llama2'
            },
            memory_config={
                'temporal': True,
                'persist_path': '/tmp/test_memory'
            }
        )

        # Chat should update both session and memory
        response = agent.chat("My name is Alice")

        # Check session has message
        assert len(agent.session.messages) == 2  # user + assistant

        # Check memory has interaction
        facts = agent.memory.semantic_memory.get_facts_about("Alice")
        assert len(facts) > 0 or agent.memory.working.has_item("Alice")

    def test_memory_provides_context(self):
        """Test memory provides context for generation"""
        agent = Agent(
            llm_config={'provider': 'mock'},
            memory_config={'temporal': True}
        )

        # Add to memory
        agent.memory.add_fact("User", "name", "Bob")

        # Chat should use memory context
        response = agent.chat("What's my name?")

        # Mock should receive context with Bob
        # (Would need mock provider to verify)
        assert agent.memory.retrieve_context("name") is not None

    def test_tools_work_across_packages(self):
        """Test tools from AbstractLLM work in Agent"""
        from abstractllm.tools import ToolDefinition

        # Define tool
        def calculator(expression: str) -> float:
            """Calculate math expression"""
            return eval(expression)

        tool = ToolDefinition.from_function(calculator)

        # Create agent with tool
        agent = Agent(
            llm_config={'provider': 'mock'},
            tools=[tool]
        )

        # Tool should be registered
        assert agent.tool_registry.has_tools()
        assert 'calculator' in [t['name'] for t in agent.tool_registry.get_tools()]
```

### 6. Performance Benchmarks (45 min)

Create `abstractllm/tests/performance/test_benchmarks.py`:
```python
"""
Performance benchmarks to ensure refactoring doesn't degrade performance.
"""

import pytest
import time
from abstractllm._refactoring.session.basic import BasicSession


class TestPerformance:
    """Performance benchmarks"""

    @pytest.mark.performance
    def test_session_message_throughput(self):
        """Test message handling performance"""
        session = BasicSession()

        start = time.time()

        # Add 1000 messages
        for i in range(1000):
            session.add_message('user', f'Message {i}')
            session.add_message('assistant', f'Response {i}')

        duration = time.time() - start

        # Should handle 2000 messages in under 1 second
        assert duration < 1.0
        assert len(session.messages) == 2000

        # Calculate throughput
        throughput = 2000 / duration
        print(f"Message throughput: {throughput:.0f} msgs/sec")

    @pytest.mark.performance
    def test_memory_retrieval_speed(self):
        """Test memory retrieval performance"""
        from abstractmemory import TemporalMemory

        memory = TemporalMemory()

        # Add 10,000 facts
        for i in range(10000):
            memory.add_fact(
                subject=f"Entity_{i % 100}",
                predicate="has_property",
                object=f"Value_{i}"
            )

        # Test retrieval speed
        start = time.time()
        results = memory.retrieve_context("Entity_50")
        duration = time.time() - start

        # Should retrieve in under 100ms
        assert duration < 0.1
        print(f"Retrieved {len(results)} facts in {duration*1000:.1f}ms")

    @pytest.mark.performance
    def test_react_cycle_overhead(self):
        """Test ReAct reasoning overhead"""
        from abstractagent import Agent

        agent = Agent(
            llm_config={'provider': 'mock'},
            enable_reasoning=True
        )

        # Time with reasoning
        start = time.time()
        response_with = agent.chat("Test", use_reasoning=True, max_iterations=3)
        time_with = time.time() - start

        # Time without reasoning
        start = time.time()
        response_without = agent.chat("Test", use_reasoning=False)
        time_without = time.time() - start

        # Overhead should be reasonable (< 3x)
        overhead = time_with / time_without
        assert overhead < 3.0
        print(f"ReAct overhead: {overhead:.1f}x")
```

### 7. Backward Compatibility Tests (30 min)

Create `abstractllm/tests/integration/test_compatibility.py`:
```python
"""
Test backward compatibility with existing code.
"""

import pytest


class TestBackwardCompatibility:
    """Ensure existing code still works"""

    def test_session_compat_wrapper(self):
        """Test Session compatibility wrapper"""
        from abstractllm.session_compat import Session

        # Old API should still work
        session = Session(
            provider='ollama',
            enable_memory=True,
            tools=[],
            system_prompt="Test"
        )

        # Old methods should work
        session.add_message('user', 'Hello')
        response = session.generate('Hi')

        assert response is not None

    def test_import_paths(self):
        """Test old import paths still work"""
        # These should not raise ImportError
        from abstractllm import Session  # Via compat
        from abstractllm import create_llm
        from abstractllm.tools import ToolDefinition
        from abstractllm.media import MediaProcessor

    def test_cli_compatibility(self):
        """Test CLI still works"""
        import subprocess

        # Old alma command should work
        result = subprocess.run(
            ['alma', '--help'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert 'ALMA' in result.stdout
```

## Validation Steps

### Run All Tests
```bash
# Run all unit tests
cd /Users/albou/projects/abstractllm
pytest tests/unit -v

cd /Users/albou/projects/abstractmemory
pytest tests/unit -v

cd /Users/albou/projects/abstractagent
pytest tests/unit -v

# Run integration tests
cd /Users/albou/projects/abstractllm
pytest tests/integration -v

# Run performance benchmarks
pytest tests/performance -v -m performance

# Generate coverage report
pytest --cov=abstractllm --cov=abstractmemory --cov=abstractagent \
       --cov-report=html --cov-report=term
```

### Validate Metrics
```bash
# Check line counts
echo "BasicSession lines:"
wc -l abstractllm/_refactoring/session/basic.py

echo "Agent lines:"
wc -l abstractagent/agent.py

# Check no circular dependencies
python -c "
import abstractllm
import abstractmemory
import abstractagent
print('No circular dependency errors')
"

# Check import times
python -c "
import time
start = time.time()
import abstractllm
print(f'AbstractLLM import: {(time.time()-start)*1000:.1f}ms')

start = time.time()
import abstractmemory
print(f'AbstractMemory import: {(time.time()-start)*1000:.1f}ms')

start = time.time()
import abstractagent
print(f'AbstractAgent import: {(time.time()-start)*1000:.1f}ms')
"
```

## Success Criteria

- [ ] All unit tests pass (100+ tests)
- [ ] Integration tests confirm packages work together
- [ ] Performance benchmarks meet targets:
  - Message throughput > 1000 msgs/sec
  - Memory retrieval < 100ms for 10k facts
  - ReAct overhead < 3x
- [ ] Backward compatibility maintained
- [ ] Code coverage > 80%
- [ ] No circular dependencies
- [ ] Import times < 200ms per package

## Issues to Track

1. **Mock Provider Needed**: Need mock LLM provider for testing
2. **Memory Persistence**: Test file vs LanceDB storage
3. **Tool Execution**: Test actual tool execution paths
4. **Streaming**: Test streaming responses
5. **Async Support**: Test async methods

## Next Task

Proceed to Task 06: Migration and Deployment