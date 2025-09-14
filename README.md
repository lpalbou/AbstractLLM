# AbstractLLM

[![PyPI version](https://badge.fury.io/py/abstractllm.svg)](https://badge.fury.io/py/abstractllm)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified interface for Large Language Models with memory, reasoning, and tool capabilities.

Version: 1.0.1

## Overview

AbstractLLM provides a consistent interface for multiple LLM providers while offering agentic capabilities including hierarchical memory systems, ReAct reasoning cycles, and universal tool support. The framework focuses on practical AI agent development.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agent Development](#agent-development)
- [Enhanced Tools System](#enhanced-tools-system)
- [Memory & Reasoning](#memory--reasoning)
- [Provider Support](#provider-support)
- [Command-Line Examples](#command-line-examples)
- [Contributing](#contributing)
- [License](#license)

## Key Features

### Core Infrastructure
- 🔄 **Universal Provider Support**: OpenAI, Anthropic, Ollama, HuggingFace, and MLX with consistent API
- 🔌 **Provider Agnostic**: Switch between providers with minimal code changes
- 🛠️ **Enhanced Tool System**: Tool creation with Pydantic validation and retry logic (alpha phase)
- 📊 **Model Capability Detection**: Automatic detection of tool support, vision capabilities, and context limits

### Agentic Capabilities (Alpha Testing)
- 🧠 **Hierarchical Memory**: Working, episodic, and semantic memory with cross-session persistence (alpha)
- 🔄 **ReAct Reasoning**: Complete reasoning cycles with scratchpad traces and fact extraction (alpha)
- 🌐 **Knowledge Graphs**: Automatic fact extraction and relationship mapping (alpha)
- 🎯 **Context-Aware Retrieval**: Memory-enhanced LLM prompting with relevant context injection (alpha)
- 📝 **Session Management**: Persistent conversations with memory consolidation

### Production Features
- 🖼️ **Vision Support**: Multimodal capabilities across compatible providers
- 📝 **Structured Output**: JSON/YAML response formatting with validation
- 🔤 **Type Safety**: Full type hints and enum-based parameters
- 🛑 **Unified Error Handling**: Consistent error handling with retry strategies
- 🍎 **Apple Silicon Optimization**: Native MLX support for M1/M2/M3 devices

## Installation

```bash
# Core installation with basic features
pip install abstractllm

# Provider-specific installations
pip install "abstractllm[openai]"       # OpenAI API support
pip install "abstractllm[anthropic]"    # Anthropic/Claude API support
pip install "abstractllm[ollama]"       # Ollama local models
pip install "abstractllm[huggingface]"  # HuggingFace models
pip install "abstractllm[mlx]"          # Apple Silicon MLX support
pip install "abstractllm[tools]"        # Enhanced tool system

# Comprehensive installation (recommended)
pip install "abstractllm[all]"          # All providers (MLX will install on Apple Silicon only)
```

**Note**: The `[all]` extra includes MLX dependencies which are Apple Silicon specific. On non-Apple platforms, MLX dependencies will be installed but MLX functionality will not be available.

## Quick Start

### Basic LLM Usage

```python
from abstractllm import create_llm

# Create an LLM instance
llm = create_llm("openai", model="gpt-4o-mini")
response = llm.generate("Explain quantum computing briefly.")
print(response)

# Switch providers seamlessly
anthropic_llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")
response = anthropic_llm.generate("Tell me about yourself.")
print(response)
```

### Third-Party Integration

AbstractLLM is designed for easy integration into existing projects:

```python
from abstractllm import create_llm
from abstractllm.session import Session

class MyAIAssistant:
    def __init__(self, provider="openai", model="gpt-4o-mini"):
        self.llm = create_llm(provider, model=model)
        self.session = Session(provider=self.llm, enable_memory=True)  # Alpha feature
    
    def ask(self, question: str) -> str:
        """Ask the assistant a question with memory (alpha)."""
        response = self.session.generate(question)
        return response.content
    
    def ask_with_tools(self, question: str, tools: list) -> str:
        """Ask with tool support."""
        response = self.session.generate_with_tools(question, tools=tools)
        return response.content

# Usage in your application
assistant = MyAIAssistant(provider="anthropic")
answer = assistant.ask("What did we discuss earlier?")
```

## Agent Development

### ALMA-Simple: Intelligent Agent Example

AbstractLLM includes `alma-simple.py`, a complete example of an agent with memory, reasoning, and tool capabilities:

```bash
# Interactive agent with memory and tools
python alma-simple.py

# Single query with provider switching
python alma-simple.py --provider openai --model gpt-4o-mini \
    --prompt "list the files in the current directory"

# Use enhanced models that work well
python alma-simple.py --provider ollama --model qwen3-coder:30b \
    --prompt "read README.md and summarize it"
```

**Note**: Our testing shows that `qwen3-coder:30b` works particularly well for coding tasks and tool usage.

### Key Agent Features Demonstrated

```python
from abstractllm.factory import create_session
from abstractllm.tools.common_tools import read_file, list_files, search_files

# Create agent session
session = create_session(
    "anthropic",
    model="claude-3-5-haiku-20241022", 
    enable_memory=True,            # Hierarchical memory (alpha)
    enable_retry=True,             # Retry strategies
    tools=[read_file, list_files], # Tool capabilities
    max_tool_calls=25,             # Prevent infinite loops
    system_prompt="You are a helpful assistant with memory and tools."
)

# Agent can reason, remember, and use tools
response = session.generate(
    prompt="Read the project files and remember the key concepts",
    use_memory_context=True,     # Use relevant memories (alpha)
    create_react_cycle=True,     # Create reasoning trace (alpha)
)
```

## Enhanced Tools System

AbstractLLM features an enhanced tool system with validation capabilities:

### Basic Tool Creation

```python
from abstractllm.tools import tool
from pydantic import Field

@tool(retry_on_error=True, timeout=30.0)
def search_web(
    query: str = Field(description="Search query", min_length=1),
    max_results: int = Field(default=10, ge=1, le=100)
) -> list[str]:
    """Search the web for information.
    
    Args:
        query: The search query to execute
        max_results: Maximum number of results
    """
    # Implementation
    return [f"Result for: {query}"]
```

### Advanced Tool Features

```python
from abstractllm.tools import tool, ToolContext
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    url: str
    relevance: float = Field(ge=0.0, le=1.0)

@tool(
    parse_docstring=True,           # Extract parameter descriptions
    retry_on_error=True,            # Retry on validation errors
    max_retries=3,                  # Maximum retry attempts
    timeout=30.0,                   # Execution timeout
    tags=["search", "web"],         # Categorization
    when_to_use="When user needs current web information",
    requires_context=True,          # Inject session context
    response_model=SearchResult     # Validate response
)
def enhanced_search(
    query: str = Field(min_length=1, max_length=500),
    context: ToolContext = None    # Auto-injected
) -> list[SearchResult]:
    """Enhanced web search with validation."""
    # Access session memory through context
    if context.memory:
        relevant_facts = context.memory.search(query)
    
    return [SearchResult(title="Example", url="http://example.com", relevance=0.9)]
```

### Tool System Features

- **Pydantic Validation**: Automatic input/output validation with LLM-friendly error messages
- **Retry Logic**: Intelligent retry on validation errors
- **Docstring Parsing**: Extract parameter descriptions from Google/NumPy/Sphinx docstrings
- **Context Injection**: Access session memory and metadata in tools
- **Timeout Support**: Prevent hanging tool executions
- **Deprecation Warnings**: Mark tools as deprecated with migration messages
- **Universal Compatibility**: Works across all providers (native and prompted)

## Memory & Reasoning (Alpha Testing)

### Hierarchical Memory System

AbstractLLM implements a hierarchical memory architecture (alpha testing):

```python
from abstractllm.memory import HierarchicalMemory
from abstractllm.factory import create_session

# Create session with memory (alpha)
session = create_session(
    "ollama",
    model="qwen3:4b",
    enable_memory=True,              # Alpha feature
    memory_config={
        'working_memory_size': 10,     # Recent context items
        'consolidation_threshold': 5,   # When to consolidate to long-term
        'cross_session_persistence': True  # Remember across sessions
    }
)

# Memory automatically:
# - Extracts facts from conversations
# - Creates knowledge graphs with relationships  
# - Consolidates important information
# - Provides relevant context for new queries
```

### Memory Components

1. **Working Memory**: Recent interactions and context
2. **Episodic Memory**: Consolidated experiences and events
3. **Semantic Memory**: Extracted facts and knowledge graph
4. **ReAct Cycles**: Complete reasoning traces with scratchpads
5. **Bidirectional Links**: Relationships between all memory components

### Example Memory Usage

```python
# Query with memory context (alpha)
response = session.generate(
    "What did I tell you about my project?",
    use_memory_context=True  # Inject relevant memories (alpha)
)

# Create reasoning cycle (alpha)
response = session.generate(
    "Analyze the project structure and make recommendations",
    create_react_cycle=True  # Full ReAct reasoning with scratchpad (alpha)
)

# Access memory directly
if session.memory:
    stats = session.memory.get_statistics()
    print(f"Facts learned: {stats['knowledge_graph']['total_facts']}")
    print(f"ReAct cycles: {stats['total_react_cycles']}")
```

## Provider Support

### OpenAI - Manual Provider Improvements
```python
# Supported through manual provider enhancements
llm = create_llm("openai", model="gpt-4o-mini") # Vision + tools
llm = create_llm("openai", model="gpt-4o")      # Latest supported model
llm = create_llm("openai", model="gpt-4-turbo")  # Multimodal support

# Enhanced parameters through manual provider improvements
llm = create_llm("openai", 
                 model="gpt-4o",
                 seed=42,                    # Reproducible outputs
                 frequency_penalty=1.0,      # Reduce repetition  
                 presence_penalty=0.5)       # Encourage new topics
```

### Anthropic - Claude Models
```python
llm = create_llm("anthropic", model="claude-3-5-sonnet-20241022")
llm = create_llm("anthropic", model="claude-3-5-haiku-20241022")  # Fast and efficient
```

### Local Models - Ollama & MLX
```python
# Ollama for various open-source models
llm = create_llm("ollama", model="qwen3:4b")         # Good balance
llm = create_llm("ollama", model="qwen3-coder:30b")  # Excellent for coding

# MLX for Apple Silicon (M1/M2/M3)
llm = create_llm("mlx", model="mlx-community/GLM-4.5-Air-4bit")
llm = create_llm("mlx", model="Qwen/Qwen3-4B-MLX-4bit")
```

### HuggingFace - Open Source Models
```python
llm = create_llm("huggingface", model="Qwen/Qwen3-4B")
llm = create_llm("huggingface", model="microsoft/Phi-4-mini-instruct")
```

## Command-Line Examples

### ALMA-Simple Agent Examples

```bash
# Basic usage with different providers
python alma-simple.py --provider anthropic --model claude-3-5-haiku-20241022 \
    --prompt "list the files in the current directory"

python alma-simple.py --provider openai --model gpt-4o-mini \
    --prompt "read README.md and summarize the key features"

python alma-simple.py --provider ollama --model qwen3-coder:30b \
    --prompt "analyze the project structure"

# Advanced usage with memory persistence
python alma-simple.py --memory agent_memory.pkl \
    --prompt "Remember that I'm working on an AI project"

# Interactive mode with verbose logging
python alma-simple.py --verbose

# Control tool usage iterations
python alma-simple.py --max-tool-calls 10 \
    --prompt "carefully examine each file in the project"
```

### Verified Working Configurations

These configurations have been tested and work reliably:

```bash
# OpenAI - Supported models through manual provider improvements
python alma-simple.py --provider openai --model gpt-4o-mini \
    --prompt "list files" --max-tool-calls 3

python alma-simple.py --provider openai --model gpt-4o \
    --prompt "list files" --max-tool-calls 3

# Anthropic - Reliable and fast
python alma-simple.py --provider anthropic --model claude-3-5-haiku-20241022 \
    --prompt "list files" --max-tool-calls 3

# Ollama - Excellent open-source option
python alma-simple.py --provider ollama --model qwen3:4b \
    --prompt "read README.md and summarize it"

# HuggingFace - Direct model usage
python alma-simple.py --provider huggingface --model Qwen/Qwen3-4B \
    --prompt "list the files"

# MLX - Apple Silicon optimized
python alma-simple.py --provider mlx --model mlx-community/GLM-4.5-Air-4bit \
    --prompt "list files"
```

**Note**: `qwen3-coder:30b` via Ollama works well for coding tasks and reasoning.

## Key Improvements in Recent Versions

### OpenAI Provider Improvements
- **Manual Provider Enhancements**: Improved OpenAI provider through custom implementation
- **Enhanced Parameters**: Support for seed, frequency_penalty, presence_penalty
- **Better Error Handling**: Improved API error management and retry logic

### Memory & Reasoning Enhancements  
- **Hierarchical Memory**: Implementation of hierarchical memory management
- **Cross-Session Persistence**: Knowledge preserved across different sessions
- **ReAct Reasoning**: Complete reasoning cycles with scratchpad traces
- **Knowledge Graphs**: Automatic fact extraction and relationship mapping
- **Context-Aware Retrieval**: Memory-enhanced prompting for better responses

### Universal Tool System
- **Enhanced @tool Decorator**: Pydantic validation, retry logic, rich metadata
- **Provider Agnostic**: Works with all providers (native tools or prompted)
- **Context Injection**: Tools can access session memory and metadata
- **Backward Compatible**: Existing @register decorator still supported
- **Production Ready**: Timeouts, confirmations, deprecation warnings

### Architecture Improvements
- **Unified Session System**: Single session class with all capabilities
- **Provider Detection**: Automatic capability detection and optimization
- **Memory Consolidation**: Integration of memory features
- **Error Recovery**: Intelligent fallback and retry strategies

## Integration Examples

### Simple Integration
```python
from abstractllm import create_llm

# Drop-in replacement for OpenAI client
def my_ai_function(prompt: str) -> str:
    llm = create_llm("openai", model="gpt-4o-mini")
    return llm.generate(prompt).content

# With provider flexibility  
def flexible_ai(prompt: str, provider: str = "anthropic") -> str:
    llm = create_llm(provider)
    return llm.generate(prompt).content
```

### Advanced Agent Integration
```python
from abstractllm.factory import create_session
from abstractllm.tools import tool

@tool
def get_user_data(user_id: str) -> dict:
    """Fetch user data from your database."""
    return {"name": "Alice", "preferences": ["AI", "coding"]}

class CustomerServiceAgent:
    def __init__(self):
        self.session = create_session(
            "anthropic", 
            model="claude-3-5-sonnet-20241022",
            enable_memory=True,         # Alpha feature
            tools=[get_user_data],
            system_prompt="You are a helpful customer service agent."
        )
    
    def handle_request(self, user_id: str, message: str) -> str:
        prompt = f"User {user_id} says: {message}"
        response = self.session.generate(
            prompt, 
            use_memory_context=True,    # Remember previous interactions (alpha)
            create_react_cycle=True     # Detailed reasoning (alpha)
        )
        return response.content
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**AbstractLLM** - Unified LLM interface with agentic capabilities.