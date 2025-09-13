#!/usr/bin/env python
"""
Test script for enhanced tool system with unified handling across providers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_llm, create_session
from abstractllm.tools.enhanced_core import (
    EnhancedToolDefinition, ParameterSchema, ToolExample,
    ToolChoice, EnhancedToolCall, ToolExecutionState
)
from abstractllm.tools.common_tools import read_file, list_files
import json
from typing import List, Dict, Any


# Define enhanced tools with rich metadata
def create_enhanced_search_tool():
    """Create an enhanced search tool with examples."""
    return EnhancedToolDefinition(
        name="search_code",
        description="Search for code patterns in the codebase",
        parameters={
            "query": ParameterSchema(
                type="string",
                description="The search query or pattern to find",
                required=True
            ),
            "file_type": ParameterSchema(
                type="string",
                description="File extension to search in (e.g., 'py', 'js')",
                enum=["py", "js", "ts", "java", "go"],
                required=False,
                default="py"
            ),
            "max_results": ParameterSchema(
                type="integer",
                description="Maximum number of results to return",
                minimum=1,
                maximum=100,
                required=False,
                default=10
            )
        },
        examples=[
            ToolExample(
                input_description="Find all class definitions in Python files",
                arguments={"query": "class.*:", "file_type": "py"},
                expected_output="List of files with class definitions"
            ),
            ToolExample(
                input_description="Search for TODO comments",
                arguments={"query": "TODO", "max_results": 20},
                expected_output="Files containing TODO comments"
            )
        ],
        category="code_analysis",
        timeout=10.0
    )


def create_enhanced_analyze_tool():
    """Create an enhanced code analysis tool."""
    return EnhancedToolDefinition(
        name="analyze_code",
        description="Analyze code complexity and quality metrics",
        parameters={
            "file_path": ParameterSchema(
                type="string",
                description="Path to the file to analyze",
                required=True
            ),
            "metrics": ParameterSchema(
                type="array",
                description="Which metrics to compute",
                items=ParameterSchema(
                    type="string",
                    enum=["complexity", "lines", "dependencies", "todos"]
                ),
                required=False,
                default=["complexity", "lines"]
            )
        },
        examples=[
            ToolExample(
                input_description="Analyze complexity of main.py",
                arguments={"file_path": "main.py", "metrics": ["complexity", "lines"]},
                expected_output="Complexity score and line count"
            )
        ],
        category="code_analysis",
        retry_config={"max_attempts": 3, "backoff": "exponential"}
    )


def test_enhanced_tools_with_provider(provider_name: str):
    """Test enhanced tools with a specific provider."""
    print(f"\n{'='*60}")
    print(f"Testing Enhanced Tools with {provider_name}")
    print('='*60)
    
    try:
        # Create provider
        if provider_name == "ollama":
            llm = create_llm(provider_name, model="qwen3:30b-a3b-q4_K_M")
        elif provider_name == "mlx":
            llm = create_llm(provider_name, model="mlx-community/Qwen3-30B-A3B-4bit")
        elif provider_name == "openai":
            llm = create_llm(provider_name, model="gpt-4o-mini")
        elif provider_name == "anthropic":
            llm = create_llm(provider_name, model="claude-3-5-sonnet-20241022")
        else:
            print(f"Unsupported provider: {provider_name}")
            return
        
        # Create enhanced tools
        search_tool = create_enhanced_search_tool()
        analyze_tool = create_enhanced_analyze_tool()
        
        # Create execution state for ReAct
        state = ToolExecutionState(
            conversation_id="test_001",
            max_iterations=5
        )
        
        # Test 1: Tool with examples in prompt
        print("\n1. Testing tool prompt with examples:")
        prompt_with_examples = search_tool.to_prompt_with_examples()
        print(prompt_with_examples)
        
        # Test 2: Tool definition for native APIs
        print("\n2. Testing tool definition conversion:")
        tool_dict = search_tool.to_dict()
        print(json.dumps(tool_dict, indent=2))
        
        # Test 3: Simulated ReAct cycle
        print("\n3. Testing ReAct execution state:")
        
        # Simulate reasoning
        state.update_scratchpad(
            thought="I need to search for class definitions to understand the codebase structure",
            action="search_code",
            observation="Found 15 class definitions across 8 files"
        )
        state.iteration_count += 1
        
        # Extract facts
        state.add_fact("abstractllm", "has_classes", 15)
        state.add_fact("abstractllm", "uses_files", 8)
        
        # Get context
        context = state.get_context()
        print(context)
        
        # Test 4: Tool execution with confidence
        print("\n4. Testing enhanced tool call:")
        tool_call = EnhancedToolCall(
            name="search_code",
            arguments={"query": "def.*test", "file_type": "py"},
            confidence=0.85,
            reasoning="Searching for test functions to understand test coverage"
        )
        print(f"Tool: {tool_call.name}")
        print(f"Confidence: {tool_call.confidence}")
        print(f"Reasoning: {tool_call.reasoning}")
        
        # Test 5: Use with session (if tools are functions)
        print("\n5. Testing with actual session:")
        
        # Create dummy implementation
        def search_code(query: str, file_type: str = "py", max_results: int = 10) -> List[str]:
            """Search for code patterns."""
            # Dummy implementation
            return [f"Found pattern '{query}' in file1.{file_type}", 
                   f"Found pattern '{query}' in file2.{file_type}"]
        
        def analyze_code(file_path: str, metrics: List[str] = None) -> Dict[str, Any]:
            """Analyze code metrics."""
            # Dummy implementation
            return {
                "file": file_path,
                "metrics": metrics or ["lines"],
                "results": {"lines": 100, "complexity": 5}
            }
        
        # Create session with tools
        session = create_session(provider=llm)
        session.add_tool(search_code)
        session.add_tool(analyze_code)
        
        # Test with provider
        response = session.generate(
            "Search for all function definitions in Python files",
            tools=[search_code, analyze_code],
            max_tool_calls=3
        )
        
        print(f"Response: {response}")
        
        print(f"\n✅ Enhanced tools test completed for {provider_name}")
        
    except Exception as e:
        print(f"❌ Error testing {provider_name}: {e}")
        import traceback
        traceback.print_exc()


def test_tool_choice_forcing():
    """Test tool choice forcing capabilities."""
    print("\n" + "="*60)
    print("Testing Tool Choice Forcing")
    print("="*60)
    
    # Different tool choice modes
    modes = [
        (ToolChoice.AUTO, "Let model decide which tools to use"),
        (ToolChoice.NONE, "Disable all tools"),
        (ToolChoice.REQUIRED, "Force model to use at least one tool"),
        (ToolChoice.SPECIFIC, "Force model to use specific tool")
    ]
    
    for mode, description in modes:
        print(f"\n{mode.value}: {description}")
        
        # This would be integrated into the request
        if mode == ToolChoice.SPECIFIC:
            print("  Example: force use of 'search_code' tool")
        
    print("\n✅ Tool choice modes demonstrated")


def test_parameter_validation():
    """Test parameter schema validation."""
    print("\n" + "="*60)
    print("Testing Parameter Schema Validation")
    print("="*60)
    
    # Create a complex parameter schema
    complex_param = ParameterSchema(
        type="object",
        description="Configuration object",
        properties={
            "name": ParameterSchema(
                type="string",
                description="Name of the configuration",
                pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$"
            ),
            "values": ParameterSchema(
                type="array",
                description="List of values",
                items=ParameterSchema(
                    type="integer",
                    minimum=0,
                    maximum=100
                )
            ),
            "enabled": ParameterSchema(
                type="boolean",
                description="Whether config is enabled",
                default=True
            )
        }
    )
    
    # Convert to JSON Schema
    schema = complex_param.to_json_schema()
    print("Generated JSON Schema:")
    print(json.dumps(schema, indent=2))
    
    print("\n✅ Parameter validation demonstrated")


def main():
    """Run all enhanced tool tests."""
    print("Enhanced Tool System Tests")
    print("="*60)
    
    # Test parameter validation
    test_parameter_validation()
    
    # Test tool choice forcing
    test_tool_choice_forcing()
    
    # Test with different providers
    providers_to_test = ["ollama"]  # Start with Ollama
    
    for provider in providers_to_test:
        try:
            test_enhanced_tools_with_provider(provider)
        except Exception as e:
            print(f"Skipping {provider}: {e}")
    
    print("\n" + "="*60)
    print("All enhanced tool tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()