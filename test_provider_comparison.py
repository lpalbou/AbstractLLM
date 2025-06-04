#!/usr/bin/env python3
"""
Comprehensive test comparing MLX vs Ollama providers using the same Qwen model.
This test ensures AbstractLLM provides truly unified behavior across providers.

Models:
- MLX: mlx-community/Qwen3-30B-A3B-4bit  
- Ollama: qwen3:30b-a3b-q4_K_M

Both are the same underlying model, so behavior should be IDENTICAL.
"""

import json
import time
import logging
from typing import Dict, Any, List
from abstractllm import create_llm
from abstractllm.session import Session

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Simple safety check
        allowed_chars = set('0123456789+-*/().,. ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def get_current_time() -> str:
    """Get the current time."""
    import datetime
    return f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

class ProviderComparison:
    """Compare tool calling behavior between MLX and Ollama providers."""
    
    def __init__(self):
        self.tools = [read_file, calculate_math, get_current_time]
        self.system_prompt = ("You are a helpful assistant that can use tools. "
                            "When asked to perform tasks, use the appropriate tools and provide clear, concise responses.")
        
        self.test_cases = [
            {
                "name": "Simple Tool Call",
                "prompt": "What is 25 * 4 + 7?",
                "expected_tool": "calculate_math"
            },
            {
                "name": "File Reading",
                "prompt": "Read the file README.md and tell me what it's about in one sentence.",
                "expected_tool": "read_file"
            },
            {
                "name": "Current Time",
                "prompt": "What time is it right now?",
                "expected_tool": "get_current_time"
            },
            {
                "name": "Multi-step Task", 
                "prompt": "Read the file pyproject.toml, then calculate 15 * 8, then tell me the current time.",
                "expected_tools": ["read_file", "calculate_math", "get_current_time"]
            },
            {
                "name": "No Tool Needed",
                "prompt": "Just say hello and introduce yourself.",
                "expected_tool": None
            }
        ]
        
    def create_provider(self, provider_type: str) -> Any:
        """Create a provider instance."""
        if provider_type == "mlx":
            return create_llm("mlx", model="mlx-community/Qwen3-30B-A3B-4bit")
        elif provider_type == "ollama":
            return create_llm("ollama", model="qwen3:30b-a3b-q4_K_M")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    def test_provider(self, provider_type: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single provider with a test case."""
        print(f"\n  Testing {provider_type.upper()} with: '{test_case['prompt']}'")
        
        start_time = time.time()
        
        try:
            # Create provider and session
            provider = self.create_provider(provider_type)
            session = Session(
                system_prompt=self.system_prompt,
                provider=provider,
                tools=self.tools
            )
            
            # Generate response
            response = session.generate(
                prompt=test_case["prompt"],
                max_tool_calls=5  # Allow multiple tool calls
            )
            
            end_time = time.time()
            
            # Analyze response
            result = {
                "provider": provider_type,
                "success": True,
                "response_time": end_time - start_time,
                "response_type": type(response).__name__,
                "has_content": hasattr(response, 'content'),
                "content_preview": None,
                "tool_calls_made": [],
                "error": None
            }
            
            # Extract content
            if hasattr(response, 'content'):
                content = response.content
                result["content_preview"] = content[:200] + "..." if len(content) > 200 else content
            else:
                content = str(response)
                result["content_preview"] = content[:200] + "..." if len(content) > 200 else content
            
            # Check session messages for tool calls
            tool_calls = []
            for message in session.messages:
                if hasattr(message, 'tool_results') and message.tool_results:
                    for tool_result in message.tool_results:
                        if 'name' in tool_result:
                            tool_calls.append(tool_result['name'])
            
            result["tool_calls_made"] = tool_calls
            
            return result
            
        except Exception as e:
            return {
                "provider": provider_type,
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e),
                "tool_calls_made": [],
                "content_preview": None
            }
    
    def compare_responses(self, mlx_result: Dict, ollama_result: Dict, test_case: Dict) -> Dict[str, Any]:
        """Compare responses between MLX and Ollama."""
        comparison = {
            "test_case": test_case["name"],
            "both_successful": mlx_result["success"] and ollama_result["success"],
            "response_time_diff": abs(mlx_result["response_time"] - ollama_result["response_time"]),
            "tool_calls_match": mlx_result["tool_calls_made"] == ollama_result["tool_calls_made"],
            "content_length_diff": 0,
            "issues": []
        }
        
        # Compare content lengths
        if mlx_result["content_preview"] and ollama_result["content_preview"]:
            comparison["content_length_diff"] = abs(
                len(mlx_result["content_preview"]) - len(ollama_result["content_preview"])
            )
        
        # Check for issues
        if not comparison["both_successful"]:
            if not mlx_result["success"]:
                comparison["issues"].append(f"MLX failed: {mlx_result['error']}")
            if not ollama_result["success"]:
                comparison["issues"].append(f"Ollama failed: {ollama_result['error']}")
        
        if not comparison["tool_calls_match"]:
            comparison["issues"].append(
                f"Tool calls differ: MLX={mlx_result['tool_calls_made']}, "
                f"Ollama={ollama_result['tool_calls_made']}"
            )
        
        # Check expected tool usage
        expected_tool = test_case.get("expected_tool")
        expected_tools = test_case.get("expected_tools", [])
        
        if expected_tool:
            if expected_tool not in mlx_result["tool_calls_made"]:
                comparison["issues"].append(f"MLX didn't use expected tool: {expected_tool}")
            if expected_tool not in ollama_result["tool_calls_made"]:
                comparison["issues"].append(f"Ollama didn't use expected tool: {expected_tool}")
        
        if expected_tools:
            for tool in expected_tools:
                if tool not in mlx_result["tool_calls_made"]:
                    comparison["issues"].append(f"MLX didn't use expected tool: {tool}")
                if tool not in ollama_result["tool_calls_made"]:
                    comparison["issues"].append(f"Ollama didn't use expected tool: {tool}")
        
        if expected_tool is None and (mlx_result["tool_calls_made"] or ollama_result["tool_calls_made"]):
            comparison["issues"].append("Unexpected tool usage when none was expected")
        
        return comparison
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run full comparison between MLX and Ollama providers."""
        print("🔄 Starting Provider Comparison: MLX vs Ollama")
        print("=" * 80)
        
        results = {
            "mlx_results": [],
            "ollama_results": [],
            "comparisons": [],
            "summary": {
                "total_tests": len(self.test_cases),
                "successful_tests": 0,
                "identical_behavior": 0,
                "tool_calling_issues": 0,
                "performance_issues": 0
            }
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📝 Test {i}/{len(self.test_cases)}: {test_case['name']}")
            print("-" * 50)
            
            # Test MLX
            mlx_result = self.test_provider("mlx", test_case)
            results["mlx_results"].append(mlx_result)
            
            # Test Ollama
            ollama_result = self.test_provider("ollama", test_case)
            results["ollama_results"].append(ollama_result)
            
            # Compare results
            comparison = self.compare_responses(mlx_result, ollama_result, test_case)
            results["comparisons"].append(comparison)
            
            # Print results
            self.print_test_results(mlx_result, ollama_result, comparison)
            
            # Update summary
            if comparison["both_successful"]:
                results["summary"]["successful_tests"] += 1
            
            if comparison["tool_calls_match"] and not comparison["issues"]:
                results["summary"]["identical_behavior"] += 1
            
            if not comparison["tool_calls_match"]:
                results["summary"]["tool_calling_issues"] += 1
            
            if comparison["response_time_diff"] > 10:  # More than 10 seconds difference
                results["summary"]["performance_issues"] += 1
        
        return results
    
    def print_test_results(self, mlx_result: Dict, ollama_result: Dict, comparison: Dict):
        """Print results for a single test."""
        print(f"    MLX:    {'✅' if mlx_result['success'] else '❌'} "
              f"({mlx_result['response_time']:.1f}s) "
              f"Tools: {mlx_result['tool_calls_made']}")
        
        print(f"    Ollama: {'✅' if ollama_result['success'] else '❌'} "
              f"({ollama_result['response_time']:.1f}s) "
              f"Tools: {ollama_result['tool_calls_made']}")
        
        if comparison["issues"]:
            print(f"    ⚠️  Issues: {', '.join(comparison['issues'])}")
        else:
            print(f"    ✅ Behavior identical")
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print final comparison summary."""
        summary = results["summary"]
        
        print("\n" + "=" * 80)
        print("📊 FINAL COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"📈 Test Results:")
        print(f"   • Total tests:           {summary['total_tests']}")
        print(f"   • Successful tests:      {summary['successful_tests']}/{summary['total_tests']}")
        print(f"   • Identical behavior:    {summary['identical_behavior']}/{summary['total_tests']}")
        print(f"   • Tool calling issues:   {summary['tool_calling_issues']}")
        print(f"   • Performance issues:    {summary['performance_issues']}")
        
        # Calculate scores
        success_rate = summary['successful_tests'] / summary['total_tests'] * 100
        unification_score = summary['identical_behavior'] / summary['total_tests'] * 100
        
        print(f"\n🎯 Scores:")
        print(f"   • Success Rate:      {success_rate:.1f}%")
        print(f"   • Unification Score: {unification_score:.1f}%")
        
        # Verdict
        if unification_score >= 90:
            print(f"\n✅ VERDICT: AbstractLLM provides excellent unified behavior!")
        elif unification_score >= 70:
            print(f"\n⚠️  VERDICT: AbstractLLM provides good but improvable unified behavior.")
        else:
            print(f"\n❌ VERDICT: AbstractLLM fails to provide unified behavior. Major issues found.")
        
        # Show specific issues
        print(f"\n🔍 Issues Found:")
        for comparison in results["comparisons"]:
            if comparison["issues"]:
                print(f"   • {comparison['test_case']}: {', '.join(comparison['issues'])}")
        
        return unification_score >= 80  # Return True if behavior is sufficiently unified

def main():
    """Run the provider comparison test."""
    try:
        comparison = ProviderComparison()
        results = comparison.run_comparison()
        success = comparison.print_final_summary(results)
        
        # Save detailed results
        with open("provider_comparison_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: provider_comparison_results.json")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 