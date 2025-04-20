#!/usr/bin/env python3
"""
Test script for the BasicAgent implementation.

This script tests the BasicAgent with various providers and configurations.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

from basic_agent import BasicAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_tool_agent")

# Test queries
TEST_QUERIES = [
    "Please read the file test_file.txt",
    "Can you read test_file.txt but only show the first 3 lines?",
    "Tell me what's in the file test_file.txt"
]

def test_provider(provider_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Test a specific provider with multiple queries.
    
    Args:
        provider_name: The provider to test
        api_key: Optional API key
        
    Returns:
        Dict with test results
    """
    logger.info(f"Testing provider: {provider_name}")
    results = {
        "provider": provider_name,
        "queries": []
    }
    
    try:
        # Create agent
        agent = BasicAgent(provider_name=provider_name, api_key=api_key, debug=True)
        logger.info(f"Agent created for {provider_name}")
        
        # Test each query
        for query in TEST_QUERIES:
            logger.info(f"Testing query: {query}")
            try:
                response = agent.run(query)
                success = "test_file.txt" in response
                results["queries"].append({
                    "query": query,
                    "success": success,
                    "response": response[:200] + "..." if len(response) > 200 else response
                })
                logger.info(f"Query test {'succeeded' if success else 'failed'}")
            except Exception as e:
                logger.error(f"Error executing query: {e}")
                results["queries"].append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
    except Exception as e:
        logger.error(f"Error testing provider {provider_name}: {e}")
        results["error"] = str(e)
    
    return results

def run_tests(providers: List[str], api_keys: Dict[str, str]) -> Dict[str, Any]:
    """
    Run tests for multiple providers.
    
    Args:
        providers: List of provider names to test
        api_keys: Dict mapping provider names to API keys
        
    Returns:
        Dict with test results for all providers
    """
    all_results = {}
    
    for provider in providers:
        api_key = api_keys.get(provider)
        results = test_provider(provider, api_key)
        all_results[provider] = results
        
        # Print summary
        print(f"\n==== Results for {provider} ====")
        if "error" in results:
            print(f"ERROR: {results['error']}")
        else:
            success_count = sum(q["success"] for q in results["queries"])
            print(f"Queries: {success_count}/{len(results['queries'])} successful")
            
            for query_result in results["queries"]:
                status = "✅ SUCCESS" if query_result["success"] else "❌ FAILURE"
                print(f"\n{status} - Query: {query_result['query']}")
                if "error" in query_result:
                    print(f"Error: {query_result['error']}")
                else:
                    print(f"Response: {query_result['response']}")
        
        print("\n" + "="*50)
    
    return all_results

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test BasicAgent with different providers")
    parser.add_argument("--providers", nargs="+", default=["anthropic", "openai"],
                        help="List of providers to test")
    parser.add_argument("--openai-key", help="OpenAI API key (optional, will use environment variable if not provided)")
    parser.add_argument("--anthropic-key", help="Anthropic API key (optional, will use environment variable if not provided)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Collect API keys
    api_keys = {}
    if args.openai_key:
        api_keys["openai"] = args.openai_key
    elif os.environ.get("OPENAI_API_KEY"):
        api_keys["openai"] = os.environ.get("OPENAI_API_KEY")
        
    if args.anthropic_key:
        api_keys["anthropic"] = args.anthropic_key
    elif os.environ.get("ANTHROPIC_API_KEY"):
        api_keys["anthropic"] = os.environ.get("ANTHROPIC_API_KEY")
    
    # Check if we have keys for the requested providers
    missing_keys = [p for p in args.providers if p not in api_keys and p != "ollama"]
    if missing_keys:
        print(f"Warning: Missing API keys for providers: {', '.join(missing_keys)}")
        print("Tests for these providers will likely fail unless keys are in environment variables.")
    
    # Run tests
    results = run_tests(args.providers, api_keys)
    
    # Determine exit code based on results
    failed = any("error" in results[p] or any(not q["success"] for q in results[p]["queries"]) 
                 for p in results if "queries" in results[p])
    
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main()) 