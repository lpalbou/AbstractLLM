#!/usr/bin/env python3
"""
Debug script to test Ollama's tool calling capabilities directly.
"""

import requests
import json
import sys

def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def main():
    # Ollama API settings
    model = "qwen2.5"  # Try with qwen2.5 or cogito
    base_url = "http://localhost:11434"
    endpoint = f"{base_url}/api/chat"
    
    # Tool definition
    read_file_tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read."
                    }
                },
                "required": ["file_path"]
            }
        }
    }
    
    # Get prompt from command line or use default
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Please read the file README.md"
    
    # STEP 1: Initial request with the user prompt
    print("\n=== STEP 1: INITIAL REQUEST ===")
    
    # Create request data
    initial_request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that can read files when needed. If you need to see a file's contents, use the read_file tool."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "tools": [read_file_tool],
        "options": {
            "temperature": 0.7,
            "num_predict": 2048
        }
    }
    
    # Print request for debugging
    print("\nSending initial request to Ollama:")
    print("-" * 50)
    print(json.dumps(initial_request, indent=2))
    print("-" * 50)
    
    try:
        # Make API call
        print(f"\nCalling Ollama API at: {endpoint}")
        response = requests.post(endpoint, json=initial_request)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Print raw response
        print("\nRaw response from Ollama:")
        print("-" * 50)
        print(json.dumps(data, indent=2))
        print("-" * 50)
        
        # Check for tool calls
        message = data.get("message", {})
        if isinstance(message, dict) and "tool_calls" in message and message["tool_calls"]:
            print("\nTool calls detected!")
            print("-" * 50)
            
            # Extract and execute tool calls
            tool_results = []
            for tc in message["tool_calls"]:
                # Extract from nested function structure
                function_data = tc.get("function", {})
                tool_name = function_data.get("name", "unknown")
                tool_args = function_data.get("arguments", {})
                
                print(f"Tool: {tool_name}")
                print(f"Arguments: {tool_args}")
                
                # Execute the tool if it's read_file
                if tool_name == "read_file" and isinstance(tool_args, dict):
                    file_path = tool_args.get("file_path")
                    if file_path:
                        result = read_file(file_path)
                        print(f"Tool result: {result[:100]}..." if len(result) > 100 else result)
                        
                        # Add to tool results
                        tool_results.append({
                            "tool_call_id": tc.get("id", "call_1"),
                            "role": "tool",
                            "name": tool_name,
                            "content": result
                        })
            
            print("-" * 50)
            
            # STEP 2: Follow-up request with tool results
            print("\n=== STEP 2: FOLLOW-UP REQUEST WITH TOOL RESULTS ===")
            
            # Create follow-up messages with tool results
            follow_up_messages = initial_request["messages"].copy()
            
            # Add assistant's tool call message
            follow_up_messages.append(message)
            
            # Add tool result messages
            for tool_result in tool_results:
                follow_up_messages.append({
                    "role": "tool",
                    "name": tool_result["name"],
                    "content": tool_result["content"],
                    "tool_call_id": tool_result["tool_call_id"]
                })
            
            # Create follow-up request
            follow_up_request = {
                "model": model,
                "messages": follow_up_messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 2048
                }
            }
            
            # Print follow-up request for debugging
            print("\nSending follow-up request to Ollama:")
            print("-" * 50)
            print(json.dumps(follow_up_request, indent=2))
            print("-" * 50)
            
            # Make follow-up API call
            print(f"\nCalling Ollama API at: {endpoint} with tool results")
            follow_up_response = requests.post(endpoint, json=follow_up_request)
            follow_up_response.raise_for_status()
            
            # Parse follow-up response
            follow_up_data = follow_up_response.json()
            
            # Print raw follow-up response
            print("\nRaw follow-up response from Ollama:")
            print("-" * 50)
            print(json.dumps(follow_up_data, indent=2))
            print("-" * 50)
            
            # Extract final answer
            final_message = follow_up_data.get("message", {})
            if isinstance(final_message, dict) and "content" in final_message:
                print("\nFinal answer:")
                print("-" * 50)
                print(final_message["content"])
                print("-" * 50)
            else:
                print("\nNo final answer content found in response.")
        else:
            print("\nNo tool calls detected in the response.")
            
            # Show the content if available
            if isinstance(message, dict) and "content" in message:
                print("\nResponse content:")
                print(message["content"])
    
    except Exception as e:
        print(f"\nError during API request: {str(e)}")
        
if __name__ == "__main__":
    main() 