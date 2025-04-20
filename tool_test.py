#!/usr/bin/env python3
"""
Simple test script for AbstractLLM tool support.

This script demonstrates how to use the AbstractLLM library with the Anthropic provider
to perform a file reading operation using tools.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List

from abstractllm.factory import create_llm
from abstractllm.tools import function_to_tool_definition
from abstractllm.types import GenerateResponse

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tool_test")

def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read and return the contents of a file.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (optional)
        
    Returns:
        The contents of the file as a string
    """
    logger.info(f"Reading file: {file_path}, max_lines: {max_lines}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if max_lines is not None:
                lines = []
                for i, line in enumerate(file):
                    if i >= max_lines:
                        break
                    lines.append(line)
                content = ''.join(lines)
                if len(lines) == max_lines and file.readline():  # Check if there are more lines
                    content += f"\n... (file truncated, showed {max_lines} lines)"
            else:
                content = file.read()
        logger.info(f"Successfully read file: {file_path}, content length: {len(content)}")
        return content
    except FileNotFoundError:
        error_msg = f"Error: File not found at path '{file_path}'"
        logger.error(error_msg)
        return error_msg
    except PermissionError:
        error_msg = f"Error: Permission denied when trying to read '{file_path}'"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(error_msg)
        return error_msg

def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """
    Execute a tool call based on the provided dictionary.
    
    Args:
        tool_call: Dictionary containing tool call information
        
    Returns:
        Result of the tool execution
    """
    logger.info(f"Executing tool call: {tool_call}")
    
    # Extract tool name and arguments
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    if tool_name == "read_file":
        # Execute the read_file function with the provided arguments
        return read_file(**arguments)
    else:
        return f"Error: Unknown tool '{tool_name}'"

def main():
    """Main function to demonstrate tool usage."""
    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return 1
    
    # Create the provider
    logger.info("Creating Anthropic provider")
    provider = create_llm(
        "anthropic",
        api_key=api_key,
        model="claude-3-5-sonnet-20241022"
    )
    
    # Create tool definition
    logger.info("Creating tool definition")
    file_reader_tool = function_to_tool_definition(read_file)
    logger.debug(f"Tool definition: {json.dumps(file_reader_tool.to_dict(), indent=2)}")
    
    # User query
    query = "Please read the file test_file.txt"
    
    try:
        # Generate a response with tools
        logger.info(f"Generating response for query: {query}")
        response = provider.generate(
            prompt=query,
            system_prompt="You are a helpful assistant with access to a read_file tool. When asked to read a file, you MUST use the read_file tool and MUST NOT make up file contents.",
            tools=[file_reader_tool],
            temperature=0.7,
            max_tokens=1024
        )
        
        logger.debug(f"Response type: {type(response)}")
        
        if isinstance(response, str):
            # If response is a string, print it directly
            print(f"Response: {response}")
        elif hasattr(response, 'content') and response.content:
            # Extract tool calls from response text if needed
            content = response.content
            print(f"Response Content: {content}")
            
            # Look for tool call in the response text (JSON block)
            import re
            tool_call_match = re.search(r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}', content)
            
            if tool_call_match:
                tool_name = tool_call_match.group(1)
                tool_args_str = tool_call_match.group(2)
                
                try:
                    # Parse the arguments
                    tool_args = json.loads(tool_args_str)
                    
                    # Execute the tool
                    tool_call = {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                    result = execute_tool_call(tool_call)
                    
                    # Print the actual file content
                    print("\nACTUAL FILE CONTENT:")
                    print(result)
                    
                except json.JSONDecodeError:
                    print(f"Error: Failed to parse tool arguments: {tool_args_str}")
            else:
                print("No tool call found in the response.")
                
            # Check if the response has tool_calls attribute (proper API tool call)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info("Response contains proper tool calls")
                
                for tool_call in response.tool_calls:
                    print(f"\nExecuting Tool Call: {tool_call}")
                    # Execute the tool
                    result = execute_tool_call({
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    })
                    print(f"Tool Result: {result}")
        else:
            # For other response types
            print(f"Response: {response}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 