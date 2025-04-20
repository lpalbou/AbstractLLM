#!/usr/bin/env python3
"""
Basic agent for AbstractLLM with tool support.

This script demonstrates how to create a simple agent that uses the AbstractLLM
framework with a file reading tool, compatible with multiple providers.
"""

import os
import sys
import re
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Callable, Union, Generator
from pathlib import Path

from abstractllm import create_llm
from abstractllm.types import Message, GenerateResponse
from abstractllm.enums import ModelParameter, ModelCapability, MessageRole
from abstractllm.session import Session, SessionManager
from abstractllm.tools import function_to_tool_definition, ToolDefinition, ToolCall, ToolCallRequest

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("basic_agent")

# Add specialized logging functions for each step in the tool call sequence
def log_step(step_number, step_name, message):
    """Log a step in the agent-LLM interaction with a clear step number and name."""
    logger.info(f"STEP {step_number}: {step_name} - {message}")

def log_llm_request(prompt, tools):
    """Log the request being sent to the LLM."""
    tool_names = [t.name if hasattr(t, 'name') else t['name'] if isinstance(t, dict) else 'unknown' for t in tools]
    logger.info(f"STEP 1: AGENT→LLM - Sending prompt with {len(tools)} tools: {', '.join(tool_names)}")
    logger.debug(f"Prompt: {prompt}")
    logger.debug(f"Tools: {tools}")

def log_tool_call_request(tool_calls):
    """Log when the LLM requests a tool call."""
    if not tool_calls:
        logger.info("STEP 2: LLM→AGENT - No tool calls requested by LLM")
        return False
    
    for i, tc in enumerate(tool_calls):
        name = tc.name if hasattr(tc, 'name') else 'unknown'
        args = tc.arguments if hasattr(tc, 'arguments') else {}
        logger.info(f"STEP 2: LLM→AGENT - LLM requested tool call: {name} with args: {args}")
    
    return True

def log_tool_execution(tool_name, args, result):
    """Log the execution of a tool and its result."""
    logger.info(f"STEP 3: AGENT EXECUTION - Executing tool: {tool_name} with args: {args}")
    logger.info(f"STEP 4: TOOL→AGENT - Tool execution completed with result length: {len(str(result)) if result else 0}")
    logger.debug(f"Tool result: {result[:200]}..." if result and len(str(result)) > 200 else f"Tool result: {result}")

def log_tool_results_to_llm():
    """Log sending tool results back to the LLM."""
    logger.info(f"STEP 5: AGENT→LLM - Sending tool results to LLM for completion")

def log_final_response(response):
    """Log the final response from the LLM."""
    logger.info(f"STEP 6: LLM→AGENT - Received final response from LLM")
    logger.debug(f"Final response: {response[:200]}..." if response and len(str(response)) > 200 else f"Final response: {response}")

def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read and return the contents of a file.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (optional)
        
    Returns:
        The contents of the file as a string
    """
    logger.debug(f"Tool call: read_file(file_path={file_path}, max_lines={max_lines})")
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
        log_tool_execution("read_file", {"file_path": file_path, "max_lines": max_lines}, content)
        return content
    except FileNotFoundError:
        error_msg = f"Error: File not found at path '{file_path}'"
        logger.error(error_msg)
        log_tool_execution("read_file", {"file_path": file_path, "max_lines": max_lines}, error_msg)
        return error_msg
    except PermissionError:
        error_msg = f"Error: Permission denied when trying to read '{file_path}'"
        logger.error(error_msg)
        log_tool_execution("read_file", {"file_path": file_path, "max_lines": max_lines}, error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(error_msg)
        log_tool_execution("read_file", {"file_path": file_path, "max_lines": max_lines}, error_msg)
        return error_msg


class BasicAgent:
    """
    A basic agent that uses AbstractLLM with tool support.
    """
    
    def __init__(self, provider_name: str = "anthropic", model_name: Optional[str] = None, 
                 api_key: Optional[str] = None, debug: bool = False):
        """
        Initialize the agent with the specified provider and model.
        
        Args:
            provider_name: Name of the provider to use (default: "anthropic")
            model_name: Specific model to use (optional, defaults to provider's default)
            api_key: Provider API key (optional, will use environment variable if not provided)
            debug: Whether to enable debug mode
        """
        logger.info(f"Initializing BasicAgent with provider: {provider_name}")
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        
        # Get provider-specific configuration
        provider_config = self._get_provider_config(provider_name, model_name, api_key)
        
        # Create the provider
        logger.info(f"Creating provider: {provider_name}")
        self.llm = create_llm(provider_name, **provider_config)
        
        # Define the file reading tool
        logger.debug("Creating file reader tool definition")
        file_reader_tool = function_to_tool_definition(read_file)
        logger.debug(f"Tool definition: {json.dumps(file_reader_tool.to_dict(), indent=2)}")
        
        # Verify that the provider supports tool calls
        logger.info("Checking provider capabilities")
        capabilities = self.llm.get_capabilities()
        logger.debug(f"Provider capabilities: {capabilities}")
        
        # Get current model for detailed logging
        current_model = self.llm.config_manager.get_param("model")
        logger.info(f"Using model: {current_model}")
        
        can_use_tools = capabilities.get(ModelCapability.TOOL_USE, False) or capabilities.get(ModelCapability.FUNCTION_CALLING, False)
        if not can_use_tools:
            logger.warning(f"Provider may not support tool calls according to capabilities with model {current_model}")
            print(f"Warning: The selected provider {provider_name} with model {current_model} may not support tool calls")
        
        # Create a session with a tool-focused system prompt
        logger.info("Creating session with tool support")
        self.session = Session(
            provider=self.llm,
            system_prompt=(
                "You are a helpful AI assistant that can read files when requested. "
                "You have access to a tool called read_file that can read file contents. "
                "When a user asks you to read a file, you MUST use the read_file tool. "
                "DO NOT make up file contents. ONLY use the read_file tool when asked to read a file. "
                "You MUST NEVER respond with 'I can help you read the file' without actually using the tool. "
                "Instead, you should immediately use the read_file tool without asking for permission. "
                "The read_file tool accepts the following parameters:\n"
                "- file_path: Path to the file to read (required)\n"
                "- max_lines: Maximum number of lines to read (optional)\n\n"
                "Examples of when to use the tool:\n"
                "1. USER: 'Please read test_file.txt'\n"
                "   YOU: *Use read_file with file_path='test_file.txt'*\n"
                "2. USER: 'Show me the first 5 lines of test_file.txt'\n"
                "   YOU: *Use read_file with file_path='test_file.txt', max_lines=5*\n"
                "3. USER: 'What's in the file test_file.txt?'\n"
                "   YOU: *Use read_file with file_path='test_file.txt'*\n\n"
                "NEVER respond with statements like 'I'll help you read that file' or 'I can assist with that' "
                "without actually using the tool. Always use the tool immediately when asked to read a file."
            ),
            tools=[file_reader_tool]
        )
        logger.debug(f"Session created with ID: {self.session.id}")
        
        # Define tool functions mapping
        self.tool_functions = {
            "read_file": read_file
        }
        logger.debug(f"Tool functions registered: {list(self.tool_functions.keys())}")
        
        # Store provider name for later use
        self.provider_name = provider_name
    
    def _get_provider_config(self, provider_name: str, model_name: Optional[str] = None, 
                           api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get provider-specific configuration, including API key and model.
        
        Args:
            provider_name: Name of the provider
            model_name: Specific model to use (optional)
            api_key: Provider API key (optional)
            
        Returns:
            Dictionary of provider configuration
        """
        config = {}
        
        # Set provider-specific defaults and get API key
        if provider_name == "openai":
            config["model"] = model_name or "gpt-4o"
            env_api_key = os.environ.get("OPENAI_API_KEY")
            key_name = "OPENAI_API_KEY"
        elif provider_name == "anthropic":
            config["model"] = model_name or "claude-3-7-sonnet-20250219"  # Updated to use 3.7 Sonnet by default
            env_api_key = os.environ.get("ANTHROPIC_API_KEY")
            key_name = "ANTHROPIC_API_KEY"
        elif provider_name == "ollama":
            config["model"] = model_name or "llama3"  # Adjust based on available models
            env_api_key = None  # Not typically needed for Ollama
            key_name = None
        else:
            config["model"] = model_name or "unknown"
            env_api_key = None
            key_name = f"{provider_name.upper()}_API_KEY"
            logger.warning(f"Unknown provider: {provider_name}")
        
        # Use provided API key or get from environment
        if api_key:
            config["api_key"] = api_key
        elif env_api_key:
            config["api_key"] = env_api_key
        elif key_name and key_name != "OLLAMA_API_KEY":  # Ollama doesn't need an API key
            logger.warning(f"{key_name} environment variable not set")
        
        logger.debug(f"Provider config: {config}")
        return config
    
    def extract_tool_call_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call JSON from text response.
        
        Some providers may include tool calls as JSON blocks in the text response
        rather than as structured fields. This method attempts to extract those.
        
        Args:
            text: Response text to parse
            
        Returns:
            Dictionary containing tool call information or None if not found
        """
        # Try to find JSON pattern that looks like a tool call
        patterns = [
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}',
            r'\{\s*"function"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}',
            r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"tool_input"\s*:\s*(\{[^}]+\})\s*\}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    name = match.group(1)
                    args_str = match.group(2)
                    args = json.loads(args_str)
                    return {"name": name, "arguments": args}
                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"Failed to parse potential tool call: {e}")
        
        return None
    
    def run(self, query: str) -> str:
        """
        Process a user query and return the response.
        
        Args:
            query: The user's query
            
        Returns:
            The agent's response
        """
        logger.info(f"Processing query: {query}")
        
        # Check if this is a file reading request
        is_file_request = any(pattern in query.lower() for pattern in [
            "read file", "read the file", "show file", "show the file", 
            "file contents", "what's in", "what is in", "content of",
            ".txt", ".md", ".py", ".json", "file"
        ])
        
        file_path_match = re.search(r'[\'"]?([a-zA-Z0-9_\-.]+\.[a-zA-Z0-9]+)[\'"]?', query)
        potential_file_path = file_path_match.group(1) if file_path_match else None
        
        max_lines_match = re.search(r'(\d+) lines', query)
        max_lines = int(max_lines_match.group(1)) if max_lines_match else None
        
        try:
            # STEP 1: Agent sending prompt to LLM along with available tools
            log_llm_request(query, self.session.tools)
            
            # Use the standard session tools approach
            logger.debug("Using session.generate_with_tools")
            response = self.session.generate_with_tools(
                tool_functions=self.tool_functions,
                model=self.llm.config_manager.get_param("model"),
                prompt=query,
                temperature=0.1  # Lower temperature to encourage more deterministic tool usage
            )
            
            # STEP 2 & 3: Check if the LLM requested tool calls and Agent executed them
            assistant_messages = [msg for msg in self.session.messages if msg.role == MessageRole.ASSISTANT]
            tool_was_used = False
            direct_tool_execution = False
            
            if assistant_messages and hasattr(assistant_messages[-1], "tool_results") and assistant_messages[-1].tool_results:
                # Get the tool results
                tool_results = assistant_messages[-1].tool_results
                tool_outputs = []
                
                # STEP 2: LLM requested tool calls
                tool_was_used = log_tool_call_request([True])  # Just indicating there were tool calls
                
                for tool_result in tool_results:
                    if "call_id" in tool_result and "output" in tool_result:
                        tool_name = tool_result.get("name", "unknown")
                        # We've already logged the execution in the tool function, but log the collection here
                        logger.info(f"STEP 4: AGENT→LLM - Collected tool result for {tool_name}")
                        
                        if "error" in tool_result and tool_result["error"]:
                            tool_outputs.append(f"Error executing tool: {tool_result['error']}")
                        else:
                            tool_outputs.append(tool_result["output"])
                
                # STEP 5: Sending tool results back to LLM
                log_tool_results_to_llm()
                
                # Combine the tool outputs with the final response content
                if tool_outputs and response.content:
                    result = f"I've read the file for you:\n\n{''.join(tool_outputs)}\n\n{response.content}"
                elif tool_outputs:
                    result = f"I've read the file for you:\n\n{''.join(tool_outputs)}"
                else:
                    result = response.content
            else:
                # No tool calls were made by the LLM
                log_tool_call_request([])
                result = response.content
                
                # If this looks like a file request but no tool was used, try direct execution
                if is_file_request and not tool_was_used and potential_file_path:
                    logger.info(f"FALLBACK: No tool calls from LLM despite file request. Using direct execution with file: {potential_file_path}")
                    try:
                        direct_tool_execution = True
                        file_content = self.tool_functions["read_file"](
                            file_path=potential_file_path, 
                            max_lines=max_lines
                        )
                        
                        # Combine the file content with the model's response
                        result = f"I've read the file for you:\n\n{file_content}\n\nHere's my analysis:\n{result}"
                        
                    except Exception as e:
                        logger.error(f"Error with direct file reading: {e}")
                        # Continue with the original result
            
            # STEP 6: Final response
            if direct_tool_execution:
                logger.info("STEP 6: AGENT→USER - Direct tool execution was used instead of LLM-requested tool call")
            else:
                log_final_response(result)
            
            logger.info(f"Final response to user: {result[:100]}..." if len(result) > 100 else result)
            return result
        
        except Exception as e:
            # Fall back to a simple direct tool execution if everything else fails
            logger.error(f"Error in run method: {e}")
            
            if is_file_request and potential_file_path:
                try:
                    logger.info(f"EMERGENCY FALLBACK: Falling back to direct tool execution for file: {potential_file_path}")
                    file_content = self.tool_functions["read_file"](
                        file_path=potential_file_path, 
                        max_lines=max_lines
                    )
                    return f"I've read the file for you:\n\n{file_content}"
                except Exception as direct_error:
                    logger.error(f"Direct tool execution also failed: {direct_error}")
            
            return f"I encountered an error processing your request: {str(e)}"
        
    def run_streaming(self, query: str) -> None:
        """
        Process a user query and stream the response with tool execution.
        
        Args:
            query: The user's query
        """
        logger.info(f"Processing streaming query: {query}")
        
        try:
            # Use the streaming version of generate_with_tools
            logger.info("Using session.generate_with_tools_streaming")
            stream = self.session.generate_with_tools_streaming(
                tool_functions=self.tool_functions,
                model=self.llm.config_manager.get_param("model"),  # Explicitly pass the model
                prompt=query  # Explicitly pass the query as the prompt
            )
            
            print("\nAssistant: ", end="", flush=True)
            for chunk in stream:
                if isinstance(chunk, str):
                    # Regular content chunk
                    print(chunk, end="", flush=True)
                elif isinstance(chunk, dict) and chunk.get("type") == "tool_result":
                    # Tool result
                    result = chunk.get("result", {})
                    tool_name = result.get("name", "unknown")
                    output = result.get("output")
                    error = result.get("error")
                    
                    if error:
                        print(f"\n[Tool {tool_name} error: {error}]", end="", flush=True)
                    else:
                        print(f"\n[Tool {tool_name} executed]", end="", flush=True)
                        
                        # For file reading, show a preview
                        if tool_name == "read_file" and output:
                            lines = output.split('\n')
                            preview = '\n'.join(lines[:5])
                            if len(lines) > 5:
                                preview += "\n[...file content truncated for display...]"
                            print(f"\nFile content preview:\n{preview}", end="", flush=True)
            
            print()  # Final newline
            
        except Exception as e:
            error_msg = f"Error in streaming mode: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"\nError: {str(e)}")
    
    def run_interactive(self):
        """
        Run the agent in interactive mode, accepting user input from the console.
        """
        logger.info("Starting interactive mode")
        print(f"Basic Agent with {self.provider_name.capitalize()} provider")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'stream' before your query to use streaming mode.")
        print("Example: 'Please read the file test_file.txt'\n")
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested to exit")
                print("\nGoodbye!")
                break
            
            # Check if user wants streaming mode
            streaming = False
            if user_input.lower().startswith("stream "):
                streaming = True
                user_input = user_input[7:]  # Remove "stream " prefix
                logger.debug("Streaming mode requested")
            
            # Process the query
            try:
                if streaming:
                    self.run_streaming(user_input)
                else:
                    response = self.run(user_input)
                    print(f"\nAssistant: {response}")
            except Exception as e:
                error_msg = f"Error in interactive mode: {str(e)}"
                logger.error(error_msg, exc_info=True)
                print(f"\nError: {str(e)}")


def main():
    """Main function to parse arguments and run the agent."""
    parser = argparse.ArgumentParser(description="Basic Agent with tool support")
    parser.add_argument("--provider", default="anthropic", choices=["openai", "anthropic", "ollama"], 
                       help="Provider to use (default: anthropic)")
    parser.add_argument("--model", help="Specific model to use (optional, defaults to provider's default)")
    parser.add_argument("--api-key", help="Provider API key (optional, will use environment variable if not provided)")
    parser.add_argument("--query", help="Single query to run (if not provided, will run in interactive mode)")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for the query")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    
    args = parser.parse_args()
    
    try:
        # Create the agent
        agent = BasicAgent(
            provider_name=args.provider, 
            model_name=args.model, 
            api_key=args.api_key, 
            debug=args.debug
        )
        
        # Run the agent
        if args.query:
            # Single query mode
            logger.info(f"Running single query: {args.query}")
            if args.stream:
                agent.run_streaming(args.query)
            else:
                response = agent.run(args.query)
                print(f"Response: {response}")
        else:
            # Interactive mode
            logger.info("Starting interactive mode")
            agent.run_interactive()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1
    
    logger.info("Agent execution completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 