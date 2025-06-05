"""
ALMA - Abstract Language Model Agent

A minimal, clean agent implementation leveraging AbstractLLM and its tool call abstraction.
"""

import os
import time
import logging
import subprocess
import argparse
from typing import Dict, List, Any, Optional, Callable, Union

from abstractllm import create_llm
from abstractllm.session import Session, SessionManager
from abstractllm.tools import function_to_tool_definition
from abstractllm.types import GenerateResponse
from abstractllm.utils.logging import log_step

# Configure logging - we'll set the level dynamically based on verbose flag
logger = logging.getLogger("alma")
# Create handlers but don't add them yet - we'll do that in main()
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("alma.log")


def configure_logging(verbose: bool = False):
    """Configure logging based on verbose flag."""
    # Set format for handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Set level based on verbose flag
    level = logging.DEBUG if verbose else logging.WARNING
    
    # Configure root logger for abstractllm package
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Make sure handlers aren't duplicated
    if console_handler not in root_logger.handlers:
        root_logger.addHandler(console_handler)
    if file_handler not in root_logger.handlers:
        root_logger.addHandler(file_handler)
    
    # Make our logger slightly more verbose
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    # Configure specific loggers for different components
    logging.getLogger("abstractllm").setLevel(level)
    logging.getLogger("alma.tools").setLevel(logging.INFO if verbose else logging.WARNING)
    
    # If not verbose, suppress httpx logging
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)


def read_file(file_path: str, max_lines: Optional[int] = None) -> str:
    """
    Read the contents of a file with security validation.
    
    Args:
        file_path: The path of the file to read
        max_lines: Maximum number of lines to read (optional)
        
    Returns:
        The file contents as a string, or an error message
    """
    # Validate max_lines
    if max_lines is not None and (max_lines <= 0 or max_lines > 10000):
        return f"Error: max_lines must be between 1 and 10000."
    
    # Execute with safeguards
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if max_lines is not None:
                lines = [next(f) for _ in range(max_lines)]
                content = ''.join(lines)
            else:
                # Limit file size for security
                content = f.read(10 * 1024 * 1024)  # 10MB limit
                if len(content) >= 10 * 1024 * 1024:
                    content += "\n... (file truncated due to size limits)"
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


def execute_command(command: str, timeout: int = 30) -> str:
    """
    Execute a shell command and return its output.
    
    Args:
        command: The command to execute
        timeout: Maximum execution time in seconds (default: 30)
        
    Returns:
        The command output (stdout and stderr) as a string
    """
    logger = logging.getLogger("alma.tools")
    logger.info(f"Executing command: {command}")
    
    try:
        # Execute command with timeout
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            if output:
                output += "\n\n--- STDERR ---\n" + result.stderr
            else:
                output = result.stderr
                
        # Add return code information
        output += f"\n\nExit code: {result.returncode}"
        
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"


class ALMA:
    """
    Abstract Language Model Agent - A minimal agent implementation leveraging AbstractLLM.
    """
    
    def __init__(
        self,
        provider_name: str = "anthropic",
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the ALMA agent.
        
        Args:
            provider_name: The LLM provider to use (default: anthropic)
            model_name: The model to use (default: claude-3-5-haiku-20241022)
            api_key: The API key for the provider (optional)
            session_id: A unique ID for this session (optional)
            verbose: Whether to log detailed information
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self.session_id = session_id or f"alma-{int(time.time())}"
        self.verbose = verbose
        
        # Initialize the provider with configuration
        provider_config = {
            "api_key": api_key,
            "model": model_name
        }
        self.provider = create_llm(provider_name, **provider_config)
        
        # Initialize session manager and create a session
        self.session_manager = SessionManager()
        
        # Define system prompt that explains the agent's capabilities
        self.system_prompt = """
        You are a helpful assistant that can use tools to answer questions.
        When you need information that might require a tool, use the appropriate tool.
        Always think step by step about which tools you need and why.
        For file access, use the read_file tool rather than making assumptions about file contents.
        For executing commands, use the execute_command tool to run shell commands when needed.
        """
        
        # Set up default available tools
        self.tool_functions = {
            "read_file": read_file,
            "execute_command": execute_command,
        }
        
        # Define tool schemas 
        self.tools = [
            function_to_tool_definition(read_file),
            function_to_tool_definition(execute_command),
        ]
        
        # Create initial session
        self.session = self._create_session()
        
        log_step(0, "INIT", f"Initialized ALMA with provider={provider_name}, model={model_name}")

    def _create_session(self) -> Session:
        """Create a new session with the current provider and system prompt."""
        # Create a new session
        session = Session(
            system_prompt=self.system_prompt,
            provider=self.provider
        )
        
        # Add tools to the session
        for tool in self.tools:
            session.add_tool(tool)
            
        # Store the session in the manager
        self.session_manager.sessions[self.session_id] = session
        
        return session

    def _get_session(self) -> Session:
        """Get or create a session for the current session ID."""
        # Try to get an existing session
        session = self.session_manager.get_session(self.session_id)
        
        # Create a new session if one doesn't exist
        if session is None:
            session = self._create_session()
            
        return session

    def run(self, query: str) -> str:
        """
        Process a user query using the LLM-first tool call flow.
        
        Args:
            query: The user's query
            
        Returns:
            The final response from the LLM
        """

        try:
            # STEP 1: User → Agent - Log the incoming query
            log_step(1, "USER→AGENT", f"Received query: {query[:100]}...")
            
            # Get or create a session
            session = self._get_session()
            
            # Add user message to session for context
            session.add_message("user", query)
            
            # STEP 2: Agent → LLM - Send query to LLM with tools
            tool_names = list(self.tool_functions.keys())
            log_step(2, "AGENT→LLM", f"Sending query to {self.provider_name} with available tools: {tool_names}")
            
            # Use the session's generate_with_tools method to handle the flow (tools are executed inside)
            # Note: tool_functions could be omitted since we registered the tools with add_tool,
            # but providing it explicitly ensures consistent behavior
            response = session.generate(
                tool_functions=self.tool_functions,
                model=self.model_name
            )
            # STEP 3: LLM→AGENT - Received LLM response (tool execution done internally)
            log_step(3, "LLM→AGENT", "Received LLM response; tool calls handled internally by session.generate_with_tools")
            
            if self.verbose:
                logger.info(f"RESPONSE: {response}")
                
            # Get the final response content
            final_response = response.content if response.content else ""
            
            # STEP 6: LLM → Agent → User - Return final response
            log_step(6, "LLM→USER", f"Final response: {final_response[:100]}..." if len(final_response) > 100 else final_response)
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error during agent execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"I encountered an error processing your request: {str(e)}"
    
    def run_streaming(self, query: str) -> None:
        """
        Process a user query with streaming output.
        
        Args:
            query: The user's query
        """
        try:
            # STEP 1: User → Agent - Log the incoming query
            log_step(1, "USER→AGENT", f"Received streaming query: {query[:100]}...")
            
            # Get or create a session
            session = self._get_session()
            
            # Add user message to session for context
            session.add_message("user", query)
            
            # STEP 2: Agent → LLM - Send query to LLM with tools
            tool_names = list(self.tool_functions.keys())
            log_step(2, "AGENT→LLM", f"Sending streaming query to {self.provider_name} with available tools: {tool_names}")
            
            # Track content for session history
            content_buffer = []
            
            # Use streaming version with tool support
            print("\nAssistant: ", end="", flush=True)
            
            # Attempt streaming; if streaming API not supported, fall back to non-streaming
            try:
                for chunk in session.generate_with_tools_streaming(
                    tool_functions=self.tool_functions,  # Optional since tools are registered, but included for clarity
                    model=self.model_name
                ):
                    # Debug the chunk type and content
                    if self.verbose:
                        logger.debug(f"Chunk type: {type(chunk)}, Content: {str(chunk)[:100]}...")
                    
                    if isinstance(chunk, str):
                        # Regular content chunk
                        print(chunk, end="", flush=True)
                        content_buffer.append(chunk)
                    elif isinstance(chunk, dict):
                        # Handle different dictionary formats
                        if chunk.get("type") == "tool_result" and (chunk.get("tool_call") or chunk.get("result")):
                            # Tool execution result
                            result = chunk.get("tool_call", {}) or chunk.get("result", {})
                            tool_name = result.get("name", "unknown")
                            tool_args = result.get("arguments", {})
                            
                            # Log tool execution
                            log_step(3, "LLM→AGENT", f"LLM requested tool: {tool_name}")
                            log_step(4, "AGENT→TOOL", f"Executing tool: {tool_name} with args: {tool_args}")
                            log_step(5, "TOOL→LLM", f"Tool execution completed, results sent to LLM")
                            
                            # Visual indicator of tool execution
                            print(f"\n[Executing tool: {tool_name}]\n", flush=True)
                    # Handle ToolCallRequest objects directly from providers
                    elif hasattr(chunk, 'tool_calls') and getattr(chunk, 'tool_calls', None):
                        tool_calls = chunk.tool_calls
                        for tool_call in tool_calls:
                            tool_name = getattr(tool_call, 'name', 'unknown')
                            tool_args = getattr(tool_call, 'arguments', {})
                            
                            # Log tool execution
                            log_step(3, "LLM→AGENT", f"LLM requested tool: {tool_name}")
                            log_step(4, "AGENT→TOOL", f"Executing tool: {tool_name} with args: {tool_args}")
                            log_step(5, "TOOL→LLM", f"Tool execution completed, results sent to LLM")
                            
                            # Visual indicator of tool execution
                            print(f"\n[Executing tool: {tool_name}]\n", flush=True)
                    # Handle individual tool call objects
                    elif hasattr(chunk, 'name') and hasattr(chunk, 'arguments'):
                        tool_name = chunk.name
                        tool_args = chunk.arguments
                        
                        # Log tool execution
                        log_step(3, "LLM→AGENT", f"LLM requested tool: {tool_name}")
                        log_step(4, "AGENT→TOOL", f"Executing tool: {tool_name} with args: {tool_args}")
                        log_step(5, "TOOL→LLM", f"Tool execution completed, results sent to LLM")
                        
                        # Visual indicator of tool execution
                        print(f"\n[Executing tool: {tool_name}]\n", flush=True)
                    else:
                        # Try to extract content from other object types (like Anthropic message chunks)
                        content = None
                        if hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                        elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                            content = chunk.delta.text
                            
                        if content:
                            print(content, end="", flush=True)
                            content_buffer.append(content)
                        elif self.verbose:
                            logger.debug(f"Unhandled chunk type: {type(chunk)}")
            except TypeError as e:
                # Streaming not supported by provider; fallback to non-streaming
                logger.warning(f"Streaming not supported, falling back to run(): {e}")
                response = self.run(query)
                print(f"\nAssistant: {response}\n")
                return
            
            # Join content chunks for session history
            final_content = "".join(content_buffer)
            
            # Add assistant response to session
            session.add_message("assistant", final_content)
            
            # Complete the output with a newline
            print("\n")
            
            # STEP 6: Log final response
            log_step(6, "LLM→USER", f"Final streaming response completed")
            
        except Exception as e:
            error_msg = f"Error during streaming agent execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"\nError: {error_msg}\n")
    
    def run_interactive(self, stream: bool = True):
        """Run the agent in interactive mode."""
        print("ALMA Interactive Mode " + 
              ("(Streaming)" if stream else "(Non-streaming)") + 
              " - Type 'exit' to quit.\n")
        
        while True:
            # Get user input
            query = input("\nYou: ")
            
            # Check for exit command
            if query.lower() in ["exit", "quit", "bye"]:
                print("Ending session. Goodbye!")
                break
                
            # Skip empty inputs
            if not query.strip():
                continue
                
            # Process the query
            if stream:
                self.run_streaming(query)
            else:
                response = self.run(query)
                print(f"\nAssistant: {response}")


def main():
    """Main entry point when script is run directly."""
    parser = argparse.ArgumentParser(description="ALMA - Abstract Language Model Agent")
    parser.add_argument("--provider", default="anthropic", help="LLM provider to use")
    parser.add_argument("--model", default="claude-3-5-haiku-20241022", help="Model name to use")
    parser.add_argument("--query", help="Query to process (if not provided, runs in interactive mode)")
    parser.add_argument("--stream", action="store_true", help="Use streaming output")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    
    args = parser.parse_args()
    
    # Configure logging based on verbose flag
    configure_logging(args.verbose)
    
    # Create agent
    agent = ALMA(
        provider_name=args.provider,
        model_name=args.model,
        verbose=args.verbose
    )
    
    # Process query or run in interactive mode
    if args.query:
        if args.stream:
            agent.run_streaming(args.query)
        else:
            response = agent.run(args.query)
            print(f"\nAssistant: {response}")
    else:
        agent.run_interactive(args.stream)


if __name__ == "__main__":
    main() 