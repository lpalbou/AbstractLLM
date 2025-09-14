#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Simple Tools - Enhanced version with new @tool decorator and Pydantic validation.

This example demonstrates the new SOTA tool creation system with:
- Pydantic validation for robust input checking
- Rich metadata for better LLM understanding
- Retry logic for error recovery
- Timeout support for long operations
- Context injection for session-aware tools
- Docstring parsing for automatic descriptions
"""

import os
import json
import subprocess
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator

from abstractllm.factory import create_session
from abstractllm.structured_response import StructuredResponseConfig, ResponseFormat
from abstractllm.tools import tool, ToolContext
from abstractllm.utils.logging import configure_logging
from abstractllm.interface import ModelParameter
from abstractllm.utils.display import display_response, display_error, display_thinking, display_success, Colors, Symbols
from abstractllm.types import GenerateResponse
from abstractllm.utils.response_helpers import enhance_string_response, save_interaction_context
from abstractllm.utils.commands import create_command_processor
import argparse
import sys
import logging

# Colors for output
BLUE = '\033[34m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
RESET = '\033[0m'

# =============================================================================
# Response Models for Structured Outputs
# =============================================================================

class FileInfo(BaseModel):
    """Information about a file."""
    path: str
    size: int = Field(ge=0, description="File size in bytes")
    is_directory: bool
    permissions: str = Field(pattern=r'^[rwx-]{9}$', description="Unix permissions")
    modified: datetime
    
    @field_validator('path')
    def validate_path(cls, v):
        if '..' in v:
            raise ValueError("Path traversal not allowed")
        return v


class CodeAnalysis(BaseModel):
    """Code analysis results."""
    language: str = Field(description="Programming language")
    lines_of_code: int = Field(ge=0)
    complexity: str = Field(pattern=r'^(low|medium|high)$')
    imports: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)


class SearchMatch(BaseModel):
    """A search result match."""
    file: str
    line_number: int = Field(ge=1)
    content: str
    context: Optional[str] = None


# =============================================================================
# Enhanced Tools with @tool Decorator
# =============================================================================

@tool(
    parse_docstring=True,
    retry_on_error=True,
    timeout=5.0,
    tags=["filesystem", "read"],
    when_to_use="When you need to read the contents of a file"
)
def read_file(
    file_path: str = Field(
        description="Path to the file to read",
        min_length=1,
        max_length=500
    ),
    max_lines: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of lines to read"
    ),
    encoding: str = Field(
        default="utf-8",
        pattern=r'^(utf-8|ascii|latin-1)$',
        description="File encoding"
    )
) -> str:
    """
    Read the contents of a file with validation and limits.
    
    Args:
        file_path: Path to the file (relative or absolute)
        max_lines: Maximum lines to read (1-10000)
        encoding: File encoding (utf-8, ascii, or latin-1)
        
    Returns:
        File contents as string
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        with open(path, 'r', encoding=encoding) as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... (truncated after {max_lines} lines)")
                    break
                lines.append(line)
            return ''.join(lines)
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")


@tool(
    parse_docstring=True,
    timeout=3.0,
    tags=["filesystem", "list"],
    when_to_use="When you need to list files in a directory"
)
def list_files(
    directory: str = Field(
        default=".",
        description="Directory to list"
    ),
    pattern: Optional[str] = Field(
        default=None,
        description="Glob pattern to filter files (e.g., '*.py')"
    ),
    recursive: bool = Field(
        default=False,
        description="Search recursively in subdirectories"
    ),
    include_hidden: bool = Field(
        default=False,
        description="Include hidden files (starting with .)"
    )
) -> List[Dict[str, Any]]:
    """
    List files in a directory with detailed information.
    
    Args:
        directory: Directory path to list
        pattern: Optional glob pattern filter
        recursive: Whether to search recursively
        include_hidden: Include hidden files
        
    Returns:
        List of file information dictionaries
    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        files = []
        
        if recursive and pattern:
            paths = dir_path.rglob(pattern)
        elif pattern:
            paths = dir_path.glob(pattern)
        elif recursive:
            paths = dir_path.rglob("*")
        else:
            paths = dir_path.iterdir()
        
        for path in paths:
            # Skip hidden files if not requested
            if not include_hidden and path.name.startswith('.'):
                continue
            
            # Get file stats
            stat = path.stat()
            
            # Format permissions (simplified)
            perms = 'rwxrwxrwx' if path.is_dir() else 'rw-rw-rw-'
            
            files.append({
                "path": str(path.relative_to(dir_path)),
                "size": stat.st_size,
                "is_directory": path.is_dir(),
                "permissions": perms,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Sort by name
        files.sort(key=lambda f: (not f['is_directory'], f['path'].lower()))
        
        # Limit results
        return files[:100]  # Max 100 files to avoid overwhelming
        
    except Exception as e:
        raise ValueError(f"Failed to list files: {e}")


@tool(
    parse_docstring=True,
    retry_on_error=True,
    timeout=10.0,
    tags=["search", "grep"],
    when_to_use="When you need to search for text patterns in files"
)
def search_files(
    pattern: str = Field(
        description="Search pattern (regex supported)",
        min_length=1,
        max_length=200
    ),
    directory: str = Field(
        default=".",
        description="Directory to search in"
    ),
    file_pattern: str = Field(
        default="*",
        description="File pattern to search (e.g., '*.py')"
    ),
    case_sensitive: bool = Field(
        default=True,
        description="Case sensitive search"
    ),
    max_results: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of results"
    )
) -> List[Dict[str, Any]]:
    """
    Search for text patterns in files using regex.
    
    Args:
        pattern: Regex pattern to search for
        directory: Directory to search in
        file_pattern: File pattern filter
        case_sensitive: Whether search is case sensitive
        max_results: Maximum results to return
        
    Returns:
        List of search matches with file, line, and content
    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        matches = []
        files_searched = 0
        
        # Search files
        for file_path in dir_path.rglob(file_pattern):
            if not file_path.is_file():
                continue
            
            files_searched += 1
            if files_searched > 100:  # Limit files searched
                break
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append({
                                "file": str(file_path.relative_to(dir_path)),
                                "line_number": line_num,
                                "content": line.strip(),
                                "context": None
                            })
                            
                            if len(matches) >= max_results:
                                return matches
            except:
                continue  # Skip files that can't be read
        
        return matches
        
    except Exception as e:
        raise ValueError(f"Search failed: {e}")


@tool(
    parse_docstring=True,
    timeout=5.0,
    tags=["code", "analysis"],
    when_to_use="When you need to analyze code structure and complexity"
)
def analyze_code(
    file_path: str = Field(
        description="Path to code file to analyze",
        min_length=1
    ),
    include_docstrings: bool = Field(
        default=True,
        description="Include docstring analysis"
    )
) -> Dict[str, Any]:
    """
    Analyze code file for structure and complexity.
    
    Args:
        file_path: Path to the code file
        include_docstrings: Whether to analyze docstrings
        
    Returns:
        Dictionary with language, complexity, and structure info
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Detect language by extension
        ext = path.suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        language = language_map.get(ext, 'unknown')
        
        # Count lines
        lines = content.split('\n')
        loc = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        
        # Extract imports (Python example)
        imports = []
        functions = []
        classes = []
        
        if language == 'python':
            import_pattern = re.compile(r'^(?:from|import)\s+(\S+)')
            func_pattern = re.compile(r'^def\s+(\w+)\s*\(')
            class_pattern = re.compile(r'^class\s+(\w+)\s*[\(:]')
            
            for line in lines:
                if match := import_pattern.match(line.strip()):
                    imports.append(match.group(1))
                elif match := func_pattern.match(line.strip()):
                    functions.append(match.group(1))
                elif match := class_pattern.match(line.strip()):
                    classes.append(match.group(1))
        
        # Estimate complexity
        if loc < 100:
            complexity = "low"
        elif loc < 500:
            complexity = "medium"
        else:
            complexity = "high"
        
        return {
            "language": language,
            "lines_of_code": loc,
            "complexity": complexity,
            "imports": imports[:20],  # Limit to 20
            "functions": functions[:20],
            "classes": classes[:20]
        }
        
    except Exception as e:
        raise ValueError(f"Code analysis failed: {e}")


@tool(
    parse_docstring=True,
    timeout=10.0,
    tags=["system", "command"],
    when_to_use="When you need to execute a shell command",
    requires_confirmation=False  # Set to True for safety in production
)
def execute_command(
    command: str = Field(
        description="Shell command to execute",
        min_length=1,
        max_length=500
    ),
    working_dir: str = Field(
        default=".",
        description="Working directory for command"
    ),
    timeout: float = Field(
        default=5.0,
        ge=0.1,
        le=30.0,
        description="Command timeout in seconds"
    )
) -> Dict[str, Any]:
    """
    Execute a shell command safely with timeout.
    
    Args:
        command: The shell command to execute
        working_dir: Directory to run command in
        timeout: Maximum execution time
        
    Returns:
        Dict with exit_code, stdout, and stderr
    """
    # Safety checks
    dangerous_commands = ['rm -rf', 'format', 'del /f', 'dd', 'mkfs']
    for dangerous in dangerous_commands:
        if dangerous in command.lower():
            raise ValueError(f"Dangerous command blocked: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout[:5000],  # Limit output
            "stderr": result.stderr[:1000],
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "success": False
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }


@tool(
    parse_docstring=True,
    tags=["utility", "hash"],
    when_to_use="When you need to calculate file checksums"
)
def calculate_hash(
    file_path: str = Field(description="Path to file"),
    algorithm: str = Field(
        default="sha256",
        pattern=r'^(md5|sha1|sha256|sha512)$',
        description="Hash algorithm"
    )
) -> str:
    """
    Calculate cryptographic hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        Hex digest of the file hash
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Select hash algorithm
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'sha512':
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Read and hash file in chunks
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
        
    except Exception as e:
        raise ValueError(f"Hash calculation failed: {e}")


@tool(
    parse_docstring=True,
    requires_context=True,
    tags=["memory", "session"],
    when_to_use="When you need to save important information to memory"
)
def remember_fact(
    subject: str = Field(description="Subject of the fact", min_length=1),
    predicate: str = Field(description="Relationship/predicate", min_length=1),
    object: str = Field(description="Object of the fact", min_length=1),
    confidence: float = Field(default=0.8, ge=0.0, le=1.0),
    context: Optional[ToolContext] = None
) -> Dict[str, str]:
    """
    Save a fact to the session's knowledge graph.
    
    Args:
        subject: The subject entity
        predicate: The relationship
        object: The object entity
        confidence: Confidence score (0-1)
        context: Session context (auto-injected)
        
    Returns:
        Confirmation of fact storage
    """
    if context and context.memory:
        fact_id = context.memory.add_fact(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence
        )
        return {
            "status": "saved",
            "fact_id": fact_id,
            "message": f"Remembered: {subject} {predicate} {object}"
        }
    else:
        return {
            "status": "no_memory",
            "message": "Memory system not available"
        }


@tool(
    parse_docstring=True,
    tags=["utility", "json"],
    when_to_use="When you need to parse or validate JSON data"
)
def parse_json(
    json_string: str = Field(description="JSON string to parse"),
    validate_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON schema for validation"
    )
) -> Dict[str, Any]:
    """
    Parse and optionally validate JSON data.
    
    Args:
        json_string: JSON string to parse
        validate_schema: Optional schema for validation
        
    Returns:
        Parsed JSON as dictionary
    """
    try:
        data = json.loads(json_string)
        
        # Basic schema validation if provided
        if validate_schema:
            required = validate_schema.get('required', [])
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
        
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"JSON parsing failed: {e}")


def create_agent(provider="ollama", model="qwen3-coder:30b", memory_path=None, max_tool_calls=25, 
                 seed=None, top_p=None, max_input_tokens=None, frequency_penalty=None, presence_penalty=None):
    """Create an enhanced agent with new tool system."""
    
    print(f"{BLUE}🧠 Creating intelligent agent with enhanced tools:{RESET}")
    print(f"  • Pydantic validation")
    print(f"  • Rich metadata and descriptions")
    print(f"  • Retry logic on errors")
    print(f"  • Timeout protection")
    print(f"  • Context-aware tools\n")
    
    # List of enhanced tools
    tools = [
        read_file,
        list_files,
        search_files,
        analyze_code,
        execute_command,
        calculate_hash,
        remember_fact,
        parse_json
    ]
    
    # Build configuration with SOTA parameters
    config = {
        'model': model,
        'enable_memory': True,
        'enable_retry': True,
        'persist_memory': memory_path,
        'memory_config': {
            'working_memory_size': 10,
            'consolidation_threshold': 5
        },
        'tools': tools,
        'system_prompt': """You are an intelligent AI assistant with advanced tool capabilities.
You have access to enhanced tools with validation and error handling.
When using tools, pay attention to the parameter requirements and constraints.
If a tool call fails with validation errors, correct the parameters and try again.""",
        'max_tokens': 2048,
        'temperature': 0.7,
        'max_tool_calls': max_tool_calls
    }
    
    # Add SOTA parameters if specified
    if seed is not None:
        config[ModelParameter.SEED] = seed
    if top_p is not None:
        config[ModelParameter.TOP_P] = top_p
    if max_input_tokens is not None:
        config[ModelParameter.MAX_INPUT_TOKENS] = max_input_tokens
    if frequency_penalty is not None:
        config[ModelParameter.FREQUENCY_PENALTY] = frequency_penalty
    if presence_penalty is not None:
        config[ModelParameter.PRESENCE_PENALTY] = presence_penalty
    
    session = create_session(provider, **config)
    
    # Display tool information
    print(f"{GREEN}📦 Enhanced tools loaded:{RESET}")
    for tool_func in tools:
        if hasattr(tool_func, 'tool_definition'):
            tool_def = tool_func.tool_definition
            tags = ', '.join(tool_def.tags) if tool_def.tags else 'none'
            print(f"  • {tool_def.name} - {tool_def.description[:50]}... [{tags}]")
    print()
    
    if memory_path:
        print(f"{GREEN}💾 Memory persisted to: {memory_path}{RESET}\n")
    
    return session


def run_query(session, prompt, structured_output=None):
    """Execute a query with the agent and display beautiful results."""
    
    # Show thinking indicator
    display_thinking("Processing your query...")
    
    # Configure structured output if requested
    config = None
    if structured_output:
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON if structured_output == "json" else ResponseFormat.YAML,
            force_valid_json=True,
            max_retries=3,
            temperature_override=0.0
        )
    
    try:
        # Try SOTA features first, fallback to simple generation
        try:
            response = session.generate(
                prompt=prompt,
                use_memory_context=True,    # Inject relevant memories
                create_react_cycle=True,     # Create ReAct cycle with scratchpad
                structured_config=config     # Structured output if configured
            )
        except Exception as sota_error:
            # Fallback to simple generation without SOTA features
            print(f"{Colors.DIM}Note: Using simplified mode due to session compatibility{Colors.RESET}")
            response = session.generate_with_tools(
                prompt=prompt,
                max_tool_calls=session.max_tool_calls if hasattr(session, 'max_tool_calls') else 25
            )
        
        # Convert string responses to enhanced GenerateResponse objects
        if isinstance(response, str):
            response = enhance_string_response(
                content=response,
                model=getattr(session._provider, 'config_manager', {}).get_param('model') if hasattr(session, '_provider') else 'unknown'
            )
        
        # Save interaction context for facts/scratchpad commands
        if isinstance(response, GenerateResponse):
            save_interaction_context(response, prompt)
            display_response(response)
        else:
            # Ultimate fallback
            print(f"\n{Colors.BRIGHT_GREEN}Response:{Colors.RESET} {response}")
        
        return response
        
    except Exception as e:
        display_error(str(e))
        return None


def show_memory_insights(session):
    """Display memory system insights."""
    
    if not hasattr(session, 'memory'):
        return
    
    memory = session.memory
    stats = memory.get_statistics()
    
    print(f"\n{BLUE}📊 Memory Insights:{RESET}")
    print(f"  • Working Memory: {stats['memory_distribution']['working_memory']} items")
    print(f"  • Episodic Memory: {stats['memory_distribution']['episodic_memory']} experiences")
    print(f"  • Knowledge Graph: {stats['knowledge_graph']['total_facts']} facts")
    print(f"  • ReAct Cycles: {stats['total_react_cycles']} ({stats['successful_cycles']} successful)")
    print(f"  • Bidirectional Links: {stats['link_statistics']['total_links']}")
    
    # Show sample facts from knowledge graph
    if memory.knowledge_graph.facts:
        print(f"\n  {GREEN}Sample Knowledge Graph Triples:{RESET}")
        for i, (fact_id, fact) in enumerate(list(memory.knowledge_graph.facts.items())[:5]):
            print(f"    {i+1}. {fact.subject} --[{fact.predicate}]--> {fact.object}")
    
    # Show current ReAct cycle if active
    if session.current_cycle:
        cycle = session.current_cycle
        print(f"\n  {GREEN}Current ReAct Cycle:{RESET}")
        print(f"    ID: {cycle.cycle_id}")
        print(f"    Query: {cycle.query[:100]}...")
        print(f"    Thoughts: {len(cycle.thoughts)}")
        print(f"    Actions: {len(cycle.actions)}")
        print(f"    Observations: {len(cycle.observations)}")


def interactive_mode(session):
    """Run enhanced interactive chat with slash command support."""
    
    # Create command processor
    cmd_processor = create_command_processor(session)
    
    print(f"\n{Colors.BRIGHT_BLUE}{Symbols.SPARKLES} Enhanced Interactive Mode{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 50}{Colors.RESET}")
    print(f"{Colors.DIM}Type {Colors.BRIGHT_BLUE}/help{Colors.DIM} for commands or ask questions directly.{Colors.RESET}")
    print(f"{Colors.DIM}Use {Colors.BRIGHT_BLUE}/exit{Colors.DIM} to quit.{Colors.RESET}\n")
    
    while True:
        try:
            user_input = input(f"{Colors.BRIGHT_GREEN}alma>{Colors.RESET} ").strip()
            
            if not user_input:
                continue
            
            # Process slash commands
            if cmd_processor.process_command(user_input):
                continue
            
            # Regular query - generate response
            response = run_query(session, user_input)
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.BRIGHT_GREEN}{Symbols.CHECKMARK} Goodbye!{Colors.RESET}")
            break
        except Exception as e:
            display_error(f"Unexpected error: {str(e)}")
            print(f"{Colors.DIM}You can continue or type {Colors.BRIGHT_BLUE}/exit{Colors.DIM} to quit.{Colors.RESET}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="ALMA Simple - Intelligent agent with AbstractLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
    Interactive chat with memory and tools
  
  %(prog)s --prompt "What files are here?"
    Single query execution
  
  %(prog)s --memory agent.pkl --prompt "Remember my name is Alice"
    Use persistent memory
  
  %(prog)s --structured json --prompt "List 3 colors with hex codes"
    Get structured JSON output
  
  %(prog)s --provider openai --seed 12345 --top-p 0.8 --prompt "Generate text"
    Use SOTA parameters for reproducible, controlled generation
  
  %(prog)s --provider openai --frequency-penalty 1.0 --presence-penalty 0.5
    Use OpenAI-specific parameters for content control
"""
    )
    
    parser.add_argument(
        "--provider",
        default="ollama",
        help="LLM provider (default: ollama)"
    )
    
    parser.add_argument(
        "--model",
        default="qwen3-coder:30b",
        help="Model to use (default: qwen3-coder:30b)"
    )
    
    parser.add_argument(
        "--prompt",
        help="Single prompt to execute (exits after)"
    )
    
    parser.add_argument(
        "--memory",
        help="Path to persist memory (e.g., agent.pkl)"
    )
    
    parser.add_argument(
        "--structured",
        choices=["json", "yaml"],
        help="Force structured output format"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )
    
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=25,
        help="Maximum number of tool call iterations (default: 25)"
    )
    
    # SOTA parameters
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible generation"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling parameter (0.0-1.0)"
    )
    
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        help="Maximum input context length"
    )
    
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        help="Frequency penalty (-2.0 to 2.0, OpenAI only)"
    )
    
    parser.add_argument(
        "--presence-penalty", 
        type=float,
        help="Presence penalty (-2.0 to 2.0, OpenAI only)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        configure_logging(console_level=logging.DEBUG)
    else:
        configure_logging(console_level=logging.WARNING)
    
    # Create agent
    session = create_agent(
        provider=args.provider,
        model=args.model,
        memory_path=args.memory,
        max_tool_calls=args.max_tool_calls,
        seed=args.seed,
        top_p=getattr(args, 'top_p', None),
        max_input_tokens=getattr(args, 'max_input_tokens', None),
        frequency_penalty=getattr(args, 'frequency_penalty', None),
        presence_penalty=getattr(args, 'presence_penalty', None)
    )
    
    # Execute single prompt or start interactive mode
    if args.prompt:
        print(f"\n{Colors.BRIGHT_CYAN}{Symbols.TARGET} Query:{Colors.RESET} {Colors.WHITE}{args.prompt}{Colors.RESET}\n")
        response = run_query(session, args.prompt, args.structured)
        
        # Only show memory insights if response was successful
        if response is not None:
            show_memory_insights(session)
    else:
        interactive_mode(session)
    
    # Save memory if persisting
    if args.memory and hasattr(session, 'memory') and session.memory:
        session.memory.save_to_disk()
        print(f"\n{Colors.BRIGHT_GREEN}{Symbols.CHECKMARK} Memory saved to {args.memory}{Colors.RESET}")


if __name__ == "__main__":
    main()