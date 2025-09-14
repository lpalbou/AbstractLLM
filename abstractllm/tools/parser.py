"""
Architecture-based tool call parsing and formatting.

This module handles the detection and parsing of tool calls from model
responses based on their architecture.
"""

import re
import json
import logging
from typing import List, Optional, Dict, Any
from enum import Enum

from abstractllm.tools.core import ToolCall, ToolDefinition
from abstractllm.architectures import detect_architecture, get_architecture_format

logger = logging.getLogger(__name__)


class ToolFormat(Enum):
    """Tool call formats for different architectures."""
    
    # JSON-based
    RAW_JSON = "raw_json"              # {"name": "...", "arguments": {...}}
    FUNCTION_CALL = "function_call"    # <function_call>...</function_call>
    SPECIAL_TOKEN = "special_token"    # <|tool_call|>...
    
    # Code-based  
    TOOL_CODE = "tool_code"           # ```tool_code\nfunc(...)```
    
    # XML-based
    XML_WRAPPED = "xml_wrapped"       # <tool_call>...</tool_call>


def detect_tool_calls(response: str, model_name: Optional[str] = None) -> bool:
    """
    Detect if response contains tool calls.
    
    Args:
        response: Model response text
        model_name: Optional model name for architecture detection
        
    Returns:
        True if tool calls detected
    """
    if not response or not response.strip():
        return False
    
    # Get expected format from architecture
    tool_format = _get_tool_format(model_name)
    
    # Check format-specific patterns
    # Be lenient - only check for opening tags since models may forget closing tags
    if tool_format == ToolFormat.TOOL_CODE:
        return "```tool_code" in response or "```tool_call" in response or "tool_call:" in response
    elif tool_format == ToolFormat.SPECIAL_TOKEN:
        return "<|tool_call|>" in response  # Just check opening tag
    elif tool_format == ToolFormat.FUNCTION_CALL:
        return "<function_call" in response or _has_json_tool_pattern(response)
    elif tool_format == ToolFormat.XML_WRAPPED:
        return "<tool_call>" in response  # Just check opening tag
    else:
        # Try common patterns - be lenient with any opening tag
        return any([
            "```tool_code" in response,
            "```tool_call" in response,  # Add support for tool_call blocks
            "tool_call:" in response,  # Add support for Gemma 3 style
            "<|tool_call|>" in response,
            "<function_call" in response,
            "<tool_call>" in response,
            _has_json_tool_pattern(response),
            # Also check for Python code blocks with tool function calls
            re.search(r'```(?:python|json)?\s*\n.*?list_files\(', response, re.DOTALL) is not None,
            re.search(r'```(?:python|json)?\s*\n.*?read_file\(', response, re.DOTALL) is not None,
            re.search(r'```(?:python|json)?\s*\n.*?search_files\(', response, re.DOTALL) is not None
        ])


def parse_tool_calls(response: str, model_name: Optional[str] = None) -> List[ToolCall]:
    """
    Parse tool calls from response.
    
    Args:
        response: Model response containing tool calls
        model_name: Optional model name for architecture detection
        
    Returns:
        List of parsed tool calls
    """
    if not response or not response.strip():
        return []
    
    # Get expected format
    tool_format = _get_tool_format(model_name)
    
    # Parse based on format
    parsers = {
        ToolFormat.TOOL_CODE: _parse_tool_code,
        ToolFormat.SPECIAL_TOKEN: _parse_special_token,
        ToolFormat.FUNCTION_CALL: _parse_function_call,
        ToolFormat.XML_WRAPPED: _parse_xml_wrapped,
        ToolFormat.RAW_JSON: _parse_raw_json
    }
    
    parser = parsers.get(tool_format, _parse_any_format)
    return parser(response)


def format_tool_prompt(tools: List[ToolDefinition], model_name: Optional[str] = None) -> str:
    """
    Format tools into a system prompt based on model architecture.
    
    Args:
        tools: List of tool definitions
        model_name: Optional model name for architecture detection
        
    Returns:
        Formatted system prompt
    """
    if not tools:
        return "You are a helpful AI assistant."
    
    # Get tool format
    tool_format = _get_tool_format(model_name)
    
    # Format based on architecture
    if tool_format == ToolFormat.TOOL_CODE:
        return _format_gemma_style(tools)
    elif tool_format == ToolFormat.SPECIAL_TOKEN:
        return _format_qwen_style(tools)
    elif tool_format == ToolFormat.FUNCTION_CALL:
        return _format_llama_style(tools)
    elif tool_format == ToolFormat.XML_WRAPPED:
        return _format_xml_style(tools)
    else:
        return _format_generic_style(tools)


# Internal helpers

def _get_tool_format(model_name: Optional[str]) -> ToolFormat:
    """Get tool format for a model."""
    if not model_name:
        return ToolFormat.RAW_JSON
    
    architecture = detect_architecture(model_name)
    if not architecture:
        return ToolFormat.RAW_JSON
    
    # Map architectures to formats
    format_map = {
        "gemma": ToolFormat.TOOL_CODE,
        "qwen": ToolFormat.SPECIAL_TOKEN,
        "llama": ToolFormat.FUNCTION_CALL,
        "phi": ToolFormat.XML_WRAPPED,
        "mistral": ToolFormat.FUNCTION_CALL
    }
    
    return format_map.get(architecture, ToolFormat.RAW_JSON)


def _has_json_tool_pattern(text: str) -> bool:
    """Check if text contains JSON tool call patterns."""
    patterns = [
        r'\{"name":\s*"[^"]+',
        r'\{"function":\s*"[^"]+',
        r'"name":\s*"[^"]+.*"arguments":\s*\{',
        # More flexible patterns to catch variations
        r'\{\s*"name"\s*:\s*"[^"]+',
        r'\{\s*"function"\s*:\s*"[^"]+',
        r'"name"\s*:\s*"[^"]+[^}]*"arguments"\s*:\s*\{',
        # Catch common tool names in JSON-like structures
        r'\{\s*"name"\s*:\s*"(?:list_files|read_file|search_files|write_file)',
        r'"(?:list_files|read_file|search_files|write_file)"\s*,?\s*"arguments"',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# Parsing functions

def _parse_tool_code(response: str) -> List[ToolCall]:
    """Parse ```tool_code format and tool_call: format."""
    tool_calls = []
    
    # Strategy 1: Parse ```tool_code blocks
    pattern = r'```tool_code\s*\n(.*?)\n```'
    for match in re.findall(pattern, response, re.DOTALL):
        # Parse function calls like: func_name(arg1="val1", arg2=123)
        func_pattern = r'(\w+)\s*\((.*?)\)'
        for func_match in re.finditer(func_pattern, match):
            name = func_match.group(1)
            args_str = func_match.group(2)
            
            # Parse arguments
            arguments = {}
            if args_str:
                # Simple argument parsing
                arg_pattern = r'(\w+)\s*=\s*([^,]+)'
                for arg_match in re.finditer(arg_pattern, args_str):
                    key = arg_match.group(1)
                    value = arg_match.group(2).strip()
                    
                    # Parse value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    
                    arguments[key] = value
            
            tool_calls.append(ToolCall(name=name, arguments=arguments))
    
    # Strategy 1.5: Parse ```tool_call blocks (Gemma 3 alternative format)
    tool_call_block_pattern = r'```tool_call\s*\n(.*?)\n```'
    for match in re.findall(tool_call_block_pattern, response, re.DOTALL):
        # Parse function calls like: func_name(arg1="val1", arg2=123) or just func_name
        func_pattern = r'(\w+)(?:\s*\((.*?)\))?'
        for func_match in re.finditer(func_pattern, match.strip()):
            name = func_match.group(1)
            args_str = func_match.group(2) if func_match.group(2) else ""
            
            # Parse arguments
            arguments = {}
            if args_str:
                # Simple argument parsing
                arg_pattern = r'(\w+)\s*=\s*([^,)]+)'
                for arg_match in re.finditer(arg_pattern, args_str):
                    key = arg_match.group(1)
                    value = arg_match.group(2).strip()
                    
                    # Parse value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    
                    arguments[key] = value
            
            tool_calls.append(ToolCall(name=name, arguments=arguments))
    
    # Strategy 2: Parse tool_call: format (Gemma 3 style)
    # Look for patterns like: tool_call: function_name(arg1="val1", arg2=val2)
    tool_call_pattern = r'tool_call:\s*(\w+)\s*\((.*?)\)'
    for match in re.finditer(tool_call_pattern, response):
        name = match.group(1)
        args_str = match.group(2)
        
        # Parse arguments
        arguments = {}
        if args_str:
            # Simple argument parsing
            arg_pattern = r'(\w+)\s*=\s*([^,)]+)'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2).strip()
                
                # Parse value
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                
                arguments[key] = value
        
        tool_calls.append(ToolCall(name=name, arguments=arguments))
    
    return tool_calls


def _parse_special_token(response: str) -> List[ToolCall]:
    """Parse <|tool_call|> format with robust fallback."""
    tool_calls = []
    
    # First, find all tool call positions to avoid duplicates from overlapping patterns
    all_matches = []
    
    # Strategy 1: Look for properly closed tags (more flexible with whitespace)
    pattern_with_close = r'<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>'
    for match in re.finditer(pattern_with_close, response, re.DOTALL | re.IGNORECASE):
        all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    # Strategy 2: Look for opening tags followed by valid JSON (no closing tag) - more flexible
    # This handles cases where the JSON might span multiple lines or have various whitespace
    pattern_no_close = r'<\|tool_call\|>\s*(\{(?:[^{}]|(?:\{[^{}]*\}))*\})\s*(?:</\|tool_call\|>|$|\n|<)'
    for match in re.finditer(pattern_no_close, response, re.DOTALL | re.IGNORECASE):
        # Check if this match overlaps with any closed tag match
        overlaps = False
        for closed_start, closed_end, _ in all_matches:
            if match.start() >= closed_start and match.start() < closed_end:
                overlaps = True
                break
        if not overlaps:
            all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    # Strategy 3: Ultra-robust pattern - just find start tag + JSON, ignore ending completely
    # This is the most important pattern - prioritize start tag detection and valid JSON
    pattern_start_json = r'<\|tool_call\|>\s*(\{[^<]*?\})'
    for match in re.finditer(pattern_start_json, response, re.DOTALL | re.IGNORECASE):
        # Check if this match overlaps with any previous matches
        overlaps = False
        for prev_start, prev_end, _ in all_matches:
            if match.start() >= prev_start and match.start() < prev_end:
                overlaps = True
                break
        if not overlaps:
            json_candidate = match.group(1).strip()
            # Basic validation that it looks like JSON and contains tool structure
            if (json_candidate.startswith('{') and json_candidate.endswith('}') and 
                ('"name"' in json_candidate or '"function"' in json_candidate)):
                all_matches.append((match.start(), match.end(), json_candidate))
    
    # Strategy 4: Even more flexible pattern to catch edge cases
    # Look for the opening tag followed by anything that looks like JSON until we hit a logical end
    pattern_flexible = r'<\|tool_call\|>\s*(\{[^<]*?\})\s*(?:</\|tool_call\|>|\|>|\n\s*\n|\[context|\n\s*<|\Z)'
    for match in re.finditer(pattern_flexible, response, re.DOTALL | re.IGNORECASE):
        # Check if this match overlaps with any previous matches
        overlaps = False
        for prev_start, prev_end, _ in all_matches:
            if match.start() >= prev_start and match.start() < prev_end:
                overlaps = True
                break
        if not overlaps:
            json_candidate = match.group(1).strip()
            # Basic validation that it looks like JSON
            if json_candidate.startswith('{') and json_candidate.endswith('}'):
                all_matches.append((match.start(), match.end(), json_candidate))
    
    # Strategy 5: Look for Python code blocks that might contain JSON-like tool calls
    # This is for models that misunderstand the format but still try to make tool calls
    pattern_code_block = r'```(?:python|json)?\s*\n.*?list_files\(([^)]*)\).*?\n```'
    for match in re.finditer(pattern_code_block, response, re.DOTALL):
        # Extract arguments from function call
        args_str = match.group(1).strip()
        arguments = {}
        
        # Parse simple keyword arguments if any
        if args_str:
            arg_pattern = r'(\w+)\s*=\s*([^,]+)'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2).strip()
                
                # Parse value
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                
                arguments[key] = value
        
        # Create a tool call for list_files
        tool_calls.append(ToolCall(
            name="list_files",
            arguments=arguments
        ))
    
    # Sort by position and parse each match
    all_matches.sort(key=lambda x: x[0])
    for _, _, json_str in all_matches:
        try:
            # Clean up the JSON string - remove any trailing content that might interfere
            json_str = json_str.strip()
            
            # Handle cases where there might be trailing text after the JSON
            if json_str.count('{') > json_str.count('}'):
                # Missing closing braces - try to add them
                missing_braces = json_str.count('{') - json_str.count('}')
                json_str += '}' * missing_braces
            
            # Try to find the JSON object boundaries more precisely
            brace_count = 0
            json_end = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > 0:
                json_str = json_str[:json_end]
            
            # Additional cleanup for common issues
            # Remove any trailing characters that might be malformed closing tags
            if json_str.endswith('}|>'):
                json_str = json_str[:-2]  # Remove |>
            elif json_str.endswith('}}>'):
                json_str = json_str[:-1]  # Remove extra >
            
            # Try normal JSON parsing first
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback: fix common LLM JSON issues (unescaped newlines)
                fixed_json = json_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                data = json.loads(fixed_json)
            
            if isinstance(data, dict) and "name" in data:
                tool_calls.append(ToolCall(
                    name=data["name"],
                    arguments=data.get("arguments", {})
                ))
        except json.JSONDecodeError as e:
            # More detailed logging for debugging
            logger.debug(f"JSON decode error for tool call: {e}, JSON string: {repr(json_str)}")
            continue
    
    return tool_calls


def _parse_function_call(response: str) -> List[ToolCall]:
    """Parse <function_call> format with robust fallback."""
    tool_calls = []
    all_matches = []
    
    # Strategy 1: Look for properly closed tags
    pattern_closed = r'<function_call>(.*?)</function_call>'
    for match in re.finditer(pattern_closed, response, re.DOTALL):
        all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    # Strategy 2: Look for opening tag followed by valid JSON (no closing tag)
    pattern_open = r'<function_call>\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    for match in re.finditer(pattern_open, response, re.DOTALL):
        # Check if this match overlaps with any closed tag match
        overlaps = False
        for closed_start, closed_end, _ in all_matches:
            if match.start() >= closed_start and match.start() < closed_end:
                overlaps = True
                break
        if not overlaps:
            all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    # Sort by position and parse each match
    all_matches.sort(key=lambda x: x[0])
    for _, _, json_str in all_matches:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "name" in data:
                tool_calls.append(ToolCall(
                    name=data["name"],
                    arguments=data.get("arguments", {})
                ))
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Try raw JSON as last resort
    if not tool_calls:
        tool_calls.extend(_parse_raw_json(response))
    
    return tool_calls


def _parse_xml_wrapped(response: str) -> List[ToolCall]:
    """Parse <tool_call> XML format with robust fallback."""
    tool_calls = []
    all_matches = []
    
    # Strategy 1: Look for properly closed tags
    pattern_closed = r'<tool_call>(.*?)</tool_call>'
    for match in re.finditer(pattern_closed, response, re.DOTALL):
        all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    # Strategy 2: Look for opening tag followed by valid JSON (no closing tag)
    pattern_open = r'<tool_call>\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    for match in re.finditer(pattern_open, response, re.DOTALL):
        # Check if this match overlaps with any closed tag match
        overlaps = False
        for closed_start, closed_end, _ in all_matches:
            if match.start() >= closed_start and match.start() < closed_end:
                overlaps = True
                break
        if not overlaps:
            all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    # Sort by position and parse each match
    all_matches.sort(key=lambda x: x[0])
    for _, _, json_str in all_matches:
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "name" in data:
                tool_calls.append(ToolCall(
                    name=data["name"],
                    arguments=data.get("arguments", {})
                ))
        except json.JSONDecodeError:
            continue
    
    return tool_calls


def _parse_raw_json(response: str) -> List[ToolCall]:
    """Parse raw JSON tool calls."""
    tool_calls = []
    
    # Find JSON objects
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.findall(json_pattern, response):
        try:
            data = json.loads(match)
            if isinstance(data, dict):
                # Check for tool call structure
                name = data.get("name") or data.get("function")
                if name:
                    args = data.get("arguments") or data.get("parameters") or {}
                    tool_calls.append(ToolCall(name=name, arguments=args))
        except json.JSONDecodeError:
            continue
    
    return tool_calls


def _parse_any_format(response: str) -> List[ToolCall]:
    """Try all parsing formats."""
    # First check for Python code blocks with common tool names
    tool_calls = []
    
    # Look for Python code blocks with list_files calls
    list_files_pattern = r'```(?:python|json)?\s*\n.*?list_files\(([^)]*)\).*?\n```'
    for match in re.finditer(list_files_pattern, response, re.DOTALL):
        args_str = match.group(1).strip()
        arguments = {}
        
        # Parse simple keyword arguments if any
        if args_str:
            arg_pattern = r'(\w+)\s*=\s*([^,]+)'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2).strip()
                
                # Parse value
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():
                    value = int(value)
                
                arguments[key] = value
        
        # Create a tool call for list_files
        tool_calls.append(ToolCall(
            name="list_files",
            arguments=arguments
        ))
    
    # If we found tool calls in code blocks, return them
    if tool_calls:
        return tool_calls
    
    # Otherwise try all standard parsers
    for parser in [_parse_tool_code, _parse_special_token, 
                   _parse_function_call, _parse_xml_wrapped, _parse_raw_json]:
        tool_calls = parser(response)
        if tool_calls:
            return tool_calls
    
    return []


# Formatting functions

def _format_gemma_style(tools: List[ToolDefinition]) -> str:
    """Format for Gemma (tool_code) - clean and pragmatic."""
    tool_list = [t.to_dict() for t in tools]
    
    # Create 2 examples for each tool
    examples = []
    for tool in tools:
        if tool.name == "list_files":
            examples.append(f"""{tool.name} - List files in a directory
Example 1: ```tool_code
list_files(directory_path="docs")
```
Example 2: ```tool_code
list_files(directory_path="src", pattern="*.py", recursive=True)
```""")
        elif tool.name == "read_file":
            examples.append(f"""{tool.name} - Read file contents
Example 1: ```tool_code
read_file(file_path="example.txt")
```
Example 2: ```tool_code
read_file(file_path="large.txt", should_read_entire_file=False, start_line_one_indexed=10, end_line_one_indexed_inclusive=20)
```""")
        elif tool.name == "search_files":
            examples.append(f"""{tool.name} - Search text in files
Example 1: ```tool_code
search_files(search_term="function")
```
Example 2: ```tool_code
search_files(search_term="class", directory_path="src", file_pattern="*.py", case_sensitive=True)
```""")
        elif tool.name == "write_file":
            examples.append(f"""{tool.name} - Write content to file
Example 1: ```tool_code
write_file(file_path="output.txt", content="Hello World")
```
Example 2: ```tool_code
write_file(file_path="log.txt", content="Error occurred", mode="a")
```""")
        else:
            # Generic examples for other tools
            params = list(tool.parameters.get("properties", {}).keys())
            if params:
                # Example 1: minimal required params
                required = tool.parameters.get("required", [])
                if required:
                    min_params = ", ".join([f'{p}="value"' for p in required[:2]])
                    examples.append(f"""{tool.name} - {tool.description}
Example 1: ```tool_code
{tool.name}({min_params})
```""")
                else:
                    examples.append(f"""{tool.name} - {tool.description}
Example 1: ```tool_code
{tool.name}()
```""")
                
                # Example 2: more params
                all_params = ", ".join([f'{p}="value"' for p in params[:3]])
                examples.append(f"""Example 2: ```tool_code
{tool.name}({all_params})
```""")
            else:
                examples.append(f"""{tool.name} - {tool.description}
Example 1: ```tool_code
{tool.name}()
```
Example 2: ```tool_code
{tool.name}()
```""")
    
    examples_text = "\n\n".join(examples)
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{json.dumps(tool_list, indent=2)}

EXAMPLES:
{examples_text}"""


def _format_qwen_style(tools: List[ToolDefinition]) -> str:
    """Format for Qwen (special token) - clean and pragmatic."""
    tool_list = [t.to_dict() for t in tools]
    
    # Create 2 examples for each tool
    examples = []
    for tool in tools:
        if tool.name == "list_files":
            examples.append(f"""{tool.name} - List files in a directory
Example 1: <|tool_call|>{{"name": "list_files", "arguments": {{"directory_path": "docs"}}}}</|tool_call|>
Example 2: <|tool_call|>{{"name": "list_files", "arguments": {{"directory_path": "src", "pattern": "*.py", "recursive": true}}}}</|tool_call|>""")
        elif tool.name == "read_file":
            examples.append(f"""{tool.name} - Read file contents
Example 1: <|tool_call|>{{"name": "read_file", "arguments": {{"file_path": "example.txt"}}}}</|tool_call|>
Example 2: <|tool_call|>{{"name": "read_file", "arguments": {{"file_path": "large.txt", "should_read_entire_file": false, "start_line_one_indexed": 10, "end_line_one_indexed_inclusive": 20}}}}</|tool_call|>""")
        elif tool.name == "search_files":
            examples.append(f"""{tool.name} - Search text in files
Example 1: <|tool_call|>{{"name": "search_files", "arguments": {{"search_term": "function"}}}}</|tool_call|>
Example 2: <|tool_call|>{{"name": "search_files", "arguments": {{"search_term": "class", "directory_path": "src", "file_pattern": "*.py", "case_sensitive": true}}}}</|tool_call|>""")
        elif tool.name == "write_file":
            examples.append(f"""{tool.name} - Write content to file
Example 1: <|tool_call|>{{"name": "write_file", "arguments": {{"file_path": "output.txt", "content": "Hello World"}}}}</|tool_call|>
Example 2: <|tool_call|>{{"name": "write_file", "arguments": {{"file_path": "log.txt", "content": "Error occurred", "mode": "a"}}}}</|tool_call|>""")
        else:
            # Generic examples for other tools
            params = list(tool.parameters.get("properties", {}).keys())
            if params:
                # Example 1: minimal required params
                required = tool.parameters.get("required", [])
                min_args = {param: "value" for param in required[:2]} if required else {}
                # Example 2: more params
                all_args = {param: "value" for param in params[:3]}
                examples.append(f"""{tool.name} - {tool.description}
Example 1: <|tool_call|>{{"name": "{tool.name}", "arguments": {min_args}}}</|tool_call|>
Example 2: <|tool_call|>{{"name": "{tool.name}", "arguments": {all_args}}}</|tool_call|>""")
            else:
                examples.append(f"""{tool.name} - {tool.description}
Example 1: <|tool_call|>{{"name": "{tool.name}", "arguments": {{}}}}</|tool_call|>
Example 2: <|tool_call|>{{"name": "{tool.name}", "arguments": {{}}}}</|tool_call|>""")
    
    examples_text = "\n\n".join(examples)
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{json.dumps(tool_list, indent=2)}

EXAMPLES:
{examples_text}"""


def _format_llama_style(tools: List[ToolDefinition]) -> str:
    """Format for Llama (function_call) - clean and pragmatic."""
    tool_list = [t.to_dict() for t in tools]
    
    # Create 2 examples for each tool
    examples = []
    for tool in tools:
        if tool.name == "list_files":
            examples.append(f"""{tool.name} - List files in a directory
Example 1: <function_call>{{"name": "list_files", "arguments": {{"directory_path": "docs"}}}}</function_call>
Example 2: <function_call>{{"name": "list_files", "arguments": {{"directory_path": "src", "pattern": "*.py", "recursive": true}}}}</function_call>""")
        elif tool.name == "read_file":
            examples.append(f"""{tool.name} - Read file contents
Example 1: <function_call>{{"name": "read_file", "arguments": {{"file_path": "example.txt"}}}}</function_call>
Example 2: <function_call>{{"name": "read_file", "arguments": {{"file_path": "large.txt", "should_read_entire_file": false, "start_line_one_indexed": 10, "end_line_one_indexed_inclusive": 20}}}}</function_call>""")
        elif tool.name == "search_files":
            examples.append(f"""{tool.name} - Search text in files
Example 1: <function_call>{{"name": "search_files", "arguments": {{"search_term": "function"}}}}</function_call>
Example 2: <function_call>{{"name": "search_files", "arguments": {{"search_term": "class", "directory_path": "src", "file_pattern": "*.py", "case_sensitive": true}}}}</function_call>""")
        elif tool.name == "write_file":
            examples.append(f"""{tool.name} - Write content to file
Example 1: <function_call>{{"name": "write_file", "arguments": {{"file_path": "output.txt", "content": "Hello World"}}}}</function_call>
Example 2: <function_call>{{"name": "write_file", "arguments": {{"file_path": "log.txt", "content": "Error occurred", "mode": "a"}}}}</function_call>""")
        else:
            # Generic examples for other tools
            params = list(tool.parameters.get("properties", {}).keys())
            if params:
                # Example 1: minimal required params
                required = tool.parameters.get("required", [])
                min_args = {param: "value" for param in required[:2]} if required else {}
                # Example 2: more params
                all_args = {param: "value" for param in params[:3]}
                examples.append(f"""{tool.name} - {tool.description}
Example 1: <function_call>{{"name": "{tool.name}", "arguments": {min_args}}}</function_call>
Example 2: <function_call>{{"name": "{tool.name}", "arguments": {all_args}}}</function_call>""")
            else:
                examples.append(f"""{tool.name} - {tool.description}
Example 1: <function_call>{{"name": "{tool.name}", "arguments": {{}}}}</function_call>
Example 2: <function_call>{{"name": "{tool.name}", "arguments": {{}}}}</function_call>""")
    
    examples_text = "\n\n".join(examples)
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{json.dumps(tool_list, indent=2)}

EXAMPLES:
{examples_text}"""


def _format_xml_style(tools: List[ToolDefinition]) -> str:
    """Format for XML style - clean and pragmatic."""
    tool_list = [t.to_dict() for t in tools]
    
    # Create 2 examples for each tool
    examples = []
    for tool in tools:
        if tool.name == "list_files":
            examples.append(f"""{tool.name} - List files in a directory
Example 1: <tool_call>{{"name": "list_files", "arguments": {{"directory_path": "docs"}}}}</tool_call>
Example 2: <tool_call>{{"name": "list_files", "arguments": {{"directory_path": "src", "pattern": "*.py", "recursive": true}}}}</tool_call>""")
        elif tool.name == "read_file":
            examples.append(f"""{tool.name} - Read file contents
Example 1: <tool_call>{{"name": "read_file", "arguments": {{"file_path": "example.txt"}}}}</tool_call>
Example 2: <tool_call>{{"name": "read_file", "arguments": {{"file_path": "large.txt", "should_read_entire_file": false, "start_line_one_indexed": 10, "end_line_one_indexed_inclusive": 20}}}}</tool_call>""")
        elif tool.name == "search_files":
            examples.append(f"""{tool.name} - Search text in files
Example 1: <tool_call>{{"name": "search_files", "arguments": {{"search_term": "function"}}}}</tool_call>
Example 2: <tool_call>{{"name": "search_files", "arguments": {{"search_term": "class", "directory_path": "src", "file_pattern": "*.py", "case_sensitive": true}}}}</tool_call>""")
        elif tool.name == "write_file":
            examples.append(f"""{tool.name} - Write content to file
Example 1: <tool_call>{{"name": "write_file", "arguments": {{"file_path": "output.txt", "content": "Hello World"}}}}</tool_call>
Example 2: <tool_call>{{"name": "write_file", "arguments": {{"file_path": "log.txt", "content": "Error occurred", "mode": "a"}}}}</tool_call>""")
        else:
            # Generic examples for other tools
            params = list(tool.parameters.get("properties", {}).keys())
            if params:
                # Example 1: minimal required params
                required = tool.parameters.get("required", [])
                min_args = {param: "value" for param in required[:2]} if required else {}
                # Example 2: more params
                all_args = {param: "value" for param in params[:3]}
                examples.append(f"""{tool.name} - {tool.description}
Example 1: <tool_call>{{"name": "{tool.name}", "arguments": {min_args}}}</tool_call>
Example 2: <tool_call>{{"name": "{tool.name}", "arguments": {all_args}}}</tool_call>""")
            else:
                examples.append(f"""{tool.name} - {tool.description}
Example 1: <tool_call>{{"name": "{tool.name}", "arguments": {{}}}}</tool_call>
Example 2: <tool_call>{{"name": "{tool.name}", "arguments": {{}}}}</tool_call>""")
    
    examples_text = "\n\n".join(examples)
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{json.dumps(tool_list, indent=2)}

EXAMPLES:
{examples_text}"""


def _format_generic_style(tools: List[ToolDefinition]) -> str:
    """Generic format - clean and pragmatic."""
    tool_list = [t.to_dict() for t in tools]
    
    # Create 2 examples for each tool
    examples = []
    for tool in tools:
        if tool.name == "list_files":
            examples.append(f"""{tool.name} - List files in a directory
Example 1: {{"name": "list_files", "arguments": {{"directory_path": "docs"}}}}
Example 2: {{"name": "list_files", "arguments": {{"directory_path": "src", "pattern": "*.py", "recursive": true}}}}""")
        elif tool.name == "read_file":
            examples.append(f"""{tool.name} - Read file contents
Example 1: {{"name": "read_file", "arguments": {{"file_path": "example.txt"}}}}
Example 2: {{"name": "read_file", "arguments": {{"file_path": "large.txt", "should_read_entire_file": false, "start_line_one_indexed": 10, "end_line_one_indexed_inclusive": 20}}}}""")
        elif tool.name == "search_files":
            examples.append(f"""{tool.name} - Search text in files
Example 1: {{"name": "search_files", "arguments": {{"search_term": "function"}}}}
Example 2: {{"name": "search_files", "arguments": {{"search_term": "class", "directory_path": "src", "file_pattern": "*.py", "case_sensitive": true}}}}""")
        elif tool.name == "write_file":
            examples.append(f"""{tool.name} - Write content to file
Example 1: {{"name": "write_file", "arguments": {{"file_path": "output.txt", "content": "Hello World"}}}}
Example 2: {{"name": "write_file", "arguments": {{"file_path": "log.txt", "content": "Error occurred", "mode": "a"}}}}""")
        else:
            # Generic examples for other tools
            params = list(tool.parameters.get("properties", {}).keys())
            if params:
                # Example 1: minimal required params
                required = tool.parameters.get("required", [])
                min_args = {param: "value" for param in required[:2]} if required else {}
                # Example 2: more params
                all_args = {param: "value" for param in params[:3]}
                examples.append(f"""{tool.name} - {tool.description}
Example 1: {{"name": "{tool.name}", "arguments": {min_args}}}
Example 2: {{"name": "{tool.name}", "arguments": {all_args}}}""")
            else:
                examples.append(f"""{tool.name} - {tool.description}
Example 1: {{"name": "{tool.name}", "arguments": {{}}}}
Example 2: {{"name": "{tool.name}", "arguments": {{}}}}""")
    
    examples_text = "\n\n".join(examples)
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{json.dumps(tool_list, indent=2)}

EXAMPLES:
{examples_text}"""