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
        return "```tool_code" in response
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
            "<|tool_call|>" in response,
            "<function_call" in response,
            "<tool_call>" in response,
            _has_json_tool_pattern(response)
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
        r'"name":\s*"[^"]+.*"arguments":\s*\{'
    ]
    return any(re.search(p, text) for p in patterns)


# Parsing functions

def _parse_tool_code(response: str) -> List[ToolCall]:
    """Parse ```tool_code format."""
    tool_calls = []
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
    
    return tool_calls


def _parse_special_token(response: str) -> List[ToolCall]:
    """Parse <|tool_call|> format with robust fallback."""
    tool_calls = []
    
    # First, find all tool call positions to avoid duplicates from overlapping patterns
    all_matches = []
    
    # Strategy 1: Look for properly closed tags
    pattern_with_close = r'<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>'
    for match in re.finditer(pattern_with_close, response, re.DOTALL):
        all_matches.append((match.start(), match.end(), match.group(1).strip()))
    
    # Strategy 2: Look for opening tags followed by valid JSON (no closing tag)
    pattern_no_close = r'<\|tool_call\|>\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    for match in re.finditer(pattern_no_close, response, re.DOTALL):
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
    for parser in [_parse_tool_code, _parse_special_token, 
                   _parse_function_call, _parse_xml_wrapped, _parse_raw_json]:
        tool_calls = parser(response)
        if tool_calls:
            return tool_calls
    return []


# Formatting functions

def _format_gemma_style(tools: List[ToolDefinition]) -> str:
    """Format for Gemma (tool_code)."""
    tool_defs = []
    
    for tool in tools:
        params = []
        if "properties" in tool.parameters:
            for name, info in tool.parameters["properties"].items():
                type_hint = {"string": "str", "integer": "int", 
                            "number": "float", "boolean": "bool"}.get(info.get("type"), "str")
                params.append(f"{name}: {type_hint}")
        
        signature = f"{tool.name}({', '.join(params)})" if params else f"{tool.name}()"
        tool_defs.append(f"def {signature}:\n    \"\"\"{tool.description}\"\"\"")
    
    tools_text = "\n\n".join(tool_defs)
    
    return f"""You are a helpful AI assistant with tool access. When using tools, wrap calls in ```tool_code blocks.

Available tools:

{tools_text}

Example usage:
```tool_code
{tools[0].name}(param="value")
```"""


def _format_qwen_style(tools: List[ToolDefinition]) -> str:
    """Format for Qwen (special token)."""
    tool_list = [t.to_dict() for t in tools]
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{json.dumps(tool_list, indent=2)}

To use a tool:
<|tool_call|>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}"""


def _format_llama_style(tools: List[ToolDefinition]) -> str:
    """Format for Llama (function_call)."""
    tool_descriptions = []
    
    for tool in tools:
        params = []
        if "properties" in tool.parameters:
            for name, info in tool.parameters["properties"].items():
                params.append(f"- {name} ({info.get('type', 'string')})")
        
        params_text = "\n  ".join(params) if params else "No parameters"
        tool_descriptions.append(f"• {tool.name}: {tool.description}\n  Parameters:\n  {params_text}")
    
    tools_text = "\n\n".join(tool_descriptions)
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{tools_text}

To use a tool:
<function_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</function_call>"""


def _format_xml_style(tools: List[ToolDefinition]) -> str:
    """Format for XML style."""
    tool_list = []
    
    for tool in tools:
        param_names = list(tool.parameters.get("properties", {}).keys())
        params = f"({', '.join(param_names)})" if param_names else "()"
        tool_list.append(f"- {tool.name}{params}: {tool.description}")
    
    tools_text = "\n".join(tool_list)
    
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{tools_text}

To use a tool:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>"""


def _format_generic_style(tools: List[ToolDefinition]) -> str:
    """Generic format."""
    tool_list = [f"- {t.name}: {t.description}" for t in tools]
    return f"""You are a helpful AI assistant with tool access.

Available tools:
{chr(10).join(tool_list)}

When you need to use a tool, format your request as JSON with 'name' and 'arguments' fields."""