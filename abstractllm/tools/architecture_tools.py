"""
Architecture-based tool calling support for AbstractLLM.

This module provides tool calling detection and parsing based on model architecture,
using the unified architecture detection system.
"""

import json
import re
import logging
from typing import List, Optional, Dict, Any, Union

from abstractllm.tools.types import ToolCall, ToolCallRequest
from abstractllm.architectures.unified_detection import (
    get_tool_call_format, get_model_config, ToolCallFormat
)

logger = logging.getLogger(__name__)


def detect_tool_calls(response: str, model_name: Optional[str] = None) -> bool:
    """
    Detect if the response contains tool calls using architecture-based detection.
    Now uses flexible multi-pattern detection for better empirical accuracy.
    
    Args:
        response: The model response to check
        model_name: Optional model name for architecture-specific detection
        
    Returns:
        True if tool calls are detected, False otherwise
    """
    if not response or not response.strip():
        return False
    
    # Get model's expected format
    expected_format = None
    if model_name:
        try:
            from abstractllm.architectures.unified_detection import get_tool_call_format
            expected_format = get_tool_call_format(model_name)
        except ImportError:
            pass
    
    # Flexible detection: Try expected format first, then fall back to common patterns
    detection_order = []
    
    if expected_format:
        detection_order.append(expected_format.value)
    
    # Add common patterns that weren't already tested
    common_patterns = ["tool_code", "raw_json", "special_token", "xml_wrapped", "function_call", 
                      "gemma_python", "gemma_json", "markdown_code"]
    for pattern in common_patterns:
        if not expected_format or pattern != expected_format.value:
            detection_order.append(pattern)
    
    # Test each pattern quickly
    for tool_format in detection_order:
        if _has_format_specific_tool_calls(response, tool_format):
            return True
    
    return False


def parse_tool_calls(response: str, model_name: Optional[str] = None) -> List[ToolCall]:
    """
    Parse tool calls from response using architecture-based parsing.
    Now uses flexible multi-pattern parsing for better empirical accuracy.
    
    Args:
        response: The model response containing tool calls
        model_name: Optional model name for architecture-specific parsing
        
    Returns:
        List of parsed ToolCall objects
    """
    if not response or not response.strip():
        return []
    
    # Get model's expected format
    expected_format = None
    if model_name:
        try:
            from abstractllm.architectures.unified_detection import get_tool_call_format
            expected_format = get_tool_call_format(model_name)
        except ImportError:
            pass
    
    # Flexible parsing: Try expected format first, then fall back to common patterns
    parsing_order = []
    
    if expected_format:
        parsing_order.append(expected_format.value)
    
    # Add common patterns that weren't already tested
    common_patterns = ["tool_code", "raw_json", "special_token", "xml_wrapped", "function_call", 
                      "gemma_python", "gemma_json", "markdown_code"]
    for pattern in common_patterns:
        if not expected_format or pattern != expected_format.value:
            parsing_order.append(pattern)
    
    # Try each format until one succeeds
    for tool_format in parsing_order:
        try:
            tool_calls = _parse_format_specific_tool_calls(response, tool_format)
            if tool_calls:  # If we found tool calls, return them
                return tool_calls
        except Exception as e:
            # Continue to next format if parsing fails
            continue
    
    return []


def format_tools_for_prompt(tools: List[Dict[str, Any]], model_name: Optional[str] = None) -> str:
    """
    Format tools into a system prompt using architecture-based tool calling formats.
    
    This function uses the modular prompt system to generate optimized prompts
    based on the number of tools and the target model architecture.
    
    Args:
        tools: List of tool definitions
        model_name: Model name to detect architecture and tool format
        
    Returns:
        Formatted system prompt for tools
    """
    if not tools:
        from abstractllm.tools.modular_prompts import generate_base_agent_prompt
        return generate_base_agent_prompt()
    
    # Import here to avoid circular imports
    from abstractllm.tools.modular_prompts import generate_tool_prompt
    from abstractllm.architectures.detection import detect_architecture
    
    # Detect model architecture
    if model_name:
        architecture = detect_architecture(model_name)
        logger.debug(f"Detected architecture '{architecture}' for model '{model_name}'")
    else:
        architecture = None
        logger.debug("No model name provided, using generic format")
    
    # Map architecture to tool format
    tool_format_map = {
        "gemma": "TOOL_CODE",      # Uses ```tool_code format with Python syntax
        "qwen": "SPECIAL_TOKEN",   # Uses <|tool_call|> format
        "llama": "FUNCTION_CALL",  # Uses <function_call> format
        "phi": "XML_WRAPPED",      # Uses <tool_call></tool_call> format
        "mistral": "FUNCTION_CALL", # Similar to Llama
        "deepseek": "SPECIAL_TOKEN" # Similar to Qwen
    }
    
    # Get tool format
    if architecture and architecture in tool_format_map:
        tool_format = tool_format_map[architecture]
        logger.debug(f"Using {tool_format} format for {architecture} architecture")
    else:
        tool_format = "GENERIC"
        logger.debug("Using generic tool format for unknown architecture")
    
    # Generate the prompt using modular system
    return generate_tool_prompt(tools, tool_format, model_name)


# Format-specific detection functions

def _has_format_specific_tool_calls(response: str, tool_format: str) -> bool:
    """Check if response has tool calls in the specified format."""
    if tool_format == ToolCallFormat.XML_WRAPPED.value:
        return _has_xml_wrapped_tool_calls(response)
    elif tool_format == ToolCallFormat.FUNCTION_CALL.value:
        return _has_function_call_tool_calls(response)
    elif tool_format == ToolCallFormat.SPECIAL_TOKEN.value:
        return _has_special_token_tool_calls(response)
    elif tool_format == ToolCallFormat.MARKDOWN_CODE.value:
        return _has_markdown_code_tool_calls(response)
    elif tool_format == ToolCallFormat.GEMMA_PYTHON.value:
        return _has_gemma_python_tool_calls(response)
    elif tool_format == ToolCallFormat.GEMMA_JSON.value:
        return _has_gemma_json_tool_calls(response)
    elif tool_format == ToolCallFormat.RAW_JSON.value:
        return _has_raw_json_tool_calls(response)
    elif tool_format == ToolCallFormat.TOOL_CODE.value:
        return _has_tool_code_tool_calls(response)
    else:
        # Fallback: check for any common patterns
        return any([
            _has_xml_wrapped_tool_calls(response),
            _has_function_call_tool_calls(response),
            _has_special_token_tool_calls(response),
            _has_gemma_python_tool_calls(response),
            _has_gemma_json_tool_calls(response),
            _has_raw_json_tool_calls(response),
            _has_tool_code_tool_calls(response)
        ])


def _has_xml_wrapped_tool_calls(response: str) -> bool:
    """Check for XML-wrapped tool calls: <tool_call>...</tool_call>"""
    return "<tool_call>" in response and "</tool_call>" in response


def _has_function_call_tool_calls(response: str) -> bool:
    """Check for function call format: <function_call ...> or raw JSON function calls"""
    # Check for proper function_call wrapper
    if "<function_call" in response:
        return True
    
    # Also check for raw JSON that looks like a function call (Llama fallback)
    # Look for JSON with "name" and "arguments" fields in text context
    function_call_pattern = r'\{\s*["\']name["\']\s*:\s*["\'][^"\']+["\']\s*,\s*["\']arguments["\']\s*:\s*\{[^}]*\}\s*\}'
    if re.search(function_call_pattern, response):
        return True
        
    return False


def _has_special_token_tool_calls(response: str) -> bool:
    """Check for special token format: <|tool_call|>[...]"""
    return "<|tool_call|>" in response


def _has_markdown_code_tool_calls(response: str) -> bool:
    """Check for markdown code block format: ```tool_call\n...```"""
    return "```tool_call" in response


def _has_raw_json_tool_calls(response: str) -> bool:
    """
    Enhanced detection for raw JSON tool calls, especially for Llama models.
    Looks for JSON objects that contain function call patterns.
    """
    # Look for common function call patterns in JSON
    patterns = [
        r'\{\s*["\']name["\']\s*:\s*["\'][^"\']+["\']',  # {"name": "function_name"}
        r'\{\s*["\']function["\']\s*:\s*["\'][^"\']+["\']',  # {"function": "function_name"}
        r'\{\s*["\']tool_name["\']\s*:\s*["\'][^"\']+["\']',  # {"tool_name": "function_name"}
        r'\{\s*["\']action["\']\s*:\s*["\'][^"\']+["\']',  # {"action": "function_name"}
    ]
    
    # Check if any pattern matches
    for pattern in patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    
    # Additional check: Look for JSON-like structures with typical tool call keys
    try:
        # Try to find JSON objects in the response
        json_objects = re.findall(r'\{[^{}]*\}', response)
        for json_str in json_objects:
            try:
                obj = json.loads(json_str)
                if isinstance(obj, dict):
                    # Check for common tool call keys
                    tool_call_keys = ['name', 'function', 'tool_name', 'action', 'parameters', 'arguments', 'args']
                    if any(key in obj for key in tool_call_keys):
                        return True
            except:
                continue
    except:
        pass
    
    # Look for multiple JSON objects (tool calls often come in arrays or multiple objects)
    json_count = len(re.findall(r'\{[^{}]*\}', response))
    if json_count > 0:
        # Check if response contains keywords that suggest tool calling
        tool_keywords = ['function', 'tool', 'call', 'invoke', 'execute', 'parameters', 'arguments']
        if any(keyword in response.lower() for keyword in tool_keywords):
            return True
    
    return False


def _has_tool_code_tool_calls(response: str) -> bool:
    """Check if response contains tool calls in ```tool_code format (actual Gemma format)."""
    return "```tool_code" in response


def _has_gemma_python_tool_calls(response: str) -> bool:
    """Check for Gemma Python-style tool calls: [func_name(param=value, ...)]"""
    # Look for pattern: [function_name(param=value, ...)]
    import re
    # Match function calls in square brackets with parameters
    pattern = r'\[[\w_]+\([^)]*\)\]'
    return bool(re.search(pattern, response))


def _has_gemma_json_tool_calls(response: str) -> bool:
    """Check for Gemma JSON-style tool calls: {"name": "func_name", "parameters": {...}}"""
    import re
    # Look for JSON objects with "name" and "parameters" fields
    # Make the pattern more flexible to handle whitespace and simple parameters
    pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^}]*\}\s*\}'
    if re.search(pattern, response):
        return True
    
    # Also check for simpler case where parameters might be empty
    simple_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{\s*\}\s*\}'
    return bool(re.search(simple_pattern, response))


# Format-specific parsing functions

def _parse_format_specific_tool_calls(response: str, tool_format: str) -> List[ToolCall]:
    """Parse tool calls in the specified format."""
    if tool_format == ToolCallFormat.XML_WRAPPED.value:
        return _parse_xml_wrapped_tool_calls(response)
    elif tool_format == ToolCallFormat.FUNCTION_CALL.value:
        return _parse_function_call_tool_calls(response)
    elif tool_format == ToolCallFormat.SPECIAL_TOKEN.value:
        return _parse_special_token_tool_calls(response)
    elif tool_format == ToolCallFormat.MARKDOWN_CODE.value:
        return _parse_markdown_code_tool_calls(response)
    elif tool_format == ToolCallFormat.GEMMA_PYTHON.value:
        return _parse_gemma_python_tool_calls(response)
    elif tool_format == ToolCallFormat.GEMMA_JSON.value:
        return _parse_gemma_json_tool_calls(response)
    elif tool_format == ToolCallFormat.RAW_JSON.value:
        return _parse_raw_json_tool_calls(response)
    elif tool_format == ToolCallFormat.TOOL_CODE.value:
        return _parse_tool_code_tool_calls(response)
    else:
        # Fallback: try all parsers
        for parser in [
            _parse_xml_wrapped_tool_calls,
            _parse_function_call_tool_calls, 
            _parse_special_token_tool_calls,
            _parse_gemma_python_tool_calls,
            _parse_gemma_json_tool_calls,
            _parse_raw_json_tool_calls,
            _parse_tool_code_tool_calls
        ]:
            tool_calls = parser(response)
            if tool_calls:
                return tool_calls
        return []


def _parse_xml_wrapped_tool_calls(response: str) -> List[ToolCall]:
    """Parse XML-wrapped tool calls: <tool_call>{"name": "...", "arguments": {...}}</tool_call>"""
    tool_calls = []
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, response, re.DOTALL)
    
    for i, match in enumerate(matches):
        try:
            tool_data = _robust_json_parse(match.strip())
            if isinstance(tool_data, dict) and "name" in tool_data:
                tool_call = ToolCall(
                    id=f"call_{i}",
                    name=tool_data.get("name"),
                    arguments=tool_data.get("arguments", {})
                )
                tool_calls.append(tool_call)
                logger.debug(f"Successfully parsed XML-wrapped tool call: {tool_data}")
        except Exception as e:
            logger.warning(f"Failed to parse XML-wrapped tool call: {match}. Error: {e}")
            continue
    
    return tool_calls


def _parse_function_call_tool_calls(response: str) -> List[ToolCall]:
    """
    Parse tool calls from <function_call> format.
    
    Handles multiple formats:
    - <function_call name="func" arguments="{...}"/>
    - <function_call>{\"name\": \"func\", \"arguments\": {...}}</function_call>
    - <function_call name="func" arguments={...}/>  (no quotes around JSON)
    
    Returns:
        List of ToolCall objects
    """
    tool_calls = []
    
    # Pattern 1: Self-closing tags with attributes (more flexible)
    # <function_call name="execute_command" arguments="{\"command\": \"ls -la\", \"working_directory\": \".\"}"/>
    # Also handles: <function_call name="func" arguments={...}/>
    self_closing_pattern = r'<function_call\s+name\s*=\s*["\']([^"\']+)["\']\s+arguments\s*=\s*(\{[^}]*\}|\{.*?\})\s*/>'
    self_closing_matches = re.findall(self_closing_pattern, response, re.DOTALL)
    
    for i, (func_name, args_str) in enumerate(self_closing_matches):
        try:
            # Clean up the arguments string - remove extra quotes if present
            args_str = args_str.strip()
            if args_str.startswith('"') and args_str.endswith('"'):
                args_str = args_str[1:-1]  # Remove surrounding quotes
            
            # Try to parse as JSON
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to clean it up
                # Handle common issues like unquoted keys
                args_str = re.sub(r'(\w+):', r'"\1":', args_str)  # Quote unquoted keys
                arguments = json.loads(args_str)
            
            tool_call = ToolCall(
                id=f"call_{i}",
                name=func_name,
                arguments=arguments
            )
            tool_calls.append(tool_call)
        except (json.JSONDecodeError, ValueError) as e:
            # If parsing fails, skip this tool call
            continue
    
    # Pattern 2: Container tags with JSON content
    # <function_call>{"name": "func", "arguments": {...}}</function_call>
    container_pattern = r'<function_call\s*>(.*?)</function_call>'
    container_matches = re.findall(container_pattern, response, re.DOTALL)
    
    for i, json_str in enumerate(container_matches):
        try:
            json_str = json_str.strip()
            data = json.loads(json_str)
            
            if isinstance(data, dict) and "name" in data:
                arguments = data.get("arguments", {})
                
                tool_call = ToolCall(
                    id=f"call_container_{i}",
                    name=data["name"],
                    arguments=arguments
                )
                tool_calls.append(tool_call)
        except (json.JSONDecodeError, ValueError) as e:
            # If parsing fails, skip this tool call
            continue
    
    return tool_calls


def _parse_special_token_tool_calls(response: str) -> List[ToolCall]:
    """Parse special token format: <|tool_call|>[{"name": "...", "arguments": {...}}] or <|tool_call|>{"name": "...", "arguments": {...}}"""
    tool_calls = []
    
    # Pattern 1: With square brackets (array format)
    pattern_array = r'<\|tool_call\|>\[(.*?)\]'
    matches_array = re.findall(pattern_array, response, re.DOTALL)
    
    for i, match in enumerate(matches_array):
        try:
            # This format expects an array of tool calls
            tool_data_list = json.loads(f"[{match}]")
            for j, tool_data in enumerate(tool_data_list):
                if isinstance(tool_data, dict) and "name" in tool_data:
                    tool_call = ToolCall(
                        id=f"call_{i}_{j}",
                        name=tool_data.get("name"),
                        arguments=tool_data.get("arguments", {})
                    )
                    tool_calls.append(tool_call)
                    logger.debug(f"Successfully parsed special token tool call (array): {tool_data}")
        except Exception as e:
            logger.warning(f"Failed to parse special token tool call (array): {match}. Error: {e}")
            continue
    
    # Pattern 2: Without square brackets (single object format) - for Qwen models
    # Use a more robust pattern that captures complete JSON objects
    pattern_single = r'<\|tool_call\|>\s*\n?\s*(\{[^{}]*\{[^{}]*\}[^{}]*\})'
    matches_single = re.findall(pattern_single, response, re.DOTALL)
    
    for i, match in enumerate(matches_single):
        try:
            # This format expects a single tool call object - use robust JSON parser
            tool_data = _robust_json_parse(match)
            if isinstance(tool_data, dict) and "name" in tool_data:
                tool_call = ToolCall(
                    id=f"call_single_{i}",
                    name=tool_data.get("name"),
                    arguments=tool_data.get("arguments", {})
                )
                tool_calls.append(tool_call)
                logger.debug(f"Successfully parsed special token tool call (single): {tool_data}")
        except Exception as e:
            logger.warning(f"Failed to parse special token tool call (single): {match}. Error: {e}")
            continue
    
    return tool_calls


def _parse_markdown_code_tool_calls(response: str) -> List[ToolCall]:
    """Parse markdown code block format: ```tool_call\nfunction_name(...)\n```"""
    tool_calls = []
    
    # Extract tool_call code blocks
    pattern = r'```tool_call\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    call_id = 0
    for match in matches:
        # Parse Python-style function call
        func_pattern = r'(\w+)\((.*?)\)'
        func_match = re.search(func_pattern, match.strip())
        
        if func_match:
            func_name = func_match.group(1)
            args_str = func_match.group(2)
            
            # Parse arguments (simplified - handles basic cases)
            arguments = {}
            if args_str.strip():
                # Handle simple parameter parsing
                for arg in args_str.split(','):
                    arg = arg.strip()
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        key = key.strip().strip('"\'')
                        value = value.strip().strip('"\'')
                        arguments[key] = value
                    else:
                        # Positional argument
                        arguments[f"arg_{len(arguments)}"] = arg.strip().strip('"\'')
            
            tool_call = ToolCall(
                id=f"call_{call_id}",
                name=func_name,
                arguments=arguments
            )
            tool_calls.append(tool_call)
            call_id += 1
            logger.debug(f"Successfully parsed markdown tool call: {func_name}")
    
    return tool_calls


def _parse_gemma_python_tool_calls(response: str) -> List[ToolCall]:
    """Parse Gemma Python-style tool calls: [func_name(param=value, ...)]"""
    tool_calls = []
    
    # Find function calls in square brackets
    pattern = r'\[(\w+)\(([^)]*)\)\]'
    matches = re.findall(pattern, response)
    
    call_id = 0
    for func_name, args_str in matches:
        # Parse arguments
        arguments = {}
        if args_str.strip():
            # Split by comma, but be careful with nested structures
            arg_parts = []
            current_part = ""
            paren_depth = 0
            quote_char = None
            
            for char in args_str:
                if quote_char:
                    current_part += char
                    if char == quote_char and (len(current_part) == 1 or current_part[-2] != '\\'):
                        quote_char = None
                elif char in ['"', "'"]:
                    quote_char = char
                    current_part += char
                elif char == '(':
                    paren_depth += 1
                    current_part += char
                elif char == ')':
                    paren_depth -= 1
                    current_part += char
                elif char == ',' and paren_depth == 0:
                    arg_parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            
            if current_part.strip():
                arg_parts.append(current_part.strip())
            
            # Parse each argument
            for arg in arg_parts:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to evaluate simple values
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]  # Remove quotes
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]  # Remove quotes
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    
                    arguments[key] = value
                else:
                    # Positional argument
                    arguments[f"arg_{len(arguments)}"] = arg.strip().strip('"\'')
        
        tool_call = ToolCall(
            id=f"call_{call_id}",
            name=func_name,
            arguments=arguments
        )
        tool_calls.append(tool_call)
        call_id += 1
        logger.debug(f"Successfully parsed Gemma Python tool call: {func_name}")
    
    return tool_calls


def _parse_gemma_json_tool_calls(response: str) -> List[ToolCall]:
    """Parse Gemma JSON-style tool calls: {"name": "func_name", "parameters": {...}}"""
    tool_calls = []
    
    # Find JSON objects with name and parameters
    pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{[^}]*\})\s*\}'
    matches = re.findall(pattern, response)
    
    call_id = 0
    for func_name, params_json in matches:
        try:
            # Parse the parameters JSON
            parameters = _robust_json_parse(params_json)
            
            tool_call = ToolCall(
                id=f"call_{call_id}",
                name=func_name,
                arguments=parameters
            )
            tool_calls.append(tool_call)
            call_id += 1
            logger.debug(f"Successfully parsed Gemma JSON tool call: {func_name}")
        except Exception as e:
            logger.warning(f"Failed to parse Gemma JSON tool call parameters: {params_json}. Error: {e}")
            continue
    
    return tool_calls


def _parse_raw_json_tool_calls(response: str) -> List[ToolCall]:
    """
    Enhanced parsing for raw JSON tool calls, especially for Llama models.
    Handles various JSON formats and extracts tool calls robustly.
    """
    tool_calls = []
    
    # Strategy 1: Look for complete JSON objects with tool call patterns
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in json_matches:
        try:
            parsed = _robust_json_parse(match)
            if isinstance(parsed, dict):
                tool_call = _extract_tool_call_from_dict(parsed)
                if tool_call:
                    tool_calls.append(tool_call)
        except Exception:
            continue
    
    # Strategy 2: Line-by-line JSON parsing (for simple single-line JSON)
    if not tool_calls:
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    parsed = _robust_json_parse(line)
                    if isinstance(parsed, dict):
                        tool_call = _extract_tool_call_from_dict(parsed)
                        if tool_call:
                            tool_calls.append(tool_call)
                except Exception:
                    continue
    
    # Strategy 3: Try parsing the entire response as JSON
    if not tool_calls:
        response_stripped = response.strip()
        if response_stripped.startswith('{') and response_stripped.endswith('}'):
            try:
                parsed = _robust_json_parse(response_stripped)
                if isinstance(parsed, dict):
                    tool_call = _extract_tool_call_from_dict(parsed)
                    if tool_call:
                        tool_calls.append(tool_call)
            except Exception:
                pass
        # Also try parsing as a JSON array
        elif response_stripped.startswith('[') and response_stripped.endswith(']'):
            try:
                parsed = _robust_json_parse(response_stripped)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            tool_call = _extract_tool_call_from_dict(item)
                            if tool_call:
                                tool_calls.append(tool_call)
            except Exception:
                pass
    
    return tool_calls


def _extract_tool_call_from_dict(data: dict) -> Optional[ToolCall]:
    """
    Extract a ToolCall from a dictionary, handling various key names and formats.
    
    Args:
        data: Dictionary that might contain tool call information
        
    Returns:
        ToolCall object if extraction successful, None otherwise
    """
    # Look for function name in various possible keys
    function_name = None
    for name_key in ['name', 'function', 'tool_name', 'action', 'function_name']:
        if name_key in data:
            function_name = data[name_key]
            break
    
    if not function_name:
        return None
    
    # Look for arguments/parameters in various possible keys
    arguments = {}
    for args_key in ['arguments', 'parameters', 'args', 'params', 'inputs']:
        if args_key in data:
            args_value = data[args_key]
            if isinstance(args_value, dict):
                arguments = args_value
            elif isinstance(args_value, str):
                # Try to parse as JSON
                try:
                    arguments = json.loads(args_value)
                except:
                    # If parsing fails, treat as single string argument
                    arguments = {"input": args_value}
            break
    
    # If no explicit arguments found, use remaining keys as arguments
    if not arguments:
        excluded_keys = {'name', 'function', 'tool_name', 'action', 'function_name', 'type'}
        arguments = {k: v for k, v in data.items() if k not in excluded_keys}
    
    return ToolCall(
        name=str(function_name),
        arguments=arguments
    )


def _parse_tool_code_tool_calls(response: str) -> List[ToolCall]:
    """
    Parse tool calls from ```tool_code format (actual Gemma format).
    
    Example format:
    ```tool_code
    search_files(directory=".", pattern="*.py")
    ```
    
    Returns:
        List of ToolCall objects
    """
    tool_calls = []
    
    # Pattern to match ```tool_code blocks
    pattern = r'```tool_code\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    for i, match in enumerate(matches):
        code_block = match.strip()
        
        # Parse function calls from the code block
        # Handle multiple function calls in one block
        lines = [line.strip() for line in code_block.split('\n') if line.strip()]
        
        for line in lines:
            # Parse function call: func_name(param=value, param2=value2)
            func_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)', line)
            if func_match:
                func_name = func_match.group(1)
                args_str = func_match.group(2).strip()
                
                # Parse arguments
                arguments = {}
                if args_str:
                    try:
                        # Use safe evaluation for simple arguments
                        # This handles: param="value", param=123, param=True, etc.
                        args_dict = {}
                        
                        # Split arguments by comma, but be careful of commas in strings
                        current_arg = ""
                        paren_level = 0
                        quote_char = None
                        
                        for char in args_str + ",":  # Add comma to process last arg
                            if char in ['"', "'"] and quote_char is None:
                                quote_char = char
                                current_arg += char
                            elif char == quote_char:
                                quote_char = None
                                current_arg += char
                            elif quote_char is not None:
                                current_arg += char
                            elif char in ['(', '[', '{']:
                                paren_level += 1
                                current_arg += char
                            elif char in [')', ']', '}']:
                                paren_level -= 1
                                current_arg += char
                            elif char == ',' and paren_level == 0 and quote_char is None:
                                # Process the argument
                                if current_arg.strip():
                                    if '=' in current_arg:
                                        key, value = current_arg.split('=', 1)
                                        key = key.strip()
                                        value = value.strip()
                                        
                                        # Parse the value
                                        try:
                                            # Try to evaluate as literal
                                            if value.startswith('"') and value.endswith('"'):
                                                parsed_value = value[1:-1]  # Remove quotes
                                            elif value.startswith("'") and value.endswith("'"):
                                                parsed_value = value[1:-1]  # Remove quotes
                                            elif value.lower() in ['true', 'false']:
                                                parsed_value = value.lower() == 'true'
                                            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                                                parsed_value = int(value)
                                            elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                                                parsed_value = float(value)
                                            else:
                                                parsed_value = value  # Keep as string
                                                
                                            arguments[key] = parsed_value
                                        except:
                                            arguments[key] = value  # Fallback to string
                                
                                current_arg = ""
                            else:
                                current_arg += char
                        
                    except Exception as e:
                        # If parsing fails, treat as simple string arguments
                        arguments = {"raw_args": args_str}
                
                # Create ToolCall
                tool_call = ToolCall(
                    id=f"tool_code_{i}_{len(tool_calls)}",
                    name=func_name,
                    arguments=arguments
                )
                tool_calls.append(tool_call)
    
    return tool_calls


def _robust_json_parse(json_str: str) -> dict:
    """Robustly parse JSON with multiple fallback strategies."""
    
    def clean_json_string(s: str) -> str:
        """Clean up JSON string for parsing."""
        # Remove leading/trailing whitespace
        s = s.strip()
        
        # Remove markdown code block markers if present
        s = re.sub(r'^```json\s*', '', s)
        s = re.sub(r'^```\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
        
        # Fix common JSON issues
        # Replace single quotes with double quotes (but not inside strings)
        s = re.sub(r"'([^']*)':", r'"\1":', s)  # Keys with single quotes
        s = re.sub(r":\s*'([^']*)'", r': "\1"', s)  # Values with single quotes
        
        # Fix trailing commas
        s = re.sub(r',\s*}', '}', s)
        s = re.sub(r',\s*]', ']', s)
        
        # Fix unquoted keys
        s = re.sub(r'(\w+):', r'"\1":', s)
        
        return s
    
    # Try original string first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Try cleaned string
    try:
        cleaned = clean_json_string(json_str)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try extracting just the JSON part
    try:
        # Look for content between first { and last }
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_part = json_str[start:end+1]
            return json.loads(clean_json_string(json_part))
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try eval as last resort (dangerous but sometimes works)
    try:
        # Replace common Python literals that aren't valid JSON
        eval_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        result = eval(eval_str)
        if isinstance(result, dict):
            return result
    except (SyntaxError, NameError, ValueError):
        pass
    
    # If all else fails, raise an exception
    raise json.JSONDecodeError(f"Could not parse JSON: {json_str}", json_str, 0)


# Tool formatting functions for different architectures

def _format_tools_gemma_python(tools: List[Dict[str, Any]]) -> str:
    """Format tools for Gemma Python-style calling based on official documentation."""
    
    # Function calling setup (from Gemma docs)
    setup = """You have access to functions. If you decide to invoke any of the function(s),
you MUST put it in the format of
[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]

You SHOULD NOT include any other text in the response if you call a function"""
    
    # Function definitions in JSON schema format (from Gemma docs)
    function_definitions = []
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            function_definitions.append(func)
        else:
            # Convert to function format
            function_definitions.append({
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {})
            })
    
    # Format as JSON array
    import json
    definitions_json = json.dumps(function_definitions, indent=2)
    
    return f"{setup}\n{definitions_json}"


def _format_tools_gemma_json(tools: List[Dict[str, Any]]) -> str:
    """Format tools for Gemma JSON-style calling based on official documentation."""
    
    # Function calling setup (from Gemma docs)
    setup = """You have access to functions. If you decide to invoke any of the function(s),
you MUST put it in the format of
{"name": function name, "parameters": dictionary of argument name and its value}

You SHOULD NOT include any other text in the response if you call a function"""
    
    # Function definitions in JSON schema format (from Gemma docs)
    function_definitions = []
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            function_definitions.append(func)
        else:
            # Convert to function format
            function_definitions.append({
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {})
            })
    
    # Format as JSON array
    import json
    definitions_json = json.dumps(function_definitions, indent=2)
    
    return f"{setup}\n{definitions_json}"


def _format_tools_special_token(tools: List[Dict[str, Any]]) -> str:
    """Format tools for Qwen/DeepSeek special token format."""
    tool_descriptions = []
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        param_names = []
        if isinstance(parameters, dict) and "properties" in parameters:
            param_names = list(parameters["properties"].keys())
        
        param_str = f"({', '.join(param_names)})" if param_names else "()"
        tool_descriptions.append(f"- {name}{param_str}: {description}")
    
    format_example = '<|tool_call|>[{"name": "tool_name", "arguments": {"param_name": "value"}}]'
    
    return f"""You have access to these tools:
{chr(10).join(tool_descriptions)}

To use a tool: {format_example}

When following multi-step procedures:
1. Read the instructions first 
2. Execute each step that requires a tool call
3. Continue to the next step based on the results
4. Complete the entire procedure unless instructed otherwise

You are an action-taking agent, not just an advisor."""


def _format_tools_function_call(tools: List[Dict[str, Any]]) -> str:
    """Format tools for Llama function call format."""
    tool_descriptions = []
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        param_names = []
        if isinstance(parameters, dict) and "properties" in parameters:
            param_names = list(parameters["properties"].keys())
        
        param_str = f"({', '.join(param_names)})" if param_names else "(No parameters)"
        tool_descriptions.append(f"• {name}: {description}")
        tool_descriptions.append(f"  Parameters: {param_str}")
    
    return f"""You are a helpful AI assistant with access to tool. You have access to this tool:

{chr(10).join(tool_descriptions)}

When you need to use a tool, use the <function_call> format:
<function_call>
{{"name": "tool_name", "arguments": {{"parameter_name": "value"}}}}
</function_call>

Always use the exact parameter names shown in the tool definitions above.

IMPORTANT: When you decide to use a tool, respond ONLY with the <function_call> block. Do not explain what you're doing - just call the tool directly."""


def _format_tools_xml_wrapped(tools: List[Dict[str, Any]]) -> str:
    """Format tools for XML wrapped format (Mistral/Phi)."""
    tool_descriptions = []
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        param_names = []
        if isinstance(parameters, dict) and "properties" in parameters:
            param_names = list(parameters["properties"].keys())
        
        param_str = f"({', '.join(param_names)})" if param_names else "()"
        tool_descriptions.append(f"- {name}{param_str}: {description}")
    
    format_example = '<tool_call>{"name": "tool_name", "arguments": {"param_name": "value"}}</tool_call>'
    
    return f"""You have access to these tools:
{chr(10).join(tool_descriptions)}

To use a tool: {format_example}

When following multi-step procedures:
1. Read the instructions first 
2. Execute each step that requires a tool call
3. Continue to the next step based on the results
4. Complete the entire procedure unless instructed otherwise

You are an action-taking agent, not just an advisor."""


def _format_tools_markdown_code(tools: List[Dict[str, Any]]) -> str:
    """Format tools for markdown code block format."""
    tool_descriptions = []
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        param_names = []
        if isinstance(parameters, dict) and "properties" in parameters:
            param_names = list(parameters["properties"].keys())
        
        param_str = f"({', '.join(param_names)})" if param_names else "()"
        tool_descriptions.append(f"- {name}{param_str}: {description}")
    
    format_example = '```tool_call\ntool_name("param_value")\n```'
    
    return f"""You have access to these tools:
{chr(10).join(tool_descriptions)}

To use a tool:
{format_example}

When following multi-step procedures:
1. Read the instructions first 
2. Execute each step that requires a tool call
3. Continue to the next step based on the results
4. Complete the entire procedure unless instructed otherwise

You are an action-taking agent, not just an advisor."""


def _format_tools_tool_code(tools: List[Dict[str, Any]]) -> str:
    """
    Format tools for ```tool_code format (actual Gemma format).
    Based on the actual format from https://www.philschmid.de/gemma-function-calling
    """
    formatted_parts = []
    
    # Add the instruction format from the blog post
    formatted_parts.append("At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. The python methods described below are imported and available, you can only use defined methods. The generated code should be readable and efficient. The response to a method will be wrapped in ```tool_output``` use it to call more tools or generate a helpful, friendly response. When using a ```tool_call``` think step by step why and how it should be used.")
    
    formatted_parts.append("\nThe following Python methods are available:\n")
    
    # Format each tool as a Python function definition
    for tool in tools:
        if "function" in tool:
            func_info = tool["function"]
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "")
            parameters = func_info.get("parameters", {})
            
            # Build function signature
            if "properties" in parameters:
                params = []
                required = parameters.get("required", [])
                
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "str")
                    
                    # Map JSON Schema types to Python types
                    if param_type == "string":
                        type_hint = "str"
                    elif param_type == "integer":
                        type_hint = "int"
                    elif param_type == "number":
                        type_hint = "float"
                    elif param_type == "boolean":
                        type_hint = "bool"
                    elif param_type == "array":
                        type_hint = "list"
                    elif param_type == "object":
                        type_hint = "dict"
                    else:
                        type_hint = "str"
                    
                    params.append(f"{param_name}: {type_hint}")
                
                param_str = ", ".join(params)
            else:
                param_str = ""
            
            # Format the function definition
            formatted_parts.append(f"```python")
            formatted_parts.append(f"def {name}({param_str}):")
            formatted_parts.append(f'    """{description}')
            
            # Add parameter descriptions
            if "properties" in parameters:
                formatted_parts.append("")
                formatted_parts.append("    Args:")
                for param_name, param_info in parameters["properties"].items():
                    param_desc = param_info.get("description", "")
                    formatted_parts.append(f"      {param_name}: {param_desc}")
            
            formatted_parts.append('    """')
            formatted_parts.append("```\n")
    
    return "\n".join(formatted_parts)


def create_tool_call_request(response_content: str, model_name: Optional[str] = None) -> Union[str, ToolCallRequest]:
    """
    Create a ToolCallRequest from response content if tool calls are detected.
    
    Args:
        response_content: Raw response content
        model_name: Model name for architecture detection
        
    Returns:
        ToolCallRequest if tool calls detected, otherwise original content
    """
    if detect_tool_calls(response_content, model_name):
        tool_calls = parse_tool_calls(response_content, model_name)
        if tool_calls:
            return ToolCallRequest(
                content=response_content,
                tool_calls=tool_calls
            )
    
    return response_content 