"""
Modular prompt generation for different tool scenarios and model architectures.

This module provides optimized system prompts based on the number of tools available
and the target model architecture.
"""

from typing import List, Dict, Any, Optional


def generate_base_agent_prompt() -> str:
    """Generate a clean base agent prompt for scenarios with no tools."""
    return """You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries."""


def generate_tool_prompt(tools: List[Dict[str, Any]], tool_format: str, model_name: Optional[str] = None) -> str:
    """
    Generate a system prompt optimized for tool usage.
    
    Args:
        tools: List of tool definition dictionaries (can be empty, single, or multiple)
        tool_format: Tool calling format (TOOL_CODE, SPECIAL_TOKEN, etc.)
        model_name: Model name for specific optimizations
        
    Returns:
        Optimized system prompt for tool usage
    """
    if not tools:
        return generate_base_agent_prompt()
    
    if tool_format == "TOOL_CODE":
        return _generate_gemma_tool_prompt(tools)
    elif tool_format == "SPECIAL_TOKEN":
        return _generate_qwen_tool_prompt(tools)
    elif tool_format == "FUNCTION_CALL":
        return _generate_llama_tool_prompt(tools)
    elif tool_format == "XML_WRAPPED":
        return _generate_xml_tool_prompt(tools)
    else:
        # Fallback to generic format
        return _generate_generic_tool_prompt(tools)


def _generate_gemma_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Generate Gemma-optimized tool prompt."""
    tool_definitions = []
    
    for tool in tools:
        if "function" in tool:
            func_info = tool["function"]
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "")
            parameters = func_info.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        # Create function signature
        if isinstance(parameters, dict) and "properties" in parameters:
            props = parameters["properties"]
            required = parameters.get("required", [])
            
            params = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "str")
                
                # Map JSON Schema types to Python types
                type_mapping = {
                    "string": "str", 
                    "integer": "int", 
                    "number": "float",
                    "boolean": "bool", 
                    "array": "list", 
                    "object": "dict"
                }
                python_type = type_mapping.get(param_type, "str")
                
                if param_name in required:
                    params.append(f"{param_name}: {python_type}")
                else:
                    params.append(f"{param_name}: {python_type} = None")
            
            signature = f"{name}({', '.join(params)})"
        else:
            signature = f"{name}()"
        
        tool_def = f"""def {signature}:
    \"\"\"{description}\"\"\"
    pass"""
        tool_definitions.append(tool_def)
    
    tools_section = "\n\n".join(tool_definitions)
    
    tool_count = len(tools)
    if tool_count == 1:
        tool_word = "tool"
        available_text = "You have access to this tool"
    else:
        tool_word = "tools" 
        available_text = f"You have access to these {tool_count} tools"
    
    return f"""You are a helpful AI assistant with access to {tool_word}. When you need to use a {tool_word}, call it using the ```tool_code format with the EXACT parameter names shown in the function signatures below.

{available_text}:

{tools_section}

CRITICAL INSTRUCTIONS:
- When you decide to use a {tool_word}, use the ```tool_code format
- Use the EXACT parameter names from the function signatures above
- Do NOT use generic placeholder names - use the actual parameter names shown
- Example format:
```tool_code
{_get_example_call(tools[0])}
```

Always use the actual parameter names like 'directory_path', 'pattern', 'recursive' etc. as shown in the function definitions."""


def _generate_qwen_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Generate Qwen-optimized tool prompt."""
    tool_definitions = []
    
    for tool in tools:
        if "function" in tool:
            func_info = tool["function"]
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "")
            parameters = func_info.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        tool_def = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        tool_definitions.append(tool_def)
    
    tool_count = len(tools)
    if tool_count == 1:
        tool_word = "tool"
        available_text = "You have access to this tool"
    else:
        tool_word = "tools"
        available_text = f"You have access to these {tool_count} tools"
    
    tools_json = str(tool_definitions).replace("'", '"')
    
    return f"""You are a helpful AI assistant with access to {tool_word}. {available_text}:

{tools_json}

When you need to use a {tool_word}, use the <|tool_call|> format:
<|tool_call|>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}

Always use the exact parameter names from the tool definitions above."""


def _generate_llama_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Generate Llama-optimized tool prompt."""
    tool_definitions = []
    
    for tool in tools:
        if "function" in tool:
            func_info = tool["function"]
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "")
            parameters = func_info.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        # Format parameters nicely
        if isinstance(parameters, dict) and "properties" in parameters:
            props = parameters["properties"]
            param_list = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                param_list.append(f"- {param_name} ({param_type}): {param_desc}")
            params_text = "\n  ".join(param_list)
        else:
            params_text = "No parameters required"
        
        tool_def = f"""• {name}: {description}
  Parameters:
  {params_text}"""
        tool_definitions.append(tool_def)
    
    tools_section = "\n\n".join(tool_definitions)
    
    tool_count = len(tools)
    if tool_count == 1:
        tool_word = "tool"
        available_text = "You have access to this tool"
    else:
        tool_word = "tools"
        available_text = f"You have access to these {tool_count} tools"
    
    return f"""You are a helpful AI assistant with access to {tool_word}. {available_text}:

{tools_section}

When you need to use a {tool_word}, use the <function_call> format:
<function_call>
{{"name": "tool_name", "arguments": {{"parameter_name": "value"}}}}
</function_call>

Always use the exact parameter names shown in the tool definitions above."""


def _generate_xml_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Generate XML-wrapped tool prompt (for Phi models)."""
    tool_definitions = []
    
    for tool in tools:
        if "function" in tool:
            func_info = tool["function"]
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "")
            parameters = func_info.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        # Format parameters
        if isinstance(parameters, dict) and "properties" in parameters:
            props = parameters["properties"]
            param_list = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                param_list.append(f"{param_name} ({param_type}): {param_desc}")
            params_text = ", ".join(param_list)
        else:
            params_text = "No parameters"
        
        tool_def = f"{name}({params_text}) - {description}"
        tool_definitions.append(tool_def)
    
    tools_section = "\n".join(f"• {tool}" for tool in tool_definitions)
    
    tool_count = len(tools)
    if tool_count == 1:
        tool_word = "tool"
        available_text = "You have access to this tool"
    else:
        tool_word = "tools"
        available_text = f"You have access to these {tool_count} tools"
    
    return f"""You are a helpful AI assistant with access to {tool_word}. {available_text}:

{tools_section}

When you need to use a {tool_word}, use this XML format:
<tool_call>
{{"name": "tool_name", "arguments": {{"parameter_name": "value"}}}}
</tool_call>

Use the exact parameter names from the tool definitions above."""


def _generate_generic_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """Generate generic tool prompt for unknown architectures."""
    tool_definitions = []
    
    for tool in tools:
        if "function" in tool:
            func_info = tool["function"]
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "")
            parameters = func_info.get("parameters", {})
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
        
        tool_def = f"- {name}: {description}"
        tool_definitions.append(tool_def)
    
    tools_section = "\n".join(tool_definitions)
    
    tool_count = len(tools)
    if tool_count == 1:
        tool_word = "tool"
        available_text = "You have access to this tool"
    else:
        tool_word = "tools"
        available_text = f"You have access to these {tool_count} tools"
    
    return f"""You are a helpful AI assistant with access to {tool_word}. {available_text}:

{tools_section}

When you need to use a {tool_word}, describe what you want to do and I will help you format the request appropriately."""


def _get_example_call(tool: Dict[str, Any]) -> str:
    """Generate an example function call with actual parameter names."""
    if "function" in tool:
        func_info = tool["function"]
        name = func_info.get("name", "unknown")
        parameters = func_info.get("parameters", {})
    else:
        name = tool.get("name", "unknown")
        parameters = tool.get("parameters", {})
    
    if isinstance(parameters, dict) and "properties" in parameters:
        props = parameters["properties"]
        example_args = []
        
        for param_name, param_info in props.items():
            param_type = param_info.get("type", "string")
            
            # Generate appropriate example values
            if param_type == "string":
                if "directory" in param_name.lower() or "path" in param_name.lower():
                    example_args.append(f'{param_name}="."')
                elif "pattern" in param_name.lower():
                    example_args.append(f'{param_name}="*.py"')
                else:
                    example_args.append(f'{param_name}="example_value"')
            elif param_type in ["integer", "number"]:
                example_args.append(f'{param_name}=100')
            elif param_type == "boolean":
                example_args.append(f'{param_name}=True')
            else:
                example_args.append(f'{param_name}="value"')
        
        return f"{name}({', '.join(example_args)})"
    else:
        return f"{name}()" 