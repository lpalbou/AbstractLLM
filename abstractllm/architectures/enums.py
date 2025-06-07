"""
Enums for architecture detection and tool formats.
"""

from enum import Enum


class ToolCallFormat(Enum):
    """Tool call format types for different architectures."""
    
    # XML-based formats
    XML_WRAPPED = "xml_wrapped"  # <tool_call>{...}</tool_call>
    
    # JSON-based formats
    RAW_JSON = "raw_json"  # {"name": "...", "arguments": {...}}
    FUNCTION_CALL = "function_call"  # <function_call name="..." arguments="..."/>
    SPECIAL_TOKEN = "special_token"  # <|tool_call|>{...}
    
    # Code-based formats
    TOOL_CODE = "tool_code"  # ```tool_code\nfunc(...)\n```
    MARKDOWN_CODE = "markdown_code"  # ```tool_call\nfunc(...)\n```
    GEMMA_PYTHON = "gemma_python"  # [func_name(param=value)]
    GEMMA_JSON = "gemma_json"  # {"name": "func", "parameters": {...}}
    
    # Special values
    NONE = "none"  # No tool support
    GENERIC = "generic"  # Unknown/generic format


class ModelType(Enum):
    """Model type classification."""
    
    BASE = "base"  # Base/foundation models (text completion only)
    INSTRUCT = "instruct"  # Instruction-tuned models (follows instructions)


class ArchitectureFamily(Enum):
    """Known architecture families."""
    
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    PHI = "phi"
    GEMMA = "gemma"
    CLAUDE = "claude"
    GPT = "gpt"
    GRANITE = "granite"
    DEEPSEEK = "deepseek"
    YI = "yi"
    GENERIC = "generic"