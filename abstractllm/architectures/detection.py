"""
Architecture Detection

Identifies model architectures from model names, supporting various naming conventions
from different providers (HuggingFace, Ollama, MLX, etc.).
"""

import re
from typing import Optional, Dict, List


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for consistent architecture detection.
    
    Args:
        model_name: Raw model name from any provider
        
    Returns:
        Normalized model name
    """
    # Convert to lowercase
    normalized = model_name.lower()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "mlx-community/",
        "microsoft/",
        "meta-llama/",
        "mistralai/",
        "google/",
        "01-ai/",
        "deepseek-ai/",
        "qwen/",
        "ibm-granite/",
        "codellama/",
    ]
    
    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    
    # Handle Ollama-style names (model:tag)
    if ":" in normalized:
        normalized = normalized.split(":")[0]
    
    return normalized


def detect_architecture(model_name: str) -> Optional[str]:
    """
    Detect the architecture family of a model from its name.
    
    Args:
        model_name: Model name (can be from any provider)
        
    Returns:
        Architecture name or None if not detected
    """
    normalized = normalize_model_name(model_name)
    
    # Architecture detection patterns (order matters - more specific first)
    architecture_patterns = {
        # Granite family (check before granite to avoid conflicts)
        "granite": [
            r"granite[\-_]?\d",  # granite-3, granite3, granite_3
            r"granite[\-_]?[a-z]+",  # granite-code, granite_instruct
            r"granite$",  # just "granite"
        ],
        
        # Qwen family (check specific versions first)
        "qwen": [
            r"qwen[\d\.]*[\-_]?vl",  # qwen2-vl, qwen2.5-vl
            r"qwen[\d\.]*",  # qwen, qwen2, qwen2.5, qwen3
        ],
        
        # Llama family
        "llama": [
            r"llama[\-_]?\d",  # llama-3, llama3, llama_2
            r"llama$",  # just "llama"
            r"codellama",  # CodeLlama
            r"alpaca",  # Alpaca (Llama-based)
            r"vicuna",  # Vicuna (Llama-based)
            r"wizard",  # WizardLM (Llama-based)
        ],
        
        # Mistral family
        "mistral": [
            r"mistral",
            r"mixtral",  # Mixtral (Mistral-based)
            r"zephyr",  # Zephyr (Mistral-based)
        ],
        
        # Phi family
        "phi": [
            r"phi[\-_]?\d",  # phi-3, phi3, phi_2
            r"phi$",  # just "phi"
        ],
        
        # Gemma family
        "gemma": [
            r"gemma",
            r"paligemma",  # PaliGemma (vision)
        ],
        
        # DeepSeek family
        "deepseek": [
            r"deepseek",
        ],
        
        # Yi family
        "yi": [
            r"yi[\-_]?\d",  # yi-6b, yi_9b
            r"yi$",  # just "yi"
        ],
        
        # Claude family (Anthropic)
        "claude": [
            r"claude",
        ],
        
        # GPT family (OpenAI)
        "gpt": [
            r"gpt[\-_]?\d",  # gpt-4, gpt4, gpt_3
            r"gpt$",  # just "gpt"
        ],
    }
    
    for architecture, patterns in architecture_patterns.items():
        for pattern in patterns:
            if re.search(pattern, normalized):
                return architecture
    
    return None


def get_tool_call_format(architecture: str) -> Optional[str]:
    """
    Get the expected tool call format for an architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        Tool call format string or None
    """
    format_map = {
        "granite": "special_token",    # <|tool_call|>[{...}]
        "llama": "function_call",      # <function_call>{...}</function_call>
        "qwen": "special_token",       # <|tool_call|>[{...}]
        "mistral": "xml_wrapped",      # <tool_call>{...}</tool_call>
        "phi": "xml_wrapped",          # <tool_call>{...}</tool_call>
        "gemma": "xml_wrapped",        # <tool_call>{...}</tool_call>
        "deepseek": "raw_json",        # Raw JSON objects
        "yi": "xml_wrapped",           # <tool_call>{...}</tool_call>
        "claude": "xml_wrapped",       # <tool_call>{...}</tool_call>
        "gpt": "json_schema",          # OpenAI function calling format
    }
    
    return format_map.get(architecture)


def get_supported_architectures() -> List[str]:
    """
    Get list of all supported architectures from detection patterns.
    
    Returns:
        List of architecture names
    """
    return [
        "granite",
        "qwen", 
        "llama",
        "mistral",
        "phi",
        "gemma", 
        "deepseek",
        "yi",
        "claude",
        "gpt",
    ] 