"""
Architecture detection and capability lookup.

Clean separation:
- Architecture: HOW to format messages (template patterns)
- Capabilities: WHAT the model can do (tool support, context length)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)

# Cache for loaded data
_formats_cache: Optional[Dict[str, Any]] = None
_capabilities_cache: Optional[Dict[str, Any]] = None


def _load_architecture_formats() -> Dict[str, Any]:
    """Load architecture format definitions."""
    global _formats_cache
    if _formats_cache is not None:
        return _formats_cache
    
    try:
        json_path = Path(__file__).parent.parent / "assets" / "architecture_formats.json"
        with open(json_path) as f:
            _formats_cache = json.load(f)
        return _formats_cache
    except Exception as e:
        logger.error(f"Failed to load architecture formats: {e}")
        return {"architectures": {}}


def _load_model_capabilities() -> Dict[str, Any]:
    """Load model-specific capabilities."""
    global _capabilities_cache
    if _capabilities_cache is not None:
        return _capabilities_cache
    
    try:
        json_path = Path(__file__).parent.parent / "assets" / "model_capabilities.json"
        with open(json_path) as f:
            _capabilities_cache = json.load(f)
        return _capabilities_cache
    except Exception as e:
        logger.error(f"Failed to load model capabilities: {e}")
        return {"models": {}}


def _normalize_model_name(model_name: str) -> str:
    """Normalize model name for matching."""
    # Remove common prefixes
    name = model_name.lower()
    for prefix in ["mlx-community/", "huggingface/", "ollama/", "local/"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Handle Ollama format: convert colon to hyphen for consistency
    # e.g., "qwen3:30b" -> "qwen3-30b"
    if ":" in name:
        # Split on colon and rejoin with hyphen
        parts = name.split(":", 1)
        if len(parts) == 2:
            name = f"{parts[0]}-{parts[1]}"
    
    # Remove file extensions and quantization suffixes ONLY
    # Keep version numbers like 3.5
    if name.endswith((".gguf", ".bin", ".safetensors")):
        name = name.rsplit(".", 1)[0]
    
    # Remove quantization indicators
    for suffix in ["-4bit", "-8bit", "-gguf", "-q4_k_m", "-q8_0", "-fp16", "-int8"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    return name


def detect_model_type(model_name: str) -> str:
    """
    Detect if a model is 'base' or 'instruct' type.
    
    Base models: Trained for text completion only
    Instruct models: Fine-tuned to follow instructions
    
    Args:
        model_name: The model name to analyze
        
    Returns:
        'base' or 'instruct'
    """
    name_lower = model_name.lower()
    
    # Explicit base indicators (check first)
    base_indicators = ["base", "foundation", "pretrain", "raw"]
    if any(indicator in name_lower for indicator in base_indicators):
        return "base"
    
    # Instruct indicators
    instruct_indicators = ["instruct", "chat", "-it", "sft", "rlhf", "assistant"]
    if any(indicator in name_lower for indicator in instruct_indicators):
        return "instruct"
    
    # Special cases
    if "gpt" in name_lower or "claude" in name_lower:
        return "instruct"  # These are chat models
    
    # Default: most distributed models are instruct variants
    return "instruct"


def detect_architecture(model_name: str) -> Optional[str]:
    """
    Detect the architecture of a model for message formatting.
    
    Args:
        model_name: The model name to analyze
        
    Returns:
        Architecture name (e.g., "llama", "qwen", "mistral") or None
    """
    formats = _load_architecture_formats()
    normalized = _normalize_model_name(model_name)
    
    # Check each architecture's patterns
    for arch_name, arch_data in formats.get("architectures", {}).items():
        if arch_name == "generic":  # Skip generic fallback
            continue
            
        patterns = arch_data.get("patterns", [])
        for pattern in patterns:
            if pattern in normalized:
                return arch_name
    
    # Special cases
    if normalized.startswith("o1") or normalized.startswith("o3") or normalized.startswith("o4"):
        return "gpt"  # OpenAI models use same format
    
    return None


def get_architecture_format(architecture: str) -> Dict[str, Any]:
    """
    Get the message format specification for an architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        Format specification dict
    """
    formats = _load_architecture_formats()
    archs = formats.get("architectures", {})
    
    if architecture in archs:
        return archs[architecture]
    
    # Return generic format as fallback
    return archs.get("generic", {})


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get capabilities for a specific model.
    
    Args:
        model_name: The model name to analyze
        
    Returns:
        Dictionary with model capabilities:
        - model_type: Type of model (base/instruct)
        - context_length: Maximum context length (active context size)
        - max_output_tokens: Maximum output tokens (output context size)
        - tool_support: Level of tool support (native/prompted/none)
        - structured_output: Structured output support level (native/prompted/none)
        - parallel_tools: Whether parallel tools are supported
        - max_tools: Maximum tools per call (-1 for unlimited)
        - vision_support: Whether vision is supported (yes/no)
        - image_resolutions: Supported image resolutions (if vision)
        - audio_support: Whether audio is supported (yes/no)
        - embeddings_support: Whether model can generate embeddings for RAG
        - chat_template: Chat template tags (from architecture)
        - tool_template: Tool template tags (from architecture)
        - notes: Any specific notes
        - source: Data source
    """
    caps_data = _load_model_capabilities()
    normalized = _normalize_model_name(model_name)
    
    # Start with defaults
    default_caps = caps_data.get("default_capabilities", {
        "context_length": 4096,
        "max_output_tokens": 2048,
        "tool_support": "none",
        "structured_output": "none",
        "parallel_tools": False,
        "vision_support": False,
        "audio_support": False,
        "embeddings_support": False
    })
    
    capabilities = default_caps.copy()
    
    # Add model type detection
    capabilities["model_type"] = detect_model_type(model_name)
    
    # Check for model matches
    models = caps_data.get("models", {})
    best_match = None
    best_match_length = 0
    
    # Find the best (longest) matching model key
    for model_key, model_caps in models.items():
        model_key_normalized = model_key.replace("-", "").replace(".", "").replace("_", "").lower()
        normalized_clean = normalized.replace("-", "").replace(".", "").replace("_", "")
        
        # Check if this model key matches
        if (model_key in normalized or 
            model_key_normalized in normalized_clean or
            normalized.startswith(model_key) or
            normalized_clean.startswith(model_key_normalized)):
            # Use the longest match (most specific)
            if len(model_key) > best_match_length:
                best_match = model_caps
                best_match_length = len(model_key)
    
    if best_match:
        capabilities.update(best_match)
    else:
        # If no match found, still check for vision/audio/embeddings in name
        if any(indicator in normalized for indicator in ["vision", "vl", "vlm", "llava", "visual"]):
            capabilities["vision_support"] = True
        if any(indicator in normalized for indicator in ["audio", "whisper", "speech"]):
            capabilities["audio_support"] = True
        if any(indicator in normalized for indicator in ["embed", "embedding", "bge", "e5", "gte"]):
            capabilities["embeddings_support"] = True
        
        # For unknown models, default to prompted tool support if it's an instruct model
        # This ensures we don't block tool usage for capable models we haven't catalogued
        if capabilities["model_type"] == "instruct":
            capabilities["tool_support"] = "prompted"
            capabilities["structured_output"] = "prompted"
    
    # Add embeddings detection for known embedding models
    embedding_models = ["text-embedding", "embed", "bge", "e5", "gte", "instructor", "sentence-transformers"]
    if any(em in normalized for em in embedding_models):
        capabilities["embeddings_support"] = True
    
    # Get architecture info for templates
    architecture = detect_architecture(model_name)
    if architecture:
        arch_format = get_architecture_format(architecture)
        # Extract template information
        if arch_format.get("user_prefix") or arch_format.get("assistant_prefix"):
            tags = []
            for key in ["user_prefix", "assistant_prefix", "system_prefix"]:
                if arch_format.get(key):
                    tags.append(arch_format[key].strip())
            capabilities["chat_template"] = ", ".join(tags[:2])  # Show main tags
        else:
            capabilities["chat_template"] = arch_format.get("message_format", "generic")
        
        # Tool template
        if arch_format.get("tool_format"):
            capabilities["tool_template"] = arch_format["tool_format"]
        else:
            capabilities["tool_template"] = "none"
    else:
        capabilities["chat_template"] = "generic"
        capabilities["tool_template"] = "none"
    
    return capabilities


def format_messages(messages: List[Dict[str, str]], architecture: Optional[str] = None) -> str:
    """
    Format messages according to architecture specification.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        architecture: Architecture name (will auto-detect if None)
        
    Returns:
        Formatted prompt string
    """
    if not messages:
        return ""
    
    # Get architecture format
    if architecture is None:
        architecture = "generic"
    
    fmt = get_architecture_format(architecture)
    
    # Special handling for specific formats
    if fmt.get("message_format") == "openai_chat":
        # OpenAI doesn't need formatting - API handles it
        return ""
    
    # Build formatted string
    formatted_parts = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system" and fmt.get("system_prefix"):
            formatted_parts.append(fmt["system_prefix"] + content + fmt.get("system_suffix", ""))
        elif role == "user" and fmt.get("user_prefix"):
            formatted_parts.append(fmt["user_prefix"] + content + fmt.get("user_suffix", ""))
        elif role == "assistant" and fmt.get("assistant_prefix"):
            formatted_parts.append(fmt["assistant_prefix"] + content + fmt.get("assistant_suffix", ""))
    
    # Add generation prompt for assistant
    result = "".join(formatted_parts)
    if messages[-1].get("role") != "assistant" and fmt.get("assistant_prefix"):
        result += fmt["assistant_prefix"]
    
    return result


# Convenience functions for simple access to capabilities
def supports_tools(model_name: str) -> bool:
    """Check if model supports tools (native or prompted)."""
    caps = get_model_capabilities(model_name)
    return caps.get("tool_support", "none") != "none"


def supports_vision(model_name: str) -> bool:
    """Check if model supports vision input."""
    caps = get_model_capabilities(model_name)
    return caps.get("vision_support", False)


def supports_audio(model_name: str) -> bool:
    """Check if model supports audio input."""
    caps = get_model_capabilities(model_name)
    return caps.get("audio_support", False)


def supports_embeddings(model_name: str) -> bool:
    """Check if model can generate embeddings for RAG."""
    caps = get_model_capabilities(model_name)
    return caps.get("embeddings_support", False)


def get_context_limits(model_name: str) -> Dict[str, int]:
    """Get context size limits for a model."""
    caps = get_model_capabilities(model_name)
    return {
        "input": caps.get("context_length", 4096),
        "output": caps.get("max_output_tokens", 2048)
    }


def get_context_length(model_name: str) -> int:
    """
    Get the context length (input limit) for a model.
    
    Args:
        model_name: Model name to check
        
    Returns:
        Context length in tokens
    """
    caps = get_model_capabilities(model_name)
    return caps.get("context_length", 4096)


def is_instruct_model(model_name: str) -> bool:
    """Check if model is instruction-tuned (vs base model)."""
    return detect_model_type(model_name) == "instruct"