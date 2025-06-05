"""
Architecture Capabilities

Defines the capabilities of each model architecture including tool calling formats,
system prompt support, vision capabilities, etc.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
from enum import Enum


class ToolCallFormat(Enum):
    """Tool call formats supported by different architectures."""
    SPECIAL_TOKEN = "special_token"       # <|tool_call|>[{...}]
    XML_WRAPPED = "xml_wrapped"           # <tool_call>{...}</tool_call>
    FUNCTION_CALL = "function_call"       # <function_call>{...}</function_call>
    RAW_JSON = "raw_json"                # Raw JSON objects
    JSON_SCHEMA = "json_schema"           # OpenAI-style function calling
    HF_NATIVE = "hf_native"              # Use HuggingFace template directly


@dataclass
class ArchitectureCapabilities:
    """Capabilities of a model architecture."""
    architecture: str
    supports_tools: bool = False
    tool_call_format: Optional[ToolCallFormat] = None
    supports_system_prompt: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    supports_functions: bool = False  # Legacy function calling
    max_context_length: Optional[int] = None
    preferred_providers: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.preferred_providers is None:
            self.preferred_providers = []


# Architecture capabilities database
ARCHITECTURE_CAPABILITIES = {
    "granite": ArchitectureCapabilities(
        architecture="granite",
        supports_tools=True,
        tool_call_format=ToolCallFormat.SPECIAL_TOKEN,
        supports_system_prompt=True,
        supports_vision=False,
        supports_streaming=True,
        max_context_length=8192,
        preferred_providers=["ollama", "mlx", "huggingface"],
        notes="IBM Granite models with special token format for tool calls"
    ),
    
    "qwen": ArchitectureCapabilities(
        architecture="qwen",
        supports_tools=True,
        tool_call_format=ToolCallFormat.SPECIAL_TOKEN,
        supports_system_prompt=True,
        supports_vision=True,  # Qwen-VL variants
        supports_streaming=True,
        max_context_length=32768,  # Qwen2.5 has long context
        preferred_providers=["ollama", "mlx", "huggingface"],
        notes="Alibaba Qwen models, including vision variants"
    ),
    
    "llama": ArchitectureCapabilities(
        architecture="llama",
        supports_tools=True,
        tool_call_format=ToolCallFormat.FUNCTION_CALL,
        supports_system_prompt=True,
        supports_vision=False,
        supports_streaming=True,
        max_context_length=8192,  # Most Llama models
        preferred_providers=["ollama", "mlx", "anthropic", "openai"],
        notes="Meta Llama family including fine-tuned variants"
    ),
    
    "mistral": ArchitectureCapabilities(
        architecture="mistral",
        supports_tools=True,
        tool_call_format=ToolCallFormat.XML_WRAPPED,
        supports_system_prompt=True,
        supports_vision=False,
        supports_streaming=True,
        max_context_length=32768,  # Mistral has long context
        preferred_providers=["ollama", "huggingface"],
        notes="Mistral AI models including Mixtral"
    ),
    
    "phi": ArchitectureCapabilities(
        architecture="phi",
        supports_tools=True,
        tool_call_format=ToolCallFormat.XML_WRAPPED,
        supports_system_prompt=True,
        supports_vision=False,  # Phi-3-Vision exists but not common
        supports_streaming=True,
        max_context_length=4096,
        preferred_providers=["ollama", "mlx", "huggingface"],
        notes="Microsoft Phi family of small language models"
    ),
    
    "gemma": ArchitectureCapabilities(
        architecture="gemma",
        supports_tools=True,
        tool_call_format=ToolCallFormat.XML_WRAPPED,
        supports_system_prompt=True,
        supports_vision=True,  # PaliGemma variant
        supports_streaming=True,
        max_context_length=8192,
        preferred_providers=["ollama", "mlx", "huggingface"],
        notes="Google Gemma models including PaliGemma vision variant"
    ),
    
    "deepseek": ArchitectureCapabilities(
        architecture="deepseek",
        supports_tools=True,
        tool_call_format=ToolCallFormat.RAW_JSON,
        supports_system_prompt=True,
        supports_vision=False,
        supports_streaming=True,
        max_context_length=16384,
        preferred_providers=["ollama", "huggingface"],
        notes="DeepSeek models including R1 reasoning variants"
    ),
    
    "yi": ArchitectureCapabilities(
        architecture="yi",
        supports_tools=True,
        tool_call_format=ToolCallFormat.XML_WRAPPED,
        supports_system_prompt=True,
        supports_vision=False,
        supports_streaming=True,
        max_context_length=4096,
        preferred_providers=["ollama", "huggingface"],
        notes="01.ai Yi language models"
    ),
    
    "claude": ArchitectureCapabilities(
        architecture="claude",
        supports_tools=True,
        tool_call_format=ToolCallFormat.XML_WRAPPED,
        supports_system_prompt=True,
        supports_vision=True,
        supports_streaming=True,
        max_context_length=200000,  # Claude has very long context
        preferred_providers=["anthropic"],
        notes="Anthropic Claude models via API"
    ),
    
    "gpt": ArchitectureCapabilities(
        architecture="gpt",
        supports_tools=True,
        tool_call_format=ToolCallFormat.JSON_SCHEMA,
        supports_system_prompt=True,
        supports_vision=True,  # GPT-4V and later
        supports_streaming=True,
        supports_functions=True,  # Legacy function calling
        max_context_length=128000,  # GPT-4 Turbo
        preferred_providers=["openai"],
        notes="OpenAI GPT models via API"
    ),
}


def get_capabilities(architecture: str) -> Optional[ArchitectureCapabilities]:
    """
    Get capabilities for an architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        ArchitectureCapabilities or None if not found
    """
    return ARCHITECTURE_CAPABILITIES.get(architecture)


def get_supported_architectures() -> List[str]:
    """
    Get list of all supported architectures.
    
    Returns:
        List of architecture names
    """
    return list(ARCHITECTURE_CAPABILITIES.keys())


def get_architectures_by_capability(capability: str, value: bool = True) -> List[str]:
    """
    Get architectures that support a specific capability.
    
    Args:
        capability: Capability name (e.g., 'supports_tools', 'supports_vision')
        value: Required value for the capability
        
    Returns:
        List of architecture names
    """
    result = []
    for arch, caps in ARCHITECTURE_CAPABILITIES.items():
        if hasattr(caps, capability) and getattr(caps, capability) == value:
            result.append(arch)
    return result


def get_architectures_by_tool_format(tool_format: ToolCallFormat) -> List[str]:
    """
    Get architectures that use a specific tool call format.
    
    Args:
        tool_format: Tool call format
        
    Returns:
        List of architecture names
    """
    result = []
    for arch, caps in ARCHITECTURE_CAPABILITIES.items():
        if caps.tool_call_format == tool_format:
            result.append(arch)
    return result


def get_architectures_for_provider(provider: str) -> List[str]:
    """
    Get architectures preferred for a specific provider.
    
    Args:
        provider: Provider name
        
    Returns:
        List of architecture names
    """
    result = []
    for arch, caps in ARCHITECTURE_CAPABILITIES.items():
        if provider in caps.preferred_providers:
            result.append(arch)
    return result


def detect_model_type(model_name: str) -> str:
    """
    Detect if a model is 'base' or 'instruct'.
    
    Base models: Trained for text completion/prediction only
    Instruct models: Fine-tuned to follow instructions and engage in conversation
    
    Args:
        model_name: Full model name
        
    Returns:
        'base' or 'instruct'
    """
    model_name_lower = model_name.lower()
    
    # Check for base indicators FIRST (more specific)
    base_indicators = ["base", "foundation", "pretrain", "raw"]
    if any(indicator in model_name_lower for indicator in base_indicators):
        return "base"
    
    # Check for instruct indicators
    instruct_indicators = ["instruct", "chat", "it", "sft", "tune", "finetune"]
    if any(indicator in model_name_lower for indicator in instruct_indicators):
        return "instruct"
    
    # Default assumption: if not explicitly marked as base, assume instruct
    # Most models in production are instruct variants
    return "instruct"


def detect_model_tool_capability(model_name: str) -> bool:
    """
    Detect if a specific model supports tool calling/function calling.
    
    Tool support requires:
    1. The model must be an instruct model (base models can't follow instructions)
    2. The model family must support tool calling (from training knowledge)
    3. This specific model variant must be trained for tools
    
    Args:
        model_name: Full model name
        
    Returns:
        True if the model supports tool calling
    """
    model_name_lower = model_name.lower()
    
    # First check: Must be an instruct model to support tools
    if detect_model_type(model_name) != "instruct":
        return False
    
    # Second check: Explicit tool indicators in model names (highest confidence)
    explicit_tool_indicators = [
        "tool", "function", "agent", "functionary", 
        "gorilla", "toolformer", "hermes"
    ]
    if any(indicator in model_name_lower for indicator in explicit_tool_indicators):
        return True
    
    # Third check: Model families that support tools (from training knowledge)
    # These are model families where instruct variants typically support tools
    tool_supporting_families = [
        # User-provided list of families that support tools
        "granite3.3", "granite3.2", "granite3-dense", "granite3.1-dense",
        "granite3.1-moe", "granite3-moe", "devstral", "llama4", "llama3.3", 
        "llama3.2", "llama3.1", "llama3-groq-tool-use", "mistral", "qwen3",
        "qwen2.5", "qwen2.5-coder", "qwen2", "qwq", "mistral-small",
        "mistral-small3.1", "mistral-large", "command-a", "command-r",
        "command-r-plus", "hermes3", "phi4-mini", "cogito", "nemotron-mini",
        "athene-v2", "nemotron", "gemma3"
    ]
    
    # Check if model name contains any of these tool-supporting family indicators
    for family in tool_supporting_families:
        family_normalized = family.lower().replace("-", "").replace(".", "")
        model_normalized = model_name_lower.replace("-", "").replace(".", "")
        if family_normalized in model_normalized:
            return True
    
    # Default: No tool support unless explicitly indicated
    return False


def detect_model_vision_capability(model_name: str) -> bool:
    """
    Detect if a specific model has vision capabilities.
    
    This is provider-agnostic - the same model should be detected consistently
    regardless of whether it's accessed through MLX, HuggingFace, Ollama, etc.
    
    Args:
        model_name: Full model name (e.g., "mlx-community/qwen2-vl-7b-4bit")
        
    Returns:
        True if the model has vision capabilities, False otherwise
    """
    model_name_lower = model_name.lower()
    
    # Vision indicators in model names - these are explicit markers
    vision_indicators = [
        "vlm", "vision", "visual", "llava", "clip", "multimodal", 
        "vit", "blip", "vqa", "image", "qwen-vl", "qwen2-vl", 
        "qwen2.5-vl", "pixtral", "llava-next", "cogvlm", "internvl", 
        "minicpm-v", "phi-3-vision", "kosmos", "flamingo", "gpt-4v", 
        "paligemma-3b-mix-448-8bit", "Qwen2-VL-2B-Instruct-4bit", 
        "Qwen2-VL-7B-Instruct-4bit", "Qwen2.5-VL-3B-Instruct-4bit", 
        "Qwen2.5-VL-7B-Instruct-4bit", "Qwen2.5-VL-32B-Instruct-4bit", 
        "Qwen2.5-VL-72B-Instruct-4bit", "qwen-vl-chat"
    ]
    
    return any(indicator in model_name_lower for indicator in vision_indicators)


def detect_model_audio_capability(model_name: str) -> bool:
    """
    Detect if a specific model has audio capabilities.
    
    Args:
        model_name: Full model name
        
    Returns:
        True if the model has audio capabilities, False otherwise
    """
    model_name_lower = model_name.lower()
    
    # Audio indicators in model names
    audio_indicators = [
        "audio", "speech", "whisper", "wav2vec", "hubert", "salmonn",
        "seamless", "musicgen", "audiogen", "bark", "xtts", "speecht5"
    ]
    
    return any(indicator in model_name_lower for indicator in audio_indicators)


def detect_model_reasoning_capability(model_name: str) -> bool:
    """
    Detect if a specific model has enhanced reasoning capabilities.
    
    Args:
        model_name: Full model name
        
    Returns:
        True if the model has enhanced reasoning capabilities
    """
    model_name_lower = model_name.lower()
    
    # Reasoning indicators in model names
    reasoning_indicators = [
        "reasoning", "think", "cot", "o1", "qwq", "deepseek-r1",
        "reflection", "step", "math", "theorem"
    ]
    
    return any(indicator in model_name_lower for indicator in reasoning_indicators)


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """
    Get comprehensive capabilities for a specific model instance.
    
    This returns what THIS specific model can actually do, based on:
    1. Model type (base vs instruct) 
    2. Specific capability indicators in the name
    3. Training knowledge of model families
    
    Args:
        model_name: Full model name (e.g., "mlx-community/Qwen3-1.7B-4bit-DWQ-053125")
        
    Returns:
        Dictionary with model-specific capabilities
    """
    model_type = detect_model_type(model_name)
    
    result = {
        "model_type": model_type,
        "supports_vision": detect_model_vision_capability(model_name),
        "supports_audio": detect_model_audio_capability(model_name),
        "supports_reasoning": detect_model_reasoning_capability(model_name),
        # Tool support requires instruct model + family support + specific training
        "supports_tools": detect_model_tool_capability(model_name),
        "supports_chat": model_type == "instruct",  # Only instruct models can chat
        "supports_completion": True,  # Both base and instruct can do completion
    }
    
    # Add capability summary
    capabilities = []
    if result["supports_chat"]:
        capabilities.append("chat")
    if result["supports_tools"]:
        capabilities.append("tools")
    if result["supports_vision"]:
        capabilities.append("vision")
    if result["supports_audio"]:
        capabilities.append("audio")
    if result["supports_reasoning"]:
        capabilities.append("reasoning")
    
    result["capability_summary"] = capabilities
    
    return result 