"""
Unified Architecture Detection System for AbstractLLM.

This module merges the proven logic from mlx_model_configs.py with the new
architecture detection capabilities to provide a single, comprehensive
system for detecting model architectures and their capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCallFormat(Enum):
    """Tool call formats supported by different architectures."""
    SPECIAL_TOKEN = "special_token"       # <|tool_call|>[{...}]
    XML_WRAPPED = "xml_wrapped"           # <tool_call>{...}</tool_call>
    FUNCTION_CALL = "function_call"       # <function_call {...}>
    RAW_JSON = "raw_json"                # Raw JSON objects
    MARKDOWN_CODE = "markdown_code"       # ```tool_call\n... ```
    TOOL_CODE = "tool_code"              # ```tool_code\nfunc_name(param=value)\n```
    GEMMA_PYTHON = "gemma_python"         # [func_name(param=value, ...)]
    GEMMA_JSON = "gemma_json"            # {"name": "func_name", "parameters": {...}}
    HF_NATIVE = "hf_native"              # Use HuggingFace template directly


@dataclass
class UnifiedModelConfig:
    """Unified configuration for a model architecture combining MLX and architecture detection."""
    
    # Basic identification
    architecture: str
    model_name: str
    
    # Token configuration (from MLX configs)
    eos_tokens: List[str]
    bos_tokens: List[str]
    
    # Generation parameters (from MLX configs)
    default_repetition_penalty: float = 1.0
    default_temperature: float = 0.7
    
    # Capabilities (from architecture detection)
    supports_tools: bool = False
    tool_call_format: Optional[ToolCallFormat] = None
    supports_system_prompt: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    max_context_length: Optional[int] = None
    
    # Chat template configuration
    chat_template_type: str = "default"
    
    # Provider preferences
    preferred_providers: List[str] = None
    
    def __post_init__(self):
        if self.preferred_providers is None:
            self.preferred_providers = []


class UnifiedArchitectureDetector:
    """
    Unified architecture detector that combines proven MLX model configurations
    with new architecture detection capabilities.
    """
    
    # Architecture mapping combining both systems
    ARCHITECTURE_CONFIGS = {
        # Qwen models (including DeepSeek R1 which is based on Qwen3)
        "qwen": {
            "patterns": ["qwen", "deepseek"],  # DeepSeek R1 is distilled from Qwen3
            "exclusions": ["vl"],  # Exclude vision-language models
            "architecture": "qwen",
            "eos_tokens": ["<|endoftext|>", "<|im_end|>", "</s>"],
            "bos_tokens": ["<|im_start|>"],
            "default_repetition_penalty": 1.2,
            "supports_tools": True,
            "tool_call_format": ToolCallFormat.SPECIAL_TOKEN,
            "supports_system_prompt": True,
            "supports_vision": False,
            "supports_streaming": True,
            "preferred_providers": ["mlx", "huggingface"]
        },
        
        # Qwen VL models (vision-language)
        "qwen_vl": {
            "patterns": ["qwen.*vl", "qwen2-vl"],
            "exclusions": [],
            "architecture": "qwen_vl",
            "eos_tokens": ["<|endoftext|>", "<|im_end|>", "</s>"],
            "bos_tokens": ["<|im_start|>"],
            "supports_tools": False,  # Vision models typically don't support tools
            "tool_call_format": None,
            "supports_system_prompt": True,
            "supports_vision": True,
            "supports_streaming": True,
            "preferred_providers": ["mlx", "huggingface"]
        },
        
        # Gemma models (including PaliGemma which is built on Gemma)
        "gemma": {
            "patterns": ["gemma", "paligemma"],  # PaliGemma is built on Gemma architecture
            "exclusions": [],
            "architecture": "gemma",
            "eos_tokens": ["<eos>", "</s>"],
            "bos_tokens": ["<bos>", "<s>"],
            "supports_tools": True,
            "tool_call_format": ToolCallFormat.TOOL_CODE,  # Actual format used by Gemma models
            "supports_system_prompt": True,
            "supports_vision": False,  # Base Gemma doesn't support vision (PaliGemma does but uses same base)
            "supports_streaming": True,
            "preferred_providers": ["mlx", "huggingface"]
        },
        
        # Llama family
        "llama": {
            "patterns": ["llama"],
            "exclusions": [],
            "architecture": "llama",
            "eos_tokens": ["</s>", "<|endoftext|>"],
            "bos_tokens": ["<s>"],
            "supports_tools": True,
            "tool_call_format": ToolCallFormat.FUNCTION_CALL,
            "supports_system_prompt": True,
            "supports_vision": False,
            "supports_streaming": True,
            "preferred_providers": ["mlx", "huggingface", "ollama"]
        },
        
        # Mistral family
        "mistral": {
            "patterns": ["mistral", "mixtral"],
            "exclusions": [],
            "architecture": "mistral",
            "eos_tokens": ["</s>"],
            "bos_tokens": ["<s>"],
            "supports_tools": True,
            "tool_call_format": ToolCallFormat.XML_WRAPPED,
            "supports_system_prompt": True,
            "supports_vision": False,
            "supports_streaming": True,
            "preferred_providers": ["huggingface", "ollama"]
        },
        
        # Phi family
        "phi": {
            "patterns": ["phi"],
            "exclusions": [],
            "architecture": "phi",
            "eos_tokens": ["<|endoftext|>", "</s>"],
            "bos_tokens": ["<s>"],
            "supports_tools": True,
            "tool_call_format": ToolCallFormat.XML_WRAPPED,
            "supports_system_prompt": True,
            "supports_vision": False,
            "supports_streaming": True,
            "preferred_providers": ["mlx", "huggingface"]
        },
        
        # Default configuration
        "default": {
            "patterns": [],
            "exclusions": [],
            "architecture": "default",
            "eos_tokens": ["</s>"],
            "bos_tokens": ["<s>"],
            "supports_tools": False,
            "tool_call_format": None,
            "supports_system_prompt": True,
            "supports_vision": False,
            "supports_streaming": True,
            "preferred_providers": ["huggingface"]
        }
    }
    
    @classmethod
    def detect_architecture(cls, model_name: str) -> str:
        """
        Detect model architecture from model name using pattern matching.
        
        Args:
            model_name: The model name/path
            
        Returns:
            Architecture name
        """
        model_name_lower = model_name.lower()
        
        # Special case: Check for vision models first since they're more specific
        if any(pattern in model_name_lower for pattern in ["vl", "vision", "visual"]):
            # Check Qwen VL models
            if any(pattern in model_name_lower for pattern in ["qwen", "qwen2"]):
                return "qwen_vl"
            # PaliGemma is still Gemma architecture, just with vision capabilities
            elif "paligemma" in model_name_lower:
                return "gemma"
        
        # Test each architecture's patterns in priority order
        for arch_name, config in cls.ARCHITECTURE_CONFIGS.items():
            # Check explicit model patterns
            patterns = config.get("patterns", [arch_name])
            exclusions = config.get("exclusions", [])
            
            # Check for exclusions first
            if any(exclusion in model_name_lower for exclusion in exclusions):
                continue
                
            # Check if any pattern matches
            if any(pattern in model_name_lower for pattern in patterns):
                return arch_name
        
        # Fallback to unknown architecture
        return "unknown"
    
    @classmethod
    def get_model_config(cls, model_name: str) -> UnifiedModelConfig:
        """
        Get unified model configuration combining MLX and architecture detection.
        
        Args:
            model_name: Model name
            
        Returns:
            Unified model configuration
        """
        architecture = cls.detect_architecture(model_name)
        
        # Get base configuration for this architecture
        if architecture in cls.ARCHITECTURE_CONFIGS:
            base_config = cls.ARCHITECTURE_CONFIGS[architecture]
        else:
            # Default configuration
            base_config = {
                "eos_tokens": ["</s>"],
                "bos_tokens": ["<s>"],
                "default_repetition_penalty": 1.0,
                "supports_tools": False,
                "tool_call_format": None,
                "supports_system_prompt": True,
                "supports_vision": False,
                "chat_template_type": "default"
            }
        
        # Create unified configuration
        config = UnifiedModelConfig(
            architecture=architecture,
            model_name=model_name,
            eos_tokens=base_config.get("eos_tokens", ["</s>"]),
            bos_tokens=base_config.get("bos_tokens", ["<s>"]),
            default_repetition_penalty=base_config.get("default_repetition_penalty", 1.0),
            default_temperature=base_config.get("default_temperature", 0.7),
            supports_tools=base_config.get("supports_tools", False),
            tool_call_format=base_config.get("tool_call_format"),
            supports_system_prompt=base_config.get("supports_system_prompt", True),
            supports_vision=base_config.get("supports_vision", False),
            supports_streaming=base_config.get("supports_streaming", True),
            max_context_length=base_config.get("max_context_length"),
            chat_template_type=base_config.get("chat_template_type", "default"),
            preferred_providers=base_config.get("preferred_providers", ["mlx", "huggingface"])
        )
        
        logger.info(f"Created unified config for {model_name}: {architecture} architecture, "
                   f"tools={config.supports_tools}, format={config.tool_call_format}")
        
        return config
    
    @classmethod
    def get_tool_call_format(cls, model_name: str) -> Optional[ToolCallFormat]:
        """Get the tool calling format for a model."""
        config = cls.get_model_config(model_name)
        return config.tool_call_format
    
    @classmethod
    def supports_tools(cls, model_name: str) -> bool:
        """Check if a model supports tool calling."""
        config = cls.get_model_config(model_name)
        return config.supports_tools
    
    @classmethod
    def get_generation_params(cls, model_name: str, temperature: float = None) -> Dict[str, Any]:
        """
        Get MLX generation parameters for a model.
        
        Args:
            model_name: Model name
            temperature: Override temperature
            
        Returns:
            Generation parameters for MLX
        """
        config = cls.get_model_config(model_name)
        
        if temperature is None:
            temperature = config.default_temperature
        
        # Ensure temperature is valid
        temperature = float(max(0.01, min(2.0, temperature)))
        
        params = {}
        
        # Create sampler if MLX is available
        try:
            import mlx_lm.sample_utils
            sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)
            params["sampler"] = sampler
            
            # Apply repetition penalty if configured
            if config.default_repetition_penalty != 1.0:
                logits_processors = mlx_lm.sample_utils.make_logits_processors(
                    repetition_penalty=config.default_repetition_penalty,
                    repetition_context_size=64
                )
                params["logits_processors"] = logits_processors
                logger.debug(f"Applied repetition penalty {config.default_repetition_penalty} for {config.architecture}")
                
        except ImportError:
            logger.warning("MLX sample utils not available, using basic parameters")
        
        return params


# Global instance
_unified_detector = UnifiedArchitectureDetector()

# Public API functions
def detect_architecture(model_name: str) -> str:
    """Detect the architecture of a model."""
    return _unified_detector.detect_architecture(model_name)

def get_model_config(model_name: str) -> UnifiedModelConfig:
    """Get unified model configuration."""
    return _unified_detector.get_model_config(model_name)

def get_tool_call_format(model_name: str) -> Optional[ToolCallFormat]:
    """Get the tool calling format for a model."""
    return _unified_detector.get_tool_call_format(model_name)

def supports_tools(model_name: str) -> bool:
    """Check if a model supports tool calling."""
    return _unified_detector.supports_tools(model_name)

def get_generation_params(model_name: str, temperature: float = None) -> Dict[str, Any]:
    """Get MLX generation parameters for a model."""
    return _unified_detector.get_generation_params(model_name, temperature) 