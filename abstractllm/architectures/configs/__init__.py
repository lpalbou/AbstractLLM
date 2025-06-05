"""
Architecture-specific configurations.

This module provides configuration classes for different model architectures
that can be used by providers to set architecture-specific parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from ..detection import detect_architecture


@dataclass
class ArchitectureConfig:
    """Base configuration for a model architecture."""
    architecture: str
    eos_tokens: List[str] = None
    bos_tokens: List[str] = None
    default_temperature: float = 0.7
    default_repetition_penalty: float = 1.0
    preferred_max_tokens: Optional[int] = None
    chat_template_format: Optional[str] = None
    
    def __post_init__(self):
        if self.eos_tokens is None:
            self.eos_tokens = ["</s>"]
        if self.bos_tokens is None:
            self.bos_tokens = ["<s>"]


# Architecture-specific configurations
ARCHITECTURE_CONFIGS = {
    "granite": ArchitectureConfig(
        architecture="granite",
        eos_tokens=["<|endoftext|>", "</s>", "<|end|>"],
        bos_tokens=["<s>", "<|start|>"],
        default_temperature=0.7,
        default_repetition_penalty=1.1,
        preferred_max_tokens=4096,
        chat_template_format="special_tokens"
    ),
    
    "qwen": ArchitectureConfig(
        architecture="qwen",
        eos_tokens=["<|endoftext|>", "<|im_end|>", "</s>"],
        bos_tokens=["<|im_start|>"],
        default_temperature=0.7,
        default_repetition_penalty=1.2,
        preferred_max_tokens=8192,
        chat_template_format="im_start_end"
    ),
    
    "llama": ArchitectureConfig(
        architecture="llama",
        eos_tokens=["</s>", "<|endoftext|>"],
        bos_tokens=["<s>"],
        default_temperature=0.7,
        default_repetition_penalty=1.0,
        preferred_max_tokens=4096,
        chat_template_format="inst_format"
    ),
    
    "mistral": ArchitectureConfig(
        architecture="mistral",
        eos_tokens=["</s>"],
        bos_tokens=["<s>"],
        default_temperature=0.7,
        default_repetition_penalty=1.0,
        preferred_max_tokens=8192,
        chat_template_format="inst_format"
    ),
    
    "phi": ArchitectureConfig(
        architecture="phi",
        eos_tokens=["<|endoftext|>", "</s>"],
        bos_tokens=["<s>"],
        default_temperature=0.7,
        default_repetition_penalty=1.0,
        preferred_max_tokens=4096,
        chat_template_format="basic"
    ),
    
    "gemma": ArchitectureConfig(
        architecture="gemma",
        eos_tokens=["<eos>", "</s>"],
        bos_tokens=["<bos>", "<s>"],
        default_temperature=0.7,
        default_repetition_penalty=1.0,
        preferred_max_tokens=4096,
        chat_template_format="basic"
    ),
    
    "deepseek": ArchitectureConfig(
        architecture="deepseek",
        eos_tokens=["<|endoftext|>", "<|im_end|>", "</s>"],
        bos_tokens=["<|im_start|>"],
        default_temperature=0.7,
        default_repetition_penalty=1.15,
        preferred_max_tokens=8192,
        chat_template_format="im_start_end"
    ),
    
    "yi": ArchitectureConfig(
        architecture="yi",
        eos_tokens=["<|endoftext|>", "</s>"],
        bos_tokens=["<s>"],
        default_temperature=0.7,
        default_repetition_penalty=1.0,
        preferred_max_tokens=4096,
        chat_template_format="basic"
    ),
}


def get_config(architecture: str) -> Optional[ArchitectureConfig]:
    """
    Get configuration for an architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        ArchitectureConfig or None if not found
    """
    return ARCHITECTURE_CONFIGS.get(architecture)


def get_config_for_model(model_name: str) -> Optional[ArchitectureConfig]:
    """
    Get configuration for a model by detecting its architecture.
    
    Args:
        model_name: Model name
        
    Returns:
        ArchitectureConfig or None if architecture not detected
    """
    architecture = detect_architecture(model_name)
    if architecture:
        return get_config(architecture)
    return None


def get_supported_architectures() -> List[str]:
    """Get list of architectures with configurations."""
    return list(ARCHITECTURE_CONFIGS.keys()) 