"""
Configuration system for HuggingFace provider.

This module provides a tree-structured configuration system that handles:
- Base model parameters
- Architecture-specific parameters
- Capability-specific parameters
- Model-specific overrides

The configuration system follows a priority order:
1. Model-specific overrides
2. Capability-specific parameters
3. Architecture-specific parameters
4. Base parameters
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union, List, Type
from enum import Enum
import logging
import torch
from pathlib import Path

from .model_types import ModelArchitecture, ModelCapability

# Configure logger
logger = logging.getLogger(__name__)

class DeviceType(str, Enum):
    """Supported device types for model loading."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon GPU

class QuantizationType(str, Enum):
    """Supported quantization types."""
    NONE = "none"
    INT8 = "8bit"
    INT4 = "4bit"

class TorchDType(str, Enum):
    """Supported torch data types."""
    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

@dataclass
class BaseConfig:
    """Base configuration parameters common to all models."""
    
    # Device configuration
    device_map: Union[str, Dict[int, str]] = "auto"
    device_type: DeviceType = DeviceType.AUTO
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    
    # Model loading
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    use_bettertransformer: bool = False
    use_safetensors: bool = True
    
    # Optimization
    torch_compile: bool = False
    compile_mode: str = "default"
    torch_dtype: TorchDType = TorchDType.AUTO
    quantization: QuantizationType = QuantizationType.NONE
    
    # Resource management
    model_cache_dir: Optional[str] = None
    max_cached_models: int = 3
    cleanup_on_error: bool = True

@dataclass
class EncoderConfig:
    """Configuration for encoder-based models."""
    max_position_embeddings: int = 512
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    intermediate_size: Optional[int] = None
    layer_norm_eps: float = 1e-12

@dataclass
class DecoderConfig:
    """Configuration for decoder-based models."""
    max_position_embeddings: int = 2048
    attention_temperature: float = 1.0
    kv_heads: Optional[int] = None
    sliding_window: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None

@dataclass
class VisionConfig:
    """Configuration for vision models."""
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    use_quick_gelu: bool = True

@dataclass
class SpeechConfig:
    """Configuration for speech models."""
    sampling_rate: int = 16000
    num_mel_bins: int = 80
    hop_length: int = 160
    win_length: int = 400

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 2048
    min_new_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False

@dataclass
class ModelConfig:
    """Complete model configuration with tree structure."""
    
    # Core attributes
    architecture: ModelArchitecture
    name: Optional[str] = None
    revision: Optional[str] = None
    
    # Component configurations
    base: BaseConfig = field(default_factory=BaseConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Architecture-specific configurations
    encoder: Optional[EncoderConfig] = None
    decoder: Optional[DecoderConfig] = None
    vision: Optional[VisionConfig] = None
    speech: Optional[SpeechConfig] = None
    
    # Capability-specific parameters
    capability_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Model-specific overrides
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize architecture-specific configurations."""
        if self.architecture == ModelArchitecture.ENCODER_ONLY:
            if not self.encoder:
                self.encoder = EncoderConfig()
                
        elif self.architecture == ModelArchitecture.DECODER_ONLY:
            if not self.decoder:
                self.decoder = DecoderConfig()
                
        elif self.architecture == ModelArchitecture.ENCODER_DECODER:
            if not self.encoder:
                self.encoder = EncoderConfig()
            if not self.decoder:
                self.decoder = DecoderConfig()
                
        elif self.architecture == ModelArchitecture.VISION_ENCODER:
            if not self.vision:
                self.vision = VisionConfig()
                
        elif self.architecture == ModelArchitecture.SPEECH:
            if not self.speech:
                self.speech = SpeechConfig()
    
    def merge_capability_params(self, capability: ModelCapability) -> None:
        """Merge capability-specific parameters."""
        if capability.name not in self.capability_params:
            self.capability_params[capability.name] = {}
        
        # Update with capability parameters
        self.capability_params[capability.name].update(capability.parameters)
        
        # Apply capability requirements if any
        if capability.requirements:
            # Update base config with requirements
            for key, value in capability.requirements.items():
                if hasattr(self.base, key):
                    setattr(self.base, key, value)
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration for model loading."""
        device_config = {}
        
        # Handle device mapping
        if isinstance(self.base.device_map, dict):
            device_config["device_map"] = self.base.device_map
        elif self.base.device_map == "auto":
            device_config["device_map"] = "auto"
        else:
            device_config["device"] = self.base.device_map
        
        # Add memory configuration if specified
        if self.base.max_memory:
            device_config["max_memory"] = self.base.max_memory
            
        # Add offload configuration if specified
        if self.base.offload_folder:
            device_config["offload_folder"] = self.base.offload_folder
        
        return device_config
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get model loading keyword arguments."""
        kwargs = {
            "trust_remote_code": self.base.trust_remote_code,
            "torch_dtype": self.base.torch_dtype.value,
            "use_safetensors": self.base.use_safetensors
        }
        
        # Add device configuration
        kwargs.update(self.get_device_config())
        
        # Add quantization configuration
        if self.base.quantization != QuantizationType.NONE:
            from transformers import BitsAndBytesConfig
            
            if self.base.quantization == QuantizationType.INT8:
                kwargs["load_in_8bit"] = True
            elif self.base.quantization == QuantizationType.INT4:
                kwargs["load_in_4bit"] = True
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
        
        # Add Flash Attention 2 if enabled
        if self.base.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
        
        # Add BetterTransformer if enabled
        if self.base.use_bettertransformer:
            kwargs["use_bettertransformer"] = True
        
        # Add model-specific overrides
        kwargs.update(self.model_overrides)
        
        return kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        
        # Remove None values
        return {k: v for k, v in config_dict.items() if v is not None}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        # Extract core attributes
        architecture = config_dict.pop("architecture")
        name = config_dict.pop("name", None)
        revision = config_dict.pop("revision", None)
        
        # Create instance
        config = cls(
            architecture=architecture,
            name=name,
            revision=revision
        )
        
        # Update nested configurations
        if "base" in config_dict:
            config.base = BaseConfig(**config_dict["base"])
        if "generation" in config_dict:
            config.generation = GenerationConfig(**config_dict["generation"])
        if "encoder" in config_dict:
            config.encoder = EncoderConfig(**config_dict["encoder"])
        if "decoder" in config_dict:
            config.decoder = DecoderConfig(**config_dict["decoder"])
        if "vision" in config_dict:
            config.vision = VisionConfig(**config_dict["vision"])
        if "speech" in config_dict:
            config.speech = SpeechConfig(**config_dict["speech"])
        
        # Update capability parameters and overrides
        config.capability_params = config_dict.get("capability_params", {})
        config.model_overrides = config_dict.get("model_overrides", {})
        
        return config
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate device configuration
        if self.base.device_type != DeviceType.AUTO:
            if self.base.device_type == DeviceType.CUDA and not torch.cuda.is_available():
                raise ValueError("CUDA device requested but not available")
            elif self.base.device_type == DeviceType.MPS and not hasattr(torch.backends, "mps"):
                raise ValueError("MPS device requested but not available")
        
        # Validate quantization
        if self.base.quantization != QuantizationType.NONE:
            try:
                import bitsandbytes
            except ImportError:
                raise ValueError(
                    "bitsandbytes package required for quantization. "
                    "Install with: pip install bitsandbytes"
                )
        
        # Validate Flash Attention 2
        if self.base.use_flash_attention:
            try:
                from transformers.utils.import_utils import is_flash_attn_available
                if not is_flash_attn_available():
                    raise ValueError(
                        "Flash Attention 2 requested but not available. "
                        "Install with: pip install flash-attn --no-build-isolation"
                    )
            except ImportError:
                raise ValueError("Flash Attention 2 not available")
        
        # Validate architecture-specific configurations
        if self.architecture == ModelArchitecture.ENCODER_ONLY:
            if not self.encoder:
                raise ValueError("Encoder configuration required for ENCODER_ONLY architecture")
        elif self.architecture == ModelArchitecture.DECODER_ONLY:
            if not self.decoder:
                raise ValueError("Decoder configuration required for DECODER_ONLY architecture")
        elif self.architecture == ModelArchitecture.ENCODER_DECODER:
            if not self.encoder or not self.decoder:
                raise ValueError("Both encoder and decoder configurations required for ENCODER_DECODER architecture")
        elif self.architecture == ModelArchitecture.VISION_ENCODER:
            if not self.vision:
                raise ValueError("Vision configuration required for VISION_ENCODER architecture")
        elif self.architecture == ModelArchitecture.SPEECH:
            if not self.speech:
                raise ValueError("Speech configuration required for SPEECH architecture") 