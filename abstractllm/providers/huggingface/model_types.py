"""
Model types and base pipeline classes for HuggingFace provider.

This module defines the core model architectures and their capabilities.
Each architecture represents a fundamental model structure that determines
how the model processes inputs and generates outputs.

Core architectures:
- ENCODER_ONLY: Models that encode input (BERT, RoBERTa)
- DECODER_ONLY: Models that generate text (GPT, LLaMA)
- ENCODER_DECODER: Models that transform text (T5, BART)
- VISION_ENCODER: Models that understand images (ViT, CLIP)
- MULTIMODAL: Models that handle multiple modalities (LLaVA, BLIP)
- SPEECH: Models that process audio (Whisper, SpeechT5)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Union, List, Set, Generator
import logging
import gc
import torch
from pathlib import Path

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import (
    ModelLoadingError,
    UnsupportedFeatureError,
    InvalidInputError
)

# Configure logger
logger = logging.getLogger(__name__)

class ModelArchitecture(str, Enum):
    """Core model architectures that determine fundamental processing patterns.
    
    Each architecture represents a different approach to processing inputs and
    generating outputs. The architecture choice affects:
    - How inputs are processed
    - What parameters are relevant
    - What capabilities are available
    - How resources are managed
    """
    
    ENCODER_ONLY = "encoder_only"
    """Models that encode input text into representations.
    Examples: BERT, RoBERTa, DeBERTa
    Good for: Understanding, classification, token-level tasks"""
    
    DECODER_ONLY = "decoder_only"
    """Models that generate text autoregressively.
    Examples: GPT, LLaMA, Phi
    Good for: Text generation, chat, completion"""
    
    ENCODER_DECODER = "enc_dec"
    """Models that transform input text to output text.
    Examples: T5, BART, mBART
    Good for: Translation, summarization, structured generation"""
    
    VISION_ENCODER = "vision_enc"
    """Models that encode images into representations.
    Examples: ViT, CLIP, DeiT
    Good for: Image understanding, classification, embeddings"""
    
    MULTIMODAL = "multimodal"
    """Models that handle multiple input types together.
    Examples: LLaVA, BLIP, GPT-4V
    Good for: Image-text tasks, multi-modal understanding"""
    
    SPEECH = "speech"
    """Models that process audio signals.
    Examples: Whisper, SpeechT5, Bark
    Good for: Speech recognition, text-to-speech"""

@dataclass
class ModelCapability:
    """Defines a specific capability that a model can perform.
    
    A capability represents a concrete task or function that a model
    can perform, along with metadata about how well it can perform it
    and what specific parameters it needs.
    """
    
    name: str
    """Identifier for the capability (e.g., 'text-generation', 'translation')"""
    
    confidence: float
    """How well the model handles this capability (0.0 to 1.0)"""
    
    requires_finetuning: bool
    """Whether specialized training is needed for optimal performance"""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Capability-specific parameters and their defaults"""
    
    requirements: Optional[Dict[str, Any]] = None
    """System or environment requirements for this capability"""

class ModelCapabilities:
    """Manages and validates model capabilities.
    
    This class handles:
    - Tracking what capabilities a model has
    - Checking if capabilities are available
    - Managing capability-specific configurations
    - Validating capability requirements
    """
    
    def __init__(self, architecture: ModelArchitecture):
        """Initialize capabilities manager.
        
        Args:
            architecture: The model's core architecture
            
        The architecture determines the base capabilities available.
        Additional capabilities may be added based on model-specific features.
        """
        self._architecture = architecture
        self._capabilities: Dict[str, ModelCapability] = {}
        self._load_architecture_capabilities()
    
    def _load_architecture_capabilities(self) -> None:
        """Load base capabilities based on architecture."""
        # Base capabilities for all architectures
        self._capabilities["basic_inference"] = ModelCapability(
            name="basic_inference",
            confidence=1.0,
            requires_finetuning=False,
            parameters={"max_length": 2048}
        )
        
        # Architecture-specific capabilities
        if self._architecture == ModelArchitecture.ENCODER_ONLY:
            self._capabilities["text_classification"] = ModelCapability(
                name="text_classification",
                confidence=0.9,
                requires_finetuning=True,
                parameters={"num_labels": 2}
            )
            self._capabilities["token_classification"] = ModelCapability(
                name="token_classification",
                confidence=0.9,
                requires_finetuning=True
            )
            
        elif self._architecture == ModelArchitecture.DECODER_ONLY:
            self._capabilities["text_generation"] = ModelCapability(
                name="text_generation",
                confidence=1.0,
                requires_finetuning=False,
                parameters={
                    "max_new_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 1.0
                }
            )
            self._capabilities["chat"] = ModelCapability(
                name="chat",
                confidence=0.8,
                requires_finetuning=True,
                parameters={"system_prompt": None}
            )
            
        elif self._architecture == ModelArchitecture.ENCODER_DECODER:
            self._capabilities["translation"] = ModelCapability(
                name="translation",
                confidence=0.9,
                requires_finetuning=True,
                parameters={
                    "src_lang": "en",
                    "tgt_lang": "fr"
                }
            )
            self._capabilities["summarization"] = ModelCapability(
                name="summarization",
                confidence=0.9,
                requires_finetuning=True,
                parameters={
                    "min_length": 50,
                    "max_length": 200
                }
            )
            
        elif self._architecture == ModelArchitecture.VISION_ENCODER:
            self._capabilities["image_classification"] = ModelCapability(
                name="image_classification",
                confidence=1.0,
                requires_finetuning=True
            )
            self._capabilities["image_embeddings"] = ModelCapability(
                name="image_embeddings",
                confidence=1.0,
                requires_finetuning=False
            )
            
        elif self._architecture == ModelArchitecture.MULTIMODAL:
            self._capabilities["image_to_text"] = ModelCapability(
                name="image_to_text",
                confidence=0.9,
                requires_finetuning=False,
                parameters={"max_new_tokens": 256}
            )
            self._capabilities["visual_qa"] = ModelCapability(
                name="visual_qa",
                confidence=0.8,
                requires_finetuning=True
            )
            
        elif self._architecture == ModelArchitecture.SPEECH:
            self._capabilities["text_to_speech"] = ModelCapability(
                name="text_to_speech",
                confidence=0.9,
                requires_finetuning=False,
                parameters={"sample_rate": 16000}
            )
            self._capabilities["speech_to_text"] = ModelCapability(
                name="speech_to_text",
                confidence=0.9,
                requires_finetuning=False
            )
    
    def can_handle(self, capability: str) -> bool:
        """Check if model can handle a specific capability.
        
        Args:
            capability: Name of the capability to check
            
        Returns:
            bool: True if the model can handle the capability
        """
        return capability in self._capabilities
    
    def get_config(self, capability: str) -> Dict[str, Any]:
        """Get capability-specific configuration.
        
        Args:
            capability: Name of the capability
            
        Returns:
            Configuration dictionary for the capability
            
        Raises:
            InvalidInputError: If capability is not supported
        """
        if not self.can_handle(capability):
            logger.info(f"Model not specifically trained for {capability}")
            raise InvalidInputError(f"Capability {capability} not supported")
        return self._capabilities[capability].parameters.copy()
    
    def add_capability(self, capability: ModelCapability) -> None:
        """Add a new capability to the model.
        
        Args:
            capability: Capability to add
            
        This allows adding model-specific capabilities beyond
        what the architecture provides by default.
        """
        self._capabilities[capability.name] = capability
    
    def remove_capability(self, capability_name: str) -> None:
        """Remove a capability from the model.
        
        Args:
            capability_name: Name of capability to remove
            
        This is useful when a model doesn't support a capability
        that its architecture typically provides.
        """
        self._capabilities.pop(capability_name, None)
    
    def get_all_capabilities(self) -> Dict[str, ModelCapability]:
        """Get all available capabilities.
        
        Returns:
            Dictionary of capability names to their definitions
        """
        return self._capabilities.copy()
    
    def check_requirements(self, 
                         capability: str,
                         system_info: Dict[str, Any]) -> bool:
        """Check if system meets capability requirements.
        
        Args:
            capability: Name of the capability to check
            system_info: Dictionary of system information
            
        Returns:
            bool: True if all requirements are met
        """
        if not self.can_handle(capability):
            return False
            
        cap = self._capabilities[capability]
        if not cap.requirements:
            return True
            
        return all(
            req in system_info and system_info[req] == value
            for req, value in cap.requirements.items()
        )

@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    architecture: ModelArchitecture
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    quantization: Optional[str] = None  # "4bit", "8bit", None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    use_safetensors: bool = True
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    revision: Optional[str] = None

class BasePipeline(ABC):
    """Base pipeline for all model types."""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._config = None
        self._generation_config = None
        self._is_loaded = False
        
        # Initialize device
        self.device = self._get_optimal_device()
    
    @staticmethod
    def _get_optimal_device() -> str:
        """Determine the optimal device for model loading."""
        try:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception as e:
            logger.warning(f"Error detecting optimal device: {e}")
        return "cpu"
    
    @abstractmethod
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load model and components."""
        pass
    
    @abstractmethod
    def process(self, 
               inputs: List[MediaInput], 
               generation_config: Optional[Dict[str, Any]] = None,
               stream: bool = False,
               **kwargs) -> Union[str, Generator[str, None, None], Dict[str, Any]]:
        """Process inputs and generate output."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            try:
                # Move model to CPU before deletion
                if hasattr(self._model, 'cpu'):
                    self._model.cpu()
                del self._model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
        
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_loaded = False
        gc.collect() 