"""
HuggingFace provider implementation for AbstractLLM.

This module provides the HuggingFace provider implementation, which:
- Manages model loading and pipeline selection
- Handles capability detection and validation
- Provides model recommendations
- Manages system requirements
"""

import os
import logging
import platform
import psutil
from typing import Optional, Dict, Any, Union, List, Generator, Type
from pathlib import Path
import torch
from huggingface_hub import HfApi, ModelFilter
from typing import Tuple

from abstractllm.interface import AbstractLLMInterface, ModelParameter
from abstractllm.exceptions import (
    ModelLoadingError,
    UnsupportedFeatureError,
    InvalidInputError,
    GenerationError,
    ResourceError
)
from abstractllm.media.factory import MediaFactory
from abstractllm.media.interface import MediaInput
from abstractllm.utils.logging import log_request, log_response

from .model_types import ModelArchitecture, ModelCapability, ModelCapabilities
from .config import ModelConfig, DeviceType
from .pipeline import (
    BasePipeline,
    EncoderDecoderPipeline,
    DecoderOnlyPipeline,
    EncoderOnlyPipeline,
    VisionEncoderPipeline,
    SpeechPipeline
)

# Configure logger
logger = logging.getLogger(__name__)

class HuggingFaceProvider(AbstractLLMInterface):
    """HuggingFace provider implementation.
    
    This provider handles:
    - Model loading and pipeline selection
    - Capability detection and validation
    - Model recommendations
    - System requirements management
    """
    
    # Pipeline mapping
    PIPELINE_MAPPING = {
        ModelArchitecture.ENCODER_DECODER: EncoderDecoderPipeline,
        ModelArchitecture.DECODER_ONLY: DecoderOnlyPipeline,
        ModelArchitecture.ENCODER_ONLY: EncoderOnlyPipeline,
        ModelArchitecture.VISION_ENCODER: VisionEncoderPipeline,
        ModelArchitecture.SPEECH: SpeechPipeline
    }
    
    # Model recommendations
    MODEL_RECOMMENDATIONS = {
        "text-generation": [
            ("meta-llama/Llama-2-7b-chat-hf", "High-quality chat model"),
            ("microsoft/phi-2", "Efficient general-purpose model"),
            ("mistralai/Mistral-7B-v0.1", "Strong open-source model")
        ],
        "text2text": [
            ("google/flan-t5-base", "Versatile text-to-text model"),
            ("facebook/bart-large", "Strong summarization model"),
            ("t5-base", "General-purpose T5 model")
        ],
        "vision": [
            ("openai/clip-vit-base-patch32", "Strong vision-language model"),
            ("microsoft/git-base", "Good for image captioning"),
            ("Salesforce/blip-image-captioning-base", "Efficient image understanding")
        ],
        "speech": [
            ("openai/whisper-base", "Reliable speech recognition"),
            ("microsoft/speecht5_tts", "High-quality text-to-speech"),
            ("facebook/wav2vec2-base", "Good for speech processing")
        ]
    }
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """Initialize the HuggingFace provider.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        
        # Initialize components
        self._pipeline: Optional[BasePipeline] = None
        self._model_config: Optional[ModelConfig] = None
        self._hf_api = HfApi()
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "microsoft/phi-2",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 2048,
            "trust_remote_code": True,
            "use_flash_attention": True,
            "device_map": "auto",
            "torch_dtype": "auto",
            "use_safetensors": True
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration from provider config."""
        # Detect if running on macOS
        is_macos = platform.system().lower() == "darwin"
        
        # Disable Flash Attention 2 on macOS
        use_flash_attention = False if is_macos else self.config_manager.get_param("use_flash_attention", True)
        
        # Get model architecture
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        architecture = self._detect_model_architecture(model_name)
        
        # Create configuration
        return ModelConfig(
            architecture=architecture,
            name=model_name,
            base=dict(
                trust_remote_code=self.config_manager.get_param("trust_remote_code", True),
                use_flash_attention=use_flash_attention,
                device_map=self.config_manager.get_param("device_map", "auto"),
                torch_dtype=self.config_manager.get_param("torch_dtype", "auto"),
                use_safetensors=self.config_manager.get_param("use_safetensors", True)
            ),
            generation=dict(
                max_new_tokens=self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048),
                temperature=self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7),
                top_p=self.config_manager.get_param("top_p", 1.0),
                do_sample=True
            )
        )
    
    def _detect_model_architecture(self, model_name: str) -> ModelArchitecture:
        """Detect model architecture from model name or configuration."""
        try:
            # Try HuggingFace Hub API first
            model_info = self._hf_api.model_info(model_name)
            pipeline_tag = model_info.pipeline_tag
            
            # Map pipeline tags to architectures
            TAG_MAPPING = {
                "text-generation": ModelArchitecture.DECODER_ONLY,
                "text2text-generation": ModelArchitecture.ENCODER_DECODER,
                "image-to-text": ModelArchitecture.VISION_ENCODER,
                "automatic-speech-recognition": ModelArchitecture.SPEECH,
                "text-classification": ModelArchitecture.ENCODER_ONLY
            }
            
            if pipeline_tag in TAG_MAPPING:
                return TAG_MAPPING[pipeline_tag]
            
        except Exception as e:
            logger.debug(f"Could not get model info from HF Hub: {e}")
        
        # Fallback to name-based detection
        name_lower = model_name.lower()
        
        if any(x in name_lower for x in ["gpt", "llama", "phi", "falcon"]):
            return ModelArchitecture.DECODER_ONLY
        elif any(x in name_lower for x in ["t5", "bart", "mt5"]):
            return ModelArchitecture.ENCODER_DECODER
        elif any(x in name_lower for x in ["bert", "roberta", "deberta"]):
            return ModelArchitecture.ENCODER_ONLY
        elif any(x in name_lower for x in ["vit", "clip", "blip"]):
            return ModelArchitecture.VISION_ENCODER
        elif any(x in name_lower for x in ["whisper", "wav2vec", "speecht5"]):
            return ModelArchitecture.SPEECH
        
        # Default to decoder-only
        return ModelArchitecture.DECODER_ONLY
    
    def _create_pipeline(self, config: ModelConfig) -> BasePipeline:
        """Create appropriate pipeline for model architecture."""
        if config.architecture not in self.PIPELINE_MAPPING:
            raise UnsupportedFeatureError(
                str(config.architecture),
                "Architecture not supported",
                provider="huggingface"
            )
        
        pipeline_class = self.PIPELINE_MAPPING[config.architecture]
        return pipeline_class()
    
    def _check_system_requirements(self, config: ModelConfig) -> None:
        """Check if system meets model requirements.
        
        Args:
            config: Model configuration
            
        Raises:
            ResourceError: If system requirements not met
        """
        # Check device requirements
        if config.base.device_type == DeviceType.CUDA:
            if not torch.cuda.is_available():
                raise ResourceError(
                    "CUDA device requested but not available",
                    provider="huggingface",
                    details={"device": "cuda"}
                )
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if config.base.max_memory and "cuda:0" in config.base.max_memory:
                required = self._parse_memory_string(config.base.max_memory["cuda:0"])
                if gpu_memory < required:
                    raise ResourceError(
                        f"Insufficient GPU memory: {gpu_memory} < {required}",
                        provider="huggingface",
                        details={
                            "available": gpu_memory,
                            "required": required
                        }
                    )
        
        # Check CPU memory
        available_memory = psutil.virtual_memory().available
        if config.base.max_memory and "cpu" in config.base.max_memory:
            required = self._parse_memory_string(config.base.max_memory["cpu"])
            if available_memory < required:
                raise ResourceError(
                    f"Insufficient CPU memory: {available_memory} < {required}",
                    provider="huggingface",
                    details={
                        "available": available_memory,
                        "required": required
                    }
                )
    
    @staticmethod
    def _parse_memory_string(memory_str: str) -> int:
        """Parse memory string (e.g., '4GiB') to bytes."""
        units = {
            'B': 1,
            'KB': 1024,
            'MB': 1024**2,
            'GB': 1024**3,
            'TB': 1024**4,
            'KiB': 1024,
            'MiB': 1024**2,
            'GiB': 1024**3,
            'TiB': 1024**4
        }
        
        # Remove spaces and convert to upper case
        memory_str = memory_str.replace(" ", "").upper()
        
        # Find the unit
        unit = ""
        for u in sorted(units.keys(), key=len, reverse=True):
            if memory_str.endswith(u):
                unit = u
                memory_str = memory_str[:-len(u)]
                break
        
        if not unit:
            return int(memory_str)
        
        try:
            number = float(memory_str)
            return int(number * units[unit])
        except ValueError:
            raise ValueError(f"Invalid memory string format: {memory_str}")
    
    def get_model_recommendations(self, task: str) -> List[Dict[str, str]]:
        """Get model recommendations for a specific task.
        
        Args:
            task: Task type (e.g., 'text-generation', 'vision')
            
        Returns:
            List of recommended models with descriptions
        """
        if task not in self.MODEL_RECOMMENDATIONS:
            return []
        
        return [
            {"model": model, "description": desc}
            for model, desc in self.MODEL_RECOMMENDATIONS[task]
        ]
    
    def update_model_recommendations(self, task: str, recommendations: List[Tuple[str, str]]) -> None:
        """Update model recommendations for a task.
        
        Args:
            task: Task type
            recommendations: List of (model_name, description) tuples
        """
        self.MODEL_RECOMMENDATIONS[task] = recommendations
    
    def load_model(self) -> None:
        """Load the model using appropriate pipeline."""
        try:
            # Create model configuration
            self._model_config = self._create_model_config()
            
            # Check system requirements
            self._check_system_requirements(self._model_config)
            
            # Create and load pipeline
            self._pipeline = self._create_pipeline(self._model_config)
            self._pipeline.load(
                self._model_config.name,
                self._model_config
            )
            
        except Exception as e:
            if self._pipeline:
                self._pipeline.cleanup()
            self._pipeline = None
            raise ModelLoadingError(f"Failed to load model: {e}")
    
    def generate(self,
                prompt: str,
                system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text using the loaded pipeline."""
        if not self._pipeline:
            self.load_model()
        
        try:
            # Convert all inputs to MediaInput objects
            inputs = []
            
            # Add system prompt if provided
            if system_prompt:
                system_input = MediaFactory.from_source(system_prompt, media_type="text")
                inputs.append(system_input)
            
            # Add main prompt
            text_input = MediaFactory.from_source(prompt, media_type="text")
            inputs.append(text_input)
            
            # Process files if any
            if files:
                for file_path in files:
                    media_input = MediaFactory.from_source(file_path)
                    inputs.append(media_input)
            
            # Get generation config
            generation_config = {
                "temperature": self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7),
                "max_new_tokens": self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048),
                "do_sample": True
            }
            generation_config.update(kwargs)
            
            # Log request
            log_request("huggingface", prompt, {
                "model": self.config_manager.get_param(ModelParameter.MODEL),
                "temperature": generation_config["temperature"],
                "max_tokens": generation_config["max_new_tokens"],
                "has_system_prompt": system_prompt is not None,
                "stream": stream,
                "has_files": bool(files)
            })
            
            # Generate
            result = self._pipeline.process(
                inputs=inputs,
                generation_config=generation_config,
                stream=stream
            )
            
            # Process through output handlers and return
            return self._handle_output(result)
            
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}")
    
    async def generate_async(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           files: Optional[List[Union[str, Path]]] = None,
                           stream: bool = False,
                           **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text asynchronously."""
        # For now, we'll use the synchronous implementation in a thread pool
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        with ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                self.generate,
                prompt,
                system_prompt,
                files,
                stream,
                **kwargs
            )
            
            # Process through output handlers and return
            return self._handle_output(result)
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """Get provider capabilities."""
        if not self._pipeline:
            self.load_model()
        
        return self._pipeline.capabilities.get_all_capabilities()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._pipeline:
            self._pipeline.cleanup()
        self._pipeline = None 