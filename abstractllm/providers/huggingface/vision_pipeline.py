"""
Vision model pipeline implementation for HuggingFace provider.
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator, Set
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    LlavaProcessor, LlavaForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq
)

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadingError, GenerationError, UnsupportedFeatureError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities, ModelArchitecture

# Configure logger
logger = logging.getLogger(__name__)

class ImageToTextPipeline(BasePipeline):
    """Pipeline for image-to-text models."""
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the image-to-text model."""
        try:
            # Determine model architecture and load appropriate model
            if "blip" in model_name.lower():
                self._load_blip(model_name, config)
            elif "llava" in model_name.lower():
                self._load_llava(model_name, config)
            else:
                self._load_vision2seq(model_name, config)
            self._is_loaded = True
        except Exception as e:
            self.cleanup()
            raise ModelLoadingError(f"Failed to load model: {e}")
    
    def _load_blip(self, model_name: str, config: ModelConfig) -> None:
        """Load a BLIP model."""
        load_kwargs = self._get_load_kwargs(config)
        
        self._processor = BlipProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code
        )
        self._model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Store model type for processing
        self._model_type = "blip"
    
    def _load_llava(self, model_name: str, config: ModelConfig) -> None:
        """Load a LLaVA model."""
        load_kwargs = self._get_load_kwargs(config)
        
        self._processor = LlavaProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code
        )
        self._model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Store model type for processing
        self._model_type = "llava"
    
    def _load_vision2seq(self, model_name: str, config: ModelConfig) -> None:
        """Load a generic vision-to-sequence model."""
        load_kwargs = self._get_load_kwargs(config)
        
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Store model type for processing
        self._model_type = "vision2seq"
    
    def _get_load_kwargs(self, config: ModelConfig) -> Dict[str, Any]:
        """Get model loading kwargs based on config."""
        load_kwargs = {
            "device_map": config.device_map,
            "torch_dtype": config.torch_dtype,
            "use_safetensors": config.use_safetensors,
            "trust_remote_code": config.trust_remote_code,
            "use_flash_attention_2": config.use_flash_attention
        }
        
        # Add quantization if specified
        if config.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs.update({
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            })
        elif config.quantization == "8bit":
            load_kwargs.update({"load_in_8bit": True})
            
        return load_kwargs
    
    def process(self, 
               inputs: List[MediaInput], 
               generation_config: Optional[Dict[str, Any]] = None,
               stream: bool = False,
               **kwargs) -> str:
        """Process inputs and generate text."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
            
        try:
            # Get image and text inputs
            image_input = self._get_image_input(inputs)
            text_input = self._get_text_input(inputs)
            
            # Process based on model type
            if self._model_type == "blip":
                return self._process_blip(image_input, text_input, generation_config)
            elif self._model_type == "llava":
                return self._process_llava(image_input, text_input, generation_config)
            else:  # vision2seq
                return self._process_vision2seq(image_input, text_input, generation_config)
                
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}")
    
    def _get_image_input(self, inputs: List[MediaInput]) -> MediaInput:
        """Get the image input from the list of inputs."""
        image_inputs = [inp for inp in inputs if inp.media_type == "image"]
        if not image_inputs:
            raise ValueError("Image input required for image-to-text model")
        if len(image_inputs) > 1:
            logger.warning("Multiple images provided, using only the first one")
        return image_inputs[0]
    
    def _get_text_input(self, inputs: List[MediaInput]) -> Optional[str]:
        """Get the text input from the list of inputs."""
        text_inputs = [inp for inp in inputs if inp.media_type == "text"]
        if text_inputs:
            formatted = text_inputs[0].to_provider_format("huggingface")
            return formatted["content"]
        return None
    
    def _process_blip(self, 
                     image_input: MediaInput,
                     text_input: Optional[str],
                     generation_config: Optional[Dict[str, Any]]) -> str:
        """Process inputs with BLIP model."""
        # Format image for BLIP
        formatted_image = image_input.to_provider_format("huggingface")
        
        # Prepare inputs
        inputs = self._processor(
            images=formatted_image["content"],
            text=text_input if text_input else "",
            return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self._model.generate(
            **inputs,
            **(generation_config or {})
        )
        
        return self._processor.decode(outputs[0], skip_special_tokens=True)
    
    def _process_llava(self,
                      image_input: MediaInput,
                      text_input: Optional[str],
                      generation_config: Optional[Dict[str, Any]]) -> str:
        """Process inputs with LLaVA model."""
        # Format image for LLaVA
        formatted_image = image_input.to_provider_format("huggingface")
        
        # Prepare inputs
        inputs = self._processor(
            images=formatted_image["content"],
            text=text_input if text_input else "Describe this image.",
            return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self._model.generate(
            **inputs,
            **(generation_config or {})
        )
        
        return self._processor.decode(outputs[0], skip_special_tokens=True)
    
    def _process_vision2seq(self,
                          image_input: MediaInput,
                          text_input: Optional[str],
                          generation_config: Optional[Dict[str, Any]]) -> str:
        """Process inputs with vision-to-sequence model."""
        # Format image for vision2seq
        formatted_image = image_input.to_provider_format("huggingface")
        
        # Prepare inputs
        inputs = self._processor(
            images=formatted_image["content"],
            text=text_input if text_input else None,
            return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self._model.generate(
            **inputs,
            **(generation_config or {})
        )
        
        return self._processor.decode(outputs[0], skip_special_tokens=True)
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            input_types={"image", "text"},
            output_types={"text"},
            supports_streaming=False,
            supports_system_prompt=False,
            context_window=self._get_context_window()
        )
    
    def _get_context_window(self) -> Optional[int]:
        """Get the model's context window size."""
        if hasattr(self._model, "config"):
            if hasattr(self._model.config, "max_position_embeddings"):
                return self._model.config.max_position_embeddings
            elif hasattr(self._model.config, "max_length"):
                return self._model.config.max_length
        return None 