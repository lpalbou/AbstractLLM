"""
Visual Question Answering pipeline implementation for HuggingFace provider.

This pipeline handles visual question answering tasks including:
- General VQA (answering questions about images)
- Scene understanding
- Object detection and counting
- Spatial relationship understanding
- Image attribute analysis
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    ViltProcessor,
    ViltForQuestionAnswering,
    BlipProcessor,
    BlipForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer
)
from PIL import Image
import numpy as np

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadError, InvalidInputError, GenerationError, UnsupportedFeatureError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities

# Configure logger
logger = logging.getLogger(__name__)

class VisualQuestionAnsweringPipeline(BasePipeline):
    """Pipeline for visual question answering tasks.
    
    This pipeline supports various VQA architectures:
    - ViLT: Vision-and-Language Transformer
    - BLIP: Bootstrapping Language-Image Pre-training
    - Other VQA-specific architectures
    
    It handles both open-ended and multiple-choice questions,
    with proper confidence scoring and answer generation.
    """
    
    # Model architecture mapping
    ARCHITECTURES = {
        "vilt": (ViltProcessor, ViltForQuestionAnswering),
        "blip": (BlipProcessor, BlipForQuestionAnswering)
    }
    
    def __init__(self) -> None:
        """Initialize the VQA pipeline."""
        super().__init__()
        self.model: Optional[PreTrainedModel] = None
        self.processor = None
        self.model_config: Optional[ModelConfig] = None
        self._model_type: str = ""
        self._max_question_length: int = 512
        self._supports_multiple_choice: bool = False
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the VQA model and processor.
        
        Args:
            model_name: Name or path of the model
            config: Model configuration
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Detect model architecture
            self._model_type = self._detect_architecture(model_name)
            processor_class, model_class = self.ARCHITECTURES.get(
                self._model_type, 
                (AutoProcessor, AutoConfig)
            )
            
            # Load processor/tokenizer
            self.processor = processor_class.from_pretrained(
                model_name,
                trust_remote_code=config.trust_remote_code
            )
            
            # Prepare loading kwargs
            load_kwargs = {
                "trust_remote_code": config.trust_remote_code,
                "device_map": config.device_map,
                "torch_dtype": config.torch_dtype
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
            
            # Load model
            self.model = model_class.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Move model to device if not using device_map="auto"
            if config.device_map != "auto":
                self.model.to(self.device)
            
            # Get model capabilities
            model_config = self.model.config
            self._max_question_length = getattr(model_config, "max_position_embeddings", 512)
            self._supports_multiple_choice = hasattr(model_config, "num_choices")
            
            self.model_config = config
            self._is_loaded = True
            
            logger.info(f"Loaded VQA model {model_name} ({self._model_type})")
            logger.debug(f"Max question length: {self._max_question_length}")
            logger.debug(f"Supports multiple choice: {self._supports_multiple_choice}")
            
        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}")
    
    def _detect_architecture(self, model_name: str) -> str:
        """Detect the model architecture from the model name."""
        name_lower = model_name.lower()
        if "vilt" in name_lower:
            return "vilt"
        elif "blip" in name_lower:
            return "blip"
        else:
            # Default to ViLT for unknown architectures
            logger.warning(f"Unknown model architecture for {model_name}, defaulting to ViLT")
            return "vilt"
    
    def process(self, 
                inputs: List[MediaInput], 
                generation_config: Optional[Dict[str, Any]] = None,
                stream: bool = False,
                **kwargs) -> Dict[str, Any]:
        """Process image and question to generate answer.
        
        Args:
            inputs: List containing image and question inputs
            generation_config: Optional generation parameters
            stream: Not used for VQA
            **kwargs: Additional arguments including:
                - answer_candidates: List of possible answers for multiple choice
                - max_answer_length: Maximum length of generated answer
                - num_beams: Number of beams for answer generation
                - return_logits: Return raw logits instead of processed answers
                
        Returns:
            Dictionary containing:
            - answer: Generated answer text
            - confidence: Confidence score
            - logits: Optional raw logits if requested
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Extract image and question from inputs
            image_input = None
            question_input = None
            
            for inp in inputs:
                if inp.media_type == "image":
                    if image_input is not None:
                        raise InvalidInputError("Multiple images provided, only one is supported")
                    image_input = inp
                elif inp.media_type == "text":
                    if question_input is not None:
                        raise InvalidInputError("Multiple questions provided, only one is supported")
                    question_input = inp
            
            if not image_input or not question_input:
                raise InvalidInputError("Both image and question inputs are required")
            
            # Get image and question
            image_data = image_input.to_provider_format("huggingface")
            question = question_input.to_provider_format("huggingface")["content"]
            
            # Get processing parameters
            answer_candidates = kwargs.get("answer_candidates")
            max_answer_length = kwargs.get("max_answer_length", 50)
            num_beams = kwargs.get("num_beams", 3)
            return_logits = kwargs.get("return_logits", False)
            
            # Handle multiple choice if supported and candidates provided
            if answer_candidates and not self._supports_multiple_choice:
                logger.warning("Model does not support multiple choice, ignoring answer candidates")
                answer_candidates = None
            
            # Prepare inputs
            if self._model_type == "vilt":
                # ViLT uses a specific input format
                inputs = self.processor(
                    images=image_data["content"],
                    text=question,
                    answer_candidates=answer_candidates,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_question_length
                ).to(self.model.device)
                
                # Generate answer
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                if answer_candidates:
                    # Multiple choice: get most likely answer
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)
                    answer_idx = torch.argmax(logits, dim=1).item()
                    confidence = probs[0][answer_idx].item()
                    answer = answer_candidates[answer_idx]
                else:
                    # Open-ended: generate answer
                    answer_ids = self.model.generate(
                        inputs["input_ids"],
                        max_length=max_answer_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                    answer = self.processor.decode(answer_ids[0], skip_special_tokens=True)
                    confidence = outputs.logits.max().item()
                
            else:  # BLIP and others
                # Standard input processing
                inputs = self.processor(
                    images=image_data["content"],
                    text=question,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_question_length
                ).to(self.model.device)
                
                # Generate answer
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_answer_length,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                    answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                    confidence = 1.0  # BLIP doesn't provide confidence scores
            
            # Prepare result
            result = {
                "answer": answer,
                "confidence": confidence
            }
            
            # Add logits if requested
            if return_logits:
                result["logits"] = outputs.logits.cpu().numpy()
            
            return result
            
        except Exception as e:
            raise GenerationError(f"VQA generation failed: {e}")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            input_types={"image", "text"},
            output_types={"text"},
            supports_streaming=False,
            supports_system_prompt=False,
            context_window=self._max_question_length
        )
    
    def _get_context_window(self) -> Optional[int]:
        """Get the model's context window size."""
        return self._max_question_length 