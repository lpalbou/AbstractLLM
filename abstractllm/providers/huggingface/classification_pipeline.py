"""
Text Classification pipeline implementation for HuggingFace provider.

This pipeline handles text classification tasks including:
- Sentiment analysis
- Topic classification
- Intent detection
- Multi-label classification
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator
import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer
)

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadError, InvalidInputError, GenerationError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities

# Configure logger
logger = logging.getLogger(__name__)

class TextClassificationPipeline(BasePipeline):
    """Pipeline for text classification tasks.
    
    This pipeline supports various classification tasks:
    - Sentiment analysis
    - Topic classification
    - Intent detection
    - Multi-label classification
    
    It handles both single-label and multi-label classification,
    with proper probability/confidence scoring.
    """
    
    def __init__(self) -> None:
        """Initialize the classification pipeline."""
        super().__init__()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_config: Optional[ModelConfig] = None
        self._is_multilabel: bool = False
        self._label_names: List[str] = []
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the classification model and tokenizer.
        
        Args:
            model_name: Name or path of the model
            config: Model configuration
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Load configuration to check model type
            model_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=config.trust_remote_code
            )
            
            # Detect if multi-label classification
            self._is_multilabel = getattr(model_config, "problem_type", None) == "multi_label_classification"
            
            # Get label names if available
            self._label_names = getattr(model_config, "id2label", {})
            if isinstance(self._label_names, dict):
                self._label_names = [self._label_names[i] for i in sorted(self._label_names.keys())]
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
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
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=model_config,
                **load_kwargs
            )
            
            # Move model to device if not using device_map="auto"
            if config.device_map != "auto":
                self.model.to(self.device)
            
            self.model_config = config
            self._is_loaded = True
            
            logger.info(f"Loaded classification model {model_name} with {len(self._label_names)} labels")
            logger.debug(f"Labels: {self._label_names}")
            logger.debug(f"Multi-label: {self._is_multilabel}")
            
        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}")
    
    def process(self, 
                inputs: List[MediaInput], 
                generation_config: Optional[Dict[str, Any]] = None,
                stream: bool = False,
                **kwargs) -> Dict[str, Any]:
        """Process text input and return classification results.
        
        Args:
            inputs: List of text inputs to classify
            generation_config: Optional generation parameters
            stream: Not used for classification
            **kwargs: Additional arguments including:
                - return_all_scores: Return scores for all labels
                - threshold: Score threshold for multi-label (default: 0.5)
                
        Returns:
            Dictionary containing:
            - labels: List of predicted labels
            - scores: Corresponding confidence scores
            - all_scores: Optional scores for all labels
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Get text from inputs
            text_inputs = [inp for inp in inputs if inp.media_type == "text"]
            if not text_inputs:
                raise InvalidInputError("No text input provided")
            
            # Combine all text inputs
            text = " ".join(
                inp.to_provider_format("huggingface")["content"] 
                for inp in text_inputs
            )
            
            # Get processing parameters
            return_all_scores = kwargs.get("return_all_scores", False)
            threshold = kwargs.get("threshold", 0.5)
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.model.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process logits based on task type
            if self._is_multilabel:
                # Multi-label: Apply sigmoid and threshold
                probs = torch.sigmoid(outputs.logits)
                predictions = (probs > threshold).float()
                scores = probs.squeeze().tolist()
                
                # Get predicted labels
                if isinstance(scores, float):
                    scores = [scores]  # Handle single label case
                labels = [
                    self._label_names[i] 
                    for i, score in enumerate(scores) 
                    if score > threshold
                ]
                
                result = {
                    "labels": labels,
                    "scores": [score for score in scores if score > threshold]
                }
                
            else:
                # Single-label: Apply softmax
                probs = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1)
                scores = probs.squeeze().tolist()
                
                # Handle single label case
                if isinstance(scores, float):
                    scores = [scores]
                
                # Get predicted label
                label_idx = prediction.item()
                label = self._label_names[label_idx] if self._label_names else str(label_idx)
                
                result = {
                    "labels": [label],
                    "scores": [scores[label_idx]]
                }
            
            # Add all scores if requested
            if return_all_scores:
                result["all_scores"] = {
                    self._label_names[i]: score
                    for i, score in enumerate(scores)
                }
            
            return result
            
        except Exception as e:
            raise GenerationError(f"Classification failed: {e}")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            input_types={"text"},
            output_types={"text"},
            supports_streaming=False,
            supports_system_prompt=False,
            context_window=self._get_context_window()
        )
    
    def _get_context_window(self) -> Optional[int]:
        """Get the model's context window size."""
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "max_position_embeddings"):
                return self.model.config.max_position_embeddings
            elif hasattr(self.model.config, "max_length"):
                return self.model.config.max_length
        return None 