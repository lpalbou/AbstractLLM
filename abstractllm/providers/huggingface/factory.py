"""
Pipeline factory for HuggingFace provider.

This module provides centralized pipeline creation and model type detection.
All model architecture detection and pipeline mapping is handled here to ensure
consistency across the system.
"""

import logging
from typing import Optional, Dict, Any, Tuple, Type

from abstractllm.exceptions import UnsupportedModelError
from .model_types import BasePipeline, ModelConfig, ModelArchitecture
from .text_pipeline import TextGenerationPipeline
from .vision_pipeline import ImageToTextPipeline
from .text2text_pipeline import Text2TextPipeline
from .qa_pipeline import QuestionAnsweringPipeline
from .classification_pipeline import TextClassificationPipeline
from .docqa_pipeline import DocumentQuestionAnsweringPipeline
from .tts_pipeline import TextToSpeechPipeline

# Configure logger
logger = logging.getLogger(__name__)

class PipelineFactory:
    """Factory for creating model pipelines."""
    
    # Mapping of pipeline types to their implementations and architectures
    _PIPELINE_MAPPING = {
        # Currently Implemented
        "text-generation": TextGenerationPipeline,
        "text2text-generation": Text2TextPipeline,
        "image-to-text": ImageToTextPipeline,
        "question-answering": QuestionAnsweringPipeline,
        "text-classification": TextClassificationPipeline,
        "document-question-answering": DocumentQuestionAnsweringPipeline,
        "text-to-speech": TextToSpeechPipeline,
        
        # Planned for future implementation
        "token-classification": None,
        "visual-question-answering": None,
        "automatic-speech-recognition": None
    }
    
    # Model name patterns for architecture detection
    _MODEL_PATTERNS = {
        "text-generation": [
            "gpt", "llama", "phi", "falcon", "bloom", "opt", "pythia"
        ],
        "text2text-generation": [
            "t5", "bart", "mt5", "mbart", "pegasus"
        ],
        "image-to-text": [
            "blip", "llava", "git", "kosmos"
        ],
        "question-answering": [
            "bert-qa", "roberta-qa", "deberta-qa", "electra-qa", "squad"
        ],
        "text-classification": [
            "sentiment", "topic", "class", "intent", "emotion"
        ],
        "document-question-answering": [
            "layoutlm", "donut", "docvqa"
        ],
        "visual-question-answering": [
            "vilt-vqa", "blip-vqa"
        ],
        "text-to-speech": [
            "speecht5", "vits", "bark", "coqui", "tts"
        ]
    }
    
    @classmethod
    def create_pipeline(cls, 
                       model_name: str, 
                       model_config: Optional[ModelConfig] = None) -> BasePipeline:
        """
        Create and configure appropriate pipeline.
        
        Args:
            model_name: Name or path of the model
            model_config: Optional model configuration
            
        Returns:
            Configured pipeline instance
            
        Raises:
            UnsupportedModelError: If model type is not supported or not yet implemented
        """
        # Detect model type and architecture
        model_type, architecture = cls.detect_model_architecture(model_name)
        
        if model_type not in cls._PIPELINE_MAPPING:
            raise UnsupportedModelError(f"Model type {model_type} not supported")
            
        pipeline_class = cls._PIPELINE_MAPPING[model_type]
        
        # Check if pipeline type is implemented
        if pipeline_class is None:
            raise UnsupportedModelError(
                f"Model type {model_type} is planned but not yet implemented"
            )
        
        # Create default config if none provided
        if model_config is None:
            model_config = ModelConfig(architecture=architecture)
            
        return pipeline_class()
    
    @classmethod
    def detect_model_architecture(cls, model_name: str) -> Tuple[str, ModelArchitecture]:
        """
        Detect both model type and architecture.
        
        This is the central method for model type detection. All pipelines should
        use this method instead of implementing their own detection logic.
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Tuple of (pipeline_type, model_architecture)
        """
        try:
            # Try HuggingFace Hub API first
            from huggingface_hub import model_info
            info = model_info(model_name)
            pipeline_tag = info.pipeline_tag
            
            # Map HF pipeline tags to our pipeline types
            TAG_MAPPING = {
                # Currently Implemented
                "text-generation": "text-generation",
                "text2text-generation": "text2text-generation",
                "image-to-text": "image-to-text",
                "vision-text-generation": "image-to-text",
                "translation": "text2text-generation",
                "summarization": "text2text-generation",
                "question-answering": "question-answering",
                "extractive-qa": "question-answering",
                "generative-qa": "question-answering",
                "text-classification": "text-classification",
                "sentiment-analysis": "text-classification",
                "topic-classification": "text-classification",
                "document-question-answering": "document-question-answering",
                "visual-question-answering": "visual-question-answering",
                
                # Planned
                "token-classification": "token-classification",
                "text-to-speech": "text-to-speech",
                "automatic-speech-recognition": "automatic-speech-recognition"
            }
            
            model_type = TAG_MAPPING.get(pipeline_tag, "text-generation")
            
        except Exception as e:
            logger.debug(f"Could not get model info from HF Hub: {e}")
            
            # Fallback to name-based detection
            model_name_lower = model_name.lower()
            
            # Special case for GGUF models
            if model_name.endswith('.gguf'):
                return "text-generation", ModelArchitecture.CAUSAL_LM
            
            # Check patterns for each model type
            for pipeline_type, patterns in cls._MODEL_PATTERNS.items():
                if any(pattern in model_name_lower for pattern in patterns):
                    model_type = pipeline_type
                    break
            else:
                # Default to text generation if no pattern matches
                model_type = "text-generation"
        
        # Get corresponding architecture
        _, architecture = cls._PIPELINE_MAPPING[model_type]
        return model_type, architecture 