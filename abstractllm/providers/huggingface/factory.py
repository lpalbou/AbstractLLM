"""
Pipeline factory for HuggingFace provider.

This module provides centralized pipeline creation and model type detection.
All model architecture detection and pipeline mapping is handled here to ensure
consistency across the system.
"""

import logging
from typing import Optional, Dict, Any, Tuple, Type
from importlib import import_module

from abstractllm.exceptions import UnsupportedModelError
from .model_types import BasePipeline, ModelConfig, ModelArchitecture

# Configure logger
logger = logging.getLogger(__name__)

class PipelineFactory:
    """Factory for creating model pipelines."""
    
    # Mapping of pipeline types to their module paths
    _PIPELINE_MAPPING = {
        # Currently Implemented
        "text-generation": ".text_pipeline.TextGenerationPipeline",
        "text2text-generation": ".text2text_pipeline.Text2TextPipeline",
        "image-to-text": ".vision_pipeline.ImageToTextPipeline",
        "question-answering": ".qa_pipeline.QuestionAnsweringPipeline",
        "text-classification": ".classification_pipeline.TextClassificationPipeline",
        "document-question-answering": ".docqa_pipeline.DocumentQuestionAnsweringPipeline",
        "text-to-speech": ".tts_pipeline.TextToSpeechPipeline",
        
        # Planned for future implementation
        "token-classification": None,
        "visual-question-answering": None,
        "automatic-speech-recognition": None
    }
    
    # Model name patterns for architecture detection
    _MODEL_PATTERNS = {
        "text-generation": [
            "gpt", "llama", "phi", "falcon", "bloom", "opt", "pythia", "mistral",
            "mpt", "stablelm", "rwkv", "qwen", "yi", "mixtral"
        ],
        "text2text-generation": [
            "t5", "bart", "mt5", "mbart", "pegasus", "flan"
        ],
        "image-to-text": [
            "blip", "llava", "git", "kosmos", "fuyu", "cogvlm", "qwen-vl"
        ],
        "question-answering": [
            "bert-qa", "roberta-qa", "deberta-qa", "electra-qa", "squad"
        ],
        "text-classification": [
            "sentiment", "topic", "class", "intent", "emotion", "bert-class",
            "roberta-class", "deberta-class"
        ],
        "document-question-answering": [
            "layoutlm", "donut", "docvqa", "mathpix"
        ],
        "visual-question-answering": [
            "vilt-vqa", "blip-vqa", "flamingo"
        ],
        "text-to-speech": [
            "speecht5", "vits", "bark", "coqui", "tts", "musicgen", "audiogen"
        ]
    }
    
    @classmethod
    def _import_pipeline_class(cls, pipeline_type: str) -> Type[BasePipeline]:
        """Import pipeline class dynamically."""
        if pipeline_type not in cls._PIPELINE_MAPPING:
            raise UnsupportedModelError(f"Pipeline type {pipeline_type} not supported")
            
        module_path = cls._PIPELINE_MAPPING[pipeline_type]
        if module_path is None:
            raise UnsupportedModelError(
                f"Pipeline type {pipeline_type} is planned but not yet implemented"
            )
            
        try:
            # Import relative to current package
            module_name, class_name = module_path.rsplit(".", 1)
            module = import_module(module_name, package=__package__)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise UnsupportedModelError(f"Failed to import pipeline {pipeline_type}: {e}")
    
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
        
        try:
            # Import pipeline class dynamically
            pipeline_class = cls._import_pipeline_class(model_type)
            
            # Create default config if none provided
            if model_config is None:
                model_config = ModelConfig(architecture=architecture)
                
            return pipeline_class()
            
        except Exception as e:
            raise UnsupportedModelError(f"Failed to create pipeline for {model_name}: {e}")
    
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
            
        Note:
            The detection follows this priority:
            1. HuggingFace Hub API pipeline tags
            2. Model name pattern matching
            3. Default to text-generation (DECODER_ONLY)
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
                "text-to-speech": "text-to-speech",
                "automatic-speech-recognition": "automatic-speech-recognition",
                
                # Additional mappings
                "feature-extraction": "text-classification",
                "fill-mask": "text-classification",
                "token-classification": "text-classification",
                "zero-shot-classification": "text-classification",
                "conversational": "text-generation"
            }
            
            model_type = TAG_MAPPING.get(pipeline_tag, "text-generation")
            
        except Exception as e:
            logger.debug(f"Could not get model info from HF Hub: {e}")
            
            # Fallback to name-based detection
            model_name_lower = model_name.lower()
            
            # Special case for GGUF models
            if model_name.endswith('.gguf'):
                return "text-generation", ModelArchitecture.DECODER_ONLY
            
            # Check patterns for each model type
            for pipeline_type, patterns in cls._MODEL_PATTERNS.items():
                if any(pattern in model_name_lower for pattern in patterns):
                    model_type = pipeline_type
                    break
            else:
                # Default to text generation if no pattern matches
                model_type = "text-generation"
        
        # Map pipeline type to architecture
        ARCHITECTURE_MAPPING = {
            "text-generation": ModelArchitecture.DECODER_ONLY,
            "text2text-generation": ModelArchitecture.ENCODER_DECODER,
            "image-to-text": ModelArchitecture.VISION_ENCODER,
            "question-answering": ModelArchitecture.ENCODER_ONLY,
            "text-classification": ModelArchitecture.ENCODER_ONLY,
            "document-question-answering": ModelArchitecture.MULTIMODAL,
            "visual-question-answering": ModelArchitecture.MULTIMODAL,
            "text-to-speech": ModelArchitecture.SPEECH,
            "automatic-speech-recognition": ModelArchitecture.SPEECH
        }
        
        architecture = ARCHITECTURE_MAPPING.get(model_type, ModelArchitecture.DECODER_ONLY)
        
        logger.debug(f"Detected pipeline type '{model_type}' and architecture '{architecture}' for model '{model_name}'")
        return model_type, architecture 