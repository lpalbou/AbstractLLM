"""
Pipeline system for HuggingFace provider.

This module provides a flexible pipeline system that handles:
- Model loading and resource management
- Input processing and validation
- Architecture-specific processing
- Capability management
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List, Generator, Type
import logging
import gc
import torch
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizer
)

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import (
    ModelLoadingError,
    InvalidInputError,
    GenerationError,
    UnsupportedFeatureError,
    ResourceError
)
from .model_types import ModelArchitecture, ModelCapability, ModelCapabilities
from .config import ModelConfig

# Configure logger
logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """Base pipeline for all model architectures.
    
    This class provides core functionality for:
    - Model loading and resource management
    - Input processing and validation
    - Capability management
    - Error handling
    """
    
    def __init__(self):
        """Initialize pipeline."""
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.processor: Optional[Any] = None
        self.config: Optional[ModelConfig] = None
        self._capabilities: Optional[ModelCapabilities] = None
        self._is_loaded: bool = False
        
        # Initialize device
        self.device = self._get_optimal_device()
    
    @staticmethod
    def _get_optimal_device() -> str:
        """Determine optimal device for model loading."""
        try:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception as e:
            logger.warning(f"Error detecting optimal device: {e}")
        return "cpu"
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load model and components.
        
        Args:
            model_name: Name or path of the model
            config: Model configuration
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=config.base.trust_remote_code
            )
            
            # Load model components
            self._load_components(model_name, model_config, config)
            
            # Initialize capabilities
            self._capabilities = ModelCapabilities(config.architecture)
            
            # Store configuration
            self.config = config
            self._is_loaded = True
            
            logger.info(f"Successfully loaded model {model_name}")
            
        except Exception as e:
            self.cleanup()
            raise ModelLoadingError(f"Failed to load model {model_name}: {str(e)}")
    
    @abstractmethod
    def _load_components(self, 
                        model_name: str,
                        model_config: Any,
                        config: ModelConfig) -> None:
        """Load model-specific components.
        
        Args:
            model_name: Name or path of the model
            model_config: Model configuration from AutoConfig
            config: Pipeline configuration
            
        This method should be implemented by each pipeline to load:
        - Model
        - Tokenizer
        - Processor (if needed)
        - Additional components
        """
        pass
    
    def process(self,
                inputs: List[MediaInput],
                generation_config: Optional[Dict[str, Any]] = None,
                stream: bool = False,
                **kwargs) -> Union[str, Generator[str, None, None], Dict[str, Any]]:
        """Process inputs and generate output.
        
        Args:
            inputs: List of media inputs
            generation_config: Optional generation parameters
            stream: Whether to stream the output
            **kwargs: Additional arguments
            
        Returns:
            Generated output (format depends on pipeline type)
            
        Raises:
            RuntimeError: If model not loaded
            InvalidInputError: If inputs are invalid
            GenerationError: If generation fails
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Validate inputs
            self._validate_inputs(inputs)
            
            # Process inputs
            processed_inputs = self._process_inputs(inputs)
            
            # Generate output
            return self._generate(
                processed_inputs,
                generation_config,
                stream,
                **kwargs
            )
            
        except Exception as e:
            if isinstance(e, (InvalidInputError, GenerationError)):
                raise
            raise GenerationError(f"Generation failed: {e}")
    
    @abstractmethod
    def _validate_inputs(self, inputs: List[MediaInput]) -> None:
        """Validate input types and formats.
        
        Args:
            inputs: List of media inputs
            
        Raises:
            InvalidInputError: If inputs are invalid
        """
        pass
    
    @abstractmethod
    def _process_inputs(self, inputs: List[MediaInput]) -> Dict[str, Any]:
        """Process inputs into model-ready format.
        
        Args:
            inputs: List of media inputs
            
        Returns:
            Dictionary of processed inputs
        """
        pass
    
    @abstractmethod
    def _generate(self,
                 processed_inputs: Dict[str, Any],
                 generation_config: Optional[Dict[str, Any]] = None,
                 stream: bool = False,
                 **kwargs) -> Union[str, Generator[str, None, None], Dict[str, Any]]:
        """Generate output from processed inputs.
        
        Args:
            processed_inputs: Dictionary of processed inputs
            generation_config: Optional generation parameters
            stream: Whether to stream the output
            **kwargs: Additional arguments
            
        Returns:
            Generated output
        """
        pass
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        if not self._capabilities:
            raise RuntimeError("Model not loaded, capabilities not available")
        return self._capabilities
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            try:
                # Move model to CPU before deletion
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._is_loaded = False
        gc.collect()

class EncoderDecoderPipeline(BasePipeline):
    """Pipeline for encoder-decoder models (T5, BART, etc.)."""
    
    def _load_components(self,
                        model_name: str,
                        model_config: Any,
                        config: ModelConfig) -> None:
        """Load encoder-decoder model components."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.base.trust_remote_code
        )
        
        # Ensure we have required tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            config=model_config,
            **config.get_model_kwargs()
        )
        
        # Move model to device if not using device_map="auto"
        if config.base.device_map != "auto":
            self.model.to(self.device)
    
    def _validate_inputs(self, inputs: List[MediaInput]) -> None:
        """Validate inputs for encoder-decoder models."""
        # Check for text inputs
        text_inputs = [inp for inp in inputs if inp.media_type == "text"]
        if not text_inputs:
            raise InvalidInputError("Text input required for encoder-decoder model")
    
    def _process_inputs(self, inputs: List[MediaInput]) -> Dict[str, Any]:
        """Process inputs for encoder-decoder models."""
        # Combine text inputs
        text = " ".join(
            inp.to_provider_format("huggingface")["content"]
            for inp in inputs
            if inp.media_type == "text"
        )
        
        # Tokenize
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.encoder.max_position_embeddings
        ).to(self.model.device)
    
    def _generate(self,
                 processed_inputs: Dict[str, Any],
                 generation_config: Optional[Dict[str, Any]] = None,
                 stream: bool = False,
                 **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text using encoder-decoder model."""
        # Update generation config
        gen_kwargs = {}
        if generation_config:
            gen_kwargs.update(generation_config)
        
        if stream:
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            # Set up streaming
            streamer = TextIteratorStreamer(self.tokenizer)
            generation_kwargs = dict(
                **processed_inputs,
                streamer=streamer,
                **gen_kwargs
            )
            
            # Run generation in a thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Return generator
            def stream_generator():
                for text in streamer:
                    yield text
            return stream_generator()
        else:
            # Generate without streaming
            outputs = self.model.generate(
                **processed_inputs,
                **gen_kwargs
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class DecoderOnlyPipeline(BasePipeline):
    """Pipeline for decoder-only models (GPT, LLaMA, etc.)."""
    
    def _load_components(self,
                        model_name: str,
                        model_config: Any,
                        config: ModelConfig) -> None:
        """Load decoder-only model components."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.base.trust_remote_code
        )
        
        # Ensure we have required tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            **config.get_model_kwargs()
        )
        
        # Move model to device if not using device_map="auto"
        if config.base.device_map != "auto":
            self.model.to(self.device)
    
    def _validate_inputs(self, inputs: List[MediaInput]) -> None:
        """Validate inputs for decoder-only models."""
        # Check for text inputs
        text_inputs = [inp for inp in inputs if inp.media_type == "text"]
        if not text_inputs:
            raise InvalidInputError("Text input required for decoder-only model")
    
    def _process_inputs(self, inputs: List[MediaInput]) -> Dict[str, Any]:
        """Process inputs for decoder-only models."""
        # Combine text inputs
        text = " ".join(
            inp.to_provider_format("huggingface")["content"]
            for inp in inputs
            if inp.media_type == "text"
        )
        
        # Tokenize
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.decoder.max_position_embeddings
        ).to(self.model.device)
    
    def _generate(self,
                 processed_inputs: Dict[str, Any],
                 generation_config: Optional[Dict[str, Any]] = None,
                 stream: bool = False,
                 **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text using decoder-only model."""
        # Update generation config
        gen_kwargs = {}
        if generation_config:
            gen_kwargs.update(generation_config)
        
        if stream:
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            # Set up streaming
            streamer = TextIteratorStreamer(self.tokenizer)
            generation_kwargs = dict(
                **processed_inputs,
                streamer=streamer,
                **gen_kwargs
            )
            
            # Run generation in a thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Return generator
            def stream_generator():
                for text in streamer:
                    yield text
            return stream_generator()
        else:
            # Generate without streaming
            outputs = self.model.generate(
                **processed_inputs,
                **gen_kwargs
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class EncoderOnlyPipeline(BasePipeline):
    """Pipeline for encoder-only models (BERT, RoBERTa, etc.)."""
    
    def _load_components(self,
                        model_name: str,
                        model_config: Any,
                        config: ModelConfig) -> None:
        """Load encoder-only model components."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.base.trust_remote_code
        )
        
        # Ensure we have required tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model based on task
        if hasattr(model_config, "num_labels"):
            # Classification model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=model_config,
                **config.get_model_kwargs()
            )
        else:
            # Base encoder model
            self.model = AutoModel.from_pretrained(
                model_name,
                config=model_config,
                **config.get_model_kwargs()
            )
        
        # Move model to device if not using device_map="auto"
        if config.base.device_map != "auto":
            self.model.to(self.device)
    
    def _validate_inputs(self, inputs: List[MediaInput]) -> None:
        """Validate inputs for encoder-only models."""
        # Check for text inputs
        text_inputs = [inp for inp in inputs if inp.media_type == "text"]
        if not text_inputs:
            raise InvalidInputError("Text input required for encoder-only model")
    
    def _process_inputs(self, inputs: List[MediaInput]) -> Dict[str, Any]:
        """Process inputs for encoder-only models."""
        # Combine text inputs
        text = " ".join(
            inp.to_provider_format("huggingface")["content"]
            for inp in inputs
            if inp.media_type == "text"
        )
        
        # Tokenize
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.encoder.max_position_embeddings
        ).to(self.model.device)
    
    def _generate(self,
                 processed_inputs: Dict[str, Any],
                 generation_config: Optional[Dict[str, Any]] = None,
                 stream: bool = False,
                 **kwargs) -> Dict[str, Any]:
        """Generate embeddings or classification outputs."""
        outputs = self.model(**processed_inputs)
        
        if hasattr(self.model.config, "num_labels"):
            # Classification output
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            return {
                "predictions": predictions.tolist(),
                "probabilities": probs.tolist(),
                "logits": logits.tolist()
            }
        else:
            # Embeddings output
            return {
                "embeddings": outputs.last_hidden_state.tolist(),
                "pooled": outputs.pooler_output.tolist() if hasattr(outputs, "pooler_output") else None
            }

class VisionEncoderPipeline(BasePipeline):
    """Pipeline for vision encoder models (ViT, CLIP, etc.)."""
    
    def _load_components(self,
                        model_name: str,
                        model_config: Any,
                        config: ModelConfig) -> None:
        """Load vision encoder model components."""
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.base.trust_remote_code
        )
        
        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            config=model_config,
            **config.get_model_kwargs()
        )
        
        # Move model to device if not using device_map="auto"
        if config.base.device_map != "auto":
            self.model.to(self.device)
    
    def _validate_inputs(self, inputs: List[MediaInput]) -> None:
        """Validate inputs for vision encoder models."""
        # Check for image inputs
        image_inputs = [inp for inp in inputs if inp.media_type == "image"]
        if not image_inputs:
            raise InvalidInputError("Image input required for vision encoder model")
    
    def _process_inputs(self, inputs: List[MediaInput]) -> Dict[str, Any]:
        """Process inputs for vision encoder models."""
        # Get images
        images = [
            inp.to_provider_format("huggingface")["content"]
            for inp in inputs
            if inp.media_type == "image"
        ]
        
        # Process images
        return self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.model.device)
    
    def _generate(self,
                 processed_inputs: Dict[str, Any],
                 generation_config: Optional[Dict[str, Any]] = None,
                 stream: bool = False,
                 **kwargs) -> Dict[str, Any]:
        """Generate vision outputs."""
        outputs = self.model(**processed_inputs)
        
        return {
            "embeddings": outputs.last_hidden_state.tolist(),
            "pooled": outputs.pooler_output.tolist() if hasattr(outputs, "pooler_output") else None
        }

class SpeechPipeline(BasePipeline):
    """Pipeline for speech models (Whisper, SpeechT5, etc.)."""
    
    def _load_components(self,
                        model_name: str,
                        model_config: Any,
                        config: ModelConfig) -> None:
        """Load speech model components."""
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=config.base.trust_remote_code
        )
        
        # Load model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            config=model_config,
            **config.get_model_kwargs()
        )
        
        # Move model to device if not using device_map="auto"
        if config.base.device_map != "auto":
            self.model.to(self.device)
    
    def _validate_inputs(self, inputs: List[MediaInput]) -> None:
        """Validate inputs for speech models."""
        # Check for audio inputs
        audio_inputs = [inp for inp in inputs if inp.media_type == "audio"]
        if not audio_inputs:
            raise InvalidInputError("Audio input required for speech model")
    
    def _process_inputs(self, inputs: List[MediaInput]) -> Dict[str, Any]:
        """Process inputs for speech models."""
        # Get audio
        audio = [
            inp.to_provider_format("huggingface")["content"]
            for inp in inputs
            if inp.media_type == "audio"
        ]
        
        # Process audio
        return self.processor(
            audio=audio,
            return_tensors="pt",
            sampling_rate=self.config.speech.sampling_rate
        ).to(self.model.device)
    
    def _generate(self,
                 processed_inputs: Dict[str, Any],
                 generation_config: Optional[Dict[str, Any]] = None,
                 stream: bool = False,
                 **kwargs) -> Union[str, Dict[str, Any]]:
        """Generate speech outputs."""
        outputs = self.model.generate(
            **processed_inputs,
            **(generation_config or {})
        )
        
        # Decode output tokens
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        return text[0] if len(text) == 1 else text 