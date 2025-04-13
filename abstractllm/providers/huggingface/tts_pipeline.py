"""
Text-to-Speech pipeline implementation for HuggingFace provider.

This pipeline handles text-to-speech models including:
- Microsoft SpeechT5
- Coqui XTTS
- Bark
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator
import torch
from transformers import (
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    VitsTokenizer,
    VitsModel,
    BarkProcessor,
    BarkModel,
    PreTrainedModel
)
import numpy as np
from datasets import load_dataset

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadError, InvalidInputError, GenerationError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities

# Configure logger
logger = logging.getLogger(__name__)

class TTSPipeline(BasePipeline):
    """Pipeline for text-to-speech synthesis.
    
    Supports multiple TTS architectures:
    - SpeechT5: High-quality speech synthesis with voice cloning
    - VITS: Fast, high-quality speech synthesis
    - Bark: Multilingual speech synthesis with emotion
    """
    
    # Model architecture mapping
    ARCHITECTURES = {
        "speecht5": (SpeechT5Processor, SpeechT5ForTextToSpeech),
        "vits": (VitsTokenizer, VitsModel),
        "bark": (BarkProcessor, BarkModel)
    }
    
    def __init__(self) -> None:
        """Initialize the TTS pipeline."""
        super().__init__()
        self.model: Optional[PreTrainedModel] = None
        self.processor = None
        self.vocoder = None  # For SpeechT5
        self.speaker_embeddings = None
        self.output_format: str = "wav"
        self.sample_rate: int = 16000  # Default, will be updated based on model
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the TTS model and processor."""
        try:
            # Import required components
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            from datasets import load_dataset
            import torch

            # Store config
            self.model_config = config

            # Load processor
            self.processor = SpeechT5Processor.from_pretrained(model_name)

            # Load model
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                model_name,
                device_map=config.device_map,
                torch_dtype=config.torch_dtype
            )

            # Load vocoder
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Load speaker embeddings
            logger.info("Loading speaker embeddings...")
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = torch.tensor(
                embeddings_dataset[7306]["xvector"]  # Using a default voice
            ).unsqueeze(0)

            # Move models and embeddings to device if not using device_map="auto"
            if config.device_map != "auto":
                self.model.to(self.device)
                self.vocoder.to(self.device)
                self.speaker_embeddings = self.speaker_embeddings.to(self.device)

            # Set sample rate from vocoder config
            self.sample_rate = self.vocoder.config.sampling_rate
            self.output_format = "wav"  # Default to WAV format

            self._is_loaded = True
            logger.info(f"Successfully loaded TTS model {model_name}")

        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}")
    
    def _detect_architecture(self, model_name: str) -> str:
        """Detect the model architecture from the model name."""
        name_lower = model_name.lower()
        if "speecht5" in name_lower:
            return "speecht5"
        elif "vits" in name_lower or "coqui" in name_lower:
            return "vits"
        elif "bark" in name_lower:
            return "bark"
        else:
            # Default to SpeechT5 for unknown models
            logger.warning(f"Unknown model architecture for {model_name}, defaulting to SpeechT5")
            return "speecht5"
    
    def process(self, 
                inputs: List[MediaInput], 
                generation_config: Optional[Dict[str, Any]] = None,
                stream: bool = False,
                **kwargs) -> Union[bytes, Generator[bytes, None, None]]:
        """Process text input and generate speech audio.
        
        Args:
            inputs: List of text inputs to convert to speech
            generation_config: Optional generation parameters
            stream: Whether to stream the audio output
            **kwargs: Additional arguments including:
                - voice_preset: Voice configuration (model-specific)
                - speaking_rate: Speech rate multiplier
                - output_format: Audio format (wav, mp3, etc.)
                
        Returns:
            Audio data as bytes or generator of audio chunks
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
            
            # Process with SpeechT5
            return self._process_speecht5(text, generation_config, stream, **kwargs)
            
        except Exception as e:
            raise GenerationError(f"Speech generation failed: {e}")
    
    def _process_speecht5(self, 
                         text: str,
                         generation_config: Optional[Dict[str, Any]] = None,
                         stream: bool = False,
                         **kwargs) -> Union[bytes, Generator[bytes, None, None]]:
        """Process text using SpeechT5."""
        try:
            # Prepare inputs
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate speech with SpeechT5
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=self.speaker_embeddings,  # Use loaded speaker embeddings
                    vocoder=self.vocoder
                )

            # Move to CPU for audio processing
            speech = speech.cpu().numpy()

            # Ensure the audio is properly shaped
            if len(speech.shape) == 1:
                speech = speech.reshape(-1, 1)

            # Convert to audio format
            if stream:
                return self._stream_audio(speech)
            else:
                return self._audio_to_bytes(speech)
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise GenerationError(f"Failed to generate speech: {e}")
    
    def _process_vits(self,
                     text: str,
                     generation_config: Optional[Dict[str, Any]] = None,
                     stream: bool = False,
                     **kwargs) -> Union[bytes, Generator[bytes, None, None]]:
        """Process text using VITS."""
        # Prepare inputs
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate speech
        with torch.no_grad():
            speech = self.model(**inputs).audio
        
        # Convert to audio format
        if stream:
            return self._stream_audio(speech.cpu().numpy())
        else:
            return self._audio_to_bytes(speech.cpu().numpy())
    
    def _process_bark(self,
                     text: str,
                     generation_config: Optional[Dict[str, Any]] = None,
                     stream: bool = False,
                     **kwargs) -> Union[bytes, Generator[bytes, None, None]]:
        """Process text using Bark."""
        # Prepare inputs
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate speech
        speech = self.model.generate(**inputs).audio
        
        # Convert to audio format
        if stream:
            return self._stream_audio(speech.cpu().numpy())
        else:
            return self._audio_to_bytes(speech.cpu().numpy())
    
    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio array to bytes in specified format."""
        try:
            import soundfile as sf
            import io

            # Ensure audio is float32 and normalized
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))

            # Ensure audio is 2D (samples, channels)
            if len(audio.shape) == 1:
                audio = audio.reshape(-1, 1)

            # Write to buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.sample_rate, format=self.output_format)
            buffer.seek(0)  # Rewind buffer
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to convert audio to bytes: {e}")
            raise GenerationError(f"Failed to convert audio to bytes: {e}")
    
    def _stream_audio(self, audio: np.ndarray, chunk_size: int = 4096) -> Generator[bytes, None, None]:
        """Stream audio array as bytes chunks."""
        try:
            import soundfile as sf
            import io

            # Ensure audio is float32 and normalized
            audio = audio.astype(np.float32)
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))

            # Write to buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.sample_rate, format=self.output_format)
            buffer.seek(0)

            while True:
                chunk = buffer.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        except Exception as e:
            raise GenerationError(f"Failed to stream audio: {e}")
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            input_types={"text"},
            output_types={"audio"},
            supports_streaming=True,
            supports_system_prompt=False,
            context_window=None  # Not applicable for TTS
        )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'model_config'):
            del self.model_config
        if hasattr(self, 'vocoder'):
            del self.vocoder
        if hasattr(self, 'speaker_embeddings'):
            del self.speaker_embeddings
        super().cleanup() 