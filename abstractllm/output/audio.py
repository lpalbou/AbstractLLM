"""
Audio output handling for AbstractLLM.

This module provides audio output capabilities, primarily for text-to-speech output.
It handles platform-specific audio playback and streaming considerations.
"""

import logging
import queue
import threading
from typing import Optional, Union, Generator, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from pathlib import Path

from abstractllm.providers.huggingface.tts_pipeline import TTSPipeline
from abstractllm.exceptions import AudioOutputError
from abstractllm.interface import OutputHandler

logger = logging.getLogger(__name__)

class AudioPlayer(ABC):
    """Abstract base class for audio playback."""
    
    @abstractmethod
    def play(self, audio_data: Union[bytes, np.ndarray], sample_rate: int) -> None:
        """Play audio data."""
        pass
    
    @abstractmethod
    def stream(self, audio_stream: Generator[bytes, None, None], sample_rate: int) -> None:
        """Stream audio data."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop audio playback."""
        pass

class SoundDevicePlayer(AudioPlayer):
    """Audio player using sounddevice library."""
    
    def __init__(self):
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_playing = False
        
    def play(self, audio_data: Union[bytes, np.ndarray], sample_rate: int) -> None:
        """Play complete audio data."""
        try:
            if isinstance(audio_data, bytes):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp.flush()
                    data, _ = sf.read(tmp.name)
                os.unlink(tmp.name)
            else:
                data = audio_data
            
            sd.play(data, sample_rate)
            sd.wait()
        except Exception as e:
            raise AudioOutputError(f"Failed to play audio: {e}")
    
    def stream(self, audio_stream: Generator[bytes, None, None], sample_rate: int) -> None:
        """Stream audio data."""
        def audio_callback(outdata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            try:
                data = self.audio_queue.get_nowait()
                if len(data) < len(outdata):
                    outdata[:len(data)] = data
                    outdata[len(data):] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:] = data[:len(outdata)]
                    # Put remaining data back in queue
                    if len(data) > len(outdata):
                        self.audio_queue.put(data[len(outdata):])
            except queue.Empty:
                outdata.fill(0)
                raise sd.CallbackStop()
        
        try:
            self.is_playing = True
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                callback=audio_callback
            ) as stream:
                for chunk in audio_stream:
                    if not self.is_playing:
                        break
                    if isinstance(chunk, bytes):
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp.write(chunk)
                            tmp.flush()
                            data, _ = sf.read(tmp.name)
                        os.unlink(tmp.name)
                    else:
                        data = chunk
                    self.audio_queue.put(data)
                    sd.sleep(int(1000 * len(data) / sample_rate))
        except Exception as e:
            self.is_playing = False
            raise AudioOutputError(f"Failed to stream audio: {e}")
    
    def stop(self) -> None:
        """Stop audio playback."""
        self.is_playing = False
        sd.stop()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

class TTSOutputHandler(OutputHandler):
    """Handler for TTS output from AbstractLLM pipelines."""
    
    def __init__(self, 
                tts_pipeline: TTSPipeline,
                output_dir: Optional[str] = None,
                playback: bool = True,
                save: bool = False,
                **kwargs):
        """Initialize TTS output handler.
        
        Args:
            tts_pipeline: TTS pipeline instance
            output_dir: Directory to save audio files (if save=True)
            playback: Whether to play audio output
            save: Whether to save audio files
            **kwargs: Additional TTS configuration
        """
        self.tts = tts_pipeline
        self.output_dir = Path(output_dir) if output_dir else None
        self.playback = playback
        self.save = save
        self.kwargs = kwargs
        
        if self.save and not self.output_dir:
            self.output_dir = Path.home() / ".abstractllm" / "audio"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize audio player if needed
        self.player = SoundDevicePlayer() if playback else None
        
        # Cache for frequently used phrases
        self.cache_dir = Path.home() / ".abstractllm" / "cache" / "tts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def handle(self, text: Union[str, Generator[str, None, None]]) -> None:
        """Handle text output by converting to speech.
        
        Args:
            text: Text to convert to speech
        """
        try:
            if isinstance(text, Generator):
                self._handle_streaming(text)
            else:
                self._handle_complete(text)
        except Exception as e:
            logger.error(f"TTS output handling failed: {e}")
            # Fall back to text output
            if isinstance(text, Generator):
                for chunk in text:
                    print(chunk, end="", flush=True)
            else:
                print(text)
    
    def _handle_complete(self, text: str) -> None:
        """Handle complete text output."""
        try:
            # Check cache first
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.wav"
            
            if cache_file.exists():
                logger.debug("Using cached audio file")
                audio_data = cache_file.read_bytes()
            else:
                logger.debug("Generating new audio")
                # Generate new audio
                from abstractllm.media.text import TextInput
                audio_data = self.tts.process([TextInput(text)], **self.kwargs)
                # Cache for future use
                cache_file.write_bytes(audio_data)
            
            # Save if requested
            if self.save:
                import time
                output_file = self.output_dir / f"tts_{int(time.time())}.{self.tts.output_format}"
                logger.info(f"Saving audio to {output_file}")
                output_file.write_bytes(audio_data)
            
            # Play if requested
            if self.playback:
                logger.debug("Playing audio")
                self.player.play(audio_data, self.tts.sample_rate)
                
        except Exception as e:
            logger.error(f"Failed to handle complete text: {e}")
            raise
    
    def _handle_streaming(self, text_stream: Generator[str, None, None]) -> None:
        """Handle streaming text output."""
        try:
            def audio_stream():
                for text_chunk in text_stream:
                    from abstractllm.media.text import TextInput
                    audio_chunk = self.tts.process(
                        [TextInput(text_chunk)],
                        stream=True,
                        **self.kwargs
                    )
                    for data in audio_chunk:
                        yield data
            
            if self.playback:
                logger.debug("Streaming audio playback")
                self.player.stream(audio_stream(), self.tts.sample_rate)
            
            if self.save:
                # For streaming, save complete audio
                import time
                output_file = self.output_dir / f"tts_stream_{int(time.time())}.{self.tts.output_format}"
                logger.info(f"Saving streamed audio to {output_file}")
                with open(output_file, 'wb') as f:
                    for chunk in audio_stream():
                        f.write(chunk)
                        
        except Exception as e:
            logger.error(f"Failed to handle streaming text: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        # Include TTS config in cache key
        config_str = f"{self.tts.model_config}_{self.kwargs}"
        key = f"{text}_{config_str}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.player:
            self.player.stop() 