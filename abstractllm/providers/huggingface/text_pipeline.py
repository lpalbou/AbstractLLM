"""
Text generation pipeline implementation for HuggingFace provider.
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator, Set
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from queue import Queue

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadingError, GenerationError
from .model_types import BasePipeline, ModelConfig, ModelCapabilities, ModelArchitecture

# Configure logger
logger = logging.getLogger(__name__)

class GGUFTokenizerWrapper:
    """Wrapper for GGUF model tokenizer to match HF interface."""
    
    def __init__(self, model):
        self.model = model
        self.eos_token = "</s>"
        self.pad_token = "</s>"
        self.bos_token = "<s>"
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token ids."""
        return self.model.tokenize(text.encode())
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token ids to text."""
        return self.model.detokenize(token_ids).decode()

class TextGenerationPipeline(BasePipeline):
    """Pipeline for text generation models."""
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the text generation model."""
        try:
            if model_name.endswith('.gguf'):
                self._load_gguf(model_name, config)
            else:
                self._load_transformers(model_name, config)
            self._is_loaded = True
        except Exception as e:
            self.cleanup()
            raise ModelLoadingError(f"Failed to load model: {e}")
    
    def _load_transformers(self, model_name: str, config: ModelConfig) -> None:
        """Load a transformers model."""
        # Load configuration
        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code
        )
        
        # Prepare loading kwargs
        load_kwargs = {
            "device_map": config.device_map,
            "torch_dtype": config.torch_dtype,
            "use_safetensors": config.use_safetensors,
            "trust_remote_code": config.trust_remote_code
        }
        
        # Add Flash Attention 2 configuration if enabled
        if config.use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
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
            
        # Load model and tokenizer
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            **load_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=config.trust_remote_code
        )
        
        # Ensure we have required tokens
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Set up generation config
        self._generation_config = self._model.generation_config
    
    def _load_gguf(self, model_path: str, config: ModelConfig) -> None:
        """Load a GGUF model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ModelLoadingError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python"
            )
        
        # Determine device configuration
        n_gpu_layers = -1 if config.device_map != "cpu" else 0
        
        self._model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Default context window
            n_gpu_layers=n_gpu_layers
        )
        
        # Create tokenizer wrapper
        self._tokenizer = GGUFTokenizerWrapper(self._model)
    
    def process(self, 
               inputs: List[MediaInput], 
               generation_config: Optional[Dict[str, Any]] = None,
               stream: bool = False,
               **kwargs) -> Union[str, Generator[str, None, None]]:
        """Process inputs and generate text."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Process text inputs
            prompt = self._combine_text_inputs(inputs)
            
            # Handle GGUF models differently
            if isinstance(self._model, object) and hasattr(self._model, 'model_path'):  # GGUF model
                return self._generate_gguf(prompt, generation_config, stream)
            else:
                return self._generate_transformers(prompt, generation_config, stream)
                
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}")
    
    def _combine_text_inputs(self, inputs: List[MediaInput]) -> str:
        """Combine text inputs into a single prompt."""
        text_parts = []
        for input_obj in inputs:
            if input_obj.media_type == "text":
                formatted = input_obj.to_provider_format("huggingface")
                text_parts.append(formatted["content"])
        return "\n".join(text_parts)
    
    def _generate_transformers(self, 
                             prompt: str,
                             generation_config: Optional[Dict[str, Any]] = None,
                             stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using transformers model."""
        # Prepare inputs
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Update generation config
        gen_kwargs = {}
        if hasattr(self._generation_config, "to_dict"):
            gen_kwargs.update(self._generation_config.to_dict())
        else:
            # Fallback to getting all attributes
            gen_kwargs.update({
                k: v for k, v in vars(self._generation_config).items()
                if not k.startswith('_')
            })
        
        # Update with provided config
        if generation_config:
            gen_kwargs.update(generation_config)
        
        if stream:
            # Set up streaming
            streamer = TextIteratorStreamer(self._tokenizer)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                **gen_kwargs
            )
            
            # Run generation in a thread
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Return generator
            def stream_generator():
                for text in streamer:
                    yield text
            return stream_generator()
        else:
            # Generate without streaming
            outputs = self._model.generate(**inputs, **gen_kwargs)
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_gguf(self,
                      prompt: str,
                      generation_config: Optional[Dict[str, Any]] = None,
                      stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Generate text using GGUF model."""
        gen_kwargs = {
            "temperature": generation_config.get("temperature", 0.7) if generation_config else 0.7,
            "max_tokens": generation_config.get("max_tokens", 2048) if generation_config else 2048,
            "stream": stream
        }
        
        completion = self._model.create_completion(prompt, **gen_kwargs)
        
        if stream:
            def stream_generator():
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].text:
                        yield chunk.choices[0].text
            return stream_generator()
        else:
            return completion["choices"][0]["text"]
    
    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            input_types={"text"},
            output_types={"text"},
            supports_streaming=True,
            supports_system_prompt=True,
            context_window=self._get_context_window()
        )
    
    def _get_context_window(self) -> Optional[int]:
        """Get the model's context window size."""
        if hasattr(self._model, "config") and hasattr(self._model.config, "max_position_embeddings"):
            return self._model.config.max_position_embeddings
        return None 