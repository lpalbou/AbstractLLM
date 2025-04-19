"""
Text generation pipeline implementation for HuggingFace provider.
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator, Set
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, GenerationConfig
from threading import Thread
from queue import Queue

from abstractllm.media.interface import MediaInput
from abstractllm.exceptions import ModelLoadingError, GenerationError, CleanupError
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
    
    # Model-specific prompt templates
    PROMPT_TEMPLATES = {
        "microsoft/phi-2": {
            "instruct": "Instruct: {prompt}\nOutput: ",
            "chat": "Human: {prompt}\nAssistant: ",
            "system": "System: {system}\nHuman: {prompt}\nAssistant: "
        }
    }
    
    def load(self, model_name: str, config: ModelConfig) -> None:
        """Load the text generation model."""
        try:
            if model_name.endswith('.gguf'):
                self._load_gguf(model_name, config)
            else:
                self._load_transformers(model_name, config)
            self._is_loaded = True
            self._model_name = model_name  # Store model name for prompt formatting
        except Exception as e:
            self.cleanup()
            raise ModelLoadingError(f"Failed to load model: {e}")
    
    def _load_transformers(self, model_name: str, config: ModelConfig) -> None:
        """Load a transformers model."""
        try:
            # Load configuration
            model_config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=config.trust_remote_code
            )
            
            # Determine optimal dtype
            if config.torch_dtype == "auto":
                if torch.cuda.is_available():
                    dtype = torch.float16
                else:
                    dtype = torch.float32
            else:
                dtype = getattr(torch, config.torch_dtype)
            
            # Load tokenizer first to ensure proper setup
            tokenizer_config = config.base.tokenizer_config or {}
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=config.trust_remote_code,
                **tokenizer_config
            )
            
            # Ensure we have required tokens
            if not self._tokenizer.pad_token:
                if self._tokenizer.eos_token:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                else:
                    # Add a new pad token if none exists
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
            if not self._tokenizer.bos_token:
                if self._tokenizer.eos_token:
                    self._tokenizer.bos_token = self._tokenizer.eos_token
                else:
                    # Add a new bos token if none exists
                    self._tokenizer.add_special_tokens({'bos_token': '<s>'})
            
            if not self._tokenizer.eos_token:
                # Add a new eos token if none exists
                self._tokenizer.add_special_tokens({'eos_token': '</s>'})
            
            # Update tokenizer configuration with actual tokens
            config.base.tokenizer_config.update({
                "pad_token": self._tokenizer.pad_token,
                "bos_token": self._tokenizer.bos_token,
                "eos_token": self._tokenizer.eos_token
            })
            
            # Prepare loading kwargs
            load_kwargs = {
                "device_map": config.device_map,
                "torch_dtype": dtype,
                "use_safetensors": config.use_safetensors,
                "trust_remote_code": config.trust_remote_code,
                "low_cpu_mem_usage": True
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
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                })
            elif config.quantization == "8bit":
                load_kwargs.update({"load_in_8bit": True})
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                **load_kwargs
            )
            
            # Resize token embeddings if we added new tokens
            self._model.resize_token_embeddings(len(self._tokenizer))
            
            # Set up generation config
            self._generation_config = self._model.generation_config
            
            # Move model to appropriate device if needed
            if config.device_map != "auto":
                self._model = self._model.to(self.device)
            
            logger.info("Successfully loaded model and tokenizer")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
    
    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt according to model-specific templates."""
        # Get model-specific templates
        templates = self.PROMPT_TEMPLATES.get(self._model_name, {})
        
        if not templates:
            # No specific template, return prompt as is
            return prompt
            
        if system_prompt:
            template = templates.get("system", "{prompt}")
            return template.format(system=system_prompt, prompt=prompt)
        else:
            template = templates.get("instruct", "{prompt}")
            return template.format(prompt=prompt)
    
    def process(self, inputs: List[MediaInput], stream: bool = False, **kwargs) -> Union[List[str], Generator[str, None, None]]:
        """Process text inputs through the pipeline.
        
        Args:
            inputs: List of MediaInput objects containing text
            stream: Whether to stream the output
            **kwargs: Additional generation parameters that override defaults
            
        Returns:
            List of generated strings or generator yielding strings if streaming
            
        Raises:
            ModelLoadingError: If model is not loaded
            GenerationError: If generation fails
        """
        if not self._is_loaded:
            raise ModelLoadingError("Model not loaded. Call load() first.")
            
        try:
            # Format inputs for HF
            text_inputs = [inp.text for inp in inputs]
            
            # Get model's special token IDs
            pad_token_id = self._tokenizer.pad_token_id
            eos_token_id = self._tokenizer.eos_token_id
            bos_token_id = getattr(self._tokenizer, 'bos_token_id', None)
            
            # Tokenize with padding and attention masks
            tokenized = self._tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt",
                return_attention_mask=True,
                pad_to_multiple_of=8,  # For efficiency on GPU
            )
            
            # Move to device
            tokenized = {k: v.to(self._model.device) for k, v in tokenized.items()}
            
            # Set up generation config with defaults and overrides
            gen_kwargs = {
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 100,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "use_cache": True,
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "bos_token_id": bos_token_id,
                # Beam search params
                "num_beams": 1,
                "early_stopping": True,
                "length_penalty": 1.0,
            }
            
            # Override with user params
            gen_kwargs.update(kwargs)
            
            if stream:
                streamer = TextIteratorStreamer(
                    self._tokenizer,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                gen_kwargs["streamer"] = streamer
                
                # Start generation in a thread
                thread = Thread(target=self._model.generate, kwargs={
                    **tokenized,
                    **gen_kwargs
                })
                thread.start()
                
                # Stream tokens
                def generate():
                    try:
                        for text in streamer:
                            yield text
                    except Exception as e:
                        logger.error(f"Error during streaming: {e}")
                        raise GenerationError(f"Streaming generation failed: {e}")
                    finally:
                        thread.join()
                        
                return generate()
                
            else:
                # Non-streaming generation
                with torch.inference_mode():
                    output_ids = self._model.generate(
                        **tokenized,
                        **gen_kwargs
                    )
                
                return self._tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise GenerationError(f"Text generation failed: {e}")
    
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

    def _process_inputs(self, inputs: List[MediaInput]) -> Dict[str, Any]:
        """Process text inputs into model inputs."""
        text_parts = []
        for input_obj in inputs:
            if input_obj.media_type == "text":
                formatted = input_obj.to_provider_format("huggingface")
                text_parts.append(formatted["content"])
        
        text = "\n".join(text_parts)
        
        # Tokenize with safe defaults
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._get_context_window() or 2048,
            padding=True,
            return_attention_mask=True
        )
        
        # Move to device
        return {k: v.to(self._model.device) for k, v in inputs.items()} 

    def cleanup(self):
        """Clean up resources used by the pipeline.
        
        This method:
        1. Moves model to CPU
        2. Deletes model and tokenizer
        3. Clears CUDA cache if available
        
        Raises:
            CleanupError: If cleanup fails
        """
        try:
            if self._is_loaded:
                logger.info("Cleaning up pipeline resources...")
                
                # Move model to CPU first
                if hasattr(self._model, "to"):
                    self._model.to("cpu")
                
                # Delete model and tokenizer
                del self._model
                del self._tokenizer
                self._model = None
                self._tokenizer = None
                self._is_loaded = False
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info("Pipeline cleanup completed successfully")
                
        except Exception as e:
            logger.error(f"Error during pipeline cleanup: {e}")
            raise CleanupError(f"Failed to clean up pipeline resources: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup is called."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in pipeline destructor: {e}") 