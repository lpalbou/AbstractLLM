"""
HuggingFace provider implementation using pipeline abstraction.
"""

import logging
from typing import Optional, Dict, Any, Union, List, Generator
from pathlib import Path

from abstractllm.interface import AbstractLLMInterface, ModelParameter
from abstractllm.exceptions import ModelLoadingError, GenerationError
from abstractllm.media.factory import MediaFactory
from abstractllm.utils.logging import log_request, log_response

from .model_types import ModelConfig, ModelArchitecture
from .factory import PipelineFactory

# Configure logger
logger = logging.getLogger(__name__)

class HuggingFaceProvider(AbstractLLMInterface):
    """HuggingFace provider using pipeline abstraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pipeline = None
        self._model_config = None
        
        # Set default configuration
        default_config = {
            ModelParameter.MODEL: "microsoft/phi-2",
            ModelParameter.TEMPERATURE: 0.7,
            ModelParameter.MAX_TOKENS: 2048,
            "trust_remote_code": True,
            "use_flash_attention": True,
            "device_map": "auto",
            "torch_dtype": "auto",
            "use_safetensors": True
        }
        
        # Merge defaults with provided config
        self.config_manager.merge_with_defaults(default_config)
    
    def _create_model_config(self) -> ModelConfig:
        """Create model configuration from provider config."""
        # Detect if running on macOS
        import platform
        is_macos = platform.system().lower() == "darwin"
        
        # Disable Flash Attention 2 on macOS
        use_flash_attention = False if is_macos else self.config_manager.get_param("use_flash_attention", True)
        
        # Get model name and detect architecture
        model_name = self.config_manager.get_param(ModelParameter.MODEL)
        model_type, architecture = PipelineFactory.detect_model_architecture(model_name)
        
        # Create base configuration
        config = ModelConfig(
            architecture=architecture,
            trust_remote_code=self.config_manager.get_param("trust_remote_code", True),
            use_flash_attention=use_flash_attention,
            quantization=self.config_manager.get_param("quantization"),
            device_map=self.config_manager.get_param("device_map", "auto"),
            torch_dtype=self.config_manager.get_param("torch_dtype", "auto"),
            use_safetensors=self.config_manager.get_param("use_safetensors", True)
        )
        
        # Add generation configuration
        config.generation.temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7)
        config.generation.max_new_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048)
        config.generation.top_p = self.config_manager.get_param(ModelParameter.TOP_P, 0.9)
        config.generation.top_k = self.config_manager.get_param(ModelParameter.TOP_K, 50)
        config.generation.repetition_penalty = self.config_manager.get_param(ModelParameter.REPETITION_PENALTY, 1.1)
        
        return config
    
    def load_model(self) -> None:
        """Load the model using appropriate pipeline."""
        try:
            model_name = self.config_manager.get_param(ModelParameter.MODEL)
            logger.info(f"Loading model {model_name} using transformers")
            
            # Create model configuration
            self._model_config = self._create_model_config()
            
            # Initialize tokenizer configuration
            from transformers import AutoConfig, AutoTokenizer
            model_config_obj = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=self._model_config.trust_remote_code
            )
            
            # Initialize tokenizer first to ensure proper setup
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self._model_config.trust_remote_code,
                padding_side="left",  # Better for generation
                model_max_length=getattr(model_config_obj, "max_position_embeddings", 2048)
            )
            
            # Ensure we have required tokens
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
                
            if not tokenizer.bos_token:
                tokenizer.bos_token = tokenizer.eos_token
            
            # Add special tokens
            special_tokens = {
                "pad_token": tokenizer.pad_token,
                "bos_token": tokenizer.bos_token,
                "eos_token": tokenizer.eos_token
            }
            tokenizer.add_special_tokens(special_tokens)
            
            # Add tokenizer configuration
            self._model_config.base.tokenizer_config = {
                "padding_side": "left",  # Better for generation
                "model_max_length": getattr(model_config_obj, "max_position_embeddings", 2048),
                "add_special_tokens": True,
                "return_attention_mask": True,
                "pad_token": tokenizer.pad_token,
                "bos_token": tokenizer.bos_token,
                "eos_token": tokenizer.eos_token
            }
            
            # Create and load pipeline
            self._pipeline = PipelineFactory.create_pipeline(
                model_name,
                self._model_config
            )
            
            # Load model and components
            self._pipeline.load(model_name, self._model_config)
            logger.info("Successfully loaded model and tokenizer")
            
        except Exception as e:
            if self._pipeline:
                self._pipeline.cleanup()
            self._pipeline = None
            raise ModelLoadingError(f"Failed to load model: {e}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                files: Optional[List[Union[str, Path]]] = None,
                stream: bool = False,
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text using the loaded pipeline."""
        if not self._pipeline:
            self.load_model()
        
        try:
            # Convert all inputs to MediaInput objects
            inputs = []
            
            # Add system prompt if provided
            if system_prompt:
                system_input = MediaFactory.from_source(system_prompt, media_type="text")
                inputs.append(system_input)
            
            # Add main prompt
            text_input = MediaFactory.from_source(prompt, media_type="text")
            inputs.append(text_input)
            
            # Process files if any
            if files:
                for file_path in files:
                    media_input = MediaFactory.from_source(file_path)
                    inputs.append(media_input)
            
            # Get generation config
            generation_config = {
                "temperature": self.config_manager.get_param(ModelParameter.TEMPERATURE, 0.7),
                "max_new_tokens": self.config_manager.get_param(ModelParameter.MAX_TOKENS, 2048),
                "do_sample": True,
                "top_p": self.config_manager.get_param(ModelParameter.TOP_P, 0.9),
                "top_k": self.config_manager.get_param(ModelParameter.TOP_K, 50),
                "repetition_penalty": self.config_manager.get_param(ModelParameter.REPETITION_PENALTY, 1.1),
                "use_cache": True,
                "num_beams": 1,
                "early_stopping": False
            }
            generation_config.update(kwargs)
            
            # Log request
            log_request("huggingface", prompt, {
                "model": self.config_manager.get_param(ModelParameter.MODEL),
                "temperature": generation_config["temperature"],
                "max_tokens": generation_config["max_new_tokens"],
                "has_system_prompt": system_prompt is not None,
                "stream": stream,
                "has_files": bool(files)
            })
            
            # Process inputs
            result = self._pipeline.process(
                inputs=inputs,
                generation_config=generation_config,
                stream=stream
            )
            
            # Process through output handlers and return
            return self._handle_output(result)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise GenerationError(f"Generation failed: {e}")
    
    async def generate_async(self,
                           prompt: str,
                           system_prompt: Optional[str] = None,
                           files: Optional[List[Union[str, Path]]] = None,
                           stream: bool = False,
                           **kwargs) -> Union[str, Generator[str, None, None]]:
        """Generate text asynchronously."""
        # For now, we'll use the synchronous implementation in a thread pool
        from concurrent.futures import ThreadPoolExecutor
        import asyncio
        
        with ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor,
                self.generate,
                prompt,
                system_prompt,
                files,
                stream,
                **kwargs
            )
            
            # Process through output handlers and return
            return self._handle_output(result)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._pipeline:
            self._pipeline.cleanup()
        self._pipeline = None 