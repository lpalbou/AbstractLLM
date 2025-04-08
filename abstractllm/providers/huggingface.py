"""
Hugging Face implementation for AbstractLLM.
"""

from typing import Dict, Any, Optional, Union, Generator, AsyncGenerator
import os
import asyncio
import logging

from abstractllm.interface import AbstractLLMInterface, ModelParameter, ModelCapability, create_config
from abstractllm.utils.logging import (
    log_request, 
    log_response,
    log_request_url
)

# Configure logger with specific class path
logger = logging.getLogger("abstractllm.providers.huggingface.HuggingFaceProvider")


class HuggingFaceProvider(AbstractLLMInterface):
    """
    Hugging Face implementation using Transformers.
    """
    
    def __init__(self, config: Optional[Dict[Union[str, ModelParameter], Any]] = None):
        """
        Initialize the Hugging Face provider.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default configuration
        if ModelParameter.MODEL not in self.config and "model" not in self.config:
            self.config[ModelParameter.MODEL] = "google/gemma-7b"
        
        self._model = None
        self._tokenizer = None
        
        # Log provider initialization
        model_name = self.config.get(ModelParameter.MODEL, self.config.get("model", "google/gemma-7b"))
        logger.info(f"Initialized HuggingFace provider with model: {model_name}")
    
    def _load_model_and_tokenizer(self):
        """
        Load the model and tokenizer if not already loaded.
        
        Raises:
            ImportError: If required packages are not installed
        """
        if self._model is not None and self._tokenizer is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Required packages not found. Install them with: "
                "pip install torch transformers"
            )
        
        model_name = self.config.get(ModelParameter.MODEL, self.config.get("model", "google/gemma-7b"))
        
        # Log model loading at INFO level
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        # Extract parameters for model loading
        load_in_8bit = self.config.get("load_in_8bit", False)
        load_in_4bit = self.config.get("load_in_4bit", False)
        device_map = self.config.get("device_map", "auto")
        cache_dir = self.config.get("cache_dir", None)
        
        # Log detailed configuration at DEBUG level
        logger.debug(f"Model loading configuration: load_in_8bit={load_in_8bit}, load_in_4bit={load_in_4bit}, device_map={device_map}")
        if cache_dir:
            logger.debug(f"Using custom cache directory: {cache_dir}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Handle pad token for tokenizers that don't have one
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.debug("Setting pad_token to eos_token for tokenizer")
            elif self._tokenizer.bos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.bos_token
                logger.debug("Setting pad_token to bos_token for tokenizer")
            elif self._tokenizer.unk_token is not None:
                self._tokenizer.pad_token = self._tokenizer.unk_token
                logger.debug("Setting pad_token to unk_token for tokenizer")
        
        # Load model with appropriate settings
        model_kwargs = {
            "device_map": device_map,
            "cache_dir": cache_dir
        }
        
        if load_in_8bit:
            try:
                import bitsandbytes
                model_kwargs["load_in_8bit"] = True
                logger.debug("Using 8-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not installed. Falling back to default precision.")
        elif load_in_4bit:
            try:
                import bitsandbytes
                model_kwargs["load_in_4bit"] = True
                logger.debug("Using 4-bit quantization")
            except ImportError:
                logger.warning("bitsandbytes not installed. Falling back to default precision.")
                
        # Load the model
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Set pad_token_id in the model's config to match the tokenizer
        if hasattr(self._model, 'config') and self._tokenizer.pad_token_id is not None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
            logger.debug(f"Set model's pad_token_id to tokenizer's pad_token_id: {self._tokenizer.pad_token_id}")
        
        logger.info(f"Successfully loaded model {model_name}")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                stream: bool = False, 
                **kwargs) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using Hugging Face model.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
            
        Returns:
            The generated response or a generator if streaming
            
        Raises:
            Exception: If model loading or generation fails
        """
        # Import here to avoid dependency issues
        import torch
        
        # Combine configuration with kwargs
        params = self.config.copy()
        params.update(kwargs)
        
        # Extract parameters (using both string and enum keys for backwards compatibility)
        model_name = params.get(ModelParameter.MODEL, params.get("model", "google/gemma-7b"))
        temperature = params.get(ModelParameter.TEMPERATURE, params.get("temperature", 0.7))
        max_tokens = params.get(ModelParameter.MAX_TOKENS, params.get("max_new_tokens", 512))
        system_prompt_from_config = params.get(ModelParameter.SYSTEM_PROMPT, params.get("system_prompt"))
        system_prompt = system_prompt or system_prompt_from_config
        top_p = params.get(ModelParameter.TOP_P, params.get("top_p", 1.0))
        stop = params.get(ModelParameter.STOP, params.get("stop"))
        
        # Log at INFO level
        logger.info(f"Generating response with HuggingFace model: {model_name}")
        
        # Log detailed parameters at DEBUG level
        logger.debug(f"Generation parameters: temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
        if system_prompt:
            logger.debug("Using system prompt")
        if stop:
            logger.debug(f"Using stop sequences: {stop}")
        
        # Log the request
        log_request("huggingface", prompt, {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "has_system_prompt": system_prompt is not None,
            "stream": stream
        })
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Prepare the input - handling system prompt if provided
        if system_prompt:
            # Adapt based on the model - this is a simplistic approach
            # Different models have different formats for system prompts
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Set up generation config
        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "pad_token_id": self._tokenizer.pad_token_id  # Explicitly set pad_token_id to avoid warnings
        }
        
        if temperature > 0:
            generation_config["temperature"] = temperature
            
        if stop:
            # Convert stop sequences to token IDs
            stop_token_ids = []
            for seq in (stop if isinstance(stop, list) else [stop]):
                ids = self._tokenizer.encode(seq, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids[-1])  # Use last token as stop
            if stop_token_ids:
                generation_config["eos_token_id"] = stop_token_ids
        
        # Handle streaming if requested (a simplified version)
        if stream:
            logger.info("Starting streaming generation")
            
            def response_generator():
                input_length = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    generated = inputs["input_ids"].clone()
                    past_key_values = None
                    
                    for _ in range(max_tokens):
                        with torch.no_grad():
                            outputs = self._model(
                                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                                past_key_values=past_key_values,
                                use_cache=True
                            )
                            
                            next_token_logits = outputs.logits[:, -1, :]
                            
                            # Apply temperature and top-p sampling
                            if temperature > 0:
                                next_token_logits = next_token_logits / temperature
                            
                            # Filter with top-p
                            if top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                next_token_logits[indices_to_remove] = -float("Inf")
                            
                            # Sample
                            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            
                            # Check for end of sequence
                            if stop_token_ids and next_token.item() in stop_token_ids:
                                break
                                
                            # Add to generated
                            generated = torch.cat([generated, next_token], dim=-1)
                            
                            # Decode the current new token
                            new_token_text = self._tokenizer.decode(next_token[0])
                            yield new_token_text
                            
                            # Update past key values
                            past_key_values = outputs.past_key_values
                            
            return response_generator()
        else:
            # Standard non-streaming response
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode and extract only the new content
            full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = full_output[len(full_prompt):].strip()
            
            # Log the response
            log_response("huggingface", result)
            logger.info("Generation completed successfully")
            
            return result
    
    async def generate_async(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None, 
                          stream: bool = False, 
                          **kwargs) -> Union[str, AsyncGenerator[str, None]]:
        """
        Asynchronously generate a response using Hugging Face model.
        
        This runs the generation in a thread pool since most HF models
        are not async-compatible.
        
        Args:
            prompt: The input prompt
            system_prompt: Override the system prompt in the config
            stream: Whether to stream the response
            **kwargs: Additional parameters to override configuration
            
        Returns:
            The generated response or an async generator if streaming
            
        Raises:
            Exception: If model loading or generation fails
        """
        loop = asyncio.get_event_loop()
        
        if not stream:
            # For non-streaming, run the synchronous method in an executor
            result = await loop.run_in_executor(
                None, 
                lambda: self.generate(
                    prompt=prompt, 
                    system_prompt=system_prompt, 
                    stream=False, 
                    **kwargs
                )
            )
            return result
        else:
            # For streaming, we need to wrap the generator
            sync_gen = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                stream=True,
                **kwargs
            )
            
            async def async_generator():
                for token in sync_gen:
                    yield token
                    # Small sleep to give other tasks a chance to run
                    await asyncio.sleep(0)
            
            return async_generator()
    
    def get_capabilities(self) -> Dict[Union[str, ModelCapability], Any]:
        """
        Return capabilities of the Hugging Face provider.
        
        Returns:
            Dictionary of capabilities
        """
        return {
            ModelCapability.STREAMING: True,
            ModelCapability.MAX_TOKENS: None,  # Varies by model and hardware
            ModelCapability.SYSTEM_PROMPT: True,
            ModelCapability.ASYNC: True,
            ModelCapability.FUNCTION_CALLING: False,  # Not typically supported natively
            ModelCapability.VISION: False  # Depends on model, assume False by default
        }
    
    @staticmethod
    def list_cached_models(cache_dir: Optional[str] = None) -> list:
        """
        List all models cached locally.
        
        Args:
            cache_dir: Custom cache directory
            
        Returns:
            List of dictionaries with model information
        """
        try:
            from huggingface_hub import scan_cache_dir
            
            cache_info = scan_cache_dir(cache_dir)
            models = []
            
            for repo in cache_info.repos:
                models.append({
                    "name": repo.repo_id,
                    "size": repo.size_on_disk,
                    "last_used": repo.last_accessed
                })
                
            return models
        except ImportError:
            raise ImportError("huggingface_hub package is required for this feature")
    
    @staticmethod
    def clear_model_cache(model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
        """
        Clear cached models.
        
        Args:
            model_name: Specific model to clear (None for all)
            cache_dir: Custom cache directory
            
        Returns:
            None
        """
        try:
            from huggingface_hub import delete_cache_folder, scan_cache_dir
            
            if model_name:
                # Delete specific model
                cache_info = scan_cache_dir(cache_dir)
                for repo in cache_info.repos:
                    if repo.repo_id == model_name:
                        delete_cache_folder(repo_id=model_name, cache_dir=cache_dir)
                        return
                raise ValueError(f"Model {model_name} not found in cache")
            else:
                # Delete entire cache
                delete_cache_folder(cache_dir=cache_dir)
        except ImportError:
            raise ImportError("huggingface_hub package is required for this feature") 