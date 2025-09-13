"""
Model-specific configurations for MLX provider.

This module provides configuration classes for different model architectures
to ensure correct handling of tokens, generation parameters, and other
model-specific behaviors.
"""

import logging
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

class MLXModelConfig:
    """Base class for model-specific configurations."""
    
    name = "generic"
    
    # Token IDs or strings
    eos_tokens: List[str] = ["</s>"]
    bos_tokens: List[str] = ["<s>"]
    
    # Generation parameters
    default_repetition_penalty = 1.0
    default_temperature = 0.7
    
    # Other configuration
    supports_vision = False
    supports_system_prompt = True
    
    @classmethod
    def apply_to_tokenizer(cls, tokenizer):
        """Apply model-specific configuration to tokenizer."""
        # Add EOS tokens
        for token in cls.eos_tokens:
            try:
                tokenizer.add_eos_token(token)
                logger.info(f"Added EOS token to tokenizer: {token}")
            except Exception as e:
                logger.warning(f"Failed to add EOS token {token}: {e}")
                
    @classmethod
    def get_generation_params(cls, temperature: float, **kwargs) -> Dict[str, Any]:
        """Get parameters for generation."""
        params = {}
        
        # Ensure temperature is a valid float
        if temperature is None:
            temperature = cls.default_temperature
            logger.warning(f"Temperature was None, using default value: {temperature}")
        
        # Ensure temperature is within valid range
        temperature = float(max(0.01, min(2.0, temperature)))
        
        # Create sampler if needed
        try:
            import mlx_lm.sample_utils
            sampler = mlx_lm.sample_utils.make_sampler(temp=temperature)
            params["sampler"] = sampler
            
            # Apply repetition penalty if configured or provided
            # Allow custom repetition_penalty to override default
            repetition_penalty = kwargs.get("repetition_penalty", cls.default_repetition_penalty)
            
            if repetition_penalty != 1.0:
                logits_processors = mlx_lm.sample_utils.make_logits_processors(
                    repetition_penalty=repetition_penalty,
                    repetition_context_size=64  # Look back at last 64 tokens
                )
                params["logits_processors"] = logits_processors
                logger.info(f"Added repetition penalty {repetition_penalty} for {cls.name} model")
        except ImportError:
            logger.warning("Could not import mlx_lm.sample_utils, temperature will be ignored")
        
        return params
    
    @classmethod
    def format_system_prompt(cls, system_prompt: str, user_prompt: str, processor) -> str:
        """Format system and user prompts according to model expectations."""
        try:
            if hasattr(processor, "apply_chat_template"):
                # For newer models that support chat templates
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    return processor.apply_chat_template(messages, tokenize=False)
                except ValueError as e:
                    # If chat template is not available, fall back to simple format
                    logger.warning(f"Chat template error: {e}. Falling back to simple format.")
                    return f"{system_prompt}\n\n{user_prompt}"
            else:
                # Simple concatenation for older models
                return f"{system_prompt}\n\n{user_prompt}"
        except Exception as e:
            # Catch any unexpected errors and fall back to user prompt only
            logger.warning(f"Error formatting system prompt: {e}. Using just the user prompt.")
            return user_prompt


class LlamaConfig(MLXModelConfig):
    """Configuration for Llama models."""
    
    name = "llama"
    eos_tokens = ["</s>", "<|endoftext|>"]
    bos_tokens = ["<s>"]
    
    @classmethod
    def format_system_prompt(cls, system_prompt: str, user_prompt: str, processor) -> str:
        """Format system and user prompts according to Llama chat template."""
        if hasattr(processor, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return processor.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback to a common Llama format
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"


class QwenConfig(MLXModelConfig):
    """Configuration for Qwen models."""
    
    name = "qwen"
    eos_tokens = ["<|endoftext|>", "<|im_end|>", "</s>"]
    bos_tokens = ["<|im_start|>"]
    default_repetition_penalty = 1.2
    
    @classmethod
    def format_system_prompt(cls, system_prompt: str, user_prompt: str, processor) -> str:
        """Format system and user prompts according to Qwen chat template."""
        if hasattr(processor, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return processor.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback to a common Qwen format
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


class MistralConfig(MLXModelConfig):
    """Configuration for Mistral models."""
    
    name = "mistral"
    eos_tokens = ["</s>"]
    bos_tokens = ["<s>"]


class PhiConfig(MLXModelConfig):
    """Configuration for Phi models."""
    
    name = "phi"
    eos_tokens = ["<|endoftext|>", "</s>"]
    bos_tokens = ["<s>"]


class PaliGemmaConfig(MLXModelConfig):
    """Configuration for PaliGemma vision models."""
    
    name = "paligemma"
    eos_tokens = ["</s>"]
    bos_tokens = ["<s>"]
    supports_vision = True
    
    @classmethod
    def format_system_prompt(cls, system_prompt: str, user_prompt: str, processor) -> str:
        """Format system and user prompts for PaliGemma."""
        # PaliGemma expects image tokens at the beginning
        if "<image>" not in user_prompt:
            user_prompt = "<image> " + user_prompt
            
        if system_prompt:
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        return user_prompt
    
    @classmethod
    def apply_to_tokenizer(cls, tokenizer):
        """Apply PaliGemma-specific configuration to tokenizer."""
        # PaliGemmaProcessor doesn't support add_eos_token
        # We've already added a dummy method in MLXProvider
        # Just skip this without error
        pass


class Qwen2VLConfig(MLXModelConfig):
    """Configuration for Qwen vision language models.
    
    This covers both Qwen2-VL and Qwen2.5-VL models.
    """
    
    name = "qwen2_vl"
    eos_tokens = ["<|endoftext|>", "</s>"]
    bos_tokens = ["<|endoftext|>", "<s>"]
    supports_vision = True
    
    @classmethod
    def format_system_prompt(cls, system_prompt: str, user_prompt: str, processor) -> str:
        """Format system and user prompts for Qwen VL models."""
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
            
        # Detect if chat template should be used
        if hasattr(processor, "apply_chat_template"):
            try:
                # Try using the model's chat template
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                formatted_prompt = processor.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}. Falling back to manual formatting.")
        
        # Fall back to manual formatting with im_start/im_end
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    @classmethod
    def get_image_size(cls, processor, config_type: str = "qwen2_vl") -> dict:
        """Get the preferred image size for Qwen VL models."""
        # Default sizes depending on model type
        if "qwen2.5-vl" in config_type or "qwen2-5-vl" in config_type:
            # For Qwen2.5-VL models
            default_size = {"height": 336, "width": 336}
        else:
            # For Qwen2-VL models
            default_size = {"height": 448, "width": 448}
            
        # Try to get the size from the processor if available
        if processor and hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            if isinstance(processor.image_processor.size, dict):
                return processor.image_processor.size
        
        return default_size


class Qwen25VLConfig(Qwen2VLConfig):
    """Configuration for Qwen2.5-VL models.
    
    Inherits from Qwen2VLConfig with specific overrides for Qwen2.5-VL.
    """
    
    name = "qwen2_5_vl"
    
    @classmethod
    def get_image_size(cls, processor, config_type: str = "qwen2_5_vl") -> dict:
        """Get the preferred image size for Qwen2.5-VL models."""
        return super().get_image_size(processor, config_type="qwen2.5-vl")


class CodeModelConfig(MLXModelConfig):
    """Configuration for code generation models (SQLCoder, etc.)."""
    
    name = "code"
    eos_tokens = ["<|endoftext|>", "</s>"]
    bos_tokens = ["<s>"]
    supports_system_prompt = False  # Many code models don't support chat templates
    
    @classmethod
    def format_system_prompt(cls, system_prompt: str, user_prompt: str, processor) -> str:
        """Format system and user prompts for code models."""
        # Most code models expect direct prompts without chat formatting
        # Prepend the system prompt as a comment
        if system_prompt:
            return f"# System: {system_prompt}\n\n{user_prompt}"
        return user_prompt


class GemmaConfig(MLXModelConfig):
    """Configuration for Gemma models."""
    
    name = "gemma"
    eos_tokens = ["<eos>", "</s>"]
    bos_tokens = ["<bos>", "<s>"]


# Create aliases for common model variations
class DefaultModelConfig(MLXModelConfig):
    """Default configuration that works for most models."""
    pass


class Phi3Config(PhiConfig):
    """Configuration for Phi-3 models (inherits from PhiConfig)."""
    name = "phi3"


class Llama3Config(LlamaConfig):
    """Configuration for Llama-3 models (inherits from LlamaConfig)."""
    name = "llama3"


class Llama2Config(LlamaConfig):
    """Configuration for Llama-2 models (inherits from LlamaConfig)."""
    name = "llama2"


class AMThinkingConfig(MLXModelConfig):
    """Configuration for A-M Team Thinking models (DeepSeek-R1 based)."""
    
    name = "am_thinking"
    eos_tokens = ["<|im_end|>", "<|endoftext|>", "</s>", "<think>", "</think>", "<answer>", "</answer>"]
    bos_tokens = ["<|im_start|>"]
    default_repetition_penalty = 1.15  # Higher to prevent loops
    default_temperature = 0.7
    
    @classmethod
    def get_generation_params(cls, temperature: float, **kwargs) -> Dict[str, Any]:
        """Get parameters for generation with stronger loop prevention."""
        params = super().get_generation_params(temperature, **kwargs)
        
        # MLX-LM uses sampler for repetition penalty, not direct parameters
        # The repetition penalty is handled in the base class through sampler creation
        
        return params
    
    @classmethod
    def format_system_prompt(cls, system_prompt: str, user_prompt: str, processor) -> str:
        """Format system and user prompts for AM-Thinking models."""
        if hasattr(processor, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for models without chat template
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


class DeepSeekR1Config(AMThinkingConfig):
    """Configuration for DeepSeek-R1 models."""
    
    name = "deepseek_r1"
    eos_tokens = ["<|endoftext|>", "<|im_end|>", "</s>", "<think>", "</think>", "<answer>", "</answer>"]
    default_repetition_penalty = 1.2  # Even higher for DeepSeek models


class ModelConfigFactory:
    """Factory class to create model configuration objects for MLX models.
    
    This class handles mapping model name patterns to configuration classes.
    """
    
    # Map of model name patterns to configuration classes
    CONFIG_MAP = {
        # Default config
        "default": DefaultModelConfig,
        
        # Phi family
        "phi-3": Phi3Config,
        "phi3": Phi3Config,
        "phi-2": PhiConfig,
        "phi2": PhiConfig,
        "phi": PhiConfig,
        
        # Llama family
        "llama-3": Llama3Config,
        "llama3": Llama3Config,
        "llama-2": Llama2Config,
        "llama2": Llama2Config,
        "llama": LlamaConfig,
        "h2o-danube": LlamaConfig,  # H2O's Danube models are based on Llama
        "wizard": LlamaConfig,       # WizardLM models are based on Llama
        "vicuna": LlamaConfig,       # Vicuna models are based on Llama
        "alpaca": LlamaConfig,       # Alpaca models are based on Llama
        
        
        # Mistral family
        "mistral": MistralConfig,
        "mixtral": MistralConfig,    # Mixtral is based on Mistral
        "zephyr": MistralConfig,     # Zephyr models are based on Mistral
        
        # Gemma family
        "gemma": GemmaConfig,
        "paligemma": PaliGemmaConfig,
        
        # Qwen family
        "qwen2.5-vl": Qwen25VLConfig, # Check this first (more specific)
        "qwen2-5-vl": Qwen25VLConfig, # Alternative naming format
        "qwen2-vl": Qwen2VLConfig,   # Qwen2-VL models
        "qwen": QwenConfig,
        
        # Code models
        "code": CodeModelConfig,
        "sqlcoder": CodeModelConfig,
        "starcoder": CodeModelConfig,
        "codellama": CodeModelConfig,
        "coder": CodeModelConfig,
        
        # AM-Thinking family
        "am-thinking": AMThinkingConfig,
        "am_thinking": AMThinkingConfig,
        "deepseek-r1": DeepSeekR1Config,
        "deepseek_r1": DeepSeekR1Config,
        "deepseek": DeepSeekR1Config,
    }
    
    @classmethod
    def create_config(cls, model_name: str) -> MLXModelConfig:
        """Create a model config instance for a given model name."""
        model_name_lower = model_name.lower()
        
        # Check for specialized configs first (order matters for Qwen variants)
        for pattern, config_class in cls.CONFIG_MAP.items():
            if pattern in model_name_lower:
                logger.debug(f"Using {config_class.__name__} for model {model_name}")
                return config_class()
                
        # Fall back to default config
        logger.debug(f"No specific config found for {model_name}, using DefaultModelConfig")
        return DefaultModelConfig() 