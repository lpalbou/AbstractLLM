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
            
            # Apply repetition penalty if configured
            if cls.default_repetition_penalty != 1.0:
                repetition_penalty = cls.default_repetition_penalty
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


class ModelConfigFactory:
    """Factory for getting the appropriate model configuration."""
    
    # Mapping of model name patterns to config classes
    CONFIG_MAP = {
        # Llama family models and variants
        "llama": LlamaConfig,
        "h2o-danube": LlamaConfig,  # H2O's Danube models are based on Llama
        "wizard": LlamaConfig,       # WizardLM models are based on Llama
        "vicuna": LlamaConfig,       # Vicuna models are based on Llama
        "alpaca": LlamaConfig,       # Alpaca models are based on Llama
        
        # Qwen family
        "qwen": QwenConfig,
        
        # Mistral family
        "mistral": MistralConfig,
        "mixtral": MistralConfig,    # Mixtral is based on Mistral
        "zephyr": MistralConfig,     # Zephyr models are based on Mistral
        
        # Phi family
        "phi": PhiConfig,
        
        # Vision models
        "paligemma": PaliGemmaConfig,
        
        # Code models
        "sqlcoder": CodeModelConfig,
        "starcoder": CodeModelConfig,
        "codellama": CodeModelConfig,
        "coder": CodeModelConfig,
        "code": CodeModelConfig,
    }
    
    @classmethod
    def get_for_model(cls, model_name: str) -> MLXModelConfig:
        """Get the appropriate configuration for a model."""
        model_name_lower = model_name.lower()
        
        # First try direct matches
        for key, config_class in cls.CONFIG_MAP.items():
            if key in model_name_lower:
                logger.info(f"Using {config_class.name} configuration for model {model_name}")
                return config_class()
        
        # If no direct match, try to detect from model config or architecture hint
        # Check for common architecture identifiers in the model name
        architecture_hints = {
            "7b": "llama",  # Most 7B models are Llama-based
            "-13b": "llama",  # Most 13B models are Llama-based
            "3.1": "llama",  # Llama 3.1
            "3.2": "llama",  # Llama 3.2
            "3-": "llama",   # Llama 3
            "70b": "llama",  # Likely Llama
            "bloom": "bloom",
            "neox": "neox",
            "gpt2": "gpt2",
            "gpt-j": "gptj",
            "opt": "opt",
        }
        
        # Try to detect architecture from hints in name
        for hint, arch_family in architecture_hints.items():
            if hint in model_name_lower:
                if arch_family in cls.CONFIG_MAP:
                    logger.info(f"Detected {arch_family} architecture for {model_name} based on name hint")
                    return cls.CONFIG_MAP[arch_family]()
        
        logger.info(f"No specific configuration found for {model_name}, using generic configuration")
        return MLXModelConfig() 