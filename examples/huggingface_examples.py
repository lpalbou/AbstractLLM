"""
HuggingFace Provider Examples

This file contains practical examples of using the HuggingFace provider in AbstractLLM.
Each example demonstrates key features and best practices.
"""

import sys
import platform
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from abstractllm import create_llm
from abstractllm.enums import ModelParameter
from abstractllm.exceptions import (
    ModelLoadingError,
    GenerationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderAPIError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_optimal_device_config() -> Dict[str, Any]:
    """
    Determine optimal device configuration based on system capabilities.
    Returns a device configuration dict.
    """
    import torch
    
    device_config = {"device_map": "cpu", "torch_dtype": "float32"}
    
    try:
        if torch.cuda.is_available():
            device_config["device_map"] = "auto"
            device_config["torch_dtype"] = "auto"
            # Disable Flash Attention on Windows due to compatibility issues
            device_config["use_flash_attention"] = platform.system() != "Windows"
            logger.info("CUDA device detected, using GPU acceleration")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_config["device_map"] = "mps"
            device_config["torch_dtype"] = "float16"
            device_config["use_flash_attention"] = False
            logger.info("Apple Silicon GPU detected, using MPS device")
        else:
            logger.info("No GPU detected, using CPU")
    except Exception as e:
        logger.warning(f"Error detecting device capabilities: {e}")
    
    return device_config

def create_base_config(model_name: str = "microsoft/phi-2") -> Dict[str, Any]:
    """
    Create a base configuration with optimal settings.
    Args:
        model_name: The model to use
    Returns:
        Configuration dictionary
    """
    # Get optimal device configuration
    device_config = get_optimal_device_config()
    
    # Create base configuration
    config = {
        ModelParameter.MODEL: model_name,
        ModelParameter.TEMPERATURE: 0.7,
        ModelParameter.MAX_TOKENS: 100,
        ModelParameter.TOP_P: 0.9,
        ModelParameter.TOP_K: 50,
        ModelParameter.REPETITION_PENALTY: 1.1,
        "trust_remote_code": True,
        "use_safetensors": True
    }
    
    # Update with device configuration
    config.update(device_config)
    
    return config

def show_model_recommendations() -> None:
    """Show model recommendations for different tasks."""
    # Define recommended models
    recommendations = {
        "text-generation": [
            ("meta-llama/Llama-2-7b-chat-hf", "High-quality chat model"),
            ("microsoft/phi-2", "Efficient general-purpose model"),
            ("mistralai/Mistral-7B-v0.1", "Strong open-source model")
        ],
        "text2text": [
            ("google/flan-t5-base", "Versatile text-to-text model"),
            ("facebook/bart-large", "Strong summarization model"),
            ("t5-base", "General-purpose T5 model")
        ],
        "vision": [
            ("openai/clip-vit-base-patch32", "Strong vision-language model"),
            ("microsoft/git-base", "Good for image captioning"),
            ("Salesforce/blip-image-captioning-base", "Efficient image understanding")
        ],
        "speech": [
            ("openai/whisper-base", "Reliable speech recognition"),
            ("microsoft/speecht5_tts", "High-quality text-to-speech"),
            ("facebook/wav2vec2-base", "Good for speech processing")
        ]
    }
    
    print("\nModel Recommendations:")
    print("=====================")
    
    for task, models in recommendations.items():
        print(f"\n{task.upper()}:")
        for model, description in models:
            print(f"- {model}")
            print(f"  {description}")
    print()

def run_example(config: Dict[str, Any], output_dir: Optional[Path] = None) -> None:
    """
    Run examples demonstrating key features.
    Args:
        config: Provider configuration
        output_dir: Optional directory for saving outputs
    """
    provider = None
    try:
        # Initialize provider
        provider = create_llm("huggingface", **config)
        logger.info(f"Created provider with model: {config[ModelParameter.MODEL]}")
        
        # Basic generation
        logger.info("Running basic generation example...")
        prompt = "Write a one-sentence story about hope."
        print(f"\nPrompt: {prompt}")
        result = provider.generate(prompt)
        print(f"Output: {result}\n")
        
        # Generation with system prompt
        logger.info("Running system prompt example...")
        prompt = "What is quantum computing?"
        system_prompt = "You are a physics professor explaining concepts to beginners."
        print(f"\nSystem: {system_prompt}")
        print(f"Prompt: {prompt}")
        result = provider.generate(prompt, system_prompt=system_prompt)
        print(f"Output: {result}\n")
        
        # Streaming generation
        logger.info("Running streaming example...")
        prompt = "Write a haiku about nature."
        print(f"\nPrompt: {prompt}")
        print("Output:", end=" ", flush=True)
        for chunk in provider.generate(prompt, stream=True):
            print(chunk, end="", flush=True)
        print("\n")
        
        # Get model capabilities
        capabilities = provider.get_capabilities()
        print("\nModel Capabilities:")
        for capability, value in capabilities.items():
            print(f"- {capability}: {value}")
        
    except ModelLoadingError as e:
        logger.error(f"Failed to load model: {e}")
        if hasattr(e, 'details'):
            logger.error(f"Details: {e.details}")
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e}")
        if e.reason:
            logger.error(f"Reason: {e.reason}")
    except GenerationError as e:
        logger.error(f"Generation failed: {e}")
        if hasattr(e, 'details'):
            logger.error(f"Details: {e.details}")
    except InvalidRequestError as e:
        logger.error(f"Invalid request: {e}")
        if hasattr(e, 'details'):
            logger.error(f"Details: {e.details}")
    except ProviderAPIError as e:
        logger.error(f"Provider API error: {e}")
        if hasattr(e, 'details'):
            logger.error(f"Details: {e.details}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if provider and hasattr(provider, '_pipeline'):
            logger.info("Cleaning up resources...")
            if provider._pipeline:
                provider._pipeline.cleanup()

def main():
    """Main entry point for examples."""
    try:
        # Create output directory
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Show model recommendations
        show_model_recommendations()
        
        # Create configuration
        config = create_base_config()
        logger.info(f"Created configuration: {config}")
        
        # Run examples
        print("\nRunning Examples:")
        run_example(config, output_dir)
        
    except KeyboardInterrupt:
        logger.info("Examples interrupted by user")
    except Exception as e:
        logger.error(f"Failed to run examples: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()