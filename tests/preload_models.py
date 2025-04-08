#!/usr/bin/env python
"""
Utility script to preload models for testing.
This helps avoid timeouts during testing by ensuring models are already downloaded and loaded.

Example usage:
    # Preload all models
    python tests/preload_models.py all
    
    # Preload just the HuggingFace models
    python tests/preload_models.py huggingface
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path

# Add the parent directory to the path so we can import abstractllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from abstractllm import create_llm, ModelParameter
from abstractllm.providers.huggingface import DEFAULT_MODEL as DEFAULT_HF_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("preload_models")


def preload_huggingface_model(model_name=DEFAULT_HF_MODEL):
    """
    Preload a HuggingFace model.
    
    Args:
        model_name: Name of the model to preload
    """
    try:
        logger.info(f"Preloading HuggingFace model: {model_name}")
        start_time = time.time()
        
        provider = create_llm("huggingface", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.DEVICE: "cpu",     # Use CPU for testing
            "auto_load": True,                # Enable auto-loading 
            "auto_warmup": True,              # Enable auto-warmup
            "load_timeout": 300,              # Longer timeout for initial load
            "trust_remote_code": True         # Allow trusted code execution if needed
        })
        
        # Make sure the model is loaded - should already be handled by auto_load flag, but let's be sure
        if not hasattr(provider, "_model_loaded") or not provider._model_loaded:
            if hasattr(provider, "load_model"):
                logger.info("Explicitly calling load_model()")
                provider.load_model()
        
        # Make sure warmup is done
        if not hasattr(provider, "_warmup_completed") or not provider._warmup_completed:
            if hasattr(provider, "warmup"):
                logger.info("Explicitly calling warmup()")
                provider.warmup()
        
        # Do a simple generation to make sure everything works
        logger.info("Testing model with a simple generation...")
        response = provider.generate("Hello world", max_tokens=5)
        logger.info(f"Test response: {response}")
        
        end_time = time.time()
        logger.info(f"Successfully preloaded HuggingFace model in {end_time - start_time:.2f}s")
        return True
    except Exception as e:
        logger.error(f"Failed to preload HuggingFace model: {e}")
        return False


def preload_openai():
    """
    Verify OpenAI settings and credentials.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, skipping OpenAI check")
        return False
    
    try:
        logger.info("Testing OpenAI provider...")
        provider = create_llm("openai")
        return True
    except Exception as e:
        logger.error(f"Failed to set up OpenAI provider: {e}")
        return False


def preload_anthropic():
    """
    Verify Anthropic settings and credentials.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set, skipping Anthropic check")
        return False
    
    try:
        logger.info("Testing Anthropic provider...")
        provider = create_llm("anthropic")
        return True
    except Exception as e:
        logger.error(f"Failed to set up Anthropic provider: {e}")
        return False


def preload_ollama():
    """
    Check if Ollama is running and models are available.
    """
    try:
        import requests
        logger.info("Checking for Ollama availability...")
        response = requests.get("http://localhost:11434/api/tags")
        
        if response.status_code != 200:
            logger.warning("Ollama API not accessible")
            return False
        
        # Check if phi4-mini model exists
        models = response.json().get("models", [])
        model_exists = any(model["name"].startswith("phi4-mini") for model in models)
        
        model_name = "phi4-mini:latest"
        if not model_exists:
            logger.warning(f"{model_name} not available, you may need to pull it first with 'ollama pull {model_name}'")
            logger.info("Continuing with test anyway using the specified model")
        else:
            logger.info(f"Found Ollama model: {model_name}")
        
        # Test the provider with phi4-mini model
        provider = create_llm("ollama", **{
            ModelParameter.MODEL: model_name
        })
        
        return True
    except Exception as e:
        logger.error(f"Failed to check Ollama: {e}")
        return False


def preload_all():
    """
    Preload all available models.
    """
    successes = []
    failures = []
    
    if preload_huggingface_model():
        successes.append("huggingface")
    else:
        failures.append("huggingface")
        
    if preload_openai():
        successes.append("openai")
    else:
        failures.append("openai")
        
    if preload_anthropic():
        successes.append("anthropic")
    else:
        failures.append("anthropic")
        
    if preload_ollama():
        successes.append("ollama")
    else:
        failures.append("ollama")
    
    logger.info(f"Successfully preloaded: {', '.join(successes)}")
    if failures:
        logger.warning(f"Failed to preload: {', '.join(failures)}")


def main():
    parser = argparse.ArgumentParser(description="Preload models for testing")
    parser.add_argument("provider", choices=["all", "huggingface", "openai", "anthropic", "ollama"],
                      help="Which provider's models to preload")
    parser.add_argument("--model", default=DEFAULT_HF_MODEL, 
                      help=f"Model name to preload for HuggingFace (default: {DEFAULT_HF_MODEL})")
    
    args = parser.parse_args()
    
    if args.provider == "all":
        preload_all()
    elif args.provider == "huggingface":
        preload_huggingface_model(args.model)
    elif args.provider == "openai":
        preload_openai()
    elif args.provider == "anthropic":
        preload_anthropic()
    elif args.provider == "ollama":
        preload_ollama()


if __name__ == "__main__":
    main() 