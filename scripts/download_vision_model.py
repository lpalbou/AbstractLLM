#!/usr/bin/env python
"""
Script to preload the DeepSeek vision model.

This downloads the model files to the cache so they can be used later without timeouts.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import abstractllm
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from abstractllm import create_llm, ModelParameter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_vision_model")

def download_deepseek_model(cache_dir=None):
    """
    Download and cache the DeepSeek vision model.
    
    Args:
        cache_dir: Optional custom cache directory
    """
    # Define parameters
    model_name = "deepseek-ai/deepseek-vl-7b-chat"
    
    logger.info(f"Downloading and caching {model_name}...")
    
    # Set up cache directory
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = "~/.cache/abstractllm/models"
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    
    logger.info(f"Using cache directory: {cache_dir}")
    
    # Create the LLM instance
    try:
        llm = create_llm("huggingface", **{
            ModelParameter.MODEL: model_name,
            ModelParameter.DEVICE: "cpu",  # Use CPU for downloading
            "trust_remote_code": True,
            "load_timeout": 3600,  # 1 hour timeout
            "disable_flash_attn": True,  # Disable FlashAttention
            "cache_dir": cache_dir
        })
        
        # Load the model - this will download files to cache
        logger.info("Starting model download and preprocessing...")
        llm.load_model()
        logger.info(f"Successfully downloaded and cached {model_name}!")
        
        # Try a simple generation to confirm it works
        logger.info("Testing model with a simple text prompt...")
        sample_response = llm.generate("Hello, please introduce yourself.")
        logger.info(f"Model test successful! Response: {sample_response[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and cache the DeepSeek vision model")
    parser.add_argument("--cache-dir", help="Custom cache directory for models")
    args = parser.parse_args()
    
    success = download_deepseek_model(args.cache_dir)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 