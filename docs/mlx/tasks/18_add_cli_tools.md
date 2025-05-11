# Task 18: Add CLI Tools for MLX Provider

## Description
Implement command-line interface (CLI) tools for managing the MLX provider, including model caching, conversion, and information.

## Requirements
1. Add CLI commands to list and manage cached MLX models
2. Implement commands to convert PyTorch models to MLX format
3. Create commands to get system information for MLX compatibility
4. Add CLI commands to run inference with MLX models

## Implementation Details

Add the following code to the appropriate CLI module (e.g., `abstractllm/cli/mlx_commands.py`):

```python
"""
Command-line tools for the MLX provider.
"""

import argparse
import sys
import platform
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from abstractllm import create_llm, ModelParameter
import logging
from abstractllm.utils.logging import log_request, log_response

logger = logging.getLogger("abstractllm.cli.mlx")

def check_mlx_compatibility() -> bool:
    """Check if the system is compatible with MLX."""
    is_macos = platform.system().lower() == "darwin"
    is_arm = platform.processor() == "arm"
    if not (is_macos and is_arm):
        logger.warning(f"MLX requires macOS with Apple Silicon. Current platform: {platform.system()} {platform.processor()}")
    return is_macos and is_arm

def list_cached_models_cmd(args: argparse.Namespace) -> None:
    """List cached MLX models."""
    if not check_mlx_compatibility():
        print("Warning: This system is not compatible with MLX.")
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print("MLX requires macOS with Apple Silicon (M1/M2/M3).")
        if not args.force:
            return
    
    try:
        from abstractllm.providers.mlx_provider import MLXProvider
        
        logger.info("Retrieving MLX model cache information")
        
        # Get cache information
        cache_info = MLXProvider.get_cache_info(args.cache_dir)
        
        # Print cache information
        print(f"MLX Models Cache ({cache_info['cache_dir']}):")
        print(f"Total size: {cache_info['total_size_human']}")
        print(f"Model count: {cache_info['model_count']}")
        
        if cache_info['model_count'] == 0:
            print("\nNo MLX models found in cache.")
            return
        
        print("\nCached Models:")
        for i, model in enumerate(cache_info['models'], 1):
            print(f"{i}. {model['name']}")
            print(f"   Size: {model['human_size']}")
            print(f"   Last used: {model['last_used']}")
            print()
    except ImportError:
        logger.error("Required dependencies not available for MLX provider")
        print("Error: Required dependencies not available.")
        print("Please install with: pip install 'abstractllm[mlx]'")

def clear_model_cache_cmd(args: argparse.Namespace) -> None:
    """Clear MLX model cache."""
    if not check_mlx_compatibility() and not args.force:
        print("Warning: This system is not compatible with MLX.")
        print("Use --force to run anyway.")
        return
    
    try:
        from abstractllm.providers.mlx_provider import MLXProvider
        
        logger.info(f"Clearing MLX model cache: {'specific model: ' + args.model if args.model else 'all models'}")
        
        if args.model:
            # Clear specific model
            MLXProvider.clear_model_cache(args.model)
            print(f"Cleared {args.model} from in-memory cache.")
        else:
            # Clear all models
            MLXProvider.clear_model_cache()
            print("Cleared all models from in-memory cache.")
            
        if args.prune:
            logger.info(f"Pruning disk cache with max size: {args.max_size} GB")
            
            # Prune disk cache
            deleted = MLXProvider.prune_cache(args.max_size, args.cache_dir)
            if deleted:
                print(f"Pruned disk cache, removed {len(deleted)} models:")
                for model in deleted:
                    print(f"  - {model}")
            else:
                print("No models needed to be pruned from disk cache.")
    except ImportError:
        logger.error("Required dependencies not available for MLX provider")
        print("Error: Required dependencies not available.")
        print("Please install with: pip install 'abstractllm[mlx]'")

def list_available_models_cmd(args: argparse.Namespace) -> None:
    """List available MLX models on Hugging Face."""
    try:
        from abstractllm.providers.mlx_provider import MLXProvider
        
        logger.info("Searching for available MLX models on Hugging Face Hub")
        print("Searching for MLX models on Hugging Face...")
        models = MLXProvider.list_available_mlx_models()
        
        if not models:
            logger.warning("No MLX models found or error connecting to Hugging Face")
            print("No MLX models found or error connecting to Hugging Face.")
            return
        
        print(f"Found {len(models)} MLX models:")
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['id']}")
            print(f"   Downloads: {model['downloads']}")
            
            if model['pipeline_tag']:
                print(f"   Pipeline: {model['pipeline_tag']}")
                
            if 'last_modified' in model and model['last_modified']:
                print(f"   Last modified: {model['last_modified']}")
                
            print()
    except ImportError:
        logger.error("Required dependencies not available for MLX provider")
        print("Error: Required dependencies not available.")
        print("Please install with: pip install 'abstractllm[mlx]'")

def convert_model_cmd(args: argparse.Namespace) -> None:
    """Convert a PyTorch model to MLX format."""
    if not check_mlx_compatibility() and not args.force:
        print("Warning: This system is not compatible with MLX.")
        print("Use --force to run anyway.")
        return
    
    try:
        from abstractllm.providers.mlx_provider import MLXProvider
        
        logger.info(f"Converting model {args.model_id} to MLX format")
        print(f"Converting {args.model_id} to MLX format...")
        
        output_dir = args.output_dir
        if output_dir:
            output_dir = os.path.expanduser(output_dir)
        
        try:
            model_dir = MLXProvider.convert_to_mlx(args.model_id, output_dir)
            logger.info(f"Successfully converted model to {model_dir}")
            print(f"Successfully converted to MLX format at: {model_dir}")
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            print(f"Error converting model: {e}")
    except ImportError:
        logger.error("Required dependencies not available for MLX provider")
        print("Error: Required dependencies not available.")
        print("Please install with: pip install 'abstractllm[mlx]'")

def run_inference_cmd(args: argparse.Namespace) -> None:
    """Run inference with an MLX model."""
    if not check_mlx_compatibility():
        print("Error: This system is not compatible with MLX.")
        print(f"Platform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print("MLX requires macOS with Apple Silicon (M1/M2/M3).")
        return
    
    try:
        from abstractllm import create_llm
        
        logger.info(f"Running inference with MLX model: {args.model}")
        print(f"Running inference with model: {args.model}")
        
        # Create LLM
        llm = create_llm(
            "mlx",
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Process input
        if args.input_file:
            with open(args.input_file, 'r') as f:
                prompt = f.read()
                logger.info(f"Using prompt from file: {args.input_file}")
        else:
            prompt = args.prompt
            logger.info("Using prompt from command line")
        
        # Process files
        files = []
        if args.files:
            files = [Path(f) for f in args.files]
            logger.info(f"Including {len(files)} file(s) in the request")
        
        # Log the request to the central logging system
        model_name = llm.config_manager.get_param(ModelParameter.MODEL)
        log_request("mlx", prompt, {
            "model": model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": args.stream,
            "has_files": bool(files)
        })
        
        # Run inference
        print("\nGenerating response...\n")
        
        if args.stream:
            # Stream response
            print("Response:")
            full_text = ""
            for chunk in llm.generate(prompt, stream=True, files=files):
                # Print just the new content
                new_text = chunk.text[len(full_text):]
                print(new_text, end="", flush=True)
                full_text = chunk.text
            print("\n")
            
            # Log the final response
            log_response("mlx", full_text)
        else:
            # Generate response
            response = llm.generate(prompt, files=files)
            
            # Print response (logging is handled in the provider)
            print("Response:")
            print(response.text)
            print(f"\nToken usage: {response.prompt_tokens} prompt, " 
                  f"{response.completion_tokens} completion")
    except ImportError:
        logger.error("Required dependencies not available for MLX provider")
        print("Error: Required dependencies not available.")
        print("Please install with: pip install 'abstractllm[mlx]'")
    except Exception as e:
        logger.error(f"Error during MLX inference: {e}")
        print(f"Error: {e}")

def check_system_cmd(args: argparse.Namespace) -> None:
    """Check system compatibility with MLX."""
    is_macos = platform.system().lower() == "darwin"
    is_arm = platform.processor() == "arm"
    
    logger.info("Running MLX compatibility check")
    
    print("MLX Compatibility Check:")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    
    print(f"\nMacOS: {'✅' if is_macos else '❌'}")
    print(f"Apple Silicon: {'✅' if is_arm else '❌'}")
    print(f"MLX compatible: {'✅' if (is_macos and is_arm) else '❌'}")
    
    # Check for required packages
    try:
        import mlx
        print(f"MLX installed: ✅ (version {mlx.__version__})")
        logger.info(f"MLX installed: version {mlx.__version__}")
    except ImportError:
        print("MLX installed: ❌")
        logger.warning("MLX package not installed")
    
    try:
        import mlx_lm
        print(f"MLX-LM installed: ✅")
        logger.info("MLX-LM package installed")
    except ImportError:
        print("MLX-LM installed: ❌")
        logger.warning("MLX-LM package not installed")
    
    if is_macos and is_arm:
        print("\nThis system is compatible with MLX! ✅")
        logger.info("System is compatible with MLX")
    else:
        print("\nThis system is NOT compatible with MLX. ❌")
        print("MLX requires macOS with Apple Silicon (M1/M2/M3).")
        logger.warning("System is not compatible with MLX")

def register_mlx_commands(subparsers) -> None:
    """Register MLX commands with the CLI parser."""
    # MLX parent parser
    mlx_parser = subparsers.add_parser(
        "mlx", 
        help="MLX provider tools"
    )
    mlx_subparsers = mlx_parser.add_subparsers(dest="mlx_command", required=True)
    
    # List cached models
    list_parser = mlx_subparsers.add_parser(
        "list-cache", 
        help="List cached MLX models"
    )
    list_parser.add_argument(
        "--cache-dir",
        help="Custom cache directory path"
    )
    list_parser.add_argument(
        "--force",
        action="store_true",
        help="Force operation even on non-compatible systems"
    )
    list_parser.set_defaults(func=list_cached_models_cmd)
    
    # Clear model cache
    clear_parser = mlx_subparsers.add_parser(
        "clear-cache", 
        help="Clear MLX model cache"
    )
    clear_parser.add_argument(
        "--model",
        help="Specific model to clear (or all if not specified)"
    )
    clear_parser.add_argument(
        "--prune",
        action="store_true",
        help="Prune disk cache to stay under size limit"
    )
    clear_parser.add_argument(
        "--max-size",
        type=float,
        default=10.0,
        help="Maximum cache size in GB for pruning"
    )
    clear_parser.add_argument(
        "--cache-dir",
        help="Custom cache directory path"
    )
    clear_parser.add_argument(
        "--force",
        action="store_true",
        help="Force operation even on non-compatible systems"
    )
    clear_parser.set_defaults(func=clear_model_cache_cmd)
    
    # List available models
    available_parser = mlx_subparsers.add_parser(
        "list-available", 
        help="List available MLX models on Hugging Face"
    )
    available_parser.set_defaults(func=list_available_models_cmd)
    
    # Convert model
    convert_parser = mlx_subparsers.add_parser(
        "convert", 
        help="Convert a PyTorch model to MLX format"
    )
    convert_parser.add_argument(
        "model_id",
        help="Hugging Face model ID to convert"
    )
    convert_parser.add_argument(
        "--output-dir",
        help="Output directory for converted model"
    )
    convert_parser.add_argument(
        "--force",
        action="store_true",
        help="Force operation even on non-compatible systems"
    )
    convert_parser.set_defaults(func=convert_model_cmd)
    
    # Run inference
    run_parser = mlx_subparsers.add_parser(
        "run", 
        help="Run inference with an MLX model"
    )
    run_parser.add_argument(
        "--model",
        default="mlx-community/phi-2",
        help="MLX model to use"
    )
    run_parser.add_argument(
        "--prompt",
        help="Text prompt for generation"
    )
    run_parser.add_argument(
        "--input-file",
        help="File containing prompt text"
    )
    run_parser.add_argument(
        "--files",
        nargs="+",
        help="Additional files to process (e.g., text, images)"
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens to generate"
    )
    run_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response"
    )
    run_parser.set_defaults(func=run_inference_cmd)
    
    # Check system
    check_parser = mlx_subparsers.add_parser(
        "check-system", 
        help="Check system compatibility with MLX"
    )
    check_parser.set_defaults(func=check_system_cmd)
```

Then, update the main CLI module to include these commands:

```python
# In abstractllm/cli/__init__.py or similar

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="AbstractLLM Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Register commands
    # ... existing commands
    
    # Register MLX commands
    from abstractllm.cli.mlx_commands import register_mlx_commands
    register_mlx_commands(subparsers)
    
    # Parse args and execute
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
```

## References
- See AbstractLLM's existing CLI structure
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- Reference the MLX Usage Examples: `docs/mlx/mlx_usage_examples.md`

## Testing
1. Test each CLI command individually:
   ```
   python -m abstractllm mlx check-system
   python -m abstractllm mlx list-available
   python -m abstractllm mlx list-cache
   python -m abstractllm mlx run --prompt "Hello, world!"
   ```
2. Test with various options and parameters
3. Test error handling and edge cases 