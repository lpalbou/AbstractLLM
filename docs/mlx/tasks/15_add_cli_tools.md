# Task 15: Add CLI Tools

## Description
Add comprehensive CLI tools for the MLX provider, including support for vision capabilities.

## Requirements
1. Add vision model support to CLI
2. Support image file inputs
3. Add memory and performance monitoring
4. Support streaming output
5. Add error handling and reporting

## Implementation Details

### CLI Implementation

Update `abstractllm/cli/mlx_cli.py`:

```python
#!/usr/bin/env python3
"""CLI tools for MLX provider."""

import os
import sys
import click
import psutil
from pathlib import Path
from typing import List, Optional
from PIL import Image
import json
import asyncio
from datetime import datetime

from abstractllm import create_llm
from abstractllm.enums import ModelParameter
from abstractllm.exceptions import (
    ImageProcessingError,
    MemoryError,
    UnsupportedFeatureError
)

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"

def monitor_resources() -> dict:
    """Get current resource usage."""
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    return {
        "rss": format_size(memory.rss),
        "vms": format_size(memory.vms),
        "cpu_percent": process.cpu_percent(),
        "memory_percent": process.memory_percent()
    }

def validate_image(ctx, param, value) -> List[Path]:
    """Validate image file inputs."""
    if not value:
        return []
    
    paths = []
    for path in value:
        try:
            path = Path(path)
            if not path.exists():
                raise click.BadParameter(f"Image file not found: {path}")
            
            # Verify it's an image
            try:
                Image.open(path).verify()
            except Exception:
                raise click.BadParameter(f"Invalid image file: {path}")
            
            paths.append(path)
        except Exception as e:
            raise click.BadParameter(str(e))
    
    return paths

@click.group()
def mlx():
    """MLX provider CLI tools."""
    pass

@mlx.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="MLX model to use (e.g., mlx-community/Qwen2.5-VL-32B-Instruct-6bit)"
)
@click.option(
    "--prompt",
    "-p",
    required=True,
    help="Generation prompt"
)
@click.option(
    "--system-prompt",
    "-s",
    help="Optional system prompt"
)
@click.option(
    "--image",
    "-i",
    multiple=True,
    callback=validate_image,
    help="Image file(s) to analyze (can be specified multiple times)"
)
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Enable streaming output"
)
@click.option(
    "--monitor/--no-monitor",
    default=False,
    help="Enable resource monitoring"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save response to file"
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output in JSON format"
)
@click.option(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Generation temperature"
)
@click.option(
    "--max-tokens",
    type=int,
    default=1024,
    help="Maximum tokens to generate"
)
@click.option(
    "--quantize/--no-quantize",
    default=True,
    help="Enable model quantization"
)
def generate(
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    image: List[Path],
    stream: bool,
    monitor: bool,
    output: Optional[str],
    json_output: bool,
    temperature: float,
    max_tokens: int,
    quantize: bool
):
    """Generate text or analyze images with MLX model."""
    try:
        # Create provider
        llm = create_llm("mlx", **{
            ModelParameter.MODEL: model,
            ModelParameter.TEMPERATURE: temperature,
            ModelParameter.MAX_TOKENS: max_tokens,
            "quantize": quantize
        })
        
        # Initialize monitoring
        start_time = datetime.now()
        start_resources = monitor_resources() if monitor else None
        
        if stream:
            # Streaming generation
            response_text = ""
            for chunk in llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                files=[str(img) for img in image] if image else None,
                stream=True
            ):
                if json_output:
                    click.echo(json.dumps({
                        "chunk": chunk.content,
                        "usage": chunk.usage
                    }))
                else:
                    click.echo(chunk.content, nl=False)
                response_text += chunk.content
        else:
            # Single response
            response = llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                files=[str(img) for img in image] if image else None
            )
            response_text = response.content
            
            if json_output:
                click.echo(json.dumps({
                    "response": response.content,
                    "usage": response.usage,
                    "model": response.model
                }))
            else:
                click.echo(response_text)
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                f.write(response_text)
        
        # Show monitoring info
        if monitor:
            end_time = datetime.now()
            end_resources = monitor_resources()
            duration = (end_time - start_time).total_seconds()
            
            monitoring_info = {
                "duration_seconds": duration,
                "start_resources": start_resources,
                "end_resources": end_resources
            }
            
            if json_output:
                click.echo(json.dumps({"monitoring": monitoring_info}))
            else:
                click.echo("\nResource Usage:", err=True)
                click.echo(f"Duration: {duration:.2f}s", err=True)
                click.echo(f"Start: {json.dumps(start_resources, indent=2)}", err=True)
                click.echo(f"End: {json.dumps(end_resources, indent=2)}", err=True)
    
    except ImageProcessingError as e:
        click.echo(f"Error processing image: {e}", err=True)
        sys.exit(1)
    except MemoryError as e:
        click.echo(f"Memory error: {e}", err=True)
        sys.exit(1)
    except UnsupportedFeatureError as e:
        click.echo(f"Unsupported feature: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@mlx.command()
@click.argument("image", type=click.Path(exists=True))
def check_image(image: str):
    """Check if an image is valid and show its properties."""
    try:
        path = Path(image)
        with Image.open(path) as img:
            info = {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "file_size": format_size(path.stat().st_size)
            }
            
            click.echo(json.dumps(info, indent=2))
            
            # Check if size is reasonable for MLX models
            width, height = img.size
            if width > 1000 or height > 1000:
                click.echo("\nWarning: Image is large and may need resizing for MLX models", err=True)
            
            # Check memory requirements
            estimated_memory = (width * height * 3) * 4  # Rough estimate for float32 RGB
            if estimated_memory > 1024 * 1024 * 1024:  # 1GB
                click.echo("\nWarning: Image may require significant memory for processing", err=True)
    
    except Exception as e:
        click.echo(f"Error checking image: {e}", err=True)
        sys.exit(1)

@mlx.command()
def list_models():
    """List available MLX models."""
    models = {
        "text_models": [
            "mlx-community/mistral-7b-v0.1",
            "mlx-community/phi-2",
            "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX"
        ],
        "vision_models": [
            "mlx-community/Qwen2.5-VL-32B-Instruct-6bit",
            "mlx-community/llava-v1.6-34b-mlx",
            "mlx-community/kimi-vl-70b-instruct-mlx"
        ]
    }
    
    click.echo(json.dumps(models, indent=2))

@mlx.command()
def system_check():
    """Check system compatibility and requirements."""
    import platform
    
    # Check system requirements
    checks = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "memory_total": format_size(psutil.virtual_memory().total),
        "memory_available": format_size(psutil.virtual_memory().available)
    }
    
    # Check if system meets requirements
    is_macos = platform.system().lower() == "darwin"
    is_arm = platform.processor() == "arm"
    has_memory = psutil.virtual_memory().total >= 32 * 1024 * 1024 * 1024  # 32GB
    
    checks["compatible"] = is_macos and is_arm
    checks["memory_sufficient"] = has_memory
    
    click.echo(json.dumps(checks, indent=2))
    
    if not checks["compatible"]:
        click.echo("\nWarning: System may not be compatible with MLX (requires Apple Silicon)", err=True)
    if not checks["memory_sufficient"]:
        click.echo("\nWarning: System may have insufficient memory for larger models", err=True)

if __name__ == "__main__":
    mlx()
```

### Example Usage

```bash
# Basic text generation
abstractllm mlx generate -m mlx-community/phi-2 -p "Hello, world!"

# Image analysis
abstractllm mlx generate \
    -m mlx-community/Qwen2.5-VL-32B-Instruct-6bit \
    -p "What's in this image?" \
    -i image.jpg

# Multiple images with streaming
abstractllm mlx generate \
    -m mlx-community/Qwen2.5-VL-32B-Instruct-6bit \
    -p "Compare these images." \
    -i image1.jpg -i image2.jpg \
    --stream

# With system prompt and monitoring
abstractllm mlx generate \
    -m mlx-community/Qwen2.5-VL-32B-Instruct-6bit \
    -p "Analyze this artwork." \
    -s "You are an art critic." \
    -i artwork.jpg \
    --monitor

# Check image properties
abstractllm mlx check-image large_image.jpg

# Check system compatibility
abstractllm mlx system-check

# List available models
abstractllm mlx list-models
```

## References
- See `docs/mlx/vision-upgrade.md` for vision implementation details
- See `docs/mlx/deepsearch-mlx-vlm.md` for MLX-VLM insights
- See Click documentation for CLI framework details

## Testing
Test the CLI tools:
1. Test all commands with various inputs
2. Verify error handling
3. Test resource monitoring
4. Test image validation
5. Test streaming output

## Success Criteria
1. All CLI commands work as expected
2. Vision capabilities are fully supported
3. Error handling is robust
4. Resource monitoring is accurate
5. Image validation is thorough
6. Documentation is clear and complete 