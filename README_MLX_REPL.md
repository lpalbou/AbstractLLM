# MLX REPL for AbstractLLM

This is a simple interactive REPL (Read-Eval-Print Loop) for testing the MLX provider in AbstractLLM.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- AbstractLLM installed
- MLX packages installed
- PyTorch (required for vision models)

## Installation

Make sure you have the necessary packages installed:

```bash
# Install MLX and related packages
pip install mlx mlx-lm mlx-vlm

# For vision models, PyTorch is REQUIRED
pip install torch

# For listing models, install huggingface_hub
pip install huggingface_hub

# For better image handling
pip install Pillow
```

## Usage

Run the REPL with:

```bash
./mlx_repl.py
```

## Available Commands

- `model [MODEL_NAME]`: Set or show the current model
- `temp [TEMPERATURE]`: Set or show temperature (0.0-1.0)
- `tokens [MAX_TOKENS]`: Set or show maximum tokens
- `stream [on|off]`: Toggle streaming mode
- `image [PATH]`: Add an image for vision models (with automatic resize option)
- `test_image [PATH]`: Test if an image can be processed correctly
- `clear_images`: Clear all added images
- `list_compatible_vision`: Show list of known working vision models
- `ask PROMPT`: Ask the model a question
- `info`: Show current settings
- `models [FILTER]`: List available MLX models from HuggingFace
- `exit`, `quit`, or Ctrl+D: Exit the REPL

## Compatible Vision Models

Not all vision models work properly with MLX. The following models have been tested and are known to work well:

1. `mlx-community/llava-1.5-7b-4bit`
2. `mlx-community/bakllava-1-4bit`
3. `mlx-community/llava-phi-2-vision-4bit`
4. `mlx-community/phi-3-vision-128k-instruct-4bit`

⚠️ **Note**: Qwen2-VL models currently have compatibility issues with the MLX framework.

## CRITICAL: PyTorch Requirement

Vision models in MLX require PyTorch to be installed. This is not optional! Without PyTorch, you will get errors like:

```
Failed to process inputs with error: total images must be the same as the number of image tags, got 0 image tags and 1 images. Please install PyTorch and try again.
```

The REPL will check for PyTorch and warn you if it's missing.

## Image Tags for Vision Models

Different vision models require specific formatting for image tags. The REPL handles this automatically, but for reference:

- **Phi-3 Vision**: `Image: <image>\nUser: {prompt}\nAssistant:`
- **LLaVA models**: `<image>\n{prompt}`
- **Qwen models**: `<img>{prompt}`

## Examples

```
# Set a text model
mlx> model mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX

# Ask a question
mlx> ask What is the capital of France?

# Set a vision model
mlx> model mlx-community/phi-3-vision-128k-instruct-4bit

# Add and test an image
mlx> test_image tests/examples/mountain_path.jpg
mlx> image tests/examples/mountain_path.jpg

# Ask about the image
mlx> ask What's in this image?

# List available models containing "mistral"
mlx> models mistral

# List compatible vision models
mlx> list_compatible_vision
```

## Image Size and Compatibility

Very large images may cause issues with some models, particularly with the error:
```
ValueError: [broadcast_shapes] Shapes cannot be broadcast
```

To avoid this:
1. Use the `test_image` command to check image size
2. The `image` command will automatically offer to resize images larger than 1024px
3. Use image formats like JPEG, PNG or WebP for best compatibility

## Troubleshooting

If you encounter errors:

1. Make sure you're running on macOS with Apple Silicon
2. Ensure all required packages are installed
3. **Make sure PyTorch is installed for vision models**
4. If using a vision model, check that the prompt includes the proper image tags (automatic in recent versions)
5. For vision models, use the known compatible models listed with `list_compatible_vision`
6. Use `clear_images` and try again with a single image (some models only support one image at a time)
7. Resize large images to less than 1024px max dimension
8. Check that the model exists and is accessible
9. For large models, ensure you have enough memory 