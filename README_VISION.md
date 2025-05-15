# Vision Capabilities with AbstractLLM

This document explains how to use vision models with AbstractLLM and provides a workaround for using MLX-VLM models that are not directly supported by the MLX provider.

## Overview

AbstractLLM can work with vision models, allowing you to process images alongside text. This functionality is built into the MLX provider but requires specific setup to work properly with certain models.

## Tested Vision Models

The following vision models have been tested with AbstractLLM:

- `mlx-community/paligemma-3b-mix-448-8bit` - Works well with the patches in this document
- `mlx-community/Qwen2-VL-2B-Instruct-4bit` - Currently has some compatibility issues

## Requirements

To use vision capabilities with AbstractLLM, you'll need:

1. macOS with Apple Silicon (M1/M2/M3)
2. Python 3.8+
3. `abstractllm` installed
4. `mlx-vlm` installed: `pip install mlx-vlm`
5. `pillow` installed for image processing: `pip install Pillow`

## Basic Usage

Here's how to use a vision model with AbstractLLM:

```python
from abstractllm import create_llm
from PIL import Image

# Load an image
image = Image.open("path/to/image.jpg")

# Create a configuration for a vision model
config = {
    "model": "mlx-community/paligemma-3b-mix-448-8bit",
    "max_tokens": 100,
    "temperature": 0.1
}

# Create the model
llm = create_llm("mlx", **config)

# Generate a response with the image
response = llm.generate("Describe this image in detail.", images=[image])

# Print the response
print(response.content)
```

## Workaround for MLX-VLM Models

Some vision models require using the `mlx_vlm` library directly rather than the default `mlx_lm` that AbstractLLM uses. The following patch can be applied to make AbstractLLM work with these models:

```python
import logging
from abstractllm import create_llm
from abstractllm.interface import ModelParameter
from abstractllm.types import GenerateResponse
from abstractllm.providers.mlx_provider import MLXProvider
from abstractllm.exceptions import GenerationError
import PIL.Image

# Import MLX-VLM
import mlx_vlm
from mlx_vlm import load as load_vlm
from mlx_vlm import generate as generate_vlm
from mlx_vlm.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store original methods before patching
original_check_vision = MLXProvider._check_vision_model
original_load_model = MLXProvider.load_model
original_generate = MLXProvider.generate

# Define a list of known vision models
VLM_MODELS = [
    "mlx-community/paligemma-3b-mix-448-8bit"
]

# Create patched methods
def patched_check_vision(self, model_name):
    """Always returns True for our vision models"""
    if any(vlm_model in model_name.lower() for vlm_model in VLM_MODELS):
        logger.info(f"Patched vision check: Forcing {model_name} to be recognized as vision model")
        return True
    return original_check_vision(self, model_name)
    
def patched_load_model(self):
    """Use MLX-VLM for our known vision models"""
    model_name = self.config_manager.get_param(ModelParameter.MODEL)
    
    # Check if this is one of our vision models
    if any(vlm_model in model_name.lower() for vlm_model in VLM_MODELS):
        try:
            logger.info(f"Patched load_model: Using MLX-VLM for {model_name}")
            self._model, self._processor = load_vlm(model_name, trust_remote_code=True)
            self._config = load_config(model_name, trust_remote_code=True)
            self._is_loaded = True
            self._is_vision_model = True
            return
        except Exception as e:
            logger.error(f"Patched load_model failed: {e}")
    
    # Fall back to original method for other models
    return original_load_model(self)

def patched_generate(self, prompt, system_prompt=None, files=None, stream=False, tools=None, **kwargs):
    """Patched generate method that uses MLX-VLM for vision generation"""
    try:
        # Check if we need to use vision generation
        images = kwargs.get("images", [])
        if images and self._is_vision_model:
            logger.info("Patched generate: Using MLX-VLM for vision generation")
            
            # Need to call load_model if not already loaded
            if not self._is_loaded:
                self.load_model()
                
            # Get generation parameters
            max_tokens = self.config_manager.get_param(ModelParameter.MAX_TOKENS, default=100)
            temperature = self.config_manager.get_param(ModelParameter.TEMPERATURE, default=0.1)
            
            # Generate using MLX-VLM
            logger.info("Generating with MLX-VLM directly...")
            result = generate_vlm(
                model=self._model,
                processor=self._processor,
                image=images[0],  # Currently only using the first image
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature
            )
            
            # Extract the text from the result
            if isinstance(result, tuple):
                text = result[0]
            else:
                text = result
                
            # Create a response object
            response = GenerateResponse(
                content=text,
                raw_response=result,
                usage={
                    "completion_tokens": max_tokens,
                    "total_tokens": len(prompt) + max_tokens  # Approximation
                }
            )
            
            return response
        else:
            # Fall back to original method for non-vision generation
            return original_generate(self, prompt, system_prompt, files, stream, tools, **kwargs)
            
    except Exception as e:
        logger.error(f"Patched generate failed: {e}")
        raise GenerationError(f"Patched vision generation failed: {str(e)}")

# Apply the patches
MLXProvider._check_vision_model = patched_check_vision
MLXProvider.load_model = patched_load_model
MLXProvider.generate = patched_generate
```

## Usage with the Patch

After applying the patch, you can use the API as normal:

```python
# Load image
image = Image.open("path/to/image.jpg")

# Create model with patched MLXProvider
llm = create_llm("mlx", model="mlx-community/paligemma-3b-mix-448-8bit")

# Generate with image
response = llm.generate("Describe this image in detail.", images=[image])

# Print response
print(response.content)
```

## Recommendations for Future Development

To better support vision models in AbstractLLM, consider:

1. Adding direct MLX-VLM support to the MLXProvider
2. Creating a detection mechanism to automatically use the right loader (MLX-LM vs MLX-VLM)
3. Implementing a more comprehensive vision model registry
4. Updating the MLXProvider to handle multi-image inputs

## Troubleshooting

If you encounter issues:

1. Make sure you're using Python 3.8+ on macOS with Apple Silicon
2. Verify that both `mlx-lm` and `mlx-vlm` are installed
3. Check if the model is supported by MLX-VLM (run the test script to check compatibility)
4. Use the direct MLX-VLM API if AbstractLLM integration fails

## Testing

You can use the `test_abstractllm_vision.py` script to test vision models:

```bash
# Test with direct MLX-VLM
python test_abstractllm_vision.py

# Test with AbstractLLM (with patch)
python test_abstractllm_vision.py --use-abstractllm
``` 