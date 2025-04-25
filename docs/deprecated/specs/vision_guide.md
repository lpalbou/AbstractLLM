# Vision Capabilities in AbstractLLM

This guide explains how to use AbstractLLM's vision capabilities to interact with multimodal models that can process both text and images.

## Supported Models

AbstractLLM supports vision capabilities for the following providers and models:

### OpenAI
- `gpt-4-vision-preview`
- `gpt-4-turbo`
- `gpt-4o`
- `gpt-4o-2024-05-13`

### Anthropic
- `claude-3-opus`
- `claude-3-sonnet`
- `claude-3-haiku`
- `claude-3.5-sonnet`
- `claude-3.5-haiku`

### Ollama
- `llama3.2-vision`
- `deepseek-janus-pro`
- `erwan2/DeepSeek-Janus-Pro-7B`

## Basic Usage

Using vision capabilities is straightforward with AbstractLLM:

```python
from abstractllm import create_llm, ModelParameter, ModelCapability

# Create an LLM instance with a vision-capable model
llm = create_llm("openai", **{
    ModelParameter.MODEL: "gpt-4o",
    ModelParameter.TEMPERATURE: 0.7
})

# Check if vision is supported
capabilities = llm.get_capabilities()
if capabilities.get(ModelCapability.VISION):
    # Use the vision capability with an image URL
    prompt = "What can you see in this image? Please describe it in detail."
    image_url = "https://example.com/image.jpg"
    
    response = llm.generate(prompt, image=image_url)
    print(response)
```

## Input Types

AbstractLLM accepts several types of image inputs:

### Image URLs

```python
# Using an image URL
image_url = "https://example.com/image.jpg"
response = llm.generate("Describe this image", image=image_url)
```

### Local Image Files

```python
# Using a local image file path
image_path = "/path/to/local/image.jpg"
response = llm.generate("Describe this image", image=image_path)
```

### Base64-Encoded Images

```python
# Using a base64-encoded image string
import base64
with open("/path/to/image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
response = llm.generate("Describe this image", image=base64_image)
```

### Multiple Images

```python
# Using multiple images
images = [
    "https://example.com/image1.jpg",
    "/path/to/local/image2.jpg"
]

response = llm.generate(
    "Compare these two images and tell me what's different", 
    images=images
)
```

## Advanced Configuration

### Image Detail Level (OpenAI)

For OpenAI models, you can specify the detail level for image analysis:

```python
response = llm.generate(
    "Describe this image in great detail", 
    image=image_url,
    image_detail="high"  # Options: "low", "high", "auto"
)
```

### Provider-Specific Formatting

If you need more control over how the image is formatted for the provider's API, you can provide a pre-formatted object:

```python
# OpenAI format
openai_format = {
    "type": "image_url",
    "image_url": {
        "url": image_url,
        "detail": "high"
    }
}

response = llm.generate("Describe this image", image=openai_format)

# Anthropic format
anthropic_format = {
    "type": "image",
    "source": {
        "type": "url",
        "url": image_url
    }
}

response = llm.generate("Describe this image", image=anthropic_format)
```

## Checking Vision Capability

Before using vision features, you should check if the selected model supports vision:

```python
llm = create_llm("anthropic", **{
    ModelParameter.MODEL: "claude-3-5-sonnet-20240620"
})

capabilities = llm.get_capabilities()
if capabilities.get(ModelCapability.VISION):
    # Use vision features
    pass
else:
    print("Vision not supported with this model")
```

## Examples

### Image Analysis

```python
llm = create_llm("openai", **{
    ModelParameter.MODEL: "gpt-4o"
})

image_url = "https://example.com/photo.jpg"
prompt = "What objects do you see in this image? List them."

response = llm.generate(prompt, image=image_url)
print(response)
```

### Image Comparison

```python
llm = create_llm("anthropic", **{
    ModelParameter.MODEL: "claude-3-5-sonnet-20240620"
})

image_url1 = "https://example.com/image1.jpg"
image_url2 = "https://example.com/image2.jpg"
prompt = "Compare these two images and describe the differences."

response = llm.generate(prompt, images=[image_url1, image_url2])
print(response)
```

### Text and Image Combined

```python
llm = create_llm("openai", **{
    ModelParameter.MODEL: "gpt-4o"
})

image_url = "https://example.com/chart.jpg"
prompt = """
This chart shows quarterly sales data. 
Please analyze the following:
1. What's the overall trend?
2. Which quarter had the highest sales?
3. What products are performing best?
"""

response = llm.generate(prompt, image=image_url)
print(response)
```

## Implementation Details

Under the hood, AbstractLLM handles the differences between provider APIs:

- For OpenAI, images are added to the `content` array of the user message
- For Anthropic, images are similarly added to the message content
- For Ollama, images are added in the format expected by the API

This abstraction allows you to use a consistent interface across all providers while still leveraging the unique capabilities of each model. 