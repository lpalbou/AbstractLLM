# Using Vision Capabilities

AbstractLLM provides a unified interface for working with vision-capable models across different providers. This guide explains how to use these capabilities to process and reason about images alongside text.

## Supported Models

Vision capabilities are available across these providers:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Anthropic**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, `claude-3.5-sonnet`, `claude-3.5-haiku`
- **Ollama**: `llava-hf/llava-1.5-7b-hf`, `llama3.2-vision`, `deepseek-janus-pro`
- **HuggingFace**: Various vision models with provider-specific handling

## Basic Usage

### Checking Vision Support

Always check if a model supports vision capabilities before attempting to use them:

```python
from abstractllm import create_llm, ModelCapability

# Create an LLM instance with a vision-capable model
llm = create_llm("openai", model="gpt-4o")

# Check if vision is supported
capabilities = llm.get_capabilities()
if capabilities.get(ModelCapability.VISION):
    # Use vision capabilities
    response = llm.generate("What's in this image?", image="path/to/image.jpg")
    print(response)
else:
    print("Vision is not supported by this model")
```

### Using Local Image Files

```python
# Using a local image file
response = llm.generate(
    prompt="Describe this image in detail.",
    image="/path/to/local/image.jpg"
)
print(response)
```

### Using Image URLs

```python
# Using an image URL
image_url = "https://example.com/image.jpg"
response = llm.generate(
    prompt="What can you tell me about this image?",
    image=image_url
)
print(response)
```

### Using Base64-Encoded Images

```python
# Using base64-encoded image data
import base64

with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

response = llm.generate(
    prompt="Analyze this image.",
    image=f"data:image/jpeg;base64,{base64_image}"
)
print(response)
```

## Multiple Images

AbstractLLM supports sending multiple images in a single request:

```python
# Using multiple images
images = [
    "/path/to/image1.jpg",
    "https://example.com/image2.jpg",
    f"data:image/jpeg;base64,{base64_image}"
]

response = llm.generate(
    prompt="Compare these images and tell me their differences.",
    images=images
)
print(response)
```

> ⚠️ **Note**: Not all providers support multiple images equally well. OpenAI and Anthropic have good support for multiple images, while Ollama and HuggingFace may have limitations depending on the model.

## Advanced Configuration

### Detail Level

For providers that support it (like OpenAI), you can control the detail level of the image processing:

```python
from abstractllm.media.factory import MediaFactory
from abstractllm.media.image import ImageInput

# Create an image input with high detail
image = MediaFactory.from_source(
    source="/path/to/image.jpg",
    media_type="image"
)
image.detail_level = "high"  # Options: "low", "medium", "high", "auto"

# Use in generation
response = llm.generate(
    prompt="Describe this image in extreme detail.",
    image=image
)
```

### Provider-Specific Options

Different providers implement vision capabilities in different ways. Here are provider-specific considerations:

#### OpenAI

```python
# OpenAI uses a specific format for image details
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    prompt="Describe what's in this image.",
    image={
        "url": "https://example.com/image.jpg",
        "detail_level": "high"
    }
)
```

#### Anthropic

```python
# Anthropic has a maximum image size limitation (100MB)
llm = create_llm("anthropic", model="claude-3-opus-20240229")
response = llm.generate(
    prompt="What's in this image?",
    image="/path/to/image.jpg"
)
```

#### Ollama

```python
# Ollama requires a vision-capable model
llm = create_llm("ollama", model="llava")
response = llm.generate(
    prompt="Describe this image.",
    image="/path/to/image.jpg"
)
```

#### HuggingFace

```python
# HuggingFace requires specific model support
llm = create_llm("huggingface", model="llava-hf/llava-1.5-7b-hf")
response = llm.generate(
    prompt="What is shown in this image?",
    image="/path/to/image.jpg"
)
```

## Implementation Details

### Image Processing

AbstractLLM handles image processing through the `MediaFactory` and `ImageInput` classes:

```python
# From abstractllm/media/factory.py
class MediaFactory:
    @classmethod
    def from_source(cls, source: Union[str, Path, Dict[str, Any]], media_type: Optional[str] = None) -> MediaInput:
        """
        Create a media input object from a source.
        
        Args:
            source: File path, URL, base64 string, or provider-specific dict
            media_type: Explicit media type (optional, auto-detected if not provided)
            
        Returns:
            Appropriate MediaInput instance
        """
```

The `ImageInput` class handles:
- Loading images from files, URLs, and base64 strings
- Converting images to provider-specific formats
- Caching for efficiency
- Metadata extraction
- Error handling

### Provider Adapters

Each provider has specific logic for handling image inputs:

```python
# From abstractllm/media/image.py
def to_provider_format(self, provider: str) -> Any:
    """
    Convert the image to a format suitable for the specified provider.
    
    Args:
        provider: The provider name ('openai', 'anthropic', 'ollama', 'huggingface')
        
    Returns:
        Provider-specific format for the image
    """
    # Convert to provider-specific format
    if provider == "openai":
        format_result = self._format_for_openai()
    elif provider == "anthropic":
        format_result = self._format_for_anthropic()
    elif provider == "ollama":
        format_result = self._format_for_ollama()
    elif provider == "huggingface":
        format_result = self._format_for_huggingface()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
```

## Best Practices

1. **Always Check Capabilities**
   ```python
   if llm.get_capabilities().get(ModelCapability.VISION):
       # Use vision capabilities
   ```

2. **Optimize Image Size**
   - Large images consume more tokens, which can be expensive
   - Resize images to appropriate dimensions before sending
   - Use lower detail levels for initial analysis

3. **Handle Errors Gracefully**
   ```python
   from abstractllm.exceptions import ImageProcessingError
   
   try:
       response = llm.generate(prompt="Describe this.", image=image_path)
   except ImageProcessingError as e:
       print(f"Image processing error: {e}")
   ```

4. **Consider Token Usage**
   - Vision models typically charge per image and resolution
   - Monitor your token usage when using vision capabilities
   - Use the appropriate detail level for your task

5. **Test Provider Compatibility**
   - Different providers have different strengths
   - Test your specific use case with different providers
   - Consider fallback options for vision tasks

## Common Use Cases

### Image Analysis

```python
prompt = "Analyze this image and tell me what objects you see."
response = llm.generate(prompt=prompt, image="/path/to/image.jpg")
```

### Multi-Image Comparison

```python
prompt = "Compare these two product images and tell me their differences."
response = llm.generate(prompt=prompt, images=["/path/to/product1.jpg", "/path/to/product2.jpg"])
```

### Chart Interpretation

```python
prompt = "Interpret this chart and explain what trends it shows."
response = llm.generate(prompt=prompt, image="/path/to/chart.png")
```

### Visual Question Answering

```python
prompt = "How many people are in this image? What are they doing?"
response = llm.generate(prompt=prompt, image="/path/to/image.jpg")
```

## Limitations

1. **Provider Inconsistencies**
   - Different providers have different vision capabilities
   - Results may vary between providers
   - HuggingFace has limited support compared to commercial providers

2. **Token Usage**
   - Vision capabilities consume significantly more tokens than text-only
   - High-resolution images or high detail levels increase costs
   - Consider optimizing image size and detail level

3. **Performance**
   - Vision processing adds latency
   - Multiple images can slow down response times
   - Local models (Ollama, HuggingFace) may have memory constraints

4. **Security Considerations**
   - Be cautious about what images you share with external providers
   - Consider privacy implications of image processing
   - Use local providers (Ollama, HuggingFace) for sensitive content

## Conclusion

AbstractLLM's vision capabilities provide a flexible and consistent interface for working with vision-enabled LLMs. By abstracting away provider-specific details, you can focus on your application logic while leveraging the latest multimodal AI capabilities across different providers. 