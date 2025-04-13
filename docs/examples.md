# AbstractLLM Examples

This document provides examples of using AbstractLLM with various providers and pipelines.

## Basic Usage

### Text Generation

```python
from abstractllm import create_llm

# Create an OpenAI provider
llm = create_llm("openai", model="gpt-4")

# Generate text
response = llm.generate("Tell me a story about a brave cat.")
print(response)

# Stream responses
for chunk in llm.generate("Tell me a story about a brave cat.", stream=True):
    print(chunk, end="")
```

### Async Generation

```python
import asyncio
from abstractllm import create_llm

async def main():
    llm = create_llm("openai", model="gpt-4")
    
    # Async generation
    response = await llm.generate_async("Tell me a story about a brave cat.")
    print(response)
    
    # Async streaming
    async for chunk in llm.generate_async("Tell me a story about a brave cat.", stream=True):
        print(chunk, end="")

asyncio.run(main())
```

## HuggingFace Provider

### Text Generation Pipeline

```python
from abstractllm import create_llm

# Create a HuggingFace provider with Phi-2
llm = create_llm("huggingface", model="microsoft/phi-2")

# Generate text
response = llm.generate("Explain quantum computing in simple terms.")
print(response)
```

### Visual Question Answering Pipeline

```python
from abstractllm import create_llm
from PIL import Image
from abstractllm.media import ImageInput, TextInput

# Create a HuggingFace provider with ViLT
llm = create_llm("huggingface", model="dandelin/vilt-b32-finetuned-vqa")

# Basic usage
image = Image.open("path/to/image.jpg")
response = llm.generate("What color is the cat in the image?", image=image)
print(response)

# Multiple choice questions
response = llm.generate(
    "What animal is in the image?",
    image=image,
    answer_candidates=["cat", "dog", "bird"]
)
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']:.2f}")

# Advanced usage with media inputs
image_input = ImageInput("path/to/image.jpg", detail_level="high")
question_input = TextInput("Describe the scene in detail.")

response = llm.generate(
    inputs=[image_input, question_input],
    generation_config={
        "max_answer_length": 100,
        "num_beams": 4,
        "return_logits": True
    }
)

print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']:.2f}")
if 'logits' in response:
    print("Raw model logits available for analysis")

# Error handling
try:
    response = llm.generate(
        "Invalid question",
        image=None  # Missing required image
    )
except InvalidInputError as e:
    print(f"Input error: {e}")
except GenerationError as e:
    print(f"Generation failed: {e}")
```

### Document Question Answering Pipeline

```python
from abstractllm import create_llm
from PIL import Image

# Create a HuggingFace provider with LayoutLM
llm = create_llm("huggingface", model="microsoft/layoutlmv3-base")

# Load a document (PDF or image)
document = Image.open("path/to/invoice.pdf")

# Ask questions about the document
response = llm.generate("What is the total amount on this invoice?", document=document)
print(response)

# Extract specific information
response = llm.generate("Find the invoice date.", document=document)
print(response)
```

## Media Handling

### Image Inputs

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create image input from various sources
image1 = ImageInput.from_file("path/to/image.jpg")
image2 = ImageInput.from_url("https://example.com/image.jpg")
image3 = ImageInput.from_base64("base64_encoded_string")

# Use with provider
llm = create_llm("openai", model="gpt-4-vision-preview")
response = llm.generate("Describe this image.", image=image1)
print(response)
```

### Document Inputs

```python
from abstractllm import create_llm
from abstractllm.media import DocumentInput

# Create document input
doc = DocumentInput.from_file("path/to/document.pdf")

# Use with provider
llm = create_llm("huggingface", model="microsoft/layoutlmv3-base")
response = llm.generate("What is the main topic of this document?", document=doc)
print(response)
```

## Advanced Usage

### Custom Configuration

```python
from abstractllm import create_llm, ModelParameter

# Create provider with detailed configuration
llm = create_llm("huggingface",
    model="microsoft/phi-2",
    temperature=0.7,
    max_tokens=1000,
    device="cuda",
    quantization="8bit",
    torch_dtype="float16"
)

# Override parameters per request
response = llm.generate(
    "Write a story.",
    temperature=0.9,
    max_tokens=2000
)
```

### Error Handling

```python
from abstractllm import create_llm
from abstractllm.exceptions import (
    AbstractLLMError,
    ModelNotFoundError,
    MediaProcessingError
)

try:
    llm = create_llm("huggingface", model="invalid-model")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")

try:
    llm = create_llm("huggingface", model="microsoft/phi-2")
    response = llm.generate("Question", image="invalid.jpg")
except MediaProcessingError as e:
    print(f"Media processing failed: {e}")
except AbstractLLMError as e:
    print(f"Other error: {e}")
```

### Resource Management

```python
from abstractllm import create_llm

# Create provider with resource constraints
llm = create_llm("huggingface",
    model="microsoft/phi-2",
    device="cuda",
    max_memory={"cuda:0": "4GiB"},
    torch_compile=True,
    compile_mode="reduce-overhead"
)

try:
    response = llm.generate("Hello")
finally:
    # Clean up resources
    llm.cleanup()
```

## Testing Examples

### Basic Test Setup

```python
import unittest
from abstractllm import create_llm

class TestLLM(unittest.TestCase):
    def setUp(self):
        self.llm = create_llm("huggingface", model="microsoft/phi-2")
    
    def test_generation(self):
        response = self.llm.generate("Hello")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def tearDown(self):
        self.llm.cleanup()
```