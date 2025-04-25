# Multi-Modal Content

This guide explains how to work with multiple content types in AbstractLLM, particularly focusing on advanced techniques for handling images, documents, and other media in your LLM applications.

## Understanding Multi-Modal Capabilities

AbstractLLM provides a unified interface for working with various content types across different providers:

```python
from abstractllm import create_llm
from abstractllm.media import MediaFactory

# Create a vision-capable model
llm = create_llm("openai", model="gpt-4o")

# Process text and image together
response = llm.generate(
    prompt="Describe what you see in this image.",
    image="/path/to/image.jpg"
)
print(response)
```

## Advanced Image Handling

### Multiple Images

Working with multiple images in a single request:

```python
from abstractllm import create_llm
from abstractllm.media import MediaFactory

# Create a vision-capable model
llm = create_llm("openai", model="gpt-4o")

# Process multiple images
images = [
    "/path/to/image1.jpg",
    "https://example.com/image2.jpg",
    MediaFactory.from_source("/path/to/image3.png")
]

response = llm.generate(
    prompt="Compare these images and tell me the key differences.",
    images=images
)
print(response)
```

### Image Preprocessing

Optimize images before sending to the model:

```python
from abstractllm import create_llm
from abstractllm.media import MediaFactory, ImageInput
from PIL import Image
import io

def preprocess_image(image_path, max_width=800, max_height=600):
    """Preprocess image for optimal token usage."""
    # Create image input
    img_input = MediaFactory.from_source(image_path, media_type="image")
    
    # Resize to optimize token usage
    img_input.resize(max_width=max_width, max_height=max_height)
    
    # Set detail level (for providers that support it)
    img_input.detail_level = "low"  # Options: "low", "medium", "high", "auto"
    
    return img_input

# Usage
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Process optimized image
processed_image = preprocess_image("/path/to/large_image.jpg")
response = llm.generate(
    prompt="What does this image show?",
    image=processed_image
)
```

### Custom Image Parameters

Control image processing with provider-specific parameters:

```python
from abstractllm import create_llm

# OpenAI with custom image parameters
llm = create_llm("openai", model="gpt-4o")
response = llm.generate(
    prompt="Describe this image in detail.",
    image={
        "url": "https://example.com/image.jpg",
        "detail": "high"  # OpenAI-specific parameter
    }
)

# Anthropic with custom parameters
llm = create_llm("anthropic", model="claude-3-opus-20240229")
response = llm.generate(
    prompt="What's in this image?",
    image={
        "path": "/path/to/image.jpg",
        "resize_factor": 0.75  # Anthropic-specific parameter
    }
)
```

## Document Processing

Handling documents and extracting information:

```python
from abstractllm import create_llm
from abstractllm.media import MediaFactory, TextInput
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF document."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def process_document(document_path, question):
    """Process a document and answer questions about it."""
    # Extract text from document
    if document_path.endswith(".pdf"):
        document_text = extract_text_from_pdf(document_path)
    else:
        # For text files
        with open(document_path, "r") as f:
            document_text = f.read()
    
    # Create LLM
    llm = create_llm("anthropic", model="claude-3-opus-20240229")
    
    # Build prompt with document content
    prompt = f"""
    Here is the content of the document:
    
    {document_text}
    
    Based on the above document, please answer the following question:
    
    {question}
    """
    
    # Generate response
    return llm.generate(
        prompt=prompt,
        system_prompt="You are an expert at analyzing documents. Extract relevant information accurately."
    )

# Usage
answer = process_document("report.pdf", "What are the main findings of the report?")
print(answer)
```

## Mixed Media Processing

Combining different media types in a single request:

```python
from abstractllm import create_llm
from abstractllm.media import MediaFactory

def analyze_report_with_charts(report_text, chart_image_path):
    """Analyze a report text along with chart images."""
    # Create a vision-capable model
    llm = create_llm("anthropic", model="claude-3-opus-20240229")
    
    # Process document text and chart together
    prompt = f"""
    Here is a report text:
    
    {report_text}
    
    There is also a chart image attached. Please analyze both the report and the chart,
    and explain how the chart relates to the information in the text.
    """
    
    response = llm.generate(
        prompt=prompt,
        image=chart_image_path
    )
    
    return response

# Example usage
with open("quarterly_report.txt", "r") as f:
    report_text = f.read()

analysis = analyze_report_with_charts(report_text, "sales_chart.png")
print(analysis)
```

## Provider-Specific Optimizations

Each provider handles multi-modal content differently. Here are provider-specific optimizations:

### OpenAI

```python
from abstractllm import create_llm
from abstractllm.media import MediaFactory

# OpenAI handles multiple images efficiently
llm = create_llm("openai", model="gpt-4o")

# Optimize for token usage with low detail for initial analysis
image_low = MediaFactory.from_source("image.jpg")
image_low.detail_level = "low"

initial_analysis = llm.generate(
    prompt="Give a brief overview of what you see in this image.",
    image=image_low
)

# Use high detail only for specific parts that need it
if "complex diagram" in initial_analysis or "fine text" in initial_analysis:
    image_high = MediaFactory.from_source("image.jpg")
    image_high.detail_level = "high"
    
    detailed_analysis = llm.generate(
        prompt="Now, examine the fine details of this image and describe any text or small elements you see.",
        image=image_high
    )
```

### Anthropic

```python
from abstractllm import create_llm
import base64

# Anthropic works well with in-context examples
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Optimize content structure for Anthropic
def analyze_document_series(document_paths, image_paths):
    """Analyze a series of documents and related images with Anthropic."""
    # Build a prompt that interleaves document content and images
    prompt = "I'm going to show you several documents and related images. Please analyze them together.\n\n"
    
    for i, (doc_path, img_path) in enumerate(zip(document_paths, image_paths)):
        # Add document content
        with open(doc_path, "r") as f:
            doc_content = f.read()
        
        prompt += f"Document {i+1}:\n{doc_content}\n\n"
        prompt += f"Image {i+1}: [Image is attached separately]\n\n"
    
    prompt += "Please analyze how the images relate to the documents and identify any inconsistencies or notable insights."
    
    # Process with all images
    response = llm.generate(
        prompt=prompt,
        images=image_paths
    )
    
    return response
```

## Advanced Techniques

### Vision Chain of Thought

Implement chain-of-thought for complex image analysis:

```python
from abstractllm import create_llm
from abstractllm.session import Session

def vision_chain_of_thought(image_path, question):
    """Apply chain-of-thought reasoning to image analysis."""
    llm = create_llm("openai", model="gpt-4o")
    
    # Create session for multi-turn interaction
    session = Session(
        provider=llm,
        system_prompt="You are an expert at analyzing images through step-by-step reasoning."
    )
    
    # Step 1: General description
    session.add_message("user", "What is shown in this image?", image=image_path)
    description = session.generate()
    
    # Step 2: Detailed elements analysis
    session.add_message("user", "Identify and list all the key elements in the image.")
    elements = session.generate()
    
    # Step 3: Relationships between elements
    session.add_message("user", "How do these elements relate to each other?")
    relationships = session.generate()
    
    # Step 4: Answer the specific question
    session.add_message("user", f"Based on your analysis, please answer this question: {question}")
    answer = session.generate()
    
    return {
        "description": description,
        "elements": elements,
        "relationships": relationships,
        "answer": answer
    }

# Usage
result = vision_chain_of_thought("complex_scene.jpg", "What time of day is it in this image and how can you tell?")
for key, value in result.items():
    print(f"\n{key.upper()}:")
    print(value)
```

### Multi-Modal Tool Use

Combining vision with tool use:

```python
from abstractllm import create_llm
from abstractllm.session import Session
import cv2
import numpy as np
import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    return pytesseract.image_to_string(Image.open(image_path))

def count_objects(image_path, object_type):
    """Count objects of a specific type in an image (simplified example)."""
    # Load pre-trained object detection model
    # This is just a placeholder - would use an actual model in practice
    return {"count": 5, "object_type": object_type}

# Register tools
tools = [extract_text_from_image, count_objects]

# Create session with vision and tools
llm = create_llm("openai", model="gpt-4o")
session = Session(
    provider=llm,
    system_prompt="You can analyze images and use tools to process them further.",
    tools=tools
)

# Process image with tools
response = session.generate_with_tools(
    "Analyze this image. If there's text, extract it. Also count the number of people visible.",
    image="meeting_photo.jpg"
)
print(response)
```

### Image Series Analysis

Analyzing a sequence of images:

```python
from abstractllm import create_llm
import os

def analyze_image_sequence(image_dir, query):
    """Analyze a sequence of images in a directory."""
    # List image files
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort by filename (assuming filenames indicate sequence)
    image_files.sort()
    
    # Create LLM
    llm = create_llm("anthropic", model="claude-3-opus-20240229")
    
    # Build prompt with context about sequence
    prompt = f"""
    I'm going to show you a sequence of {len(image_files)} images. 
    These images are ordered and represent a series of events or changes.
    
    Please analyze the sequence, paying attention to changes or progression over time.
    
    After analyzing the sequence, answer this question: {query}
    """
    
    # Generate response with all images
    response = llm.generate(
        prompt=prompt,
        images=image_files
    )
    
    return response

# Usage
analysis = analyze_image_sequence("timelapse_images/", 
                                "How does the vegetation change throughout the seasons shown in these images?")
print(analysis)
```

## Best Practices

1. **Optimize Image Size**: Resize images to reduce token usage without losing important details
2. **Use Detail Levels Appropriately**: Start with low detail and increase only when needed
3. **Combine Media Types Strategically**: Structure prompts to make clear connections between text and images
4. **Cache Large Media**: Implement caching for frequently used images to reduce processing time
5. **Validate Media Inputs**: Add validation to ensure media inputs are in supported formats
6. **Implement Fallbacks**: Have text-only fallbacks for when vision models are unavailable

## Limitations and Considerations

1. **Token Consumption**: Vision requests use significantly more tokens than text-only requests
2. **Cost Implications**: Multi-modal requests are typically more expensive
3. **Provider Variability**: Different providers handle multi-modal content differently
4. **Performance Impact**: Processing multiple or large images can increase latency
5. **Content Filtering**: Be aware that different providers have different content policies for images

## Next Steps

- [Vision Capabilities](../user-guide/vision.md): User guide for working with basic vision capabilities
- [Performance Optimization](performance.md): Optimize multi-modal applications for better performance
- [Provider-Specific Features](../providers/index.md): Learn about provider-specific multi-modal capabilities 