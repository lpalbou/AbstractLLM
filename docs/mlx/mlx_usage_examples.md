# MLX Provider Usage Examples

This document provides practical examples of using the MLX provider within AbstractLLM. These examples demonstrate how to leverage MLX-accelerated models on Apple Silicon devices.

## Prerequisites

Before using the MLX provider, ensure you have:

1. A Mac with Apple Silicon (M1/M2/M3 series)
2. Python 3.8 or newer
3. AbstractLLM installed with MLX dependencies:

```bash
# Install with MLX support
pip install "abstractllm[mlx]"

# Or for development
pip install -e ".[mlx,dev]"
```

## Basic Text Generation

### Simple Generation

The simplest way to use the MLX provider:

```python
from abstractllm import create_llm

# Create an LLM with MLX provider
llm = create_llm("mlx")  # Uses default model

# Generate text
response = llm.generate("Explain quantum computing in simple terms")
print(response.text)
```

### Customizing Parameters

You can customize various generation parameters:

```python
from abstractllm import create_llm, ModelParameter

# Create an LLM with custom parameters
llm = create_llm(
    "mlx",
    model="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
    temperature=0.8,
    max_tokens=2048,
    top_p=0.95
)

# Or set parameters using ModelParameter enum
llm = create_llm(
    "mlx", 
    **{
        ModelParameter.MODEL: "mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
        ModelParameter.TEMPERATURE: 0.8,
        ModelParameter.MAX_TOKENS: 2048,
        ModelParameter.TOP_P: 0.95
    }
)

# Generate text
response = llm.generate("Write a short story about artificial intelligence")
print(response.text)
```

### Using System Prompts

System prompts help guide the model's behavior:

```python
from abstractllm import create_llm

llm = create_llm("mlx")

system_prompt = "You are an expert programmer who explains code in simple terms."
user_prompt = "Explain how recursion works in Python with an example."

response = llm.generate(
    prompt=user_prompt,
    system_prompt=system_prompt
)

print(response.text)
```

## Streaming Responses

For real-time responses, use streaming mode:

```python
from abstractllm import create_llm

llm = create_llm("mlx")

# Generate with streaming
for chunk in llm.generate(
    "Explain the history of machine learning", 
    stream=True
):
    # Process each chunk as it arrives
    print(chunk.text, end="", flush=True)
```

## Async Support

For non-blocking operations in async environments:

```python
import asyncio
from abstractllm import create_llm

async def generate_async():
    llm = create_llm("mlx")
    
    # Simple async generation
    response = await llm.generate_async("What is the meaning of life?")
    print(response.text)
    
    # Async streaming
    async for chunk in await llm.generate_async(
        "Describe the solar system", 
        stream=True
    ):
        print(chunk.text, end="", flush=True)

# Run in an async environment
asyncio.run(generate_async())
```

## Working with Different Models

MLX supports various model architectures:

```python
from abstractllm import create_llm

# Small model (good for testing or less demanding applications)
llm_small = create_llm("mlx", model="mlx-community/phi-2")

# Medium-sized model
llm_medium = create_llm("mlx", model="mlx-community/Mistral-7B-Instruct-v0.2")

# Larger model
llm_large = create_llm("mlx", model="mlx-community/Mixtral-8x7B-Instruct-v0.1")

# Generate with different models
response_small = llm_small.generate("What is the capital of France?")
response_medium = llm_medium.generate("Explain the theory of relativity")
response_large = llm_large.generate("Write a sonnet about artificial intelligence")
```

## Vision-Capable Models (when available)

For models that support vision:

```python
from abstractllm import create_llm, ModelCapability
from pathlib import Path

# Create an LLM with a vision-capable model
llm = create_llm("mlx", model="mlx-community/llava-1.5-7b-mlx")

# Check if vision is supported
if llm.get_capabilities().get(ModelCapability.VISION, False):
    # Process an image
    image_path = Path("./example_image.jpg")
    
    # Generate description of the image
    response = llm.generate(
        "What's in this image?",
        files=[image_path]
    )
    
    print(response.text)
else:
    print("This model doesn't support vision capabilities")
```

## Integration with Other Systems

### Chat Interface Example

Creating a simple chat interface:

```python
from abstractllm import create_llm

def simple_chat():
    llm = create_llm("mlx")
    
    # Initialize chat history
    chat_history = []
    system_prompt = "You are a helpful AI assistant."
    
    print("Chat with the MLX-powered AI (type 'exit' to quit):")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        # Update chat history with user message
        chat_history.append(f"User: {user_input}")
        
        # Format full prompt with chat history context
        full_prompt = "\n".join(chat_history[-5:])  # Keep last 5 messages for context
        
        # Generate response
        response = llm.generate(
            full_prompt, 
            system_prompt=system_prompt
        )
        
        # Display and save AI response
        ai_message = response.text.strip()
        print(f"\nAI: {ai_message}")
        chat_history.append(f"AI: {ai_message}")

# Run the chat interface
simple_chat()
```

### Web Application Integration

Example of integrating with a Flask web application:

```python
from flask import Flask, request, jsonify
from abstractllm import create_llm

app = Flask(__name__)

# Initialize the LLM
llm = create_llm("mlx")

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt', '')
    system_prompt = data.get('system_prompt', '')
    
    # Generate response
    response = llm.generate(
        prompt=prompt,
        system_prompt=system_prompt
    )
    
    return jsonify({
        'text': response.text,
        'tokens': response.total_tokens
    })

@app.route('/stream', methods=['POST'])
def stream_text():
    data = request.json
    prompt = data.get('prompt', '')
    
    # Stream response (in a real app, you'd use proper streaming response)
    def generate():
        for chunk in llm.generate(prompt, stream=True):
            yield f"data: {chunk.text}\n\n"
    
    return app.response_class(
        generate(),
        mimetype='text/event-stream'
    )

if __name__ == '__main__':
    app.run(debug=True)
```

## Performance Tips

### Model Caching

The MLX provider implements model caching, but you can explicitly manage it:

```python
from abstractllm import create_llm

# Create an LLM
llm = create_llm("mlx")

# The first generation will load the model
response1 = llm.generate("What is MLX?")

# Subsequent generations reuse the loaded model
response2 = llm.generate("How does it work?")

# If you're done with the model and want to free memory
if hasattr(llm, 'clear_model_cache'):
    llm.clear_model_cache()
```

### Using Quantized Models

Quantized models offer better performance with minimal quality loss:

```python
from abstractllm import create_llm

# Use a quantized model (4-bit)
llm = create_llm(
    "mlx", 
    model="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX",
    quantize=True  # This is the default
)

response = llm.generate("Explain the benefits of quantization in machine learning")
```

## Capability Detection

Check what capabilities are supported by your model:

```python
from abstractllm import create_llm, ModelCapability

llm = create_llm("mlx")

# Get all capabilities
capabilities = llm.get_capabilities()

# Check specific capabilities
supports_streaming = capabilities.get(ModelCapability.STREAMING, False)
supports_vision = capabilities.get(ModelCapability.VISION, False)
max_tokens = capabilities.get(ModelCapability.MAX_TOKENS, 2048)

print(f"Streaming support: {supports_streaming}")
print(f"Vision support: {supports_vision}")
print(f"Maximum tokens: {max_tokens}")
```

## Multi-Modal Input Processing

Handle different types of inputs:

```python
from abstractllm import create_llm
from pathlib import Path

llm = create_llm("mlx", model="mlx-community/llava-1.5-7b-mlx")

# Process both text file and image
text_file = Path("./document.txt")
image_file = Path("./diagram.jpg")

# Generate response considering both inputs
response = llm.generate(
    "Explain the relationship between the text and the diagram",
    files=[text_file, image_file]
)

print(response.text)
```

## Conclusion

These examples demonstrate the versatility of the MLX provider in AbstractLLM. By leveraging Apple's MLX framework, you can run powerful language models efficiently on Apple Silicon hardware while maintaining the familiar AbstractLLM interface.

For more advanced usage, refer to the AbstractLLM documentation and the MLX provider implementation guide. 