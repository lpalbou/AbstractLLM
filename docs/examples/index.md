# AbstractLLM Examples

This section provides practical code examples that demonstrate how to use AbstractLLM for various tasks and use cases. These examples are designed to be copy-paste ready and illustrate best practices.

## Basic Examples

### Simple Text Generation

```python
from abstractllm import create_llm

# Create a provider
llm = create_llm("openai", model="gpt-3.5-turbo")

# Generate text
response = llm.generate("Explain what a large language model is in simple terms.")
print(response)
```

### Using Different Providers

```python
from abstractllm import create_llm

# OpenAI provider
openai_llm = create_llm("openai", model="gpt-4")
openai_response = openai_llm.generate("What are the benefits of renewable energy?")

# Anthropic provider
anthropic_llm = create_llm("anthropic", model="claude-3-opus-20240229")
anthropic_response = anthropic_llm.generate("What are the benefits of renewable energy?")

# Compare responses
print("OpenAI response:", openai_response[:100], "...")
print("Anthropic response:", anthropic_response[:100], "...")
```

## Session Management

### Multi-Turn Conversation

```python
from abstractllm import create_llm
from abstractllm.session import Session

# Create a provider
llm = create_llm("openai", model="gpt-4")

# Create a session with a system prompt
session = Session(
    provider=llm,
    system_prompt="You are a helpful AI assistant that specializes in science topics."
)

# First user message
response1 = session.generate("What is quantum entanglement?")
print("Response 1:", response1)

# Follow-up question
response2 = session.generate("How is it used in quantum computing?")
print("Response 2:", response2)

# Another follow-up
response3 = session.generate("Can you explain it using a simple analogy?")
print("Response 3:", response3)
```

## Working with Tools

### Basic Tool Usage

```python
from abstractllm import create_llm, ToolDefinition
from abstractllm.session import Session
import json

# Define a weather tool
def get_weather(location: str):
    """Get the current weather for a location."""
    # In a real app, you would call a weather API
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny",
        "humidity": 45
    }

# Create a provider with a model that supports function calling
llm = create_llm("openai", model="gpt-4")

# Create a session with the tool
session = Session(
    provider=llm,
    system_prompt="You are a helpful assistant that can provide weather information.",
    tools=[get_weather]
)

# Generate a response that will use the tool
response = session.generate_with_tools("What's the weather like in San Francisco?")
print(response)
```

### Multiple Tools

```python
from abstractllm import create_llm
from abstractllm.session import Session
import datetime

# Define multiple tools
def get_weather(location: str):
    """Get the current weather for a location."""
    # In a real app, you would call a weather API
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny",
        "humidity": 45
    }

def get_time(timezone: str = "UTC"):
    """Get the current time in the specified timezone."""
    # In a real app, you would use proper timezone handling
    return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')} in {timezone}."

def calculator(operation: str, a: float, b: float):
    """Perform a basic calculation."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    else:
        return f"Unsupported operation: {operation}"

# Create a session with multiple tools
llm = create_llm("openai", model="gpt-4")
session = Session(
    provider=llm,
    system_prompt="You are a helpful assistant that can provide weather, time, and calculation services.",
    tools=[get_weather, get_time, calculator]
)

# Generate a response that might use any of the tools
response = session.generate_with_tools(
    "I'm planning a trip to Miami. What's the weather like there? " +
    "Also, what time is it in the Eastern timezone? Lastly, what's 18% of $85?"
)
print(response)
```

## Streaming Examples

### Basic Streaming

```python
from abstractllm import create_llm

# Create a provider
llm = create_llm("openai", model="gpt-4")

# Stream the response
prompt = "Write a short story about a robot discovering emotions."
print(f"Prompt: {prompt}\n")
print("Response: ", end="", flush=True)

for chunk in llm.generate(prompt, stream=True):
    print(chunk, end="", flush=True)
print("\n")
```

### Async Streaming

```python
import asyncio
from abstractllm import create_llm

async def stream_example():
    # Create a provider
    llm = create_llm("openai", model="gpt-4")
    
    # Stream the response
    prompt = "Explain the concept of artificial intelligence."
    print(f"Prompt: {prompt}\n")
    print("Response: ", end="", flush=True)
    
    async for chunk in llm.generate_async(prompt, stream=True):
        print(chunk, end="", flush=True)
    print("\n")

# Run the async function
asyncio.run(stream_example())
```

## Vision Examples

### Image Analysis

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create a provider with a vision-capable model
llm = create_llm("openai", model="gpt-4o")

# Analyze an image from a URL
image_url = "https://example.com/sample-image.jpg"
response = llm.generate(
    "What's in this image? Describe it in detail.",
    media=[ImageInput.from_url(image_url)]
)
print(response)

# Analyze a local image
local_image = "/path/to/local/image.jpg"
response = llm.generate(
    "What's in this image? Describe it in detail.",
    media=[ImageInput.from_file(local_image)]
)
print(response)
```

### Multiple Images

```python
from abstractllm import create_llm
from abstractllm.media import ImageInput

# Create a provider with a vision-capable model
llm = create_llm("anthropic", model="claude-3-opus-20240229")

# Load multiple images
image1 = ImageInput.from_file("/path/to/image1.jpg")
image2 = ImageInput.from_file("/path/to/image2.jpg")

# Compare images
response = llm.generate(
    "Compare these two images. What are the similarities and differences?",
    media=[image1, image2]
)
print(response)
```

## Advanced Examples

### Provider Interchangeability

```python
from abstractllm import create_llm
from abstractllm.chains import FallbackChain
from abstractllm.exceptions import AbstractLLMError

# Create multiple providers
primary = create_llm("openai", model="gpt-4")
fallback1 = create_llm("anthropic", model="claude-3-opus-20240229")
fallback2 = create_llm("ollama", model="llama3")

# Create a fallback chain
chain = FallbackChain(
    providers=[primary, fallback1, fallback2],
    error_types=[AbstractLLMError]
)

# Generate with fallback capability
try:
    response = chain.generate("Explain the theory of relativity simply.")
    print(f"Response from: {chain.last_successful_provider.__class__.__name__}")
    print(response)
except AbstractLLMError as e:
    print(f"All providers failed: {e}")
```

### Custom Configuration

```python
from abstractllm import create_llm
from abstractllm.configuration import ConfigurationManager

# Create a custom configuration
config = {
    "model": "gpt-4",
    "temperature": 0.8,
    "max_tokens": 500,
    "system_prompt": "You are a creative writing assistant."
}

# Create a provider with custom configuration
llm = create_llm("openai", **config)

# Override specific parameters for a request
response = llm.generate(
    "Write a short sci-fi story.",
    temperature=0.9,  # Override just for this request
    max_tokens=1000   # Override just for this request
)
print(response)
```

## Complete Applications

### Chatbot Example

```python
from abstractllm import create_llm
from abstractllm.session import Session

def simple_chatbot():
    """A simple command-line chatbot using AbstractLLM."""
    # Create a provider
    llm = create_llm("openai", model="gpt-3.5-turbo")
    
    # Create a session
    session = Session(
        provider=llm,
        system_prompt="You are a helpful assistant named AbstractBot. You provide concise, helpful responses."
    )
    
    print("Welcome to AbstractBot! Type 'exit' to quit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("AbstractBot: Goodbye! Have a great day!")
            break
        
        # Generate response with streaming
        print("\nAbstractBot: ", end="", flush=True)
        for chunk in session.generate(user_input, stream=True):
            print(chunk, end="", flush=True)
        print()

# Run the chatbot
if __name__ == "__main__":
    simple_chatbot()
```

### Document Analysis Tool

```python
from abstractllm import create_llm
from abstractllm.session import Session
import sys
import os

def analyze_document(file_path):
    """Analyze a text document using AbstractLLM."""
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Read the document
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Create a provider
    llm = create_llm("anthropic", model="claude-3-opus-20240229")
    
    # Create a session
    session = Session(
        provider=llm,
        system_prompt="You are a document analysis assistant. You provide comprehensive analysis of documents including key themes, main points, and summary."
    )
    
    # Generate analysis
    print(f"Analyzing document: {os.path.basename(file_path)}")
    print("This may take a moment...")
    
    response = session.generate(
        f"Please analyze this document thoroughly:\n\n{document_text}"
    )
    
    print("\n--- Document Analysis ---\n")
    print(response)
    print("\n------------------------\n")

# Run the document analyzer if called directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_analyzer.py <file_path>")
    else:
        analyze_document(sys.argv[1])
```

## Further Resources

For more examples and detailed documentation, see:

- [Basic Generation Guide](../user-guide/basic-generation.md)
- [Sessions Guide](../user-guide/sessions.md)
- [Tool Usage Guide](../user-guide/tools.md)
- [Vision Guide](../user-guide/vision.md)
- [Provider Interchangeability Guide](../user-guide/interchangeability.md)

You can also find the source code for all examples in the [GitHub repository](https://github.com/lpalbou/abstractllm/tree/main/examples). 