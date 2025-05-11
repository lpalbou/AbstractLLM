# Task 13: Add Simple Usage Examples Documentation

## Description
Create a simple examples file with code snippets demonstrating how to use the MLX provider.

## Requirements
1. Create a `examples/mlx_examples.py` file
2. Include examples for basic text generation, streaming, and system prompts
3. Add comments explaining key concepts
4. Keep examples simple and focused on one concept each

## Implementation Details

Create a file at `examples/mlx_examples.py`:

```python
"""
MLX Provider Usage Examples

This file demonstrates how to use AbstractLLM with the MLX provider
on Apple Silicon devices.

Requirements:
- macOS with Apple Silicon (M1/M2/M3)
- AbstractLLM installed with MLX dependencies: pip install ".[mlx]"
"""

from abstractllm import create_llm, ModelParameter

def example_basic_generation():
    """
    Basic text generation with MLX.
    """
    print("\n=== Basic Generation ===")
    
    # Create an LLM with MLX provider (uses default model)
    llm = create_llm("mlx")
    
    # Generate text
    response = llm.generate("Explain what MLX is in simple terms.")
    
    # Display the response
    print(f"Response: {response.text}")
    print(f"Token stats: {response.prompt_tokens} prompt tokens, " 
          f"{response.completion_tokens} completion tokens")


def example_custom_model():
    """
    Using a specific MLX model.
    """
    print("\n=== Custom Model ===")
    
    # Create an LLM with a specific model
    llm = create_llm(
        "mlx",
        model="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX"
    )
    
    # Generate text
    response = llm.generate("What are the benefits of quantized models?")
    
    # Display the response
    print(f"Response: {response.text}")


def example_with_system_prompt():
    """
    Using a system prompt to guide the model's behavior.
    """
    print("\n=== System Prompt ===")
    
    # Create an LLM
    llm = create_llm("mlx")
    
    # Define a system prompt
    system_prompt = "You are an expert programmer who explains code in simple terms."
    
    # Generate text with system prompt
    response = llm.generate(
        "Explain how recursion works in Python.",
        system_prompt=system_prompt
    )
    
    # Display the response
    print(f"Response: {response.text}")


def example_streaming():
    """
    Streaming generation for real-time responses.
    """
    print("\n=== Streaming Generation ===")
    
    # Create an LLM
    llm = create_llm("mlx")
    
    # Generate with streaming
    print("Streaming response:")
    for chunk in llm.generate(
        "Explain the importance of machine learning.", 
        stream=True
    ):
        # Print each chunk as it arrives (without newlines)
        print(chunk.text, end="", flush=True)
    
    print("\n")  # Add newline at the end


def example_generation_parameters():
    """
    Customizing generation parameters.
    """
    print("\n=== Generation Parameters ===")
    
    # Create an LLM
    llm = create_llm("mlx")
    
    # Generate with custom parameters
    response = llm.generate(
        "Write a short story about artificial intelligence.",
        temperature=0.9,  # Higher temperature for more creativity
        max_tokens=200,   # Limit response length
        top_p=0.95        # Nucleus sampling parameter
    )
    
    # Display the response
    print(f"Response: {response.text}")


if __name__ == "__main__":
    # Print platform information
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Run examples
    try:
        example_basic_generation()
        example_custom_model()
        example_with_system_prompt()
        example_streaming()
        example_generation_parameters()
    except Exception as e:
        print(f"Error: {e}")
        print("Note: MLX provider requires macOS with Apple Silicon.")
```

## References
- Reference the MLX Provider Usage Guide: `docs/mlx/mlx_usage_examples.md`
- See AbstractLLM's existing examples for reference
- MLX documentation: https://github.com/ml-explore/mlx

## Testing
Run the examples script to ensure it works correctly:

```bash
python examples/mlx_examples.py
``` 