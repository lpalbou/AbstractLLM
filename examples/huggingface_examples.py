"""
HuggingFace Provider Examples

This file contains practical examples of using the HuggingFace provider in AbstractLLM.
Each example is self-contained and demonstrates a specific use case.
"""

from abstractllm import create_llm
from abstractllm.enums import ModelParameter
from abstractllm.exceptions import ResourceError
from pathlib import Path

def text_generation_example():
    """Basic text generation example."""
    # Create provider with basic configuration
    llm = create_llm("huggingface", 
        model="microsoft/phi-2",
        temperature=0.7,
        max_tokens=2048,
        device_map="auto"  # Automatically choose best device
    )
    
    try:
        # Basic generation
        response = llm.generate("Write a short story about a brave cat.")
        print("Basic generation:", response)
        
        # Generation with system prompt
        response = llm.generate(
            "Write a haiku about nature.",
            system_prompt="You are a skilled poet."
        )
        print("\nWith system prompt:", response)
        
        # Streaming generation
        print("\nStreaming response:")
        for chunk in llm.generate(
            "Explain quantum computing step by step.",
            stream=True
        ):
            print(chunk, end="", flush=True)
            
    finally:
        llm.cleanup()

def vision_example(image_path: str):
    """Vision model example."""
    # Create provider with vision model
    llm = create_llm("huggingface", 
        model="Salesforce/blip-image-captioning-base",
        device_map="auto"
    )
    
    try:
        # Image captioning
        response = llm.generate(
            "Describe this image in detail.",
            files=[image_path]
        )
        print("Image description:", response)
        
        # Get model capabilities
        caps = llm.get_capabilities()
        print("\nModel capabilities:", caps)
        
    finally:
        llm.cleanup()

def document_qa_example(document_path: str):
    """Document question answering example."""
    # Create provider with document QA model
    llm = create_llm("huggingface", 
        model="microsoft/layoutlmv3-base",
        device_map="auto"
    )
    
    try:
        # Ask questions about the document
        response = llm.generate(
            "What are the main points discussed in this document?",
            files=[document_path]
        )
        print("Document analysis:", response)
        
        # Ask specific questions
        response = llm.generate(
            "What is the conclusion of this document?",
            files=[document_path]
        )
        print("\nDocument conclusion:", response)
        
    finally:
        llm.cleanup()

def resource_managed_example():
    """Example with resource management."""
    try:
        # Create provider with resource limits
        llm = create_llm("huggingface", 
            model="microsoft/phi-2",
            device_map="cuda",
            max_memory={
                "cuda:0": "4GiB",  # GPU memory limit
                "cpu": "8GiB"      # CPU memory limit
            },
            use_flash_attention=True  # Enable optimizations
        )
        
        # Generate text
        response = llm.generate(
            "Explain how to optimize Python code.",
            max_tokens=1000
        )
        print("Generated response:", response)
        
    except ResourceError as e:
        print(f"Resource error: {e.details}")
    finally:
        llm.cleanup()

def model_recommendation_example():
    """Example using model recommendations."""
    llm = create_llm("huggingface", model="microsoft/phi-2")
    
    # Get recommendations for different tasks
    tasks = ["text-generation", "vision", "text2text"]
    
    for task in tasks:
        print(f"\nRecommended models for {task}:")
        recommendations = llm.get_model_recommendations(task)
        for rec in recommendations:
            print(f"- {rec['model']}: {rec['description']}")

async def async_example():
    """Example of async generation."""
    llm = create_llm("huggingface", model="microsoft/phi-2")
    
    try:
        # Basic async generation
        response = await llm.generate_async(
            "Write a story about AI."
        )
        print("Async response:", response)
        
        # Async streaming
        print("\nAsync streaming:")
        async for chunk in llm.generate_async(
            "Explain the future of technology.",
            stream=True
        ):
            print(chunk, end="", flush=True)
            
    finally:
        llm.cleanup()

def main():
    """Run all examples."""
    print("=== Text Generation Example ===")
    text_generation_example()
    
    print("\n=== Vision Example ===")
    # Replace with your image path
    vision_example("path/to/image.jpg")
    
    print("\n=== Document QA Example ===")
    # Replace with your document path
    document_qa_example("path/to/document.pdf")
    
    print("\n=== Resource Managed Example ===")
    resource_managed_example()
    
    print("\n=== Model Recommendation Example ===")
    model_recommendation_example()
    
    print("\n=== Async Example ===")
    import asyncio
    asyncio.run(async_example())

if __name__ == "__main__":
    main() 