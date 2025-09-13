#!/usr/bin/env python
"""
Test script for structured response system across different providers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abstractllm import create_llm
from abstractllm.structured_response import (
    StructuredResponseConfig, ResponseFormat, 
    StructuredResponseHandler, generate_pydantic_model
)
import json
from typing import List, Optional

# Try importing Pydantic for advanced tests
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  Pydantic not available. Some tests will be skipped.")


def test_json_response(provider_name: str):
    """Test basic JSON response generation."""
    print(f"\n{'='*60}")
    print(f"Testing JSON Response with {provider_name}")
    print('='*60)
    
    try:
        # Create provider
        if provider_name == "ollama":
            llm = create_llm(provider_name, model="qwen3:30b-a3b-q4_K_M")
        elif provider_name == "mlx":
            llm = create_llm(provider_name, model="mlx-community/Qwen3-30B-A3B-4bit")
        elif provider_name == "openai":
            llm = create_llm(provider_name, model="gpt-4o-mini")
        elif provider_name == "anthropic":
            llm = create_llm(provider_name, model="claude-3-5-sonnet-20241022")
        else:
            print(f"Unsupported provider: {provider_name}")
            return
        
        # Create handler
        handler = StructuredResponseHandler(provider_name)
        
        # Test 1: Simple JSON with schema
        print("\n1. Testing JSON with schema:")
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "age": {"type": "integer", "description": "Age in years"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of skills"
                },
                "experience": {
                    "type": "object",
                    "properties": {
                        "years": {"type": "integer"},
                        "level": {"type": "string", "enum": ["junior", "mid", "senior"]}
                    }
                }
            },
            "required": ["name", "age", "skills"]
        }
        
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON,
            schema=schema,
            examples=[
                {
                    "name": "Alice Johnson",
                    "age": 28,
                    "skills": ["Python", "Machine Learning", "Data Analysis"],
                    "experience": {"years": 5, "level": "mid"}
                },
                {
                    "name": "Bob Smith",
                    "age": 35,
                    "skills": ["JavaScript", "React", "Node.js"],
                    "experience": {"years": 10, "level": "senior"}
                }
            ],
            temperature_override=0.0  # Low temperature for consistency
        )
        
        # Prepare request
        request_params = handler.prepare_request(
            prompt="Generate a profile for a software engineer who specializes in backend development and has 7 years of experience.",
            config=config,
            system_prompt="You are a helpful assistant that generates structured data."
        )
        
        print("Modified prompt preview:")
        print(request_params["prompt"][:500] + "...")
        
        # Generate response
        response = llm.generate(**request_params)
        
        # Parse response
        try:
            parsed = handler.parse_response(response, config)
            print("\nParsed JSON response:")
            print(json.dumps(parsed, indent=2))
            print("✅ JSON parsing successful")
        except Exception as e:
            print(f"❌ JSON parsing failed: {e}")
        
        # Test 2: JSON with custom validation
        print("\n2. Testing JSON with custom validation:")
        
        def validate_age(data):
            """Custom validation: age must be between 18 and 65."""
            age = data.get("age", 0)
            return 18 <= age <= 65
        
        config_validated = StructuredResponseConfig(
            format=ResponseFormat.JSON,
            validation_fn=validate_age,
            max_retries=3
        )
        
        request_params = handler.prepare_request(
            prompt="Generate a JSON object with name and age for a working professional.",
            config=config_validated
        )
        
        response = llm.generate(**request_params)
        
        try:
            parsed = handler.parse_response(response, config_validated)
            print(f"Validated response: {parsed}")
            print("✅ Custom validation passed")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
        
    except Exception as e:
        print(f"❌ Error testing {provider_name}: {e}")
        import traceback
        traceback.print_exc()


def test_pydantic_response(provider_name: str):
    """Test Pydantic model-based response generation."""
    if not PYDANTIC_AVAILABLE:
        print("\n⚠️  Skipping Pydantic tests (not installed)")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing Pydantic Response with {provider_name}")
    print('='*60)
    
    try:
        # Create provider
        if provider_name == "ollama":
            llm = create_llm(provider_name, model="qwen3:30b-a3b-q4_K_M")
        elif provider_name == "openai":
            llm = create_llm(provider_name, model="gpt-4o-mini")
        else:
            print(f"Skipping Pydantic test for {provider_name}")
            return
        
        # Define Pydantic model
        class ProjectInfo(BaseModel):
            name: str = Field(description="Project name")
            description: str = Field(description="Brief project description")
            language: str = Field(description="Primary programming language")
            dependencies: List[str] = Field(description="List of main dependencies")
            version: str = Field(description="Current version", pattern=r"^\d+\.\d+\.\d+$")
            is_active: bool = Field(description="Whether project is actively maintained")
            stars: Optional[int] = Field(default=None, description="GitHub stars count", ge=0)
        
        # Create handler and config
        handler = StructuredResponseHandler(provider_name)
        
        config = StructuredResponseConfig(
            format=ResponseFormat.PYDANTIC,
            pydantic_model=ProjectInfo,
            temperature_override=0.0
        )
        
        # Prepare and execute request
        request_params = handler.prepare_request(
            prompt="Generate information about a popular Python web framework project.",
            config=config
        )
        
        response = llm.generate(**request_params)
        
        # Parse to Pydantic model
        try:
            project = handler.parse_response(response, config)
            print("\nParsed Pydantic model:")
            print(f"Name: {project.name}")
            print(f"Language: {project.language}")
            print(f"Version: {project.version}")
            print(f"Dependencies: {', '.join(project.dependencies)}")
            print(f"Active: {project.is_active}")
            if project.stars:
                print(f"Stars: {project.stars}")
            print("✅ Pydantic model validation successful")
        except Exception as e:
            print(f"❌ Pydantic validation failed: {e}")
        
    except Exception as e:
        print(f"❌ Error testing Pydantic with {provider_name}: {e}")


def test_retry_mechanism(provider_name: str):
    """Test retry mechanism for invalid responses."""
    print(f"\n{'='*60}")
    print(f"Testing Retry Mechanism with {provider_name}")
    print('='*60)
    
    try:
        # Create provider
        if provider_name == "ollama":
            llm = create_llm(provider_name, model="qwen3:30b-a3b-q4_K_M")
        else:
            print(f"Skipping retry test for {provider_name}")
            return
        
        handler = StructuredResponseHandler(provider_name)
        
        # Create a strict schema that might fail initially
        strict_schema = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "pattern": "^[A-Z]{3}-\\d{4}$",  # Strict pattern: XXX-0000
                    "description": "Product code in format XXX-0000"
                },
                "price": {
                    "type": "number",
                    "minimum": 0.01,
                    "maximum": 1000.00,
                    "description": "Price in USD"
                },
                "in_stock": {
                    "type": "boolean",
                    "description": "Whether item is in stock"
                }
            },
            "required": ["code", "price", "in_stock"]
        }
        
        config = StructuredResponseConfig(
            format=ResponseFormat.JSON,
            schema=strict_schema,
            max_retries=3,
            temperature_override=0.0
        )
        
        print("Testing with strict schema (may retry on validation failure)...")
        
        # Use generate_with_retry
        try:
            result = handler.generate_with_retry(
                generate_fn=llm.generate,
                prompt="Generate a product entry with code, price, and stock status. The code must be exactly 3 uppercase letters, a dash, and 4 digits (e.g., ABC-1234).",
                config=config
            )
            
            print(f"\nSuccessful result after retries:")
            print(json.dumps(result, indent=2))
            print("✅ Retry mechanism working")
            
        except ValueError as e:
            print(f"❌ Failed after all retries: {e}")
        
    except Exception as e:
        print(f"❌ Error testing retry mechanism: {e}")


def test_dynamic_model_generation():
    """Test dynamic Pydantic model generation."""
    if not PYDANTIC_AVAILABLE:
        print("\n⚠️  Skipping dynamic model test (Pydantic not installed)")
        return
    
    print(f"\n{'='*60}")
    print("Testing Dynamic Model Generation")
    print('='*60)
    
    # Dynamically create a model
    fields = {
        "title": (str, "Article title"),
        "author": (str, "Author name"),
        "word_count": (int, "Number of words"),
        "tags": (List[str], "List of tags"),
        "published": (bool, "Whether published")
    }
    
    ArticleModel = generate_pydantic_model("Article", fields)
    
    # Create instance
    try:
        article = ArticleModel(
            title="Understanding LLMs",
            author="Jane Doe",
            word_count=1500,
            tags=["AI", "ML", "NLP"],
            published=True
        )
        
        print("Dynamic model created successfully:")
        print(f"Schema: {article.model_json_schema()}")
        print("✅ Dynamic model generation successful")
        
    except Exception as e:
        print(f"❌ Dynamic model generation failed: {e}")


def test_format_comparison():
    """Compare different response formats."""
    print(f"\n{'='*60}")
    print("Comparing Response Formats")
    print('='*60)
    
    formats = [
        (ResponseFormat.JSON, "Standard JSON format"),
        (ResponseFormat.JSON_SCHEMA, "JSON with schema validation"),
        (ResponseFormat.YAML, "YAML format"),
        (ResponseFormat.XML, "XML format"),
        (ResponseFormat.PYDANTIC, "Pydantic model validation")
    ]
    
    example_data = {
        "name": "Test Item",
        "value": 42,
        "active": True,
        "tags": ["test", "demo"]
    }
    
    for format_type, description in formats:
        print(f"\n{format_type.value}: {description}")
        print(f"  Example: {example_data}")
        
        config = StructuredResponseConfig(format=format_type)
        
        # Show how the prompt would be modified
        handler = StructuredResponseHandler("generic")
        if format_type == ResponseFormat.XML:
            print(f"  XML: {handler._dict_to_xml(example_data)[:100]}...")
        elif format_type == ResponseFormat.YAML:
            print(f"  YAML: {handler._dict_to_yaml(example_data)[:100]}...")


def main():
    """Run all structured response tests."""
    print("Structured Response System Tests")
    print("="*60)
    
    # Test format comparison
    test_format_comparison()
    
    # Test dynamic model generation
    test_dynamic_model_generation()
    
    # Test with different providers
    providers_to_test = ["ollama"]  # Start with Ollama
    
    for provider in providers_to_test:
        try:
            # Test JSON responses
            test_json_response(provider)
            
            # Test Pydantic responses
            test_pydantic_response(provider)
            
            # Test retry mechanism
            test_retry_mechanism(provider)
            
        except Exception as e:
            print(f"Skipping {provider}: {e}")
    
    print("\n" + "="*60)
    print("All structured response tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()