#!/usr/bin/env python
"""
JSON Mode Example for AbstractLLM.

This script demonstrates how to use the JSON mode capability with OpenAI
to generate structured responses.
"""

import os
import json
import logging
from abstractllm import create_llm, ModelCapability
from abstractllm.utils.logging import setup_logging

def main():
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Check if OpenAI API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("\n=== JSON Mode Example with OpenAI ===\n")
    
    # Create the LLM instance with explicit model and API key
    llm = create_llm("openai", model="gpt-3.5-turbo", api_key=api_key)
    
    # Check if the provider supports JSON mode
    capabilities = llm.get_capabilities()
    if not capabilities.get(ModelCapability.JSON_MODE):
        print("This provider does not support JSON mode.")
        return
    
    print("Provider supports JSON mode capability!")
    
    # Example 1: Generate a simple user profile
    print("\nExample 1: Generate a simple user profile")
    prompt = "Create a user profile with the following fields: name, age, occupation, and hobbies (as an array). Use fictional data."
    
    print(f"Prompt: {prompt}")
    
    # Generate with JSON mode enabled
    response = llm.generate(prompt, json_mode=True)
    
    print("\nRaw response:")
    print(response)
    
    # Parse the JSON response
    try:
        user_profile = json.loads(response)
        print("\nParsed JSON:")
        print(f"Name: {user_profile['name']}")
        print(f"Age: {user_profile['age']}")
        print(f"Occupation: {user_profile['occupation']}")
        print(f"Hobbies: {', '.join(user_profile['hobbies'])}")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    
    # Example 2: Generate a more complex data structure
    print("\nExample 2: Generate a more complex data structure")
    prompt = """
    Generate a JSON data structure for a small e-commerce product catalog with the following requirements:
    
    1. It should contain an array of 3 products
    2. Each product should have: id, name, price, category, description, stock, and ratings
    3. The ratings field should be an object with: average_rating (number) and reviews (array of objects)
    4. Each review should have: user_name, rating (1-5), and comment
    
    Make sure all values are realistic but fictional.
    """
    
    print(f"Prompt: {prompt}")
    
    # Generate with JSON mode enabled
    response = llm.generate(prompt, json_mode=True)
    
    print("\nRaw response:")
    print(response)
    
    # Parse the JSON response
    try:
        catalog = json.loads(response)
        print("\nParsed JSON summary:")
        print(f"Number of products: {len(catalog['products'])}")
        
        # Display first product details
        product = catalog['products'][0]
        print(f"\nSample product: {product['name']}")
        print(f"  Price: ${product['price']}")
        print(f"  Category: {product['category']}")
        print(f"  In stock: {product['stock']}")
        print(f"  Average rating: {product['ratings']['average_rating']}")
        print(f"  Number of reviews: {len(product['ratings']['reviews'])}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing JSON: {e}")
    
    print("\nJSON mode makes it easy to generate structured data that can be directly used in your application!")

if __name__ == "__main__":
    main() 