#!/usr/bin/env python
"""
Example demonstrating provider chains and session management.

This example shows how to use provider chains for fallback behavior and
session management for maintaining conversation context across providers.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, Any

# Add parent directory to path to run the example without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abstractllm import create_llm, ModelParameter, ModelCapability
from abstractllm.chain import (
    ProviderChain, 
    create_fallback_chain, 
    create_capability_chain, 
    create_load_balanced_chain
)
from abstractllm.session import Session, SessionManager
from abstractllm.utils.logging import setup_logging
from abstractllm.exceptions import AbstractLLMError


def provider_change_callback(old_provider: str, new_provider: str, error: Exception) -> None:
    """
    Called when a provider chain switches providers due to an error.
    
    Args:
        old_provider: The provider that failed
        new_provider: The new provider being tried
        error: The error that caused the switch
    """
    print(f"\n[!] Switching from {old_provider} to {new_provider} due to error: {str(error)}")


async def example_fallback_chain():
    """Example of using a fallback chain with multiple providers."""
    print("\n=== Fallback Chain Example ===")
    
    # Create a chain that tries providers in sequence until one succeeds
    chain = create_fallback_chain(
        providers=["openai", "anthropic", "ollama", "huggingface"],
        on_provider_change=provider_change_callback,
        max_retries=2
    )
    
    prompt = "Explain quantum computing in simple terms."
    print(f"Prompt: {prompt}")
    
    try:
        # Try to generate with the chain
        # Will automatically fall back to the next provider if one fails
        response = chain.generate(prompt)
        print(f"Response: {response}")
    except RuntimeError as e:
        print(f"All providers failed: {e}")


async def example_capability_chain():
    """Example of using a capability-based chain."""
    print("\n=== Capability Chain Example ===")
    
    # Create a chain that only uses providers with specific capabilities
    chain = create_capability_chain(
        required_capabilities=[ModelCapability.VISION],
        preferred_providers=["openai", "anthropic", "ollama"],
        on_provider_change=provider_change_callback
    )
    
    # Example image URL (Eiffel Tower)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg"
    prompt = "What do you see in this image?"
    print(f"Prompt: {prompt}")
    print(f"Image: {image_url}")
    
    try:
        # The chain will only use providers that support vision
        response = chain.generate(prompt, image=image_url)
        print(f"Response: {response}")
    except RuntimeError as e:
        print(f"No vision-capable providers available: {e}")


async def example_load_balanced_chain():
    """Example of using a load-balanced chain for distribution."""
    print("\n=== Load Balanced Chain Example ===")
    
    # Create a chain that distributes load across providers
    chain = create_load_balanced_chain(
        providers=["openai", "anthropic", "ollama"],
        on_provider_change=provider_change_callback
    )
    
    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "What is natural language processing?",
        "How do neural networks work?",
        "What are the ethical concerns of AI?"
    ]
    
    print("Sending multiple prompts to demonstrate load balancing:")
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        try:
            response = chain.generate(prompt, max_tokens=100)
            print(f"Response from provider: {response[:150]}...")
        except RuntimeError as e:
            print(f"Error: {e}")


async def example_session_management():
    """Example of using session management for conversation context."""
    print("\n=== Session Management Example ===")
    
    # Create a session with a system prompt
    session = Session(
        system_prompt="You are a helpful and knowledgeable AI assistant specializing in physics.",
        provider="openai"  # Default provider
    )
    
    # First message with the default provider
    print("\nUser: What is the theory of relativity?")
    response = session.send("What is the theory of relativity?")
    print(f"Assistant (OpenAI): {response}")
    
    # Second message, switching to Anthropic
    print("\nUser: Can you explain it in simpler terms?")
    try:
        response = session.send(
            "Can you explain it in simpler terms?",
            provider="anthropic"  # Switch provider for this message
        )
        print(f"Assistant (Anthropic): {response}")
    except Exception as e:
        print(f"Error with Anthropic: {e}")
        # Fallback to original provider
        response = session.send("Can you explain it in simpler terms?")
        print(f"Assistant (OpenAI fallback): {response}")
    
    # Show conversation history
    print("\nConversation History:")
    for i, msg in enumerate(session.get_history()):
        if msg.role != "system":  # Skip system message
            print(f"{i}. {msg.role.capitalize()}: {msg.content[:100]}...")
    
    # Save the session
    session_file = "example_session.json"
    session.save(session_file)
    print(f"\nSession saved to {session_file}")
    
    # Load the session with a different provider
    loaded_session = Session.load(
        session_file,
        provider="ollama"  # New default provider
    )
    
    # Continue the conversation with the loaded session
    print("\nUser (loaded session): How is this related to quantum mechanics?")
    try:
        response = loaded_session.send("How is this related to quantum mechanics?")
        print(f"Assistant (Ollama): {response}")
    except Exception as e:
        print(f"Error with Ollama: {e}")
        # Fallback to a different provider
        response = loaded_session.send("How is this related to quantum mechanics?", provider="openai")
        print(f"Assistant (OpenAI fallback): {response}")
    
    # Clean up
    if os.path.exists(session_file):
        os.remove(session_file)


async def example_session_manager():
    """Example of using SessionManager to manage multiple conversations."""
    print("\n=== Session Manager Example ===")
    
    # Create a session manager with a directory for persistent storage
    sessions_dir = "example_sessions"
    os.makedirs(sessions_dir, exist_ok=True)
    manager = SessionManager(sessions_dir=sessions_dir)
    
    # Create two different sessions with different system prompts
    physics_session = manager.create_session(
        system_prompt="You are a helpful physics professor.",
        provider="openai"
    )
    
    history_session = manager.create_session(
        system_prompt="You are a historian specializing in ancient civilizations.",
        provider="anthropic"
    )
    
    # Use the physics session
    print("\n--- Physics Session ---")
    print("User: What is quantum entanglement?")
    try:
        response = physics_session.send("What is quantum entanglement?")
        print(f"Assistant: {response[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Use the history session
    print("\n--- History Session ---")
    print("User: Tell me about ancient Egypt.")
    try:
        response = history_session.send("Tell me about ancient Egypt.")
        print(f"Assistant: {response[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # List all sessions
    print("\nActive Sessions:")
    for session_id, created_at, last_updated in manager.list_sessions():
        print(f"- {session_id} (created: {created_at}, updated: {last_updated})")
    
    # Save all sessions
    manager.save_all()
    print(f"All sessions saved to {sessions_dir}")
    
    # Clean up
    import shutil
    if os.path.exists(sessions_dir):
        shutil.rmtree(sessions_dir)


async def example_async_session():
    """Example of using async session methods."""
    print("\n=== Async Session Example ===")
    
    # Create a session
    session = Session(
        system_prompt="You are a helpful AI assistant.",
        provider="openai"
    )
    
    # Send asynchronous messages
    print("User: What are the benefits of asynchronous programming?")
    try:
        response = await session.send_async("What are the benefits of asynchronous programming?")
        print(f"Assistant: {response[:200]}...")
        
        print("\nUser: Can you provide a simple example?")
        response = await session.send_async("Can you provide a simple example?")
        print(f"Assistant: {response[:200]}...")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples."""
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Check for API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("Warning: No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables.")
        print("Some examples may fail without API keys.")
    
    # Run examples
    await example_fallback_chain()
    
    if openai_key or anthropic_key:  # Skip vision example if no API keys
        await example_capability_chain()
        
    await example_load_balanced_chain()
    
    if openai_key or anthropic_key:  # Skip session examples if no API keys
        await example_session_management()
        await example_session_manager()
        await example_async_session()


if __name__ == "__main__":
    asyncio.run(main()) 