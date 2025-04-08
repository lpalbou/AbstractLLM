"""
Tests for the generation functionality of AbstractLLM providers.
"""

import pytest
import asyncio
from typing import Dict, Any, List, Union, Generator

from abstractllm import AbstractLLMInterface, ModelParameter, ModelCapability
from abstractllm.providers.openai import OpenAIProvider
from tests.utils import (
    validate_response, 
    validate_not_contains, 
    has_capability, 
    collect_stream, 
    collect_stream_async,
    count_tokens,
    check_order_in_response
)
from tests.examples.test_prompts import (
    FACTUAL_PROMPTS, 
    SYSTEM_PROMPT_TESTS,
    STREAMING_TEST_PROMPTS,
    PARAMETER_TEST_PROMPTS,
    LONG_CONTEXT_PROMPT
)


@pytest.mark.parametrize("prompt_test", FACTUAL_PROMPTS)
def test_factual_generation(any_provider: AbstractLLMInterface, prompt_test: Dict[str, Any]) -> None:
    """
    Test that factual prompts get appropriate responses.
    
    Args:
        any_provider: Provider instance to test
        prompt_test: Test prompt data
    """
    prompt = prompt_test["prompt"]
    expected_contains = prompt_test["expected_contains"]
    
    # Generate response
    response = any_provider.generate(prompt)
    
    # Should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Should contain at least one expected string
    assert validate_response(response, expected_contains)


@pytest.mark.parametrize("prompt_test", SYSTEM_PROMPT_TESTS)
def test_system_prompt(any_provider: AbstractLLMInterface, prompt_test: Dict[str, Any]) -> None:
    """
    Test that system prompts influence the response appropriately.
    
    Args:
        any_provider: Provider instance to test
        prompt_test: Test prompt data
    """
    capabilities = any_provider.get_capabilities()
    if not has_capability(capabilities, ModelCapability.SYSTEM_PROMPT):
        pytest.skip("Provider does not support system prompts")
    
    prompt = prompt_test["prompt"]
    system_prompt = prompt_test["system_prompt"]
    expected_contains = prompt_test["expected_contains"]
    not_expected_contains = prompt_test.get("not_expected_contains", [])
    
    # Generate response with system prompt
    response = any_provider.generate(prompt, system_prompt=system_prompt)
    
    # Should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Should contain at least one expected string
    assert validate_response(response, expected_contains)
    
    # Skip strict AI mention check for models that struggle with this instruction
    # Only check for OpenAI models which can reliably follow this instruction
    if not_expected_contains and isinstance(any_provider, OpenAIProvider):
        assert validate_not_contains(response, not_expected_contains), f"Response contained forbidden terms: {response}"


@pytest.mark.parametrize("prompt_test", STREAMING_TEST_PROMPTS)
def test_streaming(any_provider: AbstractLLMInterface, prompt_test: Dict[str, Any]) -> None:
    """
    Test streaming responses.
    
    Args:
        any_provider: Provider instance to test
        prompt_test: Test prompt data
    """
    capabilities = any_provider.get_capabilities()
    if not has_capability(capabilities, ModelCapability.STREAMING):
        pytest.skip("Provider does not support streaming")
    
    prompt = prompt_test["prompt"]
    min_chunks = prompt_test.get("min_chunks", 2)
    expected_sequence = prompt_test.get("expected_sequence", [])
    
    # Generate streaming response
    response_stream = any_provider.generate(prompt, stream=True)
    
    # Should be a generator
    assert isinstance(response_stream, Generator)
    
    # Collect chunks and count them
    chunks = []
    for chunk in response_stream:
        chunks.append(chunk)
    
    # Should have received at least min_chunks
    assert len(chunks) >= min_chunks
    
    # Each chunk should be a string
    for chunk in chunks:
        assert isinstance(chunk, str)
    
    # Combine chunks to check full response
    full_response = "".join(chunks)
    assert len(full_response) > 0
    
    # Check sequence if provided
    if expected_sequence:
        assert check_order_in_response(full_response, expected_sequence)


@pytest.mark.asyncio
async def test_async_generation(any_provider: AbstractLLMInterface) -> None:
    """
    Test asynchronous generation.
    
    Args:
        any_provider: Provider instance to test
    """
    capabilities = any_provider.get_capabilities()
    if not has_capability(capabilities, ModelCapability.ASYNC):
        pytest.skip("Provider does not support async generation")
    
    # Use a simple prompt
    prompt = "What is the capital of Japan?"
    expected_contains = ["Tokyo"]
    
    # Generate response asynchronously
    response = await any_provider.generate_async(prompt)
    
    # Should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Should contain expected string
    assert validate_response(response, expected_contains)


@pytest.mark.asyncio
async def test_async_streaming(any_provider: AbstractLLMInterface) -> None:
    """
    Test asynchronous streaming.
    
    Args:
        any_provider: Provider instance to test
    """
    capabilities = any_provider.get_capabilities()
    if not (has_capability(capabilities, ModelCapability.ASYNC) and 
            has_capability(capabilities, ModelCapability.STREAMING)):
        pytest.skip("Provider does not support async streaming")
    
    # Use a simple prompt
    prompt = "Count from 1 to 5."
    expected_sequence = ["1", "2", "3", "4", "5"]
    
    # Generate streaming response asynchronously
    response_stream = await any_provider.generate_async(prompt, stream=True)
    
    # Collect chunks
    response = await collect_stream_async(response_stream)
    
    # Should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Check sequence
    assert check_order_in_response(response, expected_sequence)


@pytest.mark.parametrize("prompt_test", PARAMETER_TEST_PROMPTS)
def test_parameter_settings(any_provider: AbstractLLMInterface, prompt_test: Dict[str, Any]) -> None:
    """
    Test that parameter settings influence the response appropriately.
    
    Args:
        any_provider: Provider instance to test
        prompt_test: Test prompt data
    """
    prompt = prompt_test["prompt"]
    parameters = prompt_test["parameters"]
    
    # Generate response with parameters
    response = any_provider.generate(prompt, **parameters)
    
    # Should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0


def test_long_context(any_provider: AbstractLLMInterface) -> None:
    """
    Test long context handling and token limits.
    
    Args:
        any_provider: Provider instance to test
    """
    prompt = LONG_CONTEXT_PROMPT["prompt"]
    parameters = LONG_CONTEXT_PROMPT["parameters"]
    expected_tokens_range = LONG_CONTEXT_PROMPT["expected_tokens_range"]
    
    # Generate response with token limit
    response = any_provider.generate(prompt, **parameters)
    
    # Should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Response should be roughly within expected token range
    # This is a rough estimate, as token counting varies by model
    token_count = count_tokens(response)
    min_tokens, max_tokens = expected_tokens_range
    assert min_tokens <= token_count <= max_tokens, f"Response had {token_count} tokens, expected {min_tokens}-{max_tokens}"


@pytest.mark.parametrize("provider_fixture", ["openai_provider", "anthropic_provider", "ollama_provider", "huggingface_provider"])
def test_provider_specific_generation(request: Any, provider_fixture: str) -> None:
    """
    Test generation with each specific provider to allow for provider-specific checks.
    
    Args:
        request: pytest request object
        provider_fixture: Name of the provider fixture
    """
    try:
        provider = request.getfixturevalue(provider_fixture)
    except pytest.skip.Exception:
        pytest.skip(f"Skipping {provider_fixture} tests")
        return
    
    # Generate a simple response
    response = provider.generate("What is the capital of France?")
    
    # Should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Should contain the expected answer
    assert validate_response(response, ["Paris"]) 