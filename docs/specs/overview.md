# AbstractLLM Package

## Overview

AbstractLLM is a lightweight Python package that provides a unified interface for interacting with different Large Language Models (LLMs). It abstracts away the implementation details of various providers (OpenAI, Anthropic, Ollama, Hugging Face), allowing developers to easily switch between models without changing their code.

## Core Features

- **Unified Interface**: Interact with any supported LLM through a consistent API
- **Multiple Provider Support**: OpenAI, Anthropic, Ollama, and Hugging Face models
- **Simple Configuration**: Easy setup and configuration for each provider
- **Request/Response Logging**: Built-in logging of prompts and completions
- **Capability Inspection**: Query models for their capabilities and limitations

## Target Audience

- Developers building applications that leverage LLMs
- Researchers who need to compare outputs across different models
- Projects that want to avoid vendor lock-in with a specific LLM provider
- Framework developers who need a consistent LLM interface

## Key Use Cases

1. **Model Experimentation**: Easily try different models without rewriting code
2. **Multi-model Applications**: Use different models for different tasks based on their strengths
3. **Fallback Chains**: Implement fallback strategies when a primary model is unavailable
4. **Simplified Integration**: Add LLM capabilities to applications with minimal code

## Design Philosophy

1. **Minimalism**: Focus only on the core abstraction, avoid feature bloat
2. **Consistency**: Provide a uniform experience across all supported providers
3. **Transparency**: Make it easy to understand what's happening under the hood
4. **Extensibility**: Allow for easy addition of new providers
5. **Independence**: No dependencies on other higher-level frameworks

## Non-Goals

- Not a full-featured framework like LangChain or LlamaIndex
- Does not handle prompt engineering or templating
- Does not implement caching (beyond what providers themselves offer)
- Does not manage conversation state or history