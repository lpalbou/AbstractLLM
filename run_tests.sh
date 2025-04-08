#!/bin/bash

# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/

# Run only tests for a specific provider
# pytest tests/ -m openai
# pytest tests/ -m anthropic
# pytest tests/ -m ollama
# pytest tests/ -m huggingface

# Run tests with coverage report
# pytest tests/ --cov=abstractllm --cov-report=term

# Run tests with verbose output
# pytest tests/ -v

# Run specific test file
# pytest tests/test_interface.py 