# Contributing to AbstractLLM

Thank you for your interest in contributing to AbstractLLM! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/lpalbou/abstractllm.git
   cd abstractllm
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest
   ```
4. Ensure code quality:
   ```bash
   black .
   isort .
   flake8
   ```
5. Commit your changes with a descriptive message
6. Push to your fork
7. Create a pull request

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Write tests for new features

## Testing

- Write tests for all new features
- Ensure existing tests pass
- Use pytest fixtures for common setup
- Aim for high test coverage

## Documentation

- Update [README.md](README.md) for significant changes
- Add docstrings to new functions/classes
- Update API documentation if needed

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include relevant tests
3. Update documentation as needed
4. Ensure all tests pass
5. Request review from maintainers

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## Questions?

Feel free to open an issue if you have any questions about contributing! 