# Contributing to AbstractLLM

Thank you for your interest in contributing to AbstractLLM! This guide provides information on how to contribute to the project, including setup instructions, coding standards, and the contribution workflow.

## Getting Started

### Prerequisites

To contribute to AbstractLLM, you'll need:

- Python 3.9 or higher
- Git
- A GitHub account
- [Poetry](https://python-poetry.org/) for dependency management

### Development Environment Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/abstractllm.git
   cd abstractllm
   ```
3. Install development dependencies:
   ```bash
   poetry install --with dev
   ```
4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Project Structure

The AbstractLLM project is organized as follows:

```
abstractllm/
├── abstractllm/          # Main package code
│   ├── providers/        # Provider implementations
│   ├── media/            # Media handling code
│   ├── tools/            # Tool call implementations
│   ├── interface.py      # Core interfaces
│   └── ...
├── tests/                # Test suite
├── docs/                 # Documentation
├── examples/             # Example code
└── ...
```

## Coding Standards

AbstractLLM follows these coding standards:

- **PEP 8**: Follow Python's [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- **Type Hints**: Use Python type hints for all function parameters and return types
- **Documentation**: Document all public interfaces with docstrings
- **Tests**: Write tests for new functionality
- **Clean Code**: Keep functions small and focused on a single task
- **Error Handling**: Use appropriate exception types and error messages

## Development Workflow

### Creating a Feature

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes, following these steps:
   - Write tests first (test-driven development)
   - Implement the feature
   - Add documentation
   - Ensure all tests pass

3. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: description of your feature"
   ```

4. Push your branch to your fork:
   ```bash
   git push -u origin feature/your-feature-name
   ```

5. Create a pull request against the `main` branch of the main repository

### Fixing a Bug

1. Create a new branch from `main`:
   ```bash
   git checkout -b fix/bug-description
   ```

2. Implement your fix, following these steps:
   - Write a failing test that demonstrates the bug
   - Fix the bug
   - Ensure all tests pass

3. Commit your changes:
   ```bash
   git commit -m "Fix: description of the bug fix"
   ```

4. Push your branch and create a pull request as described above

## Testing

AbstractLLM uses pytest for testing. To run the tests:

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=abstractllm

# Run a specific test file
poetry run pytest tests/test_specific_file.py
```

### Test Categories

- **Unit Tests**: Tests for individual components in isolation
- **Integration Tests**: Tests for interactions between components
- **Provider Tests**: Tests for specific provider implementations
- **End-to-End Tests**: Tests for the entire system

## Documentation

All new features should include documentation. AbstractLLM uses Markdown for documentation.

### Documentation Guidelines

- Document all public APIs with clear docstrings
- Include examples for non-trivial functionality
- Keep documentation up to date with code changes
- Follow the existing documentation structure

To build the documentation locally:

```bash
# Install documentation dependencies
poetry install --with docs

# Build the documentation
cd docs
mkdocs build

# Serve the documentation locally
mkdocs serve
```

## Pull Request Guidelines

When submitting a pull request:

1. Make sure all tests pass
2. Update documentation as needed
3. Describe your changes in the PR description
4. Link any related issues
5. Ensure your code passes all CI checks

## Versioning and Release Process

AbstractLLM follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality
- **PATCH** version for backwards-compatible bug fixes

The release process is managed by the core maintainers.

## Code Review Process

All submissions require review by project maintainers. The review process ensures:

- Code quality and adherence to standards
- Test coverage
- Documentation completeness
- Security considerations

## Community Guidelines

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Pull Requests**: For code contributions

### Code of Conduct

All contributors are expected to adhere to the project's Code of Conduct. Please review the [CODE_OF_CONDUCT.md](https://github.com/lpalbou/abstractllm/blob/main/CODE_OF_CONDUCT.md) file before contributing.

## Security Vulnerability Reporting

If you discover a security vulnerability in AbstractLLM, please DO NOT open a public issue. Instead, send an email to [security@example.com](mailto:security@example.com) with details of the vulnerability.

## License

By contributing to AbstractLLM, you agree that your contributions will be licensed under the project's [MIT License](https://github.com/lpalbou/abstractllm/blob/main/LICENSE).

Thank you for contributing to AbstractLLM! 