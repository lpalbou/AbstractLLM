# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.3] - 2025-05-04
### Added
- Added core dependencies to ensure basic functionality works without extras
- Ollama provider now explicitly checks for required dependencies (requests and aiohttp)
- Improved documentation for provider-specific dependency requirements

### Changed
- Updated providers to use lazy imports for better dependency management
- Modified README installation instructions to be more explicit about dependencies

### Fixed
- Fixed dependency issues when installing the base package without extras
- Providers now check for required dependencies and provide clear error messages
- Resolved cross-dependency issues between providers (e.g., torch dependency affecting Anthropic usage)
- Improved error handling for missing dependencies with helpful installation instructions

## [0.5.2] - 2025-05-03
### Fixed
- Fixed resolution of provider-specific dependencies
- Improved error messages when optional dependencies are missing
- Enhanced dependency management for cleaner installations

## [0.5.1] - 2025-05-02
### Fixed
- Added missing optional dependencies in pyproject.toml to properly support package extras
- Fixed installation of extras like `[all]`, `[tools]`, `[openai]`, etc.
- Added development extras for improved developer experience
- Synchronized the build system configuration between setup.py and pyproject.toml

## [0.5.0] - 2025-05-01
### Added
- Enhanced examples in README.md with simplified tool call patterns
- Added comparison table for tool call approaches
- Added clear documentation for tool dependencies and installation options
- Improved installation instructions with clear options for different use cases

### Changed
- Improved Session class to automatically use provider's model in tool calls
- Simplified tool call implementation with cleaner API
- Updated documentation with step-by-step examples
- Enhanced error messages for missing tool dependencies

### Fixed
- Fixed Session.generate_with_tools to properly use model from provider
- Fixed tool registration and execution to require less boilerplate
- Improved error handling in provider model detection
- Clarified tool dependency requirements in error messages
- Better fallbacks when optional dependencies are not installed

### Security
- N/A

## [0.4.7] - 2025-04-25
- Added tool call support for compatible models
- Added interactive ALMA command line agent
- Fixed Anthropic API issue with trailing whitespace in messages
- Fixed empty input handling in interactive mode

### Added
- Initial project setup
- Core abstractions for LLM interactions
- Support for OpenAI and Anthropic providers
- Configuration management system
- Comprehensive logging and error handling
- Test suite with real-world examples
- Documentation and contribution guidelines
- Enum-based parameter system for type-safe configuration
- Extended model capabilities detection
- Async generation support for all providers
- Streaming response support for all providers
- Additional parameters for fine-grained control
- Enhanced HuggingFace provider with model cache management
- Tool call support for compatible models
- Interactive ALMA command line agent

### Changed
- Updated interface to use typed enums for parameters
- Improved provider implementations with consistent parameter handling
- Extended README with examples of enum-based parameters

### Fixed
- Anthropic API issue with trailing whitespace in messages
- Empty input handling in interactive mode 