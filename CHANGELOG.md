# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- N/A

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.4.8] - 2025-04-28
### Added
- Enhanced examples in README.md with simplified tool call patterns
- Added comparison table for tool call approaches

### Changed
- Improved Session class to automatically use provider's model in tool calls
- Simplified tool call implementation with cleaner API
- Updated documentation with step-by-step examples

### Fixed
- Fixed Session.generate_with_tools to properly use model from provider
- Fixed tool registration and execution to require less boilerplate
- Improved error handling in provider model detection

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