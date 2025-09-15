# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] - 2025-01-15

### Added
#### CLI Enhancements
- **New `/temperature` Command**: Complete temperature control with show/set functionality and range validation (0.0-2.0)
- **Temperature Alias**: Added `/temp` as shorthand for `/temperature` command
- **Memory Exploration Commands**: Added `/working` command to inspect working memory contents (recent active items)
- **Enhanced `/links` Command**: Comprehensive memory links visualization with explanations and statistics
- **Updated Help System**: Enhanced `/help` documentation with new commands and usage examples

#### Memory System Improvements
- **Working Memory Inspection**: Users can now view recent active items in working memory with timestamps and importance scores
- **Memory Links Understanding**: Detailed explanations of how memory components connect and relate to each other
- **Educational Content**: Rich explanations help users understand AI memory architecture and reasoning processes

#### Parameter Management
- **Deterministic Generation Fix**: Resolved temperature null issue in Ollama requests caused by None value overwrites
- **Context Length Control**: Fixed `/mem` command integration with AbstractLLM's MAX_INPUT_TOKENS parameter
- **User Configuration Priority**: Ollama provider now respects user-configured token limits over model defaults

### Changed
#### CLI User Experience
- **Temperature Control**: Automatic temperature adjustment to 0.0 when seed is set for true determinism
- **Smart Mode Detection**: Temperature ranges automatically categorized (deterministic, focused, balanced, creative)
- **Memory Display Format**: Context usage now shows clear `<used tokens> / <max tokens>` format with color-coded percentages
- **Command Documentation**: Improved help text with better categorization and practical examples

#### Memory System Architecture
- **Enhanced Link Visualization**: Links now include type breakdown, statistics, and educational explanations
- **Working Memory Display**: Rich formatting with item types, timestamps, importance scores, and capacity usage
- **Memory Component Integration**: Better separation and explanation of different memory stores

### Fixed
#### Critical Parameter Issues
- **Temperature Null Bug**: Fixed Ollama provider ignoring user-set temperature values due to kwargs override issue
- **Max Tokens Integration**: Resolved `/mem` command not properly setting context limits in Ollama requests
- **Configuration Preservation**: Prevented None values in kwargs from overwriting existing configuration

#### Memory Access
- **Working Memory Visibility**: Previously inaccessible working memory contents now fully explorable
- **Link System Understanding**: Enhanced `/links` command from basic visualization to comprehensive explanation
- **Memory Navigation**: Complete toolkit for exploring all memory components with clear documentation

#### Session State Management
- **Deterministic Generation**: Fixed session ID and timestamp randomness affecting reproducibility
- **Memory Consistency**: Enhanced memory system to use deterministic values when seed is set
- **Cross-Session Persistence**: Improved memory state consistency across multiple sessions

### Technical Details
#### Files Modified
- `abstractllm/utils/commands.py` - Added `/temperature` and `/working` commands, enhanced `/links` and `/help`
- `abstractllm/providers/ollama.py` - Fixed kwargs None value override issue, added user token limit support
- `abstractllm/session.py` - Enhanced deterministic mode detection and memory initialization
- `abstractllm/memory.py` - Added session reference support for deterministic behavior

#### New Features Implementation
- **Temperature Command**: Complete show/set functionality with validation, mode detection, and educational feedback
- **Working Memory Command**: Inspection of recent active items with rich formatting and explanations
- **Enhanced Links Command**: Educational content about memory connections with statistics and type breakdown
- **Improved Help System**: Updated documentation covering all memory exploration commands

#### Bug Fixes Applied
- **Parameter Override Protection**: Filter None values from kwargs before updating configuration
- **Context Length Handling**: Check user-configured MAX_INPUT_TOKENS before falling back to model defaults
- **Memory State Determinism**: Use session reference to detect deterministic mode for consistent IDs and timestamps

### Usage Examples
```bash
# New temperature control
alma> /temperature 0.3
alma> /temp  # Show current temperature

# Memory exploration
alma> /working  # View recent active items
alma> /links    # Understand memory connections
alma> /mem 16384  # Set context limit (now works correctly)

# Deterministic generation (now fully working)
alma> /seed 123  # Automatically sets temperature to 0.0
alma> /temperature 0.5  # Can adjust independently
```

### Migration Notes
- All new commands are additive - no breaking changes to existing functionality
- Temperature and memory commands work across all providers
- Enhanced help system provides comprehensive documentation for memory exploration
- Previous memory exploration limitations now resolved with new `/working` command

## [1.0.3] - 2025-09-14

### Added
- **Global ALMA Command**: Added `alma` console script that provides global access to the intelligent agent
- **CLI Module**: New `abstractllm.cli` module that integrates all SOTA features from alma-simple.py
- **Universal Agent Access**: Users can now run `alma` from anywhere after installing AbstractLLM
- **Full Feature Integration**: The global command includes hierarchical memory, ReAct reasoning, knowledge graphs, and tool support

### Changed
- **Package Distribution**: Enhanced package to include console script entry point
- **User Experience**: Simplified access to the intelligent agent capabilities without needing to clone the repository

### Fixed
- **Tool Call Parsing**: Enhanced JSON parsing robustness for LLM-generated tool calls with unescaped newlines
- **Write File Tool**: Fixed tool call parsing when content contains literal newlines or special characters

### Installation
After upgrading to v1.0.3, users can install and use the global command:
```bash
pip install abstractllm==1.0.3
alma --help
alma --prompt "Hello, I'm testing the global command"
alma  # Interactive mode with memory and reasoning
```

## [1.0.2] - 2025-09-14

### Fixed
- **OpenAI Provider Response Format**: Fixed OpenAI provider to return proper `GenerateResponse` objects instead of raw strings, ensuring consistency with other providers and proper response structure

## [1.0.1] - 2025-09-14

### Fixed
- **MLX Dependencies in [all] Extra**: Fixed issue where `pip install "abstractllm[all]"` did not include MLX dependencies
- **User Experience**: Users can now install all provider support including MLX using the `[all]` extra
- **Platform Compatibility**: Added documentation clarifying that MLX dependencies are Apple Silicon specific

### Changed
- **Installation Documentation**: Updated README to clarify that `[all]` extra now includes MLX dependencies
- **Platform Notes**: Added note about MLX platform compatibility

## [1.0.0] - 2025-09-14

### BREAKING CHANGES
This is a major release with significant architectural changes and new capabilities. While the core API remains compatible, several advanced features have been added and some internal structures have changed.

### Added
#### Core Infrastructure
- **Hierarchical Memory System (Alpha)**: Three-tier memory architecture with working, episodic, and semantic memory
- **ReAct Reasoning Cycles (Alpha)**: Complete reasoning cycles with scratchpad traces and fact extraction
- **Knowledge Graph Integration (Alpha)**: Automatic fact extraction and relationship mapping
- **Context-Aware Retrieval (Alpha)**: Memory-enhanced LLM prompting with relevant context injection
- **Enhanced Tool System**: Tool creation with Pydantic validation and retry logic (alpha phase)
- **Structured Response System**: JSON/YAML response formatting with validation across all providers
- **Retry Strategies**: Exponential backoff, circuit breakers, and error recovery mechanisms
- **Scratchpad Manager**: Advanced reasoning trace management for agent workflows

#### Provider Enhancements
- **OpenAI Provider**: Manual improvements for better tool support and structured responses
- **Universal Tool Handler**: Architecture-aware tool handling that adapts to model capabilities
- **Enhanced Architecture Detection**: Improved model capability detection and optimization
- **Provider-Agnostic Features**: Memory and reasoning work across all 5 providers

#### Agent Development
- **ALMA-Simple Agent**: Complete example agent with memory, reasoning, and tool capabilities
- **Enhanced Session Management**: Persistent conversations with memory consolidation
- **Cross-Session Persistence**: Knowledge preserved across different sessions
- **Tool Integration**: Universal compatibility across all providers (native and prompted)

#### New Tools and Utilities
- **Common Tools**: Enhanced file operations, search, and system tools
- **Enhanced Tools Framework**: Advanced tool definition with examples and validation
- **Display Utilities**: Better formatting and output management
- **Command Utilities**: Comprehensive command execution and management
- **Response Helpers**: Structured response processing and validation

### Changed
#### Architecture Improvements
- **Major refactoring** of provider architecture for better maintainability
- **Unified detection system** for model capabilities and architecture
- **Enhanced base provider** with universal tool support
- **Improved session system** with memory and reasoning integration
- **Better error handling** with intelligent fallback and retry strategies

#### Documentation
- **Comprehensive documentation overhaul** with factual, humble language
- **Clear alpha testing markers** for experimental features
- **Accurate provider capability descriptions** 
- **Honest assessment** of features and limitations
- **New developer guides** and implementation reports

#### Tool System
- **Enhanced tool definitions** with rich parameter schemas and examples
- **Improved parsing** for real-world LLM inconsistencies
- **Better validation** and error handling
- **Universal compatibility** across all providers

### Fixed
- **Tool result formatting** across different providers
- **Session memory management** and persistence
- **Provider-specific** tool call handling
- **Architecture detection** for various model families
- **Error recovery** and fallback mechanisms
- **Memory consolidation** and fact extraction

### Deprecated
- **Old tool system** (legacy support maintained)
- **Basic memory implementations** (replaced with hierarchical system)

### Removed
- **Outdated architecture files** and unused templates
- **Deprecated utilities** and redundant code
- **Legacy test files** (moved to tmp/ directory)

### Security
- **Enhanced input validation** for tools and memory operations
- **Better error handling** to prevent information leakage
- **Improved command execution safety**

### Technical Details
#### New Files Added
- `abstractllm/memory.py` - Hierarchical memory system (1860+ lines)
- `abstractllm/retry_strategies.py` - Advanced retry strategies
- `abstractllm/scratchpad_manager.py` - ReAct reasoning management
- `abstractllm/structured_response.py` - Universal structured responses
- `abstractllm/tools/enhanced_core.py` - Enhanced tool definitions
- `abstractllm/tools/enhanced.py` - Enhanced tool framework
- `abstractllm/tools/handler.py` - Universal tool handler
- `abstractllm/tools/parser.py` - Robust tool call parsing
- `abstractllm/tools/registry.py` - Tool registry system
- `abstractllm/utils/commands.py` - Command utilities
- `abstractllm/utils/display.py` - Display formatting
- `abstractllm/utils/response_helpers.py` - Response processing

#### Files Significantly Updated
- `abstractllm/session.py` - Enhanced with memory and reasoning
- `abstractllm/providers/base.py` - Universal tool support
- `abstractllm/providers/ollama.py` - Improved tool handling
- `abstractllm/providers/openai.py` - Manual provider improvements
- `abstractllm/providers/anthropic.py` - Enhanced capabilities
- `abstractllm/architectures/detection.py` - Better model detection

#### Performance Improvements
- **Memory operations**: O(1) indexed retrieval vs O(n) scanning
- **Tool execution**: Better error recovery and fallback strategies
- **Provider switching**: Seamless switching between providers
- **Context management**: Efficient memory consolidation

### Migration Guide
#### For Existing Users
- Core API remains backward compatible
- Memory features are opt-in (enable_memory=True)
- Enhanced tools are additive to existing tool system
- No breaking changes to basic LLM usage

#### New Features Usage
```python
# Enable memory and reasoning (alpha)
session = create_session(
    "anthropic",
    enable_memory=True,              # Hierarchical memory
    memory_config={
        'working_memory_size': 10,
        'consolidation_threshold': 5
    }
)

# Use memory context and reasoning
response = session.generate(
    "Analyze the project",
    use_memory_context=True,         # Alpha feature
    create_react_cycle=True          # Alpha feature
)
```

### Notes
- **Memory and agency features** are in alpha testing
- **OpenAI support** achieved through manual provider improvements, not automatic compatibility
- **Breaking changes** are minimal and mostly affect internal architecture
- **Production readiness** varies by feature (core features stable, memory features alpha)

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