# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-09-22

### BREAKING CHANGE NOTICE
**⚠️ Important: This may be the last version before a major architectural refactoring.**
The next major version will split AbstractLLM into separate, focused packages:
- **AbstractLLM**: Core lightweight LLM abstraction (tools, media, streaming, async)
- **AbstractMemory**: Advanced hierarchical memory management and serialization
- **AbstractAgent**: Autonomous agent capabilities and orchestration
- **AbstractSwarm**: Multi-agent coordination (future)

This separation will enable better modularity, focused evolution, and lighter dependencies for users who only need basic LLM functionality.

### Added
#### Enhanced Agent Capabilities
- **Advanced Tools System**: Complete rewrite with SOTA autonomous agent capabilities
- **Tool Catalog & Discovery**: Intelligent tool categorization and recommendation system
- **Advanced File Operations**: Line-based editing with atomic transactions and multi-operation support
- **Enhanced File Search**: Case-insensitive pattern matching with multiple pattern support using "|" separator
- **Comprehensive Tool Registry**: Organized tool discovery with complexity levels and usage examples

#### New Advanced Tools
- **Edit File Tool**: Precise line-based file modifications with insert/delete/replace operations
- **Code Intelligence**: Professional-grade code analysis and manipulation tools
- **System Operations**: Enhanced system monitoring and resource management
- **Network Intelligence**: Advanced networking and web operations
- **Data Processing**: Sophisticated data manipulation and analysis capabilities

#### Documentation & Planning
- **Comprehensive Refactoring Documentation**: Detailed architectural analysis and implementation plans
- **Migration Strategy**: Complete separation roadmap with task breakdown
- **Architectural Evolution Analysis**: In-depth analysis of current limitations and future improvements
- **Implementation Guides**: Step-by-step refactoring documentation

### Changed
#### Tool System Improvements
- **Case-Insensitive File Operations**: All file search operations now support case-insensitive matching
- **Multiple Pattern Support**: Use "|" separator to search for multiple file patterns simultaneously
- **Enhanced Tool Examples**: Comprehensive examples and usage documentation for all tools
- **Improved Tool Organization**: Better categorization with complexity levels and dependency tracking

#### File Operations Enhancement
- **Pattern Matching**: `list_files()` and `search_files()` now support patterns like "*.py|*.js|*.md"
- **Case Insensitivity**: File searches now work with patterns like "*TEST*" finding both "test" and "TEST" files
- **Better Documentation**: Enhanced examples showing complex pattern usage and multiple search scenarios

#### Agent Architecture
- **Improved ReAct Capabilities**: Enhanced reasoning and action cycles with better tool integration
- **SOTA Tool Support**: Tools designed for complex multi-step autonomous operations
- **Safety & Observability**: Enhanced error handling and operation monitoring

### Fixed
- **Tool Discovery**: Improved tool availability detection and error handling
- **File Pattern Matching**: More robust pattern matching with better edge case handling
- **System Integration**: Enhanced compatibility across different operating systems

### Technical Details
#### Files Added
- `abstractllm/tools/advanced_tools.py` - SOTA autonomous agent tools (1594+ lines)
- `abstractllm/tools/tool_catalog.py` - Comprehensive tool discovery and categorization system (504+ lines)
- `docs/final-refactoring/` - Complete architectural refactoring documentation
- `docs/refactoring/` - Detailed analysis and implementation guides

#### Files Enhanced
- `abstractllm/tools/common_tools.py` - Enhanced with edit_file tool and improved pattern matching
- `abstractllm/tools/__init__.py` - Updated tool imports and organization

#### Architecture Documentation
- **Separation Analysis**: Detailed breakdown of how to split the monolithic structure
- **Task Breakdown**: Specific implementation steps for modular architecture
- **Migration Strategy**: Clear path for users during the transition
- **Benefits Analysis**: Performance and maintainability improvements from separation

### Migration Notes
- All existing APIs remain backward compatible
- New tools are additive and optional
- Enhanced file operations maintain existing parameter compatibility
- Tool catalog provides discovery without breaking existing workflows

### Future Roadmap
This version sets the foundation for the architectural split:
1. **Phase 1**: Extract AbstractMemory (hierarchical memory, serialization)
2. **Phase 2**: Extract AbstractAgent (autonomous capabilities, orchestration)
3. **Phase 3**: Create AbstractSwarm (multi-agent coordination)
4. **Phase 4**: Optimize AbstractLLM core (lightweight, focused)

The goal is to provide users with exactly what they need - from simple LLM calls to complex autonomous agent orchestration - without forcing unnecessary dependencies.

### Validation
#### Tested Configurations
- **Enhanced Tools**: All advanced tools tested across different environments ✅
- **File Operations**: Case-insensitive and multi-pattern matching validated ✅
- **Tool Discovery**: Catalog system tested with various tool combinations ✅
- **Backward Compatibility**: All existing functionality preserved ✅

## [1.0.5] - 2025-09-17

### Added
#### New Provider Support
- **LM Studio Provider**: Complete integration with LM Studio's OpenAI-compatible local model server
- **LM Studio Features**: Automatic model capability detection, unified memory management, and prompted tool support
- **Installation Support**: Added `pip install "abstractllm[lmstudio]"` option for LM Studio dependencies

#### Enhanced Architecture System
- **Comprehensive Model Detection**: Intelligent architecture detection system with 80+ models across 7 architecture families
- **Model Name Normalization**: Robust handling of provider-specific naming conventions (MLX, Ollama, LM Studio, etc.)
- **Unified Parameter System**: Consistent parameter handling and validation across all providers
- **JSON Asset Integration**: Model capabilities and architecture formats managed through comprehensive JSON databases

#### Documentation Improvements
- **Architecture Detection Guide**: Complete technical documentation in `docs/architecture-model-detection.md`
- **Provider Comparison**: Updated README with all 6 providers including LM Studio examples
- **Bug Documentation**: Comprehensive bug tracking system in `docs/backlog/bugs/`
- **LM Studio Acknowledgments**: Proper attribution to LM Studio project in ACKNOWLEDGMENTS.md

### Changed
#### Model Capability System
- **Enhanced Detection**: Improved model name normalization with special pattern handling (e.g., `qwen/qwen3-next-80b` → `qwen3-next-80b-a3b`)
- **Provider-Specific Overrides**: Better handling of provider API limitations vs model capabilities
- **Tool Support Classification**: More accurate distinction between "native" and "prompted" tool support based on actual provider APIs

#### Memory Management
- **Unified `/mem` Command**: Enhanced memory command with correct token limits and improved layout
- **Model Information Display**: Moved model details to top of memory overview for better UX
- **Cross-Provider Compatibility**: Memory management works consistently across all providers

#### Documentation Structure
- **Progressive Disclosure**: Moved detailed technical content to dedicated documentation files
- **Cleaner README**: More focused main documentation with clear references to detailed guides
- **Better Organization**: Separated implementation details from user-facing documentation

### Fixed
#### Critical Tool Support Issues
- **LM Studio Native Tools Bug**: Fixed 400 Bad Request errors when models had "native" tool support but LM Studio API doesn't support OpenAI tools parameter
- **Provider API Compatibility**: Corrected assumption that "native" model tool support means provider API supports tools
- **qwen3-next-80b Configuration**: Changed from "native" to "prompted" tool support to work with LM Studio

#### Model Detection Issues
- **Model Name Normalization**: Fixed normalization for various provider prefixes and patterns
- **Token Limit Detection**: Corrected capability lookup for qwen3-next-80b (now shows 262,144 / 16,384 instead of 32,768 / 8,192)
- **Architecture Mapping**: Enhanced pattern matching for complex model names

#### Memory Command Improvements
- **Display Layout**: Moved model information to top of `/mem` output for better information hierarchy
- **Token Calculation**: Fixed context token calculation from session messages
- **Parameter Access**: Improved unified parameter system integration

### Known Issues
#### LM Studio Streaming Behavior
- **Provider-Specific Issue**: LM Studio sends cumulative responses instead of incremental tokens during streaming
- **Confirmed Models**: Affects both `qwen/qwen3-next-80b` and `qwen3-coder:30b` when using LM Studio provider
- **Cross-Provider Comparison**: Same models work correctly with incremental streaming on MLX and Ollama providers
- **Performance Impact**: Results in inefficient token transmission and display artifacts showing progressive sentence reconstruction
- **Workaround**: Use non-streaming mode (`stream=False`) with LM Studio for clean output

### Technical Details
#### Files Added
- `docs/architecture-model-detection.md` - Comprehensive architecture detection documentation
- `docs/backlog/bugs/lmstudio-native-tools-bug.md` - Detailed bug report and analysis
- `docs/backlog/bugs/native-tools-audit.md` - Provider tool support compatibility audit
- `abstractllm/providers/lmstudio_provider.py` - Complete LM Studio provider implementation

#### Files Modified
- `abstractllm/assets/model_capabilities.json` - Updated qwen3-next-80b-a3b tool support, added model entries
- `abstractllm/architectures/detection.py` - Enhanced model name normalization and special pattern handling
- `abstractllm/utils/commands.py` - Improved `/mem` command layout and model information display
- `README.md` - Added LM Studio provider documentation and restructured architecture section
- `ACKNOWLEDGMENTS.md` - Added LM Studio attribution

#### Architecture Improvements
- **Provider Capability Override System**: Foundation for provider-specific tool support detection
- **Model vs Provider Separation**: Better distinction between model capabilities and provider API support
- **Comprehensive Testing Strategy**: Framework for validating provider/model combinations

### Migration Notes
- **LM Studio Integration**: New provider available with `create_llm("lmstudio", model="qwen/qwen3-next-80b")`
- **Tool Support Changes**: Some models changed from "native" to "prompted" tool support for better compatibility
- **Memory Command**: Enhanced `/mem` output layout, all functionality preserved
- **Documentation**: Main README streamlined, detailed docs moved to dedicated files

### Validation
#### Tested Configurations
- **LM Studio + qwen/qwen3-next-80b**: Tool calling, memory management, parameter validation ✅
- **LM Studio + qwen3-coder:30b**: Full compatibility across all features ✅
- **Cross-Provider Switching**: Memory and session state preserved ✅
- **Architecture Detection**: All 80+ models correctly identified ✅

## [1.0.4] - 2025-09-15

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