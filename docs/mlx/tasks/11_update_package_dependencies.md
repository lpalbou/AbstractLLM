# Task 11: Update Package Dependencies

## Description
Update AbstractLLM's package configuration to include MLX-related dependencies as optional extras, ensuring compatibility and proper installation.

## Requirements
1. Add MLX and MLX-LM as optional dependencies with appropriate version constraints
2. Add additional dependencies required for MLX provider functionality
3. Update package metadata to reflect MLX support
4. Maintain backward compatibility with existing installations

## Implementation Details

Update the `pyproject.toml` file (or `setup.py` depending on the project's configuration) to include MLX as an optional dependency:

```toml
# If using pyproject.toml
[project.optional-dependencies]
# Existing dependencies...
mlx = [
    "mlx>=0.0.25", 
    "mlx-lm>=0.0.7",
    "huggingface_hub>=0.15.0",
    "transformers>=4.30.0",  # For chat templating
    "numpy>=1.24.0",         # Often needed by MLX
    "humanize>=4.6.0"        # For human-readable sizes in caching utilities
]
```

Or in setup.py:

```python
# If using setup.py
extras_require={
    # Existing extras...
    "mlx": [
        "mlx>=0.0.25", 
        "mlx-lm>=0.0.7",
        "huggingface_hub>=0.15.0",
        "transformers>=4.30.0",  # For chat templating
        "numpy>=1.24.0",         # Often needed by MLX
        "humanize>=4.6.0"        # For human-readable sizes in caching utilities
    ],
},
```

Update package metadata to indicate MLX support:

```toml
# In pyproject.toml
[project]
# Other metadata...
description = "A unified interface for large language models with support for OpenAI, Anthropic, Hugging Face, Ollama, and MLX"
keywords = ["llm", "ai", "openai", "gpt", "claude", "huggingface", "ollama", "mlx", "apple silicon"]
```

Also update the classifiers to indicate OS support:

```toml
# In pyproject.toml
[project]
classifiers = [
    # Existing classifiers...
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
]
```

Add a platform-specific dependency specification to make mlx optional but automatic on macOS arm64:

```toml
# In pyproject.toml
[project.dependencies]
# ... existing dependencies
mlx = {version = ">=0.0.25", optional = true, markers = "sys_platform == 'darwin' and platform_machine == 'arm64'"}
```

## Installation Documentation

Add documentation on how to install with MLX support in the main README.md:

```markdown
## Installation

### Standard Installation
```bash
pip install abstractllm
```

### With MLX Support (for Apple Silicon)
```bash
pip install "abstractllm[mlx]"
```

MLX support is only available on macOS with Apple Silicon (M1/M2/M3 chips).
```

## References
- MLX GitHub repository: https://github.com/ml-explore/mlx
- MLX-LM GitHub repository: https://github.com/ml-explore/mlx-lm
- Reference the MLX Provider Implementation Guide: `docs/mlx/mlx_provider_implementation.md`
- PEP 508 for dependency specifications: https://peps.python.org/pep-0508/

## Testing
1. Verify installation with MLX dependencies works on Apple Silicon: `pip install ".[mlx]"`
2. Verify installation on non-Apple platforms doesn't attempt to install MLX
3. Verify that existing installations remain functional after the update
4. Test importing MLX-related modules after installation 