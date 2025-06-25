"""
Migration status tracking for provider tool support.

This module tracks which providers have been migrated to use the new
BaseProvider tool methods vs their own implementations.
"""


class ProviderMigrationStatus:
    """Track which providers have been migrated to new tool system."""
    
    # Provider migration status flags
    HUGGINGFACE_NEW_TOOLS = True  # Uses base class methods
    ANTHROPIC_NEW_TOOLS = True     # Uses base class methods
    OPENAI_NEW_TOOLS = True        # Uses base class methods
    OLLAMA_NEW_TOOLS = True        # Uses base class methods
    MLX_NEW_TOOLS = True           # Uses base class methods
    
    # Additional flags for specific features
    SUPPORTS_NATIVE_TOOLS = {
        "huggingface": False,  # Prompted only
        "anthropic": True,     # Native tool API
        "openai": True,        # Native function calling
        "ollama": True,        # Some models support native
        "mlx": False,          # Prompted only
    }