"""
Simple long input handling for the CLI.

This module provides basic single-line input with support for long queries
up to 8k tokens (~32k characters). No multiline complexity.
"""

import sys
from typing import Optional


def get_enhanced_input(prompt: str = "alma> ", max_chars: int = 32768) -> str:
    """
    Get single-line input with support for long queries.

    Simple and straightforward: just like regular input() but with
    a higher character limit for long queries up to 8k tokens.

    Controls:
    - Enter: Submit query
    - Ctrl+C: Cancel input

    Args:
        prompt: The prompt to display
        max_chars: Maximum characters allowed (default ~8k tokens)

    Returns:
        The input string
    """
    try:
        user_input = input(prompt)

        # Check character limit
        if len(user_input) > max_chars:
            print(f"⚠️ Input limit reached ({max_chars} characters). Truncating...")
            user_input = user_input[:max_chars]

        return user_input

    except (EOFError, KeyboardInterrupt):
        # Ctrl+D or Ctrl+C - return empty
        return ""


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate the number of tokens in text.

    Args:
        text: The text to estimate
        chars_per_token: Average characters per token (default 4.0)

    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)


def format_input_info(text: str) -> str:
    """
    Format information about the input.

    Args:
        text: The input text

    Returns:
        Formatted info string
    """
    chars = len(text)
    lines = text.count('\n') + 1 if text else 0
    tokens = estimate_tokens(text)

    parts = []
    if lines > 1:
        parts.append(f"{lines} lines")
    parts.append(f"{chars} chars")
    parts.append(f"~{tokens} tokens")

    return f"[{', '.join(parts)}]"