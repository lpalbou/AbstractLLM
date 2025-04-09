"""
Media handling module for AbstractLLM.

This module provides classes and utilities for handling different types of media
inputs (images, documents, etc.) with LLM providers.
"""

from abstractllm.media.interface import MediaInput
from abstractllm.media.image import ImageInput
from abstractllm.media.factory import MediaFactory
from abstractllm.media.processor import MediaProcessor

__all__ = [
    "MediaInput",
    "ImageInput",
    "MediaFactory",
    "MediaProcessor",
] 